// Prompt flash attention on Blackwell (sm_100a): tcgen05 tensor-core MMAs
// whose accumulators live in tensor memory (TMEM), FlashAttention-4's design
// at head dimension 256 with one 128-query tile per CTA.
//
// A CTA takes 128 query positions of one head. TMEM (512 columns of 128
// lanes, lane = query row) holds two score tiles S0, S1 (TC_BN columns
// each) and the output O (256). Warps:
//   0-3   softmax: each thread owns one row; it reads its scores, writes
//         the probabilities as bf16 over the upper half of the same S tile
//         (P), and keeps the row's running max and sum;
//   4-7   correction: rescales O when a row's max rises (lazily: only past
//         2^8, FA4's threshold), then writes the output;
//   8     MMA: one thread issues S = Q K^T into the S tile the softmax is not
//         using, and O += P V with P read from TMEM; it allocates TMEM;
//   9     loads: keys and values by TMA, one 16-token page per lane, keys a
//         tile ahead of values (FA3's order).
// The G heads that share a key/value head take the same query tile in one
// cluster of G CTAs: each loads a share of every key and value tile and
// multicasts it to all, so the cluster reads the cache once rather than G
// times (loads, not math, bound a CTA on its own). A tile's buffer is free
// once every CTA's MMAs are done with it: each CTA's MMA commit arrives on
// the empty barriers of all.
// Q, K and V tiles are [4 chunks of 64 columns][rows] with 128-byte
// swizzling, as the Hopper kernel stores them. Scores are in log2 units (Q
// is scaled by scale * log2 e as it is staged).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200  // tcgen05: B200, not RTX 50

// 32 columns from col of TMEM, one lane (row) per thread of the warp: v[j]
// is column col + j of row 32 * (warp % 4) + lane (taddr carries both).
__device__ __forceinline__ void tc_ld32(uint32_t taddr, uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
                 : "r"(taddr));
}
__device__ __forceinline__ void tc_st32(uint32_t taddr, const uint32_t* v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n" ::"r"(taddr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31])
                 : "memory");
}

__device__ __forceinline__ uint64_t tc_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 |
           (uint64_t)((sbo >> 4) & 0x3FFF) << 32 | (1ull << 46) | (2ull << 61);
}
// D (TMEM) (+)= A (smem) B (smem).
__device__ __forceinline__ void tc_mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d),
                 "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// D (TMEM) (+)= A (TMEM) B (smem).
__device__ __forceinline__ void tc_mma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(d),
                 "r"(a), "l"(b), "r"(idesc), "r"(acc));
}
// Arrives on bar once this thread's earlier MMAs complete.
__device__ __forceinline__ void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
// As tc_commit, arriving on bar in every CTA of the cluster in mask.
__device__ __forceinline__ void tc_commit_all(uint32_t bar, uint16_t mask) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n" ::"r"(bar),
        "h"(mask)
        : "memory");
}
// tma3 into the same offset of every CTA of the cluster in mask, completing
// bytes on bar in each.
__device__ __forceinline__ void tma3_all(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar,
                                         uint16_t mask) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, "
        "%3, %4}], [%5], %6;\n" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "h"(mask)
        : "memory");
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
#endif

// Keys per tile, and stages of keys and of values: the queries' 64 KB and
// two stages each of 64 keys and values fill 192 KB, so the next tile's
// keys and values load while this one's are in use (128-key tiles would fit
// only one stage, and loads would wait on the MMAs).
#define TC_BN 64
#define TC_NST 2
// Instruction descriptors (bf16 inputs, f32 accumulate): S = Q K^T at
// M128 N(TC_BN) with both operands K-major; O += P V at M128 N256 with V
// N-major.
#define TC_IDESC_QK (0x08000490u | (TC_BN >> 3) << 17)
#define TC_IDESC_PV 0x08410490u
#define TC_TILE (128 * HEAD_DIM * 2)       // 64 KB: 128 rows of 256 bf16
#define TC_STAGE (TC_BN * HEAD_DIM * 2)    // a stage of keys or values
// Q, the key and value stages, then barriers, row scales (2 x 128), row sums
// and the TMEM address.
#define TC_SMEM (TC_TILE + 2 * TC_NST * TC_STAGE + 256 + 3 * 128 * 4 + 16)

template <int G>
__device__ __forceinline__ void attn_tc_body(ATTN_FLASH_ARGS, const CUtensorMap* kmap, const CUtensorMap* vmap,
                                             int nslots) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200  // tcgen05: B200, not RTX 50
    constexpr int BN = TC_BN, NST = TC_NST, CHUNK = BN * 128;  // CHUNK: 64 columns of a stage
    extern __shared__ __align__(1024) unsigned char tsm[];
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(tsm);
    const uint32_t sQ = base, sK = base + TC_TILE, sV = sK + NST * TC_STAGE, bars = sV + NST * TC_STAGE;
    // Barriers: per stage of keys and of values, full (the bytes landed) and
    // empty (every CTA's MMAs are done with it); per score tile, full,
    // probabilities ready (softmax and correction) and row stats ready.
    const uint32_t q_full = bars, k_full = bars + 8, k_empty = k_full + 8 * NST, v_full = k_empty + 8 * NST,
                   v_empty = v_full + 8 * NST, s_full = v_empty + 8 * NST, p_ready = s_full + 16, stats = p_ready + 16,
                   o_full = stats + 16, sum_full = o_full + 8;
    float* sScale = reinterpret_cast<float*>(tsm + (bars - base) + 256);  // [2][128]
    float* sSum = sScale + 256;                                           // [128]
    uint32_t* sTmem = reinterpret_cast<uint32_t*>(sSum + 128);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // The item: head of the group (the CTA's rank in its cluster), query
    // tile and key/value head, longest tiles first within each key/value
    // head.
    int ntiles = 0;
    for (int i = 0; i < seqs; i++) {
        const int n = cu[i + 1] - cu[i];
        if (n > 1 && !is_spec(outs, spec, i)) ntiles += (n + 127) / 128;
    }
    const int it = blockIdx.x;
    if (it >= ntiles * G * Hkv) return;
    const int j = it % G, kvh = it / G / ntiles;
    int tile = ntiles - 1 - it / G % ntiles, s = 0;
    const uint16_t all = (1 << G) - 1;
    for (; s < seqs; s++) {
        const int n = cu[s + 1] - cu[s];
        if (n <= 1 || is_spec(outs, spec, s)) continue;
        const int tiles = (n + 127) / 128;
        if (tile < tiles) break;
        tile -= tiles;
    }
    const int h = kvh * G + j, tb = cu[s] + tile * 128, nb = min(128, cu[s + 1] - tb);
    const int L = pos[tb + nb - 1] + 1, T = (L + BN - 1) / BN;

    if (tid == 0) {
        mbar_init(q_full, 128);
        for (int i = 0; i < NST; i++) {
            mbar_init(k_full + 8 * i, 1);
            mbar_init(k_empty + 8 * i, G);
            mbar_init(v_full + 8 * i, 1);
            mbar_init(v_empty + 8 * i, G);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(s_full + 8 * i, 1);
            mbar_init(p_ready + 8 * i, 256);
            mbar_init(stats + 8 * i, 128);
        }
        mbar_init(o_full, 1);
        mbar_init(sum_full, 128);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == 8) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;\n" ::"r"(
                         (uint32_t)__cvta_generic_to_shared(sTmem))
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    // Every CTA's barriers are ready before any multicast reaches them.
    tc_before();
    cluster_sync();
    tc_after();
    // TMEM: score tiles S0 and S1 (BN columns each) from column 0, O from 256.
    const uint32_t tmem = *sTmem, tS = tmem, tO = tmem + 256;
    // This warp's TMEM lanes (rows) for the softmax and correction warps.
    const uint32_t lanes = (uint32_t)(32 * (warp & 3)) << 16;
    const int r = 32 * (warp & 3) + lane;

    if (warp < 4) {
        // Softmax. First stage this row's queries, scaled, into sQ.
        {
            float f[8];
            for (int c = 0; c < 32; c++) {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < nb) {
                    unpack8(*reinterpret_cast<const uint4*>(q + ((size_t)(tb + r) * Hq + h) * HEAD_DIM + c * 8), f);
#pragma unroll
                    for (int e = 0; e < 8; e++) f[e] *= scale * LOG2E;
                    v = pack8(f);
                }
                *reinterpret_cast<uint4*>(tsm + (c >> 3) * 16384 + r * 128 + (((c & 7) ^ (r & 7)) << 4)) = v;
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            mbar_arrive(q_full);
        }
        const int qp = r < nb ? pos[tb + r] : -1;
        float m = -INFINITY, l = 0.f;
        for (int i = 0; i < T; i++) {
            const uint32_t slot = tS + BN * (i & 1) + lanes;
            mbar_wait(s_full + 8 * (i & 1), (i >> 1) & 1);
            tc_after();
            uint32_t u[BN];
#pragma unroll
            for (int c = 0; c < BN / 32; c++) tc_ld32(slot + 32 * c, u + 32 * c);
            tc_wait_ld();
            float* x = reinterpret_cast<float*>(u);
            const int k0 = BN * i;
            if (k0 + BN - 1 > qp) {
#pragma unroll
                for (int c = 0; c < BN; c++)
                    if (k0 + c > qp) x[c] = -INFINITY;
            }
            float mx = m;
#pragma unroll
            for (int c = 0; c < BN; c++) mx = fmaxf(mx, x[c]);
            // Lazy rescaling: keep the old max unless the new one is more
            // than 2^8 larger; probabilities then stay below 256.
            float sc = 1.f;
            if (m == -INFINITY || mx > m + 8.f) {
                sc = m == -INFINITY ? 1.f : fast_exp2(m - mx);
                m = mx;
            }
            sScale[128 * (i & 1) + r] = sc;
            mbar_arrive(stats + 8 * (i & 1));
            const float b = m == -INFINITY ? 0.f : m;
            float sum = 0.f;
            uint32_t p[BN / 2];
#pragma unroll
            for (int c = 0; c < BN / 2; c++) {
                const float e0 = fast_exp2(x[2 * c] - b), e1 = fast_exp2(x[2 * c + 1] - b);
                sum += e0 + e1;
                p[c] = fa_bf2(e0, e1);
            }
            // P (bf16 pairs) over the upper half of the score tile.
#pragma unroll
            for (int c = 0; c < BN / 64; c++) tc_st32(slot + BN / 2 + 32 * c, p + 32 * c);
            tc_wait_st();
            tc_before();
            mbar_arrive(p_ready + 8 * (i & 1));
            l = l * sc + sum;
        }
        sSum[r] = l;
        mbar_arrive(sum_full);
    } else if (warp < 8) {
        // Correction, then the output.
        for (int i = 0; i < T; i++) {
            if (i > 0) {
                mbar_wait(stats + 8 * (i & 1), (i >> 1) & 1);
                const float sc = sScale[128 * (i & 1) + r];
                // Only a rescale waits for O (P V of the last tile). The
                // parity wait is exact: P V two tiles back is done (it was
                // issued before this tile's scores, which the stats
                // follow), and P V of this tile waits for this warp.
                if (__any_sync(0xffffffffu, sc != 1.f)) {
                    mbar_wait(o_full, (i - 1) & 1);
                    tc_after();
                    for (int c = 0; c < 8; c++) {
                        uint32_t u[32];
                        tc_ld32(tO + lanes + 32 * c, u);
                        tc_wait_ld();
#pragma unroll
                        for (int e = 0; e < 32; e++) u[e] = __float_as_uint(__uint_as_float(u[e]) * sc);
                        tc_st32(tO + lanes + 32 * c, u);
                    }
                    tc_wait_st();
                }
                tc_before();
            }
            mbar_arrive(p_ready + 8 * (i & 1));
        }
        mbar_wait(o_full, (T - 1) & 1);
        tc_after();
        mbar_wait(sum_full, 0);
        const float sum = sSum[r], inv = sum > 0.f ? 1.f / sum : 0.f;
        for (int c = 0; c < 8; c++) {
            uint32_t u[32];
            tc_ld32(tO + lanes + 32 * c, u);
            tc_wait_ld();
            if (r < nb) {
                const size_t t = (size_t)tb + r;
                const bf16* gate = qkv + t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + 32 * c;
                bf16* o = out + (t * Hq + h) * HEAD_DIM + 32 * c;
#pragma unroll
                for (int e = 0; e < 32; e += 2) {
                    float y[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const float ov = rbf(__uint_as_float(u[e + k]) * inv);
                        y[k] = ov / (1.f + expf(-b2f(gate[e + k])));
                    }
                    *reinterpret_cast<__nv_bfloat162*>(o + e) = __floats2bfloat162_rn(y[0], y[1]);
                }
            }
        }
    } else if (warp == 8) {
        // MMA issue, from one thread.
        if (lane == 0) {
            auto qk = [&](int i) {
                const int st = i % NST;
                mbar_wait(k_full + 8 * st, (i / NST) & 1);
                tc_after();
                const uint32_t d = tS + BN * (i & 1), k = sK + st * TC_STAGE;
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    const uint32_t qo = (kk >> 2) * 16384 + (kk & 3) * 32, ko = (kk >> 2) * CHUNK + (kk & 3) * 32;
                    tc_mma_ss(d, tc_desc(sQ + qo, 16, 1024), tc_desc(k + ko, 16, 1024), TC_IDESC_QK, kk > 0);
                }
                tc_commit(s_full + 8 * (i & 1));
                tc_commit_all(k_empty + 8 * st, all);
            };
            mbar_wait(q_full, 0);
            qk(0);
            for (int i = 0; i < T; i++) {
                if (i + 1 < T) qk(i + 1);
                const int st = i % NST;
                mbar_wait(p_ready + 8 * (i & 1), (i >> 1) & 1);
                mbar_wait(v_full + 8 * st, (i / NST) & 1);
                tc_after();
                const uint32_t pt = tS + BN * (i & 1) + BN / 2, v = sV + st * TC_STAGE;
#pragma unroll
                for (int kk = 0; kk < BN / 16; kk++)
                    tc_mma_ts(tO, pt + 8 * kk, tc_desc(v + kk * 2048, CHUNK, 1024), TC_IDESC_PV, i > 0 || kk > 0);
                tc_commit(o_full);
                tc_commit_all(v_empty + 8 * st, all);
            }
        }
        __syncwarp();
    } else {
        // Loads: of a tile's BN / 16 pages of 16 tokens, page p comes from
        // lane p of the CTA of rank p % G.
        if (lane == 0) {
            asm volatile("prefetch.tensormap [%0];\n" ::"l"(kmap) : "memory");
            asm volatile("prefetch.tensormap [%0];\n" ::"l"(vmap) : "memory");
        }
        const int* bt = block_table + (size_t)s * max_blocks;
        auto load = [&](int i, const CUtensorMap* map, uint32_t buf, uint32_t full, uint32_t empty) {
            const int st = i % NST;
            if (lane == 0) {
                if (i >= NST) mbar_wait(empty + 8 * st, (i / NST - 1) & 1);
                mbar_expect(full + 8 * st, TC_STAGE);
            }
            __syncwarp();
            if (lane < BN / 16 && lane % G == j) {
                const int kp = BN * i + 16 * lane, sl = kp < L ? bt[kp / block_size] * block_size : nslots;
#pragma unroll
                for (int a = 0; a < 4; a++)
                    tma3_all(buf + st * TC_STAGE + a * CHUNK + lane * 2048, map, 64 * a, kvh, sl, full + 8 * st, all);
            }
        };
        load(0, kmap, sK, k_full, k_empty);
        for (int i = 0; i < T; i++) {
            if (i + 1 < T) load(i + 1, kmap, sK, k_full, k_empty);
            load(i, vmap, sV, v_full, v_empty);
        }
        // Every CTA's last commits have reached this one's barriers.
        if (lane == 0)
            for (int i = max(0, T - NST); i < T; i++) {
                mbar_wait(k_empty + 8 * (i % NST), (i / NST) & 1);
                mbar_wait(v_empty + 8 * (i % NST), (i / NST) & 1);
            }
    }
    // Everyone is done with TMEM once the output is written, and no CTA
    // leaves while another may still send to it.
    tc_before();
    cluster_sync();
    tc_after();
    if (warp == 8) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n" ::"r"(tmem) : "memory");
#endif
}
#define ATTN_TC_PASS ATTN_FLASH_PASS, &kmap, &vmap, nslots
// attn_tc_g<G>: clusters of G CTAs of 320 threads.
extern "C" __global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(320, 1) attn_tc_g4(ATTN_WS_ARGS) {
    attn_tc_body<4>(ATTN_TC_PASS);
}
extern "C" __global__ void __cluster_dims__(6, 1, 1) __launch_bounds__(320, 1) attn_tc_g6(ATTN_WS_ARGS) {
    attn_tc_body<6>(ATTN_TC_PASS);
}
