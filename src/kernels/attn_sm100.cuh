// Prompt flash attention on Blackwell (sm_100a): tcgen05 tensor-core MMAs
// whose accumulators live in tensor memory (TMEM), FlashAttention-4's design
// at head dimension 256 with one 128-query tile per CTA.
//
// A CTA takes 128 query positions of one head. TMEM (512 columns of 128
// lanes, lane = query row) holds the queries Q (bf16 pairs, 128 columns),
// two score tiles S0, S1 (TC_BN columns each) and the output O (256).
// Keeping Q in TMEM is what makes 64-key tiles pay at head dimension 256:
// an MMA reading both operands from shared memory moves 6 KB per 128 x 64 x
// 16 step against 32 cycles of math, more than shared memory's 128 bytes a
// cycle; with Q in TMEM only the keys come from shared memory. Warps:
//   0-7   softmax: warps w and w + 4 share rows 32 (w % 4) .. + 31, each
//         taking half of a tile's columns. They swap row maxima at a named
//         barrier, write the probabilities as bf16 over the upper half of
//         the same S tile (P), and rescale their half of O when a row's
//         max rises (lazily: only past 2^8, FA4's threshold). They stage Q
//         first and write the output last;
//   8     MMA: one thread issues S = Q K^T into the S tile the softmax is not
//         using, and O += P V with P read from TMEM; it allocates TMEM;
//   9     loads: keys and values by TMA, one 16-token page per lane, keys a
//         tile ahead of values (FA3's order).
// The G heads that share a key/value head take the same query tile in one
// cluster of G CTAs: each loads a share of every key and value tile and
// multicasts it to all, so the cluster reads the cache once rather than G
// times. A stage is free once every CTA's MMAs are done with it: each CTA's
// MMA commit arrives on the empty barriers of all.
// K and V stages are [4 chunks of 64 columns][rows] with 128-byte
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

__device__ __forceinline__ void tc_st16(uint32_t taddr, const uint32_t* v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(taddr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15])
                 : "memory");
}
// One thread of the warp, chosen by elect.sync (ptxas wraps tcgen05 issued
// under a plain lane test in a loop over active lanes).
__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\nselp.u32 %0, 1, 0, e;\n}\n" : "=r"(p));
    return p;
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

// Keys per tile, and stages of keys and of values: three stages each of 64
// keys and values fill 192 KB of shared memory.
#define TC_BN 64
#define TC_NST 3
// Instruction descriptors (bf16 inputs, f32 accumulate): S = Q K^T at
// M128 N(TC_BN) with both operands K-major; O += P V at M128 N256 with V
// N-major.
#define TC_IDESC_QK (0x08000490u | (TC_BN >> 3) << 17)
#define TC_IDESC_PV 0x08410490u
#define TC_STAGE (TC_BN * HEAD_DIM * 2)  // a stage of keys or values
// The key and value stages, then barriers, the row maxima and sums of each
// half (2 x 2 x 128 and 2 x 128) and the TMEM address.
#define TC_SMEM (2 * TC_NST * TC_STAGE + 256 + 6 * 128 * 4 + 16)

__device__ __forceinline__ void pair_sync(int id) { asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory"); }

template <int G>
__device__ __forceinline__ void attn_tc_body(ATTN_FLASH_ARGS, const CUtensorMap* kmap, const CUtensorMap* vmap,
                                             int nslots) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200  // tcgen05: B200, not RTX 50
    constexpr int BN = TC_BN, NST = TC_NST, CHUNK = BN * 128;  // CHUNK: 64 columns of a stage
    extern __shared__ __align__(1024) unsigned char tsm[];
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(tsm);
    const uint32_t sK = base, sV = sK + NST * TC_STAGE, bars = sV + NST * TC_STAGE;
    // Barriers: per stage of keys and of values, full (the bytes landed) and
    // empty (every CTA's MMAs are done with it); per score tile, full and
    // probabilities ready.
    const uint32_t q_full = bars, k_full = bars + 8, k_empty = k_full + 8 * NST, v_full = k_empty + 8 * NST,
                   v_empty = v_full + 8 * NST, s_full = v_empty + 8 * NST, p_ready = s_full + 16, o_full = p_ready + 16;
    float* sMax = reinterpret_cast<float*>(tsm + (bars - base) + 256);  // [tile parity][half][128]
    float* sSum = sMax + 512;                                            // [half][128]
    uint32_t* sTmem = reinterpret_cast<uint32_t*>(sSum + 256);
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
        mbar_init(q_full, 256);
        for (int i = 0; i < NST; i++) {
            mbar_init(k_full + 8 * i, 1);
            mbar_init(k_empty + 8 * i, G);
            mbar_init(v_full + 8 * i, 1);
            mbar_init(v_empty + 8 * i, G);
        }
        for (int i = 0; i < 2; i++) {
            mbar_init(s_full + 8 * i, 1);
            mbar_init(p_ready + 8 * i, 256);
        }
        mbar_init(o_full, 1);
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
    // TMEM: Q in columns 0-127, S0 and S1 from 128, O from 256.
    const uint32_t tmem = *sTmem, tQ = tmem, tS = tmem + 128, tO = tmem + 256;

    if (warp < 8) {
        // Softmax. This warp's TMEM lanes (rows) and half of the columns.
        const int hf = warp >> 2, r = 32 * (warp & 3) + lane, pair = 1 + (warp & 3);
        const uint32_t lanes = (uint32_t)(32 * (warp & 3)) << 16;
        // Stage half of this row's queries, scaled, into TMEM as bf16 pairs.
        {
            uint32_t qv[64];
            const bf16* qrow = q + ((size_t)(tb + r) * Hq + h) * HEAD_DIM + 128 * hf;
#pragma unroll
            for (int c = 0; c < 16; c++) {
                float f[8];
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < nb) {
                    unpack8(*reinterpret_cast<const uint4*>(qrow + 8 * c), f);
#pragma unroll
                    for (int e = 0; e < 8; e++) f[e] *= scale * LOG2E;
                    v = pack8(f);
                }
                qv[4 * c] = v.x, qv[4 * c + 1] = v.y, qv[4 * c + 2] = v.z, qv[4 * c + 3] = v.w;
            }
            tc_st32(tQ + lanes + 64 * hf, qv);
            tc_st32(tQ + lanes + 64 * hf + 32, qv + 32);
            tc_wait_st();
            tc_before();
            mbar_arrive(q_full);
        }
        const int qp = r < nb ? pos[tb + r] : -1;
        float m = -INFINITY, l = 0.f;
        for (int i = 0; i < T; i++) {
            const uint32_t slot = tS + BN * (i & 1) + lanes;
            mbar_wait(s_full + 8 * (i & 1), (i >> 1) & 1);
            tc_after();
            uint32_t u[BN / 2];
            tc_ld32(slot + (BN / 2) * hf, u);
            tc_wait_ld();
            float* x = reinterpret_cast<float*>(u);
            const int k0 = BN * i + (BN / 2) * hf;
            if (k0 + BN / 2 - 1 > qp) {
#pragma unroll
                for (int c = 0; c < BN / 2; c++)
                    if (k0 + c > qp) x[c] = -INFINITY;
            }
            // Row max as a tree, then the partner half's.
            float mx4[4] = {x[0], x[1], x[2], x[3]};
#pragma unroll
            for (int c = 4; c < BN / 2; c += 4)
#pragma unroll
                for (int e = 0; e < 4; e++) mx4[e] = fmaxf(mx4[e], x[c + e]);
            float mx = fmaxf(fmaxf(mx4[0], mx4[1]), fmaxf(mx4[2], mx4[3]));
            float* mrow = sMax + 256 * (i & 1);
            mrow[128 * hf + r] = mx;
            pair_sync(pair);  // also: both halves have read S before either writes P over it
            mx = fmaxf(fmaxf(mx, mrow[128 * (1 - hf) + r]), m);
            // Lazy rescaling: keep the old max unless the new one is more
            // than 2^8 larger; probabilities then stay below 256.
            float sc = 1.f;
            if (m == -INFINITY || mx > m + 8.f) {
                sc = m == -INFINITY ? 1.f : fast_exp2(m - mx);
                m = mx;
            }
            if (__any_sync(0xffffffffu, sc != 1.f) && i > 0) {
                // O holds P V of the tiles before; this half's 128 columns.
                // The parity wait is exact: P V two tiles back is done (it
                // was issued before this tile's scores), and P V of this
                // tile waits for this warp.
                mbar_wait(o_full, (i - 1) & 1);
                tc_after();
#pragma unroll 1
                for (int c = 0; c < 4; c++) {
                    uint32_t o[32];
                    const uint32_t at = tO + lanes + 128 * hf + 32 * c;
                    tc_ld32(at, o);
                    tc_wait_ld();
#pragma unroll
                    for (int e = 0; e < 32; e++) o[e] = __float_as_uint(__uint_as_float(o[e]) * sc);
                    tc_st32(at, o);
                }
            }
            const float b = m == -INFINITY ? 0.f : m;
            float sum2[2] = {0.f, 0.f};
            uint32_t p[BN / 4];
#pragma unroll
            for (int c = 0; c < BN / 4; c++) {
                const float e0 = fast_exp2(x[2 * c] - b), e1 = fast_exp2(x[2 * c + 1] - b);
                sum2[c & 1] += e0 + e1;
                p[c] = fa_bf2(e0, e1);
            }
            // P (bf16 pairs) over the upper half of the score tile.
            tc_st16(slot + BN / 2 + (BN / 4) * hf, p);
            tc_wait_st();
            tc_before();
            mbar_arrive(p_ready + 8 * (i & 1));
            l = l * sc + sum2[0] + sum2[1];
        }
        // The output: this half's 128 columns, over the row's whole sum.
        sSum[128 * hf + r] = l;
        pair_sync(pair);
        const float sum = l + sSum[128 * (1 - hf) + r], inv = sum > 0.f ? 1.f / sum : 0.f;
        mbar_wait(o_full, (T - 1) & 1);
        tc_after();
        for (int c = 0; c < 4; c++) {
            uint32_t u[32];
            tc_ld32(tO + lanes + 128 * hf + 32 * c, u);
            tc_wait_ld();
            if (r < nb) {
                const size_t t = (size_t)tb + r;
                const int d0 = 128 * hf + 32 * c;
                const bf16* gate = qkv + t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d0;
                bf16* o = out + (t * Hq + h) * HEAD_DIM + d0;
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
        // MMA issue, from one elected thread.
        const bool leader = elect_one();
        auto qk = [&](int i) {
            const int st = i % NST;
            mbar_wait(k_full + 8 * st, (i / NST) & 1);
            tc_after();
            const uint32_t d = tS + BN * (i & 1), k = sK + st * TC_STAGE;
            if (leader) {
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    const uint32_t ko = (kk >> 2) * CHUNK + (kk & 3) * 32;
                    tc_mma_ts(d, tQ + 8 * kk, tc_desc(k + ko, 16, 1024), TC_IDESC_QK, kk > 0);
                }
                tc_commit(s_full + 8 * (i & 1));
                tc_commit_all(k_empty + 8 * st, all);
            }
            __syncwarp();
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
            if (leader) {
#pragma unroll
                for (int kk = 0; kk < BN / 16; kk++)
                    tc_mma_ts(tO, pt + 8 * kk, tc_desc(v + kk * 2048, CHUNK, 1024), TC_IDESC_PV, i > 0 || kk > 0);
                tc_commit(o_full);
                tc_commit_all(v_empty + 8 * st, all);
            }
            __syncwarp();
        }
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
