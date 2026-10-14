// Paged attention for decoding on AMD: see attn_body in qwen35.cu.
#pragma once

// The same work as attn_body (one block per token and kv head, keys split
// across the block's waves, each wave with its own online softmax, then
// merged), but the q.k products run on MFMA instead of a 32-lane shuffle
// tree per key: a wave takes 16 keys at a time and computes S^T = K Q^T, a
// lane holding keys 4 (lane / 16) .. + 3 of query head lane % 16 (heads past
// G are zero). Probabilities go through shared memory to the P V product,
// where each lane owns 4 of the 256 output dimensions for every head and
// reads whole value rows (8 bytes a lane). bf16 caches only.
template <int G, int WARPS>
__device__ __forceinline__ void attn_decode_mfma(
    const bf16* __restrict__ q, const void* __restrict__ kcache, const void* __restrict__ vcache,
    const bf16* __restrict__ qkv, int qstride, int Hq, int Hkv, const int* __restrict__ pos,
    const int* __restrict__ tok_seq, const int* __restrict__ block_table, int max_blocks, int block_size,
    bf16* __restrict__ out, float scale, uint8_t* fq, void* fs, int fmode, int frows, const int* __restrict__ cu,
    int skip_prompts, int min_chunk, float* __restrict__ part, const int* __restrict__ outs, int spec) {
    static_assert(G <= 16, "a head per MFMA column");
    constexpr int NW = WARPS * 32 / 64;  // 64-lane waves in the block
    __shared__ float sm_m[NW][G], sm_s[NW][G];
    __shared__ float sm_acc[NW][G][HEAD_DIM];
    __shared__ float sp[NW][16][G], sc_[NW][G];
    const int t = blockIdx.x, kvh = blockIdx.y;
    const int sq = tok_seq[t];
    if (skip_prompts && cu[sq + 1] - cu[sq] > 1 && !is_spec(outs, spec, sq)) return;
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, lr = lane & 15, lq = lane >> 4;
    const int L = pos[t] + 1;
    const int parts = part ? split_parts(L, gridDim.z, min_chunk) : 1;
    const int chunk = (L + parts - 1) / parts;
    const int j0 = part ? blockIdx.z * chunk : 0, j1 = part ? min(L, j0 + chunk) : L;
    if (j0 >= j1) return;
    const int* bt = block_table + (size_t)sq * max_blocks;
    // Query head lr's dimensions 16 kd + 4 lq .. + 3 (bf16 as stored; the
    // scale applies to the scores, so the operands stay exact).
    pe_s4 qb[HEAD_DIM / 16];
    {
        const bf16* qrow = q + ((size_t)t * Hq + kvh * G + (lr < G ? lr : 0)) * HEAD_DIM;
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM / 16; kd++) {
            const pe_s4 v = *reinterpret_cast<const pe_s4*>(qrow + 16 * kd + 4 * lq);
            qb[kd] = lr < G ? v : pe_s4{0, 0, 0, 0};
        }
    }
    // Per head (in lanes lr < G): running max and sum; per lane: outputs
    // for dimensions 4 lane .. + 3 of every head.
    float m = -INFINITY, l = 0.f;
    float o[G][4];
#pragma unroll
    for (int h = 0; h < G; h++) o[h][0] = o[h][1] = o[h][2] = o[h][3] = 0.f;
    auto wave_sync = [] {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    };
    auto slot_of = [&](int j) { return (size_t)bt[j / block_size] * block_size + (j % block_size); };
    // A tile's operands: this lane's K fragments (key k0 + lr) and V rows
    // (keys k0 .. k0 + 15, 4 dimensions each), loaded a tile ahead.
    struct Tile {
        pe_s4 k[HEAD_DIM / 16];
        uint2 v[16];
    };
    auto fetch = [&](int k0, Tile& tl) {
        const int kr = min(k0 + lr, j1 - 1);
        const size_t slot = slot_of(kr);
        const bf16* krow = (const bf16*)kcache + (slot * Hkv + kvh) * HEAD_DIM;
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM / 16; kd++) tl.k[kd] = *reinterpret_cast<const pe_s4*>(krow + 16 * kd + 4 * lq);
        // Key k0 + kk's slot from lane kk (keys past j1 repeat the last).
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const size_t sk = __shfl((unsigned long long)slot, kk, 64);
            tl.v[kk] = *reinterpret_cast<const uint2*>((const bf16*)vcache + (sk * Hkv + kvh) * HEAD_DIM + 4 * lane);
        }
    };
    Tile cur, nxt;
    int k0 = j0 + 16 * wave;
    if (k0 < j1) fetch(k0, cur);
    for (; k0 < j1; k0 += 16 * NW) {
        if (k0 + 16 * NW < j1) fetch(k0 + 16 * NW, nxt);
        pe_f4 sc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM / 16; kd++) sc = mfma16(cur.k[kd], qb[kd], sc);
        float mx = m;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int kj = k0 + 4 * lq + j;
            sc[j] = kj < j1 ? sc[j] * scale : -INFINITY;
            mx = fmaxf(mx, sc[j]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 32, 64));
        const float c = expf(m - mx);  // m = -inf at first: 0
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            sc[j] = expf(sc[j] - mx);
            sum += sc[j];
        }
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        l = l * c + sum;
        m = mx;
        if (lr < G) {
#pragma unroll
            for (int j = 0; j < 4; j++) sp[wave][4 * lq + j][lr] = sc[j];
            if (lq == 0) sc_[wave][lr] = c;
        }
        wave_sync();
#pragma unroll
        for (int h = 0; h < G; h++) {
            const float ch = sc_[wave][h];
#pragma unroll
            for (int e = 0; e < 4; e++) o[h][e] *= ch;
        }
        // Keys past j1 have probability 0 (their scores were -inf).
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const uint2 vr = cur.v[kk];
            const float v[4] = {__uint_as_float(vr.x << 16), __uint_as_float(vr.x & 0xffff0000u),
                                __uint_as_float(vr.y << 16), __uint_as_float(vr.y & 0xffff0000u)};
#pragma unroll
            for (int h = 0; h < G; h++) {
                const float p = sp[wave][kk][h];
#pragma unroll
                for (int e = 0; e < 4; e++) o[h][e] = fmaf(p, v[e], o[h][e]);
            }
        }
        wave_sync();  // sp and sc_ are rewritten by the next tile
        cur = nxt;
    }
    if (lr < G && lq == 0) {
        sm_m[wave][lr] = m;
        sm_s[wave][lr] = l;
    }
#pragma unroll
    for (int h = 0; h < G; h++)
#pragma unroll
        for (int e = 0; e < 4; e++) sm_acc[wave][h][4 * lane + e] = o[h][e];
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += WARPS * 32) {
#pragma unroll
        for (int g = 0; g < G; g++) {
            float M = -INFINITY;
#pragma unroll
            for (int k = 0; k < NW; k++) M = fmaxf(M, sm_m[k][g]);
            float S = 0.f, A = 0.f;
#pragma unroll
            for (int k = 0; k < NW; k++) {
                const float f = sm_m[k][g] == -INFINITY ? 0.f : expf(sm_m[k][g] - M);
                S += sm_s[k][g] * f;
                A += sm_acc[k][g][d] * f;
            }
            const int h = kvh * G + g;
            if (part) {
                const size_t row = (size_t)sq * spec + (t - cu[sq]);
                float* pp = part + ((row * Hq + h) * gridDim.z + blockIdx.z) * (HEAD_DIM + 2);
                if (d == 0) {
                    pp[0] = M;
                    pp[1] = S;
                }
                pp[2 + d] = A;
                continue;
            }
            const float ov = rbf(A / S);
            const float gate = b2f(qkv[(size_t)t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d]);
            const float y = ov / (1.f + expf(-gate));
            out[((size_t)t * Hq + h) * HEAD_DIM + d] = f2b(y);
            if (fmode) sm_acc[0][g][d] = y;
        }
    }
    if (fmode && !part) {
        __syncthreads();
        const Fp8Out fo{fq, fs, fmode, frows};
        const int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
        for (int g = w; g < G; g += WARPS) {
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] = sm_acc[0][g][8 * ln + i];
            quant8(v, t, (kvh * G + g) * HEAD_DIM + 8 * ln, Hq * HEAD_DIM, fo, true);
        }
    }
}
