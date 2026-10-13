// Qwen3.5 text decoder kernels. One step processes a flat batch of tokens drawn
// from several sequences: decode rows contribute one token each, prefill rows a
// chunk of their prompt. Per-token work (norms, projections, attention) runs
// over every token; the convolution and the gated delta rule walk each
// sequence's tokens in order.
//
// Activations are row-major bf16 [tokens, width]. Reductions run in f32, and
// values are rounded to bf16 where the reference model rounds them.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define GDN_DIM 128   // linear-attention key and value head width
#define HEAD_DIM 256  // full-attention head width
#define ROPE_DIM 64   // rotated leading dimensions of each attention head
#define ATTN_WARPS 8

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rbf(float x) { return b2f(f2b(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum across the block; every thread receives the total. `sh` holds one float
// per warp.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    v = warp_sum(v);
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = l < nw ? sh[l] : 0.f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}

// Max across the block, like block_sum.
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = l < nw ? sh[l] : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    __syncthreads();
    return t;
}

__device__ __forceinline__ void unpack8(uint4 u, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 x = __bfloat1622float2(p[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}

__device__ __forceinline__ void unpack4(uint2 u, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
    const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
    f[0] = a.x;
    f[1] = a.y;
    f[2] = b.x;
    f[3] = b.y;
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 u;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

#include "fp8.cuh"

// out[t] = table[ids[t]], or row emb_src[t] of the image embedding pool when
// that is not negative.
extern "C" __global__ void embed(const int* __restrict__ ids, const bf16* __restrict__ table,
                                 const int* __restrict__ emb_src, const bf16* __restrict__ pool,
                                 bf16* __restrict__ out, int hidden) {
    const int t = blockIdx.x;
    const int e = emb_src[t];
    const uint4* src = reinterpret_cast<const uint4*>(
        e >= 0 ? pool + (size_t)e * hidden : table + (size_t)ids[t] * hidden);
    uint4* dst = reinterpret_cast<uint4*>(out + (size_t)t * hidden);
    for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) dst[i] = src[i];
}

// RMSNorm with an f32 weight (zero-centred weights already offset by one).
//   add == 0: out[r] = norm(res[src])
//   add == 1: sum = x[src] + res[src]; out[r] = norm(sum); res[src] = bf16(sum)
//             unless write_res == 0.
// The norm uses the unrounded f32 sum. `gather` (nullable) maps output rows to
// source rows. Up to 4 * 8 * blockDim.x elements per row. With fmode set,
// the normalized row is also written as FP8 (see fp8.cuh).
extern "C" __global__ void rmsnorm(const bf16* __restrict__ x, bf16* __restrict__ res,
                                   const float* __restrict__ w, bf16* __restrict__ out,
                                   const int* __restrict__ gather, int hidden, float eps, int add,
                                   int write_res, uint8_t* fq, void* fs, int fmode, int frows) {
    __shared__ float sh[32];
    const int r = blockIdx.x;
    const size_t src = gather ? (size_t)gather[r] : (size_t)r;
    float v[4][8];
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < hidden) {
            unpack8(*reinterpret_cast<const uint4*>(res + src * hidden + i), v[c]);
            if (add) {
                float a[8];
                unpack8(*reinterpret_cast<const uint4*>(x + src * hidden + i), a);
#pragma unroll
                for (int e = 0; e < 8; e++) v[c][e] += a[e];
                if (write_res) *reinterpret_cast<uint4*>(res + src * hidden + i) = pack8(v[c]);
            }
#pragma unroll
            for (int e = 0; e < 8; e++) ss += v[c][e] * v[c][e];
        }
    }
    ss = block_sum(ss, sh);
    const float scale = 1.f / sqrtf(ss / hidden + eps);
    const Fp8Out fo{fq, fs, fmode, frows};
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        // hidden is a multiple of 256, so whole warps are in or out of range.
        if (c * blockDim.x * 8 >= (unsigned)hidden) break;
        float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (i < hidden) {
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = v[c][e] * scale * w[i + e];
            *reinterpret_cast<uint4*>(out + (size_t)r * hidden + i) = pack8(o);
        }
        if (fmode) quant8(o, r, i, hidden, fo, i < hidden);
    }
}

// out[t, c] = bf16(silu(gate) * up) where gu[t] = [gate(inter) | up(inter)],
// and/or the same as FP8 (see fp8.cuh); out may be null.
extern "C" __global__ void silu_mul(const bf16* __restrict__ gu, bf16* __restrict__ out, int inter,
                                    int tokens, uint8_t* fq, void* fs, int fmode, int frows) {
    const Fp8Out fo{fq, fs, fmode, frows};
    const int per_row = inter / 8;
    const long long n = (long long)tokens * per_row;
    const long long stride = (long long)gridDim.x * blockDim.x;
    // Warp-uniform trip count: FP8 scale groups reduce across lanes.
    for (long long base = (long long)blockIdx.x * blockDim.x; base < n; base += stride) {
        const long long e = base + threadIdx.x;
        const bool active = e < n;
        float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int t = 0, c = 0;
        if (active) {
            t = (int)(e / per_row);
            c = (int)(e - (long long)t * per_row) * 8;
            float g[8], u[8];
            unpack8(*reinterpret_cast<const uint4*>(gu + (size_t)t * 2 * inter + c), g);
            unpack8(*reinterpret_cast<const uint4*>(gu + (size_t)t * 2 * inter + inter + c), u);
#pragma unroll
            for (int i = 0; i < 8; i++) o[i] = silu(g[i]) * u[i];
            if (out) *reinterpret_cast<uint4*>(out + (size_t)t * inter + c) = pack8(o);
        }
        if (fmode) quant8(o, t, c, inter, fo, active);
    }
}

// Depthwise causal convolution (kernel 4) over each sequence's tokens, then
// SiLU. The state holds the last three inputs of each sequence: [slot][3][C].
// proj rows hold the convolution inputs in their first C columns.
extern "C" __global__ void gdn_conv(const bf16* __restrict__ proj, int pstride,
                                    const float* __restrict__ w, bf16* __restrict__ state,
                                    bf16* __restrict__ out, const int* __restrict__ cu,
                                    const int* __restrict__ slots, const int* __restrict__ fresh,
                                    int C) {
    const int s = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    bf16* st = state + (size_t)slots[s] * 3 * C;
    float x0 = 0.f, x1 = 0.f, x2 = 0.f;
    if (!fresh[s]) {
        x0 = b2f(st[c]);
        x1 = b2f(st[C + c]);
        x2 = b2f(st[2 * C + c]);
    }
    const float4 k = *reinterpret_cast<const float4*>(w + 4 * c);
    // Inputs are loaded 8 tokens at a time, so a prompt waits for one load
    // latency per 8 tokens rather than per token.
    for (int t = t0; t < t1; t += 8) {
        float xs[8];
#pragma unroll
        for (int u = 0; u < 8; u++) xs[u] = t + u < t1 ? b2f(proj[(size_t)(t + u) * pstride + c]) : 0.f;
#pragma unroll
        for (int u = 0; u < 8; u++) {
            if (t + u >= t1) break;
            const float x3 = xs[u];
            float acc = 0.f;
            acc += x0 * k.x;
            acc += x1 * k.y;
            acc += x2 * k.z;
            acc += x3 * k.w;
            out[(size_t)(t + u) * C + c] = f2b(silu(acc));
            x0 = x1;
            x1 = x2;
            x2 = x3;
        }
    }
    st[c] = f2b(x0);
    st[C + c] = f2b(x1);
    st[2 * C + c] = f2b(x2);
}

// Gated delta rule, one block of 256 threads per (sequence, value head). The
// head's [k, v] state lives in registers: threads 2j and 2j+1 hold rows 0..63
// and 64..127 of column j. Per token:
//   S *= exp(g); delta = (v - k^T S) * beta; S += k delta^T; o = q^T S
// then a gated RMSNorm of o with silu(z). The state is [slot][Hv][128][128],
// f32 or (state_bf16) bf16; the math is f32 either way.
// conv rows: [q (Hk*128) | k (Hk*128) | v (Hv*128)]; proj rows after the
// convolution inputs: [z (Hv*128) | b (Hv) | a (Hv)].
#define GDN_HALF (GDN_DIM / 2)
__device__ __forceinline__ void gdn_one(int s, int h, const bf16* __restrict__ conv, const bf16* __restrict__ proj,
                                        int pstride, int C, int Hk, int Hv, const float* __restrict__ neg_a,
                                        const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
                                        void* __restrict__ state, int state_bf16, const int* __restrict__ cu,
                                        const int* __restrict__ slots, const int* __restrict__ fresh,
                                        bf16* __restrict__ out, float eps, int tiled, uint8_t* fq, void* fs,
                                        int fmode, int frows) {
    __shared__ float qs[GDN_DIM], ks[GDN_DIM], sh[32], ys[GDN_DIM];
    const Fp8Out fo{fq, fs, fmode, frows};
    const int tid = threadIdx.x;
    const int j = tid >> 1, half = tid & 1, i0 = half * GDN_HALF;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    // Value head h reads key head h / (Hv / Hk) (grouped order), or h % Hk in
    // the tiled order GGUF files use.
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float* S = (float*)state + base;
    bf16* Sb = (bf16*)state + base;
    float r[GDN_HALF];
    const bool is_fresh = fresh[s] != 0;
#pragma unroll
    for (int i = 0; i < GDN_HALF; i++) {
        const int at = (i0 + i) * GDN_DIM + j;
        r[i] = is_fresh ? 0.f : state_bf16 ? b2f(Sb[at]) : S[at];
    }
    // Prefill rounds the normalized q and k to bf16; single-token decode keeps f32.
    const bool round_qk = (t1 - t0) > 1 || is_fresh;
    const float na = neg_a[h], db = dt_bias[h], nw = norm_w[j];
    const float qscale = rsqrtf((float)GDN_DIM);
    for (int t = t0; t < t1; t++) {
        const bf16* row = conv + (size_t)t * C;
        const bf16* prow = proj + (size_t)t * pstride + C;
        float q = 0.f, k = 0.f;
        if (tid < GDN_DIM) {
            q = b2f(row[kh * GDN_DIM + tid]);
            k = b2f(row[Hk * GDN_DIM + kh * GDN_DIM + tid]);
        }
        const float v = b2f(row[2 * Hk * GDN_DIM + h * GDN_DIM + j]);
        const float z = b2f(prow[h * GDN_DIM + j]);
        const float b = b2f(prow[Hv * GDN_DIM + h]);
        const float a = b2f(prow[Hv * GDN_DIM + Hv + h]);
        const float qq = block_sum(q * q, sh);
        const float kk = block_sum(k * k, sh);
        if (tid < GDN_DIM) {
            q = q / sqrtf(qq + 1e-6f);
            k = k / sqrtf(kk + 1e-6f);
            if (round_qk) {
                q = rbf(q);
                k = rbf(k);
            }
            qs[tid] = q * qscale;
            ks[tid] = k;
        }
        const float beta = 1.f / (1.f + expf(-b));
        const float x = a + db;
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        const float decay = expf(na * sp);
        __syncthreads();
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < GDN_HALF; i++) {
            r[i] *= decay;
            kv += ks[i0 + i] * r[i];
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        const float delta = (v - kv) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < GDN_HALF; i++) {
            r[i] += ks[i0 + i] * delta;
            o += qs[i0 + i] * r[i];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        const float ob = rbf(o);
        const float var = block_sum(half ? 0.f : ob * ob, sh) / GDN_DIM;
        if (!half) {
            const float y = ob * (1.f / sqrtf(var + eps)) * nw * silu(z);
            out[(size_t)t * Hv * GDN_DIM + h * GDN_DIM + j] = f2b(y);
            ys[j] = y;
        }
        if (fmode) {
            // The head's 128 outputs as FP8: 16 lanes of warp 0, 8 values each.
            __syncthreads();
            if (tid < 32) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = tid < 16 ? ys[8 * tid + i] : 0.f;
                quant8(v, t, h * GDN_DIM + 8 * (tid & 15), Hv * GDN_DIM, fo, tid < 16);
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < GDN_HALF; i++) {
        const int at = (i0 + i) * GDN_DIM + j;
        if (state_bf16) Sb[at] = f2b(r[i]);
        else S[at] = r[i];
    }
}

extern "C" __global__ void __launch_bounds__(2 * GDN_DIM)
    gdn_recurrent(const bf16* __restrict__ conv, const bf16* __restrict__ proj, int pstride, int C, int Hk, int Hv,
                  const float* __restrict__ neg_a, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
                  void* __restrict__ state, int state_bf16, const int* __restrict__ cu, const int* __restrict__ slots,
                  const int* __restrict__ fresh, bf16* __restrict__ out, float eps, int tiled, uint8_t* fq, void* fs,
                  int fmode, int frows) {
    gdn_one(blockIdx.x, blockIdx.y, conv, proj, pstride, C, Hk, Hv, neg_a, dt_bias, norm_w, state, state_bf16, cu,
            slots, fresh, out, eps, tiled, fq, fs, fmode, frows);
}

#define GDN_ARGS                                                                                       \
    const bf16 *__restrict__ conv, const bf16 *__restrict__ proj, int pstride, int C, int Hk, int Hv,  \
        const float *__restrict__ neg_a, const float *__restrict__ dt_bias,                            \
        const float *__restrict__ norm_w, void *__restrict__ state, int state_bf16,                    \
        const int *__restrict__ cu, const int *__restrict__ slots, const int *__restrict__ fresh,       \
        bf16 *__restrict__ out, float eps, int tiled, uint8_t *fq, void *fs, int fmode, int frows
#define GDN_PASS conv, proj, pstride, C, Hk, Hv, neg_a, dt_bias, norm_w, state, state_bf16, cu, slots, fresh, out, eps, tiled, fq, fs, fmode, frows

// The gated delta rule for prompts, in chunks of GC tokens on tensor cores
// (the chunked form FLA and FlashInfer use). With G the cumulative log decay
// inside a chunk, S0 the state before it and Gamma_ij = exp(G_i - G_j):
//   (I + A) D = beta * (V - exp(G) * K S0),  A_ij = beta_i Gamma_ij k_i.k_j (j < i)
//   O = exp(G) * Q S0 + tril(Q K^T * Gamma) D          (the diagonal included)
//   S = exp(G_last) S0 + (K * exp(G_last - G))^T D
// which is the token-by-token recurrence of gdn_one, exactly in exact
// arithmetic. The matrix products run on mma.sync with bf16 operands and f32
// accumulation; the triangular solve runs in f32, one thread per value
// column. One block of 8 warps per (sequence, head) walks its chunks; warp w
// keeps state rows 16w .. 16w + 15 in registers as mma accumulators, with a
// bf16 copy in shared memory for the products that read it. Sequences with
// one token take gdn_one's way. Outputs go to `out` unnormalized; gdn_norm
// finishes them.
#define GC 32
#define GSTR 136  // bf16 row stride of 128-wide tiles, padded against bank conflicts
#define GPSTR 40  // bf16 row stride of the 32-wide P tile
#define GRSTR 132 // f32 row stride of R
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ __forceinline__ void gmma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Fragment loads (lane l: q = l / 8, r = l % 8). A tile 16 x 16 at (m0, k0)
// from row-major X[m][k], or from X[k][m] (transposed); B as two 8-column
// tiles at (k0, n0) from Bt[n][k] or from B[k][n] (transposed).
__device__ __forceinline__ void frag_a(uint32_t* a, const bf16* X, int stride, int m0, int k0) {
    const int l = threadIdx.x & 31, q = l >> 3, r = l & 7;
    ldsm_x4(a, X + (size_t)(m0 + r + (q & 1) * 8) * stride + k0 + (q >> 1) * 8);
}
__device__ __forceinline__ void frag_a_t(uint32_t* a, const bf16* X, int stride, int m0, int k0) {
    const int l = threadIdx.x & 31, q = l >> 3, r = l & 7;
    ldsm_x4_t(a, X + (size_t)(k0 + r + (q >> 1) * 8) * stride + m0 + (q & 1) * 8);
}
__device__ __forceinline__ void frag_b(uint32_t* b, const bf16* Bt, int stride, int k0, int n0) {
    const int l = threadIdx.x & 31, q = l >> 3, r = l & 7;
    ldsm_x4(b, Bt + (size_t)(n0 + r + (q >> 1) * 8) * stride + k0 + (q & 1) * 8);
}
__device__ __forceinline__ void frag_b_t(uint32_t* b, const bf16* B, int stride, int k0, int n0) {
    const int l = threadIdx.x & 31, q = l >> 3, r = l & 7;
    ldsm_x4_t(b, B + (size_t)(k0 + r + (q & 1) * 8) * stride + n0 + (q >> 1) * 8);
}

// Shared memory of gdn_chunk, in bytes.
#define GDN_CHUNK_SMEM (4 * GC * GSTR * 2 + GDN_DIM * GSTR * 2 + GC * GRSTR * 4 + GC * (GC + 1) * 4 + GC * GPSTR * 2 + 4 * GC * 4)

extern "C" __global__ void __launch_bounds__(2 * GDN_DIM) gdn_chunk(GDN_ARGS) {
    extern __shared__ __align__(16) unsigned char gsm[];
    bf16* sQ = (bf16*)gsm;                      // [GC][GSTR] normalized q (unscaled)
    bf16* sK = sQ + GC * GSTR;                  // [GC][GSTR] normalized k
    bf16* sKw = sK + GC * GSTR;                 // [GC][GSTR] k * exp(G_last - G)
    bf16* sV = sKw + GC * GSTR;                 // [GC][GSTR] v, then D
    bf16* sS = sV + GC * GSTR;                  // [GDN_DIM][GSTR] the state, bf16
    float* sR = (float*)(sS + GDN_DIM * GSTR);  // [GC][GRSTR]
    float* sA = sR + GC * GRSTR;                // [GC][GC + 1]
    bf16* sP = (bf16*)(sA + GC * (GC + 1));     // [GC][GPSTR]
    float* sG = (float*)(sP + GC * GPSTR);      // [GC] cumulative log decay
    float* sB = sG + GC;                        // [GC] beta
    float* sg = sB + GC;                        // [GC] log decay
    const int s = gridDim.x - 1 - blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    if (t1 - t0 == 1) {
        gdn_one(s, h, conv, proj, pstride, C, Hk, Hv, neg_a, dt_bias, norm_w, state, state_bf16, cu, slots, fresh,
                out, eps, tiled, fq, fs, fmode, frows);
        return;
    }
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float* Sg = (float*)state + base;
    bf16* Sgb = (bf16*)state + base;
    const bool is_fresh = fresh[s] != 0;
    const float na = neg_a[h], db = dt_bias[h], qscale = rsqrtf((float)GDN_DIM);
    // The state, rows 16w + g (+8) and columns 8n + 2t (+1) of 16 tiles.
    float st[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = 16 * warp + g + (e >> 1) * 8, col = 8 * n + 2 * t + (e & 1);
            const size_t at = (size_t)row * GDN_DIM + col;
            st[n][e] = is_fresh ? 0.f : state_bf16 ? b2f(Sgb[at]) : Sg[at];
        }
    auto store_state_smem = [&]() {
#pragma unroll
        for (int n = 0; n < 16; n++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int row = 16 * warp + g + hf * 8, col = 8 * n + 2 * t;
                *reinterpret_cast<__nv_bfloat162*>(sS + (size_t)row * GSTR + col) =
                    __floats2bfloat162_rn(st[n][2 * hf], st[n][2 * hf + 1]);
            }
    };
    store_state_smem();
    bool zero_state = is_fresh;
    for (int c0 = t0; c0 < t1; c0 += GC) {
        const int n = min(GC, t1 - c0);
        // Inputs: a warp per token, four dimensions per lane; rows past n are zero.
        for (int w = warp; w < GC; w += 8) {
            float q[4] = {0, 0, 0, 0}, k[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, 0};
            if (w < n) {
                const bf16* row = conv + (size_t)(c0 + w) * C;
                unpack4(*reinterpret_cast<const uint2*>(row + kh * GDN_DIM + 4 * lane), q);
                unpack4(*reinterpret_cast<const uint2*>(row + Hk * GDN_DIM + kh * GDN_DIM + 4 * lane), k);
                unpack4(*reinterpret_cast<const uint2*>(row + 2 * Hk * GDN_DIM + h * GDN_DIM + 4 * lane), v);
            }
            float qq = 0.f, kk = 0.f;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                qq += q[e] * q[e];
                kk += k[e] * k[e];
            }
            qq = warp_sum(qq);
            kk = warp_sum(kk);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int d = 4 * lane + e;
                sQ[w * GSTR + d] = f2b(q[e] / sqrtf(qq + 1e-6f));
                sK[w * GSTR + d] = f2b(k[e] / sqrtf(kk + 1e-6f));
                sV[w * GSTR + d] = f2b(v[e]);
            }
            if (lane == 0) {
                float beta = 0.f, lg = 0.f;
                if (w < n) {
                    const bf16* prow = proj + (size_t)(c0 + w) * pstride + C;
                    const float b = b2f(prow[Hv * GDN_DIM + h]);
                    const float a = b2f(prow[Hv * GDN_DIM + Hv + h]);
                    const float x = a + db;
                    const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
                    beta = 1.f / (1.f + expf(-b));
                    lg = na * sp;
                }
                sB[w] = beta;
                sg[w] = lg;
            }
        }
        __syncthreads();
        if (warp == 0) {
            // Cumulative log decay (GC == 32: one value per lane).
            float x = sg[lane];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            sG[lane] = x;
        }
        __syncthreads();
        const float glast = sG[GC - 1];
        for (int e = tid; e < GC * GDN_DIM; e += 2 * GDN_DIM) {
            const int j = e / GDN_DIM, d = e % GDN_DIM;
            sKw[j * GSTR + d] = f2b(b2f(sK[j * GSTR + d]) * expf(glast - sG[j]));
        }
        // K K^T (warps 0-3) and Q K^T (warps 4-7), a 16 x 16 tile each.
        {
            const int m0 = (warp & 1) * 16, n0 = ((warp >> 1) & 1) * 16;
            const bf16* X = warp < 4 ? sK : sQ;
            float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
            for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
                uint32_t a[4], b[4];
                frag_a(a, X, GSTR, m0, k0);
                frag_b(b, sK, GSTR, k0, n0);
                gmma(acc[0], a, b[0], b[1]);
                gmma(acc[1], a, b[2], b[3]);
            }
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int i = m0 + g + (e >> 1) * 8, j = n0 + nt * 8 + 2 * t + (e & 1);
                    const float gam = j <= i ? expf(sG[i] - sG[j]) : 0.f;
                    if (warp < 4) sA[i * (GC + 1) + j] = j < i ? sB[i] * gam * acc[nt][e] : 0.f;
                    else sP[i * GPSTR + j] = f2b(qscale * gam * acc[nt][e]);
                }
        }
        // R = beta * (V - exp(G) * K S0): rows m0 .. m0 + 15, columns n0 .. n0 + 31 per warp.
        const int om0 = (warp & 1) * 16, on0 = (warp >> 1) * 32;
        {
            float acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
            if (!zero_state) {
#pragma unroll
                for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
                    uint32_t a[4], b0[4], b1[4];
                    frag_a(a, sK, GSTR, om0, k0);
                    frag_b_t(b0, sS, GSTR, k0, on0);
                    frag_b_t(b1, sS, GSTR, k0, on0 + 16);
                    gmma(acc[0], a, b0[0], b0[1]);
                    gmma(acc[1], a, b0[2], b0[3]);
                    gmma(acc[2], a, b1[0], b1[1]);
                    gmma(acc[3], a, b1[2], b1[3]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int i = om0 + g + (e >> 1) * 8, c = on0 + nt * 8 + 2 * t + (e & 1);
                    sR[i * GRSTR + c] = sB[i] * (b2f(sV[i * GSTR + c]) - expf(sG[i]) * acc[nt][e]);
                }
        }
        __syncthreads();
        // (I + A) D = R, by forward substitution: a thread per column.
        if (tid < GDN_DIM) {
            float d[GC];
#pragma unroll
            for (int i = 0; i < GC; i++) {
                float a0 = sR[i * GRSTR + tid], a1 = 0.f;
#pragma unroll
                for (int j = 0; j + 1 < i; j += 2) {
                    a0 -= sA[i * (GC + 1) + j] * d[j];
                    a1 -= sA[i * (GC + 1) + j + 1] * d[j + 1];
                }
                if (i & 1) a0 -= sA[i * (GC + 1) + i - 1] * d[i - 1];
                d[i] = a0 + a1;
            }
#pragma unroll
            for (int i = 0; i < GC; i++) sV[i * GSTR + tid] = f2b(d[i]);
        }
        __syncthreads();
        // O = exp(G) * qscale * Q S0 + P D, written unnormalized.
        {
            float acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
            if (!zero_state) {
#pragma unroll
                for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
                    uint32_t a[4], b0[4], b1[4];
                    frag_a(a, sQ, GSTR, om0, k0);
                    frag_b_t(b0, sS, GSTR, k0, on0);
                    frag_b_t(b1, sS, GSTR, k0, on0 + 16);
                    gmma(acc[0], a, b0[0], b0[1]);
                    gmma(acc[1], a, b0[2], b0[3]);
                    gmma(acc[2], a, b1[0], b1[1]);
                    gmma(acc[3], a, b1[2], b1[3]);
                }
                const float e0 = qscale * expf(sG[om0 + g]), e1 = qscale * expf(sG[om0 + g + 8]);
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    acc[nt][0] *= e0;
                    acc[nt][1] *= e0;
                    acc[nt][2] *= e1;
                    acc[nt][3] *= e1;
                }
            }
#pragma unroll
            for (int k0 = 0; k0 < GC; k0 += 16) {
                uint32_t a[4], b0[4], b1[4];
                frag_a(a, sP, GPSTR, om0, k0);
                frag_b_t(b0, sV, GSTR, k0, on0);
                frag_b_t(b1, sV, GSTR, k0, on0 + 16);
                gmma(acc[0], a, b0[0], b0[1]);
                gmma(acc[1], a, b0[2], b0[3]);
                gmma(acc[2], a, b1[0], b1[1]);
                gmma(acc[3], a, b1[2], b1[3]);
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int i = om0 + g + hf * 8, c = on0 + nt * 8 + 2 * t;
                    if (i < n)
                        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(c0 + i) * Hv * GDN_DIM + h * GDN_DIM + c) =
                            __floats2bfloat162_rn(acc[nt][2 * hf], acc[nt][2 * hf + 1]);
                }
        }
        // S = exp(G_last) S0 + Kw^T D: this warp's 16 rows, all 128 columns.
        {
            const float decay = expf(glast);
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) st[nt][e] *= decay;
#pragma unroll
            for (int k0 = 0; k0 < GC; k0 += 16) {
                uint32_t a[4];
                frag_a_t(a, sKw, GSTR, 16 * warp, k0);
#pragma unroll
                for (int np = 0; np < 8; np++) {
                    uint32_t b[4];
                    frag_b_t(b, sV, GSTR, k0, 16 * np);
                    gmma(st[2 * np], a, b[0], b[1]);
                    gmma(st[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
        store_state_smem();
        zero_state = false;
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < 16; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = 16 * warp + g + (e >> 1) * 8, col = 8 * nt + 2 * t + (e & 1);
            const size_t at = (size_t)row * GDN_DIM + col;
            if (state_bf16) Sgb[at] = f2b(st[nt][e]);
            else Sg[at] = st[nt][e];
        }
}

// The gated RMSNorm of one (token, head): lane l < 16 holds outputs 8l ..
// 8l + 7 in o; y = o / rms(o) * w * silu(z), written as bf16 and, with
// fmode, as FP8. Every lane of the warp calls it.
__device__ __forceinline__ void gdn_gate8(float* o, int t, int h, int Hv, const bf16* __restrict__ proj, int pstride,
                                          int C, const float* __restrict__ norm_w, bf16* __restrict__ out, float eps,
                                          const Fp8Out& fo, int fmode, int lane) {
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) ss += o[e] * o[e];
    const float var = warp_sum(ss) / GDN_DIM;
    const float inv = 1.f / sqrtf(var + eps);
    if (lane < 16) {
        const bf16* prow = proj + (size_t)t * pstride + C;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int c = 8 * lane + e;
            o[e] = o[e] * inv * norm_w[c] * silu(b2f(prow[h * GDN_DIM + c]));
        }
        *reinterpret_cast<uint4*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + 8 * lane) = pack8(o);
    }
    if (fmode) quant8(o, t, h * GDN_DIM + 8 * (lane & 15), Hv * GDN_DIM, fo, lane < 16);
}
// The gated norm for gdn_chunk's outputs, in place: a warp per (token,
// head), over the tokens of sequences with more than one.
extern "C" __global__ void gdn_norm(const bf16* __restrict__ proj, int pstride, int C, int Hv,
                                    const float* __restrict__ norm_w, bf16* __restrict__ out, int T, float eps,
                                    const int* __restrict__ cu, const int* __restrict__ tok_seq, uint8_t* fq,
                                    void* fs, int fmode, int frows) {
    const Fp8Out fo{fq, fs, fmode, frows};
    const int lane = threadIdx.x & 31;
    const long long n = (long long)T * Hv;
    for (long long e = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; e < n;
         e += ((long long)gridDim.x * blockDim.x) >> 5) {
        const int t = (int)(e / Hv), h = (int)(e % Hv);
        const int sq = tok_seq[t];
        if (cu[sq + 1] - cu[sq] == 1) continue;  // finished by gdn_one
        float o[8];
        if (lane < 16) unpack8(*reinterpret_cast<const uint4*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + 8 * lane), o);
        else
            for (int i = 0; i < 8; i++) o[i] = 0.f;
        gdn_gate8(o, t, h, Hv, proj, pstride, C, norm_w, out, eps, fo, fmode, lane);
    }
}

// One head's 256 values (v0 from thread i, v1 from i + 128) as E4M3 with a
// scale per (token, head): amax / 448, so the largest maps to E4M3's largest.
__device__ __forceinline__ void store_e4m3(float v0, float v1, uint8_t* dst, float* scale, float* sh) {
    const float amax = block_max(fmaxf(fabsf(v0), fabsf(v1)), sh);
    const float sc = amax > 0.f ? amax / 448.f : 1.f;
    const float inv = 1.f / sc;
    const int i = threadIdx.x;
    dst[i] = to_e4m3(v0 * inv);
    dst[i + 128] = to_e4m3(v1 * inv);
    if (i == 0) *scale = sc;
}

// Full-attention input preparation, one block of 128 threads per (token, head).
// qkv rows: [Hq x (q 256 | gate 256) | k (Hkv*256) | v (Hkv*256)].
// q heads: RMSNorm, rotary -> qout [t][Hq][256]. k heads: RMSNorm, rotary ->
// kcache[slot]. v heads: copied to vcache[slot]. Slots below zero are skipped.
// Rotary covers dims 0..63 as pairs (d, d + 32) with bf16 cos/sin, rounding as
// the bf16 reference does. Frequency d takes its position from one of three
// axes (time, height, width; interleaved multimodal rotary): height for
// d % 3 == 1 below 3 * sec_h, width for d % 3 == 2 below 3 * sec_w, else time.
// Text tokens carry the same position on all three.
extern "C" __global__ void __launch_bounds__(128)
    attn_prep(const bf16* __restrict__ qkv, int qstride, int Hq, int Hkv,
              const float* __restrict__ q_norm, const float* __restrict__ k_norm,
              const int* __restrict__ rope, int rope_stride, int sec_h, int sec_w,
              const int* __restrict__ slot_map, const float* __restrict__ inv_freq,
              bf16* __restrict__ qout, void* __restrict__ kcache, void* __restrict__ vcache,
              float* __restrict__ kscale, float* __restrict__ vscale, float eps) {
    __shared__ float sh[32], rot[ROPE_DIM];
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    const bf16* row = qkv + (size_t)t * qstride;
    const int slot = slot_map[t];
    if (hh >= Hq + Hkv) {
        if (slot < 0) return;
        const int vh = hh - Hq - Hkv;
        const bf16* src = row + Hq * 2 * HEAD_DIM + Hkv * HEAD_DIM + vh * HEAD_DIM;
        const size_t at = (size_t)slot * Hkv + vh;
        if (vscale) {
            store_e4m3(b2f(src[i]), b2f(src[i + 128]), (uint8_t*)vcache + at * HEAD_DIM, vscale + at, sh);
            return;
        }
        bf16* dst = (bf16*)vcache + at * HEAD_DIM;
        dst[i] = src[i];
        dst[i + 128] = src[i + 128];
        return;
    }
    const bool is_q = hh < Hq;
    if (!is_q && slot < 0) return;
    const bf16* src = is_q ? row + hh * 2 * HEAD_DIM : row + Hq * 2 * HEAD_DIM + (hh - Hq) * HEAD_DIM;
    const float* w = is_q ? q_norm : k_norm;
    const float x0 = b2f(src[i]), x1 = b2f(src[i + 128]);
    const float ss = block_sum(x0 * x0 + x1 * x1, sh);
    const float scale = 1.f / sqrtf(ss / HEAD_DIM + eps);
    float y0 = rbf(x0 * scale * w[i]);
    const float y1 = rbf(x1 * scale * w[i + 128]);
    if (i < ROPE_DIM) rot[i] = y0;
    __syncthreads();
    if (i < ROPE_DIM) {
        const int d = i & (ROPE_DIM / 2 - 1);
        const int axis = (d % 3 == 1 && d < 3 * sec_h) ? 1 : (d % 3 == 2 && d < 3 * sec_w) ? 2 : 0;
        const float f = (float)rope[axis * rope_stride + t] * inv_freq[d];
        const float c = rbf(cosf(f)), s = rbf(sinf(f));
        const float lo = rot[d], hi = rot[d + ROPE_DIM / 2];
        y0 = i < ROPE_DIM / 2 ? rbf(rbf(lo * c) - rbf(hi * s)) : rbf(rbf(hi * c) + rbf(lo * s));
    }
    const size_t at = (size_t)slot * Hkv + (hh - Hq);
    if (!is_q && kscale) {
        store_e4m3(y0, y1, (uint8_t*)kcache + at * HEAD_DIM, kscale + at, sh);
        return;
    }
    bf16* dst = is_q ? qout + ((size_t)t * Hq + hh) * HEAD_DIM : (bf16*)kcache + at * HEAD_DIM;
    dst[i] = f2b(y0);
    dst[i + 128] = f2b(y1);
}

// Paged attention for one token against its sequence's first pos+1 keys, one
// block per (token, kv head) covering the G query heads that share it. Keys
// are split across warps with an online softmax per warp, then merged. The
// output is gated: out = bf16(bf16(attn) * sigmoid(gate)).
// Cache layout: [block * block_size + offset][Hkv][256], bf16, or E4M3 (F8)
// with an f32 scale per [slot][Hkv] that folds into the dot product and the
// weighted sum of values.
template <int G, int WARPS, bool F8>
__device__ __forceinline__ void attn_body(const bf16* __restrict__ q, const void* __restrict__ kcache,
                                          const void* __restrict__ vcache,
                                          const float* __restrict__ kscale,
                                          const float* __restrict__ vscale, const bf16* __restrict__ qkv,
                                          int qstride, int Hq, int Hkv, const int* __restrict__ pos,
                                          const int* __restrict__ tok_seq,
                                          const int* __restrict__ block_table, int max_blocks,
                                          int block_size, bf16* __restrict__ out, float scale,
                                          uint8_t* fq, void* fs, int fmode, int frows,
                                          const int* __restrict__ cu, int skip_prompts, int zchunk,
                                          float* __restrict__ part) {
    __shared__ float sm_m[WARPS][G], sm_s[WARPS][G];
    __shared__ float sm_acc[WARPS][G][HEAD_DIM];
    const int t = blockIdx.x, kvh = blockIdx.y;
    // With skip_prompts, tokens of sequences with more than one (prompts)
    // are left to attn_flash.
    if (skip_prompts && cu[tok_seq[t] + 1] - cu[tok_seq[t]] > 1) return;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int L = pos[t] + 1;
    // With `part`, keys split into ranges of zchunk across blockIdx.z
    // (flash decoding); the ranges' partial results go to `part` for
    // attn_merge.
    const int j0 = part ? blockIdx.z * zchunk : 0, j1 = part ? min(L, j0 + zchunk) : L;
    if (j0 >= j1) return;
    const int* bt = block_table + (size_t)tok_seq[t] * max_blocks;
    float qr[G][8], acc[G][8], m[G], ssum[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        unpack8(*reinterpret_cast<const uint4*>(q + ((size_t)t * Hq + kvh * G + g) * HEAD_DIM + l * 8),
                qr[g]);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            qr[g][e] *= scale;
            acc[g][e] = 0.f;
        }
        m[g] = -INFINITY;
        ssum[g] = 0.f;
    }
    for (int j = j0 + w; j < j1; j += WARPS) {
        const size_t slot = (size_t)bt[j / block_size] * block_size + (j % block_size);
        const size_t at = slot * Hkv + kvh;
        const size_t off = at * HEAD_DIM + l * 8;
        float kf[8], vf[8], ks = 1.f, vs = 1.f;
        if constexpr (F8) {
            unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)kcache + off), kf);
            unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)vcache + off), vf);
            ks = kscale[at];
            vs = vscale[at];
        } else {
            unpack8(*reinterpret_cast<const uint4*>((const bf16*)kcache + off), kf);
            unpack8(*reinterpret_cast<const uint4*>((const bf16*)vcache + off), vf);
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            float d = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) d += qr[g][e] * kf[e];
            d = warp_sum(d) * ks;
            const float mn = fmaxf(m[g], d);
            const float corr = expf(m[g] - mn);
            const float p = expf(d - mn);
            const float pv = p * vs;
            ssum[g] = ssum[g] * corr + p;
#pragma unroll
            for (int e = 0; e < 8; e++) acc[g][e] = acc[g][e] * corr + pv * vf[e];
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (l == 0) {
            sm_m[w][g] = m[g];
            sm_s[w][g] = ssum[g];
        }
#pragma unroll
        for (int e = 0; e < 8; e++) sm_acc[w][g][l * 8 + e] = acc[g][e];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += WARPS * 32) {
#pragma unroll
        for (int g = 0; g < G; g++) {
            float M = -INFINITY;
#pragma unroll
            for (int k = 0; k < WARPS; k++) M = fmaxf(M, sm_m[k][g]);
            float S = 0.f, A = 0.f;
#pragma unroll
            for (int k = 0; k < WARPS; k++) {
                const float f = sm_m[k][g] == -INFINITY ? 0.f : expf(sm_m[k][g] - M);
                S += sm_s[k][g] * f;
                A += sm_acc[k][g][d] * f;
            }
            const int h = kvh * G + g;
            if (part) {
                float* pp = part + (((size_t)tok_seq[t] * Hq + h) * gridDim.z + blockIdx.z) * (HEAD_DIM + 2);
                if (d == 0) {
                    pp[0] = M;
                    pp[1] = S;
                }
                pp[2 + d] = A;
                continue;
            }
            const float o = rbf(A / S);
            const float gate = b2f(qkv[(size_t)t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d]);
            const float y = o / (1.f + expf(-gate));
            out[((size_t)t * Hq + h) * HEAD_DIM + d] = f2b(y);
            if (fmode) sm_acc[0][g][d] = y;  // reused: the merge above has read it
        }
    }
    if (fmode && !part) {
        // Each head's 256 gated outputs as FP8, one warp per head.
        __syncthreads();
        const Fp8Out fo{fq, fs, fmode, frows};
        const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
        for (int g = w; g < G; g += WARPS) {
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] = sm_acc[0][g][8 * l + i];
            quant8(v, t, (kvh * G + g) * HEAD_DIM + 8 * l, Hq * HEAD_DIM, fo, true);
        }
    }
}

// Query heads per kv head (G) select the kernel; the per-warp merge buffers
// must fit in 48 KB of shared memory, so larger groups use fewer warps.
#define ATTN_KERNEL_AS(NAME, G, WARPS, F8)                                                             \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) NAME(                                     \
        const bf16* __restrict__ q, const void* __restrict__ kcache, const void* __restrict__ vcache,  \
        const float* __restrict__ kscale, const float* __restrict__ vscale,                            \
        const bf16* __restrict__ qkv, int qstride, int Hq, int Hkv, const int* __restrict__ pos,      \
        const int* __restrict__ tok_seq, const int* __restrict__ block_table, int max_blocks,         \
        int block_size, bf16* __restrict__ out, float scale, uint8_t* fq, void* fs, int fmode,         \
        int frows, const int* __restrict__ cu, int skip_prompts, int zchunk, float* part) {            \
        attn_body<G, WARPS, F8>(q, kcache, vcache, kscale, vscale, qkv, qstride, Hq, Hkv, pos,         \
                                tok_seq, block_table, max_blocks, block_size, out, scale, fq, fs,      \
                                fmode, frows, cu, skip_prompts, zchunk, part);                         \
    }
// attn_paged_g<G> reads a bf16 cache, attn_paged8_g<G> an E4M3 one.
#define ATTN_KERNEL(G, WARPS)                                                                          \
    ATTN_KERNEL_AS(attn_paged_g##G, G, WARPS, false)                                                   \
    ATTN_KERNEL_AS(attn_paged8_g##G, G, WARPS, true)
ATTN_KERNEL(1, 8)
ATTN_KERNEL(2, 8)
ATTN_KERNEL(4, 8)
ATTN_KERNEL(6, 4)
ATTN_KERNEL(8, 4)

// Attention for prompt tokens, on tensor cores (flash attention over the
// paged cache). A block of 4 warps takes 64 query tokens of one sequence and
// one query head; each warp owns 16 of them. Keys and values stream through
// shared memory 32 at a time (gathered by the block table, and converted from
// E4M3 with F8); S = Q K^T runs on mma.sync, the softmax is online, and the
// probabilities go straight from S's accumulators into P V as A fragments.
// The output is gated as attn_body gates it. Blocks walk the step's prompt
// sequences (those with more than one token) in tiles of 64; others exit.
#define FA_BM 64
#define FA_BN 32
#define FA_STR (HEAD_DIM + 8)
#define FA_SMEM ((FA_BM + 2 * FA_BN) * FA_STR * 2)
__device__ __forceinline__ uint32_t fa_bf2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
template <bool F8>
__device__ __forceinline__ void attn_flash_body(
    const bf16* __restrict__ q, const void* __restrict__ kcache, const void* __restrict__ vcache,
    const float* __restrict__ kscale, const float* __restrict__ vscale, const bf16* __restrict__ qkv, int qstride,
    int Hq, int Hkv, const int* __restrict__ pos, const int* __restrict__ cu, int seqs,
    const int* __restrict__ block_table, int max_blocks, int block_size, bf16* __restrict__ out, float scale) {
    extern __shared__ __align__(16) unsigned char fsm[];
    bf16* sQ = (bf16*)fsm;
    bf16* sK = sQ + FA_BM * FA_STR;
    bf16* sV = sK + FA_BN * FA_STR;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int h = blockIdx.y, kvh = h / (Hq / Hkv);
    // Which (sequence, tile) this block takes.
    int s = -1, tile = blockIdx.x;
    for (int i = 0; i < seqs; i++) {
        const int n = cu[i + 1] - cu[i];
        if (n <= 1) continue;
        const int tiles = (n + FA_BM - 1) / FA_BM;
        if (tile < tiles) {
            s = i;
            break;
        }
        tile -= tiles;
    }
    if (s < 0) return;
    const int t0 = cu[s] + tile * FA_BM, n = min(FA_BM, cu[s + 1] - t0);
    const int p0 = pos[t0];  // positions are consecutive within a chunk
    const int L = p0 + n;    // keys 0 .. L - 1
    const int* bt = block_table + (size_t)s * max_blocks;
    // Queries, pre-scaled (by a power of two: exact).
    for (int e = tid; e < FA_BM * (HEAD_DIM / 8); e += 128) {
        const int r = e / (HEAD_DIM / 8), c = (e % (HEAD_DIM / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < n) {
            float f[8];
            unpack8(*reinterpret_cast<const uint4*>(q + ((size_t)(t0 + r) * Hq + h) * HEAD_DIM + c), f);
#pragma unroll
            for (int i = 0; i < 8; i++) f[i] *= scale;
            v = pack8(f);
        }
        *reinterpret_cast<uint4*>(sQ + r * FA_STR + c) = v;
    }
    float o[HEAD_DIM / 8][4];
#pragma unroll
    for (int nt = 0; nt < HEAD_DIM / 8; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int r0 = warp * 16 + g, r1 = r0 + 8;  // this lane's rows
    const int last_key = min(L, p0 + warp * 16 + 16);  // past this warp's last row's keys
    for (int k0 = 0; k0 < L; k0 += FA_BN) {
        __syncthreads();
        // Keys and values for positions k0 .. k0 + 31.
        for (int e = tid; e < FA_BN * (HEAD_DIM / 8); e += 128) {
            const int j = e / (HEAD_DIM / 8), c = (e % (HEAD_DIM / 8)) * 8, kp = k0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp < L) {
                const size_t slot = (size_t)bt[kp / block_size] * block_size + kp % block_size;
                const size_t at = (slot * Hkv + kvh) * HEAD_DIM + c;
                if constexpr (F8) {
                    float kf[8], vf[8];
                    unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)kcache + at), kf);
                    unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)vcache + at), vf);
                    const float ks = kscale[slot * Hkv + kvh], vs = vscale[slot * Hkv + kvh];
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        kf[i] *= ks;
                        vf[i] *= vs;
                    }
                    kv = pack8(kf);
                    vv = pack8(vf);
                } else {
                    kv = *reinterpret_cast<const uint4*>((const bf16*)kcache + at);
                    vv = *reinterpret_cast<const uint4*>((const bf16*)vcache + at);
                }
            }
            *reinterpret_cast<uint4*>(sK + j * FA_STR + c) = kv;
            *reinterpret_cast<uint4*>(sV + j * FA_STR + c) = vv;
        }
        __syncthreads();
        if (k0 >= last_key) continue;  // every key here is past this warp's rows
        // S = Q K^T for this warp's 16 rows and 32 keys.
        float sc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM; kd += 16) {
            uint32_t a[4], b0[4], b1[4];
            frag_a(a, sQ, FA_STR, warp * 16, kd);
            frag_b(b0, sK, FA_STR, kd, 0);
            frag_b(b1, sK, FA_STR, kd, 16);
            gmma(sc[0], a, b0[0], b0[1]);
            gmma(sc[1], a, b0[2], b0[3]);
            gmma(sc[2], a, b1[0], b1[1]);
            gmma(sc[3], a, b1[2], b1[3]);
        }
        // Causal mask and online softmax; rows r0 (sc[.][0..1]) and r1 ([2..3]).
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                const int qp = p0 + (e < 2 ? r0 : r1);
                if (kp > qp || kp >= L) sc[nt][e] = -INFINITY;
                if (e < 2) mx0 = fmaxf(mx0, sc[nt][e]);
                else mx1 = fmaxf(mx1, sc[nt][e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = mx0 == -INFINITY ? 1.f : expf(m0 - mx0), c1 = mx1 == -INFINITY ? 1.f : expf(m1 - mx1);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float mm = e < 2 ? mx0 : mx1;
                const float p = sc[nt][e] == -INFINITY ? 0.f : expf(sc[nt][e] - mm);
                sc[nt][e] = p;
                if (e < 2) s0 += p;
                else s1 += p;
            }
        s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        l0 = l0 * c0 + s0;
        l1 = l1 * c1 + s1;
        m0 = mx0;
        m1 = mx1;
#pragma unroll
        for (int nt = 0; nt < HEAD_DIM / 8; nt++) {
            o[nt][0] *= c0;
            o[nt][1] *= c0;
            o[nt][2] *= c1;
            o[nt][3] *= c1;
        }
        // O += P V: P's A fragments come from S's accumulators.
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            const uint32_t a[4] = {fa_bf2(sc[2 * ks][0], sc[2 * ks][1]), fa_bf2(sc[2 * ks][2], sc[2 * ks][3]),
                                   fa_bf2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]),
                                   fa_bf2(sc[2 * ks + 1][2], sc[2 * ks + 1][3])};
#pragma unroll
            for (int np = 0; np < HEAD_DIM / 16; np++) {
                uint32_t b[4];
                frag_b_t(b, sV, FA_STR, ks * 16, np * 16);
                gmma(o[2 * np], a, b[0], b[1]);
                gmma(o[2 * np + 1], a, b[2], b[3]);
            }
        }
    }
    // out = bf16(bf16(O / l) * sigmoid(gate)), as attn_body writes it.
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const int r = hf ? r1 : r0;
        if (r >= n) continue;
        const size_t t = (size_t)t0 + r;
        const float inv = hf ? i1 : i0;
#pragma unroll
        for (int nt = 0; nt < HEAD_DIM / 8; nt++) {
            const int d = nt * 8 + 2 * t4;
            float y[2];
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const float ov = rbf(o[nt][2 * hf + e] * inv);
                const float gate = b2f(qkv[t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d + e]);
                y[e] = ov / (1.f + expf(-gate));
            }
            *reinterpret_cast<__nv_bfloat162*>(out + (t * Hq + h) * HEAD_DIM + d) = __floats2bfloat162_rn(y[0], y[1]);
        }
    }
}

#define ATTN_FLASH_ARGS                                                                                \
    const bf16 *__restrict__ q, const void *__restrict__ kcache, const void *__restrict__ vcache,       \
        const float *__restrict__ kscale, const float *__restrict__ vscale, const bf16 *__restrict__ qkv, \
        int qstride, int Hq, int Hkv, const int *__restrict__ pos, const int *__restrict__ cu, int seqs,  \
        const int *__restrict__ block_table, int max_blocks, int block_size, bf16 *__restrict__ out,      \
        float scale
#define ATTN_FLASH_PASS q, kcache, vcache, kscale, vscale, qkv, qstride, Hq, Hkv, pos, cu, seqs, block_table, max_blocks, block_size, out, scale
extern "C" __global__ void __launch_bounds__(128) attn_flash(ATTN_FLASH_ARGS) { attn_flash_body<false>(ATTN_FLASH_PASS); }
extern "C" __global__ void __launch_bounds__(128) attn_flash8(ATTN_FLASH_ARGS) { attn_flash_body<true>(ATTN_FLASH_PASS); }

// Combines the key ranges of split decode attention (attn_body with `part`)
// in range order, then gates as attn_body does: a block of 256 threads per
// (token, query head), one per dimension, then one warp for the FP8 copy.
extern "C" __global__ void __launch_bounds__(HEAD_DIM)
    attn_merge(const float* __restrict__ part, int Z, int zchunk, const bf16* __restrict__ qkv, int qstride,
               int Hq, const int* __restrict__ pos, const int* __restrict__ tok_seq,
               const int* __restrict__ cu, int skip_prompts, bf16* __restrict__ out, uint8_t* fq, void* fs,
               int fmode, int frows) {
    __shared__ float ys[HEAD_DIM];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int sq = tok_seq[t];
    if (skip_prompts && cu[sq + 1] - cu[sq] > 1) return;
    const int nz = (pos[t] + zchunk) / zchunk;  // ranges holding keys 0 .. pos[t]
    const float* pp = part + ((size_t)sq * Hq + h) * Z * (HEAD_DIM + 2);
    float M = -INFINITY;
    for (int z = 0; z < nz; z++) M = fmaxf(M, pp[z * (HEAD_DIM + 2)]);
    float S = 0.f, A = 0.f;
    for (int z = 0; z < nz; z++) {
        const float* q = pp + z * (HEAD_DIM + 2);
        const float f = q[0] == -INFINITY ? 0.f : expf(q[0] - M);
        S += q[1] * f;
        A += q[2 + d] * f;
    }
    const float o = rbf(A / S);
    const float gate = b2f(qkv[(size_t)t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d]);
    const float y = o / (1.f + expf(-gate));
    out[((size_t)t * Hq + h) * HEAD_DIM + d] = f2b(y);
    if (fmode) {
        ys[d] = y;
        __syncthreads();
        if (d < 32) {
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; i++) v[i] = ys[8 * d + i];
            quant8(v, t, h * HEAD_DIM + 8 * d, Hq * HEAD_DIM, Fp8Out{fq, fs, fmode, frows}, true);
        }
    }
}

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Adjusts logits before sampling: for each (token, add, rep) entry of a row,
// logit = (logit > 0 ? logit / rep : logit * rep) + add. Entries of a row name
// distinct tokens. pen_cu[r]..pen_cu[r + 1] are row r's entries.
extern "C" __global__ void penalize(float* __restrict__ logits, int V, const int* __restrict__ pen_cu,
                                    const int* __restrict__ pen_tok, const float* __restrict__ pen_add,
                                    const float* __restrict__ pen_rep) {
    float* row = logits + (size_t)blockIdx.x * V;
    for (int e = pen_cu[blockIdx.x] + threadIdx.x; e < pen_cu[blockIdx.x + 1]; e += blockDim.x) {
        const int t = pen_tok[e];
        float l = row[t];
        const float rep = pen_rep[e];
        if (rep != 1.f) l = l > 0.f ? l / rep : l * rep;
        row[t] = l + pen_add[e];
    }
}

// Order-preserving map of a float to an unsigned key.
__device__ __forceinline__ uint32_t fkey(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

#define SAMPLE_THREADS 1024
#define SAMPLE_WARPS (SAMPLE_THREADS / 32)

// Radix selection over the keys of x = logit / T among elements with key >=
// floor. By count (mass == false): the largest key K with at least `target`
// elements >= K. By mass (mass == true): the largest K whose elements >= K
// hold at least `target` of the probability exp(x - m) summed over the
// survivors (target is a fraction). Returns floor when every survivor is
// needed. Four 8-bit rounds, each one pass over the row with per-warp
// histograms in shared memory.
__device__ uint32_t radix_select(const float* __restrict__ row, int V, float inv_t, float m,
                                 uint32_t floor, float target, bool mass, float* hist,
                                 float* bins, uint32_t* shared) {
    uint32_t prefix = 0, mask = 0;
    float need = target;
    for (int shift = 24; shift >= 0; shift -= 8) {
        float* mine = hist + (threadIdx.x >> 5) * 256;
        for (int b = threadIdx.x; b < SAMPLE_WARPS * 256; b += SAMPLE_THREADS) hist[b] = 0.f;
        __syncthreads();
        for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) {
            const float x = row[i] * inv_t;
            const uint32_t k = fkey(x);
            if (k >= floor && (k & mask) == prefix)
                atomicAdd(&mine[(k >> shift) & 255], mass ? expf(x - m) : 1.f);
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            float t = 0.f;
            for (int w = 0; w < SAMPLE_WARPS; w++) t += hist[w * 256 + threadIdx.x];
            bins[threadIdx.x] = t;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (mass && shift == 24) {
                float total = 0.f;
                for (int b = 0; b < 256; b++) total += bins[b];
                need = target * total;
            }
            float cum = 0.f;
            int chosen = -1;
            for (int b = 255; b >= 0; b--) {
                if (cum + bins[b] >= need) {
                    chosen = b;
                    break;
                }
                cum += bins[b];
            }
            shared[0] = chosen < 0 ? 0xffffffffu : (uint32_t)chosen;
            shared[1] = __float_as_uint(need - cum);
        }
        __syncthreads();
        const uint32_t chosen = shared[0];
        need = __uint_as_float(shared[1]);
        __syncthreads();
        if (chosen == 0xffffffffu) return floor;  // fewer survivors than needed
        prefix |= chosen << shift;
        mask |= 255u << shift;
    }
    return prefix > floor ? prefix : floor;
}

// Picks one token per row of f32 logits. temps[r] <= 0 is greedy (ties go to
// the lowest index). Otherwise it samples from softmax(logits / T) restricted,
// in order, by min_p (probability at least min_p times the top one), top_k and
// top_p (the smallest set holding top_p of the remaining probability), using
// Gumbel-max with noise hashed from (seeds[r], index).
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS)
    sample(const float* __restrict__ logits, int V, const float* __restrict__ temps,
           const int* __restrict__ top_ks, const float* __restrict__ top_ps,
           const float* __restrict__ min_ps, const int* __restrict__ seeds, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float hist[SAMPLE_WARPS * 256];
    __shared__ float bins[256];
    __shared__ uint32_t shared[2];
    const float* row = logits + (size_t)blockIdx.x * V;
    const float temp = temps[blockIdx.x];
    const uint32_t seed = mix32((uint32_t)seeds[blockIdx.x] ^ 0x9e3779b9u);
    uint32_t floor = 0;
    float inv_t = 1.f;
    if (temp > 0.f) {
        inv_t = 1.f / temp;
        const int top_k = top_ks[blockIdx.x];
        const float top_p = top_ps[blockIdx.x], min_p = min_ps[blockIdx.x];
        if (top_k > 0 || top_p < 1.f || min_p > 0.f) {
            float m = -INFINITY;
            for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) m = fmaxf(m, row[i] * inv_t);
            m = block_max(m, sv);
            if (min_p > 0.f) floor = fkey(m + logf(min_p));
            if (top_k > 0 && top_k < V)
                floor = radix_select(row, V, inv_t, m, floor, (float)top_k, false, hist, bins, shared);
            if (top_p < 1.f)
                floor = radix_select(row, V, inv_t, m, floor, top_p, true, hist, bins, shared);
        }
    }
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) {
        float x = row[i];
        if (temp > 0.f) {
            x *= inv_t;
            if (fkey(x) < floor) continue;
            const uint32_t h = mix32(seed ^ mix32((uint32_t)i));
            const float u = ((float)(h >> 8) + 0.5f) * (1.f / 16777216.f);
            x -= logf(-logf(u));
        }
        if (x > best) {
            best = x;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) {
        sv[w] = best;
        si[w] = bi;
    }
    __syncthreads();
    if (w == 0) {
        best = l < (int)(blockDim.x >> 5) ? sv[l] : -INFINITY;
        bi = l < (int)(blockDim.x >> 5) ? si[l] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (l == 0) out[blockIdx.x] = bi;
    }
}

// ---------------------------------------------------------------------------
// Vision encoder. Linear layers run as cuBLAS GEMMs with f32 output; these
// kernels add the bias, round once to bf16 as the reference linear does, and
// fuse what follows. Patches of all images in a batch are concatenated.

#define VIT_HEAD 64

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865476f)); }

// out[p] = bf16(bf16(y[p] + bias) + bf16(sum_k w[p,k] * table[idx[p,k]])): the
// patch embedding plus its bilinearly interpolated learned position.
extern "C" __global__ void vit_embed(const float* __restrict__ y, const float* __restrict__ bias,
                                     const bf16* __restrict__ table, const int* __restrict__ idx,
                                     const float* __restrict__ w, bf16* __restrict__ out, int D) {
    const int p = blockIdx.x;
    const int* ix = idx + 4 * p;
    const float* wx = w + 4 * p;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float pos = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) pos += wx[k] * b2f(table[(size_t)ix[k] * D + d]);
        out[(size_t)p * D + d] = f2b(rbf(y[(size_t)p * D + d] + bias[d]) + rbf(pos));
    }
}

// LayerNorm over rows of width D (<= 4 * 8 * blockDim.x), f32 statistics.
extern "C" __global__ void layernorm(const bf16* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ b, bf16* __restrict__ out, int D,
                                     float eps) {
    __shared__ float sh[32];
    const bf16* row = x + (size_t)blockIdx.x * D;
    float v[4][8];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < D) {
            unpack8(*reinterpret_cast<const uint4*>(row + i), v[c]);
#pragma unroll
            for (int e = 0; e < 8; e++) s += v[c][e];
        }
    }
    const float mean = block_sum(s, sh) / D;
    float q = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < D) {
#pragma unroll
            for (int e = 0; e < 8; e++) q += (v[c][e] - mean) * (v[c][e] - mean);
        }
    }
    const float inv = rsqrtf(block_sum(q, sh) / D + eps);
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < D) {
            float o[8];
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = (v[c][e] - mean) * inv * w[i + e] + b[i + e];
            *reinterpret_cast<uint4*>(out + (size_t)blockIdx.x * D + i) = pack8(o);
        }
    }
}

// Splits the qkv projection into head-major q, k, v ([head][patch][hd]),
// adding the bias and applying 2D rotary embedding to q and k in f32. With
// half = hd / 2, dims d and d + half rotate together; frequency f = d % half
// uses the patch's row for f < hd / 4 and its column otherwise, with inverse
// frequency index f % (hd / 4). One block of hd threads per (patch, head).
extern "C" __global__ void __launch_bounds__(128)
    vit_qkv_rope(const float* __restrict__ y, const float* __restrict__ bias,
                 const int* __restrict__ pos_hw, const float* __restrict__ inv_freq,
                 bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v, int P, int H) {
    __shared__ float qs[128], ks[128];
    const int p = blockIdx.x, h = blockIdx.y, d = threadIdx.x, hd = blockDim.x;
    const int D = H * hd, half = hd / 2, quarter = hd / 4;
    const float* row = y + (size_t)p * 3 * D;
    const float qv = rbf(row[h * hd + d] + bias[h * hd + d]);
    const float kv = rbf(row[D + h * hd + d] + bias[D + h * hd + d]);
    const float vv = rbf(row[2 * D + h * hd + d] + bias[2 * D + h * hd + d]);
    qs[d] = qv;
    ks[d] = kv;
    __syncthreads();
    const int f = d % half;
    const float angle = (float)(f < quarter ? pos_hw[2 * p] : pos_hw[2 * p + 1]) * inv_freq[f % quarter];
    const float c = cosf(angle), s = sinf(angle);
    const float qr = d < half ? -qs[d + half] : qs[d - half];
    const float kr = d < half ? -ks[d + half] : ks[d - half];
    const size_t o = ((size_t)h * P + p) * hd + d;
    q[o] = f2b(qv * c + qr * s);
    k[o] = f2b(kv * c + kr * s);
    v[o] = f2b(vv);
}

// P[r] = bf16(softmax(S[r] * scale)) over rows of n f32 scores.
extern "C" __global__ void softmax_rows(const float* __restrict__ S, bf16* __restrict__ out, int n,
                                        float scale) {
    __shared__ float sh[32];
    const float* row = S + (size_t)blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, row[i] * scale);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); w++) m = fmaxf(m, sh[w]);
    __syncthreads();
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += expf(row[i] * scale - m);
    z = block_sum(z, sh);
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[(size_t)blockIdx.x * n + i] = f2b(expf(row[i] * scale - m) * inv);
}

// mode 0: x = bf16(x + bf16(y + bias))   (residual add)
// mode 1: out = bf16(gelu_tanh(bf16(y + bias)))
// mode 2: out = bf16(gelu_erf(bf16(y + bias)))
// mode 3: out = bf16(y + bias)
extern "C" __global__ void vit_bias(const float* __restrict__ y, const float* __restrict__ bias,
                                    bf16* __restrict__ out, int D, long long n, int mode) {
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const float t = rbf(y[e] + bias[e % D]);
        float r;
        if (mode == 0) r = b2f(out[e]) + t;
        else if (mode == 1) r = gelu_tanh(t);
        else if (mode == 2) r = gelu_erf(t);
        else r = t;
        out[e] = f2b(r);
    }
}

#include "quant.cuh"
#include "fp4.cuh"
