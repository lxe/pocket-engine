// Qwen3.5 text decoder kernels. One step processes a flat batch of tokens drawn
// from several sequences: decode rows contribute one token each, prefill rows a
// chunk of their prompt. Per-token work (norms, projections, attention) runs
// over every token; the convolution and the gated delta rule walk each
// sequence's tokens in order.
//
// Activations are row-major bf16 [tokens, width]. Reductions run in f32, and
// values are rounded to bf16 where the reference model rounds them.

// hipcc predefines __HIP__; backend/hip.cuh then brings __HIP_PLATFORM_AMD__,
// which the rest of the sources test.
#ifdef __HIP__
#include "backend/hip.cuh"
#else
#include "backend/cuda.cuh"
#endif
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

#include <type_traits>
#include "fp8.cuh"

// Speculative decoding: a sequence that verifies drafts names, in
// outs[s * spec + i], the state slot to keep its state in after its token i.
// Its tokens are decode tokens (paged attention, the per-token recurrence)
// even though there are several. outs[s * spec] < 0 for other sequences.
__device__ __forceinline__ bool is_spec(const int* outs, int spec, int s) {
    return outs && outs[(size_t)s * spec] >= 0;
}

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
// proj rows hold the convolution inputs in their first C columns. A thread
// takes one channel of CONV_CHUNK tokens (blockIdx.z), reading the three
// inputs before them from the rows before or, at the start, the state.
#define CONV_CHUNK 64
extern "C" __global__ void gdn_conv(const bf16* __restrict__ proj, int pstride,
                                    const float* __restrict__ w, bf16* __restrict__ state,
                                    bf16* __restrict__ out, const int* __restrict__ cu,
                                    const int* __restrict__ slots, const int* __restrict__ fresh,
                                    int C, const int* __restrict__ outs, int spec,
                                    const int* __restrict__ snaps) {
    const int s = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = cu[s], t1 = cu[s + 1];
    const int ta = t0 + blockIdx.z * CONV_CHUNK, tz = min(t1, ta + CONV_CHUNK);
    if (ta >= tz) return;
    bf16* st = state + (size_t)slots[s] * 3 * C;
    const bool keep = is_spec(outs, spec, s);
    // Inputs at ta - 3, ta - 2, ta - 1.
    float x[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const int tt = ta - 3 + i;
        x[i] = tt >= t0 ? b2f(proj[(size_t)tt * pstride + c]) : fresh[s] ? 0.f : b2f(st[(size_t)(tt - t0 + 3) * C + c]);
    }
    float x0 = x[0], x1 = x[1], x2 = x[2];
    const float4 k = *reinterpret_cast<const float4*>(w + 4 * c);
    // Prefix-cache snapshots (snaps: per sequence two (tokens into the row,
    // slot) pairs, or null): the three inputs before that point.
    if (snaps && blockIdx.z == 0) {
        for (int j = 0; j < 2; j++) {
            const int rel = snaps[4 * s + 2 * j];
            if (rel < 0) continue;
            bf16* o = state + (size_t)snaps[4 * s + 2 * j + 1] * 3 * C;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                const int tt = t0 + rel - 3 + i;
                o[i * C + c] = f2b(tt >= t0 ? b2f(proj[(size_t)tt * pstride + c]) : x[tt - t0 + 3]);
            }
        }
    }
    // Inputs are loaded 8 tokens at a time, so a chunk waits for one load
    // latency per 8 tokens rather than per token.
    for (int t = ta; t < tz; t += 8) {
        float xs[8];
#pragma unroll
        for (int u = 0; u < 8; u++) xs[u] = t + u < tz ? b2f(proj[(size_t)(t + u) * pstride + c]) : 0.f;
#pragma unroll
        for (int u = 0; u < 8; u++) {
            if (t + u >= tz) break;
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
            if (keep) {
                bf16* o = state + (size_t)outs[(size_t)s * spec + (t + u - t0)] * 3 * C;
                o[c] = f2b(x0);
                o[C + c] = f2b(x1);
                o[2 * C + c] = f2b(x2);
            }
        }
    }
    // The first chunk, the only one that read the state, writes it: from
    // its own window, or the sequence's last three rows (a sequence of more
    // than one chunk has at least three).
    if (keep || blockIdx.z != 0) return;
    if (tz != t1) {
        x0 = b2f(proj[(size_t)(t1 - 3) * pstride + c]);
        x1 = b2f(proj[(size_t)(t1 - 2) * pstride + c]);
        x2 = b2f(proj[(size_t)(t1 - 1) * pstride + c]);
    }
    st[c] = f2b(x0);
    st[C + c] = f2b(x1);
    st[2 * C + c] = f2b(x2);
}

#ifdef __HIP_PLATFORM_AMD__
#include "amd/gdn_decode.cuh"
#endif

// Which prompt sequence holds unit `target` when the step's prompt
// sequences (more than one token, not verifying drafts) are cut into units
// of `unit` tokens and numbered in order; -1 past the last. The block's
// threads each count a slice of the sequences and a prefix sum finds the
// slice, so a step of many short prompts costs a few loads per thread
// rather than one thread walking every sequence. Every thread must call it.
__device__ __forceinline__ void block_find_unit(const int* __restrict__ cu, int seqs, const int* __restrict__ outs,
                                                int spec, int unit, int target, int& s_out, int& local_out) {
    __shared__ int warp_sum[32], found[2];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, T = blockDim.x;
    const int per = (seqs + T - 1) / T, i0 = min(seqs, tid * per), i1 = min(seqs, i0 + per);
    auto units = [&](int i) {
        const int n = cu[i + 1] - cu[i];
        return n <= 1 || is_spec(outs, spec, i) ? 0 : (n + unit - 1) / unit;
    };
    int mine = 0;
    for (int i = i0; i < i1; i++) mine += units(i);
    int incl = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
    }
    if (lane == 31) warp_sum[w] = incl;
    if (tid == 0) found[0] = -1;
    __syncthreads();
    for (int k = 0; k < w; k++) incl += warp_sum[k];
    if (target >= incl - mine && target < incl) {
        int left = target - (incl - mine);
        for (int i = i0; i < i1; i++) {
            const int u = units(i);
            if (left < u) {
                found[0] = i;
                found[1] = left;
                break;
            }
            left -= u;
        }
    }
    __syncthreads();
    s_out = found[0];
    local_out = found[1];
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
                                        int fmode, int frows, const int* __restrict__ outs, int spec) {
#ifdef __HIP_PLATFORM_AMD__
    gdn_one_w(s, h, conv, proj, pstride, C, Hk, Hv, neg_a, dt_bias, norm_w, state, state_bf16, cu, slots, fresh,
              out, eps, tiled, fq, fs, fmode, frows, outs, spec);
    return;
#endif
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
    // Prefill rounds the normalized q and k to bf16; decode keeps f32.
    const bool keep = is_spec(outs, spec, s);
    const bool round_qk = ((t1 - t0) > 1 && !keep) || is_fresh;
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
        if (keep) {
            // The state after each token, so rejected drafts cost nothing.
            const size_t kb = ((size_t)outs[(size_t)s * spec + (t - t0)] * Hv + h) * GDN_DIM * GDN_DIM;
#pragma unroll
            for (int i = 0; i < GDN_HALF; i++) {
                const int at = (i0 + i) * GDN_DIM + j;
                if (state_bf16) ((bf16*)state)[kb + at] = f2b(r[i]);
                else ((float*)state)[kb + at] = r[i];
            }
        }
    }
    if (keep) return;
#pragma unroll
    for (int i = 0; i < GDN_HALF; i++) {
        const int at = (i0 + i) * GDN_DIM + j;
        if (state_bf16) Sb[at] = f2b(r[i]);
        else S[at] = r[i];
    }
}

#define GDN_ARGS                                                                                       \
    const bf16 *__restrict__ conv, const bf16 *__restrict__ proj, int pstride, int C, int Hk, int Hv,  \
        const float *__restrict__ neg_a, const float *__restrict__ dt_bias,                            \
        const float *__restrict__ norm_w, void *__restrict__ state, int state_bf16,                    \
        const int *__restrict__ cu, const int *__restrict__ slots, const int *__restrict__ fresh,       \
        bf16 *__restrict__ out, float eps, int tiled, uint8_t *fq, void *fs, int fmode, int frows,       \
        const int *__restrict__ outs, int spec
#define GDN_PASS conv, proj, pstride, C, Hk, Hv, neg_a, dt_bias, norm_w, state, state_bf16, cu, slots, fresh, out, eps, tiled, fq, fs, fmode, frows, outs, spec

extern "C" __global__ void __launch_bounds__(2 * GDN_DIM) gdn_recurrent(GDN_ARGS) {
    gdn_one(blockIdx.x, blockIdx.y, GDN_PASS);
}

// The gated delta rule for prompts, in chunks of GC tokens on tensor cores
// (the chunked form FLA and FlashInfer use). With G the cumulative log decay
// inside a chunk, S0 the state before it and Gamma_ij = exp(G_i - G_j):
//   (I + A) D = beta * (V - exp(G) * K S0),  A_ij = beta_i Gamma_ij k_i.k_j (j < i)
//   O = exp(G) * Q S0 + tril(Q K^T * Gamma) D          (the diagonal included)
//   S = exp(G_last) S0 + (K * exp(G_last - G))^T D
// which is the token-by-token recurrence of gdn_one, exactly in exact
// arithmetic. The matrix products run on mma.sync with bf16 operands and f32
// accumulation; the triangular solve runs in f32, one thread per column.
#define GC 32
#define GSTR 136  // bf16 row stride of 128-wide tiles, padded against bank conflicts
#define GPSTR 40  // bf16 row stride of the 32-wide P tile
// cp.async of 16 (or 4) bytes that zero-fills when !valid (src is not read).
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

// Two passes, as FLA computes it. With T = (I + A)^-1, the chunk's update
// D = T beta (V - exp(G) K S0) splits into D = U - W S0, where U = T (beta V)
// and W = T (beta exp(G) K) depend only on the chunk's own tokens:
//   gdn_prep, one block per (chunk, head), all chunks at once: the decay
//     G, U, W, P = tril(qscale Q K^T * Gamma) and q's and k's inverse norms,
//     into per-token scratch rows [token][Hv][...];
//   gdn_scan, one block per (sequence, head, share of the value columns),
//     walking the chunks: D = U - W S0, O = exp(G) qscale Q S0 + P D and
//     S = exp(G_last) S0 + Kw^T D with Kw = K exp(G_last - G); it reads q
//     and k from conv and normalizes them in shared memory with the norms
//     gdn_prep found. Every product there is separable by value
//     column, so NS blocks split a head's columns without repeating work.
//     Warp w keeps state rows 16w .. 16w + 15 of its columns in registers
//     as mma accumulators, with a bf16 copy in shared memory for the
//     products that read it. Sequences with one token, or verifying drafts,
//     take gdn_one's way. Outputs go to `out` unnormalized; gdn_norm finishes
//     them.
#define GAS (GC + 4)  // f32 row stride of A, 16-byte aligned
#define GDN_PREP_SMEM (3 * GC * GSTR * 2 + GC * GAS * 4 + 6 * GC * 4)
extern "C" __global__ void __launch_bounds__(2 * GDN_DIM)
    gdn_prep(const bf16* __restrict__ conv, const bf16* __restrict__ proj, int pstride, int C, int Hk, int Hv,
             const float* __restrict__ neg_a, const float* __restrict__ dt_bias, const int* __restrict__ cu, int seqs,
             const int* __restrict__ outs, int spec, int tiled, bf16* __restrict__ Uo, bf16* __restrict__ Wo,
             float* __restrict__ Nqo, float* __restrict__ Nko, bf16* __restrict__ Po, float* __restrict__ Go) {
    extern __shared__ __align__(16) unsigned char psm[];
    bf16* sQ = (bf16*)psm;                      // [GC][GSTR] normalized q (unscaled)
    bf16* sK = sQ + GC * GSTR;                  // [GC][GSTR] normalized k
    bf16* sV = sK + GC * GSTR;                  // [GC][GSTR] v
    float* sA = (float*)(sV + GC * GSTR);       // [GC][GAS]
    float* sG = sA + GC * GAS;                  // [GC] cumulative log decay
    float* sB = sG + GC;                        // [GC] beta
    float* sg = sB + GC;                        // [GC] log decay
    float* sE = sg + GC;                        // [GC] beta exp(G)
    float* sN = sE + GC;                        // [2][GC] q's and k's inverse norms
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int s, chunk;
    block_find_unit(cu, seqs, outs, spec, GC, blockIdx.x, s, chunk);
    if (s < 0) return;
    const int c0 = cu[s] + chunk * GC, n = min(GC, cu[s + 1] - c0);
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    const float na = neg_a[h], db = dt_bias[h], qscale = rsqrtf((float)GDN_DIM);
    // Inputs: a warp per token (warp w takes tokens w, w + 8, w + 16, w + 24),
    // four dimensions per lane, all loads issued first; rows past n are zero.
    constexpr int R = GC / 8;
    uint2 qr[R], kr[R], vr[R];
    float br[R], ar[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int w = warp + 8 * r;
        qr[r] = kr[r] = vr[r] = make_uint2(0, 0);
        br[r] = ar[r] = 0.f;
        if (w < n) {
            const bf16* row = conv + (size_t)(c0 + w) * C;
            qr[r] = *reinterpret_cast<const uint2*>(row + kh * GDN_DIM + 4 * lane);
            kr[r] = *reinterpret_cast<const uint2*>(row + Hk * GDN_DIM + kh * GDN_DIM + 4 * lane);
            vr[r] = *reinterpret_cast<const uint2*>(row + 2 * Hk * GDN_DIM + h * GDN_DIM + 4 * lane);
            if (lane == 0) {
                const bf16* prow = proj + (size_t)(c0 + w) * pstride + C + Hv * GDN_DIM;
                br[r] = b2f(prow[h]);
                ar[r] = b2f(prow[Hv + h]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int w = warp + 8 * r;
        float q[4], k[4];
        unpack4(qr[r], q);
        unpack4(kr[r], k);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            qq += q[e] * q[e];
            kk += k[e] * k[e];
        }
        const float nq = 1.f / sqrtf(warp_sum(qq) + 1e-6f), nk = 1.f / sqrtf(warp_sum(kk) + 1e-6f);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int d = 4 * lane + e;
            sQ[w * GSTR + d] = f2b(q[e] * nq);
            sK[w * GSTR + d] = f2b(k[e] * nk);
        }
        *reinterpret_cast<uint2*>(sV + w * GSTR + 4 * lane) = vr[r];
        if (lane == 0) {
            sN[w] = nq;
            sN[GC + w] = nk;
            float beta = 0.f, lg = 0.f;
            if (w < n) {
                const float x = ar[r] + db;
                const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
                beta = 1.f / (1.f + expf(-br[r]));
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
        sE[lane] = sB[lane] * expf(x);
    }
    __syncthreads();
    if (tid < n) {
        const size_t at = (size_t)(c0 + tid) * Hv + h;
        Go[at] = sG[tid];
        Nqo[at] = sN[tid];
        Nko[at] = sN[GC + tid];
    }
    // K K^T (warps 0-3) into A and Q K^T (warps 4-7) into P, a 16 x 16 tile each.
#ifdef __HIP_PLATFORM_AMD__
    {
        // On MFMA: wave w takes tile (16 (w & 1), 16 (w >> 1)) of both.
        const int wv = tid >> 6, ml = tid & 15, mq = (tid >> 4) & 3;
        const int m0 = (wv & 1) * 16, n0 = (wv >> 1) * 16;
        pe_f4 ak = {0.f, 0.f, 0.f, 0.f}, aq = ak;
#pragma unroll
        for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
            const pe_s4 b = frag4(sK, GSTR, n0 + ml, k0 + 4 * mq);
            ak = mfma16(frag4(sK, GSTR, m0 + ml, k0 + 4 * mq), b, ak);
            aq = mfma16(frag4(sQ, GSTR, m0 + ml, k0 + 4 * mq), b, aq);
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int i = m0 + 4 * mq + e, j = n0 + ml;
            const float gam = j <= i ? expf(sG[i] - sG[j]) : 0.f;
            sA[i * GAS + j] = j < i ? sB[i] * gam * ak[e] : 0.f;
            if (i < n) Po[((size_t)(c0 + i) * Hv + h) * GC + j] = f2b(qscale * gam * aq[e]);
        }
    }
#else
    {
        const int m0 = (warp & 1) * 16, n0 = ((warp >> 1) & 1) * 16;
        const bf16* X = warp < 4 ? sK : sQ;
        float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
        for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
            uint32_t a[4], b[4];
            frag_a(a, X, GSTR, m0, k0);
            frag_b(b, sK, GSTR, k0, n0);
            mma_bf16(acc[0], a, b[0], b[1]);
            mma_bf16(acc[1], a, b[2], b[3]);
        }
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int i = m0 + g + (e >> 1) * 8, j = n0 + nt * 8 + 2 * t + (e & 1);
                const float gam = j <= i ? expf(sG[i] - sG[j]) : 0.f;
                if (warp < 4) sA[i * GAS + j] = j < i ? sB[i] * gam * acc[nt][e] : 0.f;
                else if (i < n) Po[((size_t)(c0 + i) * Hv + h) * GC + j] = f2b(qscale * gam * acc[nt][e]);
            }
    }
#endif
    __syncthreads();
    // (I + A) [U | W] = [beta v | beta exp(G) k], by forward substitution: a
    // thread per column, reading A four entries at a time.
    {
        const bf16* X = tid < GDN_DIM ? sV : sK;
        const float* fac = tid < GDN_DIM ? sB : sE;
        const int col = tid % GDN_DIM;
#ifdef __HIP_PLATFORM_AMD__
        // On gfx942 the unrolled solve below took 512 registers and still
        // spilled; each thread keeps its column of the solution in shared
        // memory instead (sD, [GC][256] f32 past GDN_PREP_SMEM), rows in turn.
        float* sD = reinterpret_cast<float*>(psm + GDN_PREP_SMEM) + tid;
#pragma unroll 1
        for (int i = 0; i < GC; i++) {
            float a0 = fac[i] * b2f(X[i * GSTR + col]), a1 = 0.f;
            for (int j = 0; j < (i & ~3); j += 4) {
                const float4 a = *reinterpret_cast<const float4*>(sA + i * GAS + j);
                a0 = fmaf(-a.x, sD[j * 2 * GDN_DIM], a0);
                a1 = fmaf(-a.y, sD[(j + 1) * 2 * GDN_DIM], a1);
                a0 = fmaf(-a.z, sD[(j + 2) * 2 * GDN_DIM], a0);
                a1 = fmaf(-a.w, sD[(j + 3) * 2 * GDN_DIM], a1);
            }
            for (int j = i & ~3; j < i; j++) a0 = fmaf(-sA[i * GAS + j], sD[j * 2 * GDN_DIM], a0);
            sD[i * 2 * GDN_DIM] = a0 + a1;
        }
        bf16* dst = tid < GDN_DIM ? Uo : Wo;
        for (int i = 0; i < n; i++) dst[((size_t)(c0 + i) * Hv + h) * GDN_DIM + col] = f2b(sD[i * 2 * GDN_DIM]);
#else
        float d[GC];
#pragma unroll
        for (int i = 0; i < GC; i++) {
            float a0 = fac[i] * b2f(X[i * GSTR + col]), a1 = 0.f;
#pragma unroll
            for (int j = 0; j < (i & ~3); j += 4) {
                const float4 a = *reinterpret_cast<const float4*>(sA + i * GAS + j);
                a0 = fmaf(-a.x, d[j], a0);
                a1 = fmaf(-a.y, d[j + 1], a1);
                a0 = fmaf(-a.z, d[j + 2], a0);
                a1 = fmaf(-a.w, d[j + 3], a1);
            }
#pragma unroll
            for (int j = i & ~3; j < i; j++) a0 = fmaf(-sA[i * GAS + j], d[j], a0);
            d[i] = a0 + a1;
        }
        bf16* dst = tid < GDN_DIM ? Uo : Wo;
#pragma unroll
        for (int i = 0; i < GC; i++)
            if (i < n) dst[((size_t)(c0 + i) * Hv + h) * GDN_DIM + col] = f2b(d[i]);
#endif
    }
}

// Shared memory of gdn_scan<NS>: NB buffers of a chunk's inputs (W, q, k at
// full width; U at the block's VW columns; P; G and q's and k's inverse
// norms), D, and the state's columns. NS > 1 double-buffers the inputs so the next chunk loads while
// this one computes; NS == 1 would not fit an RTX 5090 that way.
#define GDN_VS(NS) (GDN_DIM / (NS) + 8)
#ifdef __HIP_PLATFORM_AMD__
#define GDN_NB(NS) 1  // 64 KB of LDS per block on gfx942: no double buffering
#else
#define GDN_NB(NS) ((NS) > 1 ? 2 : 1)
#endif
#define GDN_BUF(NS) (3 * GC * GSTR * 2 + GC * GDN_VS(NS) * 2 + GC * GPSTR * 2 + 3 * GC * 4)
#define GDN_SCAN_SMEM(NS) (GDN_NB(NS) * GDN_BUF(NS) + 2 * GC * GDN_VS(NS) * 2 + GDN_DIM * GDN_VS(NS) * 2)
template <int NS>
__device__ __forceinline__ void gdn_scan_cols(GDN_ARGS, const bf16* __restrict__ Ui, const bf16* __restrict__ Wi,
                                              const float* __restrict__ Nqi, const float* __restrict__ Nki,
                                              const bf16* __restrict__ Pi, const float* __restrict__ Gi,
                                              const int* __restrict__ snaps) {
    constexpr int VW = GDN_DIM / NS, ON = VW / 32, VS = GDN_VS(NS), NB = GDN_NB(NS);
    extern __shared__ __align__(16) unsigned char ssm[];
    // Buffer b: W, q, k [GC][GSTR]; U [GC][VS] (this block's columns); P
    // [GC][GPSTR]; G, Nq, Nk [GC].
    auto buf = [&](int b) { return ssm + b * GDN_BUF(NS); };
    bf16* sD = (bf16*)buf(NB);                  // [GC][VS] D
    bf16* sDk = sD + GC * VS;                   // [GC][VS] D, row i times exp(G_last - G_i) / |k_i|
    bf16* sS = sDk + GC * VS;                   // [GDN_DIM][VS] the state's columns, bf16
    const int s = gridDim.x - 1 - blockIdx.x, h = blockIdx.y, vz = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    if (t1 - t0 == 1 || is_spec(outs, spec, s)) {
        if (vz == 0) gdn_one(s, h, GDN_PASS);
        return;
    }
    const int v0 = vz * VW, kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float* Sg = (float*)state + base;
    bf16* Sgb = (bf16*)state + base;
    const bool is_fresh = fresh[s] != 0;
    const float qscale = rsqrtf((float)GDN_DIM);
    // The state, rows 16w + g (+8) and columns v0 + 8n + 2t (+1); sS holds
    // them from column 0.
    float st[VW / 8][4];
#pragma unroll
    for (int nn = 0; nn < VW / 8; nn++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = 16 * warp + g + (e >> 1) * 8, col = v0 + 8 * nn + 2 * t + (e & 1);
            const size_t at = (size_t)row * GDN_DIM + col;
            st[nn][e] = is_fresh ? 0.f : state_bf16 ? b2f(Sgb[at]) : Sg[at];
        }
    auto store_state_smem = [&]() {
#pragma unroll
        for (int nn = 0; nn < VW / 8; nn++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int row = 16 * warp + g + hf * 8, col = 8 * nn + 2 * t;
                *reinterpret_cast<__nv_bfloat162*>(sS + (size_t)row * VS + col) =
                    __floats2bfloat162_rn(st[nn][2 * hf], st[nn][2 * hf + 1]);
            }
    };
    // This block's share of the state into a slot.
    auto store_state = [&](int slot) {
        const size_t b = ((size_t)slot * Hv + h) * GDN_DIM * GDN_DIM;
#pragma unroll
        for (int nn = 0; nn < VW / 8; nn++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = 16 * warp + g + (e >> 1) * 8, col = v0 + 8 * nn + 2 * t + (e & 1);
                const size_t at = b + (size_t)row * GDN_DIM + col;
                if (state_bf16) ((bf16*)state)[at] = f2b(st[nn][e]);
                else ((float*)state)[at] = st[nn][e];
            }
    };
    store_state_smem();
    bool zero_state = is_fresh;
    const int om0 = (warp & 1) * 16, on0 = (warp >> 1) * (VW / 4);
    // Chunk c0's scratch rows into buffer b; rows past the sequence are zero.
    auto load = [&](int c0, int b) {
        const int n = min(GC, t1 - c0);
        bf16* bW = (bf16*)buf(b);
        bf16* bQ = bW + GC * GSTR;
        bf16* bK = bQ + GC * GSTR;
        bf16* bU = bK + GC * GSTR;
        bf16* bP = bU + GC * VS;
        float* bG = (float*)(bP + GC * GPSTR);
        for (int e = tid; e < GC * (GDN_DIM / 8); e += 2 * GDN_DIM) {
            const int i = e / (GDN_DIM / 8), c = (e % (GDN_DIM / 8)) * 8, r = c0 + min(i, n - 1);
            const size_t at = ((size_t)r * Hv + h) * GDN_DIM + c;
            const bf16* row = conv + (size_t)r * C + kh * GDN_DIM + c;
            cp16z(bW + i * GSTR + c, Wi + at, i < n);
            cp16z(bQ + i * GSTR + c, row, i < n);
            cp16z(bK + i * GSTR + c, row + Hk * GDN_DIM, i < n);
            if (c < VW) cp16z(bU + i * VS + c, Ui + at + v0, i < n);
        }
        for (int e = tid; e < GC * (GC / 8); e += 2 * GDN_DIM) {
            const int i = e / (GC / 8), c = (e % (GC / 8)) * 8;
            cp16z(bP + i * GPSTR + c, Pi + ((size_t)(c0 + min(i, n - 1)) * Hv + h) * GC + c, i < n);
        }
        if (tid < GC) {
            const size_t at = (size_t)(c0 + min(tid, n - 1)) * Hv + h;
            cp4(bG + tid, Gi + at);
            cp4(bG + GC + tid, Nqi + at);
            cp4(bG + 2 * GC + tid, Nki + at);
        }
        cp_commit();
    };
    if (NB == 2) load(t0, 0);
    for (int c0 = t0, it = 0; c0 < t1; c0 += GC, it++) {
        const int n = min(GC, t1 - c0), b = NB == 2 ? it & 1 : 0;
        if (NB == 2) {
            if (c0 + GC < t1) load(c0 + GC, b ^ 1);
            else cp_commit();
            cp_wait<1>();
        } else {
            load(c0, 0);
            cp_wait<0>();
        }
        __syncthreads();
        const bf16* sW = (const bf16*)buf(b);
        bf16* sQ = (bf16*)sW + GC * GSTR;
        bf16* sKw = sQ + GC * GSTR;
        const bf16* sU = sKw + GC * GSTR;
        const bf16* sP = sU + GC * VS;
        const float* sG = (const float*)(sP + GC * GPSTR);
        const float* sNq = sG + GC;
        const float* sNk = sNq + GC;
        const float glast = sG[GC - 1];
        // D = U - W S0 and Q S0, rows om0 .. om0 + 15, this warp's columns.
        float accq[ON][4];
        {
            float acc[ON][4];
#pragma unroll
            for (int nt = 0; nt < ON; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[nt][e] = accq[nt][e] = 0.f;
            if (!zero_state) {
#pragma unroll
                for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
                    uint32_t a[4], aq[4];
                    frag_a(a, sW, GSTR, om0, k0);
                    frag_a(aq, sQ, GSTR, om0, k0);
#pragma unroll
                    for (int nt = 0; nt < ON; nt++) {
                        uint32_t b[4];
                        frag_b_t(b, sS, VS, k0, on0 + 8 * nt);
                        mma_bf16(acc[nt], a, b[0], b[1]);
                        mma_bf16(accq[nt], aq, b[0], b[1]);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < ON; nt++)
#pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int i = om0 + g + hf * 8, c = on0 + nt * 8 + 2 * t;
                    const __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(sU + i * VS + c);
                    const float d0 = __low2float(u) - acc[nt][2 * hf], d1 = __high2float(u) - acc[nt][2 * hf + 1];
                    const float f = expf(glast - sG[i]) * sNk[i];
                    *reinterpret_cast<__nv_bfloat162*>(sD + i * VS + c) = __floats2bfloat162_rn(d0, d1);
                    *reinterpret_cast<__nv_bfloat162*>(sDk + i * VS + c) = __floats2bfloat162_rn(f * d0, f * d1);
                }
            const float e0 = qscale * expf(sG[om0 + g]) * sNq[om0 + g],
                        e1 = qscale * expf(sG[om0 + g + 8]) * sNq[om0 + g + 8];
#pragma unroll
            for (int nt = 0; nt < ON; nt++) {
                accq[nt][0] *= e0;
                accq[nt][1] *= e0;
                accq[nt][2] *= e1;
                accq[nt][3] *= e1;
            }
        }
        __syncthreads();
        // O = exp(G) qscale Q S0 + P D, written unnormalized.
#pragma unroll
        for (int k0 = 0; k0 < GC; k0 += 16) {
            uint32_t a[4];
            frag_a(a, sP, GPSTR, om0, k0);
#pragma unroll
            for (int nt = 0; nt < ON; nt++) {
                uint32_t b[4];
                frag_b_t(b, sD, VS, k0, on0 + 8 * nt);
                mma_bf16(accq[nt], a, b[0], b[1]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < ON; nt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int i = om0 + g + hf * 8, c = on0 + nt * 8 + 2 * t;
                if (i < n)
                    *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(c0 + i) * Hv * GDN_DIM + h * GDN_DIM + v0 + c) =
                        __floats2bfloat162_rn(accq[nt][2 * hf], accq[nt][2 * hf + 1]);
            }
        // S = exp(G_last) S0 + Kw^T D: this warp's 16 rows, this block's columns.
        {
            const float decay = expf(glast);
#pragma unroll
            for (int nt = 0; nt < VW / 8; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) st[nt][e] *= decay;
#pragma unroll
            for (int k0 = 0; k0 < GC; k0 += 16) {
                uint32_t a[4];
                frag_a_t(a, sKw, GSTR, 16 * warp, k0);
#pragma unroll
                for (int np = 0; np < VW / 16; np++) {
                    uint32_t b[4];
                    frag_b_t(b, sDk, VS, k0, 16 * np);
                    mma_bf16(st[2 * np], a, b[0], b[1]);
                    mma_bf16(st[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        // A prefix-cache snapshot ending with this chunk.
        if (snaps)
            for (int j = 0; j < 2; j++)
                if (snaps[4 * s + 2 * j] == c0 + n - t0) store_state(snaps[4 * s + 2 * j + 1]);
        __syncthreads();
        store_state_smem();
        zero_state = false;
        __syncthreads();
    }
    store_state(slots[s]);
}
#ifdef __HIP_PLATFORM_AMD__
#include "amd/gdn_chunk.cuh"
#endif
#define GDN_SCAN_ARGS                                                                                   \
    GDN_ARGS, const bf16 *__restrict__ Ui, const bf16 *__restrict__ Wi, const float *__restrict__ Nqi, \
        const float *__restrict__ Nki, const bf16 *__restrict__ Pi, const float *__restrict__ Gi,        \
        const int *__restrict__ snaps
extern "C" __global__ void __launch_bounds__(2 * GDN_DIM) gdn_scan1(GDN_SCAN_ARGS) {
    gdn_scan_cols<1>(GDN_PASS, Ui, Wi, Nqi, Nki, Pi, Gi, snaps);
}
extern "C" __global__ void __launch_bounds__(2 * GDN_DIM) gdn_scan2(GDN_SCAN_ARGS) {
#ifdef __HIP_PLATFORM_AMD__
    gdn_scan_mfma<2>(GDN_PASS, Ui, Wi, Nqi, Nki, Pi, Gi, snaps);
#else
    gdn_scan_cols<2>(GDN_PASS, Ui, Wi, Nqi, Nki, Pi, Gi, snaps);
#endif
}
extern "C" __global__ void __launch_bounds__(2 * GDN_DIM) gdn_scan4(GDN_SCAN_ARGS) {
#ifdef __HIP_PLATFORM_AMD__
    gdn_scan_mfma<4>(GDN_PASS, Ui, Wi, Nqi, Nki, Pi, Gi, snaps);
#else
    gdn_scan_cols<4>(GDN_PASS, Ui, Wi, Nqi, Nki, Pi, Gi, snaps);
#endif
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
        float z[8];
        unpack8(*reinterpret_cast<const uint4*>(proj + (size_t)t * pstride + C + h * GDN_DIM + 8 * lane), z);
        const float4 w0 = *reinterpret_cast<const float4*>(norm_w + 8 * lane);
        const float4 w1 = *reinterpret_cast<const float4*>(norm_w + 8 * lane + 4);
        const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = o[e] * inv * w[e] * silu(z[e]);
        *reinterpret_cast<uint4*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + 8 * lane) = pack8(o);
    }
    if (fmode) quant8(o, t, h * GDN_DIM + 8 * (lane & 15), Hv * GDN_DIM, fo, lane < 16);
}
// The gated norm for gdn_scan's outputs, in place: a block per token of a
// sequence with more than one, a warp per head (warp w takes heads w, w + 8,
// ...).
extern "C" __global__ void gdn_norm(const bf16* __restrict__ proj, int pstride, int C, int Hv,
                                    const float* __restrict__ norm_w, bf16* __restrict__ out, int T, float eps,
                                    const int* __restrict__ cu, const int* __restrict__ tok_seq, uint8_t* fq,
                                    void* fs, int fmode, int frows, const int* __restrict__ outs, int spec) {
    const Fp8Out fo{fq, fs, fmode, frows};
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const int sq = tok_seq[t];
    if (cu[sq + 1] - cu[sq] == 1 || is_spec(outs, spec, sq)) return;  // finished by gdn_one or gdn_decode
#pragma unroll 2
    for (int h = threadIdx.x >> 5; h < Hv; h += blockDim.x >> 5) {
        float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (lane < 16) unpack8(*reinterpret_cast<const uint4*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + 8 * lane), o);
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
// Key ranges split decode attention divides a context of L keys into: up to
// Z, each at least min_chunk long.
__device__ __forceinline__ int split_parts(int L, int Z, int min_chunk) {
    return max(1, min(Z, (L + min_chunk - 1) / min_chunk));
}

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
                                          const int* __restrict__ cu, int skip_prompts, int min_chunk,
                                          float* __restrict__ part, const int* __restrict__ outs, int spec) {
    __shared__ float sm_m[WARPS][G], sm_s[WARPS][G];
    __shared__ float sm_acc[WARPS][G][HEAD_DIM];
    const int t = blockIdx.x, kvh = blockIdx.y;
    // With skip_prompts, tokens of prompt sequences (more than one token,
    // not verifying drafts) are left to attn_flash.
    const int sq = tok_seq[t];
    if (skip_prompts && cu[sq + 1] - cu[sq] > 1 && !is_spec(outs, spec, sq)) return;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int L = pos[t] + 1;
    // With `part`, keys split into up to gridDim.z equal ranges of at least
    // min_chunk across blockIdx.z (flash decoding); the ranges' partial
    // results go to `part` for attn_merge.
    const int parts = part ? split_parts(L, gridDim.z, min_chunk) : 1;
    const int chunk = (L + parts - 1) / parts;
    const int j0 = part ? blockIdx.z * chunk : 0, j1 = part ? min(L, j0 + chunk) : L;
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
    // Four keys per warp at a time, so their loads overlap.
    constexpr int U = 4;
    for (int j = j0 + w * U; j < j1; j += WARPS * U) {
        float kf[U][8], vf[U][8], ks[U], vs[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            ks[u] = vs[u] = 1.f;
            const int jj = min(j + u, j1 - 1);
            const size_t slot = (size_t)bt[jj / block_size] * block_size + (jj % block_size);
            const size_t at = slot * Hkv + kvh;
            const size_t off = at * HEAD_DIM + l * 8;
            if constexpr (F8) {
                unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)kcache + off), kf[u]);
                unpack8_e4m3(*reinterpret_cast<const uint2*>((const uint8_t*)vcache + off), vf[u]);
                ks[u] = kscale[at];
                vs[u] = vscale[at];
            } else {
                unpack8(*reinterpret_cast<const uint4*>((const bf16*)kcache + off), kf[u]);
                unpack8(*reinterpret_cast<const uint4*>((const bf16*)vcache + off), vf[u]);
            }
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            float d[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                d[u] = 0.f;
#pragma unroll
                for (int e = 0; e < 8; e++) d[u] += qr[g][e] * kf[u][e];
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                for (int u = 0; u < U; u++) d[u] += __shfl_xor_sync(0xffffffffu, d[u], o);
            float mn = m[g];
#pragma unroll
            for (int u = 0; u < U; u++) {
                d[u] = j + u < j1 ? d[u] * ks[u] : -INFINITY;
                mn = fmaxf(mn, d[u]);
            }
            const float corr = expf(m[g] - mn);
            float p[U], psum = 0.f;
#pragma unroll
            for (int u = 0; u < U; u++) {
                p[u] = expf(d[u] - mn);
                psum += p[u];
                p[u] *= vs[u];
            }
            ssum[g] = ssum[g] * corr + psum;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                float a = acc[g][e] * corr;
#pragma unroll
                for (int u = 0; u < U; u++) a += p[u] * vf[u][e];
                acc[g][e] = a;
            }
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
                // Partials per decode token: up to `spec` per sequence.
                const size_t row = (size_t)sq * spec + (t - cu[sq]);
                float* pp = part + ((row * Hq + h) * gridDim.z + blockIdx.z) * (HEAD_DIM + 2);
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
        int frows, const int* __restrict__ cu, int skip_prompts, int min_chunk, float* part,           \
        const int* __restrict__ outs, int spec) {                                                      \
        attn_body<G, WARPS, F8>(q, kcache, vcache, kscale, vscale, qkv, qstride, Hq, Hkv, pos,         \
                                tok_seq, block_table, max_blocks, block_size, out, scale, fq, fs,      \
                                fmode, frows, cu, skip_prompts, min_chunk, part, outs, spec);          \
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

// Attention for prompt tokens, on tensor cores: flash attention over the
// paged cache. A block takes one KV head and 16 query tokens of one prompt
// sequence, with a warp per query head of the group, so each key and value
// tile is loaded once for all of them; tiles of 16 keys are double-buffered
// (cp.async for bf16 caches, converted from E4M3 for --kv-fp8). S = Q K^T and
// P V run on mma.sync with an online softmax, P passing from the first
// product's accumulators to the second as A fragments. The output is gated
// as attn_body gates it. Blocks walk the step's prompt sequences (those with
// more than one token) in tiles of 16; others exit.
#define FA_STR (HEAD_DIM + 8)
#define LOG2E 1.4426950408889634f
__device__ __forceinline__ uint32_t fa_bf2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
#define ATTN_FLASH_ARGS                                                                                \
    const bf16 *__restrict__ q, const void *__restrict__ kcache, const void *__restrict__ vcache,       \
        const float *__restrict__ kscale, const float *__restrict__ vscale, const bf16 *__restrict__ qkv, \
        int qstride, int Hq, int Hkv, const int *__restrict__ pos, const int *__restrict__ cu, int seqs,  \
        const int *__restrict__ block_table, int max_blocks, int block_size, bf16 *__restrict__ out,      \
        float scale, const int *__restrict__ outs, int spec
#define ATTN_FLASH_PASS q, kcache, vcache, kscale, vscale, qkv, qstride, Hq, Hkv, pos, cu, seqs, block_table, max_blocks, block_size, out, scale, outs, spec

// Shared memory of a block with NT query tiles of 16 tokens for each of G
// heads and two buffers of BN keys and values.
template <int G, int NT, int BN>
__host__ __device__ constexpr int fg_smem() { return (G * 16 * NT + 4 * BN) * FA_STR * 2; }

template <int G, bool F8, int NT, int BN, bool QR>
__device__ __forceinline__ void attn_flash_group(
    const bf16* __restrict__ q, const void* __restrict__ kcache, const void* __restrict__ vcache,
    const float* __restrict__ kscale, const float* __restrict__ vscale, const bf16* __restrict__ qkv, int qstride,
    int Hq, int Hkv, const int* __restrict__ pos, const int* __restrict__ cu, int seqs,
    const int* __restrict__ block_table, int max_blocks, int block_size, bf16* __restrict__ out, float scale,
    const int* __restrict__ outs, int spec) {
    extern __shared__ __align__(16) unsigned char fsm[];
    constexpr int BT = 16 * NT;            // tokens per block
    // With QR each warp keeps its queries in registers, and the keys and
    // values reuse the queries' shared memory once they are read.
    bf16* sQ = (bf16*)fsm;                                // [NT][G][16][FA_STR]
    bf16* sK = QR ? sQ : sQ + NT * G * 16 * FA_STR;       // [2][BN][FA_STR]
    bf16* sV = sK + 2 * BN * FA_STR;                      // [2][BN][FA_STR]
    constexpr int T = NT * G * 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    // Warp w takes query head w % G of query tile w / G.
    const int kvh = blockIdx.y, h = kvh * G + warp % G, qt = warp / G;
    int s = -1, tile = blockIdx.x;
    for (int i = 0; i < seqs; i++) {
        const int n = cu[i + 1] - cu[i];
        if (n <= 1 || is_spec(outs, spec, i)) continue;
        const int tiles = (n + BT - 1) / BT;
        if (tile < tiles) {
            s = i;
            break;
        }
        tile -= tiles;
    }
    if (s < 0) return;
    const int tb = cu[s] + tile * BT, nb = min(BT, cu[s + 1] - tb);
    const int L = pos[tb] + nb;
    // This warp's 16 queries: tokens t0 .. t0 + n - 1 at positions p0 on.
    const int t0 = tb + qt * 16, n = max(0, min(16, nb - qt * 16)), p0 = pos[tb] + qt * 16;
    const int* bt = block_table + (size_t)s * max_blocks;
    for (int e = tid; e < NT * G * 16 * (HEAD_DIM / 8); e += T) {
        const int row = e / (HEAD_DIM / 8), c = (e % (HEAD_DIM / 8)) * 8;
        const int hw = row / 16 % G, r = row % 16, tq = row / (16 * G) * 16 + r;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (tq < nb) {
            float f[8];
            unpack8(*reinterpret_cast<const uint4*>(q + ((size_t)(tb + tq) * Hq + kvh * G + hw) * HEAD_DIM + c), f);
#pragma unroll
            for (int i = 0; i < 8; i++) f[i] *= scale * LOG2E;  // scores in log2 units
            v = pack8(f);
        }
        *reinterpret_cast<uint4*>(sQ + row * FA_STR + c) = v;
    }
    auto load = [&](int k0, int buf) {
        bf16* dk = sK + buf * BN * FA_STR;
        bf16* dv = sV + buf * BN * FA_STR;
        for (int e = tid; e < BN * (HEAD_DIM / 8); e += T) {
            const int j = e / (HEAD_DIM / 8), c = (e % (HEAD_DIM / 8)) * 8, kp = k0 + j;
            if (kp >= L) {
                *reinterpret_cast<uint4*>(dk + j * FA_STR + c) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(dv + j * FA_STR + c) = make_uint4(0, 0, 0, 0);
                continue;
            }
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
                *reinterpret_cast<uint4*>(dk + j * FA_STR + c) = pack8(kf);
                *reinterpret_cast<uint4*>(dv + j * FA_STR + c) = pack8(vf);
            } else {
                cp_async16(dk + j * FA_STR + c, (const bf16*)kcache + at);
                cp_async16(dv + j * FA_STR + c, (const bf16*)vcache + at);
            }
        }
        cp_async_commit();
    };
    float o[HEAD_DIM / 8][4];
#pragma unroll
    for (int nt = 0; nt < HEAD_DIM / 8; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const bf16* myQ = sQ + warp * 16 * FA_STR;
    const int last = p0 + n - 1;  // this warp's last query position
    float sc[BN / 8][4];
    float c0 = 1.f, c1 = 1.f;
    uint32_t qa[QR ? HEAD_DIM / 16 : 1][4];
    if constexpr (QR) {
        __syncthreads();
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM; kd += 16) frag_a(qa[kd / 16], myQ, FA_STR, 0, kd);
        __syncthreads();  // the queries' shared memory now holds keys and values
    }
    // Scores of the tile from k0, masked, and the online softmax into sc;
    // c0, c1 are the factors the outputs so far take.
    auto scores = [&](const bf16* cK, int k0) {
#pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM; kd += 16) {
            uint32_t a[4];
            if constexpr (QR) {
#pragma unroll
                for (int i = 0; i < 4; i++) a[i] = qa[kd / 16][i];
            } else {
                frag_a(a, myQ, FA_STR, 0, kd);
            }
#pragma unroll
            for (int np = 0; np < BN / 16; np++) {
                uint32_t b[4];
                frag_b(b, cK, FA_STR, kd, 16 * np);
                mma_bf16(sc[2 * np], a, b[0], b[1]);
                mma_bf16(sc[2 * np + 1], a, b[2], b[3]);
            }
        }
        // Only tiles reaching past the first query or the context's end need
        // the mask.
        if (k0 + BN - 1 > p0 || k0 + BN > L) {
#pragma unroll
            for (int nt = 0; nt < BN / 8; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                    const int qp = p0 + g + (e < 2 ? 0 : 8);
                    if (kp > qp || kp >= L) sc[nt][e] = -INFINITY;
                }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        // A row with no key yet keeps -inf; it subtracts 0 so exp2 gives 0.
        const float b0 = mx0 == -INFINITY ? 0.f : mx0, b1 = mx1 == -INFINITY ? 0.f : mx1;
        c0 = fast_exp2(m0 - b0);
        c1 = fast_exp2(m1 - b1);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < BN / 8; nt++) {
            sc[nt][0] = fast_exp2(sc[nt][0] - b0);
            sc[nt][1] = fast_exp2(sc[nt][1] - b0);
            sc[nt][2] = fast_exp2(sc[nt][2] - b1);
            sc[nt][3] = fast_exp2(sc[nt][3] - b1);
            s0 += sc[nt][0] + sc[nt][1];
            s1 += sc[nt][2] + sc[nt][3];
        }
        s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        l0 = l0 * c0 + s0;
        l1 = l1 * c1 + s1;
        m0 = mx0;
        m1 = mx1;
    };
    // The outputs rescaled, plus P V.
    auto values = [&](const bf16* cV) {
#pragma unroll
        for (int nt = 0; nt < HEAD_DIM / 8; nt++) {
            o[nt][0] *= c0;
            o[nt][1] *= c0;
            o[nt][2] *= c1;
            o[nt][3] *= c1;
        }
#pragma unroll
        for (int ks = 0; ks < BN / 16; ks++) {
            const uint32_t a[4] = {fa_bf2(sc[2 * ks][0], sc[2 * ks][1]), fa_bf2(sc[2 * ks][2], sc[2 * ks][3]),
                                   fa_bf2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]),
                                   fa_bf2(sc[2 * ks + 1][2], sc[2 * ks + 1][3])};
#pragma unroll
            for (int np = 0; np < HEAD_DIM / 16; np++) {
                uint32_t b[4];
                frag_b_t(b, cV, FA_STR, 16 * ks, np * 16);
                mma_bf16(o[2 * np], a, b[0], b[1]);
                mma_bf16(o[2 * np + 1], a, b[2], b[3]);
            }
        }
    };
    load(0, 0);
    int buf = 0;
    for (int k0 = 0; k0 < L; k0 += BN, buf ^= 1) {
        if (k0 + BN < L) {
            load(k0 + BN, buf ^ 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        if (k0 <= last) {
            // Keys past all of this warp's queries add nothing.
            scores(sK + buf * BN * FA_STR, k0);
            values(sV + buf * BN * FA_STR);
        }
        __syncthreads();  // this buffer is refilled two tiles on
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const int r = g + hf * 8;
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
#ifdef __HIP_PLATFORM_AMD__
#include "amd/attention.cuh"
#endif
// attn_group_g<G>_t<NT>_k<BN>: groups of G, NT query tiles, key tiles of BN;
// attn_group8_... reads an E4M3 cache; attn_groupr_... keeps the queries in
// registers. The host picks the variant whose shared memory the device
// allows.
#define ATTN_GROUP_V(G, NT, BN)                                                                                \
    extern "C" __global__ void __launch_bounds__(G * 32 * NT) attn_group_g##G##_t##NT##_k##BN(ATTN_FLASH_ARGS) {  \
        attn_flash_group<G, false, NT, BN, false>(ATTN_FLASH_PASS);                                     \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(G * 32 * NT) attn_group8_g##G##_t##NT##_k##BN(ATTN_FLASH_ARGS) { \
        attn_flash_group<G, true, NT, BN, false>(ATTN_FLASH_PASS);                                      \
    }
#define ATTN_GROUP_R(G, NT, BN)                                                                                 \
    extern "C" __global__ void __launch_bounds__(G * 32 * NT) attn_groupr_g##G##_t##NT##_k##BN(ATTN_FLASH_ARGS) { \
        attn_flash_group<G, false, NT, BN, true>(ATTN_FLASH_PASS);                                      \
    }
// Up to 99 KB (RTX 50-series): 8 warps of queries and 16-key tiles.
ATTN_GROUP_V(1, 8, 16)
ATTN_GROUP_V(2, 4, 16)
ATTN_GROUP_V(4, 2, 16)
ATTN_GROUP_V(6, 1, 16)
ATTN_GROUP_V(8, 1, 16)
// Hopper's 227 KB (fa_shape in model.rs), and a 32-key shape that fits
// 99 KB, for checking larger key tiles on any device.
ATTN_GROUP_V(4, 2, 64)
ATTN_GROUP_V(6, 2, 32)
ATTN_GROUP_V(2, 1, 32)
// Queries in registers: 32-key tiles in 99 KB, 64-key tiles in 227 KB.
ATTN_GROUP_R(4, 2, 32)
ATTN_GROUP_R(6, 1, 32)
ATTN_GROUP_R(4, 2, 64)
ATTN_GROUP_R(6, 1, 64)

// ---------------------------------------------------------------------------
// Flash attention for prompts on Hopper's wgmma (sm_90a). A block takes 64
// query tokens of two heads of one key/value group, a consumer warpgroup per
// head, so each 64-key tile serves 128 query rows. Q K^T runs as m64n64k16
// from shared memory, P V as m64n256k16 with P in registers (the score
// accumulators' layout is the A fragments') and V read transposed. Tiles of
// 64 rows x 256 bf16 are four 128-byte-swizzled atoms of 64 columns.
#define WG_TILE (64 * HEAD_DIM * 2)

// Byte offset of 16-byte chunk c (0..31) of row r in such a tile.
__device__ __forceinline__ uint32_t wg_off(int r, int c) {
    return (c >> 3) * 8192 + r * 128 + (((c & 7) ^ (r & 7)) << 4);
}

// A shared-memory matrix descriptor with 128-byte swizzling.
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | (uint64_t)((lbo & 0x3FFFF) >> 4) << 16 |
           (uint64_t)((sbo & 0x3FFFF) >> 4) << 32 | (uint64_t)1 << 62;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
// Pins registers in place around a wgmma stage, so the compiler neither
// moves their reads into it nor reuses them while it runs (CUTLASS's
// warpgroup_fence_operand).
template <int N>
__device__ __forceinline__ void wg_pin(float* r) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void wg_pin(uint32_t* r) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+r"(r[i])::"memory");
}

// d (64 x 64 f32) (+)= A (64 x 16, descriptor) B (16 x 64, K-major descriptor).
__device__ __forceinline__ void wg_s(float* d, uint64_t da, uint64_t db, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(acc));
}

// d (64 x 256 f32) += A (64 x 16, registers) B (16 x 256, transposed descriptor).
__device__ __forceinline__ void wg_o(float* d, const uint32_t* a, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128, %129, %130, %131}, %132, p, 1, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
// Arrives on bar once all of this thread's earlier cp.async copies land.
__device__ __forceinline__ void mbar_arrive_cp(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// A TMA box of a 3-D tensor map into shared memory, completing bytes on bar.
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
            dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
        : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, int parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}
#endif

// gdn_one's math for the step's first `seqs` sequences (one-token and
// verifying rows), built to keep memory busy: persistent blocks of 256
// threads take (sequence, value head) items in turn, and while one item
// computes, the next nbuf - 1 items' states (32 KB each in bf16, contiguous)
// load into shared memory, nbuf of 2 or 3 buffers. Thread (warp w,
// lane l) holds columns 8 (l % 16) .. + 7 of rows l / 16 + 2 w + 16 i,
// i < 8, so a warp reads two whole rows per step.
// With kS = k^T S and qS = q^T S of the decayed state, the update needs no
// second pass:
//   delta = (v - kS) * beta;  o = qS + delta (q . k);  S += k delta^T
extern "C" __global__ void __launch_bounds__(256, 2) gdn_decode(GDN_ARGS, int seqs, int nbuf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) unsigned char dsm[];
    __shared__ float qs[GDN_DIM], ks[GDN_DIM], ds[GDN_DIM], ys[GDN_DIM], red[2][8][GDN_DIM], sh[32];
    __shared__ __align__(8) uint64_t full[3];
    const Fp8Out fo{fq, fs, fmode, frows};
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int col = 8 * (lane & 15), row0 = (lane >> 4) + 2 * warp;
    const int bytes = (state_bf16 ? 2 : 4) * GDN_DIM * GDN_DIM;
    const int items = seqs * Hv;
    const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(full), buf0 = (uint32_t)__cvta_generic_to_shared(dsm);
    auto state_at = [&](int slot, int h) { return (size_t)slot * Hv + h; };
    // Every thread issues its share of item n's state load as 16-byte
    // cp.async copies that arrive on the item's barrier when they land (a
    // fresh or empty sequence's just arrive). TMA's bulk copy would be one
    // instruction, but a kernel using it costs about 14 MB of device memory
    // per CUDA graph node.
    auto issue = [&](int n) {
        const int it = blockIdx.x + n * gridDim.x;
        if (it >= items) return;
        const int s = it / Hv, h = it % Hv, b = n % nbuf;
        const uint32_t bar = bar0 + 8 * b;
        if (!fresh[s] && cu[s] < cu[s + 1]) {
            const char* src = (const char*)state + state_at(slots[s], h) * bytes;
            for (int c = tid; c < bytes / 16; c += blockDim.x)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(buf0 + b * bytes + 16 * c),
                             "l"(src + 16 * c)
                             : "memory");
            mbar_arrive_cp(bar);
        } else {
            mbar_arrive(bar);
        }
    };
    if (tid == 0)
        for (int b = 0; b < nbuf; b++) mbar_init(bar0 + 8 * b, blockDim.x);
    __syncthreads();
    for (int n = 0; n < nbuf - 1; n++) issue(n);
    const float qscale = rsqrtf((float)GDN_DIM);
    for (int n = 0; blockIdx.x + n * gridDim.x < items; n++) {
        const int it = blockIdx.x + n * gridDim.x, s = it / Hv, h = it % Hv, b = n % nbuf;
        const int t0 = cu[s], t1 = cu[s + 1];
        const bool is_fresh = fresh[s] != 0;
        mbar_wait(bar0 + 8 * b, (n / nbuf) & 1);
        float r[8][8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int at = (row0 + 16 * i) * GDN_DIM + col;
            if (is_fresh) {
#pragma unroll
                for (int e = 0; e < 8; e++) r[i][e] = 0.f;
            } else if (state_bf16) {
                unpack8(reinterpret_cast<const uint4*>(dsm + b * bytes)[at / 8], r[i]);
            } else {
                const float4* p = reinterpret_cast<const float4*>(dsm + b * bytes) + at / 4;
                const float4 x = p[0], y = p[1];
                r[i][0] = x.x, r[i][1] = x.y, r[i][2] = x.z, r[i][3] = x.w;
                r[i][4] = y.x, r[i][5] = y.y, r[i][6] = y.z, r[i][7] = y.w;
            }
        }
        __syncthreads();  // buffer b is read; the buffer of item n - 1 too
        issue(n + nbuf - 1);
        if (t0 == t1) continue;
        const int kh = tiled ? h % Hk : h / (Hv / Hk);
        const bool keep = is_spec(outs, spec, s);
        const bool round_qk = ((t1 - t0) > 1 && !keep) || is_fresh;
        const float na = neg_a[h], db = dt_bias[h];
        for (int t = t0; t < t1; t++) {
            const bf16* crow = conv + (size_t)t * C;
            const bf16* prow = proj + (size_t)t * pstride + C;
            float q = 0.f, k = 0.f, v = 0.f;
            if (tid < GDN_DIM) {
                q = b2f(crow[kh * GDN_DIM + tid]);
                k = b2f(crow[Hk * GDN_DIM + kh * GDN_DIM + tid]);
                v = b2f(crow[2 * Hk * GDN_DIM + h * GDN_DIM + tid]);
            }
            const float bb = b2f(prow[Hv * GDN_DIM + h]), a = b2f(prow[Hv * GDN_DIM + Hv + h]);
            const float qq = block_sum(q * q, sh), kk = block_sum(k * k, sh);
            q = q / sqrtf(qq + 1e-6f);
            k = k / sqrtf(kk + 1e-6f);
            if (round_qk) {
                q = rbf(q);
                k = rbf(k);
            }
            q *= qscale;
            if (tid < GDN_DIM) {
                qs[tid] = q;
                ks[tid] = k;
            }
            const float qk = block_sum(tid < GDN_DIM ? q * k : 0.f, sh);  // its syncs publish qs and ks
            const float beta = 1.f / (1.f + expf(-bb));
            const float x = a + db;
            const float decay = expf(na * (fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)))));
            float kS[8] = {}, qS[8] = {};
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float ki = ks[row0 + 16 * i], qi = qs[row0 + 16 * i];
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    r[i][e] *= decay;
                    kS[e] += ki * r[i][e];
                    qS[e] += qi * r[i][e];
                }
            }
#pragma unroll
            for (int e = 0; e < 8; e++) {
                kS[e] += __shfl_xor_sync(0xffffffffu, kS[e], 16);
                qS[e] += __shfl_xor_sync(0xffffffffu, qS[e], 16);
            }
            if (lane < 16) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    red[0][warp][col + e] = kS[e];
                    red[1][warp][col + e] = qS[e];
                }
            }
            __syncthreads();
            if (tid < GDN_DIM) {
                float ksum = 0.f, qsum = 0.f;
#pragma unroll
                for (int w = 0; w < 8; w++) {
                    ksum += red[0][w][tid];
                    qsum += red[1][w][tid];
                }
                const float delta = (v - ksum) * beta;
                ds[tid] = delta;
                ys[tid] = rbf(qsum + delta * qk);
            }
            __syncthreads();
            float dl[8];
#pragma unroll
            for (int e = 0; e < 8; e++) dl[e] = ds[col + e];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 8; e++) r[i][e] += ks[row0 + 16 * i] * dl[e];
            if (warp == 0) {
                float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (lane < 16)
#pragma unroll
                    for (int e = 0; e < 8; e++) o[e] = ys[col + e];
                gdn_gate8(o, t, h, Hv, proj, pstride, C, norm_w, out, eps, fo, fmode, lane);
            }
            if (keep) {
                // The state after each token, so rejected drafts cost nothing.
                const size_t kb = state_at(outs[(size_t)s * spec + (t - t0)], h) * GDN_DIM * GDN_DIM;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const size_t at = kb + (row0 + 16 * i) * GDN_DIM + col;
                    if (state_bf16) {
                        *reinterpret_cast<uint4*>((bf16*)state + at) = pack8(r[i]);
                    } else {
                        float4* p = reinterpret_cast<float4*>((float*)state + at);
                        p[0] = make_float4(r[i][0], r[i][1], r[i][2], r[i][3]);
                        p[1] = make_float4(r[i][4], r[i][5], r[i][6], r[i][7]);
                    }
                }
            }
        }
        if (keep) continue;
        const size_t sb = state_at(slots[s], h) * GDN_DIM * GDN_DIM;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const size_t at = sb + (row0 + 16 * i) * GDN_DIM + col;
            if (state_bf16) {
                *reinterpret_cast<uint4*>((bf16*)state + at) = pack8(r[i]);
            } else {
                float4* p = reinterpret_cast<float4*>((float*)state + at);
                p[0] = make_float4(r[i][0], r[i][1], r[i][2], r[i][3]);
                p[1] = make_float4(r[i][4], r[i][5], r[i][6], r[i][7]);
            }
        }
    }
#endif
}

// Warp-specialized as FlashAttention 3 is, and persistent: a block per SM
// loops over work items (64 query tokens of two heads of one key/value
// group). Warpgroup 0 produces: each item's queries, then each 64-key tile's
// keys and values into one of two stages by TMA (per 16-token page and
// 64-column swizzle atom one box of the cache's tensor map,
// [slot][kv head][256]), one page per warp; it keeps 40 registers.
// Warpgroups 1 and 2 consume, a head each, with 232. Keys and values pass
// through separate mbarrier pipelines (full: the bytes landed; empty: every
// consumer warp is done with them) that run on across items, so a key stage
// frees as soon as Q K^T has read it and the next tile's keys load while
// P V runs. Pages past the context read past the map's slots, which TMA
// fills with zeros.
//
// Items are ordered key/value head first, then head pair, then tile, the
// last tiles (the most keys) first, and block b takes items b, b + grid,
// ...: all blocks work through one head's cache together, so a long
// context's keys and values, too big for L2 when blocks drift apart, stream
// from memory about once per head.
#define WS_SMEM (6 * WG_TILE + 128 + 8 * 256)
#define ATTN_WS_ARGS                                                                                    \
    ATTN_FLASH_ARGS, const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, \
        int nslots
template <int G>
__device__ __forceinline__ void attn_ws_body(ATTN_FLASH_ARGS, const CUtensorMap* kmap, const CUtensorMap* vmap,
                                             int nslots) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char wsm[];
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(wsm);
    unsigned char* sQ = wsm;                // [2][WG_TILE]
    // Then two stages of keys (2 + stage) and values (4 + stage).
    // Barriers: queries full and empty, then full and empty for keys and
    // for values, per stage.
    const uint32_t bars = base + 6 * WG_TILE, qfull = bars, qempty = bars + 8, fullk = bars + 16,
                   fullv = bars + 32, emptyk = bars + 48, emptyv = bars + 64;
    const int tid = threadIdx.x, wg = tid >> 7, w = (tid >> 5) & 3, lane = tid & 31, pt = tid & 127;
    int ntiles = 0;
    for (int i = 0; i < seqs; i++) {
        const int n = cu[i + 1] - cu[i];
        if (n > 1 && !is_spec(outs, spec, i)) ntiles += (n + 63) / 64;
    }
    const int items = ntiles * (G / 2) * Hkv;
    // Item it: key/value head, head pair, sequence and first token.
    int kvh, pair, s, tb;
    auto item = [&](int it) {
        kvh = it / (ntiles * (G / 2));
        const int r = it - kvh * ntiles * (G / 2);
        pair = r / ntiles;
        int tile = ntiles - 1 - (r - pair * ntiles);
        for (s = 0; s < seqs; s++) {
            const int n = cu[s + 1] - cu[s];
            if (n <= 1 || is_spec(outs, spec, s)) continue;
            const int tiles = (n + 63) / 64;
            if (tile < tiles) break;
            tile -= tiles;
        }
        tb = cu[s] + tile * 64;
    };
    if (tid == 0) {
        mbar_init(qfull, 128);
        mbar_init(qempty, 8);
        for (int i = 0; i < 2; i++) {
            mbar_init(fullk + 8 * i, 4);
            mbar_init(fullv + 8 * i, 4);
            mbar_init(emptyk + 8 * i, 8);
            mbar_init(emptyv + 8 * i, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
        if (pt == 0) {
            asm volatile("prefetch.tensormap [%0];\n" ::"l"(kmap) : "memory");
            asm volatile("prefetch.tensormap [%0];\n" ::"l"(vmap) : "memory");
        }
        int n = 0, i = 0;  // tiles and items so far
        for (int it = blockIdx.x; it < items; it += gridDim.x, i++) {
            item(it);
            const int nb = min(64, cu[s + 1] - tb), L = pos[tb] + nb, T = (L + 63) / 64;
            if (i > 0) mbar_wait(qempty, (i - 1) & 1);
            for (int e = pt; e < 2 * 64 * 32; e += 128) {
                const int hw = e >> 11, r = (e >> 5) & 63, c = e & 31;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < nb) {
                    float f[8];
                    unpack8(*reinterpret_cast<const uint4*>(
                                q + ((size_t)(tb + r) * Hq + kvh * G + 2 * pair + hw) * HEAD_DIM + c * 8),
                            f);
#pragma unroll
                    for (int j = 0; j < 8; j++) f[j] *= scale * LOG2E;  // scores in log2 units
                    v = pack8(f);
                }
                *reinterpret_cast<uint4*>(sQ + hw * WG_TILE + wg_off(r, c)) = v;
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            mbar_arrive(qfull);
            if (lane == 0) {
                // Keys run a tile ahead of values: K(k + 1) goes out before
                // V(k). A stage is reused once the consumers have released
                // its previous tile.
                const int* bt = block_table + (size_t)s * max_blocks;
                auto slot = [&](int k) {
                    const int kp = 64 * k + 16 * w;
                    return kp < L ? bt[kp / block_size] * block_size : nslots;
                };
                auto load = [&](int k, int sl, const CUtensorMap* map, uint32_t full, uint32_t empty, int buf) {
                    const int j = n + k, st = j & 1;
                    if (j >= 2) mbar_wait(empty + 8 * st, ((j >> 1) - 1) & 1);
                    mbar_expect(full + 8 * st, WG_TILE / 4);
                    const uint32_t d = base + (buf + st) * WG_TILE + w * 2048;
#pragma unroll
                    for (int a = 0; a < 4; a++) tma3(d + a * 8192, map, 64 * a, kvh, sl, full + 8 * st);
                };
                int cur = slot(0);
                load(0, cur, kmap, fullk, emptyk, 2);
                for (int k = 0; k < T; k++) {
                    const int next = k + 1 < T ? slot(k + 1) : 0;
                    if (k + 1 < T) load(k + 1, next, kmap, fullk, emptyk, 2);
                    load(k, cur, vmap, fullv, emptyv, 4);
                    cur = next;
                }
            }
            __syncwarp();
            n += T;
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int c_wg = wg - 1;
    const uint32_t qa = base + c_wg * WG_TILE, spin = base + 6 * WG_TILE + 128;
    const int r0 = 16 * w + (lane >> 2);
    int n = 0, i = 0;  // tiles and items so far
    for (int it = blockIdx.x; it < items; it += gridDim.x, i++) {
    item(it);
    const int nb = min(64, cu[s + 1] - tb), p0 = pos[tb], L = p0 + nb, T = (L + 63) / 64;
    const int h = kvh * G + 2 * pair + c_wg, q0 = p0 + r0, q1 = q0 + 8;
    float o[128];
#pragma unroll
    for (int j = 0; j < 128; j++) o[j] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    // Scores of the newest tile, then its probabilities (f32) until P V of
    // the tile before has read pa; only then are they packed into pa. The
    // corrections c0, c1 rescale o just before the tile's P V is issued, so
    // they run while Q K^T is on the tensor cores.
    float sc[32], c0 = 1.f, c1 = 1.f;
    uint32_t pa[4][4];
    // Waits for tile k's keys or values (TMA writes; wgmma reads them in
    // the same async proxy, so no fence is needed).
    auto ready = [&](uint32_t full, int k) { mbar_wait(full + 8 * ((n + k) & 1), ((n + k) >> 1) & 1); };
    auto release = [&](uint32_t empty, int k) {
        if (lane == 0) mbar_arrive(empty + 8 * ((n + k) & 1));
    };
    auto issue_s = [&](int k) {
        const uint32_t ka = base + (2 + ((n + k) & 1)) * WG_TILE;
        wg_pin<32>(sc);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const uint32_t off = (kk >> 2) * 8192 + (kk & 3) * 32;
            wg_s(sc, wg_desc(qa + off, 16, 1024), wg_desc(ka + off, 16, 1024), kk > 0);
        }
        wg_commit();
    };
    auto softmax = [&](int k) {
        const int k0 = 64 * k;
        if (k0 + 63 > p0 || k0 + 64 > L) {
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kp = k0 + 8 * j + 2 * (lane & 3) + (e & 1);
                    if (kp > (e < 2 ? q0 : q1) || kp >= L) sc[4 * j + e] = -INFINITY;
                }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            mx0 = fmaxf(mx0, fmaxf(sc[4 * j], sc[4 * j + 1]));
            mx1 = fmaxf(mx1, fmaxf(sc[4 * j + 2], sc[4 * j + 3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float b0 = mx0 == -INFINITY ? 0.f : mx0, b1 = mx1 == -INFINITY ? 0.f : mx1;
        c0 = fast_exp2(m0 - b0);
        c1 = fast_exp2(m1 - b1);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            sc[i] = fast_exp2(sc[i] - ((i & 2) ? b1 : b0));
            if (i & 2) s1 += sc[i];
            else s0 += sc[i];
        }
        s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        l0 = l0 * c0 + s0;
        l1 = l1 * c1 + s1;
        m0 = mx0;
        m1 = mx1;
        wg_pin<32>(sc);
    };
    auto pack = [&]() {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            pa[kk][0] = fa_bf2(sc[8 * kk + 0], sc[8 * kk + 1]);
            pa[kk][1] = fa_bf2(sc[8 * kk + 2], sc[8 * kk + 3]);
            pa[kk][2] = fa_bf2(sc[8 * kk + 4], sc[8 * kk + 5]);
            pa[kk][3] = fa_bf2(sc[8 * kk + 6], sc[8 * kk + 7]);
        }
    };
    auto rescale = [&]() {
        if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
            for (int i = 0; i < 128; i++) o[i] *= (i & 2) ? c1 : c0;
        }
    };
    auto issue_pv = [&](int k) {
        const uint32_t va = base + (4 + ((n + k) & 1)) * WG_TILE;
        wg_pin<128>(o);
        wg_pin<16>(&pa[0][0]);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) wg_o(o, pa[kk], wg_desc(va + kk * 2048, 8192, 1024));
        wg_commit();
    };
    mbar_wait(qfull, i & 1);
    ready(fullk, 0);
    issue_s(0);
    wg_wait0();
    wg_pin<32>(sc);
    release(emptyk, 0);
    softmax(0);
    pack();
    for (int k = 0; k + 1 < T; k++) {
        ready(fullk, k + 1);
        issue_s(k + 1);
        rescale();
        ready(fullv, k);
        issue_pv(k);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        wg_pin<32>(sc);
        release(emptyk, k + 1);
        softmax(k + 1);
        // ptxas otherwise hoists the wait above the softmax: a shared store
        // of its last result (the row sums) holds it below.
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(spin + 8 * (tid - 128)), "f"(l0), "f"(l1) : "memory");
        wg_wait0();
        wg_pin<128>(o);
        wg_pin<16>(&pa[0][0]);
        release(emptyv, k);
        pack();
    }
    rescale();
    ready(fullv, T - 1);
    issue_pv(T - 1);
    wg_wait0();
    wg_pin<128>(o);
    release(emptyv, T - 1);
    if (lane == 0) mbar_arrive(qempty);  // the queries are free for the next item
    n += T;
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const int r = r0 + 8 * hf;
        if (r >= nb) continue;
        const size_t t = (size_t)tb + r;
        const float inv = hf ? i1 : i0;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int d = 8 * j + 2 * (lane & 3);
            float y[2];
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const float ov = rbf(o[4 * j + 2 * hf + e] * inv);
                const float gate = b2f(qkv[t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d + e]);
                y[e] = ov / (1.f + expf(-gate));
            }
            *reinterpret_cast<__nv_bfloat162*>(out + (t * Hq + h) * HEAD_DIM + d) = __floats2bfloat162_rn(y[0], y[1]);
        }
    }
    }
#endif
}
// attn_ws_g<G>: a block per SM, 384 threads.
#define ATTN_WS_PASS ATTN_FLASH_PASS, &kmap, &vmap, nslots
extern "C" __global__ void __launch_bounds__(384, 1) attn_ws_g4(ATTN_WS_ARGS) { attn_ws_body<4>(ATTN_WS_PASS); }
extern "C" __global__ void __launch_bounds__(384, 1) attn_ws_g6(ATTN_WS_ARGS) { attn_ws_body<6>(ATTN_WS_PASS); }
#ifndef __HIP_PLATFORM_AMD__
#include "attn_sm100.cuh"  // B200 (tcgen05) only
#endif


// Combines the key ranges of split decode attention (attn_body with `part`)
// in range order, then gates as attn_body does: a block of 256 threads per
// (token, query head), one per dimension, then one warp for the FP8 copy.
extern "C" __global__ void __launch_bounds__(HEAD_DIM)
    attn_merge(const float* __restrict__ part, int Z, int min_chunk, const bf16* __restrict__ qkv, int qstride,
               int Hq, const int* __restrict__ pos, const int* __restrict__ tok_seq,
               const int* __restrict__ cu, int skip_prompts, bf16* __restrict__ out, uint8_t* fq, void* fs,
               int fmode, int frows, const int* __restrict__ outs, int spec) {
    __shared__ float ys[HEAD_DIM];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int sq = tok_seq[t];
    if (skip_prompts && cu[sq + 1] - cu[sq] > 1 && !is_spec(outs, spec, sq)) return;
    const int nz = split_parts(pos[t] + 1, Z, min_chunk);
    const float* pp = part + (((size_t)sq * spec + (t - cu[sq])) * Hq + h) * Z * (HEAD_DIM + 2);
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
// Also applies grammar masks: a row with mask_idx >= 0 keeps only the
// tokens whose bit is set in mask number mask_idx.
extern "C" __global__ void penalize(float* __restrict__ logits, int V, const int* __restrict__ pen_cu,
                                    const int* __restrict__ pen_tok, const float* __restrict__ pen_add,
                                    const float* __restrict__ pen_rep, const int* __restrict__ mask_idx,
                                    const uint32_t* __restrict__ masks, int words) {
    float* row = logits + (size_t)blockIdx.x * V;
    const int mi = mask_idx[blockIdx.x];
    if (mi >= 0) {
        const uint32_t* bits = masks + (size_t)mi * words;
        for (int i = threadIdx.x; i < V; i += blockDim.x)
            if (!((bits[i >> 5] >> (i & 31)) & 1u)) row[i] = -INFINITY;
        __syncthreads();
    }
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

// Log-probabilities for the output rows that ask for them, from the logits
// as sampling saw them (after penalties, biases and grammar masks, before
// temperature): for entry i, output row rows[i], the picked token's
// log-probability and the ks[i] (at most LOGPROBS_TOP) most likely tokens,
// written to out + i * (1 + 2 * LOGPROBS_TOP) as [picked, (id bits, logprob) ...].
#define LOGPROBS_TOP 20
#define LOGPROBS_CAND 64
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS)
    logprobs(const float* __restrict__ logits, int V, const int* __restrict__ rows, const int* __restrict__ ks,
             const int* __restrict__ picked, float* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ float hist[SAMPLE_WARPS * 256];
    __shared__ float bins[256];
    __shared__ uint32_t shared[2];
    __shared__ float cv[LOGPROBS_CAND];
    __shared__ int ci[LOGPROBS_CAND];
    __shared__ int nc;
    const int r = rows[blockIdx.x], k = min(ks[blockIdx.x], LOGPROBS_TOP);
    const float* row = logits + (size_t)r * V;
    float* o = out + (size_t)blockIdx.x * (1 + 2 * LOGPROBS_TOP);
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) m = fmaxf(m, row[i]);
    m = block_max(m, sv);
    float s = 0.f;
    for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) s += expf(row[i] - m);
    // Block sum, as block_max does it.
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < SAMPLE_WARPS; w++) t += sv[w];
        sv[0] = t;
        nc = 0;
    }
    __syncthreads();
    const float lse = m + logf(sv[0]);
    if (threadIdx.x == 0) o[0] = row[picked[r]] - lse;
    if (k == 0) return;
    // The k-th largest key, then every element at or above it (ties may add a few).
    const uint32_t floor = radix_select(row, V, 1.f, m, 0, (float)k, false, hist, bins, shared);
    for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS)
        if (fkey(row[i]) >= floor) {
            const int at = atomicAdd(&nc, 1);
            if (at < LOGPROBS_CAND) {
                cv[at] = row[i];
                ci[at] = i;
            }
        }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int n = min(nc, LOGPROBS_CAND);
        // Largest first, ties by lower id.
        for (int a = 1; a < n; a++) {
            const float v = cv[a];
            const int id = ci[a];
            int b = a - 1;
            while (b >= 0 && (cv[b] < v || (cv[b] == v && ci[b] > id))) {
                cv[b + 1] = cv[b];
                ci[b + 1] = ci[b];
                b--;
            }
            cv[b + 1] = v;
            ci[b + 1] = id;
        }
        for (int j = 0; j < k; j++) {
            o[1 + 2 * j] = __int_as_float(j < n ? ci[j] : -1);
            o[2 + 2 * j] = j < n ? cv[j] - lse : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-token prediction: one extra layer drafts the token after next from a
// token and the model's final hidden state before it.

// The token each row's draft input takes: mtp_ids[t] when not negative, else
// the pick of output row -mtp_ids[t] - 1.
extern "C" __global__ void mtp_tokens(const int* __restrict__ mtp_ids, const int* __restrict__ picked,
                                      int* __restrict__ out, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int x = mtp_ids[t];
    out[t] = x >= 0 ? x : picked[-x - 1];
}

// The draft layer's input row: [norm_e(embedding) | norm_h(hidden)], where
// hidden is the model's final norm of x + res (unrounded f32 sum), rounded to
// bf16 as the model's output is; or, with x null, res itself (a draft's own
// normalized output, for the next draft). Weights are f32 multipliers.
extern "C" __global__ void mtp_input(const bf16* __restrict__ emb, const bf16* __restrict__ x,
                                     const bf16* __restrict__ res, const float* __restrict__ final_w,
                                     const float* __restrict__ we, const float* __restrict__ wh,
                                     bf16* __restrict__ out, int hidden, float eps) {
    __shared__ float sh[32];
    const size_t r = blockIdx.x;
    float e[4][8], v[4][8];
    float se = 0.f, sv = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < hidden) {
            float a[8];
            unpack8(*reinterpret_cast<const uint4*>(emb + r * hidden + i), e[c]);
            unpack8(*reinterpret_cast<const uint4*>(res + r * hidden + i), v[c]);
            if (x) unpack8(*reinterpret_cast<const uint4*>(x + r * hidden + i), a);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                if (x) v[c][k] += a[k];
                se += e[c][k] * e[c][k];
                sv += v[c][k] * v[c][k];
            }
        }
    }
    se = block_sum(se, sh);
    sv = block_sum(sv, sh);
    const float ie = 1.f / sqrtf(se / hidden + eps), iv = 1.f / sqrtf(sv / hidden + eps);
    float sf = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < hidden) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                if (x) v[c][k] = rbf(v[c][k] * iv * final_w[i + k]);
                sf += v[c][k] * v[c][k];
            }
        }
    }
    sf = block_sum(sf, sh);
    const float iff = 1.f / sqrtf(sf / hidden + eps);
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (i < hidden) {
            float a[8], b[8];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                a[k] = e[c][k] * ie * we[i + k];
                b[k] = v[c][k] * iff * wh[i + k];
            }
            *reinterpret_cast<uint4*>(out + r * 2 * hidden + i) = pack8(a);
            *reinterpret_cast<uint4*>(out + r * 2 * hidden + hidden + i) = pack8(b);
        }
    }
}

// For each sequence, the token row whose draft output becomes its next draft:
// a sequence verifying drafts accepts draft j + 1 while the pick at its
// token j equals it, and drafts from the last accepted row; others draft from
// their last row.
extern "C" __global__ void mtp_accept(const int* __restrict__ ids, const int* __restrict__ cu,
                                      const int* __restrict__ ocu, const int* __restrict__ picked,
                                      const int* __restrict__ outs, int spec, int* __restrict__ rows, int S) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    const int t0 = cu[s], n = cu[s + 1] - t0;
    int j = n - 1;
    if (is_spec(outs, spec, s)) {
        j = 0;
        while (j < n - 1 && picked[ocu[s] + j] == ids[t0 + j + 1]) j++;
    }
    rows[s] = t0 + j;
}

// Metadata of draft pass k (k >= 1): one row per sequence at k positions past
// its draft row, writing the draft layer's KV only up to `limits` (the last
// position the sequence holds; -1 for padding).
extern "C" __global__ void mtp_next(const int* __restrict__ rows, const int* __restrict__ pos,
                                    const int* __restrict__ rope, int rope_stride,
                                    const int* __restrict__ block_table, int max_blocks, int block_size,
                                    const int* __restrict__ limits, int k, int* __restrict__ dpos,
                                    int* __restrict__ dslot, int* __restrict__ drope, int S) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    const int r = rows[s], p = pos[r] + k;
    dpos[s] = p;
    dslot[s] = p <= limits[s] ? block_table[(size_t)s * max_blocks + p / block_size] * block_size + p % block_size
                              : -1;
    for (int axis = 0; axis < 3; axis++) drope[axis * S + s] = rope[axis * rope_stride + r] + k;
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

#ifdef __HIP_PLATFORM_AMD__
#include "amd/skinny.cuh"
#endif

#include "quant.cuh"
#include "fp4.cuh"
#include "lora.cuh"
