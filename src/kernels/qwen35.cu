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
    for (int t = t0; t < t1; t++) {
        const float x3 = b2f(proj[(size_t)t * pstride + c]);
        float acc = 0.f;
        acc += x0 * k.x;
        acc += x1 * k.y;
        acc += x2 * k.z;
        acc += x3 * k.w;
        out[(size_t)t * C + c] = f2b(silu(acc));
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    st[c] = f2b(x0);
    st[C + c] = f2b(x1);
    st[2 * C + c] = f2b(x2);
}

// Gated delta rule, one block of 256 threads per (sequence, value head). The
// head's [k, v] state lives in registers: threads 2j and 2j+1 hold rows 0..63
// and 64..127 of column j. Per token:
//   S *= exp(g); delta = (v - k^T S) * beta; S += k delta^T; o = q^T S
// then a gated RMSNorm of o with silu(z). The state is f32: [slot][Hv][128][128].
// conv rows: [q (Hk*128) | k (Hk*128) | v (Hv*128)]; proj rows after the
// convolution inputs: [z (Hv*128) | b (Hv) | a (Hv)].
#define GDN_HALF (GDN_DIM / 2)
extern "C" __global__ void __launch_bounds__(2 * GDN_DIM)
    gdn_recurrent(const bf16* __restrict__ conv, const bf16* __restrict__ proj, int pstride, int C,
                  int Hk, int Hv, const float* __restrict__ neg_a, const float* __restrict__ dt_bias,
                  const float* __restrict__ norm_w, float* __restrict__ state,
                  const int* __restrict__ cu, const int* __restrict__ slots,
                  const int* __restrict__ fresh, bf16* __restrict__ out, float eps, int tiled,
                  uint8_t* fq, void* fs, int fmode, int frows) {
    __shared__ float qs[GDN_DIM], ks[GDN_DIM], sh[32], ys[GDN_DIM];
    const Fp8Out fo{fq, fs, fmode, frows};
    const int s = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int j = tid >> 1, half = tid & 1, i0 = half * GDN_HALF;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    // Value head h reads key head h / (Hv / Hk) (grouped order), or h % Hk in
    // the tiled order GGUF files use.
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    float* S = state + ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float r[GDN_HALF];
    const bool is_fresh = fresh[s] != 0;
#pragma unroll
    for (int i = 0; i < GDN_HALF; i++) r[i] = is_fresh ? 0.f : S[(i0 + i) * GDN_DIM + j];
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
    for (int i = 0; i < GDN_HALF; i++) S[(i0 + i) * GDN_DIM + j] = r[i];
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
                                          uint8_t* fq, void* fs, int fmode, int frows) {
    __shared__ float sm_m[WARPS][G], sm_s[WARPS][G];
    __shared__ float sm_acc[WARPS][G][HEAD_DIM];
    const int t = blockIdx.x, kvh = blockIdx.y;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int L = pos[t] + 1;
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
    for (int j = w; j < L; j += WARPS) {
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
            const float o = rbf(A / S);
            const float gate = b2f(qkv[(size_t)t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d]);
            const float y = o / (1.f + expf(-gate));
            out[((size_t)t * Hq + h) * HEAD_DIM + d] = f2b(y);
            if (fmode) sm_acc[0][g][d] = y;  // reused: the merge above has read it
        }
    }
    if (fmode) {
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
        int frows) {                                                                                   \
        attn_body<G, WARPS, F8>(q, kcache, vcache, kscale, vscale, qkv, qstride, Hq, Hkv, pos,         \
                                tok_seq, block_table, max_blocks, block_size, out, scale, fq, fs,      \
                                fmode, frows);                                                         \
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
