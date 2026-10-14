// Matrix-vector products for decoding on AMD (Gpu::gemm_ld).
#pragma once

// y[r][n] = x[r] . W[n] for r < R (1 to 4 rows) with bf16 W [N][K]: the
// products of decoding, which hipBLASLt runs at about 60% of HBM bandwidth on
// an MI325X. As vLLM's wvSplitK does, each 64-lane wave takes YT whole rows
// of W over all of K, with UN 16-byte loads per row and lane in flight, and
// reduces within the wave only (deterministic, no shared memory).
typedef unsigned int pe_u4 __attribute__((ext_vector_type(4)));
// A 16-byte load that bypasses the caches' retention (weights are read once).
__device__ __forceinline__ uint4 ld_once(const bf16* p) {
    const pe_u4 v = __builtin_nontemporal_load(reinterpret_cast<const pe_u4*>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
}
template <int R>
__device__ __forceinline__ void skinny_bf16(const bf16* __restrict__ W, const bf16* __restrict__ x, void* __restrict__ y,
                                            int K, int N, int ldy, int f32_out) {
    constexpr int YT = 2, UN = 4, STEP = 64 * 8;
    const int lane = threadIdx.x & 63, n0 = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * YT;
    if (n0 >= N) return;
    const bf16* w[YT];
#pragma unroll
    for (int i = 0; i < YT; i++) w[i] = W + (size_t)min(n0 + i, N - 1) * K;
    float acc[YT][R];
#pragma unroll
    for (int i = 0; i < YT; i++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[i][r] = 0.f;
    auto dot = [&](const uint4& wv, const uint4& xv, float a) {
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
        const uint32_t* xp = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            a = fmaf(__uint_as_float(wp[e] << 16), __uint_as_float(xp[e] << 16), a);
            a = fmaf(__uint_as_float(wp[e] & 0xffff0000u), __uint_as_float(xp[e] & 0xffff0000u), a);
        }
        return a;
    };
    int k = 8 * lane;
    for (; k + (UN - 1) * STEP < K; k += UN * STEP) {
        uint4 wv[YT][UN];
#pragma unroll
        for (int i = 0; i < YT; i++)
#pragma unroll
            for (int u = 0; u < UN; u++) wv[i][u] = ld_once(w[i] + k + u * STEP);
#pragma unroll
        for (int u = 0; u < UN; u++)
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)r * K + k + u * STEP);
#pragma unroll
                for (int i = 0; i < YT; i++) acc[i][r] = dot(wv[i][u], xv, acc[i][r]);
            }
    }
    for (; k < K; k += STEP)
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)r * K + k);
#pragma unroll
            for (int i = 0; i < YT; i++) acc[i][r] = dot(*reinterpret_cast<const uint4*>(w[i] + k), xv, acc[i][r]);
        }
#pragma unroll
    for (int i = 0; i < YT; i++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[i][r];
#pragma unroll
            for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
            if (lane == 0 && n0 + i < N) {
                if (f32_out) ((float*)y)[(size_t)r * ldy + n0 + i] = v;
                else ((bf16*)y)[(size_t)r * ldy + n0 + i] = f2b(v);
            }
        }
}
#define SKINNY(R)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) skinny_bf16_r##R(const bf16* W, const bf16* x, void* y,   \
                                                                       int K, int N, int ldy, int f32_out) { \
        skinny_bf16<R>(W, x, y, K, N, ldy, f32_out);                                                           \
    }
SKINNY(1) SKINNY(2) SKINNY(3) SKINNY(4)
