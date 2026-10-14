// Matrix-vector products for decoding on AMD (Gpu::gemm_ld).
#pragma once

// y[r][n] = x[r] . W[n] for r < R (1 to 4 rows) with bf16 W [N][K]: the
// matrix-vector products of decoding, which hipBLASLt runs at about 60% of
// HBM bandwidth on an MI325X. A block of four 64-lane waves takes four rows
// of W, each wave a quarter of K in 16-byte loads; partial sums reduce over
// lanes, then over waves in a fixed order (deterministic).
template <int R>
__device__ __forceinline__ void skinny_bf16(const bf16* __restrict__ W, const bf16* __restrict__ x, void* __restrict__ y,
                                            int K, int N, int ldy, int f32_out) {
    constexpr int NR = 4;
    __shared__ float part[4][NR * R];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, n0 = blockIdx.x * NR;
    const int kq = K / 4, kb = wave * kq, ke = kb + kq;
    float acc[NR][R];
#pragma unroll
    for (int i = 0; i < NR; i++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[i][r] = 0.f;
    for (int k = kb + 8 * lane; k < ke; k += 512) {
        uint4 w[NR], xv[R];
#pragma unroll
        for (int i = 0; i < NR; i++)
            w[i] = n0 + i < N ? *reinterpret_cast<const uint4*>(W + (size_t)(n0 + i) * K + k) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int r = 0; r < R; r++) xv[r] = *reinterpret_cast<const uint4*>(x + (size_t)r * K + k);
#pragma unroll
        for (int i = 0; i < NR; i++) {
            const uint32_t* wp = reinterpret_cast<const uint32_t*>(&w[i]);
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint32_t* xp = reinterpret_cast<const uint32_t*>(&xv[r]);
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    acc[i][r] = fmaf(__uint_as_float(wp[e] << 16), __uint_as_float(xp[e] << 16), acc[i][r]);
                    acc[i][r] = fmaf(__uint_as_float(wp[e] & 0xffff0000u), __uint_as_float(xp[e] & 0xffff0000u), acc[i][r]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NR; i++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[i][r];
#pragma unroll
            for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
            if (lane == 0) part[wave][i * R + r] = v;
        }
    __syncthreads();
    if (threadIdx.x < NR * R) {
        const int i = threadIdx.x / R, r = threadIdx.x % R, n = n0 + i;
        const float v = ((part[0][threadIdx.x] + part[1][threadIdx.x]) + part[2][threadIdx.x]) + part[3][threadIdx.x];
        if (n < N) {
            if (f32_out) ((float*)y)[(size_t)r * ldy + n] = v;
            else ((bf16*)y)[(size_t)r * ldy + n] = f2b(v);
        }
    }
}
#define SKINNY(R)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) skinny_bf16_r##R(const bf16* W, const bf16* x, void* y,   \
                                                                       int K, int N, int ldy, int f32_out) { \
        skinny_bf16<R>(W, x, y, K, N, ldy, f32_out);                                                           \
    }
SKINNY(1) SKINNY(2) SKINNY(3) SKINNY(4)
