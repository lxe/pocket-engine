// FP8 weights for decoding on AMD (--fp8): see Kind::Fp8Row in linear.rs.
#pragma once

// gfx942's E4M3 is the FNUZ variant: the largest value is 240.
#define FP8W_MAX 240.f
typedef float pe_f2 __attribute__((ext_vector_type(2)));

// One block of 256 threads per row: the row's scale, a power of two so that
// every E4M3 weight times it is exact in bf16; the E4M3 weights; and that
// bf16 copy (`dense`, for batches past the matrix-vector kernel, which thus
// computes on exactly the same values). cols is a multiple of 8.
extern "C" __global__ void __launch_bounds__(256) fp8w_quantize(const bf16* __restrict__ w, int cols,
                                                                 uint8_t* __restrict__ q, float* __restrict__ scales,
                                                                 bf16* __restrict__ dense) {
    __shared__ float sh[4];
    const size_t row = blockIdx.x;
    const bf16* wr = w + row * cols;
    float amax = 0.f;
    for (int i = threadIdx.x * 8; i < cols; i += blockDim.x * 8) {
        float v[8];
        unpack8(*reinterpret_cast<const uint4*>(wr + i), v);
#pragma unroll
        for (int e = 0; e < 8; e++) amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 64));
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    // The smallest power of two with amax / scale <= 240 (frexp, not log2f,
    // so exact powers of two land on the right side).
    float scale = 1.f;
    if (amax > 0.f) {
        int e;
        const float m = frexpf(amax / FP8W_MAX, &e);  // amax / 240 = m 2^e, m in [0.5, 1)
        scale = ldexpf(1.f, m == 0.5f ? e - 1 : e);
    }
    const float inv = 1.f / scale;  // exact: a power of two
    for (int i = threadIdx.x * 8; i < cols; i += blockDim.x * 8) {
        float v[8];
        unpack8(*reinterpret_cast<const uint4*>(wr + i), v);
        int p0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * inv, v[1] * inv, 0, false);
        p0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * inv, v[3] * inv, p0, true);
        int p1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] * inv, v[5] * inv, 0, false);
        p1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] * inv, v[7] * inv, p1, true);
        *reinterpret_cast<uint2*>(q + row * cols + i) = make_uint2((uint32_t)p0, (uint32_t)p1);
        float d[8];
        const pe_f2 a = __builtin_amdgcn_cvt_pk_f32_fp8(p0, false), b = __builtin_amdgcn_cvt_pk_f32_fp8(p0, true);
        const pe_f2 c = __builtin_amdgcn_cvt_pk_f32_fp8(p1, false), g = __builtin_amdgcn_cvt_pk_f32_fp8(p1, true);
        d[0] = a.x * scale, d[1] = a.y * scale, d[2] = b.x * scale, d[3] = b.y * scale;
        d[4] = c.x * scale, d[5] = c.y * scale, d[6] = g.x * scale, d[7] = g.y * scale;
        *reinterpret_cast<uint4*>(dense + row * cols + i) = pack8(d);
    }
    if (threadIdx.x == 0) scales[row] = scale;
}

// y[r][n] = scale[n] * (x[r] . W[n]) for r < R (1 to 4 rows), W in E4M3: as
// skinny_bf16 (amd/skinny.cuh), each 64-lane wave takes two rows over all of
// K, now 16 weights per 16-byte load, half the bytes of bf16.
__device__ __forceinline__ float dot16_fp8(const uint4& wv, const bf16* __restrict__ xp, float a) {
    const uint4 x0 = *reinterpret_cast<const uint4*>(xp), x1 = *reinterpret_cast<const uint4*>(xp + 8);
    const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const int ww[4] = {(int)wv.x, (int)wv.y, (int)wv.z, (int)wv.w};
#pragma unroll
    for (int e = 0; e < 4; e++) {
        const pe_f2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(ww[e], false), hi = __builtin_amdgcn_cvt_pk_f32_fp8(ww[e], true);
        a = fmaf(lo.x, __uint_as_float(xw[2 * e] << 16), a);
        a = fmaf(lo.y, __uint_as_float(xw[2 * e] & 0xffff0000u), a);
        a = fmaf(hi.x, __uint_as_float(xw[2 * e + 1] << 16), a);
        a = fmaf(hi.y, __uint_as_float(xw[2 * e + 1] & 0xffff0000u), a);
    }
    return a;
}
template <int R>
__device__ __forceinline__ void skinny_fp8(const uint8_t* __restrict__ W, const float* __restrict__ scales,
                                           const bf16* __restrict__ x, void* __restrict__ y, int K, int N, int ldy,
                                           int f32_out) {
    constexpr int YT = 2, UN = 4, STEP = 64 * 16;
    const int lane = threadIdx.x & 63, n0 = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * YT;
    if (n0 >= N) return;
    const uint8_t* w[YT];
#pragma unroll
    for (int i = 0; i < YT; i++) w[i] = W + (size_t)min(n0 + i, N - 1) * K;
    float acc[YT][R];
#pragma unroll
    for (int i = 0; i < YT; i++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[i][r] = 0.f;
    int k = 16 * lane;
    for (; k + (UN - 1) * STEP < K; k += UN * STEP) {
        uint4 wv[YT][UN];
#pragma unroll
        for (int i = 0; i < YT; i++)
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const pe_u4 v = __builtin_nontemporal_load(reinterpret_cast<const pe_u4*>(w[i] + k + u * STEP));
                wv[i][u] = make_uint4(v.x, v.y, v.z, v.w);
            }
#pragma unroll
        for (int u = 0; u < UN; u++)
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int i = 0; i < YT; i++) acc[i][r] = dot16_fp8(wv[i][u], x + (size_t)r * K + k + u * STEP, acc[i][r]);
    }
    for (; k < K; k += STEP) {
        uint4 wv[YT];
#pragma unroll
        for (int i = 0; i < YT; i++) wv[i] = *reinterpret_cast<const uint4*>(w[i] + k);
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int i = 0; i < YT; i++) acc[i][r] = dot16_fp8(wv[i], x + (size_t)r * K + k, acc[i][r]);
    }
#pragma unroll
    for (int i = 0; i < YT; i++) {
        const float s = scales[min(n0 + i, N - 1)];
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[i][r];
#pragma unroll
            for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
            v *= s;
            if (lane == 0 && n0 + i < N) {
                if (f32_out) ((float*)y)[(size_t)r * ldy + n0 + i] = v;
                else ((bf16*)y)[(size_t)r * ldy + n0 + i] = f2b(v);
            }
        }
    }
}
#define SKINNY_FP8(R)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                          \
        skinny_fp8_r##R(const uint8_t* W, const float* scales, const bf16* x, void* y, int K, int N, int ldy,  \
                        int f32_out) {                                                                          \
        skinny_fp8<R>(W, scales, x, y, K, N, ldy, f32_out);                                                    \
    }
SKINNY_FP8(1) SKINNY_FP8(2) SKINNY_FP8(3) SKINNY_FP8(4)
