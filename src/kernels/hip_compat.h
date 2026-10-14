// Builds the CUDA kernel sources with hipcc for AMD Instinct GPUs (gfx942).
// build.rs force-includes this header; on NVIDIA it is never seen.
//
// The kernels are written for 32-lane warps. A gfx942 wavefront has 64
// lanes, so a "warp" here is half a wavefront: lane = threadIdx.x & 31 as
// before, and every warp-level primitive below works within its 32-lane half
// (shuffles with width 32, ballots shifted to the half).
//
// Tensor-core fragments (mma.sync, ldmatrix) keep NVIDIA's per-lane layout
// and are emulated with shuffles inside the 32-lane warp: correct and
// portable, not fast. Kernels that matter get native MFMA versions.
#pragma once

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <stdint.h>

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
// CUDA's spellings of HIP's round-to-nearest-even conversions.
__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __float22bfloat162_rn(make_float2(a, b));
}

// FP8 E4M3 in the OCP encoding NVIDIA uses (bytes in the KV cache and FP8
// checkpoints keep their meaning); gfx942 converts it in software.
typedef __hip_fp8x2_storage_t __nv_fp8x2_storage_t;
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2

// TMA descriptors appear in Hopper kernels' signatures only.
struct alignas(64) CUtensorMap {
    unsigned long long opaque[16];
};
#define __trap() __builtin_trap()
#ifndef __grid_constant__
#define __grid_constant__
#endif

// Warp primitives on the 32-lane half of the wavefront. Masks are ignored:
// every caller passes a full warp.
template <class T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned, T v, int m, int w = 32) {
    return __shfl_xor(v, m, w);
}
template <class T>
__device__ __forceinline__ T __shfl_sync(unsigned, T v, int src, int w = 32) {
    return __shfl(v, src, w);
}
template <class T>
__device__ __forceinline__ T __shfl_up_sync(unsigned, T v, unsigned d, int w = 32) {
    return __shfl_up(v, d, w);
}
template <class T>
__device__ __forceinline__ T __shfl_down_sync(unsigned, T v, unsigned d, int w = 32) {
    return __shfl_down(v, d, w);
}
__device__ __forceinline__ unsigned __ballot_sync(unsigned, int p) {
    return (unsigned)(__ballot(p) >> (threadIdx.x & 32));
}
__device__ __forceinline__ bool __any_sync(unsigned m, int p) { return __ballot_sync(m, p) != 0; }
__device__ __forceinline__ bool __all_sync(unsigned m, int p) { return __ballot_sync(m, !p) == 0; }
__device__ __forceinline__ void __syncwarp(unsigned = 0xffffffffu) { __builtin_amdgcn_wave_barrier(); }

namespace pe {

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
#if __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
#else
#pragma unroll
    for (int i = 0; i < 4; i++) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return c;
#endif
}

// mma.sync m16n8k16 bf16 x bf16 + f32 (row.col), NVIDIA's fragments:
//   a0 = A[g][2t, 2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..]
//   b0 = B[2t, 2t+1][g], b1 = B[2t+8, 2t+9][g]
//   d0, d1 = C[g][2t, 2t+1], d2, d3 = C[g+8][2t, 2t+1]
// with g = lane / 4, t = lane % 4.
__device__ __forceinline__ void mma_16816_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int u = 0; u < 4; u++) {
        const int la = 4 * g + u, l0 = 8 * t + u, l1 = 8 * t + 4 + u;
        const uint32_t A0 = __shfl(a[0], la, 32), A1 = __shfl(a[1], la, 32);
        const uint32_t A2 = __shfl(a[2], la, 32), A3 = __shfl(a[3], la, 32);
        const uint32_t B00 = __shfl(b0, l0, 32), B01 = __shfl(b1, l0, 32);
        const uint32_t B10 = __shfl(b0, l1, 32), B11 = __shfl(b1, l1, 32);
        d[0] += bf_lo(A0) * bf_lo(B00) + bf_hi(A0) * bf_hi(B00) + bf_lo(A2) * bf_lo(B01) + bf_hi(A2) * bf_hi(B01);
        d[1] += bf_lo(A0) * bf_lo(B10) + bf_hi(A0) * bf_hi(B10) + bf_lo(A2) * bf_lo(B11) + bf_hi(A2) * bf_hi(B11);
        d[2] += bf_lo(A1) * bf_lo(B00) + bf_hi(A1) * bf_hi(B00) + bf_lo(A3) * bf_lo(B01) + bf_hi(A3) * bf_hi(B01);
        d[3] += bf_lo(A1) * bf_lo(B10) + bf_hi(A1) * bf_hi(B10) + bf_lo(A3) * bf_lo(B11) + bf_hi(A3) * bf_hi(B11);
    }
}

// mma.sync m16n8k32 s8 x s8 -> s32 with zero accumulators: as above with
// four bytes per register, a0 = A[g][4t..4t+3], a2 = A[g][16+4t..], b0 =
// B[4t..4t+3][g], b1 = B[16+4t..][g].
__device__ __forceinline__ void mma_16832_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    c[0] = c[1] = c[2] = c[3] = 0;
#pragma unroll
    for (int u = 0; u < 4; u++) {
        const int la = 4 * g + u, l0 = 8 * t + u, l1 = 8 * t + 4 + u;
        const uint32_t A0 = __shfl(a[0], la, 32), A1 = __shfl(a[1], la, 32);
        const uint32_t A2 = __shfl(a[2], la, 32), A3 = __shfl(a[3], la, 32);
        const uint32_t B00 = __shfl(b0, l0, 32), B01 = __shfl(b1, l0, 32);
        const uint32_t B10 = __shfl(b0, l1, 32), B11 = __shfl(b1, l1, 32);
        c[0] = dot4(A2, B01, dot4(A0, B00, c[0]));
        c[1] = dot4(A2, B11, dot4(A0, B10, c[1]));
        c[2] = dot4(A3, B01, dot4(A1, B00, c[2]));
        c[3] = dot4(A3, B11, dot4(A1, B10, c[3]));
    }
}

// ldmatrix.x4 (b16): lanes 8i..8i+7 name the rows of matrix i; lane l gets
// from matrix i the pair at row l / 4, columns 2 (l % 4). With .trans, the
// pair at rows 2 (l % 4) and 2 (l % 4) + 1, column l / 4.
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p, bool trans) {
    const int lane = threadIdx.x & 31;
    const unsigned long long addr = (unsigned long long)p;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (!trans) {
            const unsigned long long row = __shfl(addr, 8 * i + (lane >> 2), 32);
            r[i] = *reinterpret_cast<const uint32_t*>(row + 4 * (lane & 3));
        } else {
            const unsigned long long r0 = __shfl(addr, 8 * i + 2 * (lane & 3), 32);
            const unsigned long long r1 = __shfl(addr, 8 * i + 2 * (lane & 3) + 1, 32);
            const uint32_t lo = *reinterpret_cast<const uint16_t*>(r0 + 2 * (lane >> 2));
            const uint32_t hi = *reinterpret_cast<const uint16_t*>(r1 + 2 * (lane >> 2));
            r[i] = lo | (hi << 16);
        }
    }
}

}  // namespace pe
