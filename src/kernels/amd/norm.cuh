// RMSNorm on AMD: see rmsnorm in qwen35.cu.
#pragma once

// Sum across the block, one wave64 reduction and one barrier (block_sum
// reduces 32-lane halves and takes two): every thread receives the total.
// For a single use per kernel; `sh` holds one float per wave.
__device__ __forceinline__ float block_sum_once(float v, float* sh) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 6); w++) t += sh[w];
    return t;
}

// rmsnorm's math with every load issued before any is used. Branching around
// loads (the row's end, `add`) made the compiler wait for each in turn, eight
// dependent memory round trips; lanes past the row load its first elements
// instead and ignore them. 4.2 to 3.4 us a row in a graph on an MI325X.
__device__ __forceinline__ void rmsnorm_amd(const bf16* __restrict__ x, bf16* __restrict__ res,
                                            const float* __restrict__ w, bf16* __restrict__ out,
                                            const int* __restrict__ gather, int hidden, float eps, int add,
                                            int write_res, uint8_t* fq, void* fs, int fmode, int frows) {
    __shared__ float sh[16];
    const int r = blockIdx.x;
    const size_t src = gather ? (size_t)gather[r] : (size_t)r;
    uint4 rv[4], xv[4];
    float4 wv[4][2];
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8, j = i < hidden ? i : 0;
        rv[c] = *reinterpret_cast<const uint4*>(res + src * hidden + j);
        xv[c] = add ? *reinterpret_cast<const uint4*>(x + src * hidden + j) : make_uint4(0, 0, 0, 0);
        wv[c][0] = *reinterpret_cast<const float4*>(w + j);
        wv[c][1] = *reinterpret_cast<const float4*>(w + j + 4);
    }
    float v[4][8];
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        float a[8];
        unpack8(rv[c], v[c]);
        unpack8(xv[c], a);
#pragma unroll
        for (int e = 0; e < 8; e++) v[c][e] += a[e];
        if (i < hidden) {
            if (add && write_res) *reinterpret_cast<uint4*>(res + src * hidden + i) = pack8(v[c]);
#pragma unroll
            for (int e = 0; e < 8; e++) ss += v[c][e] * v[c][e];
        }
    }
    ss = block_sum_once(ss, sh);
    const float scale = 1.f / sqrtf(ss / hidden + eps);
    const Fp8Out fo{fq, fs, fmode, frows};
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int i = (c * blockDim.x + threadIdx.x) * 8;
        if (c * blockDim.x * 8 >= (unsigned)hidden) break;
        float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (i < hidden) {
            const float* wc = reinterpret_cast<const float*>(wv[c]);
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = v[c][e] * scale * wc[e];
            *reinterpret_cast<uint4*>(out + (size_t)r * hidden + i) = pack8(o);
        }
        if (fmode) quant8(o, r, i, hidden, fo, i < hidden);
    }
}
