// FP8 (E4M3) weights and activations for cuBLASLt's scaled matmuls.
//
// Two scale layouts, chosen by what the device's cuBLASLt runs:
//   mode 1, Block128: f32 scales, one per 128 x 128 weight tile and one per
//     128 consecutive activations of a row (Hopper; Qwen's FP8 checkpoints).
//   mode 2, Mx32 (MXFP8): power-of-two E8M0 scales, one per 32 consecutive
//     values of a row, stored in cuBLASLt's 128-row x 4-block tiles.
// Activations are quantized where they are produced (RMSNorm, SiLU, the
// gated delta rule's output norm, attention's output gate), from registers.


__device__ __forceinline__ uint8_t to_e4m3(float x) { return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3); }
__device__ __forceinline__ float from_e4m3(uint8_t v) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)));
}

// Eight E4M3 values to floats.
__device__ __forceinline__ void unpack8_e4m3(uint2 u, float* f) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 x = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3)));
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}

// Byte offset of the scale for (row r, 32-block kb) in MXFP8's tiled layout.
__device__ __forceinline__ size_t mx_index(int r, int kb, int KB) {
    const int kt = (KB + 3) / 4;
    return ((size_t)(r / 128) * kt + kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + (kb % 4);
}

// Smallest e with amax / 2^e <= 448 (the largest E4M3 value).
__device__ __forceinline__ int mx_exp(float amax) {
    if (!(amax > 0.f)) return -127;
    int ex;
    const float m = frexpf(amax / 448.f, &ex);
    if (m == 0.5f) ex -= 1;
    return max(-127, min(127, ex));
}

// Block128 activation scale index for (row t, 128-block kb): K-block major,
// `rows` scales per block.
__device__ __forceinline__ size_t act128_index(int t, int kb, int rows) {
    return (size_t)kb * rows + t;
}

// Where a kernel writes FP8 activations: values [rows][K] and their scales.
struct Fp8Out {
    uint8_t* q;
    void* s;
    int mode;  // 0 none, 1 Block128, 2 Mx32
    int rows;  // row capacity, for Block128 scale indexing
};

// Quantizes 8 consecutive values v of row t starting at column e (K wide).
// A scale group spans 16 lanes (Block128) or 4 lanes (Mx32) of 8 values each,
// consecutive within the warp; every lane of the warp must call this
// together, passing active = false for lanes without values.
__device__ __forceinline__ void quant8(const float* v, int t, int e, int K, const Fp8Out& o, bool active) {
    float a = 0.f;
    if (active) {
#pragma unroll
        for (int i = 0; i < 8; i++) a = fmaxf(a, fabsf(v[i]));
    }
    const int lanes = o.mode == 1 ? 16 : 4;
    for (int s = 1; s < lanes; s <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    if (!active) return;
    float inv;
    if (o.mode == 1) {
        const float sc = a > 0.f ? a / 448.f : 1.f;
        inv = 1.f / sc;
        if (e % 128 == 0) ((float*)o.s)[act128_index(t, e / 128, o.rows)] = sc;
    } else {
        const int ex = mx_exp(a);
        inv = ldexpf(1.f, -ex);
        if (e % 32 == 0) ((uint8_t*)o.s)[mx_index(t, e / 32, K / 32)] = (uint8_t)(127 + ex);
    }
    uint2 out;
    uint8_t* b = reinterpret_cast<uint8_t*>(&out);
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = to_e4m3(v[i] * inv);
    *reinterpret_cast<uint2*>(o.q + (size_t)t * K + e) = out;
}

// Weights from bf16, Mx32: one thread per (row, 32-block).
extern "C" __global__ void fp8_weights_mx(const bf16* __restrict__ w, int N, int K, uint8_t* __restrict__ q,
                                          uint8_t* __restrict__ s) {
    const int KB = K / 32;
    const long long total = (long long)N * KB;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += (long long)gridDim.x * blockDim.x) {
        const int n = (int)(g / KB), kb = (int)(g % KB);
        const bf16* src = w + (size_t)n * K + kb * 32;
        float a = 0.f;
        for (int i = 0; i < 32; i++) a = fmaxf(a, fabsf(b2f(src[i])));
        const int ex = mx_exp(a);
        const float inv = ldexpf(1.f, -ex);
        for (int i = 0; i < 32; i++) q[(size_t)n * K + kb * 32 + i] = to_e4m3(b2f(src[i]) * inv);
        s[mx_index(n, kb, KB)] = (uint8_t)(127 + ex);
    }
}

// Weights from bf16, Block128: one block of 256 threads per 128 x 128 tile.
// Scales are [N/128][K/128], row-major, as Qwen's checkpoints store them.
extern "C" __global__ void fp8_weights_block(const bf16* __restrict__ w, int N, int K, uint8_t* __restrict__ q,
                                             float* __restrict__ s) {
    __shared__ float sh[32];
    const int nb = blockIdx.y, kb = blockIdx.x, KB = gridDim.x;
    float a = 0.f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        const int r = nb * 128 + i / 128, c = kb * 128 + i % 128;
        if (r < N && c < K) a = fmaxf(a, fabsf(b2f(w[(size_t)r * K + c])));
    }
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = a;
    __syncthreads();
    a = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) a = fmaxf(a, sh[i]);
    const float sc = a > 0.f ? a / 448.f : 1.f, inv = 1.f / sc;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        const int r = nb * 128 + i / 128, c = kb * 128 + i % 128;
        if (r < N && c < K) q[(size_t)r * K + c] = to_e4m3(b2f(w[(size_t)r * K + c]) * inv);
    }
    if (threadIdx.x == 0) s[nb * KB + kb] = sc;
}

// Block-scaled E4M3 weights (Qwen's checkpoints: scales [N/128][K/128]) to
// MXFP8: each 32-block is rescaled to a power of two. One thread per (row, 32-block).
extern "C" __global__ void fp8_block_to_mx(const uint8_t* __restrict__ w, const float* __restrict__ sb, int N, int K,
                                           uint8_t* __restrict__ q, uint8_t* __restrict__ s) {
    const int KB = K / 32, KB128 = (K + 127) / 128;
    const long long total = (long long)N * KB;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += (long long)gridDim.x * blockDim.x) {
        const int n = (int)(g / KB), kb = (int)(g % KB);
        const float scale = sb[(n / 128) * KB128 + (kb * 32) / 128];
        float v[32], a = 0.f;
        for (int i = 0; i < 32; i++) {
            v[i] = from_e4m3(w[(size_t)n * K + kb * 32 + i]) * scale;
            a = fmaxf(a, fabsf(v[i]));
        }
        const int ex = mx_exp(a);
        const float inv = ldexpf(1.f, -ex);
        for (int i = 0; i < 32; i++) q[(size_t)n * K + kb * 32 + i] = to_e4m3(v[i] * inv);
        s[mx_index(n, kb, KB)] = (uint8_t)(127 + ex);
    }
}

// Activations from bf16 rows (for checks and inputs no kernel quantizes).
extern "C" __global__ void fp8_activations(const bf16* __restrict__ x, int rows, int K, uint8_t* fq, void* fs,
                                           int fmode, int frows) {
    const Fp8Out o{fq, fs, fmode, frows};
    const int per_row = K / 8;
    const long long total = (long long)rows * per_row;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long base = (long long)blockIdx.x * blockDim.x; base < total; base += stride) {
        const long long e = base + threadIdx.x;
        const bool active = e < total;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int t = 0, c = 0;
        if (active) {
            t = (int)(e / per_row);
            c = (int)(e % per_row) * 8;
            unpack8(*reinterpret_cast<const uint4*>(x + (size_t)t * K + c), v);
        }
        quant8(v, t, c, K, o, active);
    }
}
