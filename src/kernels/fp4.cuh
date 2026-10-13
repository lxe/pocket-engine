// FP4 weights (MXFP4: E2M1 with a power-of-two scale per 32) times MXFP8
// activations, on Blackwell's block-scaled tensor cores
// (mma.sync ... kind::mxf8f6f4.block_scale, sm_120a).
//
// Weights use quant.cuh's tiled order: chunks of 16 rows x 256 inputs, 8
// steps of 32, lane (g, t) holding inputs 32s + 8t .. 32s + 8t + 7 of rows g
// and g + 8 as one 32-bit word each. Within a word, input i < 4 sits at bits
// 8i and input i >= 4 at bits 8(i - 4) + 4, so two masks put four E2M1 codes
// in the middle of four bytes, where the instruction reads them. After the
// 2048 bytes of codes, each row has 8 UE8M0 scales, one per step (2176 bytes).
// Activations are the MXFP8 copies the norm and activation kernels write
// (fp8.cuh), row-major, with scales in cuBLASLt's tiled layout.
//
// NVFP4 checkpoints (E4M3 scale per 16, global f32 scale) are re-quantized
// to this layout at load: the scale per 32 is the smallest power of two that
// keeps the block's largest value within E2M1's 6.

#define FP4_CHUNK 2176

// 0, 0.5, 1, 1.5, 2, 3, 4, 6 for magnitudes 0..7, negated with bit 3.
__device__ __forceinline__ float e2m1_value(int c) {
    const int m = c & 7;
    const float v = m < 2 ? 0.5f * m : ldexpf(1.f + 0.5f * (m & 1), (m >> 1) - 1);
    return (c & 8) ? -v : v;
}
// Nearest E2M1 code, ties to the even code.
__device__ __forceinline__ int e2m1_code(float x) {
    const float a = fabsf(x);
    int m;
    if (a <= 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a < 3.5f) m = 5;
    else if (a <= 5.f) m = 6;
    else m = 7;
    return (x < 0.f && m != 0) ? (m | 8) : m;
}

// One row's 32 values as MXFP4: codes into the tile, the scale after them.
__device__ __forceinline__ void fp4_store32(const float* v, int row, int gs, int K, uint8_t* out) {
    const int kb = gs / 8, s = gs % 8, r = row & 15;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    int ex = -127;
    if (amax > 0.f) {
        int k;
        const float m = frexpf(amax / 6.f, &k);  // amax / 6 = m * 2^k, m in [0.5, 1)
        ex = max(-127, min(127, m == 0.5f ? k - 1 : k));  // smallest ex with amax / 2^ex <= 6
    }
    const float inv = ldexpf(1.f, -ex);
    uint8_t* c = out + ((size_t)(row / 16) * (K / 256) + kb) * FP4_CHUNK;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < 8; i++)
            word |= (uint32_t)e2m1_code(v[8 * t + i] * inv) << (i < 4 ? 8 * i : 8 * (i - 4) + 4);
        *reinterpret_cast<uint32_t*>(c + tile_word(r, s, t)) = word;
    }
    c[2048 + 8 * r + s] = (uint8_t)(127 + ex);
}

// NVFP4 [N][K/2] (element 2i in the low nibble of byte i), scales [N][K/16]
// as E4M3 and a global scale, to MXFP4 tiles; one thread per (row, 32 inputs).
extern "C" __global__ void nvfp4_to_tiles(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                          float global, int N, int K, uint8_t* __restrict__ out) {
    const int steps = K / 32;
    const long long n = (long long)N * steps;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int row = (int)(e / steps), gs = (int)(e % steps);
        float v[32];
#pragma unroll
        for (int i = 0; i < 32; i++) {
            const uint8_t b = w[(size_t)row * (K / 2) + gs * 16 + i / 2];
            const float sc = from_e4m3(ws[(size_t)row * (K / 16) + gs * 2 + i / 16]) * global;
            v[i] = e2m1_value((b >> (4 * (i & 1))) & 0xF) * sc;
        }
        fp4_store32(v, row, gs, K, out);
    }
}

// bf16 [N][K] to MXFP4 tiles (quantizing a bf16 checkpoint at load).
extern "C" __global__ void bf16_to_fp4_tiles(const bf16* __restrict__ w, int N, int K, uint8_t* __restrict__ out) {
    const int steps = K / 32;
    const long long n = (long long)N * steps;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int row = (int)(e / steps), gs = (int)(e % steps);
        float v[32];
#pragma unroll
        for (int i = 0; i < 32; i++) v[i] = b2f(w[(size_t)row * K + gs * 32 + i]);
        fp4_store32(v, row, gs, K, out);
    }
}

// MXFP8 rows (values and tiled scales, fp8.cuh) back to bf16, exactly (for checks).
extern "C" __global__ void mx8_dequant(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sc, int rows, int K,
                                       bf16* __restrict__ out) {
    const long long n = (long long)rows * K;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e / K), k = (int)(e % K);
        out[e] = f2b(from_e4m3(q[e]) * ldexpf(1.f, (int)sc[mx_index(r, k / 32, K / 32)] - 127));
    }
}

// Rows r0 .. r0 + rows of MXFP4 tiles as MXFP8 for cuBLASLt: values [rows][K]
// and scales in its tiled layout. E2M1 values are exact in E4M3 under the
// same power-of-two scale, so nothing is lost. One thread per 8 inputs.
extern "C" __global__ void fp4_to_mx8(const uint8_t* __restrict__ W, int r0, int rows, int cols,
                                      uint8_t* __restrict__ q, uint8_t* __restrict__ sx) {
    const int per_row = cols / 8;
    const long long n = (long long)rows * per_row;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e / per_row), k8 = (int)(e % per_row), row = r0 + r;
        const int kb = k8 / 32, s = (k8 / 4) % 8, t = k8 % 4;
        const uint8_t* c = W + ((size_t)(row / 16) * (cols / 256) + kb) * FP4_CHUNK;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(c + tile_word(row & 15, s, t));
        uint2 out;
        uint8_t* o = reinterpret_cast<uint8_t*>(&out);
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = to_e4m3(e2m1_value((word >> (i < 4 ? 8 * i : 8 * (i - 4) + 4)) & 0xF));
        *reinterpret_cast<uint2*>(q + (size_t)r * cols + 8 * k8) = out;
        if (t == 0 && (k8 & 3) == 0) sx[mx_index(r, k8 / 4, cols / 32)] = c[2048 + 8 * (row & 15) + s];
    }
}

// out[r - r0][:] = bf16(W[r][:]) from MXFP4 tiles (for checks).
extern "C" __global__ void dequant_fp4(const uint8_t* __restrict__ W, int r0, int rows, int cols,
                                       bf16* __restrict__ out) {
    const int per_row = cols / 8;
    const long long n = (long long)rows * per_row;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e / per_row), k8 = (int)(e % per_row), row = r0 + r;
        const int kb = k8 / 32, s = (k8 / 4) % 8, t = k8 % 4;
        const uint8_t* c = W + ((size_t)(row / 16) * (cols / 256) + kb) * FP4_CHUNK;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(c + tile_word(row & 15, s, t));
        const float sc = ldexpf(1.f, (int)c[2048 + 8 * (row & 15) + s] - 127);
        float v[8];
        for (int i = 0; i < 8; i++) v[i] = e2m1_value((word >> (i < 4 ? 8 * i : 8 * (i - 4) + 4)) & 0xF) * sc;
        *reinterpret_cast<uint4*>(out + (size_t)r * cols + 8 * k8) = pack8(v);
    }
}

__device__ __forceinline__ void mma_fp4_fp8(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                            uint32_t sfb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
          "h"(z));
#else
    __trap();
#endif
}

// Y[m][off + n] = X[m] . W[n] for MXFP4 tiles W and MXFP8 rows X (values Xq,
// row-major with ldq bytes per row; scales Xs). Each warp owns 16 features
// and MT rows over a slice of K (chunks of 256), like qtile; the lane's scale
// for A is that of row g + 8 * (lane & 1), and for B that of its token g.
template <int MT>
__device__ void qfp4(const uint8_t* __restrict__ W, int N, int K, const uint8_t* __restrict__ Xq, int ldq,
                     const uint8_t* __restrict__ Xs, int M, void* __restrict__ Y, int ldy, int off, int f32_out,
                     float* __restrict__ acc_ws, int ksplit) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp, n0 = tile * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * MT;
    const int KB = K / 256, per = (KB + ksplit - 1) / ksplit;
    const int b0 = blockIdx.z * per, b1 = min(KB, b0 + per);
    const uint8_t* base = W + (size_t)tile * KB * FP4_CHUNK;
    const int sfrow = g + 8 * (lane & 1);
    const uint8_t* xr[MT / 8];
    int tok[MT / 8];
    bool xok[MT / 8];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) {
        const int m = m0 + 8 * j + g;
        xok[j] = m < M;
        tok[j] = xok[j] ? m : 0;
        xr[j] = Xq + (size_t)tok[j] * ldq + 8 * t;
    }
    float acc[MT / 8][4];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    struct Chunk {
        uint4 q[4];
        uint2 sc;
    };
    // Scale bytes of block kb of token row m sit 4 to a word, words 512 apart.
    size_t sb_base[MT / 8];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) sb_base[j] = mx_index(tok[j], 0, K / 32);
    auto fetch = [&](int kb, Chunk& ch) {
        const uint8_t* c = base + (size_t)kb * FP4_CHUNK;
#pragma unroll
        for (int i = 0; i < 4; i++) ch.q[i] = ld_stream(c + 512 * i + 16 * lane);
        ch.sc = *reinterpret_cast<const uint2*>(c + 2048 + 8 * sfrow);
    };
    auto compute = [&](int kb, const Chunk& ch) {
        uint32_t sbw[MT / 8][2];
#pragma unroll
        for (int j = 0; j < MT / 8; j++) {
            sbw[j][0] = *reinterpret_cast<const uint32_t*>(Xs + sb_base[j] + (size_t)(2 * kb) * 512);
            sbw[j][1] = *reinterpret_cast<const uint32_t*>(Xs + sb_base[j] + (size_t)(2 * kb + 1) * 512);
        }
#pragma unroll
        for (int s = 0; s < 8; s++) {
            const uint32_t wa = word_of(ch.q[s >> 1], 2 * (s & 1)), wb = word_of(ch.q[s >> 1], 2 * (s & 1) + 1);
            const uint32_t a[4] = {(wa << 2) & 0x3C3C3C3Cu, (wb << 2) & 0x3C3C3C3Cu, (wa >> 2) & 0x3C3C3C3Cu,
                                   (wb >> 2) & 0x3C3C3C3Cu};
            const uint32_t sfa = ((s < 4 ? ch.sc.x : ch.sc.y) >> (8 * (s & 3))) & 0xFF;
#pragma unroll
            for (int j = 0; j < MT / 8; j++) {
                const uint2 xv = xok[j] ? *reinterpret_cast<const uint2*>(xr[j] + 256 * kb + 32 * s) : make_uint2(0, 0);
                const uint32_t sfb = (sbw[j][s >> 2] >> (8 * (s & 3))) & 0xFF;
                mma_fp4_fp8(acc[j], a, xv.x, xv.y, sfa, sfb);
            }
        }
    };
    Chunk c0, c1;
    if (b0 < b1) fetch(b0, c0);
    if (b0 + 1 < b1) fetch(b0 + 1, c1);
    for (int kb = b0; kb < b1; kb += 2) {
        const Chunk w0 = c0;
        if (kb + 2 < b1) fetch(kb + 2, c0);
        compute(kb, w0);
        if (kb + 1 < b1) {
            const Chunk w1 = c1;
            if (kb + 3 < b1) fetch(kb + 3, c1);
            compute(kb + 1, w1);
        }
    }
#pragma unroll
    for (int j = 0; j < MT / 8; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int n = n0 + g + (e >> 1) * 8, m = m0 + 8 * j + 2 * t + (e & 1);
            if (n < N && m < M) {
                if (ksplit > 1) acc_ws[((size_t)blockIdx.z * M + m) * N + n] = acc[j][e];
                else if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc[j][e];
                else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc[j][e]);
            }
        }
    }
}

#define QFP4(MT)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128) qfp4_##MT(                                       \
        const uint8_t* W, int N, int K, const uint8_t* Xq, int ldq, const uint8_t* Xs, int M, void* Y, \
        int ldy, int off, int f32_out, float* acc_ws, int ksplit) {                                    \
        qfp4<MT>(W, N, K, Xq, ldq, Xs, M, Y, ldy, off, f32_out, acc_ws, ksplit);                       \
    }
QFP4(8)
QFP4(16)
QFP4(32)
QFP4(64)
