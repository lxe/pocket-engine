// Quantized weights in ggml's block formats (GGUF). Each format gets one
// decoder for 8 consecutive weights of a row; dequantization, the
// matrix-vector product for small batches and embedding lookup all build on
// it. The math follows ggml's CUDA dequantization kernels.

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

// ggml type ids.
#define T_Q4_0 2
#define T_Q4_1 3
#define T_Q5_0 6
#define T_Q5_1 7
#define T_Q8_0 8
#define T_Q2_K 10
#define T_Q3_K 11
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ2_XXS 16
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ1_S 19
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_IQ1_M 29

__device__ __forceinline__ float hf(half x) { return __half2float(x); }
__device__ __forceinline__ float2 hf2(half2 x) { return __half22float2(x); }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}


// 8 bytes from p, which is aligned to A bytes, as two words (little-endian).
template <int A>
__device__ __forceinline__ uint2 ld8(const uint8_t* p) {
    if constexpr (A >= 8) {
        return *reinterpret_cast<const uint2*>(p);
    } else if constexpr (A >= 4) {
        const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
        return make_uint2(q[0], q[1]);
    } else {
        const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
        return make_uint2(q[0] | ((uint32_t)q[1] << 16), q[2] | ((uint32_t)q[3] << 16));
    }
}
__device__ __forceinline__ int byte_of(uint2 v, int i) { return ((i < 4 ? v.x : v.y) >> (8 * (i & 3))) & 0xFF; }

// Device layouts of repacked formats (see gguf.rs repack): the same fields,
// padded so weight bytes are 4- or 8-byte aligned.
struct block_q8_0p { half d; uint16_t pad; int8_t qs[32]; };
struct block_q4_0p { half d; uint16_t pad; uint8_t qs[16]; };
struct block_q5_0p { half d; uint16_t pad; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q6_Kp { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; uint8_t pad[6]; };
struct block_q3_Kp { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; uint16_t pad; };
static_assert(sizeof(block_q8_0p) == 36 && sizeof(block_q4_0p) == 20 && sizeof(block_q5_0p) == 24, "");
static_assert(sizeof(block_q6_Kp) == 216 && sizeof(block_q3_Kp) == 112, "");

// Decoder for format T: QK weights per block; dec8(b, c, out) writes weights
// 8c .. 8c + 7 of block b.
template <int T> struct Q;

template <> struct Q<T_Q4_0> {
    typedef block_q4_0p B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float d = hf(b->d);
        const uint2 q = ld8<4>(b->qs + (c & 1) * 8);
        const int sh = (c >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = (((byte_of(q, i) >> sh) & 0xF) - 8) * d;
    }
};
template <> struct Q<T_Q4_1> {
    typedef block_q4_1 B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float2 dm = hf2(b->dm);
        const uint2 q = ld8<4>(b->qs + (c & 1) * 8);
        const int sh = (c >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = ((byte_of(q, i) >> sh) & 0xF) * dm.x + dm.y;
    }
};
template <> struct Q<T_Q5_0> {
    typedef block_q5_0p B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float d = hf(b->d);
        const uint32_t qh = *reinterpret_cast<const uint32_t*>(b->qh);
        const uint2 q = ld8<8>(b->qs + (c & 1) * 8);
        const int sh = (c >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = 8 * c + i;
            o[i] = ((((byte_of(q, i) >> sh) & 0xF) | (((qh >> e) & 1) << 4)) - 16) * d;
        }
    }
};
template <> struct Q<T_Q5_1> {
    typedef block_q5_1 B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float2 dm = hf2(b->dm);
        uint32_t qh;
        memcpy(&qh, b->qh, 4);
        const uint8_t* q = b->qs + (c & 1) * 8;
        const int sh = (c >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = 8 * c + i;
            o[i] = (((q[i] >> sh) & 0xF) | (((qh >> e) & 1) << 4)) * dm.x + dm.y;
        }
    }
};
template <> struct Q<T_Q8_0> {
    typedef block_q8_0p B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float d = hf(b->d);
        const uint2 q = ld8<4>(reinterpret_cast<const uint8_t*>(b->qs) + 8 * c);
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = (int8_t)byte_of(q, i) * d;
    }
};
template <> struct Q<T_IQ4_NL> {
    typedef block_q4_0p B;
    static constexpr int QK = 32;
    __device__ static void dec8(const B* b, int c, float* o) {
        const float d = hf(b->d);
        const uint2 q = ld8<4>(b->qs + (c & 1) * 8);
        const int sh = (c >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = d * kvalues_iq4nl[(byte_of(q, i) >> sh) & 0xF];
    }
};
template <> struct Q<T_Q2_K> {
    typedef block_q2_K B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, n = e / 128, r = e % 128, j = r / 32, l = r % 32;
        const uint8_t sc = b->scales[8 * n + 2 * j + l / 16];
        const float2 dm = hf2(b->dm);
        const float dl = dm.x * (sc & 0xF), ml = dm.y * (sc >> 4);
        const uint2 q = ld8<4>(b->qs + 32 * n + l);
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = dl * ((byte_of(q, i) >> (2 * j)) & 3) - ml;
    }
};
template <> struct Q<T_Q3_K> {
    typedef block_q3_Kp B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, n = e / 128, r = e % 128, j = r / 32, l = r % 32;
        const int is = 8 * n + 2 * j + l / 16;
        const uint8_t* s = b->scales;
        const int8_t us = is < 4    ? (s[is] & 0xF) | (((s[is + 8] >> 0) & 3) << 4)
                          : is < 8  ? (s[is] & 0xF) | (((s[is + 4] >> 2) & 3) << 4)
                          : is < 12 ? (s[is - 8] >> 4) | (((s[is] >> 4) & 3) << 4)
                                    : (s[is - 8] >> 4) | (((s[is - 4] >> 6) & 3) << 4);
        const float dl = hf(b->d) * (us - 32);
        const int m = 1 << (4 * n + j);
        const uint2 q = ld8<8>(b->qs + 32 * n + l);
        const uint2 hm = ld8<8>(b->hmask + l);
#pragma unroll
        for (int i = 0; i < 8; i++)
            o[i] = dl * ((int8_t)((byte_of(q, i) >> (2 * j)) & 3) - ((byte_of(hm, i) & m) ? 0 : 4));
    }
};
template <> struct Q<T_Q4_K> {
    typedef block_q4_K B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, il = e / 64, r = e % 64, hi = r >= 32, l = r % 32;
        int sc, m;
        scale_min_k4(2 * il + hi, b->scales, sc, m);
        const float2 dm = hf2(b->dm);
        const float d = dm.x * sc, mn = dm.y * m;
        const uint2 q = ld8<8>(b->qs + 32 * il + l);
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = d * ((byte_of(q, i) >> (4 * hi)) & 0xF) - mn;
    }
};
template <> struct Q<T_Q5_K> {
    typedef block_q5_K B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, il = e / 64, r = e % 64, hi = r >= 32, l = r % 32;
        int sc, m;
        scale_min_k4(2 * il + hi, b->scales, sc, m);
        const float2 dm = hf2(b->dm);
        const float d = dm.x * sc, mn = dm.y * m;
        const uint2 q = ld8<8>(b->qs + 32 * il + l);
        const uint2 qh = ld8<8>(b->qh + l);
        const int hb = 2 * il + hi;
#pragma unroll
        for (int i = 0; i < 8; i++)
            o[i] = d * (((byte_of(q, i) >> (4 * hi)) & 0xF) + (((byte_of(qh, i) >> hb) & 1) << 4)) - mn;
    }
};
template <> struct Q<T_Q6_K> {
    typedef block_q6_Kp B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, ip = e / 128, r = e % 128, g = r / 32, l = r % 32;
        const float d = hf(b->d) * b->scales[8 * ip + l / 16 + 2 * g];
        const uint2 ql = ld8<8>(b->ql + 64 * ip + 32 * (g & 1) + l);
        const uint2 qh = ld8<8>(b->qh + 32 * ip + l);
        const int lsh = 4 * (g >> 1), hsh = 2 * g;
#pragma unroll
        for (int i = 0; i < 8; i++)
            o[i] = d * ((int8_t)(((byte_of(ql, i) >> lsh) & 0xF) | (((byte_of(qh, i) >> hsh) & 3) << 4)) - 32);
    }
};
template <> struct Q<T_IQ4_XS> {
    typedef block_iq4_xs B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int e = 8 * c, ib = e / 32, r = e % 32;
        const int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((b->scales_h >> 2 * ib) & 3) << 4);
        const float d = hf(b->d) * (ls - 32);
        const uint2 q = ld8<8>(b->qs + 16 * ib + (r % 16));
        const int sh = 4 * (r / 16);
#pragma unroll
        for (int i = 0; i < 8; i++) o[i] = d * kvalues_iq4nl[(byte_of(q, i) >> sh) & 0xF];
    }
};
template <> struct Q<T_IQ2_XXS> {
    typedef block_iq2_xxs B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint16_t* q2 = b->qs + 4 * ib;
        const uint8_t* aux8 = (const uint8_t*)q2;
        const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = hf(b->d) * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
};
template <> struct Q<T_IQ2_XS> {
    typedef block_iq2_xs B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint16_t* q2 = b->qs + 4 * ib;
        const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[il] & 511));
        const float d = hf(b->d) * (0.5f + ((b->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
};
template <> struct Q<T_IQ2_S> {
    typedef block_iq2_s B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint8_t* grid = (const uint8_t*)(iq2s_grid + (b->qs[4 * ib + il] | ((b->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float d = hf(b->d) * (0.5f + ((b->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = b->qs[QK_K / 8 + 4 * ib + il];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
};
template <> struct Q<T_IQ3_XXS> {
    typedef block_iq3_xxs B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint8_t* q3 = b->qs + 8 * ib;
        const uint16_t* gas = (const uint16_t*)(b->qs + QK_K / 4) + 2 * ib;
        const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = hf(b->d) * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            o[j] = d * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            o[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
    }
};
template <> struct Q<T_IQ3_S> {
    typedef block_iq3_s B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint8_t* qs = b->qs + 8 * ib;
        const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 0] | ((b->qh[ib] << (8 - 2 * il)) & 256)));
        const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 1] | ((b->qh[ib] << (7 - 2 * il)) & 256)));
        const float d = hf(b->d) * (1 + 2 * ((b->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
        const uint8_t signs = b->signs[4 * ib + il];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            o[j] = d * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            o[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
    }
};
template <> struct Q<T_IQ1_S> {
    typedef block_iq1_s B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const float delta = b->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
        const float d = hf(b->d) * (2 * ((b->qh[ib] >> 12) & 7) + 1);
        uint32_t grid32[2];
        const int8_t* q = (const int8_t*)grid32;
        grid32[0] = iq1s_grid_gpu[b->qs[4 * ib + il] | (((b->qh[ib] >> 3 * il) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = d * (q[j] + delta);
    }
};
template <> struct Q<T_IQ1_M> {
    typedef block_iq1_m B;
    static constexpr int QK = 256;
    __device__ static void dec8(const B* b, int c, float* o) {
        const int ib = c / 4, il = c % 4;
        const uint16_t* sc = (const uint16_t*)b->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const int ib16 = 2 * ib + il / 2;
        const float d = hf(scale.f16) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
        const float delta = b->qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t grid32[2];
        const int8_t* q = (const int8_t*)grid32;
        grid32[0] = iq1s_grid_gpu[b->qs[4 * ib + il] | (((b->qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = d * (q[j] + delta);
    }
};

// Weights 8k .. 8k + 7 of a quantized row starting at `row`.
template <int T>
__device__ __forceinline__ void qdec8(const uint8_t* row, int k8, float* o) {
    typedef typename Q<T>::B B;
    constexpr int per = Q<T>::QK / 8;
    Q<T>::dec8(reinterpret_cast<const B*>(row) + k8 / per, k8 % per, o);
}

template <int T>
__device__ __forceinline__ size_t row_bytes(int cols) {
    return (size_t)(cols / Q<T>::QK) * sizeof(typename Q<T>::B);
}

// out[r - r0][:] = bf16(W[r][:]) for rows r0 .. r0 + rows.
template <int T>
__device__ void qdequant(const uint8_t* __restrict__ W, int r0, int rows, int cols, bf16* __restrict__ out) {
    const int per_row = cols / 8;
    const long long n = (long long)rows * per_row;
    const size_t rb = row_bytes<T>(cols);
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e / per_row), k8 = (int)(e % per_row);
        float v[8];
        qdec8<T>(W + (size_t)(r0 + r) * rb, k8, v);
        *reinterpret_cast<uint4*>(out + (size_t)r * cols + 8 * k8) = pack8(v);
    }
}

// y[r][off + n] = W[n] . x[r] for the R rows of x, one warp per output row n.
// Memory-bound: each weight is read once and decoded in registers.
template <int T, int R>
__device__ void qgemv(const uint8_t* __restrict__ W, int N, int K, const bf16* __restrict__ x, int ldx,
                      void* __restrict__ y, int ldy, int off, int f32_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    if (n >= N) return;
    const uint8_t* row = W + (size_t)n * row_bytes<T>(K);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int k8 = lane; k8 < K / 8; k8 += 32) {
        float w[8];
        qdec8<T>(row, k8, w);
#pragma unroll
        for (int r = 0; r < R; r++) {
            float xv[8];
            unpack8(*reinterpret_cast<const uint4*>(x + (size_t)r * ldx + 8 * k8), xv);
#pragma unroll
            for (int i = 0; i < 8; i++) acc[r] += w[i] * xv[i];
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const float s = warp_sum(acc[r]);
        if (lane == 0) {
            if (f32_out) ((float*)y)[(size_t)r * ldy + off + n] = s;
            else ((bf16*)y)[(size_t)r * ldy + off + n] = f2b(s);
        }
    }
}

// Y[m][off + n] = X[m] . W[n] for many rows of X, on tensor cores.
//
// Each warp owns 16 output features and MT rows of X over a slice of K, and
// works alone: no shared memory, no block barriers. The trick is the order of
// K inside each 32-wide step. A dot product does not care how K is ordered,
// so the step's 32 columns are assigned to mma.sync m16n8k16 fragment slots
// such that lane (g, t) = (lane / 4, lane % 4) needs exactly the 8
// consecutive weights 8t .. 8t + 7 of rows g and g + 8 (one decode each), and
// the 8 consecutive inputs 8t .. 8t + 7 of each X row it holds (one 16-byte
// load). Weights are read once, still quantized, and decoded in registers.
// Small grids split K across blocks, accumulating in an f32 workspace.
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int T, int MT>
__device__ void qmma(const uint8_t* __restrict__ W, int N, int K, const bf16* __restrict__ X, int ldx, int M,
                     void* __restrict__ Y, int ldy, int off, int f32_out, float* __restrict__ acc_ws, int ksplit) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * MT;
    const size_t rb = row_bytes<T>(K);
    const uint8_t* ra = W + (size_t)min(n0 + g, N - 1) * rb;
    const uint8_t* rbp = W + (size_t)min(n0 + g + 8, N - 1) * rb;
    const int steps = K / 32, per = (steps + ksplit - 1) / ksplit;
    const int s0 = blockIdx.z * per, s1 = min(steps, s0 + per);
    // X rows this lane feeds: token 8j + g of the tile.
    const bf16* xr[MT / 8];
    bool xok[MT / 8];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) {
        const int m = m0 + 8 * j + g;
        xok[j] = m < M;
        xr[j] = X + (size_t)(xok[j] ? m : 0) * ldx + 8 * t;
    }
    float acc[MT / 8][4];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll 4
    for (int s = s0; s < s1; s++) {
        float wa[8], wb[8];
        qdec8<T>(ra, 4 * s + t, wa);
        qdec8<T>(rbp, 4 * s + t, wb);
        uint4 xv[MT / 8];
#pragma unroll
        for (int j = 0; j < MT / 8; j++)
            xv[j] = xok[j] ? *reinterpret_cast<const uint4*>(xr[j] + 32 * s) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t a[4] = {bf2(wa[4 * h], wa[4 * h + 1]), bf2(wb[4 * h], wb[4 * h + 1]),
                                   bf2(wa[4 * h + 2], wa[4 * h + 3]), bf2(wb[4 * h + 2], wb[4 * h + 3])};
#pragma unroll
            for (int j = 0; j < MT / 8; j++)
                mma_bf16(acc[j], a, h ? xv[j].z : xv[j].x, h ? xv[j].w : xv[j].y);
        }
    }
    // acc[j][0..1]: feature g, tokens 8j + 2t + {0, 1}; acc[j][2..3]: feature g + 8.
#pragma unroll
    for (int j = 0; j < MT / 8; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int n = n0 + g + (e >> 1) * 8, m = m0 + 8 * j + 2 * t + (e & 1);
            if (n < N && m < M) {
                if (ksplit > 1) atomicAdd(acc_ws + (size_t)m * N + n, acc[j][e]);
                else if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc[j][e];
                else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc[j][e]);
            }
        }
    }
}

// Writes split-K sums from the f32 workspace into the output.
extern "C" __global__ void qmma_finish(const float* __restrict__ acc_ws, int M, int N, void* __restrict__ Y,
                                       int ldy, int off, int f32_out) {
    const long long total = (long long)M * N;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const int m = (int)(e / N), n = (int)(e % N);
        if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc_ws[e];
        else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc_ws[e]);
    }
}

// Embedding lookup from a quantized table, or the image pool (see embed).
template <int T>
__device__ void qembed(const int* __restrict__ ids, const uint8_t* __restrict__ table,
                       const int* __restrict__ emb_src, const bf16* __restrict__ pool,
                       bf16* __restrict__ out, int hidden) {
    const int t = blockIdx.x, e = emb_src[t];
    for (int k8 = threadIdx.x; k8 < hidden / 8; k8 += blockDim.x) {
        uint4 v;
        if (e >= 0) {
            v = *reinterpret_cast<const uint4*>(pool + (size_t)e * hidden + 8 * k8);
        } else {
            float w[8];
            qdec8<T>(table + (size_t)ids[t] * row_bytes<T>(hidden), k8, w);
            v = pack8(w);
        }
        *reinterpret_cast<uint4*>(out + (size_t)t * hidden + 8 * k8) = v;
    }
}

#define QUANT_KERNELS(T, name)                                                                         \
    extern "C" __global__ void dequant_##name(const uint8_t* W, int r0, int rows, int cols, bf16* out) { \
        qdequant<T>(W, r0, rows, cols, out);                                                           \
    }                                                                                                  \
    extern "C" __global__ void embed_##name(const int* ids, const uint8_t* table, const int* emb_src,  \
                                            const bf16* pool, bf16* out, int hidden) {                 \
        qembed<T>(ids, table, emb_src, pool, out, hidden);                                             \
    }                                                                                                  \
    QMMA(T, name, 8) QMMA(T, name, 16) QMMA(T, name, 32) QMMA(T, name, 64)                                              \
    QGEMV(T, name, 1) QGEMV(T, name, 2) QGEMV(T, name, 3) QGEMV(T, name, 4)                            \
    QGEMV(T, name, 5) QGEMV(T, name, 6) QGEMV(T, name, 7) QGEMV(T, name, 8)

#define QMMA(T, name, MT)                                                                              \
    extern "C" __global__ void __launch_bounds__(128) qmma_##name##_##MT(                               \
        const uint8_t* W, int N, int K, const bf16* X, int ldx, int M, void* Y, int ldy, int off,       \
        int f32_out, float* acc_ws, int ksplit) {                                                      \
        qmma<T, MT>(W, N, K, X, ldx, M, Y, ldy, off, f32_out, acc_ws, ksplit);                          \
    }

#define QGEMV(T, name, R)                                                                              \
    extern "C" __global__ void __launch_bounds__(128) qgemv_##name##_##R(                               \
        const uint8_t* W, int N, int K, const bf16* x, int ldx, void* y, int ldy, int off, int f32_out) { \
        qgemv<T, R>(W, N, K, x, ldx, y, ldy, off, f32_out);                                            \
    }

QUANT_KERNELS(T_Q4_0, q4_0)
QUANT_KERNELS(T_Q4_1, q4_1)
QUANT_KERNELS(T_Q5_0, q5_0)
QUANT_KERNELS(T_Q5_1, q5_1)
QUANT_KERNELS(T_Q8_0, q8_0)
QUANT_KERNELS(T_Q2_K, q2_k)
QUANT_KERNELS(T_Q3_K, q3_k)
QUANT_KERNELS(T_Q4_K, q4_k)
QUANT_KERNELS(T_Q5_K, q5_k)
QUANT_KERNELS(T_Q6_K, q6_k)
QUANT_KERNELS(T_IQ2_XXS, iq2_xxs)
QUANT_KERNELS(T_IQ2_XS, iq2_xs)
QUANT_KERNELS(T_IQ2_S, iq2_s)
QUANT_KERNELS(T_IQ3_XXS, iq3_xxs)
QUANT_KERNELS(T_IQ3_S, iq3_s)
QUANT_KERNELS(T_IQ1_S, iq1_s)
QUANT_KERNELS(T_IQ1_M, iq1_m)
QUANT_KERNELS(T_IQ4_NL, iq4_nl)
QUANT_KERNELS(T_IQ4_XS, iq4_xs)
