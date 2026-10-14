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
// Small grids split K across blocks; each slice writes its partial sums to an
// f32 workspace and qmma_finish adds them in a fixed order.
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
                if (ksplit > 1) acc_ws[((size_t)blockIdx.z * M + m) * N + n] = acc[j][e];
                else if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc[j][e];
                else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc[j][e]);
            }
        }
    }
}

// Sums the split-K slices' partial results in slice order (so results do
// not depend on scheduling) and writes them to the output.
extern "C" __global__ void qmma_finish(const float* __restrict__ acc_ws, int M, int N, int ksplit,
                                       void* __restrict__ Y, int ldy, int off, int f32_out) {
    const long long total = (long long)M * N;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        float v = 0.f;
        for (int z = 0; z < ksplit; z++) v += acc_ws[z * total + e];
        const int m = (int)(e / N), n = (int)(e % N);
        if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = v;
        else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(v);
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

// Rewrites ggml blocks of `bs` bytes as device blocks of `dbs`: the first
// `head` bytes stay, then dbs - bs zero bytes, then the rest of the block.
extern "C" __global__ void repack_blocks(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                         int blocks, int bs, int dbs, int head) {
    const int pad = dbs - bs;
    const long long n = (long long)blocks * dbs;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const long long b = i / dbs;
        const int j = (int)(i - b * dbs);
        uint8_t v = 0;
        if (j < head) v = src[b * bs + j];
        else if (j >= head + pad) v = src[b * bs + j - pad];
        dst[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Tiled 4-bit layout (Q4_0, Q4_K), rewritten at load for the tensor-core path.
//
// A matrix becomes chunks of 16 output rows x 256 inputs, row tiles outer.
// Within a chunk, 8 steps of 32 inputs; at step s, lane (g, t) of a warp
// needs inputs 32s + 8t .. 32s + 8t + 7 of rows g and g + 8 (the K order of
// qmma), one 32-bit word of nibbles each. Each 512-byte block holds two steps,
// 16 bytes per lane: [step 2i, row g | step 2i, row g + 8 | step 2i + 1, row g
// | step 2i + 1, row g + 8], so a warp reads it with one coalesced 16-byte
// load per lane. In a word, input 2p sits at bits 4p and input 2p + 1 at bits
// 16 + 4p, so pairs come out with one mask.
//   Q4_0 (2304 bytes): 2048 of nibbles, then per row 8 halves, d per step.
//   Q4_K (2368 bytes): 2048 of nibbles, then per row 16 bytes (the 8 steps'
//   6-bit scales, then their mins), then per row (d, dmin) as two halves.
//   IQ4_NL (2304 bytes): Q4_0's layout; weights are d * kvalues_iq4nl[q].
//   IQ4_XS (2368 bytes): 2048 of nibbles, then per row 16 bytes (the 8
//   steps' scales less 32, as int8), then per row d as a half and 2 spare.
//   Q5_K (2880 bytes): 2048 of low nibbles; 512 of high bits, 16 bytes per
//   lane, a 16-bit half per step with row g in its low byte and row g + 8 in
//   its high byte, inputs in the order 0 4 1 5 2 6 3 7; then Q4_K's scales,
//   mins and (d, dmin).
//   Q6_K (3392 bytes): 2048 of low nibbles; 1024 of high bits, 16 bytes per
//   lane per four steps, a word per step with row g in its low half and row
//   g + 8 in its high half, two bits per input; then per row the 16 int8
//   scales (one per 16 inputs) and d.
// Weights are computed in f32 exactly as the decoders compute them, so tiled
// and plain dequantization agree bit for bit.
template <int T> struct Tiled;
template <> struct Tiled<T_Q4_0> { static constexpr int CHUNK = 2304; };
template <> struct Tiled<T_Q4_K> { static constexpr int CHUNK = 2368; };
template <> struct Tiled<T_Q5_K> { static constexpr int CHUNK = 2880; };
template <> struct Tiled<T_IQ4_NL> { static constexpr int CHUNK = 2304; };
template <> struct Tiled<T_IQ4_XS> { static constexpr int CHUNK = 2368; };
template <int T> constexpr bool IQ4 = T == T_IQ4_NL || T == T_IQ4_XS;

// kvalues_iq4nl[q] for each byte q (0 to 15) of n, as signed bytes: two
// 8-entry lookups by the low 3 bits, then a pick by bit 3. Byte selectors
// are nibbles, so the four indices are packed first.
__device__ __forceinline__ uint32_t iq4_bytes(uint32_t n) {
    const uint32_t sel = __byte_perm(n | (n >> 4), 0, 0x0020);
    // The table, four bytes to a word: -127 -104 -83 -65 | -49 -35 -22 -10 | 1 13 25 38 | 53 69 89 113.
    const uint32_t lo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel & 0x7777u);
    const uint32_t hi = __byte_perm(0x26190D01u, 0x71594535u, sel & 0x7777u);
    return __byte_perm(lo, hi, 0x3210u | ((sel & 0x8888u) >> 1));
}
template <> struct Tiled<T_Q6_K> { static constexpr int CHUNK = 3392; };

// Byte offset of the nibble word for (row r of the tile, step s, lane t).
__device__ __forceinline__ int tile_word(int r, int s, int t) {
    return (s >> 1) * 512 + (4 * (r & 7) + t) * 16 + (s & 1) * 8 + (r >> 3) * 4;
}

// q as an exact float without an integer conversion: 2^23 + q, less 2^23.
__device__ __forceinline__ float nibf(uint32_t w, int sh) {
    return __uint_as_float(((w >> sh) & 0xFu) | 0x4B000000u) - 8388608.f;
}
// The 8 weights of a word, rounded as the decoders round them: Q4_K as
// q * scale - min in one step, Q4_0 as (q - 8) * d (which keeps the sign of
// a zero weight, as the decoder does).
// Q5_K: q * scale - min with q's high bit from `hi` (input i at bit 2(i & 3) + (i >> 2)).
// Q6_K: (q - 32) * scale, with q's two high bits from `hi` (input i at bit 2i).
template <int T>
__device__ __forceinline__ void tile_dec8(uint32_t w, float sc, float mn, float* o, uint32_t hi = 0) {
    if constexpr (IQ4<T>) {
        // Inputs 0 4 1 5, then 2 6 3 7, as the nibble masks give them.
        const uint32_t a = iq4_bytes(w & 0x0F0F0F0Fu), b = iq4_bytes((w >> 4) & 0x0F0F0F0Fu);
        constexpr int AT[8] = {0, 2, 4, 6, 1, 3, 5, 7};
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int k = AT[i];
            o[i] = (float)(int8_t)(((k < 4 ? a : b) >> (8 * (k & 3))) & 0xFF) * sc;
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int sh = (i & 1) * 16 + (i >> 1) * 4;
        if constexpr (T == T_Q4_0) {
            o[i] = (__uint_as_float(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.f) * sc;
        } else if constexpr (T == T_Q6_K) {
            const uint32_t q = ((w >> sh) & 0xFu) | (((hi >> (2 * i)) & 3u) << 4);
            o[i] = (__uint_as_float(q | 0x4B000000u) - 8388640.f) * sc;
        } else if constexpr (T == T_Q5_K) {
            const uint32_t q = ((w >> sh) & 0xFu) | (((hi >> (2 * (i & 3) + (i >> 2))) & 1u) << 4);
            o[i] = fmaf(__uint_as_float(q | 0x4B000000u) - 8388608.f, sc, -mn);
        } else {
            o[i] = fmaf(nibf(w, sh), sc, -mn);
        }
    }
}

// Per-row scale data of a chunk: Q4_0 eight d values; Q4_K scales, mins, (d, dmin).
struct TileRow {
    uint4 a;
    uint32_t b;
};
template <int T>
__device__ __forceinline__ TileRow tile_row(const uint8_t* c, int r) {
    TileRow x;
    constexpr int SC = T == T_Q6_K ? 3072 : T == T_Q5_K ? 2560 : 2048;
    constexpr int D = T == T_Q6_K ? 3328 : T == T_Q5_K ? 2816 : 2304;
    x.a = *reinterpret_cast<const uint4*>(c + SC + 16 * r);
    x.b = T == T_Q4_0 || T == T_IQ4_NL ? 0u : *reinterpret_cast<const uint32_t*>(c + D + 4 * r);
    return x;
}
__device__ __forceinline__ uint32_t word_of(uint4 v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
template <int T>
__device__ __forceinline__ void tile_scale(const TileRow& x, int s, int t, float& sc, float& mn) {
    if constexpr (T == T_Q6_K) {
        const int k = 2 * s + (t >> 1);
        const int8_t q = (int8_t)((word_of(x.a, k >> 2) >> (8 * (k & 3))) & 0xFF);
        sc = __half2float(__ushort_as_half((unsigned short)x.b)) * q;
        mn = 0.f;
    } else if constexpr (T == T_IQ4_XS) {
        const int8_t q = (int8_t)((word_of(x.a, s >> 2) >> (8 * (s & 3))) & 0xFF);
        sc = __half2float(__ushort_as_half((unsigned short)x.b)) * q;
        mn = 0.f;
    } else if constexpr (T == T_Q4_K || T == T_Q5_K) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(&x.b));
        sc = dm.x * (float)((word_of(x.a, s >> 2) >> (8 * (s & 3))) & 0xFF);
        mn = dm.y * (float)((word_of(x.a, 2 + (s >> 2)) >> (8 * (s & 3))) & 0xFF);
    } else {
        const uint32_t h = word_of(x.a, s >> 1) >> (16 * (s & 1));
        sc = __half2float(__ushort_as_half((unsigned short)h));
        mn = T == T_IQ4_NL ? 0.f : 8.f * sc;
    }
}

// Rewrites a matrix of ggml blocks (N rows of K) in the tiled layout, one
// thread per (row, step of 32).
template <int T>
__device__ void tile_rewrite(const uint8_t* __restrict__ src, int N, int K, uint8_t* __restrict__ dst) {
    const int steps = K / 32;
    const long long n = (long long)N * steps;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int row = (int)(e / steps), gs = (int)(e % steps), kb = gs / 8, s = gs % 8, r = row & 15;
        uint8_t* c = dst + ((size_t)(row / 16) * (K / 256) + kb) * Tiled<T>::CHUNK;
        uint8_t q[32];
        if constexpr (T == T_Q6_K) {
            const block_q6_K* b = reinterpret_cast<const block_q6_K*>(src) + (size_t)row * (K / 256) + kb;
            const int ip = s / 4, g = s % 4;
            uint8_t h[32];
            for (int l = 0; l < 32; l++) {
                q[l] = (b->ql[64 * ip + 32 * (g & 1) + l] >> (4 * (g >> 1))) & 0xF;
                h[l] = (b->qh[32 * ip + l] >> (2 * g)) & 3;
            }
            for (int t = 0; t < 4; t++) {
                uint32_t w = 0;
                for (int i = 0; i < 8; i++) w |= (uint32_t)h[8 * t + i] << (2 * i);
                *reinterpret_cast<uint16_t*>(c + 2048 + (s >> 2) * 512 + (4 * (r & 7) + t) * 16 + (s & 3) * 4 +
                                             (r >> 3) * 2) = (uint16_t)w;
            }
            if (s == 0) {
                for (int i = 0; i < 16; i++) c[3072 + 16 * r + i] = (uint8_t)b->scales[i];
                *reinterpret_cast<half*>(c + 3328 + 4 * r) = b->d;
            }
        } else if constexpr (T == T_IQ4_XS) {
            const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(src) + (size_t)row * (K / 256) + kb;
            for (int i = 0; i < 32; i++) q[i] = (b->qs[16 * s + (i & 15)] >> (4 * (i >> 4))) & 0xF;
            const int ls = ((b->scales_l[s / 2] >> 4 * (s % 2)) & 0xF) | (((b->scales_h >> 2 * s) & 3) << 4);
            c[2048 + 16 * r + s] = (uint8_t)(int8_t)(ls - 32);
            if (s == 0) *reinterpret_cast<half*>(c + 2304 + 4 * r) = b->d;
        } else if constexpr (T == T_Q5_K) {
            const block_q5_K* b = reinterpret_cast<const block_q5_K*>(src) + (size_t)row * (K / 256) + kb;
            const int il = s / 2, hi = s & 1;
            for (int i = 0; i < 32; i++) q[i] = (b->qs[32 * il + i] >> (4 * hi)) & 0xF;
            for (int t = 0; t < 4; t++) {
                uint32_t h = 0;
                for (int i = 0; i < 8; i++) h |= (uint32_t)((b->qh[8 * t + i] >> s) & 1) << (2 * (i & 3) + (i >> 2));
                c[2048 + 16 * (4 * (r & 7) + t) + 4 * (s >> 1) + 2 * (s & 1) + (r >> 3)] = (uint8_t)h;
            }
            int sc, m;
            scale_min_k4(s, b->scales, sc, m);
            c[2560 + 16 * r + s] = (uint8_t)sc;
            c[2560 + 16 * r + 8 + s] = (uint8_t)m;
            if (s == 0) *reinterpret_cast<half2*>(c + 2816 + 4 * r) = b->dm;
        } else if constexpr (T == T_Q4_K) {
            const block_q4_K* b = reinterpret_cast<const block_q4_K*>(src) + (size_t)row * (K / 256) + kb;
            const int il = s / 2, hi = s & 1;
            for (int i = 0; i < 32; i++) q[i] = (b->qs[32 * il + i] >> (4 * hi)) & 0xF;
            int sc, m;
            scale_min_k4(s, b->scales, sc, m);
            c[2048 + 16 * r + s] = (uint8_t)sc;
            c[2048 + 16 * r + 8 + s] = (uint8_t)m;
            if (s == 0) *reinterpret_cast<half2*>(c + 2304 + 4 * r) = b->dm;
        } else {
            const block_q4_0* b = reinterpret_cast<const block_q4_0*>(src) + (size_t)row * (K / 32) + gs;
            for (int i = 0; i < 32; i++) q[i] = (b->qs[i & 15] >> (4 * (i >> 4))) & 0xF;
            *reinterpret_cast<half*>(c + 2048 + 16 * r + 2 * s) = b->d;
        }
        for (int t = 0; t < 4; t++) {
            uint32_t w = 0;
            for (int p = 0; p < 4; p++)
                w |= ((uint32_t)q[8 * t + 2 * p] << (4 * p)) | ((uint32_t)q[8 * t + 2 * p + 1] << (16 + 4 * p));
            *reinterpret_cast<uint32_t*>(c + tile_word(r, s, t)) = w;
        }
    }
}

// out[r - r0][:] = bf16(W[r][:]) from the tiled layout; r0 and rows are
// multiples of 16.
template <int T>
__device__ void tile_dequant(const uint8_t* __restrict__ W, int r0, int rows, int cols, bf16* __restrict__ out) {
    const int per_row = cols / 8;
    const long long n = (long long)rows * per_row;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e / per_row), k8 = (int)(e % per_row), row = r0 + r;
        const int kb = k8 / 32, s = (k8 / 4) % 8, t = k8 % 4;
        const uint8_t* c = W + ((size_t)(row / 16) * (cols / 256) + kb) * Tiled<T>::CHUNK;
        float sc, mn, v[8];
        tile_scale<T>(tile_row<T>(c, row & 15), s, t, sc, mn);
        uint32_t hi = 0;
        if constexpr (T == T_Q6_K)
            hi = *reinterpret_cast<const uint16_t*>(c + 2048 + (s >> 2) * 512 + (4 * ((row & 15) & 7) + t) * 16 +
                                                    (s & 3) * 4 + ((row & 15) >> 3) * 2);
        if constexpr (T == T_Q5_K)
            hi = c[2048 + 16 * (4 * ((row & 15) & 7) + t) + 4 * (s >> 1) + 2 * (s & 1) + ((row & 15) >> 3)];
        tile_dec8<T>(*reinterpret_cast<const uint32_t*>(c + tile_word(row & 15, s, t)), sc, mn, v, hi);
        *reinterpret_cast<uint4*>(out + (size_t)r * cols + 8 * k8) = pack8(v);
    }
}

__device__ __forceinline__ uint4 ld_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Y[m][off + n] = X[m] . W[n] from the tiled layout, on tensor cores: qmma's
// fragment order, with each warp's weights for two steps in one coalesced
// 16-byte load per lane, the next chunk's loads issued before this chunk's
// math, and split-K over chunks of 256 inputs.
template <int T, int MT>
__device__ void qtile(const uint8_t* __restrict__ W, int N, int K, const bf16* __restrict__ X, int ldx, int M,
                      void* __restrict__ Y, int ldy, int off, int f32_out, float* __restrict__ acc_ws, int ksplit) {
    constexpr int CH = Tiled<T>::CHUNK;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp, n0 = tile * 16;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * MT;
    const int KB = K / 256, per = (KB + ksplit - 1) / ksplit;
    const int b0 = blockIdx.z * per, b1 = min(KB, b0 + per);
    const uint8_t* base = W + (size_t)tile * KB * CH;
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
    // Two chunks in flight: while one is computed, the next two are loading.
    struct Chunk {
        uint4 q[4];
        uint4 h[T == T_Q6_K ? 2 : 1];
        TileRow ra, rb;
    };
    auto fetch = [&](int kb, Chunk& ch) {
        const uint8_t* c = base + (size_t)kb * CH;
#pragma unroll
        for (int i = 0; i < 4; i++) ch.q[i] = ld_stream(c + 512 * i + 16 * lane);
        if constexpr (T == T_Q6_K) {
#pragma unroll
            for (int i = 0; i < 2; i++) ch.h[i] = ld_stream(c + 2048 + 512 * i + 16 * lane);
        }
        if constexpr (T == T_Q5_K) ch.h[0] = ld_stream(c + 2048 + 16 * lane);
        ch.ra = tile_row<T>(c, g);
        ch.rb = tile_row<T>(c, g + 8);
    };
    auto compute = [&](int kb, const Chunk& ch) {
#pragma unroll
        for (int s = 0; s < 8; s++) {
            uint4 xv[MT / 8];
#pragma unroll
            for (int j = 0; j < MT / 8; j++)
                xv[j] = xok[j] ? *reinterpret_cast<const uint4*>(xr[j] + 256 * kb + 32 * s) : make_uint4(0, 0, 0, 0);
            float sa, ma, sb, mb, wa[8], wb[8];
            tile_scale<T>(ch.ra, s, t, sa, ma);
            tile_scale<T>(ch.rb, s, t, sb, mb);
            uint32_t ha = 0, hb = 0;
            if constexpr (T == T_Q6_K) {
                const uint32_t hw = word_of(ch.h[(s >> 2) & 1], s & 3);
                ha = hw & 0xFFFFu, hb = hw >> 16;
            } else if constexpr (T == T_Q5_K) {
                const uint32_t hw = word_of(ch.h[0], s >> 1) >> (16 * (s & 1));
                ha = hw & 0xFFu, hb = (hw >> 8) & 0xFFu;
            }
            tile_dec8<T>(word_of(ch.q[s >> 1], 2 * (s & 1)), sa, ma, wa, ha);
            tile_dec8<T>(word_of(ch.q[s >> 1], 2 * (s & 1) + 1), sb, mb, wb, hb);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t a[4] = {bf2(wa[4 * h], wa[4 * h + 1]), bf2(wb[4 * h], wb[4 * h + 1]),
                                       bf2(wa[4 * h + 2], wa[4 * h + 3]), bf2(wb[4 * h + 2], wb[4 * h + 3])};
#pragma unroll
                for (int j = 0; j < MT / 8; j++)
                    mma_bf16(acc[j], a, h ? xv[j].z : xv[j].x, h ? xv[j].w : xv[j].y);
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

// ---------------------------------------------------------------------------
// W4A8: the tiled 4-bit layout against int8 activations, on int8 tensor cores
// (mma m16n8k32), llama.cpp's MMQ scheme. Each step of 32 inputs gives an
// exact integer dot product per (row, token), scaled by the row's step scale
// and the token's activation scale.

// Xq = round(x / d) in int8, d = max|x| / 127 per (m, step of 32). Each
// group of 8 is stored as inputs 0 4 1 5 2 6 3 7, the byte order the nibble
// words of the tiled layout unpack to under one mask, and each row's chunk
// of 256 as [step pair][lane t][step & 1][8 bytes], so a lane reads two
// steps in one 16-byte load. Xd holds d per chunk and token pair as
// [step pair][step & 1][token & 1] (16 floats per pair); Xsum[step][Mp] the
// sums of x. One thread per group of 8.
extern "C" __global__ void quant_act8(const bf16* __restrict__ X, int ldx, int M, int K, int Mp,
                                      int8_t* __restrict__ Xq, float* __restrict__ Xd) {
    float* Xsum = Xd + (size_t)(K / 32) * Mp;
    const int groups = K / 8;
    const long long n = (long long)M * groups;
    // Four consecutive threads share a step; n is a multiple of 4.
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e - (threadIdx.x & 3) < n;
         e += (long long)gridDim.x * blockDim.x) {
        const bool ok = e < n;
        const int m = ok ? (int)(e / groups) : 0, k8 = ok ? (int)(e % groups) : 0;
        float v[8];
        const uint4 raw = ok ? *reinterpret_cast<const uint4*>(X + (size_t)m * ldx + 8 * k8) : make_uint4(0, 0, 0, 0);
        const bf16* h = reinterpret_cast<const bf16*>(&raw);
        float amax = 0.f, sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            v[i] = __bfloat162float(h[i]);
            amax = fmaxf(amax, fabsf(v[i]));
            sum += v[i];
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        if (!ok) continue;
        const float d = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
        constexpr int ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        uint32_t w[2] = {0, 0};
#pragma unroll
        for (int i = 0; i < 8; i++)
            w[i >> 2] |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(v[ORDER[i]] * id) << (8 * (i & 3));
        const int kb = k8 >> 5, st = (k8 >> 2) & 7, t = k8 & 3;
        *reinterpret_cast<uint2*>(Xq + (size_t)m * K + 256 * kb + 64 * (st >> 1) + 16 * t + 8 * (st & 1)) =
            make_uint2(w[0], w[1]);
        if (t == 0) {
            Xd[(((size_t)kb * (Mp / 2) + m / 2) * 4 + (st >> 1)) * 4 + (st & 1) * 2 + (m & 1)] = d;
            Xsum[(size_t)(k8 >> 2) * Mp + m] = sum;
        }
    }
}

__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

// Q6_K's two high bits of a word's 8 inputs (input i at bit 2i) moved to bit
// 4 of each byte, in the byte order of w & 0x0F0F0F0F (inputs 0 4 1 5); hi >> 4
// gives the other word's (2 6 3 7).
__device__ __forceinline__ uint32_t hi2_bytes(uint32_t h) {
    return ((h & 0x0303u) | ((h & 0x0C0Cu) << 14)) << 4;
}
// Q5_K's high bits of a word's 8 inputs (inputs 0 4 1 5 at bits 0 to 3) moved
// to bit 4 of each byte; hi >> 4 gives inputs 2 6 3 7.
__device__ __forceinline__ uint32_t hi1_bytes(uint32_t h) {
    return (((h & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}
// q - 8 in each byte of q (0 to 15), as signed bytes.
__device__ __forceinline__ uint32_t less8(uint32_t q) {
    return ((q | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}
__device__ __forceinline__ uint32_t tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma_tf32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// q - 32 in each byte of q (0 to 63), as signed bytes.
__device__ __forceinline__ uint32_t less32(uint32_t q) {
    return ((q | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}

// qtile's loop with int8 activations. For Q4_K and Q5_K a row's weight at
// step s is q * sc - mn, so its dot product with token x is
// sc * dx * sum(q * xq) - mn * sum(x). The first term is a step's int8
// product; the second, dmin * sum over steps of m * sum(x), is one TF32
// product per chunk, with sum(x) split in two TF32 parts so it stays
// exact. Q4_0 takes q - 8 as signed bytes and IQ4 the table's values, so
// neither has a second term. Q6_K's weights are
// (q - 32) * d * sc with a scale per 16 inputs, so each step takes two
// products, one per half (lanes t < 2 hold the first half's inputs), summed
// in integers with the scales. The block's four warps share each chunk's
// activations, staged in shared memory a chunk ahead with cp.async.
#define Q8_XSTR 320  // byte stride of an activation row: 256 plus 64 against bank conflicts
template <int MT>
struct Q8Stage {
    int8_t x[MT * Q8_XSTR];
    float d[8 * MT];    // Xd's layout
    float sum[8 * MT];  // [step][token]
};
template <int T, int MT>
__device__ void qtile8(const uint8_t* __restrict__ W, int N, int K, const int8_t* __restrict__ Xq,
                       const float* __restrict__ Xd, int Mp, int M, void* __restrict__ Y, int ldy, int off,
                       int f32_out, float* __restrict__ acc_ws, int ksplit) {
    constexpr int CH = Tiled<T>::CHUNK;
    constexpr bool Q6 = T == T_Q6_K, Q5 = T == T_Q5_K;
    __shared__ __align__(16) Q8Stage<MT> st[2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp, n0 = tile * 16;
    // Warps past the last row still help stage activations.
    const bool live = n0 < N;
    const int m0 = blockIdx.y * MT;
    const int KB = K / 256, per = (KB + ksplit - 1) / ksplit;
    const int b0 = blockIdx.z * per, b1 = min(KB, b0 + per);
    const uint8_t* base = W + (size_t)(live ? tile : 0) * KB * CH;
    float acc[MT / 8][4];
#pragma unroll
    for (int j = 0; j < MT / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    auto stage = [&](int kb, Q8Stage<MT>& sb) {
#pragma unroll
        for (int i = threadIdx.x; i < MT * 16; i += 128) {
            const int r = i >> 4, c = (i & 15) * 16, m = m0 + r;
            cp16z(sb.x + r * Q8_XSTR + c, Xq + (size_t)(m < M ? m : 0) * K + 256 * kb + c, m < M);
        }
#pragma unroll
        for (int i = threadIdx.x; i < MT * 2; i += 128)
            cp16z(sb.d + 4 * i, Xd + ((size_t)kb * (Mp / 2) + m0 / 2) * 16 + 4 * i, true);
        const float* Xsum = Xd + (size_t)(K / 32) * Mp;
#pragma unroll
        for (int i = threadIdx.x; i < MT * 2; i += 128) {
            const int s = i / (MT / 4), c = i % (MT / 4);
            cp16z(sb.sum + s * MT + 4 * c, Xsum + (size_t)(8 * kb + s) * Mp + m0 + 4 * c, true);
        }
        cp_commit();
    };
    struct Chunk {
        uint4 q[4];
        uint4 h[Q6 ? 2 : 1];
        TileRow ra, rb;
    };
    auto fetch = [&](int kb, Chunk& ch) {
        if (!live) return;
        const uint8_t* c = base + (size_t)kb * CH;
#pragma unroll
        for (int i = 0; i < 4; i++) ch.q[i] = ld_stream(c + 512 * i + 16 * lane);
        if constexpr (Q6) {
#pragma unroll
            for (int i = 0; i < 2; i++) ch.h[i] = ld_stream(c + 2048 + 512 * i + 16 * lane);
        }
        if constexpr (Q5) ch.h[0] = ld_stream(c + 2048 + 16 * lane);
        ch.ra = tile_row<T>(c, g);
        ch.rb = tile_row<T>(c, g + 8);
    };
    auto compute = [&](const Chunk& ch, const Q8Stage<MT>& sb) {
        if (!live) return;
#pragma unroll
        for (int p = 0; p < 4; p++) {
            // Two steps of activations and their scales: x of steps 2p, 2p + 1
            // (two words each), d of (step, token) (2p, 2t) (2p, 2t + 1)
            // (2p + 1, 2t) (2p + 1, 2t + 1).
            auto xv = [&](int j) {
                return *reinterpret_cast<const uint4*>(sb.x + (8 * j + g) * Q8_XSTR + 64 * p + 16 * t);
            };
            auto dd = [&](int j) { return *reinterpret_cast<const float4*>(sb.d + ((4 * j + t) * 4 + p) * 4); };
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int s = 2 * p + h;
                const uint32_t wa = word_of(ch.q[s >> 1], 2 * (s & 1)), wb = word_of(ch.q[s >> 1], 2 * (s & 1) + 1);
                uint32_t a[4] = {wa & 0x0F0F0F0Fu, wb & 0x0F0F0F0Fu, (wa >> 4) & 0x0F0F0F0Fu, (wb >> 4) & 0x0F0F0F0Fu};
                if constexpr (Q6) {
                    const uint32_t hw = word_of(ch.h[s >> 2], s & 3), ha = hw & 0xFFFFu, hb = hw >> 16;
                    a[0] = less32(a[0] | hi2_bytes(ha));
                    a[1] = less32(a[1] | hi2_bytes(hb));
                    a[2] = less32(a[2] | hi2_bytes(ha >> 4));
                    a[3] = less32(a[3] | hi2_bytes(hb >> 4));
                    // The step's two int8 scales per row, and d.
                    const int sa0 = (int8_t)(word_of(ch.ra.a, s >> 1) >> (16 * (s & 1)));
                    const int sa1 = (int8_t)(word_of(ch.ra.a, s >> 1) >> (16 * (s & 1) + 8));
                    const int sb0 = (int8_t)(word_of(ch.rb.a, s >> 1) >> (16 * (s & 1)));
                    const int sb1 = (int8_t)(word_of(ch.rb.a, s >> 1) >> (16 * (s & 1) + 8));
                    const float da = __half2float(__ushort_as_half((unsigned short)ch.ra.b));
                    const float db = __half2float(__ushort_as_half((unsigned short)ch.rb.b));
                    const bool first = t < 2;
                    const uint32_t lo[4] = {first ? a[0] : 0u, first ? a[1] : 0u, first ? a[2] : 0u, first ? a[3] : 0u};
                    const uint32_t hi[4] = {first ? 0u : a[0], first ? 0u : a[1], first ? 0u : a[2], first ? 0u : a[3]};
#pragma unroll
                    for (int j = 0; j < MT / 8; j++) {
                        const uint4 x = xv(j);
                        const float4 dj = dd(j);
                        const uint32_t b0 = h ? x.z : x.x, b1 = h ? x.w : x.y;
                        const float d0 = h ? dj.z : dj.x, d1 = h ? dj.w : dj.y;
                        int c0[4], c1[4];
                        mma_s8(c0, lo, b0, b1);
                        mma_s8(c1, hi, b0, b1);
                        acc[j][0] = fmaf((float)(c0[0] * sa0 + c1[0] * sa1), da * d0, acc[j][0]);
                        acc[j][1] = fmaf((float)(c0[1] * sa0 + c1[1] * sa1), da * d1, acc[j][1]);
                        acc[j][2] = fmaf((float)(c0[2] * sb0 + c1[2] * sb1), db * d0, acc[j][2]);
                        acc[j][3] = fmaf((float)(c0[3] * sb0 + c1[3] * sb1), db * d1, acc[j][3]);
                    }
                    continue;
                }
                if constexpr (IQ4<T>) {
#pragma unroll
                    for (int k = 0; k < 4; k++) a[k] = iq4_bytes(a[k]);
                }
                if constexpr (T == T_Q4_0) {
#pragma unroll
                    for (int k = 0; k < 4; k++) a[k] = less8(a[k]);
                }
                if constexpr (Q5) {
                    const uint32_t hw = word_of(ch.h[0], s >> 1) >> (16 * (s & 1));
                    a[0] |= hi1_bytes(hw);
                    a[1] |= hi1_bytes(hw >> 8);
                    a[2] |= hi1_bytes(hw >> 4);
                    a[3] |= hi1_bytes(hw >> 12);
                }
                float sa, ma, sb_, mb;
                tile_scale<T>(ch.ra, s, t, sa, ma);
                tile_scale<T>(ch.rb, s, t, sb_, mb);
#pragma unroll
                for (int j = 0; j < MT / 8; j++) {
                    const uint4 x = xv(j);
                    const float4 dj = dd(j);
                    const uint32_t b0 = h ? x.z : x.x, b1 = h ? x.w : x.y;
                    const float d0 = h ? dj.z : dj.x, d1 = h ? dj.w : dj.y;
                    int c[4];
                    mma_s8(c, a, b0, b1);
                    acc[j][0] = fmaf((float)c[0], sa * d0, acc[j][0]);
                    acc[j][1] = fmaf((float)c[1], sa * d1, acc[j][1]);
                    acc[j][2] = fmaf((float)c[2], sb_ * d0, acc[j][2]);
                    acc[j][3] = fmaf((float)c[3], sb_ * d1, acc[j][3]);
                }
            }
        }
        if constexpr (T == T_Q4_K || T == T_Q5_K) {
            // Mins of rows g and g + 8 at steps t and t + 4 (6-bit, exact in TF32).
            auto m6 = [&](const TileRow& r, int s) {
                return __float_as_uint((float)((word_of(r.a, 2 + (s >> 2)) >> (8 * (s & 3))) & 0xFF));
            };
            const uint32_t am[4] = {m6(ch.ra, t), m6(ch.rb, t), m6(ch.ra, t + 4), m6(ch.rb, t + 4)};
            const float dma = __half2float(__ushort_as_half((unsigned short)(ch.ra.b >> 16)));
            const float dmb = __half2float(__ushort_as_half((unsigned short)(ch.rb.b >> 16)));
#pragma unroll
            for (int j = 0; j < MT / 8; j++) {
                const float x0 = sb.sum[t * MT + 8 * j + g], x1 = sb.sum[(t + 4) * MT + 8 * j + g];
                const uint32_t h0 = tf32(x0), h1 = tf32(x1);
                const uint32_t l0 = tf32(x0 - __uint_as_float(h0)), l1 = tf32(x1 - __uint_as_float(h1));
                float r[4] = {0.f, 0.f, 0.f, 0.f};
                mma_tf32(r, am, l0, l1);
                mma_tf32(r, am, h0, h1);
                acc[j][0] = fmaf(-dma, r[0], acc[j][0]);
                acc[j][1] = fmaf(-dma, r[1], acc[j][1]);
                acc[j][2] = fmaf(-dmb, r[2], acc[j][2]);
                acc[j][3] = fmaf(-dmb, r[3], acc[j][3]);
            }
        }
    };
    // Weights two chunks ahead in registers, activations one ahead in shared memory.
    auto step = [&](int kb, const Chunk& w) {
        if (kb + 1 < b1) {
            stage(kb + 1, st[(kb + 1 - b0) & 1]);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        compute(w, st[(kb - b0) & 1]);
        __syncthreads();
    };
    Chunk c0, c1;
    if (b0 < b1) fetch(b0, c0), stage(b0, st[0]);
    if (b0 + 1 < b1) fetch(b0 + 1, c1);
    for (int kb = b0; kb < b1; kb += 2) {
        const Chunk w0 = c0;
        if (kb + 2 < b1) fetch(kb + 2, c0);
        step(kb, w0);
        if (kb + 1 < b1) {
            const Chunk w1 = c1;
            if (kb + 3 < b1) fetch(kb + 3, c1);
            step(kb + 1, w1);
        }
    }
    if (!live) return;
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

// Large batches: a block takes 128 rows (8 tiles of 16) by 64 tokens over
// all of K, with each chunk's weights and activations staged in shared
// memory a chunk ahead, and 4 x WM warps of 32 rows by 64 / WM tokens
// reading both from there. Blocks run token tiles first, so the blocks sharing weights
// run together and read them from L2. One int8 product a step, so Q6_K
// stays on qtile8.
#define QB_M 64
template <int T, int WM>
__device__ void qbig8(const uint8_t* __restrict__ W, int N, int K, const int8_t* __restrict__ Xq,
                      const float* __restrict__ Xd, int Mp, int M, void* __restrict__ Y, int ldy, int off,
                      int f32_out) {
    static_assert(T != T_Q6_K, "one product a step");
    // Bytes per stage: weights, activations, then d and sums (Rust's Int8Kernels::big repeats this).
    constexpr int CH = Tiled<T>::CHUNK, STAGE = 8 * CH + QB_M * Q8_XSTR + (10 + 8) * QB_M * 4;
    extern __shared__ __align__(16) uint8_t qb_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int NT = 128 * WM, J = 8 / WM;
    const int wn = warp & 3, wm = warp >> 2;
    const int m0 = blockIdx.x * QB_M, tile0 = blockIdx.y * 8;
    const int KB = K / 256, ntiles = (N + 15) / 16;
    const float* Xsum = Xd + (size_t)(K / 32) * Mp;
    float acc[2][J][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int j = 0; j < J; j++) acc[rt][j][0] = acc[rt][j][1] = acc[rt][j][2] = acc[rt][j][3] = 0.f;
    auto stage = [&](int kb, uint8_t* sm) {
        for (int i = threadIdx.x; i < 8 * (CH / 16); i += NT) {
            const int r = i / (CH / 16), c = (i % (CH / 16)) * 16, tile = tile0 + r;
            const bool ok = tile < ntiles;
            cp16z(sm + r * CH + c, W + ((size_t)(ok ? tile : 0) * KB + kb) * CH + c, ok);
        }
        int8_t* sx = reinterpret_cast<int8_t*>(sm + 8 * CH);
        for (int i = threadIdx.x; i < QB_M * 16; i += NT) {
            const int r = i >> 4, c = (i & 15) * 16, m = m0 + r;
            cp16z(sx + r * Q8_XSTR + c, Xq + (size_t)(m < M ? m : 0) * K + 256 * kb + c, m < M);
        }
        // d per token pair, 16 floats padded to 20 against bank conflicts.
        float* sd = reinterpret_cast<float*>(sx + QB_M * Q8_XSTR);
        for (int i = threadIdx.x; i < QB_M * 2; i += NT)
            cp16z(sd + 20 * (i >> 2) + 4 * (i & 3), Xd + ((size_t)kb * (Mp / 2) + m0 / 2) * 16 + 4 * i, true);
        float* ss = sd + 10 * QB_M;
        for (int i = threadIdx.x; i < QB_M * 2; i += NT) {
            const int s = i / (QB_M / 4), c = i % (QB_M / 4);
            cp16z(ss + s * QB_M + 4 * c, Xsum + (size_t)(8 * kb + s) * Mp + m0 + 4 * c, true);
        }
        cp_commit();
    };
    auto compute = [&](const uint8_t* sm) {
        const int8_t* sx = reinterpret_cast<const int8_t*>(sm + 8 * CH);
        const float* sd = reinterpret_cast<const float*>(sx + QB_M * Q8_XSTR);
        const float* ss = sd + 10 * QB_M;
        TileRow ra[2], rb[2];
        uint4 hq[2];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            const uint8_t* c = sm + (2 * wn + rt) * CH;
            ra[rt] = tile_row<T>(c, g);
            rb[rt] = tile_row<T>(c, g + 8);
            if constexpr (T == T_Q5_K) hq[rt] = *reinterpret_cast<const uint4*>(c + 2048 + 16 * lane);
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            uint4 w4[2];
#pragma unroll
            for (int rt = 0; rt < 2; rt++)
                w4[rt] = *reinterpret_cast<const uint4*>(sm + (2 * wn + rt) * CH + 512 * p + 16 * lane);
            // A step pair's activations and scales for token tile j (re-read
            // for the second step: registers run out before shared bandwidth).
            auto xv = [&](int j) {
                return *reinterpret_cast<const uint4*>(sx + ((64 / WM) * wm + 8 * j + g) * Q8_XSTR + 64 * p + 16 * t);
            };
            auto dd = [&](int j) {
                return *reinterpret_cast<const float4*>(sd + ((32 / WM) * wm + 4 * j + t) * 20 + 4 * p);
            };
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int s = 2 * p + h;
                uint32_t a[2][4];
                float sa[2], sb[2];
#pragma unroll
                for (int rt = 0; rt < 2; rt++) {
                    const uint32_t wa = word_of(w4[rt], 2 * h), wb = word_of(w4[rt], 2 * h + 1);
                    a[rt][0] = wa & 0x0F0F0F0Fu, a[rt][1] = wb & 0x0F0F0F0Fu;
                    a[rt][2] = (wa >> 4) & 0x0F0F0F0Fu, a[rt][3] = (wb >> 4) & 0x0F0F0F0Fu;
                    if constexpr (T == T_Q5_K) {
                        const uint32_t hw = word_of(hq[rt], p) >> (16 * h);
                        a[rt][0] |= hi1_bytes(hw);
                        a[rt][1] |= hi1_bytes(hw >> 8);
                        a[rt][2] |= hi1_bytes(hw >> 4);
                        a[rt][3] |= hi1_bytes(hw >> 12);
                    }
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if constexpr (IQ4<T>) a[rt][k] = iq4_bytes(a[rt][k]);
                        if constexpr (T == T_Q4_0) a[rt][k] = less8(a[rt][k]);
                    }
                    float ma, mb;
                    tile_scale<T>(ra[rt], s, t, sa[rt], ma);
                    tile_scale<T>(rb[rt], s, t, sb[rt], mb);
                }
#pragma unroll
                for (int j = 0; j < J; j++) {
                    const uint4 x = xv(j);
                    const float4 dj = dd(j);
                    const uint32_t b0 = h ? x.z : x.x, b1 = h ? x.w : x.y;
                    const float d0 = h ? dj.z : dj.x, d1 = h ? dj.w : dj.y;
#pragma unroll
                    for (int rt = 0; rt < 2; rt++) {
                        int c[4];
                        mma_s8(c, a[rt], b0, b1);
                        acc[rt][j][0] = fmaf((float)c[0], sa[rt] * d0, acc[rt][j][0]);
                        acc[rt][j][1] = fmaf((float)c[1], sa[rt] * d1, acc[rt][j][1]);
                        acc[rt][j][2] = fmaf((float)c[2], sb[rt] * d0, acc[rt][j][2]);
                        acc[rt][j][3] = fmaf((float)c[3], sb[rt] * d1, acc[rt][j][3]);
                    }
                }
            }
        }
        if constexpr (T == T_Q4_K || T == T_Q5_K) {
            auto m6 = [&](const TileRow& r, int s) {
                return __float_as_uint((float)((word_of(r.a, 2 + (s >> 2)) >> (8 * (s & 3))) & 0xFF));
            };
#pragma unroll
            for (int j = 0; j < J; j++) {
                const int tok = (64 / WM) * wm + 8 * j + g;
                const float x0 = ss[t * QB_M + tok], x1 = ss[(t + 4) * QB_M + tok];
                const uint32_t h0 = tf32(x0), h1 = tf32(x1);
                const uint32_t l0 = tf32(x0 - __uint_as_float(h0)), l1 = tf32(x1 - __uint_as_float(h1));
#pragma unroll
                for (int rt = 0; rt < 2; rt++) {
                    const uint32_t am[4] = {m6(ra[rt], t), m6(rb[rt], t), m6(ra[rt], t + 4), m6(rb[rt], t + 4)};
                    const float dma = __half2float(__ushort_as_half((unsigned short)(ra[rt].b >> 16)));
                    const float dmb = __half2float(__ushort_as_half((unsigned short)(rb[rt].b >> 16)));
                    float r[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_tf32(r, am, l0, l1);
                    mma_tf32(r, am, h0, h1);
                    acc[rt][j][0] = fmaf(-dma, r[0], acc[rt][j][0]);
                    acc[rt][j][1] = fmaf(-dma, r[1], acc[rt][j][1]);
                    acc[rt][j][2] = fmaf(-dmb, r[2], acc[rt][j][2]);
                    acc[rt][j][3] = fmaf(-dmb, r[3], acc[rt][j][3]);
                }
            }
        }
    };
    stage(0, qb_smem);
    for (int kb = 0; kb < KB; kb++) {
        if (kb + 1 < KB) {
            stage(kb + 1, qb_smem + ((kb + 1) & 1) * STAGE);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        compute(qb_smem + (kb & 1) * STAGE);
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int j = 0; j < J; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int n = (tile0 + 2 * wn + rt) * 16 + g + (e >> 1) * 8;
                const int m = m0 + (64 / WM) * wm + 8 * j + 2 * t + (e & 1);
                if (n < N && m < M) {
                    if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc[rt][j][e];
                    else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc[rt][j][e]);
                }
            }
}
#define QBIG8W(T, name, WM)                                                                            \
    extern "C" __global__ void __launch_bounds__(128 * WM) qbig8_##name##_##WM(                        \
        const uint8_t* W, int N, int K, const int8_t* Xq, const float* Xd, int Mp, int M, void* Y,     \
        int ldy, int off, int f32_out) {                                                               \
        qbig8<T, WM>(W, N, K, Xq, Xd, Mp, M, Y, ldy, off, f32_out);                                    \
    }
#define QBIG8(T, name) QBIG8W(T, name, 1) QBIG8W(T, name, 2)

#define QTILE8(T, name, MT)                                                                            \
    extern "C" __global__ void __launch_bounds__(128) qtile8_##name##_##MT(                            \
        const uint8_t* W, int N, int K, const int8_t* Xq, const float* Xd, int Mp, int M, void* Y,     \
        int ldy, int off, int f32_out, float* acc_ws, int ksplit) {                                    \
        qtile8<T, MT>(W, N, K, Xq, Xd, Mp, M, Y, ldy, off, f32_out, acc_ws, ksplit);                   \
    }
QTILE8(T_Q4_0, q4_0, 8) QTILE8(T_Q4_0, q4_0, 16) QTILE8(T_Q4_0, q4_0, 32) QTILE8(T_Q4_0, q4_0, 64)
QTILE8(T_Q4_K, q4_k, 8) QTILE8(T_Q4_K, q4_k, 16) QTILE8(T_Q4_K, q4_k, 32) QTILE8(T_Q4_K, q4_k, 64)
QTILE8(T_Q5_K, q5_k, 8) QTILE8(T_Q5_K, q5_k, 16) QTILE8(T_Q5_K, q5_k, 32) QTILE8(T_Q5_K, q5_k, 64)
QTILE8(T_IQ4_NL, iq4_nl, 8) QTILE8(T_IQ4_NL, iq4_nl, 16) QTILE8(T_IQ4_NL, iq4_nl, 32) QTILE8(T_IQ4_NL, iq4_nl, 64)
QTILE8(T_IQ4_XS, iq4_xs, 8) QTILE8(T_IQ4_XS, iq4_xs, 16) QTILE8(T_IQ4_XS, iq4_xs, 32) QTILE8(T_IQ4_XS, iq4_xs, 64)
QBIG8(T_Q4_0, q4_0) QBIG8(T_Q4_K, q4_k) QBIG8(T_Q5_K, q5_k) QBIG8(T_IQ4_NL, iq4_nl) QBIG8(T_IQ4_XS, iq4_xs)
QTILE8(T_Q6_K, q6_k, 8) QTILE8(T_Q6_K, q6_k, 16) QTILE8(T_Q6_K, q6_k, 32) QTILE8(T_Q6_K, q6_k, 64)

#define QTILE(T, name, MT)                                                                             \
    extern "C" __global__ void __launch_bounds__(128) qtile_##name##_##MT(                             \
        const uint8_t* W, int N, int K, const bf16* X, int ldx, int M, void* Y, int ldy, int off,      \
        int f32_out, float* acc_ws, int ksplit) {                                                      \
        qtile<T, MT>(W, N, K, X, ldx, M, Y, ldy, off, f32_out, acc_ws, ksplit);                        \
    }
#define TILED_KERNELS(T, name)                                                                         \
    extern "C" __global__ void tile_##name(const uint8_t* src, int N, int K, uint8_t* dst) {           \
        tile_rewrite<T>(src, N, K, dst);                                                               \
    }                                                                                                  \
    extern "C" __global__ void dequant_tiled_##name(const uint8_t* W, int r0, int rows, int cols,      \
                                                    bf16* out) {                                       \
        tile_dequant<T>(W, r0, rows, cols, out);                                                       \
    }                                                                                                  \
    QTILE(T, name, 8) QTILE(T, name, 16) QTILE(T, name, 32) QTILE(T, name, 64)
TILED_KERNELS(T_Q4_0, q4_0)
TILED_KERNELS(T_Q4_K, q4_k)
TILED_KERNELS(T_Q5_K, q5_k)
TILED_KERNELS(T_IQ4_NL, iq4_nl)
TILED_KERNELS(T_IQ4_XS, iq4_xs)
TILED_KERNELS(T_Q6_K, q6_k)

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
