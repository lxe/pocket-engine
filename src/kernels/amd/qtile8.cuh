// qtile8 (int8 activations against the tiled 4- to 6-bit layouts) on AMD's
// int8 matrix cores: see qtile8 in quant.cuh for the math.
#pragma once

// MFMA 16x16x32 int8 (gfx942): a lane holds A[m][k..k+7] and B[k..k+7][n]
// (m, n = lane % 16, k = 8 (lane / 16)) and C[m..m+3][n] (m = 4 (lane / 16)).
// A lane's 8 inputs of a weight row at one step are one nibble word of the
// tiled layout (tile_word), which unmasks to bytes in the order 0 4 1 5 2 6 3
// 7: the order quant_act8 stores activations in, so B is a plain 8-byte load
// (at quant_act8's [step pair][lane][step & 1] place in the chunk).
// The block's two waves take two 16-row tiles each of its 64 rows; tokens
// go in tiles of 16.
__device__ __forceinline__ int4 mfma_i8(long a, long b) {
    typedef int i4 __attribute__((ext_vector_type(4)));
    const i4 c = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, i4{0, 0, 0, 0}, 0, 0, 0);
    return make_int4(c[0], c[1], c[2], c[3]);
}
template <int T, int MT>
__device__ void qtile8_mfma(const uint8_t* __restrict__ W, int N, int K, const int8_t* __restrict__ Xq,
                            const float* __restrict__ Xd, int Mp, int M, void* __restrict__ Y, int ldy, int off,
                            int f32_out, float* __restrict__ acc_ws, int ksplit) {
    const float* Xsum = Xd + (size_t)(K / 32) * Mp;
    constexpr int CH = Tiled<T>::CHUNK, TT = (MT + 15) / 16;
    constexpr bool Q6 = T == T_Q6_K, Q5 = T == T_Q5_K;
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63, lr = lane & 15, lq = lane >> 4;
    const int m0 = blockIdx.y * MT;
    const int KB = K / 256, per = (KB + ksplit - 1) / ksplit;
    const int b0 = blockIdx.z * per, b1 = min(KB, b0 + per);
    int tiles[2];
    bool live[2];
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        tiles[rt] = blockIdx.x * 4 + 2 * wave + rt;
        live[rt] = tiles[rt] * 16 < N;
    }
    if (!live[0]) return;
    float acc[2][TT][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int tt = 0; tt < TT; tt++) acc[rt][tt][0] = acc[rt][tt][1] = acc[rt][tt][2] = acc[rt][tt][3] = 0.f;
    // This lane's tokens: m0 + 16 tt + lr (clamped into range for loads).
    const int8_t* xr[TT];
    bool xok[TT];
    int xm[TT];
#pragma unroll
    for (int tt = 0; tt < TT; tt++) {
        const int m = m0 + 16 * tt + lr;
        xok[tt] = m < M && 16 * tt + lr < MT;
        xm[tt] = xok[tt] ? m : m0;
        xr[tt] = Xq + (size_t)xm[tt] * K + 16 * lq;
    }
    const int g = lr & 7, half = lr >> 3;
    for (int kb = b0; kb < b1; kb++) {
        const uint8_t* c[2];
        TileRow tr[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            c[rt] = W + ((size_t)(live[rt] ? tiles[rt] : tiles[0]) * KB + kb) * CH;
#pragma unroll
            for (int j = 0; j < 4; j++) tr[rt][j] = tile_row<T>(c[rt], 4 * lq + j);
        }
#pragma unroll 2
        for (int s = 0; s < 8; s++) {
            const int gs = 8 * kb + s;
            long b[TT];
            float2 ds[TT];
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                const int m = xm[tt];
                b[tt] = xok[tt] ? *reinterpret_cast<const long*>(xr[tt] + 256 * kb + 64 * (s >> 1) + 8 * (s & 1)) : 0;
                ds[tt] = make_float2(Xd[(((size_t)kb * (Mp / 2) + m / 2) * 4 + (s >> 1)) * 4 + (s & 1) * 2 + (m & 1)],
                                     Xsum[(size_t)gs * Mp + m]);
            }
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                if (!live[rt]) continue;
                const uint32_t w = *reinterpret_cast<const uint32_t*>(c[rt] + tile_word(lr, s, lq));
                uint32_t lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
                if constexpr (Q6) {
                    const uint32_t h = *reinterpret_cast<const uint16_t*>(
                        c[rt] + 2048 + (s >> 2) * 512 + (4 * g + lq) * 16 + (s & 3) * 4 + half * 2);
                    lo = less32(lo | hi2_bytes(h));
                    hi = less32(hi | hi2_bytes(h >> 4));
                } else if constexpr (Q5) {
                    const uint32_t h = c[rt][2048 + 16 * (4 * g + lq) + 4 * (s >> 1) + 2 * (s & 1) + half];
                    lo |= hi1_bytes(h);
                    hi |= hi1_bytes(h >> 4);
                } else if constexpr (IQ4<T>) {
                    lo = iq4_bytes(lo);
                    hi = iq4_bytes(hi);
                }
                const long a = (long)((unsigned long)lo | ((unsigned long)hi << 32));
                if constexpr (Q6) {
                    // Two products, one per 16-input scale: lanes lq < 2 hold the first half.
                    const long a0 = lq < 2 ? a : 0, a1 = lq < 2 ? 0 : a;
                    int s0[4], s1[4];
                    float dr[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t wd = word_of(tr[rt][j].a, s >> 1);
                        s0[j] = (int8_t)(wd >> (16 * (s & 1)));
                        s1[j] = (int8_t)(wd >> (16 * (s & 1) + 8));
                        dr[j] = __half2float(__ushort_as_half((unsigned short)tr[rt][j].b));
                    }
#pragma unroll
                    for (int tt = 0; tt < TT; tt++) {
                        const int4 p0 = mfma_i8(a0, b[tt]), p1 = mfma_i8(a1, b[tt]);
                        const int q0[4] = {p0.x, p0.y, p0.z, p0.w}, q1[4] = {p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            acc[rt][tt][j] = fmaf((float)(q0[j] * s0[j] + q1[j] * s1[j]), dr[j] * ds[tt].x, acc[rt][tt][j]);
                    }
                } else {
                    float sc[4], mn[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) tile_scale<T>(tr[rt][j], s, 0, sc[j], mn[j]);
#pragma unroll
                    for (int tt = 0; tt < TT; tt++) {
                        const int4 p = mfma_i8(a, b[tt]);
                        const int q[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            float v = fmaf((float)q[j], sc[j] * ds[tt].x, acc[rt][tt][j]);
                            if constexpr (!IQ4<T>) v = fmaf(-mn[j], ds[tt].y, v);
                            acc[rt][tt][j] = v;
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        if (!live[rt]) continue;
#pragma unroll
        for (int tt = 0; tt < TT; tt++) {
            if (!xok[tt]) continue;
            const int m = m0 + 16 * tt + lr;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = tiles[rt] * 16 + 4 * lq + j;
                if (n >= N) continue;
                if (ksplit > 1) acc_ws[((size_t)blockIdx.z * M + m) * N + n] = acc[rt][tt][j];
                else if (f32_out) ((float*)Y)[(size_t)m * ldy + off + n] = acc[rt][tt][j];
                else ((bf16*)Y)[(size_t)m * ldy + off + n] = f2b(acc[rt][tt][j]);
            }
        }
    }
}
