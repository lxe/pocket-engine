// The gated delta rule for decoding on AMD (backend/hip.cuh): see gdn_one in qwen35.cu.
#pragma once

// Sums of two values across the block, one wave64 reduction and a barrier
// pair; `sh` holds two floats per wave.
__device__ __forceinline__ float2 block_sum2(float a, float b, float* sh) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) a += __shfl_xor(a, o, 64), b += __shfl_xor(b, o, 64);
    const int w = threadIdx.x >> 6, nw = blockDim.x >> 6;
    if ((threadIdx.x & 63) == 0) sh[2 * w] = a, sh[2 * w + 1] = b;
    __syncthreads();
    float2 t = make_float2(0.f, 0.f);
    for (int i = 0; i < nw; i++) t.x += sh[2 * i], t.y += sh[2 * i + 1];
    __syncthreads();
    return t;
}

// gdn_one for AMD, the same math laid out for the hardware (measured on an
// MI325X with 48 heads of 27B: 15.0 to 8.8 us at 1 sequence, 118 to 77 at
// 64, 447 to 256 at 256):
// - Thread t holds columns 8 (t / 16).. + 7 of rows 8 (t % 16).. + 7, so a
//   column group's 16 row groups sit in one wave (sums over rows are
//   shuffles) and the state moves in 16-byte loads.
// - Every load is issued before any is used. A branch between a load and its
//   use (the state's format, freshness) made the compiler wait for each load
//   in turn: 16 dependent memory round trips, 5 us of a block's 12.
// - q's and k's norms share one reduction.
// Splitting a head's columns across blocks (as FLA does) gained at most
// 0.5 us at 1 sequence, lost beyond, and needs the gated norm in a separate
// kernel; handing the row to the last block instead needs agent-scope
// fences, which on gfx942 write back and invalidate the L2.
__device__ __forceinline__ void gdn_one_w(int s, int h, const bf16* __restrict__ conv,
                                          const bf16* __restrict__ proj, int pstride, int C, int Hk, int Hv,
                                          const float* __restrict__ neg_a, const float* __restrict__ dt_bias,
                                          const float* __restrict__ norm_w, void* __restrict__ state, int state_bf16,
                                          const int* __restrict__ cu, const int* __restrict__ slots,
                                          const int* __restrict__ fresh, bf16* __restrict__ out, float eps, int tiled,
                                          uint8_t* fq, void* fs, int fmode, int frows, const int* __restrict__ outs,
                                          int spec) {
    constexpr int CW = 8, RG = 16, RW = GDN_DIM / RG;
    __shared__ float qs[GDN_DIM], ks[GDN_DIM], sh[16], ys[GDN_DIM];
    const Fp8Out fo{fq, fs, fmode, frows};
    const int tid = threadIdx.x, rg = tid % RG, c0 = CW * (tid / RG), r0 = RW * rg;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float r[RW][CW];
    const bool is_fresh = fresh[s] != 0;
    auto load = [&](size_t b) {
        const size_t at = b + (size_t)r0 * GDN_DIM + c0;
        if (is_fresh) {
#pragma unroll
            for (int i = 0; i < RW; i++)
#pragma unroll
                for (int c = 0; c < CW; c++) r[i][c] = 0.f;
        } else if (state_bf16) {
            uint4 u[RW];
#pragma unroll
            for (int i = 0; i < RW; i++) u[i] = *reinterpret_cast<const uint4*>((const bf16*)state + at + i * GDN_DIM);
#pragma unroll
            for (int i = 0; i < RW; i++) unpack8(u[i], r[i]);
        } else {
            float4 f[RW][2];
#pragma unroll
            for (int i = 0; i < RW; i++)
#pragma unroll
                for (int c4 = 0; c4 < 2; c4++)
                    f[i][c4] = *reinterpret_cast<const float4*>((const float*)state + at + i * GDN_DIM + 4 * c4);
#pragma unroll
            for (int i = 0; i < RW; i++)
#pragma unroll
                for (int c4 = 0; c4 < 2; c4++) {
                    float* d = &r[i][4 * c4];
                    d[0] = f[i][c4].x, d[1] = f[i][c4].y, d[2] = f[i][c4].z, d[3] = f[i][c4].w;
                }
        }
    };
    auto store = [&](size_t b) {
        const size_t at = b + (size_t)r0 * GDN_DIM + c0;
        if (state_bf16) {
#pragma unroll
            for (int i = 0; i < RW; i++) *reinterpret_cast<uint4*>((bf16*)state + at + i * GDN_DIM) = pack8(r[i]);
        } else {
#pragma unroll
            for (int i = 0; i < RW; i++)
#pragma unroll
                for (int c4 = 0; c4 < CW; c4 += 4)
                    *reinterpret_cast<float4*>((float*)state + at + i * GDN_DIM + c4) =
                        make_float4(r[i][c4], r[i][c4 + 1], r[i][c4 + 2], r[i][c4 + 3]);
        }
    };
    load(base);
    auto rows = [](float v) {
#pragma unroll
        for (int o = 1; o < RG; o <<= 1) v += __shfl_xor(v, o, 64);
        return v;
    };
    const bool keep = is_spec(outs, spec, s);
    const bool round_qk = ((t1 - t0) > 1 && !keep) || is_fresh;
    const float na = neg_a[h], db = dt_bias[h];
    const float qscale = rsqrtf((float)GDN_DIM);
    for (int t = t0; t < t1; t++) {
        const bf16* row = conv + (size_t)t * C;
        const bf16* prow = proj + (size_t)t * pstride + C;
        // The token's inputs, all issued together: q and k (threads < 128
        // take one element each; the rest load element 0 and ignore it), v,
        // the gate's z, b and a.
        const int qi = tid < GDN_DIM ? tid : 0;
        const bf16 qb = row[kh * GDN_DIM + qi], kb = row[Hk * GDN_DIM + kh * GDN_DIM + qi];
        const uint4 vu = *reinterpret_cast<const uint4*>(row + 2 * Hk * GDN_DIM + h * GDN_DIM + c0);
        const uint4 zu = *reinterpret_cast<const uint4*>(prow + h * GDN_DIM + c0);
        const float b = b2f(prow[Hv * GDN_DIM + h]);
        const float a = b2f(prow[Hv * GDN_DIM + Hv + h]);
        float q = tid < GDN_DIM ? b2f(qb) : 0.f, k = tid < GDN_DIM ? b2f(kb) : 0.f;
        const float2 qk = block_sum2(q * q, k * k, sh);
        if (tid < GDN_DIM) {
            q = q / sqrtf(qk.x + 1e-6f);
            k = k / sqrtf(qk.y + 1e-6f);
            if (round_qk) {
                q = rbf(q);
                k = rbf(k);
            }
            qs[tid] = q * qscale;
            ks[tid] = k;
        }
        const float beta = 1.f / (1.f + expf(-b));
        const float x = a + db;
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        const float decay = expf(na * sp);
        float v[CW];
        unpack8(vu, v);
        __syncthreads();
        float kv[CW];
#pragma unroll
        for (int c = 0; c < CW; c++) kv[c] = 0.f;
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const float kk_ = ks[r0 + i];
#pragma unroll
            for (int c = 0; c < CW; c++) {
                r[i][c] *= decay;
                kv[c] += kk_ * r[i][c];
            }
        }
        float delta[CW], o[CW];
#pragma unroll
        for (int c = 0; c < CW; c++) {
            delta[c] = (v[c] - rows(kv[c])) * beta;
            o[c] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const float kk_ = ks[r0 + i], qq_ = qs[r0 + i];
#pragma unroll
            for (int c = 0; c < CW; c++) {
                r[i][c] += kk_ * delta[c];
                o[c] += qq_ * r[i][c];
            }
        }
        float ob[CW], ss = 0.f;
#pragma unroll
        for (int c = 0; c < CW; c++) {
            ob[c] = rbf(rows(o[c]));
            ss += ob[c] * ob[c];
        }
        const float var = block_sum2(rg == 0 ? ss : 0.f, 0.f, sh).x / GDN_DIM;
        if (rg == 0) {
            const float inv = 1.f / sqrtf(var + eps);
            float z[CW], y[CW];
            unpack8(zu, z);
            const float4 w0 = *reinterpret_cast<const float4*>(norm_w + c0);
            const float4 w1 = *reinterpret_cast<const float4*>(norm_w + c0 + 4);
            const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int c = 0; c < CW; c++) {
                y[c] = ob[c] * inv * w[c] * silu(z[c]);
                ys[c0 + c] = y[c];
            }
            *reinterpret_cast<uint4*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + c0) = pack8(y);
        }
        if (fmode) {
            __syncthreads();
            if (tid < 32) {
                float vv[8];
#pragma unroll
                for (int i = 0; i < 8; i++) vv[i] = tid < 16 ? ys[8 * tid + i] : 0.f;
                quant8(vv, t, h * GDN_DIM + 8 * (tid & 15), Hv * GDN_DIM, fo, tid < 16);
            }
        }
        __syncthreads();  // qs and ks are rewritten by the next token
        if (keep) store(((size_t)outs[(size_t)s * spec + (t - t0)] * Hv + h) * GDN_DIM * GDN_DIM);
    }
    if (!keep) store(base);
}
