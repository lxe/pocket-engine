// The gated delta rule for decoding on AMD (backend/hip.cuh): see gdn_one in qwen35.cu.
#pragma once

// gdn_one for AMD: the same math with each thread holding 4 adjacent
// columns of 16 rows (thread t: columns 4 (t / 8).., rows 16 (t % 8)..), so
// the state moves in 8- or 16-byte loads instead of 2-byte ones; a column's
// sums over rows reduce across the 8 lanes of its group.
__device__ __forceinline__ void gdn_one_w(int s, int h, const bf16* __restrict__ conv,
                                          const bf16* __restrict__ proj, int pstride, int C, int Hk, int Hv,
                                          const float* __restrict__ neg_a, const float* __restrict__ dt_bias,
                                          const float* __restrict__ norm_w, void* __restrict__ state, int state_bf16,
                                          const int* __restrict__ cu, const int* __restrict__ slots,
                                          const int* __restrict__ fresh, bf16* __restrict__ out, float eps, int tiled,
                                          uint8_t* fq, void* fs, int fmode, int frows, const int* __restrict__ outs,
                                          int spec) {
    constexpr int RW = 16, CW = 4;
    __shared__ float qs[GDN_DIM], ks[GDN_DIM], sh[32], ys[GDN_DIM];
    const Fp8Out fo{fq, fs, fmode, frows};
    const int tid = threadIdx.x, rg = tid & 7, c0 = CW * (tid >> 3), r0 = RW * rg;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    const int kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    float r[RW][CW];
    const bool is_fresh = fresh[s] != 0;
    auto load = [&](size_t b) {
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const size_t at = b + (size_t)(r0 + i) * GDN_DIM + c0;
            if (is_fresh) {
                r[i][0] = r[i][1] = r[i][2] = r[i][3] = 0.f;
            } else if (state_bf16) {
                const uint2 u = *reinterpret_cast<const uint2*>((const bf16*)state + at);
                r[i][0] = __uint_as_float(u.x << 16), r[i][1] = __uint_as_float(u.x & 0xffff0000u);
                r[i][2] = __uint_as_float(u.y << 16), r[i][3] = __uint_as_float(u.y & 0xffff0000u);
            } else {
                const float4 f = *reinterpret_cast<const float4*>((const float*)state + at);
                r[i][0] = f.x, r[i][1] = f.y, r[i][2] = f.z, r[i][3] = f.w;
            }
        }
    };
    auto store = [&](size_t b) {
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const size_t at = b + (size_t)(r0 + i) * GDN_DIM + c0;
            if (state_bf16) {
                __nv_bfloat162 lo = __floats2bfloat162_rn(r[i][0], r[i][1]), hi = __floats2bfloat162_rn(r[i][2], r[i][3]);
                *reinterpret_cast<uint2*>((bf16*)state + at) =
                    make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
            } else {
                *reinterpret_cast<float4*>((float*)state + at) = make_float4(r[i][0], r[i][1], r[i][2], r[i][3]);
            }
        }
    };
    load(base);
    auto rows8 = [](float v) {
        v += __shfl_xor(v, 1, 32);
        v += __shfl_xor(v, 2, 32);
        return v + __shfl_xor(v, 4, 32);
    };
    const bool keep = is_spec(outs, spec, s);
    const bool round_qk = ((t1 - t0) > 1 && !keep) || is_fresh;
    const float na = neg_a[h], db = dt_bias[h];
    const float qscale = rsqrtf((float)GDN_DIM);
    for (int t = t0; t < t1; t++) {
        const bf16* row = conv + (size_t)t * C;
        const bf16* prow = proj + (size_t)t * pstride + C;
        float q = 0.f, k = 0.f;
        if (tid < GDN_DIM) {
            q = b2f(row[kh * GDN_DIM + tid]);
            k = b2f(row[Hk * GDN_DIM + kh * GDN_DIM + tid]);
        }
        const float b = b2f(prow[Hv * GDN_DIM + h]);
        const float a = b2f(prow[Hv * GDN_DIM + Hv + h]);
        const float qq = block_sum(q * q, sh);
        const float kk = block_sum(k * k, sh);
        if (tid < GDN_DIM) {
            q = q / sqrtf(qq + 1e-6f);
            k = k / sqrtf(kk + 1e-6f);
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
        __syncthreads();
        float kv[CW] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const float kk_ = ks[r0 + i];
#pragma unroll
            for (int c = 0; c < CW; c++) {
                r[i][c] *= decay;
                kv[c] += kk_ * r[i][c];
            }
        }
        float delta[CW], o[CW] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < CW; c++) {
            const float v = b2f(row[2 * Hk * GDN_DIM + h * GDN_DIM + c0 + c]);
            delta[c] = (v - rows8(kv[c])) * beta;
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
            ob[c] = rbf(rows8(o[c]));
            ss += ob[c] * ob[c];
        }
        const float var = block_sum(rg == 0 ? ss : 0.f, sh) / GDN_DIM;
        if (rg == 0) {
            const float inv = 1.f / sqrtf(var + eps);
            float y[CW];
#pragma unroll
            for (int c = 0; c < CW; c++) {
                const int j = c0 + c;
                y[c] = ob[c] * inv * norm_w[j] * silu(b2f(prow[h * GDN_DIM + j]));
                ys[j] = y[c];
            }
            __nv_bfloat162 lo = __floats2bfloat162_rn(y[0], y[1]), hi = __floats2bfloat162_rn(y[2], y[3]);
            *reinterpret_cast<uint2*>(out + (size_t)t * Hv * GDN_DIM + h * GDN_DIM + c0) =
                make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
        }
        if (fmode) {
            __syncthreads();
            if (tid < 32) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = tid < 16 ? ys[8 * tid + i] : 0.f;
                quant8(v, t, h * GDN_DIM + 8 * (tid & 15), Hv * GDN_DIM, fo, tid < 16);
            }
            __syncthreads();
        }
        if (keep) store(((size_t)outs[(size_t)s * spec + (t - t0)] * Hv + h) * GDN_DIM * GDN_DIM);
    }
    if (!keep) store(base);
}
