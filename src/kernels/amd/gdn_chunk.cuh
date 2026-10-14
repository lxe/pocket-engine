// The chunked gated delta rule for prompts on AMD: see gdn_scan_cols in qwen35.cu.
#pragma once

// gdn_scan on AMD's matrix cores (MFMA 16x16x16 bf16, 64-lane waves): the
// same products as gdn_scan_cols, tiled per wave. A lane holds A[m][k..k+3]
// (m = lane % 16, k = 4 (lane / 16)), B[k..k+3][n] and C[m..m+3][n], so both
// operands want k contiguous: K, D, Dk and the state's bf16 copy sit
// transposed in shared memory. Wave w keeps state rows 32w .. 32w + 31 of
// the block's columns in registers.
#define GKT (GC + 8)  // bf16 stride of token-contiguous rows
template <int NS>
__host__ __device__ constexpr int gdn_mfma_smem() {
    return (2 * GC * GSTR + GDN_DIM * GKT + GC * (GDN_DIM / NS + 8) + GC * GPSTR + 2 * (GDN_DIM / NS) * GKT +
            (GDN_DIM / NS) * GSTR) * 2 + 3 * GC * 4;
}
template <int NS>
__device__ __forceinline__ void gdn_scan_mfma(GDN_ARGS, const bf16* __restrict__ Ui, const bf16* __restrict__ Wi,
                                              const float* __restrict__ Nqi, const float* __restrict__ Nki,
                                              const bf16* __restrict__ Pi, const float* __restrict__ Gi,
                                              const int* __restrict__ snaps) {
    constexpr int VW = GDN_DIM / NS, CT = VW / 16, CW = CT / 2, VS = VW + 8;
    static_assert(CT >= 2, "two waves share a row tile's columns");
    extern __shared__ __align__(16) unsigned char ssm[];
    bf16* sW = (bf16*)ssm;                // [GC][GSTR] W
    bf16* sQ = sW + GC * GSTR;            // [GC][GSTR] q (unnormalized)
    bf16* sKt = sQ + GC * GSTR;           // [GDN_DIM][GKT] k, transposed
    bf16* sU = sKt + GDN_DIM * GKT;       // [GC][VS] U, this block's columns
    bf16* sP = sU + GC * VS;              // [GC][GPSTR] P
    bf16* sDt = sP + GC * GPSTR;          // [VW][GKT] D, transposed
    bf16* sDkt = sDt + VW * GKT;          // [VW][GKT] D rows times exp(G_last - G_i) |k_i|^-1, transposed
    bf16* sSt = sDkt + VW * GKT;          // [VW][GSTR] the state's columns, transposed, bf16
    float* sG = (float*)(sSt + VW * GSTR);  // [GC] G, then Nq, Nk
    float* sNq = sG + GC;
    float* sNk = sNq + GC;
    const int s = gridDim.x - 1 - blockIdx.x, h = blockIdx.y, vz = blockIdx.z, tid = threadIdx.x;
    const int wave = tid >> 6, lane = tid & 63, lr = lane & 15, lq = lane >> 4;
    const int t0 = cu[s], t1 = cu[s + 1];
    if (t0 == t1) return;
    if (t1 - t0 == 1 || is_spec(outs, spec, s)) {
        if (vz == 0) gdn_one(s, h, GDN_PASS);
        return;
    }
    const int v0 = vz * VW, kh = tiled ? h % Hk : h / (Hv / Hk);
    const size_t base = ((size_t)slots[s] * Hv + h) * GDN_DIM * GDN_DIM;
    const bool is_fresh = fresh[s] != 0;
    const float qscale = rsqrtf((float)GDN_DIM);
    // State tile (rt, ct): rows 32 wave + 16 rt + 4 lq + j, columns 16 ct + lr.
    pe_f4 st[2][CT];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int ct = 0; ct < CT; ct++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const size_t at = base + (size_t)(32 * wave + 16 * rt + 4 * lq + j) * GDN_DIM + v0 + 16 * ct + lr;
                st[rt][ct][j] = is_fresh ? 0.f : state_bf16 ? b2f(((const bf16*)state)[at]) : ((const float*)state)[at];
            }
    auto store_state_smem = [&]() {
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < CT; ct++)
                *reinterpret_cast<uint2*>(sSt + (16 * ct + lr) * GSTR + 32 * wave + 16 * rt + 4 * lq) =
                    pack4(st[rt][ct][0], st[rt][ct][1], st[rt][ct][2], st[rt][ct][3]);
    };
    auto store_state = [&](int slot) {
        const size_t b = ((size_t)slot * Hv + h) * GDN_DIM * GDN_DIM;
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < CT; ct++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const size_t at = b + (size_t)(32 * wave + 16 * rt + 4 * lq + j) * GDN_DIM + v0 + 16 * ct + lr;
                    if (state_bf16) ((bf16*)state)[at] = f2b(st[rt][ct][j]);
                    else ((float*)state)[at] = st[rt][ct][j];
                }
    };
    store_state_smem();
    bool zero_state = is_fresh;
    // This wave's output tiles: token rows 16 (wave & 1) .., columns CW tiles from (wave >> 1) CW.
    const int om0 = 16 * (wave & 1), oc0 = (wave >> 1) * CW;
    for (int c0 = t0; c0 < t1; c0 += GC) {
        const int n = min(GC, t1 - c0);
        for (int e = tid; e < GC * (GDN_DIM / 8); e += 2 * GDN_DIM) {
            const int i = e / (GDN_DIM / 8), c = (e % (GDN_DIM / 8)) * 8, r = c0 + min(i, n - 1);
            const bool ok = i < n;
            const size_t at = ((size_t)r * Hv + h) * GDN_DIM + c;
            const bf16* row = conv + (size_t)r * C + kh * GDN_DIM + c;
            const uint4 z = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(sW + i * GSTR + c) = ok ? *reinterpret_cast<const uint4*>(Wi + at) : z;
            *reinterpret_cast<uint4*>(sQ + i * GSTR + c) = ok ? *reinterpret_cast<const uint4*>(row) : z;
            const uint4 kv = ok ? *reinterpret_cast<const uint4*>(row + Hk * GDN_DIM) : z;
            const bf16* kb = reinterpret_cast<const bf16*>(&kv);
#pragma unroll
            for (int d = 0; d < 8; d++) sKt[(c + d) * GKT + i] = kb[d];
            if (c < VW) *reinterpret_cast<uint4*>(sU + i * VS + c) = ok ? *reinterpret_cast<const uint4*>(Ui + at + v0) : z;
        }
        for (int e = tid; e < GC * (GC / 8); e += 2 * GDN_DIM) {
            const int i = e / (GC / 8), c = (e % (GC / 8)) * 8;
            *reinterpret_cast<uint4*>(sP + i * GPSTR + c) =
                i < n ? *reinterpret_cast<const uint4*>(Pi + ((size_t)(c0 + i) * Hv + h) * GC + c) : make_uint4(0, 0, 0, 0);
        }
        if (tid < GC) {
            const size_t at = (size_t)(c0 + min(tid, n - 1)) * Hv + h;
            sG[tid] = Gi[at];
            sNq[tid] = Nqi[at];
            sNk[tid] = Nki[at];
        }
        __syncthreads();
        const float glast = sG[GC - 1];
        // D = U - W S0 and Q S0 for this wave's tiles.
        pe_f4 accq[CW];
#pragma unroll
        for (int cw = 0; cw < CW; cw++) {
            const int ct = oc0 + cw;
            pe_f4 acc = {0.f, 0.f, 0.f, 0.f};
            accq[cw] = acc;
            if (!zero_state) {
#pragma unroll
                for (int k0 = 0; k0 < GDN_DIM; k0 += 16) {
                    const pe_s4 b = frag4(sSt, GSTR, 16 * ct + lr, k0 + 4 * lq);
                    acc = mfma16(frag4(sW, GSTR, om0 + lr, k0 + 4 * lq), b, acc);
                    accq[cw] = mfma16(frag4(sQ, GSTR, om0 + lr, k0 + 4 * lq), b, accq[cw]);
                }
            }
            const int c = 16 * ct + lr, i0 = om0 + 4 * lq;
            float d[4], dk[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int i = i0 + j;
                d[j] = b2f(sU[i * VS + c]) - acc[j];
                dk[j] = d[j] * expf(glast - sG[i]) * sNk[i];
                accq[cw][j] *= qscale * expf(sG[i]) * sNq[i];
            }
            *reinterpret_cast<uint2*>(sDt + c * GKT + i0) = pack4(d[0], d[1], d[2], d[3]);
            *reinterpret_cast<uint2*>(sDkt + c * GKT + i0) = pack4(dk[0], dk[1], dk[2], dk[3]);
        }
        __syncthreads();
        // O = exp(G) qscale Q S0 + P D, written unnormalized.
#pragma unroll
        for (int cw = 0; cw < CW; cw++) {
            const int ct = oc0 + cw;
#pragma unroll
            for (int k0 = 0; k0 < GC; k0 += 16)
                accq[cw] = mfma16(frag4(sP, GPSTR, om0 + lr, k0 + 4 * lq), frag4(sDt, GKT, 16 * ct + lr, k0 + 4 * lq),
                                  accq[cw]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int i = om0 + 4 * lq + j;
                if (i < n) out[(size_t)(c0 + i) * Hv * GDN_DIM + h * GDN_DIM + v0 + 16 * ct + lr] = f2b(accq[cw][j]);
            }
        }
        // S = exp(G_last) S0 + K^T Dk.
        const float decay = expf(glast);
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int ct = 0; ct < CT; ct++) {
                st[rt][ct] *= decay;
#pragma unroll
                for (int k0 = 0; k0 < GC; k0 += 16)
                    st[rt][ct] = mfma16(frag4(sKt, GKT, 32 * wave + 16 * rt + lr, k0 + 4 * lq),
                                        frag4(sDkt, GKT, 16 * ct + lr, k0 + 4 * lq), st[rt][ct]);
            }
        if (snaps)
            for (int j = 0; j < 2; j++)
                if (snaps[4 * s + 2 * j] == c0 + n - t0) store_state(snaps[4 * s + 2 * j + 1]);
        __syncthreads();
        store_state_smem();
        zero_state = false;
        __syncthreads();
    }
    store_state(slots[s]);
}
