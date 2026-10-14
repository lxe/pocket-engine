// Flash attention for prompts on AMD: see attn_flash_group in qwen35.cu.
#pragma once

// Flash attention for prompts on AMD's matrix cores (MFMA 16x16x16 bf16,
// 64-lane waves): wave w takes 16 queries of query head w % G, tile w / G,
// and computes scores transposed, S^T = K Q^T, so a lane's accumulators
// (keys 4 (lane / 16) .. + 3 of query lane % 16) are already the B operand
// of O^T += V^T P^T and each lane rescales only its own query. Queries stay
// in registers; keys and values (transposed) are staged per tile of AM_BN.
#define AM_BN 32
#define AM_VS (AM_BN + 8)
template <int G, int NT>
__device__ __forceinline__ void attn_mfma(ATTN_FLASH_ARGS) {
    extern __shared__ __align__(16) unsigned char fsm[];
    bf16* sK = (bf16*)fsm;            // [AM_BN][FA_STR]
    bf16* sVt = sK + AM_BN * FA_STR;  // [HEAD_DIM][AM_VS]
    constexpr int BT = 16 * NT, T = NT * G * 64;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63, lr = lane & 15, lq = lane >> 4;
    const int kvh = blockIdx.y, h = kvh * G + wave % G, qt = wave / G;
    int s = -1, tile = blockIdx.x;
    for (int i = 0; i < seqs; i++) {
        const int n = cu[i + 1] - cu[i];
        if (n <= 1 || is_spec(outs, spec, i)) continue;
        const int tiles = (n + BT - 1) / BT;
        if (tile < tiles) {
            s = i;
            break;
        }
        tile -= tiles;
    }
    if (s < 0) return;
    const int tb = cu[s] + tile * BT, nb = min(BT, cu[s + 1] - tb);
    const int L = pos[tb] + nb;
    const int t0 = tb + qt * 16, n = max(0, min(16, nb - qt * 16)), p0 = pos[tb] + qt * 16;
    const int* bt = block_table + (size_t)s * max_blocks;
    // Query lr's dimensions 16 kd + 4 lq .. + 3, scaled into log2 units.
    pe_s4 qb[HEAD_DIM / 16];
    {
        const bool ok = lr < n;
        const bf16* qrow = q + ((size_t)(t0 + (ok ? lr : 0)) * Hq + h) * HEAD_DIM;
#pragma unroll
        for (int kd = 0; kd < HEAD_DIM / 16; kd++) {
            const uint2 raw = ok ? *reinterpret_cast<const uint2*>(qrow + 16 * kd + 4 * lq) : make_uint2(0, 0);
            const float f = scale * LOG2E;
            const uint2 v = pack4(__uint_as_float(raw.x << 16) * f, __uint_as_float(raw.x & 0xffff0000u) * f,
                                  __uint_as_float(raw.y << 16) * f, __uint_as_float(raw.y & 0xffff0000u) * f);
            qb[kd] = *reinterpret_cast<const pe_s4*>(&v);
        }
    }
    pe_f4 o[HEAD_DIM / 16];
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; dt++) o[dt] = pe_f4{0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;
    const int last = p0 + n - 1;
    for (int k0 = 0; k0 < L; k0 += AM_BN) {
        __syncthreads();  // the previous tile is consumed
        for (int e = tid; e < AM_BN * (HEAD_DIM / 8); e += T) {
            const int j = e / (HEAD_DIM / 8), c = (e % (HEAD_DIM / 8)) * 8, kp = k0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (kp < L) {
                const size_t slot = (size_t)bt[kp / block_size] * block_size + kp % block_size;
                const size_t at = (slot * Hkv + kvh) * HEAD_DIM + c;
                kv = *reinterpret_cast<const uint4*>((const bf16*)kcache + at);
                vv = *reinterpret_cast<const uint4*>((const bf16*)vcache + at);
            }
            *reinterpret_cast<uint4*>(sK + j * FA_STR + c) = kv;
            const bf16* vb = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
            for (int d = 0; d < 8; d++) sVt[(c + d) * AM_VS + j] = vb[d];
        }
        __syncthreads();
        if (k0 > last) continue;  // keys past all of this wave's queries add nothing
        pe_f4 sc[AM_BN / 16];
#pragma unroll
        for (int kt = 0; kt < AM_BN / 16; kt++) {
            sc[kt] = pe_f4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kd = 0; kd < HEAD_DIM / 16; kd++)
                sc[kt] = mfma16(frag4(sK, FA_STR, 16 * kt + lr, 16 * kd + 4 * lq), qb[kd], sc[kt]);
        }
        if (k0 + AM_BN - 1 > p0 || k0 + AM_BN > L) {
#pragma unroll
            for (int kt = 0; kt < AM_BN / 16; kt++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int kp = k0 + 16 * kt + 4 * lq + j;
                    if (kp > p0 + lr || kp >= L) sc[kt][j] = -INFINITY;
                }
        }
        float mx = m;
#pragma unroll
        for (int kt = 0; kt < AM_BN / 16; kt++)
#pragma unroll
            for (int j = 0; j < 4; j++) mx = fmaxf(mx, sc[kt][j]);
        mx = fmaxf(mx, __shfl_xor(mx, 16, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 32, 64));
        // A query with no key yet keeps -inf; it subtracts 0 so exp2 gives 0.
        const float b = mx == -INFINITY ? 0.f : mx, c = fast_exp2(m - b);
        float sum = 0.f;
#pragma unroll
        for (int kt = 0; kt < AM_BN / 16; kt++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sc[kt][j] = fast_exp2(sc[kt][j] - b);
                sum += sc[kt][j];
            }
        sum += __shfl_xor(sum, 16, 64);
        sum += __shfl_xor(sum, 32, 64);
        l = l * c + sum;
        m = mx;
#pragma unroll
        for (int dt = 0; dt < HEAD_DIM / 16; dt++) o[dt] *= c;
#pragma unroll
        for (int kt = 0; kt < AM_BN / 16; kt++) {
            const uint2 pv = pack4(sc[kt][0], sc[kt][1], sc[kt][2], sc[kt][3]);
            const pe_s4 pb = *reinterpret_cast<const pe_s4*>(&pv);
#pragma unroll
            for (int dt = 0; dt < HEAD_DIM / 16; dt++)
                o[dt] = mfma16(frag4(sVt, AM_VS, 16 * dt + lr, 16 * kt + 4 * lq), pb, o[dt]);
        }
    }
    if (lr >= n) return;
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const size_t t = (size_t)t0 + lr;
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; dt++) {
        const int d = 16 * dt + 4 * lq;
        const uint2 graw = *reinterpret_cast<const uint2*>(qkv + t * qstride + h * 2 * HEAD_DIM + HEAD_DIM + d);
        const float gate[4] = {__uint_as_float(graw.x << 16), __uint_as_float(graw.x & 0xffff0000u),
                               __uint_as_float(graw.y << 16), __uint_as_float(graw.y & 0xffff0000u)};
        float y[4];
#pragma unroll
        for (int j = 0; j < 4; j++) y[j] = rbf(o[dt][j] * inv) / (1.f + expf(-gate[j]));
        *reinterpret_cast<uint2*>(out + (t * Hq + h) * HEAD_DIM + d) = pack4(y[0], y[1], y[2], y[3]);
    }
}
#define ATTN_MFMA(G, NT)                                                                                       \
    extern "C" __global__ void __launch_bounds__(G * 64 * NT) attn_mfma_g##G##_t##NT(ATTN_FLASH_ARGS) {         \
        attn_mfma<G, NT>(ATTN_FLASH_PASS);                                                                     \
    }
ATTN_MFMA(4, 2)
ATTN_MFMA(6, 2)
