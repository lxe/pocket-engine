// LoRA on AMD: lora_shrink and lora_expand (lora.cuh) without mma.sync, which
// AMD emulates with shuffles. Same grids, arguments and rounding (S goes to
// bf16 before B, as the originals' fragments take it). The work is small (a
// tile is at most 16 tokens, ranks are tens), so plain FMAs from loads issued
// ahead of their use.

// Grid: (max tiles, ceil(rank / 32), ceil(K / 256)), 128 threads: column
// n0 + (tid % 32) of S for tokens tid / 32, + 4, + 8, + 12 of the tile, over
// this block's 256 inputs, added into S.
extern "C" __global__ void __launch_bounds__(128) lora_shrink_amd(const bf16* __restrict__ X, int ldx, int K,
                                                                  const int* __restrict__ tiles,
                                                                  const LoraDesc* __restrict__ table,
                                                                  float* __restrict__ S, int lds) {
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int t0 = tiles[1 + 3 * tile], len = tiles[2 + 3 * tile], slot = tiles[3 + 3 * tile];
    const LoraDesc& d = table[slot];
    const int n0 = blockIdx.y * 32;
    if (n0 >= d.rank) return;
    const int c = threadIdx.x & 31, g = threadIdx.x >> 5, n = n0 + c;
    const bf16* a = d.a + (size_t)min(n, d.rank - 1) * K;
    const bf16* x[4];
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = X + (size_t)(t0 + min(g + 4 * j, len - 1)) * ldx;
    const int k0 = blockIdx.z * LORA_KCHUNK, k1 = min(K, k0 + LORA_KCHUNK);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int k = k0; k < k1; k += 8) {
        const uint4 av = *reinterpret_cast<const uint4*>(a + k);
        uint4 xv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) xv[j] = *reinterpret_cast<const uint4*>(x[j] + k);
        float af[8];
        unpack8(av, af);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float xf[8];
            unpack8(xv[j], xf);
#pragma unroll
            for (int e = 0; e < 8; e++) acc[j] = fmaf(xf[e], af[e], acc[j]);
        }
    }
    if (n >= d.rank) return;
#pragma unroll
    for (int j = 0; j < 4; j++)
        if (g + 4 * j < len) atomicAdd(&S[(size_t)(t0 + g + 4 * j) * lds + n], acc[j]);
}

// Grid: (max tiles, ceil(max part rows / 128), parts), 128 threads: output
// column blockIdx.y * 128 + tid of part blockIdx.z for every token of the
// tile; zeroes `next` as lora_expand does.
extern "C" __global__ void __launch_bounds__(128) lora_expand_amd(const float* __restrict__ S, int lds,
                                                                  float* __restrict__ next,
                                                                  const int* __restrict__ tiles,
                                                                  const LoraDesc* __restrict__ table, void* Y, int ldy,
                                                                  int f32_out) {
    __shared__ float ss[16][64 + 1];
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int t0 = tiles[1 + 3 * tile], len = tiles[2 + 3 * tile], slot = tiles[3 + 3 * tile];
    if (blockIdx.y == 0 && blockIdx.z == 0)
        for (int i = threadIdx.x; i < len * lds; i += 128) next[(size_t)t0 * lds + i] = 0.f;
    const LoraDesc& d = table[slot];
    const int p = blockIdx.z, r = d.r[p];
    if (r == 0) return;
    const int rows = d.rows[p], col = blockIdx.y * 128 + threadIdx.x;
    if (blockIdx.y * 128 >= rows) return;
    // The tile's S for this part, rounded to bf16 (ranks of up to 64; more
    // in passes of 64).
    const bf16* b = d.b[p] + (size_t)min(col, rows - 1) * r;
    float acc[16];
#pragma unroll
    for (int t = 0; t < 16; t++) acc[t] = 0.f;
    for (int k0 = 0; k0 < r; k0 += 64) {
        const int kn = min(64, r - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < len * kn; i += 128) {
            const int t = i / kn, k = i % kn;
            ss[t][k] = rbf(S[(size_t)(t0 + t) * lds + d.r_off[p] + k0 + k]);
        }
        __syncthreads();
        for (int k = 0; k < kn; k += 8) {
            float bf[8];
            unpack8(*reinterpret_cast<const uint4*>(b + k0 + k), bf);
#pragma unroll
            for (int t = 0; t < 16; t++) {
                if (t >= len) break;
#pragma unroll
                for (int e = 0; e < 8; e++) acc[t] = fmaf(ss[t][k + e], bf[e], acc[t]);
            }
        }
    }
    if (col >= rows) return;
#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t >= len) break;
        const size_t at = (size_t)(t0 + t) * ldy + d.off[p] + col;
        if (f32_out) ((float*)Y)[at] += acc[t];
        else ((bf16*)Y)[at] = f2b(b2f(((bf16*)Y)[at]) + acc[t]);
    }
}

// lora_shrink for decoding's tiles of a few tokens (grid and arguments as
// lora_shrink): there, lora_shrink_amd keeps 32 of its 128 threads busy for
// one token and waits on a load per step (10.7 us a call on 9B). Here wave w
// takes columns n0 + 16 w .. + 15, two at a time (half-waves), its lanes
// splitting the block's 256 inputs 8 each: all 8 of a lane's A loads issue
// first, then each token's inputs, and sums over the half-wave are shuffles.
extern "C" __global__ void __launch_bounds__(128) lora_shrink_dec(const bf16* __restrict__ X, int ldx, int K,
                                                                  const int* __restrict__ tiles,
                                                                  const LoraDesc* __restrict__ table,
                                                                  float* __restrict__ S, int lds) {
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int t0 = tiles[1 + 3 * tile], len = tiles[2 + 3 * tile], slot = tiles[3 + 3 * tile];
    const LoraDesc& d = table[slot];
    const int n0 = blockIdx.y * 32;
    if (n0 >= d.rank) return;
    const int w = threadIdx.x >> 6, lane = threadIdx.x & 63, half = lane >> 5;
    const int k = blockIdx.z * LORA_KCHUNK + 8 * (lane & 31);
    if (k >= K) return;  // whole half-waves: K is a multiple of 256
    uint4 av[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = min(n0 + 16 * w + 2 * j + half, d.rank - 1);
        av[j] = *reinterpret_cast<const uint4*>(d.a + (size_t)n * K + k);
    }
    float af[8][8];
#pragma unroll
    for (int j = 0; j < 8; j++) unpack8(av[j], af[j]);
    for (int t = 0; t < len; t++) {
        float xf[8];
        unpack8(*reinterpret_cast<const uint4*>(X + (size_t)(t0 + t) * ldx + k), xf);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) s = fmaf(xf[e], af[j][e], s);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 64);
            const int n = n0 + 16 * w + 2 * j + half;
            if ((lane & 31) == 0 && n < d.rank) atomicAdd(&S[(size_t)(t0 + t) * lds + n], s);
        }
    }
}
