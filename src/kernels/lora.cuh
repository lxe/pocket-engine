// LoRA: per step, tiles of up to 16 tokens that share an adapter, and per
// linear layer a table of descriptors by adapter slot (src/lora.rs builds
// both). lora_shrink computes S = x A^T for all of the layer's parts at once
// (A stacks every targeted part's factor); lora_expand adds S B^T into each
// part's output columns. Ranks are multiples of 16; the scale is folded into B.

struct LoraDesc {
    const bf16* a;     // [rank, K]
    const bf16* b[4];  // per part [rows, r]
    int rank;          // total, 0 when the adapter leaves this layer alone
    int r[4];          // per part, 0 when untargeted
    int r_off[4];      // per part, its first column of S
    int off[4];        // per part, its first output column
    int rows[4];
    int pad_;
};

// tiles: [count, (start, len, slot) x count]. Grid: (max tiles, ceil(rank /
// 32), ceil(K / 256)): each block adds 32 columns of S for one tile over 256
// inputs, so a decode step's few tokens still spread over the GPU. S must be
// zero where it accumulates: see lora_expand.
#define LORA_KCHUNK 256
extern "C" __global__ void __launch_bounds__(128) lora_shrink(const bf16* __restrict__ X, int ldx, int K,
                                                              const int* __restrict__ tiles,
                                                              const LoraDesc* __restrict__ table,
                                                              float* __restrict__ S, int lds) {
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int t0 = tiles[1 + 3 * tile], len = tiles[2 + 3 * tile], slot = tiles[3 + 3 * tile];
    const LoraDesc& d = table[slot];
    const int n0 = blockIdx.y * 32;
    if (n0 >= d.rank) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool ra = g < len, rb = g + 8 < len;
    const bf16* xa = X + (size_t)(t0 + (ra ? g : 0)) * ldx;
    const bf16* xb = X + (size_t)(t0 + (rb ? g + 8 : 0)) * ldx;
    const int k0 = blockIdx.z * LORA_KCHUNK, k1 = min(K, k0 + LORA_KCHUNK);
    float acc[4][4] = {};
#pragma unroll
    for (int k = k0 + 16 * warp; k < k1; k += 64) {
        uint32_t a[4];
        a[0] = ra ? *reinterpret_cast<const uint32_t*>(xa + k + 2 * t) : 0u;
        a[1] = rb ? *reinterpret_cast<const uint32_t*>(xb + k + 2 * t) : 0u;
        a[2] = ra ? *reinterpret_cast<const uint32_t*>(xa + k + 2 * t + 8) : 0u;
        a[3] = rb ? *reinterpret_cast<const uint32_t*>(xb + k + 2 * t + 8) : 0u;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (n0 + 8 * j >= d.rank) break;
            const bf16* an = d.a + (size_t)(n0 + 8 * j + g) * K + k;
            mma_bf16(acc[j], a, *reinterpret_cast<const uint32_t*>(an + 2 * t),
                     *reinterpret_cast<const uint32_t*>(an + 2 * t + 8));
        }
    }
    // Sum the four warps' partial products, then add the block's share.
    __shared__ float red[4][16][33];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) red[warp][g + (e >> 1) * 8][8 * j + 2 * t + (e & 1)] = acc[j][e];
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 32; i += 128) {
        const int row = i >> 5, col = i & 31;
        if (row < len && n0 + col < d.rank)
            atomicAdd(&S[(size_t)(t0 + row) * lds + n0 + col],
                      red[0][row][col] + red[1][row][col] + red[2][row][col] + red[3][row][col]);
    }
}

// Grid: (max tiles, ceil(max part rows / 128), parts). Each warp adds 32
// output columns of one part for one tile. Layers alternate between two S
// buffers: while this layer reads one, it zeroes the step's rows of the other
// for the next layer's lora_shrink to accumulate into. A step runs an even
// number of layers, so both are clean for the next step's first.
extern "C" __global__ void __launch_bounds__(128) lora_expand(const float* __restrict__ S, int lds,
                                                              float* __restrict__ next,
                                                              const int* __restrict__ tiles,
                                                              const LoraDesc* __restrict__ table, void* Y, int ldy,
                                                              int f32_out) {
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int t0 = tiles[1 + 3 * tile], len = tiles[2 + 3 * tile], slot = tiles[3 + 3 * tile];
    if (blockIdx.y == 0 && blockIdx.z == 0)
        for (int i = threadIdx.x; i < len * lds; i += 128) next[(size_t)t0 * lds + i] = 0.f;
    const LoraDesc& d = table[slot];
    const int p = blockIdx.z, r = d.r[p];
    if (r == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rows = d.rows[p], c0 = blockIdx.y * 128 + 32 * warp;
    if (c0 >= rows) return;
    const bool ra = g < len, rb = g + 8 < len;
    const float* sa = S + (size_t)(t0 + (ra ? g : 0)) * lds + d.r_off[p];
    const float* sb = S + (size_t)(t0 + (rb ? g + 8 : 0)) * lds + d.r_off[p];
    const bf16* B = d.b[p];
    float acc[4][4] = {};
    for (int k = 0; k < r; k += 16) {
        auto pair = [&](bool ok, const float* s, int c) {
            if (!ok) return 0u;
            const float2 v = *reinterpret_cast<const float2*>(s + c);
            return bf2(v.x, v.y);
        };
        const uint32_t a[4] = {pair(ra, sa, k + 2 * t), pair(rb, sb, k + 2 * t), pair(ra, sa, k + 2 * t + 8),
                               pair(rb, sb, k + 2 * t + 8)};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = c0 + 8 * j + g;
            uint32_t b0 = 0u, b1 = 0u;
            if (n < rows) {
                const bf16* bn = B + (size_t)n * r + k;
                b0 = *reinterpret_cast<const uint32_t*>(bn + 2 * t);
                b1 = *reinterpret_cast<const uint32_t*>(bn + 2 * t + 8);
            }
            mma_bf16(acc[j], a, b0, b1);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = g + (e >> 1) * 8, col = c0 + 8 * j + 2 * t + (e & 1);
            if (row < len && col < rows) {
                const size_t at = (size_t)(t0 + row) * ldy + d.off[p] + col;
                if (f32_out) ((float*)Y)[at] += acc[j][e];
                else ((bf16*)Y)[at] = f2b(b2f(((bf16*)Y)[at]) + acc[j][e]);
            }
        }
}
