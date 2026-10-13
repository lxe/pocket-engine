# Design

## A step

Each step is one forward pass over a flat batch of tokens. Every running sequence contributes its next token. Prompt chunks of newly admitted requests fill the rest of the step's token budget (`--max-batch-tokens`, 4,096 by default). A new request therefore starts on the next step rather than waiting for the current batch to finish, and long prompts are spread over several steps.

The scheduler (`src/sched.rs`) has no GPU code. It admits requests, plans each step and writes the step's metadata: token ids, positions, KV slots, per-sequence state slots, block tables and sampling settings. A request is admitted only when KV blocks for its whole prompt plus `max_tokens` are free, so nothing is ever preempted.

The engine (`src/engine.rs`) copies that metadata to the device and runs the step. A step with no prompt tokens replays a CUDA graph captured at startup for the next padded batch size (1 to 256), which costs a single launch from the host. Other steps run eagerly, at about 220 kernel launches.

## Per layer

Each layer runs three or four cuBLAS GEMMs over all of the step's tokens, plus a few kernels of its own (`src/kernels/qwen35.cu`):

- **Fused projections.** One GEMM computes all linear-attention inputs (q, k, v, z, b, a). One computes the attention inputs (query with its gate, key, value). One computes the MLP's gate and up projections.
- **Linear attention (gated delta rule), three kernels.**
  - The causal convolution walks each sequence's tokens and carries the last three inputs across steps.
  - The recurrence. A decode token updates each head's 128×128 state in registers. A prompt runs in chunks of 32 tokens on tensor cores, in the chunked form FlashInfer and FLA use: with G the cumulative log decay, the chunk solves (I + A)D = β(V − e^G K S₀), outputs e^G Q S₀ + tril(QKᵀ ⊙ Γ) D and carries e^{G_last} S₀ + (K ⊙ e^{G_last − G})ᵀ D to the next chunk. Operands are bf16 and every product accumulates in f32.
  - The gated RMSNorm, in place on the recurrence's output.
  - Each sequence's states live in a slot: about 18.6 MB for the 2B model.
- **Full attention, two or three kernels.** One applies the q/k norms and rotary embedding and writes keys and values into a paged cache of 16-token blocks. Decode tokens then run paged attention with the output gate fused in, a warp loading four keys at a time. When a step has few tokens, each sequence's keys split into ranges that run in parallel (as many as give every SM about four blocks, each at least 256 keys), and a merge kernel combines them. Prompt tokens run flash attention: a block takes up to eight warps, one per query head of a key/value group for each of 8/G tiles of 16 query tokens, so each 16-key tile is read once for the whole group, double-buffered with `cp.async`, and the probabilities feed the second product straight from the accumulators.
- **Norms and activation.** The residual add is folded into the next RMSNorm, and SiLU-and-multiply is one kernel.

## Prefix cache

A request whose prompt begins like an earlier one skips the shared part. Attention needs only the earlier keys and values, but linear attention needs its recurrent state as it stood at the end of the prefix, and that state is overwritten token by token. So the cache keeps snapshots:

- **Where.** Prompts are hashed in units of 64 tokens, each unit's hash chained to the one before. A prompt can leave up to two prefixes: the end of the last whole unit before its final message (the last `<|im_start|>`), which the next turn of a conversation or another question under the same system prompt shares, and the last whole unit before its final token.
- **What.** At a point on the recurrence's 32-token chunk grid (or at the end of a prompt chunk), the prompt step itself writes the sequence's recurrent and convolution state into a spare state slot (`--prefix-cache` of them). At any other point the scheduler ends the prompt chunk, and the engine copies the state into a spare slot before the next step. Either way the prefix's full KV blocks gain a reference.
- **Reuse.** At admission the scheduler finds the longest cached prefix, copies the snapshot into the new sequence's slot, points its block table at the shared blocks and starts the prompt after them. Blocks are refcounted; the least recently used prefix is evicted when slots or blocks run short.

Output with a cached prefix is identical to output without one (`tests/gpu.rs`). On an RTX 5090 with Qwen3.5-9B (bf16, `scripts/workloads.py`):

| Workload | Cache off | Cache on |
|---|---|---|
| 2,000-token system prompt, 128 short questions, 16 at a time | 174 output tok/s, first token 0.49 s | 992 tok/s, 0.059 s |
| 16 conversations of 8 turns, each turn resending the history | 606 tok/s, 0.49 s | 955 tok/s, 0.28 s |

Without the cache, time to first token in the conversations grows from 0.34 s to 0.59 s by the fourth turn; with it, it stays between 0.23 and 0.35 s.

## Long contexts

Long prompts run on the chunked recurrence above and flash attention for prompt tokens. Long-context decoding runs split-key decode attention, which spreads one long sequence's keys over the whole GPU instead of one block per head. The split runs inside CUDA graphs: each graph has a fixed token count, which fixes the number of ranges, and each sequence divides its keys among as many of them as its length fills.

Qwen3.5-9B, bf16, one request at a time on an RTX 5090, 128 output tokens (`scripts/workloads.py long`):

| Prompt | First token | Prefill | Decode |
|---:|---:|---:|---:|
| 8k | 0.70 s | 11,700 tok/s | 84 tok/s |
| 32k | 3.2 s | 10,300 tok/s | 81 tok/s |
| 64k | 7.6 s | 8,600 tok/s | 78 tok/s |

Decoding with a short context runs at 89 tok/s; at 64k, reading the 2 GB of cached keys and values each step accounts for the rest. Four 32k prompts at once share the GPU at 9,400 prompt tokens per second.

Flash attention runs at 115 TFLOPS on the 5090 (cuBLAS's bf16 matmuls reach 211). Its blocks take eight warps of queries: only one block fits per SM, and four warps are too few to hide latency. The convolution splits each sequence into 64-token chunks rather than walking it in one thread per channel.

On an H100, prompts with a bf16 cache run a flash kernel on Hopper's wgmma instead, warp-specialized and persistent as FlashAttention 3 is. A block per SM loops over work items of 64 query tokens of two heads of one key/value group. A producer warpgroup (40 registers) writes each item's queries and loads each 64-key tile's keys and values by TMA into one of two shared-memory stages, one warp per 16-token page, one box of the cache's tensor map per page and swizzle atom; two consumer warpgroups (232 registers), a head each, compute. Keys and values pass through separate mbarrier pipelines that run on across items, so a key stage frees as soon as Q Kᵀ has read it and the next tile's keys load while P V runs. Q Kᵀ runs as m64n64k16 from shared memory and P V as m64n256k16 with P in registers. Each consumer issues the next tile's Q Kᵀ, rescales its output, issues this tile's P V and runs the next softmax while P V is on the tensor cores; the probabilities stay in f32 until P V has read the registers they replace, and a shared-memory store of the row sums keeps ptxas from hoisting the P V wait above the softmax. Scores are kept in log₂ units for `ex2.approx`, and only tiles reaching past the block's first query apply the causal mask. Items run key/value head first, the longest tiles first, and block b takes every grid-th one, so all blocks stream one head's cache together: a long context's keys and values, more than L2 holds, are read from memory about once per head rather than once per block. It reaches 665 TFLOPS for groups of 4 and 658 for groups of 6 at 61k cached tokens.

The chunked recurrence runs in two passes, as FLA's does. `gdn_prep` computes every chunk's local terms at once: normalized q and k, the decay, P, and the WY form U = T βV, W = T β e^G K with T = (I + A)⁻¹. `gdn_scan` then walks the chunks with only D = U − W S, the outputs and the state update, which are separable by value column, so up to four blocks share a head when a step has few sequences. A 4,096-token chunk of 9B's recurrence takes 0.65 ms on an RTX 5090.

Time to first token, one request at a time, on an H100 against vLLM 0.31.0 run on the same machine ([receipt](receipts/h100-mtp-ttft.json)):

| Prompt | 9B | 9B, vLLM | 27B | 27B, vLLM |
|---:|---:|---:|---:|---:|
| 8k | 0.29 s | 0.33 s | 0.82 s | 0.79 s |
| 32k | 1.13 s | 1.10 s | 3.36 s | 3.22 s |
| 64k | 2.40 s | 2.29 s | 7.38 s | 6.90 s |

Each prefill kernel alone on the same shapes (4,096 new tokens after 28,672 cached for attention; 4,096 tokens with a carried state for the recurrence):

| Kernel | H100, pocket-engine | H100, vLLM | RTX 5090, pocket-engine | RTX 5090, vLLM |
|---|---:|---:|---:|---:|
| Attention, groups of 4 | 661 TFLOPS | 776 (FlashAttention 3) | 130 | 167 (FlashAttention 2) |
| Attention, groups of 6 | 664 TFLOPS | 684 | 108 | 174 |
| Recurrence, 32 heads | 0.67 ms | 0.34 ms (FlashInfer) | 0.66 ms | 0.41 ms |
| Recurrence, 48 heads | 0.83 ms | 0.32 ms | 0.95 ms | 0.60 ms |

At 64k on 27B the attention is about 840 TFLOP, 1.3 s at either kernel's rate. Most of the remaining 0.5 s between the engines is the recurrence: 48 layers, 16 chunks of 4,096 tokens each, 0.5 ms apart per chunk.

## Time to first token

A short prompt's first token waits for one prompt step. Two details keep it from waiting longer:

- **`TCP_NODELAY`.** Streamed tokens are small writes; with Nagle's algorithm the first token would wait for the previous write's acknowledgement, which Linux delays by up to 40 ms.
- **Prefix-cache boundaries.** The recurrence writes its state at a boundary on its 32-token chunk grid straight into the snapshot slot ([Prefix cache](#prefix-cache)), so a 150-token prompt runs whole as one prompt step rather than two, each reading every weight.

Median time to first token for single short prompts (60 to 200 words, chat template):

| | pocket-engine | vLLM |
|---|---:|---:|
| RTX 5090, Qwen3.5-9B | 19.6 ms | 22 ms (0.30) |
| H100, Qwen3.5-9B | 13.3 ms | 55.0 ms (0.31.0) |
| H100, Qwen3.8-27B | 36.3 ms | 113.6 ms (0.31.0) |

## Multi-token prediction

Qwen3.5 checkpoints carry one extra layer (`mtp.*`) trained to predict the token after next. `--mtp N` uses it to draft N tokens a step and verifies them in the next one. Output is the model's own: with greedy decoding a sequence keeps drafts up to the first one the model would not have picked, and with sampling it keeps them while the sample at each position equals the draft, which leaves each token distributed exactly as sequential sampling would.

- **The draft layer.** It embeds a token, normalizes the embedding and the model's final hidden state before that token, joins them through a projection (`fc`) and runs one full-attention layer with its own KV cache. Its output layer is the model's, cut to the first 65,536 tokens (BPE ids are ordered roughly by frequency): drafts stay greedy, and the cut layer costs a quarter of the full one. On Qwen3.5-9B it keeps 74% of drafts accepted against 76% with the whole vocabulary.
- **Every step.** After the model picks its tokens, the draft layer runs over every token row of the step, so its KV cache follows the model's through prompts and decoding. Each sequence drafts from its last accepted row; further drafts run one row per sequence from the draft layer's own output, with positions and KV slots the GPU computes, so the whole step stays inside one CUDA graph.
- **Verification.** A decoding sequence feeds its last token and its drafts, and gets a pick for each. The linear-attention recurrence keeps the state after every fed token (and the convolution its window), each in a state slot lent from the free ones for the step: after acceptance the sequence keeps the slot holding the state after its last accepted token and returns the rest. A rejected draft costs no recomputation, and drafting reserves no memory per sequence.
- **When.** Sequences verify drafts when the free slots suffice and the step stays within 96 rows (256 on Hopper); otherwise the step decodes plainly. Past that, verification turns the step's matmuls compute-bound and costs more than it saves; an H100 does about 2.5 times the matmul work per byte read of an RTX 5090. Penalized requests and teacher-forced scoring never draft.

Qwen3.5-9B, bf16, RTX 5090, `scripts/load.py`, output tokens per second:

| In flight | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Without drafting | 85 | 566 | 917 | 1,351 | 1,811 |
| `--mtp 2` | 152 | 824 | 1,274 | 1,497 | 1,797 |
| `--mtp 3` | 151 | 820 | 1,286 | 1,264 | 1,743 |

(The `--mtp 3` row ran without the 96-row limit, which stops drafting at 32 in flight.) Over longer answers, one request runs at 89 tok/s plainly, 144 with one draft (74% accepted), 170 with two (62%), 181 with three (52%) and 182 with four (44%).

On an H100 against vLLM 0.31.0's MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`), on the same machine. Output tokens per second: one stream of four 512-token answers, then `scripts/load.py` by requests in flight. [Receipt](receipts/h100-mtp-ttft.json).

| Qwen3.5-9B | One stream | 1 | 8 | 16 | 32 | 64 | burst 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pocket-engine, plain | 150 | 144 | 948 | 1,666 | 2,612 | 3,387 | 2,325 |
| vLLM, plain | 144 | 126 | 653 | 1,094 | 1,797 | 2,720 | 2,278 |
| pocket-engine, 1 draft | 240 | 220 | 1,318 | 2,286 | 3,214 | 4,027 | 2,664 |
| vLLM, 1 draft | 215 | 178 | 801 | 1,238 | 2,248 | 2,935 | 2,620 |
| pocket-engine, 2 drafts | 280 | 253 | 1,457 | 2,375 | 3,413 | 4,154 | 2,994 |
| vLLM, 2 drafts | 245 | 201 | 918 | 1,438 | 2,444 | 3,024 | 2,809 |
| pocket-engine, 3 drafts | 296 | 253 | 1,427 | 2,452 | 3,368 | 4,066 | 2,783 |
| vLLM, 3 drafts | 247 | 194 | 907 | 1,543 | 2,140 | 3,220 | 2,821 |

| Qwen3.8-27B | One stream | 1 | 8 | 16 | 32 | 64 | burst 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pocket-engine, plain | 50 | 48 | 292 | 520 | 802 | 1,034 | 750 |
| vLLM, plain | 48 | 44 | 249 | 448 | 681 | 1,019 | 774 |
| pocket-engine, 1 draft | 85 | 74 | 446 | 772 | 1,039 | 1,113 | 908 |
| vLLM, 1 draft | 79 | 62 | 363 | 574 | 884 | 1,022 | 884 |
| pocket-engine, 2 drafts | 106 | 87 | 497 | 853 | 1,067 | 1,098 | 896 |
| vLLM, 2 drafts | 97 | 72 | 402 | 632 | 800 | 830 | 801 |
| pocket-engine, 3 drafts | 118 | 95 | 526 | 885 | 808 | 1,080 | 891 |
| vLLM, 3 drafts | 107 | 76 | 432 | 732 | 913 | 886 | 812 |

Both engines accept drafts at about the same rate: here 76%, 61% and 50% of drafts on 9B and 79%, 62% and 53% on 27B with one, two and three drafts; vLLM reports mean acceptance lengths of 1.79, 2.29 and 2.64 on 9B (79%, 65%, 55% of drafts) and 1.81, 2.34, 2.72 on 27B (81%, 67%, 57%). pocket-engine drafts with the cut output layer and no extra memory; a 27B sequence's state is 75 MB, so at 32 in flight with three drafts the free slots run out and those steps decode plainly, where vLLM, holding a slot per draft, keeps drafting.

Greedy output with drafting matches teacher-forced picks except at near ties: on Qwen3.5-9B, 1 of 320 positions departs, by 0.012 in log-probability. With FP8 activations, verification's matmuls run with four times the rows and take other cuBLAS kernels, and E4M3 rounding magnifies their last-bit differences: 10 of 320 positions depart, against 4 for the same comparison without drafting.

## Images

The vision encoder (`src/vision.rs`) runs for a request when the scheduler admits it, just before the step that holds the request's first prompt tokens.

1. **Preprocessing (CPU, `src/image.rs`).** Each image is resized so its sides are multiples of 32 pixels, normalized and split into 16-pixel patches. The patches are ordered in 2×2 blocks.
2. **Encoder.** A GEMM embeds the patches; interpolated learned positions and 2D rotary attention follow. Attention runs per image as batched cuBLAS GEMMs with a softmax kernel in between, in query chunks sized to a fixed scratch buffer. The merger turns each 2×2 block into one text embedding.
3. **Embedding pool.** The embeddings wait in a pool of rows, reserved at admission, until the prompt steps that read them have run. Prompt steps take image rows from the pool in place of the token embedding.
4. **Positions.** Rotary positions in the text model become three-dimensional (time, height, width). Image tokens spread over their grid, and the text after an image resumes past the grid's longer side, as Transformers computes them.

## Quantized weights

GGUF weights stay in their ggml block formats on the GPU (`src/kernels/quant.cuh`). Every format has one decoder that produces 8 consecutive weights of a row. The decoders follow ggml's own dequantization, and `scripts/quant_check.py` confirms they match gguf-py bit for bit after bf16 rounding. Formats whose blocks would leave weight bytes misaligned (Q8_0, Q4_0, Q5_0, IQ4_NL, Q3_K, Q6_K) are repacked with padding at load, so decoders can use 8-byte loads.

A quantized matrix multiply picks one of three paths by the number of rows in the step:

- **Up to 4 rows: matrix-vector kernel.** A warp per output feature streams the row once and decodes in registers, at about 1.75 TB/s on an RTX 5090.
- **5 to 160 rows: tensor cores, decoding into registers.** Each warp owns 16 output features and does mma.sync m16n8k16 in bf16. A dot product does not depend on the order of K, so each 32-wide step of K is assigned to fragment slots such that each lane needs 8 consecutive weights of its two rows and 8 consecutive inputs: two decodes and one 16-byte load per step. There is no shared memory and no barrier. When the grid alone would leave SMs idle, K is split across blocks into an f32 workspace.
- **Q4_0, Q4_K and Q6_K: a tiled layout, 1 to 160 rows on tensor cores.** At load these formats are rewritten in chunks of 16 rows × 256 inputs, in the order the tensor-core loop consumes them. One 16-byte load per lane fetches its nibbles for two steps, and a warp's load covers 512 contiguous bytes. Q4_K's packed 6-bit scales are unpacked to bytes, which costs 4.6 bits per weight instead of 4.5; Q4_0 stays at 4.5. Q6_K adds a word of high bits per lane and step, at 6.6 bits instead of 6.56. A nibble becomes an exact float without an integer conversion (`bits(0x4B000000 | q) - 2^23`), then one FMA (or, for Q4_0, a subtract and a multiply). That is the same arithmetic as the decoders, so tiled and plain dequantization agree bit for bit, and `dequant` checks it. Each warp loads the next two chunks while it computes the current one.
- **Above 160 rows (prompt chunks): cuBLAS.** The matrix is dequantized to bf16 scratch and handed to cuBLAS; one decode then serves the whole chunk.

All three issue no allocation and no host synchronization, so quantized decode steps run inside CUDA graphs like bf16 ones.

`bench-linear` times one matrix with its weights cycling through copies that exceed the L2 cache, as in serving. On Qwen3.5-9B's `ffn_up` (Q4_K, 12288 × 4096) on the RTX 5090:

| Rows | Q4_K, plain blocks | Q4_K, tiled | bf16 cuBLAS |
|---:|---|---|---|
| 1 | 22.8 µs | 21.0 µs | 61.6 µs |
| 8 | 40.8 µs | 22.2 µs | 64.2 µs |
| 16 | 47.2 µs | 26.8 µs | 65.5 µs |
| 32 | 65.4 µs | 29.5 µs | 66.8 µs |
| 64 | 71.3 µs | 49.9 µs | 78.9 µs |

The larger Qwen3.8-27B `ffn_gate` (17408 × 5120) at 8 rows takes 34 µs tiled, against 77 µs plain and 109 µs in bf16. Small matrices gain less: a 0.8B matrix gives too few warps to hide latency. Q6_K (a third of a Q4_K_M file) gains the most: 9B's `ffn_down` takes 33.5 µs at 8 rows tiled, against 77 µs plain and 64 µs in bf16. 27B's 1 GB output layer takes 0.70 ms at 8 rows against 1.80. Whole models:

| Qwen3.5-9B Q4_K_M, RTX 5090 | 1 | 8 | 16 | 32 | 64 in flight |
|---|---:|---:|---:|---:|---:|
| plain blocks | 166 | 573 | 863 | 1,232 | 1,566 tok/s |
| Q4_K tiled | 166 | 680 | 978 | 1,361 | 1,598 |
| Q4_K and Q6_K tiled | 170 | 782 | 1,087 | 1,444 | 1,666 |

Qwen3.8-27B Q4_K_M serves 183 tok/s at 8 in flight tiled, against 155 in plain blocks. The tiles are about 2% larger, so 29 requests fit beside it instead of 33.

## FP8

FP8 layers multiply E4M3 weights by E4M3 activations on cuBLASLt with per-block scales, accumulating in f32. Weights come from two places:

- **Qwen's FP8 checkpoints** (`Qwen3.8-27B-FP8`, `Qwen3.5-27B-FP8`) store E4M3 weights with an f32 scale per 128×128 tile (`weight_scale_inv`). Their `modules_to_not_convert` stay bf16: the embedding, output layer, convolution, the linear-attention `a` and `b` projections, and the vision tower.
- **`--fp8`** quantizes a bf16 checkpoint's projections on the GPU at load, keeping the same layers in bf16.

The scale layout follows the GPU:

- **Hopper (H100, H200)** runs block-scaled FP8: the checkpoint's 128×128 weight scales as stored, and one activation scale per row and 128 columns. cuBLASLt needs a multiple of 4 rows there, so steps are padded.
- **Blackwell (RTX 5090)** has only MXFP8 in cuBLASLt: a power-of-two scale per 32 values, for weights and activations. Checkpoint tiles are converted at load.

Activations are converted where they are produced, not in a separate pass. RMSNorm, SiLU-and-multiply, the gated-delta-rule output norm and the gated attention output write their bf16 result and its FP8 copy with scales in the same kernel. The FP8 multiply then reads only the FP8 copy. Every step stays free of allocation and host synchronization, so FP8 decode runs in CUDA graphs.

FP8 output depends slightly on what shares a step. cuBLAS and cuBLASLt choose kernels by row count, so a row's sums can differ in the last bit between batch sizes. bf16 rounding usually absorbs that, but an FP8 rounding that flips carries the difference into later layers. Scored alone and among other requests, the same tokens differ by a median of 0.01 in log-probability on the 0.8B and 2B models, about FP8's own error against bf16; the 4B and 9B models produce identical greedy text either way.

### FP8 KV cache

With `--kv-fp8`, attention keys and values are stored as E4M3, 256 bytes per token and head, with one f32 scale each (the head's largest magnitude over 448). The kernel that applies rotary embedding writes them quantized. Paged attention reads 8 bytes per lane and folds the key scale into the dot product and the value scale into the softmax weight, so values are never expanded in memory. A token and head take 260 bytes instead of 512.

On the RTX 5090 the cache holds 1.97 times the tokens: 7,522 blocks of 16 for Qwen3.5-9B instead of 3,818, and 3,652 for Qwen3.8-27B Q4_K_M instead of 1,850. That lets the 27B model take a full 32,768-token context, which a bf16 cache cannot fit. Concurrency does not change, because recurrent state still caps the number of sequences (54 for 9B, 17 for 27B). Throughput at the short contexts of the load test is the same: 1,527 against 1,516 tok/s for 9B at 64 in flight.

Accuracy against Transformers, with prompts of about 3,300 tokens (`scripts/accuracy.py long`) so that most of the context comes from the cache:

| Model | KV cache | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|---|
| Qwen3.5-2B | bf16 | 285 of 288 (all misses within one bf16 step) | 0.0012, 0.042 |
| Qwen3.5-2B | FP8 | 285 of 288 (all within one bf16 step) | 0.0024, 0.071 |
| Qwen3.5-9B | bf16 | 284 of 288 (all within one bf16 step) | 0.0005, 0.036 |
| Qwen3.5-9B | FP8 | 281 of 288 (5 within one bf16 step) | 0.0007, 0.076 |

On the short fixture prompts, the 0.8B, 2B and 9B models pass the bf16 accuracy bar unchanged with an FP8 cache.

## FP4

On the RTX 5090, FP4 layers run `mma.sync … kind::mxf8f6f4.block_scale` at m16n8k32. The instruction takes E2M1 weights and the MXFP8 activations the norm kernels already write. Its scales must be UE8M0 (powers of two), one per 32 values.

- **Weights.** They use the tiled order of the 4-bit GGUF formats (16 rows × 256 inputs). Each lane holds 8 inputs of a row as one 32-bit word, arranged so that two masks put four E2M1 codes in the middle bits of four bytes, where the instruction reads them. Decoding costs two operations per four weights. Each row adds one scale byte per 32 inputs, for 4.25 bits per weight.
- **Scale operands.** Each lane passes A's scale for row `g + 8 × (lane & 1)` and B's scale for its token `g`, the layout CUTLASS uses for SM120.
- **Prompt chunks.** Above 160 rows, tiles expand to MXFP8 for cuBLASLt. E2M1 values under the same power-of-two scale are exact in E4M3.

NVIDIA's NVFP4 checkpoints scale every 16 values by an E4M3 factor, which this instruction cannot take. The NVFP4-native instruction (`kind::mxf4nvf4`) needs FP4 activations too. So NVFP4 weights are re-quantized at load: each 32 values get the smallest power-of-two scale that keeps their largest within E2M1's 6, then round to the nearest code. Clipping to `floor(log2 amax) - 2`, the OCP rule, measured worse. The checkpoint's per-tensor FP8 layers load like Qwen's FP8 ones, with the scalar spread over the block-scale grid.

| Qwen3.8-27B on the RTX 5090 | Top-1 agreement (213) | Median, p95 log-prob difference |
|---|---|---|
| NVFP4 checkpoint, as MXFP4 | 197 | 0.0053, 0.42 |
| Q4_K_M (GGUF) | 207 | 0.0019, 0.17 |
| Qwen3.8-27B-FP8 (H100) | 209 | 0.0009, 0.088 |

| In flight | NVFP4 as MXFP4 | Q4_K_M |
|---:|---|---|
| 1 | 56 | 52 |
| 8 | 304 | 182 |
| 16 | 455 | 250 |
| 32 | 571 | 321 |
| 64 | 551, 1.9 s | 335, 1.8 s |

Both ran with `--memory-fraction 0.96` on a card that also drives a desktop; 27 requests fit beside the NVFP4 weights (19.8 GB) and 41 beside Q4_K_M. Qwen3.5-9B with `--fp4` agrees on 201 of 220 and serves 149, 858, 1,337, 1,837 and 2,477 tok/s at 1, 8, 16, 32 and 64 in flight. With `--fp8` the same figures are 100, 684, 1,171, 1,785 and 2,457.

## Loading

Weights go from the file to the GPU without passing through mapped memory. Tensor ranges are read with parallel `pread` calls into four 64 MB pinned buffers, and the copy engine drains each buffer while the next one fills. GGUF blocks that need padding are repacked on the GPU, and FP8 conversions run on the GPU too. All of this is ordered on one stream, so loading never waits for the device. From the page cache on the RTX 5090 machine:

| Model | Weights | Ready (to serving) |
|---|---|---|
| Qwen3.5-2B, bf16 | 3.8 GB | 1.5 s |
| Qwen3.5-9B, bf16 | 17.9 GB | 1.8 s |
| Qwen3.5-9B, `--fp8` | 11.2 GB | 2.0 s |
| Qwen3.5-9B, Q4_K_M | 5.7 GB | 2.0 s |
| Qwen3.8-27B, Q4_K_M | 19.4 GB | 1.9 s |

Copying through mapped pages takes 8.8 s for the 9B model and 24.7 s for the 27B one.

## Numerics

Activations are bf16 with f32 accumulation, rounded where the bf16 reference rounds them. Logits are f32. Sampling runs on the GPU, so only one token id per sequence returns to the host each step.

The linear-attention state is stored in bf16 between steps; the recurrence loads it into f32 registers, does all of its math in f32 and rounds once when it stores. That halves the memory per request (25 MB on 9B instead of 49, 75 MB on 27B instead of 146), and the state is what limits how many requests run at once. `--state-f32` keeps it in f32. Rounding once per step costs nothing measurable against Transformers, even over 512 generated tokens, where every token rounds the state:

| Check | f32 state | bf16 state |
|---|---|---|
| Qwen3.5-2B, 949 positions over generations of up to 512 tokens | 943 agree, median 0.0002 | 944, 0.0002 |
| Qwen3.5-2B, 3,300-token prompts | 285 of 288, 0.0012 | 286, 0.0011 |
| Qwen3.5-9B, 1,293 positions over generations of up to 512 tokens | 1,291, 0.0000 | 1,291, 0.0000 |
| Qwen3.5-9B, 3,300-token prompts in 64-token chunks | 284 of 288, 0.0005 | 286, 0.0004 |
| Fixtures, 0.8B / 2B / 4B / 9B | 214, 214, 209, 218 agree | 214, 215, 209, 219 |

On the RTX 5090, twice the requests fit and throughput at high load rises (output tokens per second, median time to first token at 64 in flight):

| Model | Requests that fit, f32 → bf16 | 64 in flight, f32 | 64 in flight, bf16 |
|---|---|---|---|
| Qwen3.5-9B, bf16 weights | 54 → 101 | 1,516, 0.31 s | 1,687, 0.14 s |
| Qwen3.5-9B, `--fp8` | 119 → 231 | 2,508, 0.08 s | 2,675, 0.08 s |
| Qwen3.8-27B, Q4_K_M | 17 → 33 | 231, 4.7 s | 308, 2.4 s |

One request alone runs at the same speed either way.

## Accuracy

`tests/gpu.rs` and `scripts/accuracy.py` compare teacher-forced picks and log-probabilities with Transformers (bf16). The prompt runs as prefill and the continuation as decode steps.

| Model | Positions | Top-1 agreement | Misses within one bf16 step | Median, p95 log-prob difference |
|---|---|---|---|---|
| Qwen3.5-0.8B | 222 | 213 | 9 of 9 | 0.0032, 0.056 |
| Qwen3.5-2B | 217 | 214 | 3 of 3 | 0.0015, 0.043 |
| Qwen3.5-4B | 210 | 209 | 1 of 1 | 0.0004, 0.032 |
| Qwen3.5-9B | 220 | 219 | 1 of 1 | 0.0003, 0.032 |
| Qwen3.8-27B | 213 | 211 | 2 of 2 | 0.0004, 0.035 |

The small models ran on an RTX 5090. Qwen3.8-27B in bf16 does not fit there, so its reference and run used an H100. An H100 build of the 2B model agreed on 215 of 217.

The same comparison with image prompts covers three images. Two keep their sides; the third (300×211) is resized. Prompt tokens match Transformers' processor exactly for all three models:

| Model | Positions | Top-1 agreement | Misses within one bf16 step | Median, p95 log-prob difference |
|---|---|---|---|---|
| Qwen3.5-0.8B | 87 | 85 | 2 of 2 | 0.0007, 0.054 |
| Qwen3.5-2B | 87 | 87 | none | 0.0003, 0.088 |
| Qwen3.5-4B | 62 | 62 | none | 0.0005, 0.043 |
| Qwen3.5-9B | 85 | 85 | none | 0.0000, 0.040 |
| Qwen3.8-27B | 91 | 91 | none | 0.0001, 0.050 |

GGUF quantizations are measured against the same bf16 references, so they show the quantization's own error:

| GGUF | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|
| Qwen3.5-0.8B Q8_0 | 217 of 222 (every miss within one bf16 step) | 0.0035, 0.051 |
| Qwen3.5-0.8B Q4_K_M | 199 of 222 | 0.044, 0.57 |
| Qwen3.5-0.8B Q4_0 | 195 of 222 | 0.068, 0.78 |
| Qwen3.8-27B Q4_K_M | 207 of 213 | 0.0019, 0.17 |
| Qwen3.8-27B UD-Q4_K_XL | 210 of 213 | 0.0011, 0.16 |
| Qwen3.8-27B UD-IQ4_XS | 207 of 213 | 0.0022, 0.19 |

FP8 against the same bf16 references. The 27B checkpoints ran on an H100; `--fp8` ran on the RTX 5090 (MXFP8). [Receipt](receipts/fp8.json).

| FP8 | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|
| Qwen3.8-27B-FP8 (checkpoint) | 209 of 213 (2 misses within one bf16 step) | 0.0009, 0.088 |
| Qwen3.8-27B-FP8, images | 91 of 91 | 0.0005, 0.14 |
| Qwen3.5-27B-FP8 (checkpoint) | 215 of 217 (both within one bf16 step) | 0.0003, 0.066 |
| Qwen3.5-27B, bf16, for comparison | 217 of 217 | 0.0001, 0.032 |
| Qwen3.5-9B, `--fp8` | 213 of 220 (6 within one bf16 step) | 0.0011, 0.16 |
| Qwen3.5-9B, `--fp8`, images | 84 of 85 | 0.0000, 0.097 |
| Qwen3.5-4B, `--fp8` | 201 of 210 | 0.0023, 0.18 |
| Qwen3.5-0.8B, `--fp8` | 209 of 222 (10 within one bf16 step) | 0.019, 0.25 |

## Throughput

`scripts/load.py` sends streamed chat completions. Prompts are about 60 to 200 tokens and outputs 16 to 64 tokens, decoded greedily. The table gives output tokens per second and median time to first token. [Receipt](receipts/load.json).

| In flight | H100: pocket-engine | H100: vLLM 0.31.0 | RTX 5090: pocket-engine |
|---:|---|---|---|
| 1 | 415 tok/s, 0.044 s | 346, 0.028 s | 306, 0.043 s |
| 8 | 2,258, 0.046 s | 1,633, 0.060 s | 1,812, 0.043 s |
| 16 | 3,674, 0.045 s | 2,676, 0.077 s | 3,065, 0.043 s |
| 32 | 5,484, 0.046 s | 4,198, 0.10 s | 4,308, 0.045 s |
| 64 | 7,502, 0.056 s | 5,467, 0.13 s | 5,554, 0.051 s |
| burst of 64 | 4,939 | 4,299 | 4,593 |

The model was Qwen3.5-2B. Both H100 runs used the same H100 SXM (driver 580.126.09), one after the other, on 2026-10-12. pocket-engine was ready 5 seconds after launch and vLLM after 79. The H100 runs built the engine from Switchyard commit b4b3d7c, which measures the same as this repository's build on an RTX 5090.

### Quantized weights, RTX 5090

Qwen3.5-9B under the same load, with the recurrent state in f32, output tokens per second and median time to first token at 64 in flight. With `--max-seqs` lowered to fit the recurrent state, 53 requests fit beside the bf16 weights, 119 beside FP8 and 187 beside Q4_K_M.

| In flight | bf16 (safetensors) | `--fp8` (MXFP8) | Q4_K_M (GGUF) | Q8_0 (GGUF) |
|---:|---|---|---|---|
| 1 | 87 | 109 | 175 | 135 |
| 8 | 546 | 739 | 607 | 402 |
| 16 | 909 | 1,259 | 903 | 660 |
| 32 | 1,284 | 1,873 | 1,285 | 982 |
| 64 | 1,519, 0.34 s | 2,508, 0.08 s | 1,562, 0.15 s | 1,301, 0.17 s |
| burst of 64 | 1,294 | 2,194 | 1,546 | 1,150 |

### Qwen3.8-27B

Output tokens per second under the same load, and median time to first token. [Receipt](receipts/qwen3.8-27b.json).

| In flight | RTX 5090, Q4_K_M (GGUF) | H100, bf16 | H100, FP8 checkpoint |
|---:|---|---|---|
| 1 | 54, 0.07 s | 46, 0.05 s | 69, 0.05 s |
| 8 | 154, 0.14 s | 256, 0.07 s | 349, 0.06 s |
| 16 | 226, 0.18 s | 450, 0.08 s | 558, 0.07 s |
| 32 | 241, 1.6 s | 662, 0.10 s | 810, 0.10 s |
| 64 | 231, 4.7 s | 898, 0.15 s | 1,107, 0.14 s |

These runs kept the recurrent state in f32: 146 MB per request (48 linear-attention layers × 48 heads × 128 × 128), plus 64 KB of KV cache per token (16 attention layers). On the 5090, the Q4_K_M weights (19 GB) left room for 17 requests at once, so throughput levelled off from 16 in flight. With the state in bf16 (the default) 33 fit, and 32 in flight reach 318 tok/s. The H100 holds 63 beside the bf16 weights and 142 beside the FP8 ones (`Qwen3.8-27B-FP8`, about 30 GB).
