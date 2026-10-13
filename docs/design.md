# Design

## A step

Each step is one forward pass over a flat batch of tokens. Every running sequence contributes its next token. Prompt chunks of newly admitted requests fill the rest of the step's token budget (`--max-batch-tokens`, 4,096 by default). A new request therefore starts on the next step rather than waiting for the current batch to finish, and long prompts are spread over several steps.

The scheduler (`src/sched.rs`) has no GPU code. It admits requests, plans each step and writes the step's metadata: token ids, positions, KV slots, per-sequence state slots, block tables and sampling settings. A request is admitted only when KV blocks for its whole prompt plus `max_tokens` are free, so nothing is ever preempted.

The engine (`src/engine.rs`) copies that metadata to the device and runs the step. A step with no prompt tokens replays a CUDA graph captured at startup for the next padded batch size (1 to 256), which costs a single launch from the host. Other steps run eagerly, at about 220 kernel launches.

## Per layer

Each layer runs three or four cuBLAS GEMMs over all of the step's tokens, plus a few kernels of its own (`src/kernels/qwen35.cu`):

- **Fused projections.** One GEMM computes all linear-attention inputs (q, k, v, z, b, a). One computes the attention inputs (query with its gate, key, value). One computes the MLP's gate and up projections.
- **Linear attention (gated delta rule), two kernels.**
  - The causal convolution walks each sequence's tokens and carries the last three inputs across steps.
  - The recurrence keeps each head's 128×128 state in registers while it walks a sequence's tokens, then applies the gated RMSNorm.
  - Each sequence's states live in a slot: about 18.6 MB for the 2B model.
- **Full attention, two kernels.** One applies the q/k norms and rotary embedding and writes keys and values into a paged cache of 16-token blocks. The other is paged attention with the output gate fused in.
- **Norms and activation.** The residual add is folded into the next RMSNorm, and SiLU-and-multiply is one kernel.

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
- **Above 160 rows (prompt chunks): cuBLAS.** The matrix is dequantized to bf16 scratch and handed to cuBLAS; one decode then serves the whole chunk.

All three issue no allocation and no host synchronization, so quantized decode steps run inside CUDA graphs like bf16 ones.

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

## Loading

Weights go from the file to the GPU without passing through mapped memory. Tensor ranges are read with parallel `pread` calls into four 64 MB pinned buffers, and the copy engine drains each buffer while the next one fills. GGUF blocks that need padding are repacked on the GPU, and FP8 conversions run on the GPU too. All of this is ordered on one stream, so loading never waits for the device. From the page cache on the RTX 5090 machine:

| Model | Weights | Ready (to serving) |
|---|---|---|
| Qwen3.5-2B, bf16 | 3.8 GB | 1.5 s |
| Qwen3.5-9B, bf16 | 17.9 GB | 1.8 s |
| Qwen3.5-9B, `--fp8` | 11.2 GB | 2.0 s |
| Qwen3.5-9B, Q4_K_M | 5.7 GB | 2.0 s |
| Qwen3.8-27B, Q4_K_M | 19.4 GB | 1.9 s |

Copying through mapped pages took 8.8 s for the 9B model and 24.7 s for the 27B one.

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

The small models ran on an RTX 5090. Qwen3.8-27B in bf16 does not fit there, so its reference and run used a rented H100. An H100 build of the 2B model agreed on 215 of 217.

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

The model was Qwen3.5-2B. Both H100 runs used the same RunPod H100 SXM (driver 580.126.09), one after the other. pocket-engine was ready 5 seconds after launch and vLLM after 79. The H100 numbers come from the code before it moved to this repository (Switchyard commit b4b3d7c). Back-to-back RTX 5090 runs of that build and this one measure the same.

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

These runs kept the recurrent state in f32: 146 MB per request (48 linear-attention layers × 48 heads × 128 × 128), plus 64 KB of KV cache per token (16 attention layers). On the 5090, the Q4_K_M weights (19 GB) left room for 17 requests at once, so throughput levelled off from 16 in flight. With the state in bf16 (now the default) 33 fit, and 32 in flight reach 318 tok/s. The H100 holds 63 beside the bf16 weights and 142 beside the FP8 ones (`Qwen3.8-27B-FP8`, about 30 GB).
