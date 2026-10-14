# Design

How pocket-engine serves a request, from scheduling to kernels. Speed numbers are in [benchmarks](benchmarks.md) and accuracy in [accuracy](accuracy.md).

## Overview

Qwen3.5 and Qwen3.8 are hybrid models: three of every four layers use linear attention (a gated delta rule with a fixed-size recurrent state per head), and the fourth uses ordinary full attention with a KV cache. The engine is a single Rust binary:

- `src/sched.rs` decides what runs in each step. It has no GPU code.
- `src/engine.rs` runs steps on the GPU and streams tokens back.
- `src/model.rs` launches each layer's kernels.
- `src/kernels/` holds the CUDA kernels. Large matrix multiplies go to cuBLAS.
- `src/server.rs` is the OpenAI-compatible HTTP API.

## Steps

Every step is one forward pass over a flat batch of tokens:

- each running request contributes its next token;
- new requests' prompts fill the rest of the step, up to `--max-batch-tokens`.

A new request starts on the next step instead of waiting for others to finish, and a long prompt is spread over several steps. A request is admitted only when its whole prompt and `max_tokens` fit in the KV cache, so nothing is ever preempted.

Steps that only decode replay a CUDA graph captured at startup, which costs one launch. Sampling runs on the GPU, so only one token id per request comes back to the host.

## Linear attention

Each request keeps a 128×128 state per head (25 MB per request on 9B), stored in bf16 between steps and computed in f32.

- **Decoding** updates the state one token at a time. Blocks stay resident and load the next heads' states into shared memory while they compute one, so the update runs near memory bandwidth.
- **Prompts** run in chunks of 32 tokens on tensor cores, in the same chunked form FLA and FlashInfer use. A first kernel computes each chunk's local terms for all chunks at once; a second walks the chunks in order, carrying the state. The second kernel loads the next chunk while it computes the current one.

## Full attention

Keys and values live in a paged cache of 16-token blocks.

- **Decoding** reads a request's cached keys and values directly. For long contexts, one request's keys are split across the whole GPU and the partial results merged.
- **Prompts** run flash attention. On H100 and H200 it is a FlashAttention-3-style kernel: one warpgroup loads key and value tiles with TMA while two others compute, overlapping the softmax with the matrix multiplies. On B200 it is FlashAttention-4-style: tcgen05 matrix multiplies accumulate in tensor memory, and separate warps issue them, load, run the softmax and correct the output. The query heads that share a key/value head run as one cluster, which loads each key and value tile once for all of them. On other GPUs it is a FlashAttention-2-style kernel on `mma.sync`.

## Prefix cache

A prompt that starts like an earlier one skips the shared part. Attention only needs the cached keys and values, but linear attention needs its state exactly as it stood at the end of the prefix. So the cache keeps state snapshots:

- prompts are hashed in 64-token units;
- a prompt saves up to two snapshots: before its last message (shared by the next turn of a conversation) and near its end;
- a new request copies the longest matching snapshot and shares the prefix's KV blocks.

Output with a cached prefix is the same as without one.

## Speculative decoding

The checkpoints carry one extra layer trained to predict the token after next. With `--mtp N` it drafts N tokens each step, and the next step checks them. Output is exactly what the model would produce without drafting, for greedy and sampled requests alike.

- The draft layer runs inside the same step and CUDA graph as the model.
- When checking drafts, the linear-attention state after each draft goes into a spare state slot, so a rejected draft needs no recomputation and drafting reserves no memory.
- With `--ngram`, drafts also come from prompt lookup: each sequence keeps an index of its 2- to 4-token n-grams, and the latest earlier occurrence of its current suffix proposes what followed. A sequence with no match pads its drafts so steps stay uniform enough to replay graphs, and steps where no sequence has a real guess decode plainly.
- Drafting turns itself off when a step gets too large to profit, or when spare slots run out under heavy load.

## LoRA adapters

Each adapter is prepared once on the host in the layout the kernels read (bf16, scale folded in, ranks padded to 16, value heads reordered for GGUF bases) and copied into one of a fixed set of GPU slots when a request needs it. Per step, the scheduler lists tiles of up to 16 tokens that share an adapter. After each linear layer's own matrix multiply, two kernels add the low-rank update: one computes x Aᵀ for all of the layer's fused projections at once, on a second stream so it overlaps the main matrix multiply; the other adds its product with Bᵀ into the output. Swapping an adapter changes only a table the kernels read, so steps with adapters replay CUDA graphs too. Prefix-cache entries are kept per adapter.

## Images

1. The CPU resizes each image to multiples of 32 pixels and cuts it into patches.
2. The vision encoder runs when the request is admitted, turning each 32×32 block into one prompt token.
3. The prompt steps read those embeddings in place of token embeddings, with the 3-D rotary positions Qwen's vision models use.

## Quantized weights

**GGUF** weights stay in their original block formats on the GPU, and each format has one decoder, checked bit for bit against llama.cpp's reference. A matrix multiply picks a path by the number of rows in the step:

| Rows | Path |
|---|---|
| 1 to 4 | a matrix-vector kernel that streams the weights once |
| 5 to 160 | tensor cores, decoding weights straight into registers |
| more than 160 | dequantize to bf16, then cuBLAS |

Q4_0, Q4_K, Q5_K and Q6_K are rewritten at load into the exact order the tensor-core loop reads them, which roughly doubles their speed at 8 to 32 rows.

In batches of 16 rows or more, those formats run on int8 tensor cores, against activations quantized to int8 per 32 inputs (llama.cpp's scheme). This skips decoding weights to bf16 and replaces dequantize-then-cuBLAS for prompts. The linear-attention output projection stays in bf16, since its input loses too much precision in int8.

**FP8** weights and activations run on cuBLASLt, with 128×128 block scales on Hopper and MXFP8 on Blackwell. Kernels that produce activations write the FP8 copy directly.

**FP4** (RTX 50-series) runs on block-scaled `mma.sync`. NVIDIA's NVFP4 checkpoints are converted to MXFP4 at load, since the instruction takes power-of-two scales.

## AMD

The `rocm` feature builds the same kernels with hipcc for gfx942 and runs them through HIP and hipBLAS (`src/hip.rs`). The few hardware primitives the kernels need (asynchronous copies, `ldmatrix`, `mma.sync`, streaming loads) live in `kernels/backend/cuda.cuh` and `kernels/backend/hip.cuh`. On AMD they emulate NVIDIA's fragments with shuffles, so the hot paths have native versions in `kernels/amd/`, written for 64-lane waves and MFMA matrix instructions: the chunked delta rule, prompt and decode attention, int8 GGUF tiles, the decode recurrence, and matrix-vector products for 1 to 4 rows. FP8 and FP4 are NVIDIA only.

## Loading

Weights are read from the file with parallel `pread` calls into pinned buffers, while the GPU copies the previous buffer. Nothing goes through memory-mapped pages, which is about five times slower. A 9B model is ready to serve in under 2 seconds from the page cache.
