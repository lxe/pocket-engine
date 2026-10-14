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

- **Decoding** updates the state one token at a time.
- **Prompts** run in chunks of 32 tokens on tensor cores, in the same chunked form FLA and FlashInfer use. A first kernel computes each chunk's local terms for all chunks at once; a second walks the chunks in order, carrying the state. The second kernel loads the next chunk while it computes the current one.

## Full attention

Keys and values live in a paged cache of 16-token blocks.

- **Decoding** reads a request's cached keys and values directly. For long contexts, one request's keys are split across the whole GPU and the partial results merged.
- **Prompts** run flash attention. On H100 and H200 it is a FlashAttention-3-style kernel: one warpgroup loads key and value tiles with TMA while two others compute, overlapping the softmax with the matrix multiplies. On other GPUs it is a FlashAttention-2-style kernel on `mma.sync`.

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
- Drafting turns itself off when a step gets too large to profit, or when spare slots run out under heavy load.

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

Q4_0, Q4_K and Q6_K are rewritten at load into the exact order the tensor-core loop reads them, which roughly doubles their speed at 8 to 32 rows.

**FP8** weights and activations run on cuBLASLt, with 128×128 block scales on Hopper and MXFP8 on Blackwell. Kernels that produce activations write the FP8 copy directly.

**FP4** (RTX 50-series) runs on block-scaled `mma.sync`. NVIDIA's NVFP4 checkpoints are converted to MXFP4 at load, since the instruction takes power-of-two scales.

## Loading

Weights are read from the file with parallel `pread` calls into pinned buffers, while the GPU copies the previous buffer. Nothing goes through memory-mapped pages, which is about five times slower. A 9B model is ready to serve in under 2 seconds from the page cache.
