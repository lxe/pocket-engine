# Design

How pocket-engine serves a request. Speed is in [benchmarks](benchmarks.md), accuracy in [accuracy](accuracy.md).

## The model

Qwen3.5 and Qwen3.8 mix two kinds of layers. Three in four use linear attention, which keeps a fixed-size state per request. The fourth is ordinary attention over a KV cache.

## The engine

One Rust binary, no Python:

| Part | Job |
|---|---|
| `src/sched.rs` | decides what runs in each step (no GPU code) |
| `src/engine.rs` | runs steps and streams tokens back |
| `src/model.rs` | launches each layer's kernels |
| `src/kernels/` | the CUDA kernels, also built for [AMD](#amd) |
| `src/server.rs` | the OpenAI-compatible API |

Planning a step takes the CPU 0.01 to 0.04 ms, so the GPU never waits on it.

## Steps

Each step is one batch: the next token of every running request, plus new prompts up to `--max-batch-tokens`.

- New requests join on the next step. Long prompts spread over several steps.
- A request starts only when its prompt and `max_tokens` fit in memory, so nothing is ever preempted.
- Steps that only decode replay a CUDA graph: one launch per step.
- Sampling runs on the GPU, so only token ids come back to the CPU.

## Sampling

- Top-k, top-p, min-p, penalties and `logit_bias` run in one pass on the GPU.
- Structured output: llguidance works out the allowed tokens on the CPU, and the GPU masks the rest.
- Log-probabilities come back in OpenAI's format.

## Linear attention

Each request keeps a 128×128 state per head, 25 MB on 9B, stored in bf16.

| Phase | How |
|---|---|
| Decoding | one token at a time; each block loads the next heads' state while it updates the current one |
| Prompts | chunks of 32 tokens on tensor cores, as FLA and FlashInfer do: each chunk's own work in parallel, then one pass that carries the state |

The short convolution before it and the norm after it are folded into neighbouring kernels where possible.

## Attention

Keys and values live in a paged cache of 16-token blocks.

| Phase | GPU | Kernel |
|---|---|---|
| Decoding | all | reads the cache directly; long contexts split across the GPU |
| Prompts | H100, H200 | FlashAttention 3 style |
| Prompts | B200 | FlashAttention 4 style, two heads per pair of SMs |
| Prompts | others | FlashAttention 2 style |

One kernel prepares q and k and writes the cache. The output gate is applied as attention writes its result.

## Prefix cache

A prompt that starts like an earlier one skips the shared part. Linear attention needs its exact state at that point, so the cache saves state snapshots: before the last message, which the next turn of a chat reuses, and near the end. Output is identical with or without the cache.

## Speculative decoding

| Flag | Drafts come from |
|---|---|
| `--mtp N` | the checkpoint's own layer for predicting the token after next |
| `--ngram` | earlier text in the same request |

- Output is exactly what the model would produce without drafting.
- Each draft's state goes into a spare slot, so rejecting a draft costs nothing.
- Drafting turns itself off when it would not pay.

## LoRA adapters

- Adapters are converted once and copied into GPU slots when needed. They load and unload while the server runs, through vLLM's endpoints.
- Requests for different adapters, and for the base model, share a batch.
- The adapter's update runs beside the layer's own matrix multiply.
- Steps with adapters still replay CUDA graphs.

## Images

1. The CPU resizes each image and cuts it into patches.
2. The vision encoder runs when the request is admitted, making one token per 32×32 block.
3. The prompt reads those tokens in place of text.

## Quantized weights

**GGUF** weights stay in their block formats on the GPU, decoded exactly as llama.cpp does. The path depends on how many rows a step multiplies:

| Rows | Path |
|---|---|
| 1 to 4 | stream the weights once |
| a few | tensor cores, decoding weights into registers |
| more | int8 tensor cores, with activations quantized as llama.cpp does |

Common formats are reordered at load to match the order the kernels read them, about twice as fast at small batches.

**FP8** runs on cuBLASLt: block scales on Hopper, MXFP8 on Blackwell.

**FP4** runs on RTX 50-series cards. NVFP4 checkpoints are converted to MXFP4 at load.

## AMD

`--features rocm` builds the same kernels for the MI325X. The hot paths have their own versions in `kernels/amd/`, written for AMD's 64-lane waves and matrix instructions. FP8 and FP4 are NVIDIA only.

- Decoding streams weights with a kernel modelled on vLLM's; larger products go to hipBLASLt.
- Adapter updates run in order on one stream, since a second stream changed outputs on AMD.

What costs time on an MI325X (details in the [receipt](receipts/mi325x.json)):

| Cost | Measured | Response |
|---|---|---|
| Each kernel launch | 1.5 µs | Fewer, fused kernels |
| A load waiting on a branch | ~0.3 µs | Issue loads first |
| A device-wide fence | 83 µs | No hand-offs between blocks |

## Loading

Weights are read in parallel straight into pinned memory while the GPU copies the previous buffer. Memory-mapped reads were about five times slower. The tokenizer loads at the same time. A 9B model is ready in under 2 seconds.
