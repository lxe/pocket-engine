# Design

pocket-engine is built on one bet: an engine that serves a single model family can be far simpler than a general one, and simplicity is what makes it fast to start and cheap to run. This page follows a request from the scheduler down to the kernels and explains why each piece looks the way it does. Speed is in [benchmarks](benchmarks.md), accuracy in [accuracy](accuracy.md).

## The models

Qwen3.5 and Qwen3.8 are hybrids. Three of every four layers use linear attention, a gated delta rule that folds everything a request has seen into a fixed 128×128 state per head. The fourth layer is ordinary attention over a KV cache.

Much of the design follows from that mix. The recurrent state is cheap to keep but has to be exact, which shapes the prefix cache and speculative decoding. Full attention is only a quarter of the layers, but it dominates long prompts, which is where the most specialised kernels live.

## One process, no Python

The engine is a single Rust binary. The scheduler, the HTTP server, the tokenizer and the GPU work share one process, with no Python or PyTorch on the serving path.

| Part | Job |
|---|---|
| `src/sched.rs` | decides what runs in each step; no GPU code, unit-tested on the CPU |
| `src/engine.rs` | runs steps on the GPU and streams tokens back |
| `src/model.rs` | launches each layer's kernels |
| `src/kernels/` | the CUDA kernels, also built for [AMD](#amd) |
| `src/server.rs` | the OpenAI-compatible API |

This pays off twice. The server is ready seconds after launch, since there is nothing to import or compile. And planning a step costs the CPU 0.01 to 0.04 ms, so the GPU never waits on the host. Engines written in Python need an overlap scheduler to hide that cost; this one has nothing to hide.

## Steps

The engine runs in steps. Each step is one forward pass over a flat batch: the next token of every running request, topped up with new prompt tokens to `--max-batch-tokens`. A new request joins on the next step, and a long prompt simply spreads over several.

A request is admitted only when its whole prompt and `max_tokens` fit in the KV cache. That rules out preemption: once a request starts it runs to the end, and memory pressure never forces recomputation.

Steps that only decode replay a CUDA graph captured at startup, so a whole forward pass costs one launch. Sampling happens on the GPU, and only token ids travel back to the CPU.

## Sampling

Sampling is a single pass on the GPU: penalties and `logit_bias`, then top-k, top-p and min-p, then the draw. Structured output fits into the same pass. llguidance works out each request's allowed tokens on the CPU, for all requests in parallel, and the GPU masks everything else. Log-probabilities come back in OpenAI's format.

## Linear attention

Each request keeps a 128×128 state per head, 25 MB per request on the 9B model, stored in bf16 and updated in f32.

Decoding is bound by memory: every token reads and writes the whole state. The decode kernel keeps its blocks resident and loads the next heads' state while it updates the current one, so memory never sits idle.

Prompts use the chunked form that FLA and FlashInfer use. Within a chunk of 32 tokens the recurrence can be rewritten as small matrix multiplies, so the work local to each chunk runs on tensor cores for all chunks at once. A second pass then walks the chunks in order, carrying the state from one to the next.

## Attention

Keys and values live in a paged cache of 16-token blocks. Decoding reads the cache directly, and a very long context splits one request's keys across the whole GPU.

Prompts run flash attention, in the style each GPU generation was designed for:

| GPU | Prompt kernel |
|---|---|
| RTX 50-series | FlashAttention 2 style |
| H100, H200 | FlashAttention 3 style: one warpgroup loads tiles while two compute |
| B200 | FlashAttention 4 style: results accumulate in tensor memory, each warp has one job |

The B200 kernel handles an unusual shape. Qwen's attention heads are 256 wide, twice the usual, and at that width one SM spends longer pulling in keys and values than computing with them. So two heads that share keys and values run as a single matrix multiply across a pair of SMs, each holding half of every tile, and the other heads of the group share those loads.

The small steps around attention are fused away. One kernel normalises queries and keys, applies rotary positions and writes the cache, and the output gate is applied as attention writes its result.

## Prefix cache

Chat traffic sends the same history again with every turn. For full attention, reusing a prefix just means sharing its cached keys and values. Linear attention is harder: it needs its state exactly as it stood at the end of the prefix, and that state cannot be rebuilt from the cache.

So the cache keeps snapshots of the state. Each prompt saves at most two: one just before its last message, which the next turn of the conversation will reuse, and one near its end. A new request copies the longest matching snapshot and skips straight past the shared part. Output is identical with or without the cache.

## Speculative decoding

Speculative decoding guesses a few tokens ahead and has the full model check all the guesses in one step. Every guess the model accepts saves a step, and the answer is exactly what it would have been without guessing, for greedy and sampled requests alike.

| Flag | Drafts come from |
|---|---|
| `--mtp N` | the checkpoint's own layer for predicting the token after next |
| `--ngram` | earlier text in the same request |

Prompt lookup pays off when an answer repeats text from its prompt, as in editing, extraction or quoting: the engine indexes the request's recent word sequences and proposes whatever followed the last time the current one appeared. Such answers come out up to four times faster, and ordinary chat is unaffected.

Hybrid models make rejection tricky, because checking a draft has already advanced the linear-attention state. The engine writes the state after each draft into a spare slot, so rolling back is a matter of picking the right slot. Drafting also tracks its own acceptance and steps aside when it stops paying.

## LoRA adapters

One server can host many fine-tunes of the same base model. Each adapter is converted once into the layout the kernels read and copied into a GPU slot when a request needs it. Adapters load and unload while the server runs, through vLLM's endpoints. Reading and converting an adapter happens on a separate thread, so requests already running don't pause while it loads. With `--lora-dir`, the first request that names an adapter loads it, and the least recently used idle ones are dropped to keep memory bounded.

Requests for different adapters, and for the base model, share batches. Each layer adds the adapter's low-rank update beside its own matrix multiply, on a second stream. Switching adapters only changes a table the kernels read, so steps with adapters still replay CUDA graphs.

## Images

1. The CPU resizes each image and cuts it into patches.
2. The vision encoder runs when the request is admitted, turning each 32×32 block into one token.
3. The prompt reads those tokens in place of text, with the 3-D positions Qwen's vision models use.

## Quantized weights

GGUF files run as they are. Weights stay in their block formats on the GPU and are decoded exactly as llama.cpp decodes them, by a path chosen for the batch:

| Rows in the step | Path |
|---|---|
| 1 to 4 | stream the weights once |
| a few | tensor cores, decoding weights straight into registers |
| more | int8 tensor cores, with activations quantized as llama.cpp does |

The common formats are also reordered at load into the exact order the kernels read them, which roughly doubles their speed at small batches.

FP8 runs on cuBLASLt, with block scales on Hopper and MXFP8 on Blackwell. On RTX 50-series cards FP4 runs natively, and NVIDIA's NVFP4 checkpoints are converted to MXFP4 at load.

## AMD

`--features rocm` builds the same kernels for the MI325X. The hot paths have their own versions in `kernels/amd/`, written for AMD's 64-lane waves and matrix instructions. FP4 is NVIDIA only, and FP8 on AMD covers weights only.

Decoding streams weights with a kernel modelled on vLLM's, and larger products go to hipBLASLt, with solutions tuned for the MI325X's shapes (`tuning/`), up to 1.9 times as fast as its own picks. GGUF and FP8 weights also keep a bf16 copy for batches, made once at load; FP8 decoding reads its E4M3 rows directly, with power-of-two scales so both copies hold the same values. Adapters decode with their own kernels and run in order on one stream, since a second stream changed outputs on AMD.

What costs time on an MI325X (details in the [receipt](receipts/mi325x.json)):

| Cost | Measured | Response |
|---|---|---|
| Each kernel launch | 1.5 µs | Fuse only where parallelism stays |
| A load waiting on a branch | ~0.3 µs | Issue loads first |
| A device-wide fence | 83 µs | No hand-offs between blocks |

## Tuning per GPU

Some choices depend more on the GPU than on the model, so they are measured on the GPU itself. `pocket-engine tune` loads a model, records every matrix product it runs, and times the alternatives on that GPU with the model's own weights, replayed from CUDA graphs as decoding runs them. The results go into `tuning/cuda/`, one table per GPU, which the server reads at startup.

| Setting | Alternatives |
|---|---|
| Matrix products | cuBLASLt's own candidates against cuBLAS's default, per shape and range of rows |
| Linear-attention prompts | how many blocks share each head's columns |
| Linear-attention decoding | state buffers per block and blocks per SM |
| GGUF matrices | the batch sizes at which they switch between paths |

A setting replaces the built-in choice only where it is faster everywhere in its range, at least 3% faster on average, gives the same output, and holds up when timed a second time. Everything else keeps the built-in rule. Each table also records the cuBLAS version its algorithms came from; with another version the server uses cuBLAS's own choices.

## Loading

Weights are read in parallel straight into pinned memory while the GPU copies the previous buffer; memory-mapped reads were about five times slower. The tokenizer and the structured-output tables load at the same time. A 9B model is ready to serve in under 2 seconds.
