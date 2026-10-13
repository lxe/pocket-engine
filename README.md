# pocket-engine

An inference server for Qwen3.5 models, written from scratch in Rust and CUDA. It batches requests continuously, captures decode steps in CUDA graphs and serves an OpenAI-compatible chat API, for text and images, from bf16, FP8, FP4 or GGUF weights. It is ready 2 to 5 seconds after launch.

On one H100, against vLLM 0.31.0 on the same machine:

- **Throughput.** Qwen3.5-2B serves 7,530 output tokens per second with 64 requests in flight, against vLLM's 5,172; it leads on every dense model in bf16 up to 9B and in FP8.
- **First token.** A short prompt's first token arrives in 13.5 ms on Qwen3.5-9B, against vLLM's 57.5. Long prompts prefill at about the same speed: a 64k-token prompt takes 2.29 s on 9B, against 2.28, and 7.27 s on 27B, against 6.93.
- **Speculative decoding.** With the checkpoint's multi-token-prediction layer drafting two tokens a step (`--mtp`), one 9B stream decodes at 293 tokens per second, against vLLM's 253 with the same drafting, and 64 in flight reach 4,155 against 2,882.

[Design and measurements](docs/design.md); [against vLLM and llama.cpp on an H100](docs/h100-comparison.md).

## Performance

Output tokens per second with 64 requests in flight on one H100, from `scripts/load.py` (streamed chat, prompts of 60 to 200 tokens, 16 to 64 output tokens). vLLM 0.31.0 runs with its prefix cache off, and llama.cpp with its prompt cache off; [the comparison](docs/h100-comparison.md) has both settings, every load from 1 to 64, startup times and the method.

| Model | Format | pocket-engine | vLLM | llama.cpp |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | bf16 | **9,114** | 6,057 | 752 |
| Qwen3.5-2B | bf16 | **7,530** | 5,172 | 656 |
| Qwen3.5-4B | bf16 | **3,736** | 3,291 | 471 |
| Qwen3.5-9B | bf16 | **3,408**¹ | 2,938¹ | 390 |
| Qwen3.8-27B | bf16 | 978¹ | **1,088**¹ | 158 |
| Qwen3.5-9B | FP8 | **3,523** | 3,460 | — |
| Qwen3.8-27B | FP8 | **1,314**¹ | 1,302¹ | — |
| Qwen3.5-9B | bf16, 2 drafts a step | **4,155**² | 2,882² | — |
| Qwen3.8-27B | bf16, 2 drafts a step | **1,124**² | 897² | — |
| Qwen3.5-9B | Q4_K_M | **2,315** | does not load | 395 |
| Qwen3.8-27B | Q4_K_M | **675** | does not load | 156 |

¹ From [a separate H100 run](docs/h100-comparison.md#27b-multi-token-prediction-and-shared-prefixes), with vLLM at `--max-num-seqs 256`. ² Both engines drafting two tokens a step with their multi-token prediction (pocket-engine `--mtp 2`, vLLM's `mtp` method with 2 speculative tokens); [all loads and draft counts](docs/design.md#multi-token-prediction).

Time to first token, one request at a time, bf16 ([receipt](docs/receipts/h100-mtp-ttft.json)):

| Prompt | Qwen3.5-9B | vLLM | Qwen3.8-27B | vLLM |
|---|---:|---:|---:|---:|
| 60 to 200 tokens (median) | **13.5 ms** | 57.5 ms | **36.5 ms** | 97.4 ms |
| 8k tokens | **0.29 s** | 0.34 s | 0.81 s | **0.79 s** |
| 32k tokens | **1.06 s** | 1.10 s | 3.28 s | **3.22 s** |
| 64k tokens | 2.29 s | **2.28 s** | 7.27 s | **6.93 s** |

Prompts run on a FlashAttention-3-style warp-specialized flash kernel and a two-pass chunked linear-attention recurrence, both written for this engine ([long contexts](docs/design.md#long-contexts)).

## Supported

| | |
|---|---|
| Models | Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B (Qwen3.5- and 3.6-27B share its architecture), with images. All dense |
| Not supported | Mixture-of-experts models (Qwen3.5-35B-A3B and larger), video, GPTQ and AWQ checkpoints |
| Weights | Hugging Face safetensors checkpoints in bf16 (f32 tensors are converted), Qwen's FP8 (`Qwen3.8-27B-FP8`, `Qwen3.5-27B-FP8`) or NVIDIA ModelOpt's NVFP4 and per-tensor FP8, or a GGUF file (text only; projector files are not read). GGUF files with a multi-token-prediction layer load; the layer is skipped (`--mtp` needs a safetensors checkpoint) |
| Images | JPEG, PNG, WebP, GIF or BMP, as `data:` or `http(s)` URLs. Images are resized so each side is a multiple of 32 pixels, and every 32×32 block becomes one prompt token. The area is held between 65,536 and `--max-image-pixels` (default 4,194,304). |
| Quantization | GGUF weights stay quantized on the GPU: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, mixed freely within a file. Their activations run in bf16 with f32 accumulation |
| FP8 | E4M3 weights and activations on H100, H200 and RTX 50-series: Qwen's FP8 checkpoints, or any bf16 checkpoint with `--fp8`. Block scales of 128×128 per weight tile and 128 per activation row on Hopper; MXFP8 (a scale per 32 values) on Blackwell. Images, samplers and CUDA graphs work with it. The KV cache can be FP8 too (`--kv-fp8`) |
| FP4 | RTX 50-series only: MXFP4 weights (E2M1, a power-of-two scale per 32) times MXFP8 activations on block-scaled tensor cores. NVIDIA's NVFP4 checkpoints (`nvidia/Qwen3.8-27B-NVFP4`) are re-quantized to MXFP4 at load; `--fp4` quantizes a bf16 checkpoint's MLP to FP4 and the rest to FP8 |
| Context | Up to `--max-context` tokens per request (default 4,096). On an RTX 5090, Qwen3.5-9B takes 0.66 s to a first token at 8k, 2.97 s at 32k and 7.04 s at 64k, and decodes at 77 to 85 tok/s at every length |
| Prefix cache | Prompts that share a beginning (a system prompt, or a conversation's earlier turns) reuse its KV cache and recurrent state. On by default |
| Speculative decoding | `--mtp` drafts tokens with the checkpoint's multi-token-prediction layer and verifies them in the next step, for greedy and sampled requests alike (not with penalties or `logit_bias`). Qwen3.5-9B with two drafts: 293 tok/s for one stream instead of 150 on an H100 (vLLM's MTP: 253), 152 instead of 85 on an RTX 5090 |
| Sampling | Greedy, or temperature with `top_k`, `top_p`, `min_p` and `seed`; `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`; `stop` strings |
| GPUs | NVIDIA, built per architecture: `90a` (H100), `120a` (RTX 50-series); others via `CUDA_COMPUTE_CAP` |
| Driver | CUDA 13 (driver 580 or newer); cuBLAS 13 at run time |

## Install

You need Rust 1.98 or newer and the CUDA 13 toolkit (`nvcc`).

```sh
git clone https://github.com/lxe/pocket-engine && cd pocket-engine
CUDA_COMPUTE_CAP=90a cargo build --release --features cuda   # 120a for an RTX 5090
```

The binary is `target/release/pocket-engine`. It loads `libcuda`, `libcublas.so.13` and `libcublasLt.so.13` at run time, so they must be on the library path.

## Run

Download a model:

```sh
hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
```

The model directory needs `config.json`, the safetensors files, `tokenizer.json` and `chat_template.jinja`. `preprocessor_config.json` is optional: without it, Qwen's image defaults apply.

A GGUF file serves on its own: its metadata carries the configuration, vocabulary and chat template.

```sh
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models
pocket-engine serve --model models/Qwen3.5-9B-Q4_K_M.gguf --name qwen3.5-9b
```

Serve it:

```sh
pocket-engine serve --model models/qwen3.5-2b --name qwen3.5-2b --bind 0.0.0.0:8000
```

Then call it:

```sh
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.5-2b", "max_tokens": 64, "temperature": 0,
  "messages": [{"role": "user", "content": "Hello!"}]}'
```

The server answers on three paths:

| Path | What it does |
|---|---|
| `POST /v1/chat/completions` | Streamed (`"stream": true`, with `stream_options.include_usage`) or whole responses. Thinking is off. Sampling fields are listed below. Images go in content parts: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`. |
| `GET /v1/models` | Lists the served model. |
| `GET /health` | Reports running and waiting requests and free KV blocks. Returns 503 once the GPU has failed. |

Sampling fields follow OpenAI, plus the vLLM extensions `top_k`, `min_p` and `repetition_penalty`. All of them run on the GPU. A request without `temperature` samples at 1.0.

| Field | Range | Effect |
|---|---|---|
| `temperature` | 0 to 100 | 0 picks the most likely token |
| `top_k` | -1, 0 or more | keep the k most likely tokens; -1 and 0 keep all |
| `top_p` | (0, 1] | keep the smallest set of tokens holding this much probability |
| `min_p` | 0 to 1 | keep tokens at least this fraction as likely as the top one |
| `presence_penalty`, `frequency_penalty` | -2 to 2 | subtract from the logits of tokens already generated: once, and per occurrence |
| `repetition_penalty` | above 0 | divide positive logits, and multiply negative ones, for tokens in the prompt or output |
| `logit_bias` | -100 to 100 | add to chosen token ids' logits |
| `stop` | up to 16 strings | end the response before the first occurrence |
| `seed` | integer | makes sampling reproducible |

Filters apply in order: min_p, then top_k, then top_p. Each one judges the tokens the previous one kept.

`serve` takes these options:

| Option | Default | Meaning |
|---|---|---|
| `--max-seqs` | 256 | Most requests decoding at once; lowered at startup so their recurrent state fits in half of the free memory (25 MB per request on 9B, 75 MB on 27B) |
| `--max-batch-tokens` | 4096 | Most tokens in one step, decode rows plus prompt chunks |
| `--max-context` | 4096 | Longest prompt plus output, in tokens |
| `--prefix-cache` | a quarter of `--max-seqs`, 2 to 64 | Prompt prefixes to keep for reuse, each holding its KV blocks and a snapshot of the recurrent state; 0 turns the cache off |
| `--mtp [N]` | off (2 when given without N) | Draft N tokens a step with the multi-token-prediction layer. Sequences verify drafts while the step stays within 96 rows (256 on an H100) and free state slots can hold the states after each draft, so drafting needs no memory of its own and stops under heavy load |
| `--memory-fraction` | 0.9 | Share of GPU memory to use; the KV cache gets what is left after the weights |
| `--fp8` | off | Quantize a bf16 checkpoint's projections to FP8 at load (Hopper or Blackwell). FP8 checkpoints need no flag |
| `--fp4` | off | Quantize a bf16 checkpoint's MLP to FP4 and its other projections to FP8 at load (RTX 50-series) |
| `--kv-fp8` | off | Store the KV cache as FP8 (E4M3 with a scale per token and head): about twice the tokens in the same memory |
| `--state-f32` | off | Store the linear-attention state as f32 instead of bf16, at twice the memory per request. The math is f32 either way |
| `--wait-for PATH` | none | Initialize the GPU, then wait for `PATH` to exist before reading the model. A downloader can create the file when it finishes, so device setup overlaps the download |
| `--no-graphs` | off | Run every step without CUDA graphs |
| `--no-vision` | off | Serve text only; the vision encoder is not loaded |
| `--max-image-pixels` | 4194304 | Largest image after resizing |
| `--image-pool` | 16384 | Image embeddings that can wait at once for their prompt steps; also the most one request can carry |

## Test

```sh
cargo test                                   # scheduler, config and layout tests; no GPU needed
POCKET_ENGINE_MODEL=models/qwen3.5-2b \
  CUDA_COMPUTE_CAP=120a cargo test --release --features cuda
```

With a model set, `tests/gpu.rs` runs eight checks against it. The model can be Qwen3.5 0.8B, 2B, 4B or 9B or Qwen3.8-27B, as a directory or a GGUF file. `POCKET_ENGINE_FP8=1` runs the suite with `--fp8`, `POCKET_ENGINE_FP4=1` with `--fp4`, `POCKET_ENGINE_KV_FP8=1` with `--kv-fp8`, `POCKET_ENGINE_STATE_F32=1` with `--state-f32` and `POCKET_ENGINE_MTP=N` with `--mtp N`:

- its predictions match stored Transformers reference outputs;
- image prompts tokenize exactly as Transformers' processor does, and their predictions match too;
- batching does not change predictions beyond rounding: the same tokens scored alone and among other requests agree except at near ties;
- a prompt extending a cached one gives exactly the output it gives without the prefix cache, and scored over whole prompt steps it agrees with an uncached run except for rounding;
- with three drafts a step, greedy output departs from the model's own picks (that output scored teacher-forced) only at near ties, and more than a third of drafts are accepted;
- the samplers keep exactly the right tokens and draw them in proportion to the model's probabilities;
- penalties cut repetition;
- the HTTP API streams and reports usage correctly.

Quantized weights are held to looser bars on the accuracy checks. For GGUF, 85% of picks agree with bf16 Transformers and the median log-probability difference stays under 0.1; the image check is skipped. For FP8, 93% agree with a median under 0.03, and 95% with a median under 0.01 on images; FP4 takes the GGUF bar, and 90% with a median under 0.05 on images. An FP8 KV cache takes the FP8 bar too. The batching check allows rounding because GEMM kernels differ by batch size in the last bit; with FP8 or FP4 activations or an FP8 KV cache, which magnify that about as much as their own error, it takes 90% with a median under 0.03. For the same reason the drafting check lets quantized runs depart at up to 5% of positions.

To check a model or prompt that isn't in the fixtures, `scripts/accuracy.py` makes new reference data with Transformers:

```sh
uv venv && uv pip install torch transformers
.venv/bin/python scripts/accuracy.py reference models/qwen3.5-2b out
pocket-engine score --model models/qwen3.5-2b out.pairs.json out.ours.json
python scripts/accuracy.py compare out.ref.json out.ours.json
```

`accuracy.py long` makes reference data over three prompts of about 3,300 tokens, for checking the KV cache at length. `scripts/image_reference.py` does the same for image prompts.

`scripts/quant_check.py` checks GPU dequantization bit for bit against gguf-py (llama.cpp's reference) for every quantized format in the GGUF files it is given. `pocket-engine bench-linear` times one quantized matrix (and, for Q4_0 and Q4_K, its tiled layout) against the same matrix in bf16 on cuBLAS, with the L2 cache cold.

`scripts/load.py URL MODEL OUT.json` is the load test behind the throughput numbers.

## License

MIT
