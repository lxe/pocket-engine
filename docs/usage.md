# Usage

The server binary is `target/release/pocket-engine`; the examples write `pocket-engine`. It loads `libcuda`, `libcublas.so.13` and `libcublasLt.so.13` at run time, so they must be on the library path.

## Models and hardware

**Models.** Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B (Qwen3.5- and 3.6-27B share its architecture), all dense, with images. Mixture-of-experts models (Qwen3.5-35B-A3B and larger), video, and GPTQ and AWQ checkpoints are not supported.

**GPUs.** The build targets one architecture, set by `CUDA_COMPUTE_CAP`; other NVIDIA GPUs build with their own.

| GPU | `CUDA_COMPUTE_CAP` | Notes |
|---|---|---|
| H100, H200 | `90a` (the default) | FP8 with block scales |
| B200 | `100a` | Prompts run on generic tensor-core kernels; no Blackwell-specific attention or recurrence yet |
| RTX 50-series | `120a` | FP8 as MXFP8, and FP4 |

**Weights.**

- **bf16.** Hugging Face safetensors checkpoints; f32 tensors are converted.
- **FP8** (H100, H200 and RTX 50-series). E4M3 weights and activations: Qwen's FP8 checkpoints (`Qwen3.8-27B-FP8`, `Qwen3.5-27B-FP8`), NVIDIA ModelOpt's per-tensor FP8, or any bf16 checkpoint with `--fp8`. Block scales of 128×128 per weight tile and 128 per activation row on Hopper; MXFP8 (a scale per 32 values) on Blackwell. Images, samplers and CUDA graphs work with it. The KV cache can be FP8 too (`--kv-fp8`).
- **FP4** (RTX 50-series only). MXFP4 weights (E2M1, a power-of-two scale per 32) times MXFP8 activations on block-scaled tensor cores. NVIDIA ModelOpt's NVFP4 checkpoints (`nvidia/Qwen3.8-27B-NVFP4`) are re-quantized to MXFP4 at load; `--fp4` quantizes a bf16 checkpoint's MLP to FP4 and the rest to FP8.
- **GGUF**, text only (projector files are not read). Weights stay quantized on the GPU: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS, mixed freely within a file. Their activations run in bf16 with f32 accumulation. GGUF files with a multi-token-prediction layer load, but the layer is skipped: `--mtp` needs a safetensors checkpoint.

**Images.** JPEG, PNG, WebP, GIF or BMP, as `data:` or `http(s)` URLs. Images are resized so each side is a multiple of 32 pixels, and every 32×32 block becomes one prompt token. The area is held between 65,536 and `--max-image-pixels` (default 4,194,304).


## Model files

A model directory needs `config.json`, the safetensors files, `tokenizer.json` and `chat_template.jinja`. `preprocessor_config.json` is optional: without it, Qwen's image defaults apply.

A GGUF file serves on its own: its metadata carries the configuration, vocabulary and chat template.

```sh
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models
pocket-engine serve --model models/Qwen3.5-9B-Q4_K_M.gguf --name qwen3.5-9b --bind 0.0.0.0:8000
```

## API

| Path | What it does |
|---|---|
| `POST /v1/chat/completions` | Streamed (`"stream": true`, with `stream_options.include_usage`) or whole responses. Thinking is off. Images go in content parts: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`. |
| `GET /v1/models` | Lists the served model. |
| `GET /health` | Reports running and waiting requests and free KV blocks. Returns 503 once the GPU has failed. |

Prompts that share a beginning, such as a system prompt or a conversation's earlier turns, reuse its KV cache and recurrent state.

## Sampling

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

Filters apply in order: min_p, then top_k, then top_p. Each one judges the tokens the previous one kept. Requests with penalties or `logit_bias` do not draft tokens under `--mtp`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | required | Model directory or GGUF file |
| `--name NAME` | `qwen3.5` | Model name reported by the API |
| `--bind ADDR` | `127.0.0.1:8190` | Address to listen on |
| `--device N` | 0 | CUDA device index |
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
