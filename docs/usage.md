# Usage

The binary is `target/release/pocket-engine`. It needs `libcublas.so.13` and `libcublasLt.so.13` on the library path, or for AMD, ROCm 7's `libamdhip64` and `libhipblas`.

## Models and hardware

Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B, with images. Mixture-of-experts models, video, GPTQ and AWQ are not supported.

| GPU | Build with |
|---|---|
| H100, H200 | `CUDA_COMPUTE_CAP=90a` |
| B200 | `CUDA_COMPUTE_CAP=100a` |
| RTX 50-series | `CUDA_COMPUTE_CAP=120a` |
| AMD MI325X (gfx942) | `--features rocm` in place of `--features cuda` (`ROCM_ARCH`, `ROCM_PATH` to override) |

## Weights

| Format | GPUs | Load |
|---|---|---|
| bf16 | all | a Hugging Face checkpoint |
| FP8 | H100, H200, RTX 50-series | an FP8 checkpoint, or any bf16 one with `--fp8` |
| FP4 | RTX 50-series | an NVFP4 checkpoint, or any bf16 one with `--fp4` |
| GGUF | all | a `.gguf` file (Q4_K_M, Q8_0, IQ4_XS and the other common types); text only |

`--ngram` drafts by prompt lookup: the tokens that followed the last earlier occurrence of the latest 2 to 4 tokens, in the prompt or the output. It needs no draft layer and pays off when output copies its input (extraction, editing, quoting): repeating a passage runs 3.7 times as fast on 9B and 4 times on 27B. Steps without a match decode as usual, so it costs nothing elsewhere. It combines with `--mtp`, which then guesses where the lookup finds nothing.

`--mtp` needs a checkpoint or GGUF file that includes the draft layer; among GGUF files, Unsloth's Qwen3.8-27B UD quantizations do. Use it for a few streams at a time: it speeds up each one (64 to 112 tok/s on 27B on an RTX 5090) and costs a few percent at high load.

GGUF weights in 4- to 6-bit formats run batches against int8 activations, which serves about 20% more tokens/s under load on RTX GPUs and a few percent more on H100, with the same agreement with Transformers. `--no-int8` keeps activations in bf16.

`--kv-fp8` stores the KV cache in FP8, which fits about twice the tokens. How each format runs is in [the design notes](design.md#quantized-weights).

Images can be JPEG, PNG, WebP, GIF or BMP, as `data:` or `http(s)` URLs.

## Model files

A model directory needs `config.json`, the safetensors files, `tokenizer.json` and `chat_template.jinja`. A GGUF file serves on its own:

```sh
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models
pocket-engine serve --model models/Qwen3.5-9B-Q4_K_M.gguf --bind 0.0.0.0:8000
```

## API

| Path | |
|---|---|
| `POST /v1/chat/completions` | Chat, streamed or not. Images go in `image_url` content parts. |
| `GET /v1/models` | The served model and its adapters. |
| `POST /v1/load_lora_adapter` | Load an adapter: `{"lora_name": ..., "lora_path": ...}`. |
| `POST /v1/unload_lora_adapter` | Unload one no request is using: `{"lora_name": ...}`. |
| `GET /health` | Running and waiting requests, free KV blocks. |

## LoRA adapters

Serve fine-tuned adapters over one base model, as vLLM does: requests pick one with `"model": "<adapter name>"`, and requests for different adapters and the base run in the same batches.

```sh
pocket-engine serve --model models/Qwen3.5-9B-Q4_K_M.gguf --lora support=adapters/support --lora billing=adapters/billing.gguf
```

Adapters are PEFT directories (`adapter_config.json` and `adapter_model.safetensors`) or llama.cpp GGUF adapters, over any base format. `--max-loras` adapters sit on the GPU at once; others wait on the host and swap in when a slot frees. On an RTX 5090, adapters cost 7% at one request on 9B Q4_K_M and about 25% at 16 to 64. Without adapters there is no cost.

Requests accept the OpenAI sampling fields (`temperature`, `top_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`) plus vLLM's `top_k`, `min_p` and `repetition_penalty`. The default temperature is 1.0. Prompts that share a beginning reuse its cache.

### Structured output and log-probabilities

- `response_format`: `{"type": "json_object"}`, or `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` for output that follows a JSON Schema. vLLM's `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar` (Lark) work too. Grammars run on [llguidance](https://github.com/guidance-ai/llguidance).
- `logprobs: true` returns each token's log-probability in OpenAI's `choices[].logprobs.content`, with up to 20 alternatives by `top_logprobs`. They come from the logits sampling saw: after penalties, `logit_bias` and the grammar, before temperature (vLLM's `processed_logprobs`).

Neither turns on speculative decoding for the request. On an RTX 5090 with 9B Q4_K_M, JSON output costs 4 to 12% of throughput and log-probabilities 2 to 4%.

## Options

| Option | Default | |
|---|---|---|
| `--model PATH` | required | Model directory or GGUF file |
| `--name NAME` | file or directory name | Model name in the API |
| `--bind ADDR` | `127.0.0.1:8190` | Listen address |
| `--device N` | 0 | GPU index |
| `--max-context N` | 4096 | Longest prompt plus output |
| `--max-seqs N` | 256 | Most requests at once (lowered to fit memory) |
| `--max-batch-tokens N` | 4096 | Most tokens per step |
| `--mtp [N]` | off | Draft N tokens a step (2 if N is omitted) |
| `--ngram [N]` | off | Draft N tokens a step by prompt lookup (4 if N is omitted) |
| `--prefix-cache N` | `--max-seqs` / 4 | Prompt prefixes to keep; 0 turns caching off |
| `--memory-fraction F` | 0.9 | Share of GPU memory to use |
| `--fp8` | off | Quantize a bf16 checkpoint to FP8 |
| `--fp4` | off | Quantize a bf16 checkpoint to FP4 (RTX 50-series) |
| `--kv-fp8` | off | FP8 KV cache |
| `--state-f32` | off | f32 linear-attention state (twice the memory) |
| `--no-int8` | off | bf16 activations for GGUF weights (see above) |
| `--lora NAME=PATH` | none | Serve an adapter; repeat for more |
| `--max-loras N` | 4 with `--lora`, else off | Adapters on the GPU at once; also allows loading them at runtime |
| `--max-lora-rank N` | 16 | Largest adapter rank |
| `--no-vision` | off | Text only |
| `--max-image-pixels N` | 4194304 | Largest image after resizing |
| `--image-pool N` | 16384 | Image embeddings waiting at once |
| `--wait-for PATH` | none | Wait for `PATH` to exist before loading, so GPU setup overlaps a download |
| `--no-graphs` | off | Disable CUDA graphs |
