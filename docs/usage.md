# Usage

The binary is `target/release/pocket-engine`. It needs `libcublas.so.13` and `libcublasLt.so.13` on the library path.

## Models and hardware

Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B, with images. Mixture-of-experts models, video, GPTQ and AWQ are not supported.

| GPU | Build with |
|---|---|
| H100, H200 | `CUDA_COMPUTE_CAP=90a` |
| B200 | `CUDA_COMPUTE_CAP=100a` |
| RTX 50-series | `CUDA_COMPUTE_CAP=120a` |

## Weights

| Format | GPUs | Load |
|---|---|---|
| bf16 | all | a Hugging Face checkpoint |
| FP8 | H100, H200, RTX 50-series | an FP8 checkpoint, or any bf16 one with `--fp8` |
| FP4 | RTX 50-series | an NVFP4 checkpoint, or any bf16 one with `--fp4` |
| GGUF | all | a `.gguf` file (Q4_K_M, Q8_0, IQ4_XS and the other common types); text only |

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
| `GET /v1/models` | The served model. |
| `GET /health` | Running and waiting requests, free KV blocks. |

Requests accept the OpenAI sampling fields (`temperature`, `top_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`) plus vLLM's `top_k`, `min_p` and `repetition_penalty`. The default temperature is 1.0. Prompts that share a beginning reuse its cache.

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
| `--prefix-cache N` | `--max-seqs` / 4 | Prompt prefixes to keep; 0 turns caching off |
| `--memory-fraction F` | 0.9 | Share of GPU memory to use |
| `--fp8` | off | Quantize a bf16 checkpoint to FP8 |
| `--fp4` | off | Quantize a bf16 checkpoint to FP4 (RTX 50-series) |
| `--kv-fp8` | off | FP8 KV cache |
| `--state-f32` | off | f32 linear-attention state (twice the memory) |
| `--no-int8` | off | bf16 activations for GGUF weights (see below) |
| `--no-vision` | off | Text only |
| `--max-image-pixels N` | 4194304 | Largest image after resizing |
| `--image-pool N` | 16384 | Image embeddings waiting at once |
| `--wait-for PATH` | none | Wait for `PATH` to exist before loading, so GPU setup overlaps a download |
| `--no-graphs` | off | Disable CUDA graphs |
