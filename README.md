# pocket-engine

An inference server for Qwen3.5 models, written from scratch in Rust and CUDA. It batches requests continuously, captures decode steps in CUDA graphs and serves an OpenAI-compatible chat API. On one H100 it serves Qwen3.5-2B at 7,500 output tokens per second with 64 requests in flight, against vLLM's 5,500 on the same machine, and is ready 5 seconds after launch ([design and measurements](docs/design.md)).

## Supported

| | |
|---|---|
| Models | Qwen3.5 0.8B, 2B, 4B and 9B, with images. The 27B shapes (Qwen3.5, 3.6 and 3.8-27B) load but are not yet checked |
| Architecture | Dense. Mixture-of-experts Qwen3.5 models are not supported |
| Weights | Hugging Face safetensors checkpoints (bf16; f32 tensors are converted), or a GGUF file (text only; projector files are not read yet) |
| Images | JPEG, PNG, WebP, GIF or BMP, as `data:` or `http(s)` URLs. Images are resized so each side is a multiple of 32 pixels, and every 32×32 block becomes one prompt token. The area is held between 65,536 and `--max-image-pixels` (default 4,194,304). |
| Quantization | GGUF weights stay quantized on the GPU: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, mixed freely within a file. Activations run in bf16 with f32 accumulation; recurrent state is f32 |
| Context | Up to `--max-context` tokens per request (default 4,096) |
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
| `--max-seqs` | 256 | Most requests decoding at once; lowered at startup so their recurrent state fits in half of the free memory (49 MB per request on 9B) |
| `--max-batch-tokens` | 4096 | Most tokens in one step, decode rows plus prompt chunks |
| `--max-context` | 4096 | Longest prompt plus output, in tokens |
| `--memory-fraction` | 0.9 | Share of GPU memory to use; the KV cache gets what is left after the weights |
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

With a model directory set, `tests/gpu.rs` runs six checks against the model, which can be 0.8B, 2B, 4B or 9B:

- its predictions match stored Transformers reference outputs;
- image prompts tokenize exactly as Transformers' processor does, and their predictions match too;
- batching does not change greedy output;
- the samplers keep exactly the right tokens and draw them in proportion to the model's probabilities;
- penalties cut repetition;
- the HTTP API streams and reports usage correctly.

To check a model or prompt that isn't in the fixtures, `scripts/accuracy.py` makes new reference data with Transformers:

```sh
uv venv && uv pip install torch transformers
.venv/bin/python scripts/accuracy.py reference models/qwen3.5-2b out
pocket-engine score --model models/qwen3.5-2b out.pairs.json out.ours.json
python scripts/accuracy.py compare out.ref.json out.ours.json
```

`scripts/image_reference.py` does the same for image prompts.

`scripts/quant_check.py` checks GPU dequantization bit for bit against gguf-py (llama.cpp's reference) for every quantized format in the GGUF files it is given. `pocket-engine bench-linear` times one quantized matrix against the same matrix in bf16 on cuBLAS.

`scripts/load.py URL MODEL OUT.json` is the load test behind the throughput numbers.

## License

MIT
