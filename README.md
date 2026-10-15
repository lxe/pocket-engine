# pocket-engine

An inference server for Qwen3.5 and Qwen3.8 models, written from scratch in Rust and CUDA, with an OpenAI-compatible chat API. Runs on NVIDIA GPUs and on AMD's MI325X.

- **[Starts in 2 to 3 seconds.](docs/design.md#one-process-no-python)** A single Rust binary, no Python.
- **Fast first tokens.** 13.5 ms for a short prompt on an H100.
- **[Speculative decoding](docs/design.md#speculative-decoding)** with Qwen's built-in prediction layer. Same answers as without it, nearly twice as fast: 293 tokens/s for one 9B request on an H100.
- **[Drafting from the prompt.](docs/design.md#speculative-decoding)** Up to 4 times faster when answers quote or edit their input.
- **[Prefix caching for hybrid models.](docs/design.md#prefix-cache)** Each turn of a chat reuses the last one's work.
- **[GGUF support.](docs/design.md#quantized-weights)** Serve the quantized files you already have from llama.cpp, with no conversion, at 2,372 tokens/s for 9B on an RTX 5090.
- **[Load and serve adapters on a live server.](docs/design.md#lora-adapters)** Add or remove a fine-tune without a restart, while requests for the base model keep full speed.
- **bf16, FP8, FP4 and GGUF weights;** JSON output, log-probabilities and images.

Output tokens per second against vLLM 0.31.0 on the same GPU model. Each cell shows one request / 64 requests at once.

| GPU | Model | pocket-engine | vLLM |
|---|---|---:|---:|
| B200 | Qwen3.5-9B | **230** / **5,408** | 224 / 4,782 |
| B200 | Qwen3.8-27B | 86 / 1,766 | **87** / **1,803** |
| H100 | Qwen3.5-9B | **150** / **3,534** | 143 / 2,567 |
| H100 | Qwen3.8-27B | **50** / 1,069 | 45 / **1,092** |
| MI325X | Qwen3.5-9B | **225** / **4,103** | 207 / 3,273 |
| MI325X | Qwen3.8-27B | 74 / **1,269** | **76** / 1,166 |

All bf16. More GPUs and formats, first-token times, and the runs behind every number are in [benchmarks](docs/benchmarks.md).

## Quick start

Requires Rust 1.98 or newer and either an NVIDIA GPU with driver 580 or newer and the CUDA 13 toolkit, or an AMD MI325X with ROCm 7 (hipcc, HIP and hipBLAS).

```sh
git clone https://github.com/lxe/pocket-engine && cd pocket-engine
CUDA_COMPUTE_CAP=90a cargo build --release --features cuda   # 100a for B200, 120a for RTX 50-series
# cargo build --release --features rocm                       # AMD MI325X (gfx942), ROCm 7

hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
target/release/pocket-engine serve --model models/qwen3.5-2b --bind 0.0.0.0:8000
```

```sh
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen3.5-2b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Supported

- **Models:** Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B, with images. Dense models only.
- **GPUs:** H100, H200, B200 and RTX 50-series; AMD MI325X (ROCm).
- **Weights:** bf16, FP8 (NVIDIA), FP4 (RTX 50-series) and GGUF.
- **Adapters:** PEFT and llama.cpp LoRA, over any of those weights.

## Documentation

- [Usage](docs/usage.md): API, sampling, server options, weight formats
- [Benchmarks](docs/benchmarks.md): every GPU, model and format, against vLLM
- [Design](docs/design.md): how it works, from scheduling to kernels
- [Accuracy](docs/accuracy.md): comparisons with Transformers
- [Testing](docs/testing.md): running the test suite

## License

MIT
