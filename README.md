# pocket-engine

An inference server for Qwen3.5 and Qwen3.8 models, written from scratch in Rust and CUDA, with an OpenAI-compatible chat API.

- **Starts in seconds.** Serving 3 seconds after launch, where vLLM takes 1.5 to 3 minutes.
- **Fast first tokens.** 13.5 ms for a short prompt against vLLM's 57.5 ms.
- **Higher throughput** on most models, with optional speculative decoding from the checkpoint's own multi-token-prediction layer.
- **bf16, FP8, FP4 and GGUF weights,** and images.

Qwen3.5-9B on one H100, against vLLM 0.31.0 on the same machine:

| | pocket-engine | vLLM |
|---|---:|---:|
| Launch to serving | **2.6–3.5 s** | 89–165 s |
| First token, short prompt | **13.5 ms** | 57.5 ms |
| First token, 64k-token prompt | 2.29 s | **2.28 s** |
| Output tokens/s, 64 requests in flight | **3,534** | 2,567 |

H200, B200, RTX 5090, Qwen3.8-27B and other formats are in [benchmarks](docs/benchmarks.md).

## Quick start

Requires an NVIDIA GPU with driver 580 or newer, the CUDA 13 toolkit and Rust 1.98 or newer.

```sh
git clone https://github.com/lxe/pocket-engine && cd pocket-engine
CUDA_COMPUTE_CAP=90a cargo build --release --features cuda   # 100a for B200, 120a for RTX 50-series

hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
target/release/pocket-engine serve --model models/qwen3.5-2b --name qwen3.5-2b --bind 0.0.0.0:8000
```

```sh
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "qwen3.5-2b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Supported

- **Models:** Qwen3.5 0.8B, 2B, 4B and 9B, and Qwen3.8-27B, with images. Dense models only.
- **GPUs:** H100, H200, B200 and RTX 50-series.
- **Weights:** bf16, FP8, FP4 (RTX 50-series) and GGUF.

## Documentation

- [Usage](docs/usage.md): API, sampling, server options, weight formats
- [Benchmarks](docs/benchmarks.md): every GPU, model and format, against vLLM
- [Design](docs/design.md): how it works, from scheduling to kernels
- [Accuracy](docs/accuracy.md): comparisons with Transformers
- [Testing](docs/testing.md): running the test suite

## License

MIT
