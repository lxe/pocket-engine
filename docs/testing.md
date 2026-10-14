# Testing

## Without a GPU

```sh
cargo test
```

This covers the scheduler, configs and memory layout, and runs in seconds.

## On a GPU

Point the tests at a model and build for your GPU (`90a` for H100 or H200, `100a` for B200, `120a` for RTX 50-series):

```sh
hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
POCKET_ENGINE_MODEL=models/qwen3.5-2b CUDA_COMPUTE_CAP=120a cargo test --release --features cuda
```

Any supported model works, as a directory or a GGUF file. The run takes about a minute after the build. It checks that:

- outputs match Transformers, for text and images;
- batching and the prefix cache change outputs only by rounding;
- speculative decoding gives the same tokens as decoding without it;
- sampling and penalties behave as specified;
- the HTTP API streams and reports usage.

## Testing an option

Add one of these to the command above to run the same checks with that option on:

| Variable | Option |
|---|---|
| `POCKET_ENGINE_FP8=1` | `--fp8` |
| `POCKET_ENGINE_FP4=1` | `--fp4` |
| `POCKET_ENGINE_KV_FP8=1` | `--kv-fp8` |
| `POCKET_ENGINE_STATE_F32=1` | `--state-f32` |
| `POCKET_ENGINE_MTP=3` | `--mtp 3` |
| `POCKET_ENGINE_NO_INT8=1` | `--no-int8` |
| `POCKET_ENGINE_LORA=DIR` | checks that adapter against `tests/fixtures/<size>.lora.*.json` |

Quantized runs allow slightly larger differences from Transformers.

## Comparing with Transformers yourself

For a model or prompt the tests don't cover:

```sh
uv venv && uv pip install torch transformers
.venv/bin/python scripts/accuracy.py reference models/qwen3.5-2b out    # Transformers' answers
target/release/pocket-engine score --model models/qwen3.5-2b out.pairs.json out.ours.json
python scripts/accuracy.py compare out.ref.json out.ours.json             # agreement report
```

## Profiling

`POCKET_PROFILE=1` times every kernel and cuBLAS call with CUDA events and adds GPU time by kernel to the server's 10-second report. CUDA graphs hide the kernels inside them, so profile with `--no-graphs`. It works where `ncu` cannot read counters, as in most rented containers.

`POCKET_PROFILE=shapes` also splits cuBLAS matmuls by shape: output by input columns, and rows rounded down to a power of two.

`POCKET_NO_WS=1` sends prompts on Hopper and B200 to the portable flash kernel instead of the warp-specialized one, for A/B runs.

`POCKET_CHECK_HEADS=1 pocket-engine bench-attn ... --check` also reports the largest difference per head, and for heads off by more than 5e-3 which tokens and columns: a wrong warp shows up as a block of 32 tokens.

`scripts/perf_check.py` reruns the matmul, attention and linear-attention benchmarks and short load tests against a stored baseline for the GPU, and fails on a regression.

