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

Quantized runs allow slightly larger differences from Transformers.

## Comparing with Transformers yourself

For a model or prompt the tests don't cover:

```sh
uv venv && uv pip install torch transformers
.venv/bin/python scripts/accuracy.py reference models/qwen3.5-2b out    # Transformers' answers
target/release/pocket-engine score --model models/qwen3.5-2b out.pairs.json out.ours.json
python scripts/accuracy.py compare out.ref.json out.ours.json             # agreement report
```
