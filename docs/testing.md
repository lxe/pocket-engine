# Testing

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

## Reference data and tools

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
