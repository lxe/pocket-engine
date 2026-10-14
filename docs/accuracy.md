# Accuracy

Every comparison here is teacher-forced against Transformers in bf16: the same prompt and continuation, comparing the top pick and the log-probability at each position. "Within one bf16 step" means the two picks were tied up to bf16 rounding. `tests/gpu.rs` checks the same thing on every run, and `scripts/accuracy.py` repeats it for any model or prompt.

## bf16

| Model | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|
| Qwen3.5-0.8B | 213 of 222 (all misses within one bf16 step) | 0.0032, 0.056 |
| Qwen3.5-2B | 214 of 217 (all within one step) | 0.0015, 0.043 |
| Qwen3.5-4B | 209 of 210 (within one step) | 0.0004, 0.032 |
| Qwen3.5-9B | 219 of 220 (within one step) | 0.0003, 0.032 |
| Qwen3.8-27B | 211 of 213 (both within one step) | 0.0004, 0.035 |

With images, prompt tokens match Transformers' processor exactly:

| Model | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|
| Qwen3.5-0.8B | 85 of 87 | 0.0007, 0.054 |
| Qwen3.5-2B | 87 of 87 | 0.0003, 0.088 |
| Qwen3.5-4B | 62 of 62 | 0.0005, 0.043 |
| Qwen3.5-9B | 85 of 85 | 0.0000, 0.040 |
| Qwen3.8-27B | 91 of 91 | 0.0001, 0.050 |

## Quantized weights

Against the same bf16 references, so these show each format's own error.

| Weights | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|
| Qwen3.8-27B-FP8 checkpoint | 209 of 213 | 0.0009, 0.088 |
| Qwen3.5-27B-FP8 checkpoint | 215 of 217 | 0.0003, 0.066 |
| Qwen3.5-9B, `--fp8` | 213 of 220 | 0.0011, 0.16 |
| Qwen3.8-27B Q4_K_M (GGUF) | 207 of 213 | 0.0019, 0.17 |
| Qwen3.8-27B UD-Q4_K_XL (GGUF) | 210 of 213 | 0.0011, 0.16 |
| Qwen3.8-27B NVFP4 checkpoint, as MXFP4 | 197 of 213 | 0.0053, 0.42 |
| Qwen3.5-0.8B Q8_0 (GGUF) | 217 of 222 | 0.0035, 0.051 |
| Qwen3.5-0.8B Q4_K_M (GGUF) | 199 of 222 | 0.044, 0.57 |

FP8 receipts: [fp8.json](receipts/fp8.json).

## Long prompts and the FP8 KV cache

Prompts of about 3,300 tokens, so most of the context comes from the KV cache:

| Model | KV cache | Top-1 agreement | Median, p95 log-prob difference |
|---|---|---|---|
| Qwen3.5-2B | bf16 | 285 of 288 | 0.0012, 0.042 |
| Qwen3.5-2B | FP8 | 285 of 288 | 0.0024, 0.071 |
| Qwen3.5-9B | bf16 | 284 of 288 | 0.0005, 0.036 |
| Qwen3.5-9B | FP8 | 281 of 288 | 0.0007, 0.076 |

## Recurrent state in bf16

The linear-attention state is stored in bf16 between steps, which halves the memory per request. Over generations of up to 512 tokens it makes no measurable difference:

| Check | f32 state | bf16 state (default) |
|---|---|---|
| Qwen3.5-2B, 949 positions | 943 agree, median 0.0002 | 944, 0.0002 |
| Qwen3.5-9B, 1,293 positions | 1,291, 0.0000 | 1,291, 0.0000 |
| Qwen3.5-9B, 3,300-token prompts | 284 of 288, 0.0005 | 286, 0.0004 |
