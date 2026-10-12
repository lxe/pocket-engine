# Design

## A step

Each step is one forward pass over a flat batch of tokens. Every running sequence contributes its next token. Prompt chunks of newly admitted requests fill the rest of the step's token budget (`--max-batch-tokens`, 4,096 by default). A new request therefore starts on the next step rather than waiting for the current batch to finish, and long prompts are spread over several steps.

The scheduler (`src/sched.rs`) has no GPU code. It admits requests, plans each step and writes the step's metadata: token ids, positions, KV slots, per-sequence state slots, block tables and sampling settings. A request is admitted only when KV blocks for its whole prompt plus `max_tokens` are free, so nothing is ever preempted.

The engine (`src/engine.rs`) copies that metadata to the device and runs the step. A step with no prompt tokens replays a CUDA graph captured at startup for the next padded batch size (1 to 256), which costs a single launch from the host. Other steps run eagerly, at about 220 kernel launches.

## Per layer

Each layer runs three or four cuBLAS GEMMs over all of the step's tokens, plus a few kernels of its own (`src/kernels/qwen35.cu`):

- **Fused projections.** One GEMM computes all linear-attention inputs (q, k, v, z, b, a). One computes the attention inputs (query with its gate, key, value). One computes the MLP's gate and up projections.
- **Linear attention (gated delta rule), two kernels.**
  - The causal convolution walks each sequence's tokens and carries the last three inputs across steps.
  - The recurrence keeps each head's 128×128 state in registers while it walks a sequence's tokens, then applies the gated RMSNorm.
  - Each sequence's states live in a slot: about 18.6 MB for the 2B model.
- **Full attention, two kernels.** One applies the q/k norms and rotary embedding and writes keys and values into a paged cache of 16-token blocks. The other is paged attention with the output gate fused in.
- **Norms and activation.** The residual add is folded into the next RMSNorm, and SiLU-and-multiply is one kernel.

## Numerics

Activations are bf16 with f32 accumulation, rounded where the bf16 reference rounds them. Recurrent states are f32. Logits are f32. Sampling runs on the GPU, so only one token id per sequence returns to the host each step.

## Accuracy

`tests/gpu.rs` and `scripts/accuracy.py` compare teacher-forced picks and log-probabilities with Transformers (bf16). The prompt runs as prefill and the continuation as decode steps.

| Model | Positions | Top-1 agreement | Misses within one bf16 step | Median, p95 log-prob difference |
|---|---|---|---|---|
| Qwen3.5-0.8B | 222 | 213 | 9 of 9 | 0.0032, 0.056 |
| Qwen3.5-2B | 217 | 214 | 3 of 3 | 0.0015, 0.043 |
| Qwen3.5-4B | 210 | 209 | 1 of 1 | 0.0004, 0.032 |

These ran on an RTX 5090. An H100 build of the 2B model agreed on 215 of 217.

## Throughput

`scripts/load.py` sends streamed chat completions. Prompts are about 60 to 200 tokens and outputs 16 to 64 tokens, decoded greedily. The table gives output tokens per second and median time to first token. [Receipt](receipts/load.json).

| In flight | H100: pocket-engine | H100: vLLM 0.31.0 | RTX 5090: pocket-engine |
|---:|---|---|---|
| 1 | 415 tok/s, 0.044 s | 346, 0.028 s | 306, 0.043 s |
| 8 | 2,258, 0.046 s | 1,633, 0.060 s | 1,812, 0.043 s |
| 16 | 3,674, 0.045 s | 2,676, 0.077 s | 3,065, 0.043 s |
| 32 | 5,484, 0.046 s | 4,198, 0.10 s | 4,308, 0.045 s |
| 64 | 7,502, 0.056 s | 5,467, 0.13 s | 5,554, 0.051 s |
| burst of 64 | 4,939 | 4,299 | 4,593 |

The model was Qwen3.5-2B. Both H100 runs used the same RunPod H100 SXM (driver 580.126.09), one after the other. pocket-engine was ready 5 seconds after launch and vLLM after 79. The H100 numbers come from the code before it moved to this repository (Switchyard commit b4b3d7c). Back-to-back RTX 5090 runs of that build and this one measure the same.
