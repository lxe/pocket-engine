# Benchmarks

pocket-engine against vLLM 0.31.0, both in bf16 on the same rented GPU. Startup is launch to serving. First tokens are for a 60-to-200-token prompt (median) and a 64k-token prompt. Throughput is output tokens per second from [`scripts/load.py`](../scripts/load.py). Receipts: [H100](receipts/h100-mtp-ttft.json), [H200](receipts/h200-gauntlet.json), [B200](receipts/b200-gauntlet.json), [RTX 5090](receipts/rtx5090-gauntlet.json).

## Across GPUs

Qwen3.5-9B:

| GPU | Engine | Startup (s) | First token, short (ms) | First token, 64k (s) | One stream (tok/s) | 64 in flight (tok/s) | One stream, 2 drafts (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | pocket-engine | **2.6–3.5** | **13.5** | 2.29 | **150** | **3,534** | **293** |
| | vLLM | 89–165 | 57.5 | **2.28** | 143 | 2,567 | 253 |
| H200 | pocket-engine | **2.4–3.5** | **13.9** | 2.24 | **197** | **4,065** | **376** |
| | vLLM | 70–108 | 41.0 | **2.15** | 189 | 3,382 | 332 |
| B200 | pocket-engine | **1.7–5.1** | **7.4** | 2.86 | **253** | **5,457** | **469** |
| | vLLM | 61–132 | 19.0 | **1.12** | 242 | 4,782 | 426 |
| RTX 5090 | pocket-engine | **1.9–2.9** | **19.1** | **6.53** | **97** | **2,030** | **185** |
| | vLLM | 62–178 | 56.7 | 6.57 | 95 | 1,872 | 153 |

Qwen3.8-27B (too large for an RTX 5090 in bf16):

| GPU | Engine | Startup (s) | First token, short (ms) | First token, 64k (s) | One stream (tok/s) | 64 in flight (tok/s) | One stream, 2 drafts (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | pocket-engine | **5.1–6.1** | **36.5** | 7.27 | **49** | **1,037** | **111** |
| | vLLM | 96–180 | 97.4 | **6.93** | 48 | 1,029 | 100 |
| H200 | pocket-engine | **3.7–6.5** | **26.5** | 6.91 | **69** | 1,200 | **146** |
| | vLLM | 93–136 | 56.2 | **6.61** | 67 | **1,254** | 136 |
| B200 | pocket-engine | **3.3–8.7** | **18.7** | 7.94 | **93** | 1,698 | **197** |
| | vLLM | 68–174 | 26.9 | **3.54** | 92 | **1,803** | 178 |

On a B200 the engine has no Blackwell-specific prompt kernels yet, so long prompts lag vLLM there.

## First token by prompt length, H100

One request at a time ([receipt](receipts/h100-mtp-ttft.json)).

| Model | Engine | Launch to serving | 60 to 200 tokens (median) | 8k tokens | 32k tokens | 64k tokens |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | pocket-engine | **2.7 s** | **13.5 ms** | **0.29 s** | **1.06 s** | 2.29 s |
| | vLLM | 89–165 s | 57.5 ms | 0.34 s | 1.10 s | **2.28 s** |
| Qwen3.8-27B | pocket-engine | **5.4 s** | **36.5 ms** | 0.81 s | 3.28 s | 7.27 s |
| | vLLM | 96–124 s | 97.4 ms | **0.79 s** | **3.22 s** | **6.93 s** |

## Throughput by model and format, H100

Output tokens per second with 64 requests in flight, prefix caches off. Drafting rows use two drafts a step in both engines. [The H100 comparison](h100-comparison.md) has every load from 1 to 64.

| Model | Format | pocket-engine | vLLM | llama.cpp | Run |
|---|---|---:|---:|---:|---|
| Qwen3.5-0.8B | bf16 | **9,114** | 6,057 | 752 | [h100-engines](receipts/h100-engines.json) |
| Qwen3.5-2B | bf16 | **7,530** | 5,172 | 656 | [h100-engines](receipts/h100-engines.json) |
| Qwen3.5-4B | bf16 | **3,736** | 3,291 | 471 | [h100-engines](receipts/h100-engines.json) |
| Qwen3.5-9B | bf16 | **3,408** | 2,938 | 390 | [h100-retest](receipts/h100-retest.json) |
| Qwen3.8-27B | bf16 | 978 | **1,088** | 158 | [h100-retest](receipts/h100-retest.json) |
| Qwen3.5-9B | FP8 | **3,523** | 3,460 | — | [h100-engines](receipts/h100-engines.json) |
| Qwen3.8-27B | FP8 | **1,314** | 1,302 | — | [h100-retest](receipts/h100-retest.json) |
| Qwen3.5-9B | bf16, 2 drafts | **4,155** | 2,882 | — | [h100-mtp-ttft](receipts/h100-mtp-ttft.json) |
| Qwen3.8-27B | bf16, 2 drafts | **1,124** | 897 | — | [h100-mtp-ttft](receipts/h100-mtp-ttft.json) |
| Qwen3.5-9B | Q4_K_M | **2,315** | does not load | 395 | [h100-engines](receipts/h100-engines.json) |
| Qwen3.8-27B | Q4_K_M | **675** | does not load | 156 | [h100-engines](receipts/h100-engines.json) |
