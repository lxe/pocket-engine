# Benchmarks

pocket-engine against vLLM 0.31.0 on the same rented GPU, and llama.cpp on the H100. Startup is launch to serving. First tokens are for a 60-to-200-token prompt (median) and a 64k-token prompt. Throughput is output tokens per second from [`scripts/load.py`](../scripts/load.py). Receipts: [H100](receipts/h100-mtp-ttft.json), [H200](receipts/h200-gauntlet.json), [B200](receipts/b200-gauntlet.json), [RTX 5090](receipts/rtx5090-gauntlet.json), [MI325X](receipts/mi325x.json).

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

The B200 rows predate its own prompt attention and decode kernels: see [B200 kernels](#b200-kernels).

## B200 kernels

On a B200, prompt attention runs on tcgen05 tensor cores with tensor memory, as FlashAttention 4 does. The two heads of a pair run as one 256-row matrix multiply across two SMs, each staging half of every key and value tile; the heads that share keys and values form one cluster that loads each tile once. The linear-attention decode keeps the next heads' states and inputs loading while one computes. Measured on pods of the same kind ([receipt](receipts/b200-kernels.json)):

| | Before | Now | vLLM, same pod |
|---|---:|---:|---:|
| Qwen3.5-9B, first token for a 64k-token prompt | 2.50 s | 1.44 s | 1.32–1.38 s |
| Prompt attention at 64k tokens (`bench-attn`) | 251 TFLOPS | 1,050 TFLOPS | |
| Linear-attention decode, 64 sequences of 27B (`bench-gdn`) | 115 µs | 69 µs | |
| Qwen3.8-27B, 64 requests in flight | 1,583 tok/s | 1,655 tok/s | 1,236 tok/s |

These pods capped each container at 20 CPUs on a busy host. That slows vLLM's scheduler: its one-stream rates there were 20–30% under its own in the table above, so its last number here understates it. Against that table's 1,803 tok/s, 27B at 64 in flight is close but not yet matched.

## AMD MI325X

One MI325X (ROCm 7.14) against vLLM 0.31.0's ROCm image with AITER attention. Output tokens/s from `scripts/load.py`, plus 128 and 256 in flight with the same prompts. vLLM renders Qwen's template with thinking on, so its answers run to `max_tokens`; pocket-engine ran with `"chat_template_kwargs": {"enable_thinking": true}` to match (about 40 tokens per request on both). pocket-engine measured 2026-10-14, vLLM 2026-10-13. Receipts: [mi325x.json](receipts/mi325x.json).

| In flight | 9B pocket-engine | 9B vLLM | 27B pocket-engine | 27B vLLM |
|---:|---:|---:|---:|---:|
| 1 | **215** | 207 | 72 | **76** |
| 8 | **1,067** | 772 | **364** | 349 |
| 16 | **1,919** | 1,263 | **650** | 587 |
| 32 | **2,682** | 2,046 | **887** | 873 |
| 64 | **3,660** | 3,273 | 1,152 | **1,166** |
| 128 | **4,808** | 4,580 | 1,402 | **1,485** |
| 256 | 5,323 | **5,445** | 1,492 | **1,669** |

Startup takes 2 to 3 s against vLLM's 260 to 316 s (plus up to 5 s while the driver frees a previous process's memory), and the median first token at 64 in flight is 79 ms against 135 (9B) and 275 against 353 (27B). Qwen3.5-9B Q4_K_M serves 81, 879 and 2,146 tokens/s at 1, 16 and 64 in flight (thinking off).

## First token by prompt length, H100

One request at a time ([receipt](receipts/h100-mtp-ttft.json)).

| Model | Engine | Launch to serving | 60 to 200 tokens (median) | 8k tokens | 32k tokens | 64k tokens |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | pocket-engine | **2.7 s** | **13.5 ms** | **0.29 s** | **1.06 s** | 2.29 s |
| | vLLM | 89–165 s | 57.5 ms | 0.34 s | 1.10 s | **2.28 s** |
| Qwen3.8-27B | pocket-engine | **5.4 s** | **36.5 ms** | 0.81 s | 3.28 s | 7.27 s |
| | vLLM | 96–124 s | 97.4 ms | **0.79 s** | **3.22 s** | **6.93 s** |

## Every model and format, H100

The three engines ran one after another on one H100. Columns: output tokens per second by requests in flight, a burst of 64, launch-to-serving time, and median time to first token at 64 in flight. Builds and flags are in the [receipt](receipts/h100-engines.json).

- **vLLM + prefix cache:** vLLM's default cache, which skips this test's repeated prompts.
- **llama.cpp, default cache:** its prompt cache and checkpoints left on.
- **pocket-engine, plain blocks:** without the tiled Q4 layout.

bf16:

| Model | Engine | Ready | 1 | 8 | 16 | 32 | 64 | burst 64 | TTFT p50 at 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8b | pocket-engine | 2 s | 590 | 3,142 | 4,586 | 7,087 | 9,114 | 4,764 | 0.065 s |
| 0.8b | vLLM | 61 s | 440 | 1,911 | 3,124 | 4,855 | 6,057 | 4,478 | 0.166 s |
| 0.8b | vLLM + prefix cache | 62 s | 426 | 1,799 | 2,985 | 4,972 | 5,619 | 4,897 | 0.143 s |
| 0.8b | llama.cpp | 2 s | 334 | 580 | 490 | 471 | 752 | 563 | 0.173 s |
| 0.8b | llama.cpp, default cache | 3 s | 256 | 359 | 360 | 344 | 482 | 391 | 0.82 s |
| 2b | pocket-engine | 2 s | 428 | 2,332 | 3,760 | 5,646 | 7,530 | 4,581 | 0.066 s |
| 2b | vLLM | 61 s | 346 | 1,592 | 2,753 | 4,137 | 5,172 | 4,276 | 0.135 s |
| 2b | vLLM + prefix cache | 90 s | 334 | 1,542 | 2,354 | 3,571 | 5,144 | 3,694 | 0.144 s |
| 2b | llama.cpp | 3 s | 296 | 489 | 398 | 390 | 656 | 540 | 0.157 s |
| 2b | llama.cpp, default cache | 3 s | 234 | 362 | 340 | 328 | 481 | 376 | 0.673 s |
| 4b | pocket-engine | 3 s | 228 | 1,105 | 1,780 | 2,643 | 3,736 | 2,809 | 0.071 s |
| 4b | vLLM | 66 s | 193 | 868 | 1,518 | 2,443 | 3,291 | 2,644 | 0.131 s |
| 4b | vLLM + prefix cache | 71 s | 200 | 902 | 1,557 | 2,251 | 3,133 | 2,730 | 0.132 s |
| 4b | llama.cpp | 3 s | 166 | 277 | 232 | 224 | 471 | 339 | 0.19 s |
| 4b | llama.cpp, default cache | 5 s | 114 | 158 | 153 | 122 | 190 | 169 | 1.079 s |
| 9b | pocket-engine | 3 s | 148 | 884 | 1,436 | 2,343 | 3,133 | 2,545 | 0.066 s |
| 9b | vLLM | 69 s | 126 | 712 | 1,187 | 2,022 | 2,921 | 2,474 | 0.153 s |
| 9b | vLLM + prefix cache | 72 s | 134 | 750 | 1,252 | 2,155 | 3,238 | 2,455 | 0.126 s |
| 9b | llama.cpp | 4 s | 124 | 217 | 172 | 174 | 390 | 274 | 0.232 s |
| 9b | llama.cpp, default cache | 4 s | 94 | 149 | 135 | 120 | 197 | 158 | 1.358 s |
| 27b | pocket-engine | 5 s | 50 | 301 | 552 | 824 | 1,069 | 797 | 0.117 s |
| 27b | vLLM | 101 s | 45 | 275 | 513 | 778 | 1,092 | 814 | 0.194 s |
| 27b | vLLM + prefix cache | 102 s | 45 | 268 | 492 | 771 | 1,046 | 767 | 0.228 s |
| 27b | llama.cpp | 10 s | 45 | 72 | 62 | 72 | 158 | 116 | 0.643 s |

pocket-engine's 27B rows in this table and the next were measured again after the linear-attention changes of October 13 ([receipt](receipts/h100-27b.json)).

FP8 (pocket-engine `--fp8`, vLLM `--quantization fp8`; the Qwen3.8-27B-FP8 checkpoint for 27B):

| Model | Engine | Ready | 1 | 8 | 16 | 32 | 64 | burst 64 | TTFT p50 at 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8b | pocket-engine | 2 s | 553 | 3,124 | 4,476 | 6,618 | 8,335 | 5,180 | 0.066 s |
| 0.8b | vLLM | 78 s | 433 | 1,826 | 3,178 | 4,797 | 6,335 | 4,984 | 0.166 s |
| 0.8b | vLLM + prefix cache | 84 s | 398 | 1,771 | 3,086 | 4,597 | 5,662 | 4,748 | 0.175 s |
| 2b | pocket-engine | 2 s | 451 | 2,509 | 3,994 | 5,870 | 7,395 | 4,984 | 0.063 s |
| 2b | vLLM | 83 s | 381 | 1,697 | 2,776 | 4,406 | 5,990 | 4,653 | 0.149 s |
| 2b | vLLM + prefix cache | 96 s | 371 | 1,590 | 2,818 | 4,356 | 5,800 | 4,423 | 0.159 s |
| 4b | pocket-engine | 3 s | 265 | 1,216 | 1,931 | 2,714 | 3,860 | 2,965 | 0.066 s |
| 4b | vLLM | 92 s | 229 | 1,058 | 1,874 | 2,767 | 3,646 | 3,099 | 0.16 s |
| 4b | vLLM + prefix cache | 94 s | 232 | 1,112 | 1,612 | 2,676 | 3,983 | 3,125 | 0.13 s |
| 9b | pocket-engine | 3 s | 192 | 1,072 | 1,738 | 2,549 | 3,523 | 2,994 | 0.069 s |
| 9b | vLLM | 93 s | 178 | 943 | 1,509 | 2,642 | 3,460 | 2,921 | 0.128 s |
| 9b | vLLM + prefix cache | 96 s | 185 | 972 | 1,548 | 2,460 | 3,499 | 2,997 | 0.124 s |
| 27b | pocket-engine | 4 s | 71 | 425 | 736 | 1,082 | 1,380 | 1,096 | 0.096 s |
| 27b | vLLM | 117 s | 64 | 352 | 611 | 925 | 1,299 | 1,093 | 0.249 s |
| 27b | vLLM + prefix cache | 170 s | 64 | 351 | 628 | 921 | 1,306 | 1,049 | 0.192 s |

Q4_K_M (GGUF; vLLM does not load these files):

| Model | Engine | Ready | 1 | 8 | 16 | 32 | 64 | burst 64 | TTFT p50 at 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8b | pocket-engine | 2 s | 459 | 2,512 | 3,963 | 5,594 | 7,439 | 4,924 | 0.062 s |
| 0.8b | pocket-engine, plain blocks | 2 s | 539 | 2,429 | 3,696 | 5,854 | 7,461 | 4,276 | 0.071 s |
| 0.8b | llama.cpp | 2 s | 373 | 591 | 482 | 459 | 713 | 554 | 0.206 s |
| 0.8b | llama.cpp, default cache | 2 s | 280 | 359 | 354 | 326 | 456 | 351 | 0.921 s |
| 2b | pocket-engine | 2 s | 400 | 1,890 | 3,078 | 4,514 | 5,902 | 4,334 | 0.059 s |
| 2b | pocket-engine, plain blocks | 3 s | 379 | 1,820 | 2,970 | 4,312 | 5,559 | 4,040 | 0.051 s |
| 2b | llama.cpp | 2 s | 323 | 481 | 412 | 405 | 630 | 533 | 0.178 s |
| 2b | llama.cpp, default cache | 2 s | 252 | 361 | 337 | 327 | 443 | 378 | 0.814 s |
| 4b | pocket-engine | 3 s | 220 | 908 | 1,410 | 1,980 | 2,786 | 2,373 | 0.065 s |
| 4b | pocket-engine, plain blocks | 3 s | 198 | 815 | 1,246 | 1,807 | 2,573 | 2,213 | 0.062 s |
| 4b | llama.cpp | 2 s | 198 | 296 | 260 | 265 | 426 | 357 | 0.217 s |
| 9b | pocket-engine | 3 s | 157 | 749 | 1,150 | 1,666 | 2,315 | 2,159 | 0.083 s |
| 9b | pocket-engine, plain blocks | 3 s | 130 | 614 | 985 | 1,528 | 2,073 | 1,880 | 0.086 s |
| 9b | llama.cpp | 3 s | 167 | 257 | 218 | 223 | 395 | 309 | 0.311 s |
| 9b | llama.cpp, default cache | 3 s | 129 | 163 | 159 | 148 | 206 | 179 | 1.237 s |
| 27b | pocket-engine | 5 s | 64 | 241 | 358 | 523 | 675 | 644 | 0.217 s |
| 27b | pocket-engine, plain blocks | 4 s | 48 | 187 | 276 | 445 | 595 | 526 | 0.226 s |
| 27b | llama.cpp | 5 s | 62 | 95 | 87 | 91 | 156 | 138 | 0.645 s |

## Shared prefixes, H100

Both engines with their prefix caches on (`scripts/workloads.py`). Output tokens per second and median time to first token. [Receipt](receipts/h100-retest.json).

| Workload | Model | pocket-engine | vLLM |
|---|---|---|---|
| 2,000-token system prompt, 128 short questions, 16 at a time | 9b | 1,338 tok/s, 0.059 s | 1,040 tok/s, 0.206 s |
| | 27b | 448 tok/s, 0.079 s | 359 tok/s, 0.240 s |
| 16 conversations of 8 turns, each resending its history | 9b | 1,613 tok/s, 0.118 s | 1,419 tok/s, 0.220 s |
| | 27b | 420 tok/s, 0.202 s | 423 tok/s, 0.451 s |

## Speculative decoding, H100

Output tokens per second: one stream of four 512-token answers, then `scripts/load.py` by requests in flight. vLLM uses its own MTP drafting (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`). [Receipt](receipts/h100-mtp-ttft.json).

| Qwen3.5-9B | One stream | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| pocket-engine, plain | 150 | 145 | 952 | 1,673 | 2,606 | 3,534 |
| vLLM, plain | 143 | 124 | 636 | 1,052 | 1,821 | 2,567 |
| pocket-engine, 2 drafts | 293 | 258 | 1,488 | 2,444 | 3,470 | 4,155 |
| vLLM, 2 drafts | 253 | 192 | 894 | 1,509 | 2,281 | 2,882 |
| pocket-engine, 3 drafts | 296 | 256 | 1,461 | 2,445 | 3,420 | 3,987 |
| vLLM, 3 drafts | 248 | 192 | 985 | 1,522 | 2,479 | 3,165 |

| Qwen3.8-27B | One stream | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| pocket-engine, plain | 49 | 47 | 292 | 531 | 812 | 1,037 |
| vLLM, plain | 48 | 43 | 246 | 446 | 712 | 1,029 |
| pocket-engine, 2 drafts | 111 | 89 | 510 | 854 | 1,112 | 1,124 |
| vLLM, 2 drafts | 100 | 72 | 380 | 616 | 830 | 897 |

Both engines accept about the same share of drafts: 62% with two drafts on 9B, 63% on 27B.

## Prefix cache, RTX 5090

Qwen3.5-9B, bf16, `scripts/workloads.py`. Output tokens per second and median time to first token:

| Workload | Cache off | Cache on |
|---|---|---|
| 2,000-token system prompt, 128 short questions, 16 at a time | 174 tok/s, 0.49 s | 992 tok/s, 0.059 s |
| 16 conversations of 8 turns, each resending its history | 606 tok/s, 0.49 s | 955 tok/s, 0.28 s |

## Long prompts, RTX 5090

Qwen3.5-9B, bf16, one request at a time (`scripts/workloads.py long`):

| Prompt | First token | Decode afterwards |
|---:|---:|---:|
| 8k | 0.66 s | 85 tok/s |
| 32k | 2.97 s | 81 tok/s |
| 64k | 7.04 s | 77 tok/s |

## Quantized weights, RTX 5090

GGUF batches multiply on int8 tensor cores, with activations quantized to int8 per 32 values as llama.cpp does them; `--no-int8` dequantizes the weights to bf16 instead. Output tokens per second by requests in flight ([receipt](receipts/rtx5090-features.json)):

| | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Q4_K_M | 177 | 1,022 | 1,807 | 2,074 | 2,372 |
| Qwen3.5-9B Q4_K_M, `--no-int8` | 176 | 933 | 1,511 | 1,599 | 1,780 |
| Qwen3.8-27B UD-Q4_K_XL | 56 | 266 | 431 | 484 | 542 |
| Qwen3.8-27B UD-Q4_K_XL, `--no-int8` | 54 | 234 | 333 | 390 | 378 |
| Qwen3.8-27B NVFP4 checkpoint (as MXFP4) | 56 | 304 | 455 | 571 | 551 |

## Adapters, RTX 5090

Qwen3.5-9B Q4_K_M with four copies of a rank-16 adapter on every linear layer ([Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)), by requests in flight ([receipt](receipts/rtx5090-features.json)):

| Requests name | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| the base model | 181 | 1,054 | 1,847 | 2,113 | 2,391 |
| one adapter | 161 | 872 | 1,471 | 1,550 | 1,779 |
| four adapters, in turn | 163 | 859 | 1,498 | 1,563 | 1,761 |

Requests for the base model cost nothing extra with adapters loaded. Adapters cost 11% at one request and 26% at 64 (issue [#11](https://github.com/lxe/pocket-engine/issues/11)).

## Structured output and log-probabilities, RTX 5090

Qwen3.5-9B Q4_K_M, by requests in flight ([receipt](receipts/rtx5090-features.json)):

| | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| plain | 177 | 1,022 | 1,807 | 2,074 | 2,372 |
| `"response_format": {"type": "json_object"}` | 173 | 924 | 1,670 | 1,863 | 2,196 |
| `"logprobs": true, "top_logprobs": 5` | 176 | 1,014 | 1,804 | 2,033 | 2,347 |

## Prompt lookup, RTX 5090

`--ngram` drafts tokens by finding the last few generated tokens earlier in the prompt. It pays when answers copy from their prompts (editing, extraction, quoting) and stays out of the way otherwise. Output tokens per second, greedy ([receipt](receipts/rtx5090-features.json)):

| | Plain | `--ngram` |
|---|---:|---:|
| Qwen3.5-9B Q4_K_M, one copy-heavy request | 182–191 | 662–719 |
| Qwen3.5-9B Q4_K_M, 16 copy-heavy requests at once | 2,072–2,410 | 3,557–4,602 |
| Qwen3.5-9B Q4_K_M, standard load, 64 in flight | 2,372 | 2,394 |
| Qwen3.8-27B UD-Q4_K_XL, one copy-heavy request | 61 (`--mtp 2`: 143–148) | 233–236 |
