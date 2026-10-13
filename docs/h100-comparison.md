# pocket-engine, vLLM and llama.cpp on one H100

All runs used one RunPod H100 80GB SXM (driver 580.126.09), one after another, on 2026-10-12. Each engine served Qwen3.5 0.8B, 2B, 4B and 9B and Qwen3.8-27B under `scripts/load.py`: streamed chat completions with prompts of about 60 to 200 tokens and outputs of 16 to 64 tokens, decoded greedily with thinking off. The tables give output tokens per second at 1, 8, 16, 32 and 64 requests in flight and for a burst of 64, the time from launch to serving, and the median time to first token at 64 in flight. [Receipt](receipts/h100-engines.json).

- **pocket-engine** ran with its defaults. FP8 is `--fp8` (block-scaled weights and activations, quantized at load) for 0.8B to 9B and the `Qwen3.8-27B-FP8` checkpoint for 27B. Q4_K_M ran on the build with tiled Q4_K weights, before Q6_K was tiled too.
- **vLLM 0.31.0** (the official image) ran with `--max-model-len 4096 --gpu-memory-utilization 0.9`. FP8 is `--quantization fp8` and the same 27B checkpoint. `load.py` repeats prompts across buckets, and vLLM's prefix cache skips their prefill, which pocket-engine cannot do. So the first vLLM rows have the cache off, and the "+ prefix cache" rows use vLLM's default. vLLM 0.31.0 does not load Qwen3.5 GGUF files, so it has no Q4_K_M rows. 27B needed `--max-num-seqs 256`, and its FP8 needed `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` because the image's FlashInfer FP8 kernel is missing a cubin.
- **llama.cpp** (8f574b0, `llama-server -ngl 999 -c 65536 -np 64 -fa on`) ran the GGUF files: unsloth's BF16 and Q4_K_M for 0.8B to 9B, ggml-org's for 27B. Its host prompt cache and per-slot context checkpoints snapshot the recurrent state of these hybrid models, several hundred MB per request on 27B. The main rows turn them off (`--cache-ram 0 --ctx-checkpoints 0`); the "default cache" rows keep them.

### bf16

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
| 27b | pocket-engine | 5 s | 48 | 270 | 478 | 689 | 924 | 736 | 0.165 s |
| 27b | vLLM | 101 s | 45 | 275 | 513 | 778 | 1,092 | 814 | 0.194 s |
| 27b | vLLM + prefix cache | 102 s | 45 | 268 | 492 | 771 | 1,046 | 767 | 0.228 s |
| 27b | llama.cpp | 10 s | 45 | 72 | 62 | 72 | 158 | 116 | 0.643 s |

### FP8

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
| 27b | pocket-engine | 4 s | 70 | 355 | 579 | 866 | 1,150 | 964 | 0.168 s |
| 27b | vLLM | 117 s | 64 | 352 | 611 | 925 | 1,299 | 1,093 | 0.249 s |
| 27b | vLLM + prefix cache | 170 s | 64 | 351 | 628 | 921 | 1,306 | 1,049 | 0.192 s |

### Q4_K_M

| Model | Engine | Ready | 1 | 8 | 16 | 32 | 64 | burst 64 | TTFT p50 at 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8b | pocket-engine | 2 s | 459 | 2,512 | 3,963 | 5,594 | 7,439 | 4,924 | 0.062 s |
| 0.8b | pocket-engine, before tiling | 2 s | 539 | 2,429 | 3,696 | 5,854 | 7,461 | 4,276 | 0.071 s |
| 0.8b | llama.cpp | 2 s | 373 | 591 | 482 | 459 | 713 | 554 | 0.206 s |
| 0.8b | llama.cpp, default cache | 2 s | 280 | 359 | 354 | 326 | 456 | 351 | 0.921 s |
| 2b | pocket-engine | 2 s | 400 | 1,890 | 3,078 | 4,514 | 5,902 | 4,334 | 0.059 s |
| 2b | pocket-engine, before tiling | 3 s | 379 | 1,820 | 2,970 | 4,312 | 5,559 | 4,040 | 0.051 s |
| 2b | llama.cpp | 2 s | 323 | 481 | 412 | 405 | 630 | 533 | 0.178 s |
| 2b | llama.cpp, default cache | 2 s | 252 | 361 | 337 | 327 | 443 | 378 | 0.814 s |
| 4b | pocket-engine | 3 s | 220 | 908 | 1,410 | 1,980 | 2,786 | 2,373 | 0.065 s |
| 4b | pocket-engine, before tiling | 3 s | 198 | 815 | 1,246 | 1,807 | 2,573 | 2,213 | 0.062 s |
| 4b | llama.cpp | 2 s | 198 | 296 | 260 | 265 | 426 | 357 | 0.217 s |
| 9b | pocket-engine | 3 s | 157 | 749 | 1,150 | 1,666 | 2,315 | 2,159 | 0.083 s |
| 9b | pocket-engine, before tiling | 3 s | 130 | 614 | 985 | 1,528 | 2,073 | 1,880 | 0.086 s |
| 9b | llama.cpp | 3 s | 167 | 257 | 218 | 223 | 395 | 309 | 0.311 s |
| 9b | llama.cpp, default cache | 3 s | 129 | 163 | 159 | 148 | 206 | 179 | 1.237 s |
| 27b | pocket-engine | 5 s | 64 | 241 | 358 | 523 | 675 | 644 | 0.217 s |
| 27b | pocket-engine, before tiling | 4 s | 48 | 187 | 276 | 445 | 595 | 526 | 0.226 s |
| 27b | llama.cpp | 5 s | 62 | 95 | 87 | 91 | 156 | 138 | 0.645 s |


## What it shows

- **Small and mid-sized models.** pocket-engine leads vLLM at every load from 0.8B to 9B in bf16: 1.5 times at 64 in flight on 0.8B and 2B, 1.1 times on 4B and 9B. In FP8 the lead is the same on the small models, and from 4B up the two are about even.
- **Qwen3.8-27B.** vLLM is faster from 16 in flight up: 1,092 against 924 tok/s at 64 in bf16, and 1,299 against 1,150 in FP8. Up to 8 in flight they are level.
- **Startup.** pocket-engine is ready in 2 to 5 seconds. vLLM takes 61 to 170, compiling and capturing graphs on every launch.
- **Time to first token.** pocket-engine's median at 64 in flight is 0.06 to 0.07 s up to 9B, against vLLM's 0.12 to 0.17 s.
- **llama.cpp** barely gains from concurrency on these hybrid models. Its best is 752 tok/s, on 0.8B at 64 in flight, where pocket-engine serves 7,439 from Q4_K_M. Its default cache roughly halves even that.
- **Tiled Q4 weights** raised pocket-engine's Q4_K_M throughput 10 to 29% from 8 in flight. A single request on 0.8B got slower (459 against 539 tok/s), because its matrices are too small for the tiled kernel to hide latency. On the H100 the tiled kernel also falls behind bf16 cuBLAS above 8 rows (57 against 39 µs at 64 rows on 9B `ffn_up`), because it reaches only about 1 TB/s of the H100's 3.35.
