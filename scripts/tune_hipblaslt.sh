#!/bin/bash
# tune_hipblaslt.sh MODEL... : tune hipBLASLt for the matrix multiplies pocket-engine runs on this AMD GPU.
#
# Serves each MODEL under scripts/load.py with hipBLASLt logging each matmul, then searches every solution for the
# shapes decoding replays (the engine's graph batch sizes from 8 to 256 rows; fewer rows run on pocket-engine's own
# kernel) and appends the fastest to tuning.txt. Copy the result to tuning/hipblaslt-<arch>-<CUs>.txt to bundle it.
# Solutions are specific to the hipBLASLt build: retune after a ROCm upgrade. Needs hipblaslt-bench
# (amdrocm-blas-test) and a release build with --features rocm.
set -euo pipefail
BENCH=${HIPBLASLT_BENCH:-$(command -v hipblaslt-bench || echo /opt/rocm/bin/hipblaslt-bench)}
BIN=${POCKET_ENGINE:-target/release/pocket-engine}
OUT=${OUT:-tuning.txt}
LOGS=$(mktemp -d)
for model in "$@"; do
  HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE="$LOGS/log-%i.txt" POCKET_HIPBLASLT_TUNING=0 \
    "$BIN" serve --model "$model" --bind 127.0.0.1:8797 --name m > "$LOGS/serve.log" 2>&1 &
  pid=$!
  until curl -s localhost:8797/v1/models > /dev/null; do sleep 0.5; done
  LOAD_CASES=1:8,16:64,64:128,128:256,256:512 LOAD_BURST=0 python3 scripts/load.py http://127.0.0.1:8797 m "$LOGS/load.json" > /dev/null
  kill -INT "$pid"; wait "$pid" || true
done
# m n k of each logged matmul (A transposed, bf16) whose n is a graph batch size (engine.rs BUCKETS).
cat "$LOGS"/log-*.txt | awk 'BEGIN { split("8 12 16 24 32 48 64 96 128 160 192 224 256", b, " "); for (i in b) want[b[i]] = 1 }
  {for (i = 1; i <= NF; i++) { if ($i == "-m") m = $(i+1); if ($i == "-n") n = $(i+1); if ($i == "-k") k = $(i+1) }
  if (n in want) print m, n, k}' | sort -u > "$LOGS/shapes.txt"
echo "$(wc -l < "$LOGS/shapes.txt") shapes"
while read -r m n k; do
  HIPBLASLT_TUNING_FILE="$OUT" "$BENCH" -m "$m" -n "$n" -k "$k" --lda "$k" --ldb "$k" --ldc "$m" --ldd "$m" \
    --transA T --transB N --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r \
    --scale_type f32_r --rotating 512 --use_gpu_timer --algo_method all --cold_iters 5 --iters 30 \
    --skip_slow_solution_ratio 0.8 > /dev/null 2>&1
done < "$LOGS/shapes.txt"
echo "wrote $OUT"
