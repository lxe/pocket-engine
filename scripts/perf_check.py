"""Quick performance regression check: kernel micro-benchmarks and short load tests against a stored baseline.

Usage: perf_check.py [--save] [--only PREFIX] [--models DIR]
  Runs every metric (best of a few runs), compares with scripts/perf/<gpu>.json and exits 1 if any is worse than its
  tolerance. --save writes the current numbers as the new baseline. Needs a built target/release/pocket-engine and the
  model files below; metrics whose files are missing are skipped.
"""
import json, os, re, subprocess, sys, time, signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/pocket-engine"
MODELS = Path(os.environ.get("POCKET_MODELS", Path.home() / "models"))
G27 = MODELS / "pocket/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"
G9 = MODELS / "pocket/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
G08 = MODELS / "bench/Qwen3.5-0.8B-Q4_K_M.gguf"
B9 = MODELS / "pocket/Qwen3.5-9B"
KERNEL_TOL, E2E_TOL = 0.08, 0.06   # allowed slowdown
PORT = 8799

def run(cmd, timeout=180):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT).stdout

def linear(tensor, path, rows):
    """bench-linear: the path the engine takes at each row count (int8 from 16 rows when the format has it)."""
    out = run([BIN, "bench-linear", "--model", path, "--tensor", tensor, "--rows", ",".join(map(str, rows))])
    res = {}
    for line in out.splitlines():
        m = re.match(r"rows\s+(\d+):", line)
        if not m:
            continue
        r = int(m.group(1))
        i8 = re.search(r"int8\s+([\d.]+) us", line)
        ti = re.search(r"tiled\s+([\d.]+) us", line)
        v = i8 if (i8 and r >= 16) else ti
        if v:
            res[f"r{r}"] = float(v.group(1))
    return res

def us(cmd, pattern):
    m = re.search(pattern, run([BIN] + cmd))
    return float(m.group(1)) * (1000 if "ms" in pattern else 1) if m else None

def serve(model, extra=()):
    log = open("/tmp/perf_check_serve.log", "w")
    p = subprocess.Popen([BIN, "serve", "--model", model, "--bind", f"127.0.0.1:{PORT}", "--max-seqs", "64", *extra],
                         stdout=log, stderr=log, cwd=ROOT)
    import urllib.request
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server exited; see /tmp/perf_check_serve.log")
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("server did not start")

def load(model, cases="1:16,16:96,64:192"):
    p = serve(model)
    try:
        out = "/tmp/perf_check_load.json"
        env = dict(os.environ, LOAD_CASES=cases, LOAD_BURST="0")
        subprocess.run([sys.executable, ROOT / "scripts/load.py", f"http://127.0.0.1:{PORT}", "m", out], env=env,
                       capture_output=True, timeout=600, check=True)
        return {r["case"].replace("closed c=", "c"): r["out_tok_s"] for r in json.load(open(out))}
    finally:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(20)
        except subprocess.TimeoutExpired:
            p.kill()

# name -> (lower is better?, tolerance, needed file, function returning {suffix: value})
ROWS = [1, 16, 64, 512, 2048]
METRICS = {
    "linear.q4_k": (True, KERNEL_TOL, G27, lambda: linear("blk.3.ffn_gate.weight", G27, ROWS)),
    "linear.q5_k": (True, KERNEL_TOL, G27, lambda: linear("blk.3.ffn_up.weight", G27, ROWS)),
    "linear.iq4_xs": (True, KERNEL_TOL, G27, lambda: linear("blk.0.ffn_gate.weight", G27, ROWS)),
    "linear.q6_k": (True, KERNEL_TOL, G27, lambda: linear("blk.3.attn_output.weight", G27, ROWS)),
    "attn": (True, KERNEL_TOL, None, lambda: {
        "prefill.g4": us(["bench-attn", "--group", "4", "--tokens", "4096"], r": ([\d.]+) ms"),
        "prefill.g6": us(["bench-attn", "--group", "6", "--tokens", "4096"], r": ([\d.]+) ms"),
    }),
    "gdn": (True, KERNEL_TOL, None, lambda: {
        "chunk4096": us(["bench-gdn", "--chunk", "--tokens", "4096"], r"heads: ([\d.]+) us"),
        "decode64": us(["bench-gdn", "--seqs", "64", "--tokens", "1"], r"heads: ([\d.]+) us"),
    }),
    "load.9b_q4km": (False, E2E_TOL, G9, lambda: load(G9)),
    "load.0.8b_q4km": (False, E2E_TOL, G08, lambda: load(G08)),
    "load.9b_bf16": (False, E2E_TOL, B9, lambda: load(B9, "16:96,64:192")),
}

def gpu_name():
    return run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()[0].strip()

def measure(names, reps):
    got = {}
    for name in names:
        lower, tol, need, fn = METRICS[name]
        if need is not None and not Path(need).exists():
            print(f"skip {name}: {need} missing")
            continue
        best = {}
        for _ in range(reps):
            for k, v in fn().items():
                if v is None:
                    continue
                key = f"{name}.{k}"
                best[key] = v if key not in best else (min if lower else max)(best[key], v)
        got.update(best)
        print(f"{name}: " + ", ".join(f"{k[len(name) + 1:]}={v:g}" for k, v in best.items()), flush=True)
    return got

def main():
    args = sys.argv[1:]
    only = args[args.index("--only") + 1] if "--only" in args else ""
    names = [n for n in METRICS if n.startswith(only)]
    busy = run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"]).split()
    if any(int(x) > 1000 for x in busy if x.isdigit()):
        sys.exit("another process holds more than 1 GB of GPU memory; timings would be off")
    base_file = ROOT / "scripts/perf" / (re.sub(r"[^a-z0-9]+", "-", gpu_name().lower()).strip("-") + ".json")
    got = measure(names, reps=2)
    if "--save" in args:
        base = json.load(open(base_file)) if base_file.exists() else {}
        base.update(got)
        base_file.parent.mkdir(exist_ok=True)
        json.dump(dict(sorted(base.items())), open(base_file, "w"), indent=1)
        print(f"saved {len(got)} metrics to {base_file.relative_to(ROOT)}")
        return
    base = json.load(open(base_file))
    def group(k):
        return next(n for n in METRICS if k.startswith(n + "."))
    def worse(k, v):
        lower, tol = METRICS[group(k)][:2]
        b = base.get(k)
        if b is None:
            return None
        change = (v - b) / b if lower else (b - v) / b
        return change if change > tol else None
    bad = {k: v for k, v in got.items() if worse(k, v) is not None}
    if bad:
        # Confirm on a second measurement before failing.
        again = measure(sorted({group(k) for k in bad}), reps=2)
        bad = {k: again[k] for k in bad if k in again and worse(k, again[k]) is not None}
    print(f"\n{'metric':40} {'baseline':>10} {'now':>10} {'change':>8}")
    for k, v in got.items():
        b = base.get(k)
        if b is None:
            continue
        ch = (v - b) / b * 100
        flag = "  REGRESSION" if k in bad else ""
        print(f"{k:40} {b:10.1f} {v:10.1f} {ch:+7.1f}%{flag}")
    if bad:
        sys.exit(f"\n{len(bad)} regression(s)")
    print("\nno regressions")

if __name__ == "__main__":
    main()
