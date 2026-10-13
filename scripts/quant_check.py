"""Checks pocket-engine's GPU dequantization against gguf-py (llama.cpp's Python reference) for every quantized
tensor type in the given GGUF files: the first rows of one tensor per type, compared after rounding the reference to
bf16 (the engine's weight precision).

  PYTHONPATH=path/to/llama.cpp/gguf-py python scripts/quant_check.py target/release/pocket-engine FILE.gguf ..."""
import subprocess, sys, tempfile, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

binary, files = sys.argv[1], sys.argv[2:]
seen, bad = set(), 0
for f in files:
    r = GGUFReader(f)
    for t in r.tensors:
        ty = t.tensor_type
        if ty.name in ("F32", "F16", "BF16") or ty.name in seen or len(t.shape) != 2:
            continue
        seen.add(ty.name)
        cols = int(t.shape[0]); rows = 64
        ref = dequantize(t.data, ty).reshape(-1, cols)[:rows].astype(np.float32)
        ref_bf16 = (ref.view(np.uint32) + 0x7FFF + ((ref.view(np.uint32) >> 16) & 1)) & 0xFFFF0000  # round to nearest even
        ref_bf16 = ref_bf16.view(np.float32)
        with tempfile.NamedTemporaryFile() as out:
            subprocess.run([binary, "dequant", "--model", f, "--tensor", t.name, "--rows", str(rows), out.name], check=True, capture_output=True)
            got = np.fromfile(out.name, dtype=np.float32).reshape(ref.shape)
        exact = np.mean(got == ref_bf16)
        err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
        ok = exact > 0.999 and err < 1e-2
        bad += not ok
        print(f"{ty.name:8s} {t.name:28s} exact {exact*100:7.3f}%  max rel err {err:.2e}  {'ok' if ok else 'MISMATCH'}")
print(f"{len(seen)} types, {bad} mismatches")
sys.exit(1 if bad else 0)
