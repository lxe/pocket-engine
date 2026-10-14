"""Accuracy check against Transformers.

  python scripts/accuracy.py reference MODEL_DIR OUT [N] [--adapter LORA_DIR]   # greedy + teacher-forced top-5
  python scripts/accuracy.py long MODEL_DIR OUT            # the same over ~3,300-token prompts (docs/)
  pocket-engine score --model MODEL_DIR OUT.pairs.json OUT.ours.json
  python scripts/accuracy.py compare OUT.ref.json OUT.ours.json

`reference` needs torch and transformers (see docs/testing.md); `compare` needs only Python. With --adapter,
the PEFT adapter runs unmerged, as PEFT and vLLM run it: y += scale * (x A^T) B^T in f32. ADAPTER_MERGED=1
merges it into the bf16 weights instead, which rounds every weight it touches again. REF_PAIRS=FILE scores the
continuations of an earlier pairs file instead of generating new ones."""
import json, os, sys


def hook_adapter(model, path):
    """Applies a PEFT adapter unmerged, as PEFT runs it: y += scale * (x A^T) B^T, in f32."""
    import torch, re
    from safetensors.torch import load_file
    cfg = json.load(open(path + "/adapter_config.json"))
    sd = load_file(path + "/adapter_model.safetensors")
    mods = dict(model.named_modules())
    n = 0
    for k, a in sd.items():
        if not k.endswith(".lora_A.weight"):
            continue
        stem = k[: -len(".lora_A.weight")]
        layer, module = re.search(r"layers\.(\d+)\.(.+)$", stem).groups()
        b = sd[stem + ".lora_B.weight"]
        r = a.shape[0]
        scale = cfg["lora_alpha"] / (r ** 0.5 if cfg.get("use_rslora") else r)
        m = mods[f"model.layers.{layer}.{module}"]
        A, B = a.float().to(m.weight.device), (b.float() * scale).to(m.weight.device)
        m.register_forward_hook(lambda mod, inp, out, A=A, B=B: (out.float() + (inp[0].float() @ A.T) @ B.T).to(out.dtype))
        n += 1
    print(f"hooked {n} LoRA modules from {path}")


def merge_adapter(model, path):
    import torch, re
    from safetensors.torch import load_file
    cfg = json.load(open(path + "/adapter_config.json"))
    sd = load_file(path + "/adapter_model.safetensors")
    mods = dict(model.named_modules())
    n = 0
    for k, a in sd.items():
        if not k.endswith(".lora_A.weight"):
            continue
        stem = k[: -len(".lora_A.weight")]
        layer, module = re.search(r"layers\.(\d+)\.(.+)$", stem).groups()
        b = sd[stem + ".lora_B.weight"]
        r = a.shape[0]
        scale = cfg["lora_alpha"] / (r ** 0.5 if cfg.get("use_rslora") else r)
        w = mods[f"model.layers.{layer}.{module}"].weight
        with torch.no_grad():
            w.copy_((w.float() + scale * (b.float().to(w.device) @ a.float().to(w.device))).to(w.dtype))
        n += 1
    print(f"merged {n} LoRA modules from {path}")


def reference(M, OUT, N=48, prompts=None, adapter=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
    if adapter and os.environ.get("ADAPTER_MERGED"):
        merge_adapter(model, adapter)
    elif adapter:
        hook_adapter(model, adapter)
    if callable(prompts):
        prompts = prompts(tok)
    prompts = prompts or [
        "Hello! Who are you?",
        "Summarize briefly: the receipt shows a total amount paid by card at the store on the date with tax and tip included for the meal",
        "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
        "List three facts about the Moon, one sentence each. " * 6,
        "What is 17 * 23? Show your work step by step.",
    ]
    pairs, ref = [], []
    # REF_PAIRS scores the continuations of an earlier pairs file instead of generating.
    given = json.load(open(os.environ["REF_PAIRS"])) if os.environ.get("REF_PAIRS") else None
    for i, p in enumerate(given or prompts):
        if given:
            ids, g = p["prompt"], p["forced"]
        else:
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            if not given:
                g = model.generate(x, max_new_tokens=N, do_sample=False, temperature=None, top_p=None, top_k=None)[0, len(ids):].tolist()
            full = torch.tensor([ids + g], device="cuda")
            lp = torch.log_softmax(model(full).logits[0, len(ids) - 1:-1].float(), -1)
        top = lp.topk(5, -1)
        ref.append([{"argmax": int(top.indices[i, 0]), "top": [[int(a), float(b)] for a, b in zip(top.indices[i], top.values[i])]} for i in range(len(g))])
        pairs.append({"prompt": ids, "forced": g})
        print(len(ids), len(g), repr(tok.decode(g))[:120])
    json.dump(pairs, open(OUT + ".pairs.json", "w"))
    json.dump(ref, open(OUT + ".ref.json", "w"))


def long_prompts(tok):
    """Three questions about this repository's documentation, each after a different 3,300-token
    slice of it: long enough that most of the context is in the KV cache."""
    import glob, os
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    files = ["README.md"] + sorted(os.path.relpath(f, root) for f in glob.glob(os.path.join(root, "docs", "*.md")))
    text = "".join(open(os.path.join(root, f)).read() + "\n\n" for f in files)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    qs = ["Summarize the document above in detail, section by section.",
          "Using the document above, explain how a quantized matrix multiply picks its path, and why.",
          "List every number the document above reports for the RTX 5090, with what each measures."]
    assert len(ids) >= 2 * 200 + 3300, "the documentation is too short for three 3,300-token slices"
    return [tok.decode(ids[k * 200: k * 200 + 3300]) + "\n\n" + q for k, q in enumerate(qs)]


def compare(ref_path, ours_path):
    ref = json.load(open(ref_path)); got = json.load(open(ours_path))
    agree = tot = ties = 0; diffs = []
    for r_seq, g_seq in zip(ref, got):
        for i, (r, g) in enumerate(zip(r_seq, g_seq)):
            tot += 1; agree += r["argmax"] == g["argmax"]
            rt = dict((a, b) for a, b in r["top"]); gt = dict((a, b) for a, b in g["top"])
            diffs.append(abs(rt[r["argmax"]] - gt.get(r["argmax"], -99)))
            if r["argmax"] != g["argmax"]:
                # One bf16 rounding step of the reference's logits at these sizes.
                tie = r["top"][0][1] - r["top"][1][1] <= 0.126; ties += tie
                print("mismatch at", i, "near tie in the reference" if tie else "", r["top"][:2], g["top"][:2])
    diffs.sort()
    print(f"top-1 agreement {agree}/{tot} ({ties} of the misses are within one bf16 step in the reference); "
          f"|dlogprob| of the reference pick: median {diffs[len(diffs)//2]:.4f} p95 {diffs[int(.95*len(diffs))]:.4f} max {diffs[-1]:.4f}")


if __name__ == "__main__":
    adapter = None
    if "--adapter" in sys.argv:
        i = sys.argv.index("--adapter")
        adapter = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if sys.argv[1] == "long":
        reference(sys.argv[2], sys.argv[3], 96, long_prompts, adapter)
    elif sys.argv[1] == "reference":
        reference(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 48, adapter=adapter)
    else:
        compare(sys.argv[2], sys.argv[3])
