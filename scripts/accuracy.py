"""Accuracy check against Transformers.

  python scripts/accuracy.py reference MODEL_DIR OUT [N]   # greedy continuations + teacher-forced top-5
  pocket-engine score --model MODEL_DIR OUT.pairs.json OUT.ours.json
  python scripts/accuracy.py compare OUT.ref.json OUT.ours.json

`reference` needs torch and transformers (see README); `compare` needs only Python."""
import json, sys


def reference(M, OUT, N=48):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = [
        "Hello! Who are you?",
        "Summarize briefly: the receipt shows a total amount paid by card at the store on the date with tax and tip included for the meal",
        "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
        "List three facts about the Moon, one sentence each. " * 6,
        "What is 17 * 23? Show your work step by step.",
    ]
    pairs, ref = [], []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            g = model.generate(x, max_new_tokens=N, do_sample=False, temperature=None, top_p=None, top_k=None)[0, len(ids):].tolist()
            full = torch.tensor([ids + g], device="cuda")
            lp = torch.log_softmax(model(full).logits[0, len(ids) - 1:-1].float(), -1)
        top = lp.topk(5, -1)
        ref.append([{"argmax": int(top.indices[i, 0]), "top": [[int(a), float(b)] for a, b in zip(top.indices[i], top.values[i])]} for i in range(len(g))])
        pairs.append({"prompt": ids, "forced": g})
        print(len(ids), len(g), repr(tok.decode(g))[:120])
    json.dump(pairs, open(OUT + ".pairs.json", "w"))
    json.dump(ref, open(OUT + ".ref.json", "w"))


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
    if sys.argv[1] == "reference":
        reference(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 48)
    else:
        compare(sys.argv[2], sys.argv[3])
