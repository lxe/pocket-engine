"""Transformers reference for image prompts: greedy continuations, teacher-forced top-5 and the processor's
prompt ids, for tests/fixtures/*.images.*.json. Needs torch, transformers, torchvision and pillow, and a model
directory with preprocessor_config.json.

  python scripts/image_reference.py MODEL_DIR OUT
  pocket-engine score --model MODEL_DIR OUT.pairs.json OUT.ours.json   # also writes OUT.ours.prompts.json
  python scripts/accuracy.py compare OUT.ref.json OUT.ours.json"""
import base64, io, json, sys, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
M, OUT = sys.argv[1], sys.argv[2]
N = 40
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()

def receipt(w, h):
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    lines = ["CORNER CAFE", "2026-10-12 14:05", "Latte        4.50", "Croissant    3.25", "Tax          0.70", "TOTAL        8.45", "VISA ****1234"]
    for i, t in enumerate(lines): d.text((20, 20 + i * 28), t, fill="black", font_size=22)
    return im

def shapes(w, h):
    im = Image.new("RGB", (w, h), (235, 235, 245)); d = ImageDraw.Draw(im)
    d.ellipse((30, 30, 150, 150), fill="red"); d.rectangle((200, 60, 330, 190), fill="blue"); d.polygon([(400, 200), (470, 60), (540, 200)], fill="green")
    return im

cases = [  # sides multiple of 32: no resize, so preprocessing matches exactly
    (receipt(384, 256), "What is the total on this receipt?"),
    (shapes(576, 256), "Describe the shapes and their colors."),
    (receipt(300, 211), "Read the store name."),  # resized: approximate
]
pairs, ref, prompts = [], [], []
for img, q in cases:
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
    ids = inp["input_ids"][0].tolist()
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=N, do_sample=False, temperature=None, top_p=None, top_k=None)[0, len(ids):].tolist()
        full = dict(inp); full["input_ids"] = torch.tensor([ids + g], device="cuda"); full["attention_mask"] = torch.ones_like(full["input_ids"])
        full.pop("mm_token_type_ids", None)
        if "mm_token_type_ids" in inp:
            full["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros((1, len(g)), dtype=inp["mm_token_type_ids"].dtype, device="cuda")], 1)
        lp = torch.log_softmax(model(**full).logits[0, len(ids) - 1:-1].float(), -1)
    top = lp.topk(5, -1)
    ref.append([{"argmax": int(top.indices[i, 0]), "top": [[int(a), float(b)] for a, b in zip(top.indices[i], top.values[i])]} for i in range(len(g))])
    buf = io.BytesIO(); img.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    pairs.append({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}], "forced": g})
    prompts.append(ids)
    print(img.size, inp["image_grid_thw"].tolist(), len(ids), repr(proc.decode(g))[:100])
json.dump(pairs, open(OUT + ".pairs.json", "w")); json.dump(ref, open(OUT + ".ref.json", "w")); json.dump(prompts, open(OUT + ".hfprompts.json", "w"))
