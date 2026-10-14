"""Closed-loop and burst load against an OpenAI-style /v1/chat/completions endpoint (streamed): output tok/s, TTFT and
latency percentiles. Usage: load.py URL MODEL OUT.json
LOAD_CASES="1:16,16:96" picks other closed-loop cases (requests in flight:total); LOAD_BURST=0 skips the burst;
LOAD_MODELS="a,b" sends requests to those model names in turn (LoRA adapters) instead of MODEL."""
import os
import asyncio, json, random, sys, time
import httpx

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WORDS = "the receipt shows a total amount paid by card at the store on the date with tax and tip included for".split()
rng = random.Random(7)
def prompt():
    n = rng.randint(40, 150)   # (~60-200 tokens with the chat template)
    return "Summarize briefly: " + " ".join(rng.choice(WORDS) for _ in range(n))
REQS = [(prompt(), rng.choice([16, 32, 48, 64])) for _ in range(512)]

MODELS = os.environ.get("LOAD_MODELS", "").split(",") if os.environ.get("LOAD_MODELS") else [MODEL]
SENT = [0]

async def one(c, p, mt):
    t0 = time.perf_counter(); ttft = None; toks = 0
    model = MODELS[SENT[0] % len(MODELS)]; SENT[0] += 1
    async with c.stream("POST", URL + "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": p}],
            "max_tokens": mt, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}) as r:
        if r.status_code != 200:
            return {"err": r.status_code, "body": (await r.aread())[:200].decode(errors="replace")}
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                toks = d["usage"].get("completion_tokens") or toks
            ch = (d.get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content") and ttft is None:
                ttft = time.perf_counter() - t0
    return {"ttft": ttft, "lat": time.perf_counter() - t0, "toks": toks}

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None); return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 3) if xs else None

async def closed(c, conc, total):
    it = iter(REQS[:total]); res = []
    async def worker():
        for p, mt in it:
            res.append(await one(c, p, mt))
    t0 = time.perf_counter(); await asyncio.gather(*[worker() for _ in range(conc)]); wall = time.perf_counter() - t0
    return summarize(f"closed c={conc}", res, wall)

async def burst(c, n):
    t0 = time.perf_counter(); res = await asyncio.gather(*[one(c, p, mt) for p, mt in REQS[:n]]); wall = time.perf_counter() - t0
    return summarize(f"burst n={n}", res, wall)

def summarize(name, res, wall):
    ok = [r for r in res if "err" not in r]; toks = sum(r["toks"] for r in ok)
    s = {"case": name, "ok": len(ok), "errors": len(res) - len(ok), "wall_s": round(wall, 2), "out_tok_s": round(toks / wall, 1),
         "ttft_p50": pct([r["ttft"] for r in ok], .5), "ttft_p95": pct([r["ttft"] for r in ok], .95),
         "lat_p50": pct([r["lat"] for r in ok], .5), "lat_p95": pct([r["lat"] for r in ok], .95),
         "first_error": next((r for r in res if "err" in r), None)}
    print(json.dumps(s), flush=True); return s

async def main():
    out = []
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=256)) as c:
        await one(c, "hello", 8)   # warm
        cases = os.environ.get("LOAD_CASES", "1:16,8:64,16:96,32:128,64:192")
        for conc, total in (map(int, x.split(":")) for x in cases.split(",")):
            out.append(await closed(c, conc, total))
        if os.environ.get("LOAD_BURST", "1") != "0":
            out.append(await burst(c, 64))
    json.dump(out, open(OUT, "w"), indent=1)
asyncio.run(main())
