"""Workloads beyond scripts/load.py, against an OpenAI-style /v1/chat/completions endpoint (streamed, greedy).

  workloads.py URL MODEL shared OUT.json [--system-tokens 2000] [--requests 128] [--conc 16]
      one long system prompt, then distinct short questions
  workloads.py URL MODEL multiturn OUT.json [--conversations 16] [--turns 8]
      conversations whose every turn resends the history, replies included
  workloads.py URL MODEL long OUT.json [--tokens 8192,32768] [--conc 1]
      long prompts: prompt tokens per second, time to first token, decode speed

Prompt sizes are approximate: text is built from common words, about one token each."""
import argparse, asyncio, json, random, time
import httpx

WORDS = ("the of and to in is was for on that with as by at from his her an which or be are this have it had not "
         "but they were all one their there been has more who will would its can when out up about into than "
         "them only other new some could time these two may then first any like now my such make over our even "
         "most after also made many did before must through back years where much your way well down should "
         "because each just those people how too little state good very make world still own see men work long "
         "get here between both life being under never day same another know while last might us great old year").split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def chat(c, url, model, messages, max_tokens):
    t0 = time.perf_counter(); ttft = None; out = []; usage = {}
    async with c.stream("POST", url + "/v1/chat/completions", json={
            "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}) as r:
        if r.status_code != 200:
            return {"error": r.status_code, "body": (await r.aread())[:200].decode(errors="replace")}
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(delta)
    return {"ttft": ttft, "lat": time.perf_counter() - t0, "text": "".join(out),
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 3) if xs else None


def summary(name, res, wall, **extra):
    ok = [r for r in res if "error" not in r]
    out_toks = sum(r["completion_tokens"] for r in ok)
    in_toks = sum(r["prompt_tokens"] for r in ok)
    s = {"case": name, "ok": len(ok), "errors": len(res) - len(ok), "wall_s": round(wall, 2),
         "out_tok_s": round(out_toks / wall, 1), "prompt_tok_s": round(in_toks / wall, 1),
         "ttft_p50": pct([r["ttft"] for r in ok], .5), "ttft_p95": pct([r["ttft"] for r in ok], .95),
         "lat_p50": pct([r["lat"] for r in ok], .5), **extra}
    print(json.dumps(s), flush=True)
    return s


async def shared(c, a):
    rng = random.Random(3)
    system = text(rng, a.system_tokens)
    qs = [f"Question {i}: in one sentence, what is {text(rng, 8)}?" for i in range(a.requests)]
    it = iter(qs); res = []
    async def worker():
        for q in it:
            res.append(await chat(c, a.url, a.model, [{"role": "system", "content": system},
                                                     {"role": "user", "content": q}], 32))
    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(a.conc)])
    return [summary(f"shared system {a.system_tokens} tok, c={a.conc}", res, time.perf_counter() - t0)]


async def multiturn(c, a):
    rng = random.Random(5)
    res_by_turn = [[] for _ in range(a.turns)]
    async def conversation(k):
        msgs = [{"role": "system", "content": "You are a helpful assistant. " + text(rng, 200)}]
        for turn in range(a.turns):
            msgs.append({"role": "user", "content": f"Turn {turn}: tell me about {text(rng, 60)}"})
            r = await chat(c, a.url, a.model, msgs, 64)
            res_by_turn[turn].append(r)
            msgs.append({"role": "assistant", "content": r.get("text", "")})
    t0 = time.perf_counter()
    await asyncio.gather(*[conversation(k) for k in range(a.conversations)])
    wall = time.perf_counter() - t0
    out = [summary(f"multiturn {a.conversations} x {a.turns}", [r for t in res_by_turn for r in t], wall)]
    out[0]["ttft_p50_by_turn"] = [pct([r.get("ttft") for r in t], .5) for t in res_by_turn]
    print(json.dumps({"ttft_p50_by_turn": out[0]["ttft_p50_by_turn"]}), flush=True)
    return out


async def long(c, a):
    out = []
    for n in [int(x) for x in a.tokens.split(",")]:
        rng = random.Random(n)
        prompts = [f"Read this and summarize it in one line. {text(rng, n)}" for _ in range(a.conc)]
        t0 = time.perf_counter()
        res = await asyncio.gather(*[chat(c, a.url, a.model, [{"role": "user", "content": p}], 128) for p in prompts])
        wall = time.perf_counter() - t0
        ok = [r for r in res if "error" not in r]
        decode = [(r["completion_tokens"] - 1) / (r["lat"] - r["ttft"]) for r in ok
                  if r["ttft"] and r["completion_tokens"] > 1 and r["lat"] > r["ttft"]]
        out.append(summary(f"long {n} tok, c={a.conc}", res, wall,
                           decode_tok_s_per_request=round(sum(decode) / len(decode), 1) if decode else None,
                           prefill_tok_s=round(sum(r["prompt_tokens"] for r in ok) / max(r["ttft"] or 1e-9 for r in ok), 1)
                           if ok else None,
                           first_error=next((r for r in res if "error" in r), None)))
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("url"); p.add_argument("model"); p.add_argument("mode", choices=["shared", "multiturn", "long"])
    p.add_argument("out")
    p.add_argument("--system-tokens", type=int, default=2000); p.add_argument("--requests", type=int, default=128)
    p.add_argument("--conc", type=int, default=16)
    p.add_argument("--conversations", type=int, default=16); p.add_argument("--turns", type=int, default=8)
    p.add_argument("--tokens", default="8192,32768")
    a = p.parse_args()
    async with httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=512)) as c:
        await chat(c, a.url, a.model, [{"role": "user", "content": "hello"}], 4)
        res = await {"shared": shared, "multiturn": multiturn, "long": long}[a.mode](c, a)
    json.dump(res, open(a.out, "w"), indent=1)

asyncio.run(main())
