"""Writes tests/fixtures/tools.templates.json: tool conversations in OpenAI's
format, and each chat template's rendering of them by transformers, the way
vLLM hands them over (tool-call arguments parsed from their JSON strings).

    python scripts/tool_template_reference.py   # needs transformers and jinja2
"""
import json
import pathlib

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city, in °C.",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer"}}}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search notes.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "filters": {"type": "object"},
        "exact": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["query"]}}}


def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CASES = [
    {"name": "tools and a question", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Zürich?"}]},
    {"name": "tools without a system message", "tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find my notes on <tags> & \"quotes\"."}]},
    {"name": "two calls, two results", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Oslo and Lima, and my trip notes."},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            call(0, "get_weather", {"city": "Oslo", "days": 3}),
            call(1, "search", {"query": "trip\nnotes", "filters": {"year": 2026, "kind": ["a", "b"]},
                                "exact": True, "tags": ["travel", "été"]})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": -2}"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Packed light."}]},
    {"name": "a call with no text, then an answer and a new question", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "You help with weather."},
        {"role": "user", "content": "Lima?"},
        {"role": "assistant", "content": None, "tool_calls": [call(2, "get_weather", {"city": "Lima"})]},
        {"role": "tool", "tool_call_id": "call_2", "content": "19 C, cloudy"},
        {"role": "assistant", "content": "19 °C and cloudy."},
        {"role": "user", "content": "And tomorrow?"}]},
]


def hf_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def main():
    out = {"cases": CASES, "renders": {}}
    for t in sorted((FIX / "templates").glob("*.jinja")):
        template = t.read_text()
        for thinking in (False, True):
            key = f"{t.stem} thinking={str(thinking).lower()}"
            out["renders"][key] = [
                render_jinja_template([hf_messages(c["messages"])], tools=c["tools"], chat_template=template,
                                      add_generation_prompt=True, enable_thinking=thinking)[0][0]
                for c in CASES]
    (FIX / "tools.templates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
