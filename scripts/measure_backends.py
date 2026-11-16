"""Backend measurements for D4 (docs/spec_llm_backbone.md): tokenizer exactness, determinism, GPU budget.

Synthetic prompts only, never MIMIC data. Needs ARGUS_RUN_DIR (owner-only, outside the repo) and Ollama.

    uv run python scripts/measure_backends.py tokenizer   [--n 50]    criterion 6
    uv run python scripts/measure_backends.py determinism [--n 5]     criterion 9
    uv run python scripts/measure_backends.py gpu         [--windows 1000]  criterion 14

Results are printed as JSON; docs/backends/*.md cite them.

Sequence cost model (per window, one GPU, calls serialized):

    cost = Σ_calls [ prompt / prefill_rate + (answer + reasoning) / decode_rate ] + swaps × swap_time
"""

import argparse
import json
import os
import random
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel

from argus.llm import (
    CallContext, Capability, ContextOverflowError, LlmClient, Message, Request, Role, StructuredOutputError,
    ToolCall, ToolCallError, ToolSpec,
)
from argus.llm import policy
from argus.llm.backends.ollama_admin import OllamaAdmin
from argus.llm.config import DEFAULT_CONFIG, _backend
from argus.llm.sink import RECORD_FILE, read_records

BACKENDS = tomllib.loads(DEFAULT_CONFIG.read_text())["backends"]
CTX = CallContext("measure", "measure")
SEED = 3

# Context candidates for "max context fully on GPU" (tokens).
CONTEXT_CANDIDATES = (8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536, 81920, 98304, 131072, 196608, 262144)
SPEED_CONTEXTS = (4096, 16384)
FILLER_TOKENS_PER_LINE = 30     # upper estimate; reported speeds use the server's actual prompt count
MAX_CONTEXT_FILL = 0.85
DECODE_TOKENS = 128
REASONING_DECODE_TOKENS = 512

# Sequence-cost assumptions until D5 measures real windows (tokens per call).
WINDOW_PROMPT = 8000
ANSWER = 512
REASONING = 2048
WORDS = "lactate MAP norepinephrine 0.15µg/kg/min SpO₂ 92% FiO₂ PEEP 8 cmH₂O creatinine 1.8 mg/dL Ünïcödé 日本 😀 <b>tag</b> & \"quoted\" {json: 1} \n\n tab\there".split(" ")


class LabQuery(BaseModel):
    name: str
    hours: int | None = None


class Window(BaseModel):
    start: str
    end: str


class Assessment(BaseModel):
    risk: float
    reasons: list[str]
    windows: list[Window]


TOOLS = (ToolSpec("get_lab", "Fetch labs <since> & until", LabQuery), ToolSpec("get_window", "Vitals window", Window))


def _config(path: Path, backend: str) -> Path:
    """Committed backends + agents `off`/`on` (reasoning) on `backend`, max_tokens 1: prompt counting only."""
    lines = []
    for name, entry in BACKENDS.items():
        lines.append(f"[backends.{name}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in entry.items()]
    for agent, reasoning in (("off", "OFF"), ("on", "ON")):
        lines += [f"[agents.{agent}]", f'backend = "{backend}"', f'reasoning = "{reasoning}"', "max_tokens = 1"]
    path.write_text("\n".join(lines) + "\n")
    return path


def _text(rnd: random.Random, n: int) -> str:
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def _random_request(rnd: random.Random, tools: bool) -> Request:
    """System (80%), 1–3 user turns, optional tool exchange with two linked results."""
    messages = []
    if rnd.random() < 0.8:
        messages.append(Message(Role.SYSTEM, _text(rnd, rnd.randint(3, 60))))
    turns = rnd.randint(1, 3)
    use_tools = tools and rnd.random() < 0.5
    for t in range(turns):
        messages.append(Message(Role.USER, _text(rnd, rnd.randint(1, 300))))
        if t == turns - 1:
            continue
        if use_tools and rnd.random() < 0.6:
            calls = (ToolCall("call_a", "get_lab", {"name": "lactate", "hours": 6}),
                     ToolCall("call_b", "get_window", {"start": "08:00", "end": "14:00"}))
            messages += [
                Message(Role.ASSISTANT, _text(rnd, rnd.randint(0, 10)) if rnd.random() < 0.5 else "", tool_calls=calls),
                Message(Role.TOOL, json.dumps({"lactate": [2.1, 4.8]}), tool_call_id="call_a", tool_name="get_lab"),
                Message(Role.TOOL, _text(rnd, 20), tool_call_id="call_b", tool_name="get_window"),
                Message(Role.USER, _text(rnd, 5)),
            ]
        else:
            messages.append(Message(Role.ASSISTANT, _text(rnd, rnd.randint(1, 80))))
    return Request(tuple(messages), TOOLS if use_tools else ())


def tokenizer_check(work: Path, n: int) -> dict:
    """Criterion 6: pre-count vs server prompt_eval_count on `n` varied requests per backend."""
    results = {}
    for backend in BACKENDS:
        config = _config(work / f"{backend}.toml", backend)
        clients = {"off": LlmClient.for_agent("off", config)}
        if Capability.REASONING in clients["off"].capabilities.supported:
            clients["on"] = LlmClient.for_agent("on", config)
        supported = clients["off"].capabilities.supported
        start = len(_records())

        rnd = random.Random(SEED)
        for _ in range(n):
            client = clients[rnd.choice(sorted(clients))]
            request = _random_request(rnd, Capability.TOOLS in supported)
            try:
                if Capability.JSON_SCHEMA in supported and not request.tools and rnd.random() < 0.3:
                    client.structured(request, Assessment, CTX)
                else:
                    client.generate(request, CTX)
            except (StructuredOutputError, ToolCallError, ContextOverflowError):
                pass  # max_tokens = 1: cut-off answers; counts are in the record

        records = [r for r in _records()[start:] if r["raw_response"] is not None]
        diffs = [_server_count(r) - r["pre_count"] for r in records]
        results[backend] = {
            "requests": len(records), "exact": sum(d == 0 for d in diffs), "diffs": sorted(set(diffs)),
            "server_version": records[0]["server_version"] if records else None,
        }
        print(json.dumps({backend: results[backend]}), file=sys.stderr)
    return results


def _server_count(record: dict) -> int:
    raw = record["raw_response"]
    return raw["prompt_eval_count"] if "prompt_eval_count" in raw else raw["usage"]["prompt_tokens"]


def determinism(work: Path, n: int) -> dict:
    """Criterion 9: `n` fresh clients, same request, seed, temperature 0 -> number of distinct answers."""
    prompts = [
        "List three causes of lactic acidosis, comma-separated.",
        "Synthetic case: MAP 58 mmHg on norepinephrine 0.3 µg/kg/min, lactate 4.8. Name the next two steps.",
        "Explain in two sentences why PEEP can lower cardiac output.",
    ]
    results = {}
    for backend in BACKENDS:
        config = _config(work / f"{backend}.toml", backend).read_text().replace("max_tokens = 1", "max_tokens = 256")
        path = work / f"{backend}-det.toml"
        path.write_text(config)
        distinct = []
        for prompt in prompts:
            request = Request((Message(Role.USER, prompt),))
            answers = {LlmClient.for_agent("off", path).generate(request, CTX).message.content for _ in range(n)}
            distinct.append(len(answers))
        results[backend] = {"runs_per_prompt": n, "distinct_answers_per_prompt": distinct}
        print(json.dumps({backend: results[backend]}), file=sys.stderr)
    return results


def _filler(tokens: int, salt: int) -> str:
    """≈`tokens` tokens of synthetic chart text; unique salt defeats the KV prefix cache."""
    line = "hour {h} HR {hr} MAP {m} SpO2 {s} lactate {l};"
    rnd = random.Random(salt)
    out = [f"case {salt}:"]
    while len(out) * FILLER_TOKENS_PER_LINE < tokens:
        out.append(line.format(h=rnd.randint(0, 23), hr=rnd.randint(60, 140), m=rnd.randint(50, 100), s=rnd.randint(85, 100), l=rnd.randint(5, 60) / 10))
    return " ".join(out) + " Summarise in one sentence."


def gpu_budget(windows: int) -> dict:
    """Criterion 14: max context on GPU, prefill/decode rates, swap times, sequence cost per topology."""
    admin_config = _backend(next(iter(BACKENDS)), next(iter(BACKENDS.values())))
    admin = OllamaAdmin(admin_config, policy.check_response_status)
    models = {}
    for name, entry in BACKENDS.items():
        models.setdefault(entry["model"], name)

    measured = {}
    for model, backend in models.items():
        reasoning = BACKENDS[backend]["model"].startswith("qwen")
        best = 0
        for ctx in CONTEXT_CANDIDATES:
            admin.unload_all()
            try:
                admin.load(model, ctx)
            except Exception as e:  # out of memory or refused: stop searching
                print(f"{model} @ {ctx}: {type(e).__name__}", file=sys.stderr)
                break
            resident = next(r for r in admin.resident() if r.model == model)
            print(f"{model} @ {ctx}: vram {resident.size_vram} / {resident.size}, ctx {resident.context_length}", file=sys.stderr)
            if not resident.fully_on_gpu or resident.context_length != ctx:
                break
            best = ctx

        admin.unload_all()
        admin.load(model, best)
        speeds = {}
        for i, target in enumerate((*SPEED_CONTEXTS, int(best * MAX_CONTEXT_FILL))):
            timing = admin.generate(model, _filler(target, salt=i), best, DECODE_TOKENS, False if reasoning else None)
            speeds[str(timing.prompt_tokens)] = {
                "prefill_tok_s": timing.prompt_tokens / timing.prefill_s,
                "decode_tok_s": timing.completion_tokens / timing.decode_s,
            }
        if reasoning:
            timing = admin.generate(model, _filler(SPEED_CONTEXTS[0], salt=99), best, REASONING_DECODE_TOKENS, True)
            speeds["reasoning_on"] = {"decode_tok_s": timing.completion_tokens / timing.decode_s}
        measured[model] = {"backend": backend, "max_context_on_gpu": best, "speeds": speeds}
        print(json.dumps({model: measured[model]}), file=sys.stderr)

    swaps = {}
    names = list(models)
    for a in names:
        for b in names:
            if a == b:
                continue
            admin.unload_all()
            admin.load(a, measured[a]["max_context_on_gpu"])
            timing = admin.generate(b, "Say ok.", measured[b]["max_context_on_gpu"], 1, False if b.startswith("qwen") else None)
            swaps[f"{a} -> {b}"] = timing.wall_s
    print(json.dumps({"swaps": swaps}), file=sys.stderr)

    return {"models": measured, "swap_s": swaps, "sequences": _sequence_costs(measured, swaps, windows)}


# Candidate topologies (D8 / A7 arms): ordered (agent, reasoning) calls per window.
TOPOLOGIES = {
    "single_llm": [("expert", True)],
    "independent": [("summary", False), ("expert", True), ("research", False), ("adjudication", False)],
    "sequential": [("summary", False), ("expert", True), ("research", False), ("adjudication", False)],
    "collaborative": [("summary", False), ("expert", True), ("research", False), ("expert", True), ("research", False), ("adjudication", False)],
}
# Agent -> model assignments to compare.
ASSIGNMENTS = {
    "all_qwen": {"summary": "qwen3.8:27b", "expert": "qwen3.8:27b", "research": "qwen3.8:27b", "adjudication": "qwen3.8:27b"},
    "medgemma_summary_adjudication": {"summary": "medgemma-argus:27b", "expert": "qwen3.8:27b", "research": "qwen3.8:27b", "adjudication": "medgemma-argus:27b"},
    "mistral_research": {"summary": "qwen3.8:27b", "expert": "qwen3.8:27b", "research": "mistral-small3.2-argus:24b", "adjudication": "qwen3.8:27b"},
}


def _sequence_costs(measured: dict, swaps: dict, windows: int) -> dict:
    """GPU seconds per window and GPU hours per `windows`, per topology × assignment (assumed token sizes)."""
    out = {}
    for topology, calls in TOPOLOGIES.items():
        for assignment, models in ASSIGNMENTS.items():
            seconds, previous, prompt = 0.0, None, WINDOW_PROMPT
            chained = topology in ("sequential", "collaborative")
            for agent, reasoning in calls:
                model = models[agent]
                speeds = measured[model]["speeds"]
                prefill = min(v["prefill_tok_s"] for k, v in speeds.items() if k != "reasoning_on")
                decode = min(v["decode_tok_s"] for k, v in speeds.items() if k != "reasoning_on")
                think = reasoning and "reasoning_on" in speeds
                seconds += prompt / prefill + (ANSWER + (REASONING if think else 0)) / decode
                if previous is not None and previous != model:
                    seconds += swaps[f"{previous} -> {model}"]
                previous = model
                if chained:
                    prompt += ANSWER
            # Next window starts with the first call's model: one more swap if the window ended elsewhere.
            first = models[calls[0][0]]
            if previous != first:
                seconds += swaps[f"{previous} -> {first}"]
            out[f"{topology}/{assignment}"] = {"gpu_s_per_window": seconds, f"gpu_h_per_{windows}_windows": seconds * windows / 3600}
    return out


def _records() -> list[dict]:
    path = Path(os.environ[policy.RUN_DIR_ENV]) / RECORD_FILE
    return read_records(path) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["tokenizer", "determinism", "gpu"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--windows", type=int, default=1000)
    args = parser.parse_args()

    work = policy.check_run_dir(os.environ) / "measure"
    work.mkdir(mode=0o700, exist_ok=True)
    if args.what == "tokenizer":
        result = tokenizer_check(work, args.n or 50)
    elif args.what == "determinism":
        result = determinism(work, args.n or 5)
    else:
        result = gpu_budget(args.windows)
    print(json.dumps(result, indent=1, default=asdict))


if __name__ == "__main__":
    main()
