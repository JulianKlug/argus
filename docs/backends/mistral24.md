# Backend `mistral24` — LOCAL

| Item | Value |
|---|---|
| Adapter | `openai_compat` (Ollama `/v1/chat/completions`); control API `http://127.0.0.1:11434` |
| Model | `mistral-small3.2-argus:24b` (derived, `config/modelfiles/mistral-small3.2-argus.Modelfile`), digest `810b24690f536970d4ad962dafda4db0a2d8b48b3a877639d537a3348b3558a3` |
| Weights | same blob as `mistral-small3.2:24b`: `sha256-41a5b0c3…` (15.2 GB, local), Q4_K_M |
| Server | Ollama 0.34.1, Go template + built-in `ministral` parser |
| Verified | 2026-11-16 |

Data governance: as in `qwen27.md` (same server).

## Why a derived model

On Ollama 0.34.1 the stock `mistral-small3.2:24b` reports `tools`, but its calls are lost: the model
emits `[TOOL_CALLS]name[ARGS]{…}`, the stock template expects `[TOOL_CALLS]name[CALL_ID]i[ARGS]{…}`,
and the generic parser drops the turn (empty content, no calls, `finish = stop`). The live TOOLS
probe (L6) removes TOOLS from the stock model; the client also treats an empty `stop` turn with tools
offered as invalid (L5).

The derived Modelfile keeps the weights and changes only:
- `PARSER ministral` (Ollama built-in, parses `[TOOL_CALLS]name[ARGS]{…}`),
- tool history rendered in the format the model emits, tool results as `[TOOL_RESULTS]…[/TOOL_RESULTS]`,
- tool definitions before the last user message (stock: only when it is among the last two messages,
  so tools vanished during round trips).

The Modelfile inherits the vendor system prompt ("Le Chat"); agents always send their own system message.

## Capabilities (probed, L6)

`/api/show`: completion, vision, tools. Live probes: tool call parsed, JSON schema honoured.
Effective: `TOOLS, JSON_SCHEMA, SEED`. No reasoning switch.

## Context on /v1

`/v1` cannot set `num_ctx` or `truncate`; the server default applies (32768 on this 24 GB GPU,
lowered to 4096 if it does not fit). Over it, Ollama drops old messages silently, also in render-only
mode. The adapter therefore pre-counts through the native render (`truncate: false`): a truncated /v1
evaluation then differs from the pre-count and raises `ContextOverflowError` (contract test
`test_forced_truncation_detected[mistral24]`). `context_tokens` must not exceed the server default.

## Tokenizer (criterion 6)

Candidate 2: GGUF vocabulary (`gpt2`, pre `llama-bpe`), BOS added; prompt rendered natively by the server.

Check: `uv run python scripts/measure_backends.py tokenizer` (2026-11-16, Ollama 0.34.1): 50 seeded
requests mixing system prompts, 1–3 turns, unicode, digits, markup, tool definitions and two-call tool exchanges, JSON-schema `format`.

| Requests | pre-count = `prompt_eval_count` | Differences |
|---|---|---|
| 50 | 50 | none |

Exact: `count_tolerance = 0`. Re-run after any Ollama upgrade or model change.

## Determinism (criterion 9)

`measure_backends.py determinism`: 3 prompts × 5 fresh clients, temperature 0, seed 7, reasoning off → distinct answers per prompt [1, 1, 1].
Identical on this GPU, server and load; not a claim of bitwise determinism across loads, batch sizes or upgrades.
