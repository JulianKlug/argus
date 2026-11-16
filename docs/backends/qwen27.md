# Backend `qwen27` — LOCAL

| Item | Value |
|---|---|
| Adapter | `ollama` (native `/api/chat`) |
| Model | `qwen3.8:27b`, Q4_K_M, digest `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| Weights | `/usr/share/ollama/.ollama/models/blobs/sha256-f5f1dd89…` (16.8 GB, local) |
| Server | Ollama 0.34.1, renderer `qwen3.8`, parser `qwen3.5` (built-in, Go) |
| Verified | 2026-11-16 |

## Data governance

Shared by all LOCAL backends (same server):

- Bind address: `OLLAMA_HOST=127.0.0.1:11434` (systemd unit); `ss -ltn` shows `127.0.0.1:11434` only.
- Cloud: no configured model is cloud-form, `/api/tags` and `/api/show` report no `remote_host`.
  Ollama 0.34.1 proxies any POST whose model ends in `:cloud` / `-cloud` to ollama.com before
  local lookup; `policy.py` refuses such names before any request. `OLLAMA_NO_CLOUD` is **not set**:
  setting it in the systemd unit would also stop the 4-hourly model-recommendation fetch from
  ollama.com (no request content, but outbound traffic). Operational action for the owner.
- Telemetry: none found in the 0.34.1 server source (grep `telemetry|analytics`); outbound calls
  are cloud proxying (unused) and model recommendations (see above).
- Network-level enforcement (firewall) is out of scope (spec, Out of Scope).

## Capabilities (probed, L6)

`/api/show`: completion, vision, tools, thinking. Live probes: tool call parsed, JSON schema honoured.
Effective: `TOOLS, JSON_SCHEMA, REASONING, SEED`.

Reasoning switch: the renderer defaults to `xhigh` reasoning instructions when `think` is absent, so
the client always sends `think` explicitly (`true` = medium effort, no extra instructions).

## Tokenizer (criterion 6)

Candidate 2: GGUF vocabulary (`tokenizer.ggml.model = gpt2`, pre `qwen35`) via `tokenizers`, prompt
rendered by the server (`_debug_render_only`, `truncate: false`).

Check: `uv run python scripts/measure_backends.py tokenizer` (2026-11-16, Ollama 0.34.1): 50 seeded
requests mixing system prompts, 1–3 turns, unicode, digits, markup, tool definitions and two-call tool exchanges, reasoning on/off, JSON-schema `format`.

| Requests | pre-count = `prompt_eval_count` | Differences |
|---|---|---|
| 50 | 50 | none |

Exact: `count_tolerance = 0`. Re-run after any Ollama upgrade or model change.

## Determinism (criterion 9)

`measure_backends.py determinism`: 3 prompts × 5 fresh clients, temperature 0, seed 7, reasoning off → distinct answers per prompt [1, 1, 1].
Identical on this GPU, server and load; not a claim of bitwise determinism across loads, batch sizes or upgrades.
