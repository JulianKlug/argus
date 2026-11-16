# Backend `medgemma27` — LOCAL

| Item | Value |
|---|---|
| Adapter | `ollama` (native `/api/chat`) |
| Model | `medgemma-argus:27b` (derived, `config/modelfiles/medgemma-argus.Modelfile`), digest `852e6fdb039823b0cf87695ee4546f749897d7b14d22cc4fd46cc335b1c3295c` |
| Weights | same blob as `medgemma:27b`: `sha256-4b97da6b…` (17.4 GB, local), Q4_K_M, Gemma 3 |
| Server | Ollama 0.34.1, Go template |
| Verified | 2026-11-16 |

Data governance: as in `qwen27.md` (same server).

## Why a derived model

The stock template trims the newline after `<start_of_turn>user` and joins system and user text with
one newline: a system prompt `SYS` and user `hi` render as `<start_of_turn>userSYS\nhi`. The derived
template follows Gemma 3's reference layout `<start_of_turn>user\nSYS\n\nhi<end_of_turn>\n` (checked with
`_debug_render_only`). Weights unchanged.

## Capabilities (probed, L6)

`/api/show`: completion, vision. Live probe: JSON schema honoured. Effective: `JSON_SCHEMA, SEED`.
No tools, no reasoning: requests with tools raise `CapabilityError` before sending (contract suite).

## Tokenizer (criterion 6)

Candidate 2: GGUF vocabulary (`llama`/SentencePiece with merges), BOS added. Ollama differs from
Hugging Face here: a `▁` prefix is added to every text fragment between special tokens, not once per
text (found by raw-prompt probes: `"1"` → 3 tokens on the server, 2 without the per-fragment prefix).

Check: `uv run python scripts/measure_backends.py tokenizer` (2026-11-16, Ollama 0.34.1): 50 seeded
requests mixing system prompts, 1–3 turns, unicode, digits, markup, JSON-schema `format`.

| Requests | pre-count = `prompt_eval_count` | Differences |
|---|---|---|
| 50 | 50 | none |

Exact: `count_tolerance = 0`. Re-run after any Ollama upgrade or model change.

## Determinism (criterion 9)

`measure_backends.py determinism`: 3 prompts × 5 fresh clients, temperature 0, seed 7, reasoning off → distinct answers per prompt [2, 2, 2].

Cause (reproduced, 5 calls after unloading): the first call after a model load (no prompt-cache hit,
`prompt_eval_cached_count = 0`) differs from every later call that reuses the cached prefix (identical
among themselves). Prefill numerics depend on KV-cache state. Example: "certain medications/toxins"
cold vs "certain medications (e.g., metformin, salicylates)" warm. Stock `medgemma:27b` agreed on this
prompt, but the mechanism is the same for every Ollama model: fresh re-runs measure variability; replay
uses records (criterion 10). The contract test compares warm-cache calls only (one warm-up call first).
