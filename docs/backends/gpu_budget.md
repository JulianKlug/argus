# GPU budget (criterion 14)

Measured 2026-11-16 with `uv run python scripts/measure_backends.py gpu`: 1× RTX 3090 (24 GB),
Ollama 0.34.1, all Q4_K_M, one model resident at a time. Synthetic chart text, temperature 0.

## Per model

| Model (backend) | Max context 100 % on GPU | Prefill tok/s (≈3.4k / 13.5k / 45k prompt) | Decode tok/s (same prompts) |
|---|---|---|---|
| `qwen3.8:27b` (qwen27) | 65,536 (81,920: 91 % GPU) | 1,162 / 1,173 / 1,003 | 48 / 58 / 50; reasoning on: 72 |
| `mistral-small3.2-argus:24b` (mistral24) | 49,152 (57,344: 90 % GPU) | 1,776 / 1,726 / 1,431 (34k) | 52 / 48 / 41 |
| `medgemma-argus:27b` (medgemma27) | 65,536 (81,920: 89 % GPU) | 1,327 / 1,400 / 1,324 | 41 / 39 / 35 |

- Search grid: 8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k, 80k tokens; "100 % GPU" = `size_vram == size` in `/api/ps`.
  The true maximum lies between the last fitting and first failing grid point.
- Qwen3.8 is hybrid (attention every 4th layer): VRAM grows only ~0.1 GB from 8k to 64k. Mistral's grows ~0.16 GB per 1k tokens.
- Qwen decodes faster with reasoning on (72 vs ~50 tok/s): its Modelfile enables draft prediction
  (`draft_num_predict 4`), which pays off on predictable thinking text. One measurement; treat as indicative.
- `config/llm.toml` uses these maxima, except mistral24: Ollama's `/v1` cannot set `num_ctx` and runs at the server default (32,768).

## Model swap

Unload A, load B, first token from B (wall clock, 1-token answer):

| From → to | s |
|---|---|
| qwen → mistral | 6.6 |
| qwen → medgemma | 7.2 |
| mistral → qwen | 7.8 |
| mistral → medgemma | 7.4 |
| medgemma → qwen | 7.6 |
| medgemma → mistral | 6.3 |

Models had been loaded minutes earlier, so weights were likely in the page cache (61 GB RAM fits all three). A cold load from disk was not measured.

## Sequence cost per window

Model: `Σ_calls [prompt / prefill + (answer + reasoning) / decode] + swaps × swap time`, using the
slowest measured rate per model. Calls are serialized on one GPU, so parallel topologies queue.

Assumptions until D5 measures real windows: window prompt 8,000 tokens, answer 512, reasoning 2,048
(expert only, Qwen). Chained topologies add 512 tokens per previous answer. No KV-prefix reuse across
agents (conservative). One swap at window start if the window ends on a different model than it starts.

| Topology (calls) | all Qwen | MedGemma summary + adjudication | Mistral research |
|---|---|---|---|
| single LLM (expert) | 61 s · 17.0 GPU-h | 61 s · 17.0 | 61 s · 17.0 |
| independent (summary, expert, research, adjudication) | 117 s · 32.6 | 136 s · 37.7 | 131 s · 36.4 |
| sequential (same, chained) | 120 s · 33.4 | 139 s · 38.5 | 134 s · 37.2 |
| collaborative (+1 expert/research revision round) | 205 s · 56.9 | 223 s · 61.9 | 232 s · 64.4 |

Cells: GPU seconds per window · GPU hours per 1,000 windows.

## GPU hours per study arm

GPU hours = windows (D6) × seconds per window / 3600. D6 has not fixed the window count yet, so the
table above is per 1,000 windows. Example: 50 validation patients × 20 windows = 1,000 windows →
17 GPU-h (single LLM) to 64 GPU-h (collaborative, mixed models). Swaps add 14–27 s per window for
mixed assignments. Batching windows by model (harness, A1) removes most of that.

## Re-run

After any Ollama upgrade, model change or GPU change: `scripts/measure_backends.py gpu`, then update this file and `context_tokens`.
