# Spec: Provider-independent LLM interface (D4)

## Context

Agents (A2–A6) need generation, structured output and tool calls from an LLM. Models change
during the study: A7 compares backends, V1 freezes one per agent. Agents must therefore never
talk to a provider; swapping a model is a config change (roadmap, L11).

Hosting: local models first. A verified zero-retention remote backend is added once the
pipeline works end to end on local models and must pass the same tests. MIMIC text leaves
the machine only through such a backend (PhysioNet LLM policy).

Hardware: 1× RTX 3090 (24 GB), 61 GB RAM. Ollama 0.34 installed. Candidate local models
all Q4_K_M, ~15–19 GB, one resident at a time. Probed 2026-11-16:

| Model | Tools | Reasoning | JSON schema | Use |
|---|---|---|---|---|
| `qwen3.8:27b` | ✓ | ✓ | ✓ | any agent |
| `mistral-small3.2:24b` | ✓ | — | ✓ | any agent, reasoning OFF |
| `medgemma:27b` (Gemma 3, medical fine-tune) | — (Ollama error) | — (Ollama error) | ✓ | agents without tools (summary, adjudication), reasoning OFF |

## Design

```
 agents / harness (A1–A6)
   │  client = LlmClient.for_agent("expert")
   │  client.generate(messages) / .structured(messages, Schema) / .count_tokens(text)
   ▼
 client.py ──────► policy.py      refuse backend not LOCAL or verified ZERO_RETENTION
   │  validate, retry, overflow check, CallRecord ──► sink.py  (JSONL, run dir)
   ▼
 backends/base.py   Backend protocol + Capabilities
   ├── ollama.py             native /api/chat (format = JSON schema, tools, num_ctx, seed)
   └── openai_compat.py      /v1/chat/completions (vLLM, Ollama /v1, remote zero-retention)
   ▲
 config.py ◄── config/llm.toml   backends + per-agent model and inference settings
```

- Each layer calls only the one below. Agents see messages, schemas and usage, never URLs,
  provider SDKs or HTTP.
- Two adapters, both tested locally: `openai_compat` runs against Ollama's `/v1` endpoint now
  and against the remote endpoint later. No new adapter is needed for the remote phase.
- No provider SDK; `httpx` only. Structured schemas are pydantic models.

### Decisions (locked)

| # | Decision |
|---|---|
| L1 | Backend = (adapter, base URL, model, digest). Two local backends = two model families (backend sensitivity in A7 needs different models, not different servers) |
| L2 | Sync API. One GPU: Ollama serializes calls; parallel topologies (D8) queue |
| L3 | Context overflow raises `ContextOverflowError`; silent truncation (Ollama default) never reaches an agent |
| L4 | Structured output: backend-native schema constraint, then pydantic validation; one retry on invalid output, then `StructuredOutputError` with raw text logged |
| L5 | D4 returns tool calls, never executes them; the harness (A1) runs tools and sends results back as messages |
| L6 | Token counts come from the backend's own tokenizer (`prompt_eval_count` / `usage.prompt_tokens`), not a substitute tokenizer. D5 uses `count_tokens` |
| L7 | Defaults: `temperature = 0`, fixed `seed`. Every call records model digest and full settings (A1 replay) |
| L8 | Call records hold patient text: written only under `ARGUS_RUN_DIR` (outside the repo, same controls as the store) |
| L9 | Reasoning (“thinking”) tokens recorded separately from the answer and never parsed as output |

### Public API (`src/argus/llm/__init__.py`)

```python
class Role(Enum):           SYSTEM, USER, ASSISTANT, TOOL
class DataPolicy(Enum):     LOCAL, ZERO_RETENTION
class FinishReason(Enum):   STOP, LENGTH, TOOL_CALLS

@dataclass(frozen=True)
class Message:      role: Role; content: str; tool_call_id: str | None = None
@dataclass(frozen=True)
class ToolSpec:     name: str; description: str; parameters: type[BaseModel]
@dataclass(frozen=True)
class ToolCall:     id: str; name: str; arguments: dict
@dataclass(frozen=True)
class Usage:        prompt_tokens: int; completion_tokens: int; reasoning_tokens: int
                    latency_s: float; load_s: float; cost: float     # local: GPU-s; remote: currency per config
@dataclass(frozen=True)
class Capabilities: context_tokens: int; json_schema: bool; tools: bool; reasoning: bool; seed: bool
@dataclass(frozen=True)
class Completion:   text: str; tool_calls: list[ToolCall]; finish: FinishReason
                    usage: Usage; record_id: str

class LlmClient:
    @classmethod
    def for_agent(cls, agent: str, config: Path | None = None) -> LlmClient
    capabilities: Capabilities
    def generate(self, messages, tools: list[ToolSpec] = []) -> Completion
    def structured(self, messages, schema: type[T]) -> tuple[T, Completion]
    def count_tokens(self, text: str) -> int

errors: BackendUnavailableError, CapabilityError, ContextOverflowError,
        StructuredOutputError, PolicyViolationError
```

Agent names are config keys; the agents layer owns the constants.

### Config (`config/llm.toml`, committed; no secrets)

```toml
[backends.qwen27]
adapter = "ollama"
base_url = "http://127.0.0.1:11434"
model = "qwen3.8:27b"
policy = "LOCAL"
context_tokens = 32768        # measured, see GPU budget
gpu_cost_per_s = 1.0

[backends.mistral24]
adapter = "openai_compat"
base_url = "http://127.0.0.1:11434/v1"
model = "mistral-small3.2:24b"
policy = "LOCAL"
context_tokens = 32768

[backends.medgemma27]
adapter = "ollama"
base_url = "http://127.0.0.1:11434"
model = "medgemma:27b"
policy = "LOCAL"
context_tokens = 32768

[agents.expert]
backend = "qwen27"
temperature = 0.0
seed = 7
max_tokens = 2048
reasoning = "ON"              # ON | OFF; ON on a backend without reasoning → CapabilityError

[agents.summary]
backend = "mistral24"
...
```

Remote backends later add `policy = "ZERO_RETENTION"`, `verification = "docs/backends/<name>.md"`,
`api_key_env = "..."` (key read from env, never from the file).

### Policy rules (`policy.py`)

Checked at `LlmClient` construction; violation → `PolicyViolationError`, no request sent.

| Policy | Allowed iff |
|---|---|
| LOCAL | host is loopback; Ollama model is not a cloud model (`-cloud` / `:cloud` tag, or `ollama show` reports a remote host) |
| ZERO_RETENTION | `verification` file exists and states: provider, contract / setting ensuring no retention and no training, date verified, PhysioNet DUA reference |

Every backend has a verification note in `docs/backends/<name>.md`; for LOCAL: bind address,
no cloud models, Ollama telemetry status.

### Call record (`sink.py`)

One JSONL line per attempt (retries included) in `$ARGUS_RUN_DIR/llm_calls.jsonl`:
`record_id, agent, backend, model, digest, settings, messages, tools, schema, raw_response,
reasoning, completion, usage, error, started_at`. A1 builds its run log on these records.
`ARGUS_RUN_DIR` unset → `LlmClient` refuses to start (no silent default inside the repo).

### GPU budget

Measured once per local backend, written to `docs/backends/gpu_budget.md`:

| Measure | How |
|---|---|
| Max context fully on GPU | Largest `num_ctx` with 100% GPU layers (`ollama ps`), KV cache included |
| Prefill / decode tokens·s⁻¹ | At 4k, 16k and max context |
| Model swap time | Unload A, first token from B |
| GPU-hours per study arm | windows (50 patients × windows per patient, D6) × calls per window × measured time |

`context_tokens` in config is the measured maximum, not the model's nominal one.
If D5 windows exceed it, D5 must cut the window, not D4.

## Acceptance Criteria

1. Agent code imports only `argus.llm`; no module outside `backends/` imports `httpx` or names a URL path.
2. Same contract suite passes on every configured backend: plain generation, structured output into a nested pydantic schema, reasoning on and off; on backends declaring tools: single and multiple tool calls, tool-result round trip. ≥2 backends pass the full suite including tools.
3. Structured output invalid on first attempt → one retry → valid result, or `StructuredOutputError`; both attempts recorded.
4. Prompt larger than `context_tokens` → `ContextOverflowError` before or after the call; never a completion from a truncated prompt (test: sentinel instruction at the start of an oversized prompt).
5. Schema or tools requested on a backend whose capabilities lack them → `CapabilityError`, no request sent.
6. Non-loopback LOCAL backend, Ollama cloud model, or ZERO_RETENTION without verification file → `PolicyViolationError`, no request sent (fake transport asserts zero requests).
7. `count_tokens` equals the backend's `prompt_tokens` for the same text (± chat template overhead, measured and constant per backend).
8. Same agent, prompt, seed, temperature 0: two calls give identical text on each local backend, or the variability is measured and reported (feeds A1 fresh-run variability).
9. Every call writes one record with model digest, full settings, usage, latency; reasoning tokens separate from answer.
10. Swapping an agent's backend changes only `config/llm.toml`; contract suite re-run green.
11. GPU budget table filled for each local backend.
12. `docs/backends/<name>.md` exists for every configured backend.
13. Ollama down → `BackendUnavailableError` with backend name; no retry loop longer than configured timeout.

## Testing Plan

| Layer | What | Marker |
|---|---|---|
| Unit | policy rules, config parsing, retry, overflow, capability refusal, record writing — fake `httpx` transport | — |
| Unit | each adapter maps request / response / tool calls / usage correctly from recorded fixtures | — |
| Contract | criteria 2, 4, 7, 8 parametrized over configured backends | `llm` (needs running Ollama) |
| Measurement | GPU budget script `scripts/measure_backends.py` | manual |

Tests never send MIMIC data; contract prompts are synthetic.

## Rules for the implementing agent

Same as `spec_snapshot_engine.md` (rules 1–5). Additionally:
6. Policy checks live only in `policy.py`; provider specifics live only in `backends/`.
7. No provider SDK dependency. Add `httpx`, `pydantic` to `pyproject.toml`.

## Files Reference

| File | Change |
|---|---|
| `pyproject.toml` | `httpx`, `pydantic`; `llm` pytest marker |
| `config/llm.toml` | New |
| `src/argus/llm/__init__.py` | New: public exports |
| `src/argus/llm/client.py` | New: `LlmClient` |
| `src/argus/llm/types.py` | New: messages, usage, capabilities, enums |
| `src/argus/llm/config.py` | New: TOML loading, validation |
| `src/argus/llm/policy.py` | New |
| `src/argus/llm/sink.py` | New: JSONL call records |
| `src/argus/llm/errors.py` | New |
| `src/argus/llm/backends/{base,ollama,openai_compat}.py` | New |
| `scripts/measure_backends.py` | New: GPU budget |
| `docs/backends/*.md` | New: per-backend verification, GPU budget |
| `tests/llm/` | New |

## Out of Scope

- Remote zero-retention backend (after the local pipeline works; config + verification note only).
- Tool execution, multi-agent messaging, orchestration (D8, A1).
- Rendering windows to text (D5).
- Fine-tuning; time-series encoders (TimeOmni-1, OpenTSLM).
- Async / batched inference servers (vLLM) unless the GPU budget shows Ollama too slow.

## Known Residual Risks

- Reasoning on: more tokens per call, slower windows and less context left for the prompt; GPU budget measures with reasoning on.
- Temperature 0 on GPU is not bitwise deterministic; criterion 8 measures it.
- Q4 quantization degrades reasoning vs full precision; reported as a limitation, compared against the remote backend later.
- 24 GB caps context near ~32k for 27B models; long windows depend on D5 budgeting.
- Ollama Modelfiles carry default system prompts (`mistral-small3.2` presents itself as “Le Chat”): agents always send their own system message; the digest pins the Modelfile.
- Ollama-specific behaviours (silent truncation, model unload) handled in the adapter; a server upgrade can change them: contract suite re-run after upgrades.

## Related

- `docs/roadmap.md` D4, D5, D8, A1, A7; `literature_review.md` §8 points 11, 12.
