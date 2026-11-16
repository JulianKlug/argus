# Spec: Provider-independent LLM interface (D4)

## Context

Agents (A2–A6) need generation, structured output and tool calls from an LLM. Models change
during the study: A7 compares backends, V1 freezes one per agent. All model inference
therefore goes through one interface; swapping a model is a config change (roadmap, L11).

D4 owns: requests, capability checks, request budgeting, results, usage, call records.
D5 owns what clinical text enters a prompt; A1/D8 own when agents are called, tools, evidence.

Hosting: local models first. A verified zero-retention remote backend is added once the
pipeline works end to end on local models and must pass the same tests. MIMIC text leaves
the machine only through such a backend (PhysioNet LLM policy).

Hardware: 1× RTX 3090 (24 GB), 61 GB RAM. Ollama 0.34.1. Candidate local models,
all Q4_K_M, ~15–19 GB, one resident at a time. Probed 2026-11-16 (`/api/show` capabilities
+ live calls):

| Model | Tools | Reasoning | JSON schema | Use |
|---|---|---|---|---|
| `qwen3.8:27b` | ✓ | ✓ | ✓ | any agent |
| `mistral-small3.2:24b` | ✓ | — | ✓ | any agent, reasoning OFF |
| `medgemma:27b` (Gemma 3, medical fine-tune) | — | — | ✓ | agents without tools (summary, adjudication), reasoning OFF |

Ollama facts measured 2026-11-16 (they drive L3):
- Prompt over `num_ctx` is **silently truncated**: ~3.6k-token prompt at `num_ctx = 2048` →
  `prompt_eval_count = 1027`, no error, instruction at the start lost.
- No tokenize endpoint (`/api/tokenize` → 404). `num_predict = 0` ignored.
- `prompt_eval_count` is only known after the call; it cannot prove the input was complete.

## Design

```
 agents / harness (A1–A6)
   │  client = LlmClient.for_agent("expert")
   │  client.generate(request, ctx) / .structured(request, Schema, ctx)
   │  client.count_text_tokens(text) / .count_request_tokens(request)
   ▼
 client.py ──────► policy.py      fail closed: locality, digest, run dir, network settings
   │  budget, validate, retry, CallRecord ──► sink.py  (JSONL, $ARGUS_RUN_DIR, 0600)
   ▼
 backends/base.py   Backend protocol + probed Capabilities + Tokenizer
   ├── ollama.py             native /api/chat
   └── openai_compat.py      /v1/chat/completions (Ollama /v1 now, remote zero-retention later)
   ▲
 config.py ◄── config/llm.toml   backends (pinned digest) + per-agent settings and requirements
```

- Each layer calls only the one below. Agents see messages, schemas and usage, never URLs,
  provider SDKs or HTTP.
- `openai_compat` runs against Ollama's `/v1` now, against the remote endpoint later.
- No provider SDK; `httpx` only (`trust_env=False`, `follow_redirects=False`). Pydantic schemas.

### Decisions (locked)

| # | Decision |
|---|---|
| L1 | Backend = (adapter, base URL, model, pinned digest). Two local backends = two model families |
| L2 | Sync API. One GPU: Ollama serializes calls; parallel topologies (D8) queue |
| L3 | Budget before the call, verify after: request rejected unless `count_request_tokens + max_tokens ≤ context_tokens`; after the call `prompt_tokens` must equal the pre-count (± measured template tolerance), else `ContextOverflowError` (catches silent truncation). No exact or conservative count for a backend → backend refused (fail closed) |
| L4 | Structured output: backend-native schema constraint, then pydantic validation. Invalid → one retry with the invalid answer and the validation errors appended as corrective feedback; `finish = LENGTH` → no retry. Then `StructuredOutputError`. Every attempt recorded |
| L5 | D4 returns tool calls, never executes them. Returned calls validated against the offered tools (name, argument schema, strict: no type coercion, arguments returned as sent); invalid → same retry rule as L4, then `ToolCallError`. Nothing invalid reaches the harness |
| L6 | Capabilities are probed from the backend (`/api/show`, live probe) and intersected with config; config can restrict, never grant |
| L7 | Defaults: `temperature = 0`, fixed `seed`. Records hold digest, server version, settings |
| L8 | Call records hold patient text: only under `ARGUS_RUN_DIR`, outside the repo, owner-only |
| L9 | Reasoning text stored apart from the answer, never parsed as output. Reasoning token count only if the backend reports it, else null |
| L10 | Agent config declares required capabilities; a backend lacking one fails at `LlmClient` construction, never degrades silently |
| L11 | Exceptions and ordinary logging carry record ids, never message content |

### Public API (`src/argus/llm/__init__.py`)

```python
class Role(Enum):           SYSTEM, USER, ASSISTANT, TOOL
class DataPolicy(Enum):     LOCAL, ZERO_RETENTION
class FinishReason(Enum):   STOP, LENGTH, TOOL_CALLS
class Capability(Enum):     JSON_SCHEMA, TOOLS, REASONING, SEED

@dataclass(frozen=True)
class ToolCall:     id: str; name: str; arguments: dict      # id normalised by the adapter
@dataclass(frozen=True)
class Message:      role: Role; content: str
                    tool_calls: tuple[ToolCall, ...] = ()    # ASSISTANT requesting tools
                    tool_call_id: str | None = None          # TOOL result → which call
                    tool_name: str | None = None             # TOOL result (Ollama native needs name)
@dataclass(frozen=True)
class ToolSpec:     name: str; description: str; parameters: type[BaseModel]
@dataclass(frozen=True)
class Request:      messages: tuple[Message, ...]; tools: tuple[ToolSpec, ...] = ()
@dataclass(frozen=True)
class CallContext:  run_id: str; step_id: str; parent_record_id: str | None
                    labels: Mapping[str, str]                # e.g. arm, subject, window; opaque to D4
@dataclass(frozen=True)
class Usage:        prompt_tokens: int; completion_tokens: int; reasoning_tokens: int | None
                    latency_s: float; queue_s: float | None; load_s: float | None
                    gpu_s: float | None; cost: float | None; currency: str | None
@dataclass(frozen=True)
class Capabilities: context_tokens: int; supported: frozenset[Capability]
@dataclass(frozen=True)
class Completion:   message: Message                         # ASSISTANT, incl. tool_calls
                    reasoning: str | None; finish: FinishReason; usage: Usage; record_id: str

class LlmClient:
    @classmethod
    def for_agent(cls, agent: str, config: Path | None = None) -> LlmClient
    capabilities: Capabilities
    def generate(self, request: Request, ctx: CallContext) -> Completion
    def structured(self, request: Request, schema: type[T], ctx: CallContext) -> tuple[T, Completion]
    def count_text_tokens(self, text: str) -> int            # D5 per-domain measurement
    def count_request_tokens(self, request: Request, schema: type | None = None) -> int
                                                             # full input: template, history, tools, schema

errors: BackendUnavailableError, CapabilityError, ContextOverflowError, StructuredOutputError,
        ToolCallError, PolicyViolationError, BackendMismatchError
```

A completed assistant turn goes back into the next `Request` as `completion.message`; tool results
follow as `Role.TOOL` messages. Adapters map ids and names to each wire format and back.

### Tokenizer (first experiment of D4)

Needed for L3 before anything else. Candidates per backend, first one that passes wins:

| Candidate | Exactness check |
|---|---|
| Server tokenize endpoint (vLLM `/tokenize`, remote provider count API) | equals `prompt_tokens` on 50 fitting requests |
| Local tokenizer from the model's own GGUF blob (llama.cpp vocab-only) + rendered chat template incl. tools and schema | equals `prompt_tokens` on 50 fitting requests, or bounded by a measured constant |
| Conservative bound (chars × measured max ratio + margin) | never below `prompt_tokens` on 50 requests; documented as bound, wastes context |

None passes → backend refused. Result per backend in `docs/backends/<name>.md`.

### Config (`config/llm.toml`, committed; no secrets)

```toml
[backends.qwen27]
adapter = "ollama"
base_url = "http://127.0.0.1:11434"
model = "qwen3.8:27b"
digest = "22130167c4c2…"      # full digest; mismatch → BackendMismatchError
policy = "LOCAL"
context_tokens = 32768        # measured, see GPU budget
gpu_cost_per_s = 1.0

[backends.mistral24]
adapter = "openai_compat"
base_url = "http://127.0.0.1:11434/v1"
model = "mistral-small3.2:24b"
digest = "5a408ab55df5…"
policy = "LOCAL"
context_tokens = 32768

[backends.medgemma27]
adapter = "ollama"
base_url = "http://127.0.0.1:11434"
model = "medgemma:27b"
digest = "58238ae38f99…"
policy = "LOCAL"
context_tokens = 32768

[agents.expert]
backend = "qwen27"
requires = ["TOOLS", "JSON_SCHEMA", "REASONING"]
temperature = 0.0
seed = 7
max_tokens = 4096             # includes reasoning tokens
reasoning = "ON"

[agents.summary]
backend = "medgemma27"
requires = ["JSON_SCHEMA"]
reasoning = "OFF"
...
```

Remote backends later add `policy = "ZERO_RETENTION"`, `verification = "docs/backends/<name>.md"`,
`api_key_env = "..."` (key from env, never from the file).

### Policy rules (`policy.py`)

Checked at `LlmClient` construction and fail closed: `PolicyViolationError`, no request sent.

| Scope | Allowed iff |
|---|---|
| All | `httpx` client built with `trust_env=False` (no proxy from env), `follow_redirects=False`; any 3xx → error |
| All | Server-reported digest equals config `digest` (`/api/tags`); server version recorded |
| LOCAL | host is loopback; model has no remote host and no cloud tag (`/api/show`, `/api/tags`) |
| ZERO_RETENTION | verification note states provider, contract/setting for no retention, no training, no human review, logging/caching coverage, PhysioNet DUA reference, `verified_on` ≤ 180 days old |
| Run dir | `ARGUS_RUN_DIR` set, resolved (symlinks followed) path outside the repo, owned by the user, mode 0700; files created 0600 |

Every backend has `docs/backends/<name>.md`. LOCAL: Ollama bind address (`OLLAMA_HOST`),
no cloud models, weights present locally, telemetry status, tokenizer result.

Retention: run dirs are covered by the same PhysioNet controls as the store; deleted with the
store at project end; never copied into the repo or issue trackers.

### Call record (`sink.py`)

One JSONL line per attempt (retries included) in `$ARGUS_RUN_DIR/llm_calls.jsonl`:
`record_id, run_id, step_id, parent_record_id, labels, agent, backend, model, digest,
server_version, settings, request (messages, tools, schema), pre_count, raw_response,
reasoning, message, finish, usage, error, rejection, attempt, started_at`.
`rejection`: why a returned answer failed validation (tool arguments, schema). Accepted attempt:
`error` and `rejection` both null.

Records are sufficient to rebuild every conversation, tool exchange and setting without
re-running inference. Fresh re-runs (criterion 9) measure variability; they are not replay.

### GPU budget

Measured per local backend, written to `docs/backends/gpu_budget.md`:

| Measure | How |
|---|---|
| Max context fully on GPU | Largest `num_ctx` with 100% GPU (`ollama ps`), KV cache included |
| Prefill / decode tokens·s⁻¹ | At 4k, 16k and max context; reasoning on where supported |
| Model swap time | Unload A, first token from B |
| Sequence cost | Per candidate topology (D8): ordered agent→model calls per window incl. swaps and queueing |
| GPU-hours per study arm | windows (D6) × sequence cost |

`context_tokens` in config is the measured maximum, not the model's nominal one. If D5 windows
exceed it, D5 cuts the window, not D4. A7 reports load and queue time apart from inference,
so arms that swap models are not mistaken for slower models.

## Acceptance Criteria

1. All model inference goes through `argus.llm`; no module outside `argus/llm/backends/` imports `httpx` or names a provider URL path.
2. Contract suite, capability-conditional, on every configured backend: each supported capability works; each unsupported one raises `CapabilityError` with no request sent. Covers generation, nested structured output, reasoning on/off.
3. Tool round trip on ≥2 tool-capable backends: assistant turn with ≥2 tool calls → each result linked to its call → follow-up assistant answer using both. Unknown tool name or arguments failing the schema → retry, then `ToolCallError`; never returned to the caller.
4. Structured output: invalid first answer → retry carrying the validation errors → valid, or `StructuredOutputError`; `LENGTH` → no retry. All attempts recorded.
5. Request budgeting: full requests (long synthetic summary, multi-turn history, tool definitions, output schema) whose `count_request_tokens + max_tokens` exceeds `context_tokens` → `ContextOverflowError`, no request sent. Accepted requests: post-call `prompt_tokens` matches the pre-count within the backend's measured tolerance. A forced silent truncation (server `num_ctx` below config) → `ContextOverflowError`.
6. Tokenizer per backend passes one of the three checks in *Tokenizer*; result documented.
7. Policy refusals send zero requests (fake transport): non-loopback LOCAL, cloud model, digest mismatch, redirect, proxy env set, ZERO_RETENTION note missing or stale.
8. Run dir: unset, inside repo, symlink into repo, or group/world-accessible → refused; record files created 0600; record file that is a symlink or hard link → refused.
9. Same agent, request, seed, temperature 0: two fresh calls identical per local backend, or variability measured and reported.
10. Records alone rebuild every conversation (messages, tool calls and results, settings, context ids) byte-identical to what was sent; no inference.
11. Every call writes one record per attempt with digest, server version, settings, usage; reasoning text apart; unavailable metrics null.
12. Swapping an agent's backend changes only `config/llm.toml`; a swap to a backend lacking a `requires` capability fails at construction.
13. Error messages and log output in all failure tests contain no request content (sentinel string search).
14. GPU budget table filled for each local backend, including sequence cost.
15. Ollama down → `BackendUnavailableError` naming the backend, within the configured timeout.

## Testing Plan

| Layer | What | Marker |
|---|---|---|
| Unit | policy, run dir, config, budgeting, retries, tool validation, capability refusal, records, log redaction — fake `httpx` transport | — |
| Unit | each adapter maps messages, tool calls, ids, usage from recorded fixtures, both directions | — |
| Contract | criteria 2, 3, 5, 6, 9 parametrized over configured backends | `llm` (needs running Ollama) |
| Measurement | `scripts/measure_backends.py`: tokenizer check, GPU budget | manual |

Tests never send MIMIC data; contract prompts are synthetic.

## Implementation order

1. Tokenizer experiment (criterion 6) on the three local models: decides whether L3 is exact or a bound.
2. Types, config, policy, sink.
3. Ollama adapter, then `openai_compat`.
4. Contract suite, GPU budget.

## Rules for the implementing agent

Same as `spec_snapshot_engine.md` (rules 1–5). Additionally:
6. Policy checks live only in `policy.py`; provider specifics live only in `backends/`.
7. No provider SDK dependency. Add `httpx`, `pydantic` to `pyproject.toml`.

## Files Reference

| File | Change |
|---|---|
| `pyproject.toml` | `httpx`, `pydantic` (+ tokenizer dependency chosen in step 1); `llm` pytest marker |
| `config/llm.toml` | New |
| `src/argus/llm/__init__.py` | New: public exports |
| `src/argus/llm/client.py` | New: `LlmClient`, budgeting, retries, validation |
| `src/argus/llm/types.py` | New: messages, request, context, usage, capabilities, enums |
| `src/argus/llm/config.py` | New: TOML loading, validation |
| `src/argus/llm/policy.py` | New |
| `src/argus/llm/sink.py` | New: JSONL call records |
| `src/argus/llm/errors.py` | New |
| `src/argus/llm/backends/{base,ollama,openai_compat,tokenizer}.py` | New |
| `scripts/measure_backends.py` | New |
| `docs/backends/*.md` | New: per-backend verification, tokenizer, GPU budget |
| `tests/llm/` | New |

## Out of Scope

- Remote zero-retention backend (after the local pipeline works; config + verification note only).
  Open before any remote use: `openai_compat` needs a pre-count path without the Ollama control API;
  HTTPS required; verification fields must assert the protections, not only be present.
- Tool execution, multi-agent messaging, orchestration (D8, A1).
- Rendering windows to text (D5).
- Fine-tuning; time-series encoders (TimeOmni-1, OpenTSLM).
- Async / batched inference servers (vLLM) unless the GPU budget shows Ollama too slow.
- Network-level enforcement on the Ollama server (firewall): operational, documented in the backend note.

## Known Residual Risks

- Conservative token bound (if no exact tokenizer) wastes context.
- Reasoning on: more tokens per call, slower windows, less context for the prompt.
- Temperature 0 on GPU is not bitwise deterministic; criterion 9 measures it.
- Q4 quantization degrades reasoning vs full precision; compared against the remote backend later.
- 24 GB caps context near ~32k for 27B models; long windows depend on D5 budgeting.
- Ollama Modelfiles carry default system prompts (`mistral-small3.2` presents itself as “Le Chat”): agents always send their own system message; the digest pins the Modelfile.
- Ollama upgrade can change truncation, templates, capability reporting: contract suite and tokenizer check re-run after upgrades; server version in every record.
- Policy checks what the server reports; a compromised local server is out of scope.

## Related

- `docs/roadmap.md` D4, D5, D8, A1, A7; `literature_review.md` §8 points 11, 12.
- External review 2026-11-16 (critical: budgeting, tool round trip, governance).
