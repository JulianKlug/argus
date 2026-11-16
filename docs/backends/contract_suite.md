# Live contract suite

`uv run pytest -m llm tests/llm/test_contract.py -rA` on 2026-11-16: Ollama 0.34.1, RTX 3090
(driver 580.95.05), digests as in `config/llm.toml`. Code: `f1da700`. Synthetic prompts only.

| Test | qwen27 | mistral24 | medgemma27 |
|---|---|---|---|
| `test_generation` | pass | pass | pass |
| `test_nested_structured_output_or_refusal` | pass | pass | pass |
| `test_reasoning_on_off_or_refusal` | pass | pass (refusal) | pass (refusal) |
| `test_tools_or_refusal` | pass | pass | pass (refusal) |
| `test_tool_round_trip` | pass | pass | skipped: no TOOLS |
| `test_budget_refuses_over_context_and_accepts_exact_count` | pass | pass | pass |
| `test_forced_truncation_detected` | pass | pass | pass |
| `test_pre_count_matches_server_count` | pass | pass | pass |
| `test_two_fresh_calls_identical` (warm cache) | pass | pass | pass |

`test_enough_tool_backends` (≥ 2 backends with TOOLS): pass. Total: 27 passed, 1 skipped, 199 s.

No CI runs this suite: it needs the local GPU and pinned models. Re-run after any Ollama upgrade,
model change or client change, and update this file.
