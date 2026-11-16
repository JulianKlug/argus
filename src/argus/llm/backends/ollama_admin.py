"""Ollama server administration for measurements (GPU budget). Synthetic prompts only; never agent calls."""

import time
from dataclasses import dataclass

from argus.llm.backends.http import HttpSession
from argus.llm.backends.ollama import NS_PER_S, dumps
from argus.llm.config import BackendConfig

KEEP_ALIVE = "30m"


@dataclass(frozen=True)
class Residency:
    model: str
    size: int
    size_vram: int
    context_length: int

    @property
    def fully_on_gpu(self) -> bool:
        return self.size_vram == self.size


@dataclass(frozen=True)
class Timing:
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    load_s: float
    prefill_s: float
    decode_s: float


class OllamaAdmin:
    def __init__(self, config: BackendConfig, guard) -> None:
        url = config.control_url or config.base_url
        self._session = HttpSession(config.name, url, config.connect_timeout_s, config.request_timeout_s, guard)

    def resident(self) -> list[Residency]:
        return [
            Residency(m["name"], m["size"], m["size_vram"], m.get("context_length", 0))
            for m in self._session.get_json("/api/ps")["models"]
        ]

    def load(self, model: str, num_ctx: int) -> float:
        """Load `model` at `num_ctx`; returns wall seconds."""
        started = time.monotonic()
        self._session.post_json("/api/generate", dumps({"model": model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": num_ctx}}))
        return time.monotonic() - started

    def unload_all(self) -> None:
        for resident in self.resident():
            self._session.post_json("/api/generate", dumps({"model": resident.model, "keep_alive": 0}))

    def generate(self, model: str, prompt: str, num_ctx: int, num_predict: int, think: bool | None) -> Timing:
        body = {
            "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "truncate": False,
            "keep_alive": KEEP_ALIVE, "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0, "seed": 7},
        }
        if think is not None:
            body["think"] = think
        started = time.monotonic()
        data = self._session.post_json("/api/chat", dumps(body))
        return Timing(
            prompt_tokens=data["prompt_eval_count"],
            completion_tokens=data["eval_count"],
            wall_s=time.monotonic() - started,
            load_s=data["load_duration"] / NS_PER_S,
            prefill_s=data["prompt_eval_duration"] / NS_PER_S,
            decode_s=data["eval_duration"] / NS_PER_S,
        )
