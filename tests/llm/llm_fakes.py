"""Fake Ollama server (httpx.MockTransport) and a byte-level GGUF vocabulary.

    LlmClient ── HttpSession ── MockTransport ── FakeOllama.handle
                                                  ├─ /api/version, /api/tags, /api/show
                                                  ├─ /api/chat            (native; render or answer)
                                                  └─ /v1/chat/completions (OpenAI-compatible)

The tiny vocabulary has one token per byte and no merges, so the exact prompt count is the
UTF-8 length of the fake's rendered prompt. Answers come from a queue (`replies`).
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from argus.llm import client as client_module

MODEL = "fake:1b"
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
VERSION = "0.34.1"
NOMINAL_CONTEXT = 131072
BASE_URL = "http://127.0.0.1:11434"
SENTINEL = "SENTINEL-PATIENT-TEXT-7f3c"

GGUF_UINT32, GGUF_STRING, GGUF_ARRAY, GGUF_INT32, GGUF_BOOL = 4, 8, 9, 5, 7


# ---------- tiny GGUF ----------

def byte_level_alphabet() -> list[str]:
    """GPT-2 bytes_to_unicode: printable stand-in for each byte."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return [chars[b] for b in range(256)]


def _gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, metadata: dict) -> Path:
    """Minimal GGUF v3 with metadata only (bool, uint32, str, list[str], list[int32])."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value in metadata.items():
        out += _gguf_string(key)
        if isinstance(value, bool):
            out += struct.pack("<I?", GGUF_BOOL, value)
        elif isinstance(value, int):
            out += struct.pack("<II", GGUF_UINT32, value)
        elif isinstance(value, str):
            out += struct.pack("<I", GGUF_STRING) + _gguf_string(value)
        elif all(isinstance(v, str) for v in value):
            out += struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(value)) + b"".join(_gguf_string(v) for v in value)
        else:
            out += struct.pack("<IIQ", GGUF_ARRAY, GGUF_INT32, len(value)) + struct.pack(f"<{len(value)}i", *value)
    path.write_bytes(bytes(out))
    return path


def write_byte_gguf(path: Path) -> Path:
    tokens = byte_level_alphabet()
    return write_gguf(path, {
        "general.architecture": "fake",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "llama-bpe",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [1] * len(tokens),
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.add_bos_token": False,
    })


# ---------- fake server ----------

def ollama_message(content: str = "", thinking: str | None = None, tool_calls: list | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "function": {"index": i, "name": name, "arguments": args}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return message


@dataclass
class Reply:
    content: str = ""
    thinking: str | None = None
    tool_calls: list = field(default_factory=list)      # [(name, arguments)]
    done_reason: str = "stop"


@dataclass
class FakeOllama:
    gguf: Path
    digest: str = DIGEST
    capabilities: tuple = ("completion", "tools", "thinking")
    remote_host: str | None = None
    tools_parse: bool = True            # False: server drops tool calls (mistral3 template bug)
    json_schema_ok: bool = True
    prompt_eval_delta: int = 0          # != 0 simulates silent truncation
    redirect: bool = False
    v1_num_ctx: int | None = None       # /v1 cannot set num_ctx; over it, Ollama drops old messages silently
    replies: list = field(default_factory=list)
    requests: list = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.redirect:
            return httpx.Response(307, headers={"location": "http://198.51.100.7/"})

        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": VERSION})
        if path == "/api/tags":
            entry = {"name": MODEL, "model": MODEL, "digest": self.digest}
            if self.remote_host:
                entry["remote_host"] = self.remote_host
            return httpx.Response(200, json={"models": [entry]})
        if path == "/api/show":
            return httpx.Response(200, json={
                "capabilities": list(self.capabilities),
                "model_info": {"general.architecture": "fake", "fake.context_length": NOMINAL_CONTEXT},
                "modelfile": f"# Modelfile\nFROM {self.gguf}\nTEMPLATE x\n",
            })

        body = json.loads(request.content)
        if path.endswith("/chat/completions"):
            body = _native_view(body)
            if self.v1_num_ctx is not None:
                body = _drop_until_fits(body, self.v1_num_ctx)
        rendered = render(body)
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": rendered}})

        num_ctx = body.get("options", {}).get("num_ctx")
        prompt_tokens = len(rendered.encode())
        if body.get("truncate") is False and num_ctx is not None and prompt_tokens > num_ctx:
            return httpx.Response(400, json={"error": f"exceed_context_size_error {SENTINEL}"})

        reply = self._reply(body)
        if path == "/api/chat":
            return httpx.Response(200, json=self._native(reply, prompt_tokens))
        return httpx.Response(200, json=self._openai(reply, prompt_tokens))

    def inference_requests(self) -> list[httpx.Request]:
        """Chat requests that would run the model (render-only excluded)."""
        return [
            r for r in self.requests
            if r.url.path.endswith(("/api/chat", "/chat/completions")) and not json.loads(r.content).get("_debug_render_only")
        ]

    def agent_requests(self) -> list[dict]:
        """Inference request bodies other than capability probes."""
        probes = (client_module.PROBE_TOOL_PROMPT, client_module.PROBE_SCHEMA_PROMPT)
        bodies = [json.loads(r.content) for r in self.inference_requests()]
        return [b for b in bodies if b["messages"][0]["content"] not in probes]

    def _reply(self, body: dict) -> Reply:
        first = body["messages"][0]["content"]
        if first == client_module.PROBE_TOOL_PROMPT:
            return Reply(tool_calls=[("get_lab", {"name": "lactate"})] if self.tools_parse else [])
        if first == client_module.PROBE_SCHEMA_PROMPT:
            return Reply(content='{"ok": true}' if self.json_schema_ok else "no")
        return self.replies.pop(0)

    def _native(self, reply: Reply, prompt_tokens: int) -> dict:
        return {
            "model": MODEL, "done": True, "done_reason": reply.done_reason,
            "message": ollama_message(reply.content, reply.thinking, reply.tool_calls),
            "prompt_eval_count": prompt_tokens + self.prompt_eval_delta, "eval_count": 5,
            "load_duration": 1_000_000, "prompt_eval_duration": 2_000_000, "eval_duration": 3_000_000,
        }

    def _openai(self, reply: Reply, prompt_tokens: int) -> dict:
        message = {"role": "assistant", "content": reply.content}
        if reply.thinking:
            message["reasoning"] = reply.thinking
        if reply.tool_calls:
            message["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(reply.tool_calls)
            ]
        finish = "tool_calls" if reply.tool_calls else reply.done_reason
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens + self.prompt_eval_delta, "completion_tokens": 5},
        }


def _native_view(body: dict) -> dict:
    """What Ollama's /v1 middleware hands the template: the native chat request."""
    messages = []
    for m in body["messages"]:
        native = {"role": m["role"], "content": m["content"]}
        if m.get("tool_calls"):
            native["tool_calls"] = [
                {"id": c["id"], "function": {"index": i, "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
                for i, c in enumerate(m["tool_calls"])
            ]
        if m["role"] == "tool":
            native["tool_name"] = m.get("name")
            native["tool_call_id"] = m.get("tool_call_id")
        messages.append(native)
    out = {"messages": messages, "tools": body.get("tools", []), "_debug_render_only": body.get("_debug_render_only")}
    if "reasoning_effort" in body:
        out["think"] = body["reasoning_effort"] != "none"
    return out


def _drop_until_fits(body: dict, num_ctx: int) -> dict:
    """Ollama chatPrompt with truncate on: drop oldest non-system messages, keep the last one."""
    messages = body["messages"]
    while len(messages) > 1 and len(render(body | {"messages": messages}).encode()) > num_ctx:
        keep = [m for m in messages[:-1] if m["role"] == "system"]
        dropped = next(i for i, m in enumerate(messages) if m["role"] != "system")
        messages = keep + [m for i, m in enumerate(messages) if i > dropped and m["role"] != "system"]
    return body | {"messages": messages}


def render(body: dict) -> str:
    """Deterministic stand-in for the server template: every field that reaches the prompt."""
    parts = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in body["messages"]]
    parts += [json.dumps(t, sort_keys=True, ensure_ascii=False) for t in body.get("tools", [])]
    parts.append(f"think={body.get('think')}")
    return "\n".join(parts)


# ---------- config + environment ----------

def backend_toml(name: str = "fake", adapter: str = "ollama", **overrides) -> str:
    entry = {
        "adapter": adapter,
        "base_url": BASE_URL if adapter == "ollama" else f"{BASE_URL}/v1",
        "model": MODEL,
        "digest": DIGEST,
        "policy": "LOCAL",
        "context_tokens": 4096,
        "gpu_cost_per_s": 2.0,
        "currency": "GPU_S",
    }
    if adapter == "openai_compat":
        entry["control_url"] = BASE_URL
    entry |= overrides
    return f"[backends.{name}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in entry.items() if v is not None)


def agent_toml(name: str = "agent", backend: str = "fake", **overrides) -> str:
    entry = {"backend": backend, "requires": [], "temperature": 0.0, "seed": 7, "max_tokens": 256, "reasoning": "OFF"} | overrides
    return f"[agents.{name}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in entry.items())
