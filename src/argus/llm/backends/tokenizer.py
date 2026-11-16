"""Exact token counts from the model's own GGUF vocabulary (no inference, no server tokenize).

Checked against Ollama 0.34.1 `prompt_eval_count`: identical on 50/50 chat requests per local
model (docs/backends/*.md). Ollama's tokenizers differ from Hugging Face in one place:
SentencePiece models get a '▁' prefix on every text fragment between special tokens.

    GGUF metadata ──► vocab, merges, token types ──► tokenizers.BPE
                                                     ├─ gpt2:  regex split + byte-level
                                                     └─ llama: '▁' prefix + spaces→'▁', byte fallback
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from argus.llm.errors import CapabilityError

GGUF_MAGIC = b"GGUF"
SPM_SPACE = "▁"

# Pre-tokenizer regex per `tokenizer.ggml.pre`, as in llama.cpp / Ollama.
PRE_TOKENIZER_REGEX = {
    "qwen35": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
        r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "llama-bpe": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}


class GgufType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR = {
    GgufType.UINT8: "<B", GgufType.INT8: "<b", GgufType.UINT16: "<H", GgufType.INT16: "<h",
    GgufType.UINT32: "<I", GgufType.INT32: "<i", GgufType.FLOAT32: "<f", GgufType.BOOL: "<?",
    GgufType.UINT64: "<Q", GgufType.INT64: "<q", GgufType.FLOAT64: "<d",
}


class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


@dataclass(frozen=True)
class GgufTokenizer:
    tokenizer: Tokenizer
    add_bos: bool

    def count(self, text: str) -> int:
        """Tokens of raw text; special-token strings count as one token, as on the server."""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def count_prompt(self, rendered: str) -> int:
        """Tokens of a fully rendered prompt, including the BOS the server prepends."""
        return self.count(rendered) + (1 if self.add_bos else 0)


@lru_cache(maxsize=8)
def load_gguf_tokenizer(path: Path) -> GgufTokenizer:
    meta = read_tokenizer_metadata(path)
    model = meta.get("tokenizer.ggml.model")
    tokens = meta["tokenizer.ggml.tokens"]
    types = meta["tokenizer.ggml.token_type"]
    merges = [tuple(m.split(" ", 1)) for m in meta.get("tokenizer.ggml.merges", [])]
    vocab = {t: i for i, t in enumerate(tokens)}

    if model == "gpt2":
        regex = PRE_TOKENIZER_REGEX.get(meta.get("tokenizer.ggml.pre"))
        if regex is None:
            raise CapabilityError(f"no exact tokenizer for pre-tokenizer {meta.get('tokenizer.ggml.pre')!r}")
        tokenizer = Tokenizer(models.BPE(vocab, merges))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(regex), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = decoders.ByteLevel()
    elif model == "llama" and merges:
        unknown = tokens[meta["tokenizer.ggml.unknown_token_id"]]
        tokenizer = Tokenizer(models.BPE(vocab, merges, byte_fallback=True, unk_token=unknown, fuse_unk=True))
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Prepend(SPM_SPACE), normalizers.Replace(" ", SPM_SPACE),
        ])
    else:
        raise CapabilityError(f"no exact tokenizer for vocabulary model {model!r}")

    # Control and user-defined tokens are matched literally in rendered prompts.
    special = (TokenType.CONTROL, TokenType.USER_DEFINED)
    tokenizer.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t, ty in zip(tokens, types) if ty in special
    ])
    return GgufTokenizer(tokenizer, add_bos=bool(meta.get("tokenizer.ggml.add_bos_token", False)))


def read_tokenizer_metadata(path: Path) -> dict:
    """`tokenizer.*` key/values from a GGUF header; tensor data is never read."""
    with path.open("rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise CapabilityError("model blob is not GGUF")
        _version, _tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))

        meta = {}
        for _ in range(kv_count):
            key = _read_string(f)
            value = _read_value(f, GgufType(_unpack(f, "<I")))
            if key.startswith("tokenizer.") and key != "tokenizer.chat_template":
                meta[key] = value
        return meta


def _read_value(f: BinaryIO, kind: GgufType):
    if kind is GgufType.STRING:
        return _read_string(f)
    if kind is not GgufType.ARRAY:
        return _unpack(f, _SCALAR[kind])

    element = GgufType(_unpack(f, "<I"))
    count = _unpack(f, "<Q")
    if element is GgufType.STRING:
        return [_read_string(f) for _ in range(count)]

    fmt = _SCALAR[element]
    size = struct.calcsize(fmt)
    return list(struct.unpack(f"<{count}{fmt[1]}", f.read(size * count)))


def _read_string(f: BinaryIO) -> str:
    return f.read(_unpack(f, "<Q")).decode("utf-8", errors="replace")


def _unpack(f: BinaryIO, fmt: str):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
