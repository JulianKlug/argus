"""GGUF metadata reader and tokenizer construction (exactness on real models: contract suite + docs)."""

import pytest

from argus.llm.backends.tokenizer import load_gguf_tokenizer, read_tokenizer_metadata
from argus.llm.errors import CapabilityError
from llm_fakes import write_gguf

SPM_TOKENS = ["<unk>", "<s>", "</s>", "<start_of_turn>", "▁", "a", "b", "▁a"]
SPM_TYPES = [2, 3, 3, 3, 1, 1, 1, 1]


def test_reads_tokenizer_keys_only(byte_gguf):
    """protects=GGUF header parsing (strings, bools, string and int arrays);
    fails_when=values are misread or non-tokenizer keys leak in;
    no_claim=GGUF versions other than 3."""
    meta = read_tokenizer_metadata(byte_gguf)
    assert meta["tokenizer.ggml.model"] == "gpt2" and meta["tokenizer.ggml.add_bos_token"] is False
    assert len(meta["tokenizer.ggml.tokens"]) == 256 and meta["tokenizer.ggml.token_type"] == [1] * 256
    assert "general.architecture" not in meta


def test_byte_level_counts_utf8_bytes(byte_gguf):
    """protects=gpt2 path: regex split + byte-level mapping;
    fails_when=a byte is lost or merged without a merge rule;
    no_claim=real merge tables."""
    tokenizer = load_gguf_tokenizer(byte_gguf)
    for text in ["", "a", "héllo wörld", "日本 😀\n\t x"]:
        assert tokenizer.count(text) == len(text.encode())
    assert tokenizer.count_prompt("ab") == 2   # add_bos false


def test_spm_prefixes_each_fragment_and_counts_bos(tmp_path):
    """protects=Ollama SentencePiece behaviour (measured on medgemma): '▁' before every fragment between
    special tokens, spaces as '▁', specials matched literally, BOS added;
    fails_when=the prefix is applied once per text (HF behaviour) or specials are split;
    no_claim=score-based merges (GGUF merges are used)."""
    path = write_gguf(tmp_path / "spm.gguf", {
        "tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": SPM_TOKENS, "tokenizer.ggml.token_type": SPM_TYPES,
        "tokenizer.ggml.merges": ["\u2581 a"], "tokenizer.ggml.unknown_token_id": 0, "tokenizer.ggml.add_bos_token": True,
    })
    tokenizer = load_gguf_tokenizer(path)
    assert tokenizer.count("a") == 1                     # ▁a
    assert tokenizer.count("ab") == 2                    # ▁a b
    assert tokenizer.count("a a") == 2                   # ▁a ▁a
    assert tokenizer.count("<start_of_turn>a") == 2      # special, ▁a (prefix after the special too)
    assert tokenizer.count("b<start_of_turn>b") == 5     # ▁ b special ▁ b
    assert tokenizer.count_prompt("a") == 2              # BOS


@pytest.mark.parametrize("meta", [
    {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "unknown-pre"},
    {"tokenizer.ggml.model": "bert"},
])
def test_unknown_vocabulary_refused(tmp_path, meta):
    """protects=L3 fail closed: no exact tokenizer -> CapabilityError (backend refused);
    fails_when=an unrecognised pre-tokenizer or vocab model gets a guessed tokenizer;
    no_claim=none."""
    path = write_gguf(tmp_path / "x.gguf", meta | {"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.token_type": [1], "tokenizer.ggml.merges": []})
    with pytest.raises(CapabilityError):
        load_gguf_tokenizer(path)


def test_non_gguf_refused(tmp_path):
    """protects=blob sanity;
    fails_when=a non-GGUF file is parsed as a vocabulary;
    no_claim=none."""
    path = tmp_path / "x.bin"
    path.write_bytes(b"NOTGGUF" + b"\0" * 32)
    with pytest.raises(CapabilityError):
        read_tokenizer_metadata(path)
