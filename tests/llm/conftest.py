"""Fixtures for LLM tests; fakes live in llm_fakes.py."""

from pathlib import Path

import pytest

from argus.llm import client as client_module
from llm_fakes import FakeOllama, write_byte_gguf


@pytest.fixture
def byte_gguf(tmp_path) -> Path:
    return write_byte_gguf(tmp_path / "byte.gguf")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Owner-only run dir outside the repo, no proxies, fresh probe cache."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    run_dir.chmod(0o700)
    monkeypatch.setenv("ARGUS_RUN_DIR", str(run_dir))
    monkeypatch.setattr(client_module, "_probe_cache", {})
    return run_dir


@pytest.fixture
def fake(byte_gguf) -> FakeOllama:
    return FakeOllama(byte_gguf)


@pytest.fixture
def write_config(tmp_path):
    def write(*sections: str) -> Path:
        path = tmp_path / "llm.toml"
        path.write_text("\n".join(sections))
        return path
    return write
