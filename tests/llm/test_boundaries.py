"""Criterion 1: only argus/llm/backends speaks HTTP or names provider URL paths."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "argus"
BACKENDS = SRC / "llm" / "backends"
HTTP_IMPORT = re.compile(r"^\s*(import|from)\s+(httpx|requests|urllib3|aiohttp|openai|anthropic|ollama)\b", re.M)
PROVIDER_PATH = re.compile(r"/api/(chat|generate|show|tags|version)|/chat/completions|/v1/")


def test_http_and_provider_paths_only_in_backends():
    """protects=criterion 1: all inference goes through argus.llm backends;
    fails_when=any module outside argus/llm/backends imports an HTTP client or provider SDK, or names a provider path;
    no_claim=dynamic imports or paths assembled at runtime."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.is_relative_to(BACKENDS):
            continue
        text = path.read_text()
        if HTTP_IMPORT.search(text) or PROVIDER_PATH.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == []
