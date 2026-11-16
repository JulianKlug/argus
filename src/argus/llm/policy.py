"""Data governance checks (PhysioNet DUA). Every check fails closed with PolicyViolationError.

Order at LlmClient construction:

    environment, endpoint, run dir    static: zero requests sent
    server digest, locality           metadata queries only: zero inference requests
"""

import errno
import ipaddress
import os
import stat
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from argus.llm.config import REPO_ROOT, BackendConfig
from argus.llm.errors import BackendMismatchError, PolicyViolationError
from argus.llm.types import DataPolicy

RUN_DIR_ENV = "ARGUS_RUN_DIR"
RUN_DIR_MODE = 0o700
RECORD_FILE_MODE = 0o600
PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy")
VERIFICATION_MAX_AGE = timedelta(days=180)
HTTP_REDIRECT = range(300, 400)
CLOUD_SOURCE = "cloud"

# Fields a ZERO_RETENTION verification note must state (front matter between '---' lines).
VERIFICATION_FIELDS = (
    "provider", "no_retention", "no_training", "no_human_review", "logging_caching",
    "dua_reference", "verified_on",
)


def check_environment(env: Mapping[str, str]) -> None:
    """Proxy variables are ignored by the client (trust_env=False), but their presence means the
    machine routes traffic elsewhere: refuse rather than guess."""
    present = [k for k in env if k.lower() in PROXY_ENV and env[k]]
    if present:
        raise PolicyViolationError(f"proxy environment set: {sorted(present)}")


def check_transport(trust_env: bool, follow_redirects: bool) -> None:
    if trust_env or follow_redirects:
        raise PolicyViolationError("HTTP client must ignore env proxies and never follow redirects")


def check_response_status(status: int, backend: str) -> None:
    """Any redirect is refused: the request could land on an unvetted host."""
    if status in HTTP_REDIRECT:
        raise PolicyViolationError(f"backend {backend}: redirect refused (HTTP {status})")


def check_endpoint(backend: BackendConfig, today: date) -> None:
    """LOCAL: every URL is a loopback IP literal. ZERO_RETENTION: verification note is complete and fresh."""
    if backend.policy is DataPolicy.LOCAL:
        for url in filter(None, (backend.base_url, backend.control_url)):
            if not _is_loopback(url):
                raise PolicyViolationError(f"backend {backend.name}: LOCAL policy needs a loopback IP host")
        if _is_cloud_ref(backend.model):
            raise PolicyViolationError(f"backend {backend.name}: LOCAL policy forbids cloud models")
        return

    _check_verification_note(backend, today)


def check_server(backend: BackendConfig, reported_digest: str | None, remote_host: str | None) -> None:
    """Server must serve exactly the pinned weights, and a LOCAL model must not proxy to a remote host."""
    if reported_digest != backend.digest:
        raise BackendMismatchError(f"backend {backend.name}: server digest differs from pinned digest")
    if backend.policy is DataPolicy.LOCAL and remote_host:
        raise PolicyViolationError(f"backend {backend.name}: LOCAL policy forbids cloud models")


def check_run_dir(env: Mapping[str, str], repo_root: Path = REPO_ROOT) -> Path:
    """Call records hold patient text: owner-only directory outside the repo (L8)."""
    raw = env.get(RUN_DIR_ENV)
    if not raw:
        raise PolicyViolationError(f"{RUN_DIR_ENV} not set")

    run_dir = Path(raw).resolve()
    if not run_dir.is_dir():
        raise PolicyViolationError(f"{RUN_DIR_ENV} is not a directory")
    if run_dir == repo_root.resolve() or run_dir.is_relative_to(repo_root.resolve()):
        raise PolicyViolationError(f"{RUN_DIR_ENV} resolves inside the repository")

    info = run_dir.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != RUN_DIR_MODE:
        raise PolicyViolationError(f"{RUN_DIR_ENV} must be owned by the user with mode 0700")
    return run_dir


def open_record_file(path: Path) -> int:
    """Append-only descriptor to an owner-only 0600 regular file with a single name (L8).
    A symlink or hard link could lead into the repo: refused. Checks run on the opened descriptor,
    and all writes use it, so the vetted file is the written file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(path, flags, RECORD_FILE_MODE)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise PolicyViolationError("call record file must not be a symlink") from None
        raise

    info = os.fstat(fd)
    vetted = (
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == RECORD_FILE_MODE
    )
    if not vetted:
        os.close(fd)
        raise PolicyViolationError("call record file must be a single-link regular file, user-owned, mode 0600")
    return fd


def _is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # hostname, not an IP literal: DNS could point anywhere


def _is_cloud_ref(model: str) -> bool:
    """Ollama 0.34.1 routes a POST to ollama.com when the suffix after the last ':' is 'cloud' or ends
    in '-cloud' (case-insensitive), before any local lookup. Example: 'gpt-oss:120b-cloud'."""
    _, sep, suffix = model.rpartition(":")
    suffix = suffix.strip().lower()
    return bool(sep) and (suffix == CLOUD_SOURCE or ("/" not in suffix and suffix.endswith("-" + CLOUD_SOURCE)))


def _check_verification_note(backend: BackendConfig, today: date) -> None:
    note = backend.verification
    if note is None or not note.is_file():
        raise PolicyViolationError(f"backend {backend.name}: zero-retention verification note missing")

    fields = _front_matter(note.read_text())
    missing = [f for f in VERIFICATION_FIELDS if not fields.get(f)]
    if missing:
        raise PolicyViolationError(f"backend {backend.name}: verification note lacks {missing}")

    try:
        verified_on = date.fromisoformat(fields["verified_on"])
    except ValueError:
        raise PolicyViolationError(f"backend {backend.name}: verified_on is not an ISO date") from None
    if not today - VERIFICATION_MAX_AGE <= verified_on <= today:
        raise PolicyViolationError(f"backend {backend.name}: verification older than 180 days")


def _front_matter(text: str) -> dict[str, str]:
    """`key: value` lines between the first two '---' lines."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return {}
