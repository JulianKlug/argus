"""Policy refusals (criteria 7, 8): fail closed before any inference request."""

import os
import stat
from datetime import date, timedelta

import httpx
import pytest

from argus.llm import BackendMismatchError, LlmClient, PolicyViolationError
from argus.llm import policy
from argus.llm.config import REPO_ROOT, _backend
from argus.llm.sink import RECORD_FILE
from llm_fakes import OTHER_DIGEST, agent_toml, backend_toml

TODAY = date(2026, 11, 16)
STALE = TODAY - timedelta(days=181)


def _client(write_config, fake, backend: str | None = None):
    path = write_config(backend or backend_toml(), agent_toml())
    return LlmClient.for_agent("agent", path, transport=httpx.MockTransport(fake.handle))


def test_accepts_vetted_local_backend(write_config, fake):
    """protects=the fixtures below fail for their stated reason only;
    fails_when=a clean LOCAL loopback backend with pinned digest is refused;
    no_claim=real server behaviour (contract suite)."""
    client = _client(write_config, fake)
    assert client.capabilities.context_tokens == 4096


@pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "http://localhost:11434", "http://ollama.internal:11434"])
def test_local_policy_refuses_non_loopback_host_with_zero_requests(write_config, fake, url):
    """protects=criterion 7 non-loopback LOCAL;
    fails_when=a LOCAL backend on a routable IP or a DNS name (could resolve anywhere) is used, or any request is sent first;
    no_claim=firewalling of the Ollama host itself (out of scope)."""
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake, backend_toml(base_url=url))
    assert fake.requests == []


def test_local_policy_refuses_non_loopback_control_url(write_config, fake):
    """protects=criterion 7 for the openai_compat control API;
    fails_when=metadata traffic may leave the machine while inference stays local;
    no_claim=remote backends."""
    backend = backend_toml(adapter="openai_compat", control_url="http://192.0.2.1:11434")
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake, backend)
    assert fake.requests == []


def test_local_policy_refuses_server_reported_remote_model_without_inference(write_config, fake):
    """protects=criterion 7 cloud model reported by the server (remote_host);
    fails_when=a LOCAL backend proxies to a remote host;
    no_claim=cloud models that report nothing (server trust is out of scope)."""
    fake.remote_host = "https://ollama.com:443"
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake)
    assert fake.inference_requests() == []


@pytest.mark.parametrize("model", ["fake:1b-cloud", "fake:1b:cloud", "fake:1b:CLOUD", "gpt-oss:120b-Cloud"])
def test_local_policy_refuses_cloud_model_name_with_zero_requests(write_config, fake, model):
    """protects=criterion 7 cloud model: Ollama 0.34.1 proxies any POST whose model ends in ':cloud' or
    '-cloud' (any case) to ollama.com before local lookup, metadata calls such as /api/show included;
    fails_when=any request is sent for a cloud-form model name;
    no_claim=cloud routing rules of other Ollama versions (contract suite re-run after upgrades)."""
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake, backend_toml(model=model))
    assert fake.requests == []


def test_digest_mismatch_refused_without_inference(write_config, fake):
    """protects=criterion 7 digest pin (L1);
    fails_when=a re-pulled or different model with the same name is used;
    no_claim=integrity of the blob on disk."""
    fake.digest = OTHER_DIGEST
    with pytest.raises(BackendMismatchError):
        _client(write_config, fake)
    assert fake.inference_requests() == []


def test_redirect_refused_and_not_followed(write_config, fake):
    """protects=criterion 7 redirect;
    fails_when=a 3xx is followed (request could reach an unvetted host) or accepted;
    no_claim=redirects issued by intermediaries outside httpx."""
    fake.redirect = True
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake)
    assert len(fake.requests) == 1
    assert all(r.url.host == "127.0.0.1" for r in fake.requests)


@pytest.mark.parametrize("var", ["HTTPS_PROXY", "http_proxy", "ALL_PROXY"])
def test_proxy_environment_refused_with_zero_requests(write_config, fake, monkeypatch, var):
    """protects=criterion 7 proxy env;
    fails_when=client starts while the environment routes traffic through a proxy;
    no_claim=system-level proxies not exposed as env variables."""
    monkeypatch.setenv(var, "http://proxy.example:3128")
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake)
    assert fake.requests == []


def test_transport_settings_checked():
    """protects=policy row 'httpx client built with trust_env=False, follow_redirects=False';
    fails_when=either setting is accepted when unsafe;
    no_claim=other httpx settings."""
    policy.check_transport(trust_env=False, follow_redirects=False)
    for unsafe in ((True, False), (False, True)):
        with pytest.raises(PolicyViolationError):
            policy.check_transport(*unsafe)


# ---------- ZERO_RETENTION verification note ----------

NOTE_FIELDS = {
    "provider": "Example AI",
    "no_retention": "Enterprise ZDR addendum §3",
    "no_training": "DPA §5",
    "no_human_review": "ZDR addendum §4",
    "logging_caching": "abuse logs disabled under ZDR; no prompt caching",
    "dua_reference": "PhysioNet credentialed DUA v1.5.0 §4",
}


def _note(tmp_path, verified_on: date, drop: str | None = None):
    fields = NOTE_FIELDS | {"verified_on": verified_on.isoformat()}
    fields.pop(drop, None)
    path = tmp_path / "remote.md"
    path.write_text("---\n" + "".join(f"{k}: {v}\n" for k, v in fields.items()) + "---\n# Remote backend\n")
    return path


def _remote_config(note):
    entry = {
        "adapter": "openai_compat", "base_url": "https://api.example.com/v1", "model": "m", "digest": "c" * 64,
        "policy": "ZERO_RETENTION", "context_tokens": 1000,
    }
    if note is not None:
        entry["verification"] = str(note)
    return _backend("remote", entry)


def test_zero_retention_note_complete_and_fresh_accepted(tmp_path):
    """protects=a valid note passes (so refusals below are specific);
    fails_when=a complete note dated today is refused;
    no_claim=truth of the note's statements."""
    policy.check_endpoint(_remote_config(_note(tmp_path, TODAY)), TODAY)


@pytest.mark.parametrize("case", ["missing", "stale", "future", "incomplete"])
def test_zero_retention_note_missing_stale_or_incomplete_refused(tmp_path, write_config, fake, case):
    """protects=criterion 7 ZERO_RETENTION note missing or stale;
    fails_when=a remote backend is used without a complete note verified within 180 days;
    no_claim=contract validity."""
    note = {
        "missing": None,
        "stale": _note(tmp_path, STALE),
        "future": _note(tmp_path, TODAY + timedelta(days=1)),
        "incomplete": _note(tmp_path, TODAY, drop="no_training"),
    }[case]
    with pytest.raises(PolicyViolationError):
        policy.check_endpoint(_remote_config(note), TODAY)


def test_zero_retention_refusal_sends_zero_requests(write_config, fake, tmp_path):
    """protects=criterion 7 end to end through LlmClient;
    fails_when=a stale-note backend sends any request;
    no_claim=remote adapter behaviour (not implemented)."""
    note = _note(tmp_path, STALE)
    backend = backend_toml(adapter="openai_compat", policy="ZERO_RETENTION", control_url=None,
                           base_url="https://api.example.com/v1", verification=str(note))
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake, backend)
    assert fake.requests == []


# ---------- run dir (criterion 8) ----------

def test_run_dir_unset_refused(write_config, fake, monkeypatch):
    """protects=criterion 8 unset;
    fails_when=records would be written to a default location;
    no_claim=none."""
    monkeypatch.delenv("ARGUS_RUN_DIR")
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake)
    assert fake.requests == []


def test_run_dir_inside_repo_or_symlink_into_repo_refused(tmp_path, monkeypatch):
    """protects=criterion 8 inside repo / symlink into repo;
    fails_when=patient text could land in the git working tree;
    no_claim=hard links."""
    monkeypatch.setenv("ARGUS_RUN_DIR", str(REPO_ROOT / "docs"))
    with pytest.raises(PolicyViolationError):
        policy.check_run_dir(os.environ)

    link = tmp_path / "link"
    link.symlink_to(REPO_ROOT / "docs")
    monkeypatch.setenv("ARGUS_RUN_DIR", str(link))
    with pytest.raises(PolicyViolationError):
        policy.check_run_dir(os.environ)


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o777, 0o500])
def test_run_dir_not_owner_only_refused(clean_env, mode):
    """protects=criterion 8 group/world-accessible;
    fails_when=a run dir other than exactly 0700 is accepted;
    no_claim=ACLs."""
    clean_env.chmod(mode)
    try:
        with pytest.raises(PolicyViolationError):
            policy.check_run_dir(os.environ)
    finally:
        clean_env.chmod(0o700)


def test_record_file_created_owner_only(write_config, fake, clean_env):
    """protects=criterion 8 record files created 0600 (whatever the umask);
    fails_when=llm_calls.jsonl is readable by group or others;
    no_claim=copies made by other tools."""
    old = os.umask(0)
    try:
        _client(write_config, fake)
    finally:
        os.umask(old)
    assert stat.S_IMODE((clean_env / RECORD_FILE).stat().st_mode) == 0o600


def test_existing_world_readable_record_file_refused(write_config, fake, clean_env):
    """protects=criterion 8 for an existing record file;
    fails_when=new records are appended to a file others can read;
    no_claim=none."""
    path = clean_env / RECORD_FILE
    path.write_text("")
    path.chmod(0o644)
    with pytest.raises(PolicyViolationError):
        _client(write_config, fake)
    assert fake.inference_requests() == []
