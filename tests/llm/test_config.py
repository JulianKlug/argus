"""config/llm.toml loading: fail closed on malformed or unknown entries."""

import pytest

from argus.llm.config import DEFAULT_CONFIG, Adapter, load_agent
from argus.llm.types import Capability, DataPolicy, Reasoning
from llm_fakes import agent_toml, backend_toml


def test_committed_config_loads_every_agent():
    """protects=the committed config is valid and pins full digests;
    fails_when=a committed agent or backend entry is malformed;
    no_claim=the pinned digests match the server (policy at construction)."""
    text = DEFAULT_CONFIG.read_text()
    agents = [line[len("[agents."):-1] for line in text.splitlines() if line.startswith("[agents.")]
    assert agents
    for agent in agents:
        config = load_agent(agent)
        assert config.backend.policy is DataPolicy.LOCAL and len(config.backend.digest) == 64
        assert config.settings.temperature == 0.0 and config.settings.seed is not None


def test_agent_settings_and_requirements_parsed(write_config):
    """protects=agent entry → settings and required capabilities;
    fails_when=a field is dropped or defaults differ from L7 (temperature 0, fixed seed);
    no_claim=none."""
    path = write_config(backend_toml(), agent_toml(requires=["TOOLS"], reasoning="ON"))
    config = load_agent("agent", path)
    assert config.requires == {Capability.TOOLS} and config.settings.reasoning is Reasoning.ON
    assert config.backend.adapter is Adapter.OLLAMA and config.backend.count_tolerance == 0

    path = write_config(backend_toml(), "[agents.agent]\nbackend = \"fake\"\nmax_tokens = 10\n")
    defaults = load_agent("agent", path).settings
    assert (defaults.temperature, defaults.seed, defaults.reasoning) == (0.0, 7, Reasoning.OFF)


@pytest.mark.parametrize("backend,agent", [
    (backend_toml(digest="22130167c4c2"), agent_toml()),                 # truncated digest
    (backend_toml(capabilites=["TOOLS"]), agent_toml()),                 # typo in a restriction
    (backend_toml(currency=None), agent_toml()),                         # cost without unit
    (backend_toml(), agent_toml(backend="missing")),
    (backend_toml(), agent_toml(max_tokens=0)),
    (backend_toml(), agent_toml(requires=["TELEPATHY"])),
])
def test_malformed_config_refused(write_config, backend, agent):
    """protects=config validation;
    fails_when=a short digest, unknown key, unpriced cost, dangling backend, bad max_tokens or unknown
    capability is accepted;
    no_claim=semantic sanity of numbers."""
    with pytest.raises(ValueError):
        load_agent("agent", write_config(backend, agent))
