"""Integration test for the LLM code path, against a local stub endpoint.

Every shipped result so far came from the analytic surrogate, so prompt
assembly, HTTP transport, concurrency, JSON parsing, repair and cost accounting
were never exercised against an actual endpoint. These tests run the real
`openai` client against a local OpenAI-compatible server, so the only thing
stubbed is the model.

The stub deliberately misbehaves the way real models do -- fenced JSON, prose
instead of JSON, lowercase action words -- because a pipeline that cannot
survive that will not survive a real provider.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("openai")

SERVER = Path(__file__).resolve().parents[1] / "scripts" / "mock_llm_server.py"
PORT = 8091
BASE_URL = f"http://127.0.0.1:{PORT}/v1"


@pytest.fixture(scope="module")
def stub_server():
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--port", str(PORT), "--malformed-rate", "0.3"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for _ in range(80):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=0.5).read()
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.skip("stub server did not start")
    yield f"http://127.0.0.1:{PORT}"
    proc.terminate()
    proc.wait(timeout=10)


def _spec(model: str = "stub-model", cache: bool = False) -> dict:
    return {"provider": "openai", "model": model, "base_url": BASE_URL,
            "cache": cache, "retry": False}


def _config(pool, library, **kw):
    from nudgesim.env.episode import EpisodeConfig
    from nudgesim.intervention.scheduler import Timing
    from nudgesim.intervention.tone import Tone

    return EpisodeConfig(
        episode_id=kw.pop("episode_id", "llm"),
        claim=kw.pop("claim", pool.false_claims[0]),
        motif=kw.pop("motif", library.motifs[0]),
        timing=kw.pop("timing", Timing.EARLY),
        tone=kw.pop("tone", Tone.AGGRESSIVE),
        seed=kw.pop("seed", 1),
        backend_spec=kw.pop("backend_spec", _spec()),
        **kw,
    )


def test_citizens_run_on_a_language_model(stub_server, pool, library, monkeypatch):
    from nudgesim.env.episode import run_episode

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    result = asyncio.run(run_episode(_config(pool, library)))

    assert result.ledger.rounds_settled == 12
    citizen_backbones = {
        r.backbone for r in result.records if r.role.startswith("citizen")
    }
    assert citizen_backbones == {"stub-model"}
    assert all(
        r.meta.get("backbone_kind") == "llm"
        for r in result.records if r.role.startswith("citizen")
    )


def test_malformed_replies_are_repaired_and_counted(stub_server, pool, library, monkeypatch):
    """A 30% malformed rate must not lose a single turn."""
    from nudgesim.env.episode import run_episode

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    result = asyncio.run(run_episode(_config(pool, library, seed=7)))

    for agent in ("A1", "A2", "B1", "B2", "C"):
        rounds = {r.round_index for r in result.records if r.agent_id == agent}
        assert rounds == set(range(12)), f"{agent} lost a turn"
    assert 0.0 < result.repair_rate < 1.0, "repair rate not being tracked"


def test_response_cache_collapses_identical_prompts(stub_server, pool, library, monkeypatch, tmp_path):
    """Caching is the main cost lever; a miss on every call would be silent."""
    from nudgesim.backends.base import build_backend, LLMRequest

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    backend = build_backend({**_spec(cache=True), "cache_dir": str(tmp_path / "c")})
    request = LLMRequest(system="s", user="u", temperature=0.0)

    first = asyncio.run(backend.complete(request))
    second = asyncio.run(backend.complete(request))
    assert first.text == second.text
    assert not first.cached and second.cached
    assert backend.stats() == {"hits": 1, "misses": 1}


def test_usage_is_recorded_for_cost_accounting(stub_server, monkeypatch):
    from nudgesim.backends.base import build_backend, LLMRequest

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    backend = build_backend(_spec())
    response = asyncio.run(backend.complete(LLMRequest(system="sys", user="a user turn")))
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0


def test_a_run_on_models_is_labelled_llm_not_surrogate(stub_server, pool, library, monkeypatch, tmp_path):
    """Provenance must distinguish a model run from the analytic surrogate."""
    from nudgesim.agents.bounded_rational import NormParams
    from nudgesim.game.payoff import PayoffParams
    from nudgesim.runner import GridSpec, RunManifest, expand_grid, run_grid, write_run

    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    specs = {"stub-model": _spec()}
    spec = GridSpec(backbones=("stub-model",), backend_specs=specs, core_seeds=1,
                    placebo_seeds=1, ablation_payoff_seeds=1, ablation_ratio_seeds=1,
                    scale_seeds=1, include_arms=("core",))
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    assert all(c.backend_spec for c in configs)

    rows, raw = asyncio.run(run_grid(configs, concurrency=4))
    manifest = RunManifest(
        run_id="t", grid=spec, claim_provenance=pool.provenance,
        motif_provenance=library.provenance, payoff=PayoffParams(), norms=NormParams(),
        dgp_label="n/a", backbone_kind="llm", scale_backend="n/a",
    )
    written = json.loads(Path(write_run(tmp_path, manifest, rows, raw)["manifest"]).read_text())
    assert written["backbone_kind"] == "llm"
    assert {r["backbone_kind"] for r in rows} == {"llm"}
