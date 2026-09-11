"""End-to-end: expand a small grid, run it, write it, analyse it."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from nudgesim.agents.bounded_rational import NormParams
from nudgesim.game.payoff import PayoffParams
from nudgesim.runner import (
    ABLATION_RATIOS,
    GridSpec,
    RunManifest,
    expand_grid,
    run_grid,
    write_run,
)


def test_expected_episode_counts_match_the_plan():
    counts = GridSpec().expected_episodes()
    assert counts["core"] == 600
    assert counts["placebo"] == 200
    assert counts["ablation_payoff"] + counts["ablation_ratio"] == 240
    assert counts["scale"] == 40
    assert counts["perturbation"] == 320
    assert counts["total"] == 1400


def test_grid_expansion_covers_every_cell(pool, library):
    spec = GridSpec(core_seeds=2, placebo_seeds=1, ablation_payoff_seeds=1,
                    ablation_ratio_seeds=1, scale_seeds=1)
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    core = [c for c in configs if c.arm == "core"]
    assert len({c.condition_id for c in core}) == 20
    assert all(c.claim.is_false for c in core)
    assert all(not c.claim.is_false for c in configs if c.arm == "placebo")
    assert all(not c.payoff_visible for c in configs if c.arm == "ablation_payoff")
    ratios = {
        round(c.payoff.engagement_accuracy_ratio, 4)
        for c in configs if c.arm == "ablation_ratio"
    }
    assert ratios == {round(r, 4) for r in ABLATION_RATIOS}


def test_adding_an_arm_does_not_perturb_existing_cells(pool, library):
    """Cell streams are seeded per cell, so arms are independent."""
    common = dict(core_seeds=2, placebo_seeds=1, ablation_payoff_seeds=1,
                  ablation_ratio_seeds=1, scale_seeds=1)
    only_core = expand_grid(GridSpec(**common, include_arms=("core",)), pool, library,
                            payoff=PayoffParams(), norms=NormParams())
    with_more = expand_grid(GridSpec(**common, include_arms=("core", "placebo")),
                            pool, library, payoff=PayoffParams(), norms=NormParams())
    core_of = lambda cs: [(c.episode_id, c.claim.claim_id, c.motif.motif_id, c.seed)
                          for c in cs if c.arm == "core"]
    assert core_of(only_core) == core_of(with_more)


def test_scale_arm_uses_the_fifty_agent_society(pool, library):
    spec = GridSpec(core_seeds=1, placebo_seeds=1, ablation_payoff_seeds=1,
                    ablation_ratio_seeds=1, scale_seeds=1, include_arms=("scale",))
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    assert configs and all(c.society.size == 50 for c in configs)


def test_end_to_end_run_and_analysis(tmp_path, pool, library):
    spec = GridSpec(core_seeds=6, placebo_seeds=2, ablation_payoff_seeds=1,
                    ablation_ratio_seeds=1, scale_seeds=2)
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    rows, raw = asyncio.run(run_grid(configs, concurrency=8))
    assert len(rows) == len(configs)

    manifest = RunManifest(
        run_id="test", grid=spec, claim_provenance=pool.provenance,
        motif_provenance=library.provenance, payoff=PayoffParams(), norms=NormParams(),
        dgp_label="test", backbone_kind="surrogate", scale_backend="test",
    )
    paths = write_run(tmp_path, manifest, rows, raw)
    frame = pd.read_parquet(paths["results_parquet"])
    assert len(frame) == len(configs)
    assert frame["episode_id"].is_unique

    rounds = pd.read_parquet(paths["rounds_parquet"])
    assert set(rounds["episode_id"]) == set(frame["episode_id"])

    written = json.loads(open(paths["manifest"]).read())
    assert written["claim_provenance"] == pool.provenance
    assert written["n_episodes_written"] == len(configs)

    from analysis.preregistered import run_analysis

    report = run_analysis(tmp_path, out_dir=tmp_path / "analysis")
    assert report["analysis_set"]["n_core_episodes"] > 0
    for outcome in ("cumulative_fpr", "cumulative_epc"):
        assert report["primary"][outcome]["dunnett_vs_own_model_control"]
    assert "h3_required_n_per_cell_for_80pct_power" in report["headline"]
    assert (tmp_path / "analysis" / "cell_means.csv").exists()


def test_a_surrogate_run_is_never_labelled_an_llm_result(tmp_path, pool, library):
    spec = GridSpec(core_seeds=1, placebo_seeds=1, ablation_payoff_seeds=1,
                    ablation_ratio_seeds=1, scale_seeds=1, include_arms=("core",))
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    rows, raw = asyncio.run(run_grid(configs, concurrency=4))
    manifest = RunManifest(
        run_id="t", grid=spec, claim_provenance=pool.provenance,
        motif_provenance=library.provenance, payoff=PayoffParams(), norms=NormParams(),
        dgp_label="declared", backbone_kind="surrogate", scale_backend="engine",
    )
    written = json.loads(open(write_run(tmp_path, manifest, rows, raw)["manifest"]).read())
    assert written["backbone_kind"] == "surrogate"
    assert "SYNTHETIC" in written["claim_provenance"]
