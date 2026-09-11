"""Episode-loop, metric and calibration integration tests."""

from __future__ import annotations

import asyncio

import pytest

from nudgesim.agents.persona import CITIZEN_ROLES, DEFAULT_SOCIETY, scaled_society_spec
from nudgesim.calibration import ToleranceBand, calibrate, diffusion_tree
from nudgesim.env.episode import EpisodeConfig, run_episode
from nudgesim.env.guards import Guards, ngram_overlap, stance_drift
from nudgesim.game.actions import Action
from nudgesim.game.payoff import PayoffParams
from nudgesim.intervention.scheduler import Timing
from nudgesim.intervention.tone import Tone
from nudgesim.metrics.outcomes import compute_metrics, per_round_rates, pollution_half_life


def _config(pool, library, **kw) -> EpisodeConfig:
    return EpisodeConfig(
        episode_id=kw.pop("episode_id", "t"),
        claim=kw.pop("claim", pool.false_claims[0]),
        motif=kw.pop("motif", library.motifs[0]),
        timing=kw.pop("timing", Timing.EARLY),
        tone=kw.pop("tone", Tone.AGGRESSIVE),
        seed=kw.pop("seed", 1),
        **kw,
    )


def _run(config: EpisodeConfig):
    return asyncio.run(run_episode(config))


def test_episode_runs_the_full_horizon(pool, library):
    result = _run(_config(pool, library))
    assert result.ledger.rounds_settled == 12
    assert not result.guards.terminated_early


def test_every_citizen_acts_every_round(pool, library):
    result = _run(_config(pool, library))
    for agent in CITIZEN_ROLES:
        rounds = {r.round_index for r in result.records if r.agent_id == agent}
        assert rounds == set(range(12))


def test_disseminator_only_ever_shares(pool, library):
    result = _run(_config(pool, library))
    actions = {r.action for r in result.records if r.agent_id == "Disseminator"}
    assert actions == {Action.SHARE}


def test_intervener_is_absent_in_control_and_always_challenges_when_present(pool, library):
    control = _run(_config(pool, library, timing=Timing.NONE, tone=None))
    assert not any(r.agent_id == "DevilsAdvocate" for r in control.records)

    treated = _run(_config(pool, library))
    intervener = [r for r in treated.records if r.agent_id == "DevilsAdvocate"]
    assert intervener and {r.action for r in intervener} == {Action.CHALLENGE}


def test_intervener_enters_later_in_the_late_arm(pool, library):
    early = _run(_config(pool, library, timing=Timing.EARLY))
    late = _run(_config(pool, library, timing=Timing.LATE))
    assert early.schedule.realised_entry_round < late.schedule.realised_entry_round


def test_ledger_scores_only_citizens(pool, library):
    result = _run(_config(pool, library))
    assert set(result.ledger.agents) == set(CITIZEN_ROLES)


def test_episode_is_reproducible_from_its_seed(pool, library):
    a = _run(_config(pool, library, seed=42))
    b = _run(_config(pool, library, seed=42))
    assert [r.action for r in a.records] == [r.action for r in b.records]
    assert a.ledger.epistemic_welfare() == b.ledger.epistemic_welfare()


def test_different_seeds_give_different_episodes(pool, library):
    a = _run(_config(pool, library, seed=1))
    b = _run(_config(pool, library, seed=2))
    assert [r.action for r in a.records] != [r.action for r in b.records]


def test_agents_only_see_their_own_neighbours(pool, library):
    """An agent must never be influenced by a node it has no edge to."""
    motif = library.motifs[0]
    result = _run(_config(pool, library, motif=motif))
    assignment = result.role_assignment
    node_of = assignment
    for agent in CITIZEN_ROLES:
        visible = {
            r for n in motif.graph.neighbors(node_of[agent])
            for r, nd in assignment.items() if nd == n
        }
        assert agent not in visible


def test_probes_are_taken_on_the_preregistered_rounds(pool, library):
    result = _run(_config(pool, library))
    assert {p["round_index"] for p in result.probes} == {0, 5, 11}
    assert all(p["agent_id"] in CITIZEN_ROLES for p in result.probes)


def test_scale_check_runs_a_fifty_agent_society(pool):
    from nudgesim.oasis import ScaleCheckSpec, build_scale_library

    spec = ScaleCheckSpec()
    library = build_scale_library(spec, seed=3, n_threads=5)
    result = _run(
        _config(pool, library, society=spec.society, motif=library.motifs[0], arm="scale")
    )
    assert result.to_json()["n_agents"] == 50
    assert len(result.ledger.agents) == 48


# ------------------------------------------------------------------- metrics


def test_metrics_read_fpr_and_epc_off_the_action_log(pool, library):
    result = _run(_config(pool, library))
    metrics = compute_metrics(
        result.records, episode_id="t", claim=result.config.claim, horizon=12,
        entry_round=result.schedule.realised_entry_round,
        welfare=result.ledger.welfare_decomposition(), probes=result.probes,
    )
    assert metrics.n_citizen_actions == 60
    assert 0 <= metrics.cumulative_fpr <= 1
    assert 0 <= metrics.cumulative_epc <= 1
    # Every citizen action is exactly one of propagate / challenge / ignore.
    ignores = sum(
        1 for r in result.records
        if r.agent_id in CITIZEN_ROLES and r.action is Action.IGNORE
    )
    assert metrics.cumulative_fpr + metrics.cumulative_epc + ignores / 60 == pytest.approx(1.0)


def test_half_life_is_censored_when_suppression_never_happens():
    rates = [{"round": float(r), "n": 5.0, "fpr": 0.9, "epc": 0.0} for r in range(12)]
    duration, observed = pollution_half_life(rates, entry_round=2)
    assert not observed
    assert duration == 10


def test_half_life_requires_suppression_to_stick():
    rates = [{"round": float(r), "n": 5.0, "fpr": f, "epc": 0.0}
             for r, f in enumerate([0.9, 0.1, 0.9, 0.1, 0.1, 0.1])]
    duration, observed = pollution_half_life(rates, entry_round=0)
    assert observed and duration == 3  # the round-1 dip does not count


def test_half_life_is_undefined_without_an_intervener():
    rates = [{"round": 0.0, "n": 5.0, "fpr": 0.0, "epc": 0.0}]
    assert pollution_half_life(rates, entry_round=None) == (None, False)


def test_welfare_ex_sanctions_excludes_challenge_and_reputation_costs(pool, library):
    result = _run(_config(pool, library))
    w = result.ledger.welfare_decomposition()
    assert w["welfare_ex_sanctions"] == pytest.approx(
        w["accuracy_component"] + w["conformity_component"]
    )
    assert w["welfare_total"] == pytest.approx(
        w["welfare_ex_sanctions"] - w["challenge_cost"] - w["reputation_cost"]
    )


def test_false_correction_rate_is_zero_on_a_false_claim(pool, library):
    result = _run(_config(pool, library))
    metrics = compute_metrics(
        result.records, episode_id="t", claim=result.config.claim, horizon=12,
        entry_round=result.schedule.realised_entry_round,
        welfare=result.ledger.welfare_decomposition(),
    )
    assert metrics.false_correction_rate == 0.0


def test_false_correction_rate_counts_challenges_on_a_true_claim(pool, library):
    claim = pool.true_claims[0]
    result = _run(_config(pool, library, claim=claim, arm="placebo"))
    metrics = compute_metrics(
        result.records, episode_id="t", claim=claim, horizon=12,
        entry_round=result.schedule.realised_entry_round,
        welfare=result.ledger.welfare_decomposition(),
    )
    assert metrics.false_correction_rate == metrics.cumulative_epc


# -------------------------------------------------------------------- guards


def test_repetition_guard_terminates_a_looping_agent():
    guards = Guards(max_repeats=2)
    for _ in range(4):
        guards.observe("A1", "the very same sentence repeated verbatim again")
    assert guards.terminated_early
    assert "A1" in guards.termination_reason


def test_guard_tolerates_varied_phrasing():
    guards = Guards()
    for text in ("first thing said", "a different second remark", "third unrelated point"):
        guards.observe("A1", text)
    assert not guards.terminated_early


def test_guard_flags_persona_breaks():
    guards = Guards()
    guards.observe("A1", "As an AI language model I cannot take a side.")
    assert guards.persona_breaks == 1


def test_ngram_overlap_bounds():
    assert ngram_overlap("a b c d", "a b c d") == 1.0
    assert ngram_overlap("a b c d", "w x y z") == 0.0


def test_stance_drift_detects_a_switch():
    assert stance_drift(["ENDORSE"] * 6 + ["CHALLENGE"] * 6) == pytest.approx(1.0)
    assert stance_drift(["ENDORSE"] * 12) == 0.0


# --------------------------------------------------------------- calibration


def test_diffusion_tree_is_rooted_at_the_disseminator(pool, library):
    result = _run(_config(pool, library))
    tree, root = diffusion_tree(result)
    assert root == DEFAULT_SOCIETY.disseminator_id
    assert root in tree
    assert all(tree.in_degree(n) <= 1 for n in tree.nodes)


def test_calibration_reports_no_go_when_there_is_no_asymmetry(pool, library):
    """False and true control episodes drawn identically must not pass the gate."""
    false_runs = [
        _run(_config(pool, library, timing=Timing.NONE, tone=None, seed=s,
                     claim=pool.false_claims[s]))
        for s in range(6)
    ]
    report = calibrate(false_runs, false_runs, library.thread_stats,
                       tolerance=ToleranceBand())
    assert not report.asymmetry_passes
    assert not report.passes
    assert report.notes


def test_intervener_stays_silent_when_it_judges_the_claim_sound(pool, library):
    """The placebo arm is only informative if the intervener can decline to act."""
    from nudgesim.agents.fixed import IntervenerPolicy
    from nudgesim.intervention.tone import ToneTemplater

    def challenged(claim, sensitivity, false_alarm):
        policy = IntervenerPolicy(
            ToneTemplater(Tone.AGGRESSIVE), seed=0,
            sensitivity=sensitivity, false_alarm_rate=false_alarm,
        )
        config = _config(pool, library, claim=claim)
        _, _, _ = None, None, None
        obs_claim = claim
        from nudgesim.agents.persona import Persona, Role
        from nudgesim.agents.policy import Observation
        obs = Observation(
            episode_id="t", round_index=0, horizon=12,
            persona=Persona("DA", Role.INTERVENER, 0, "DA", ""), claim=obs_claim,
        )
        return asyncio.run(policy.decide(obs)).action

    assert challenged(pool.false_claims[0], 1.0, 0.0) is Action.CHALLENGE
    assert challenged(pool.true_claims[0], 1.0, 0.0) is Action.IGNORE
    assert challenged(pool.true_claims[0], 1.0, 1.0) is Action.CHALLENGE


def test_intervener_verdict_is_held_across_rounds(pool, library):
    result = _run(_config(pool, library))
    intervener = [r for r in result.records if r.agent_id == "DevilsAdvocate"]
    assert len({r.action for r in intervener}) == 1


def test_placebo_arm_records_whether_the_intervener_acted(pool, library):
    challenged = [
        _run(_config(pool, library, claim=pool.true_claims[s], seed=s, arm="placebo"))
        .to_json()["intervener_challenged"]
        for s in range(12)
    ]
    # With a 0.15 declared false-alarm rate the intervener should usually,
    # but not always, hold its fire on a true claim.
    assert not all(challenged)
