"""Tone templating and scheduling tests -- the manipulation checks in code form."""

from __future__ import annotations

import asyncio

import pytest

from nudgesim.data.claims import ClaimPool
from nudgesim.env.episode import EpisodeConfig, run_episode
from nudgesim.game.actions import Action, Stance
from nudgesim.intervention.scheduler import InterventionSchedule, Timing
from nudgesim.intervention.tone import (
    TONE_STIMULUS,
    Tone,
    ToneTemplater,
    build_payload,
    length_match_report,
    lexical_overlap,
    payload_preservation,
)


def test_payload_is_identical_across_tones(pool: ClaimPool):
    for claim in pool.false_claims[:20]:
        emp = ToneTemplater(Tone.EMPATHETIC, rotate=False).render(claim, 0)
        agg = ToneTemplater(Tone.AGGRESSIVE, rotate=False).render(claim, 0)
        assert emp.payload == agg.payload
        assert lexical_overlap(emp.payload, agg.payload) == 1.0


def test_payload_survives_intact_in_both_wrappers(pool: ClaimPool):
    for claim in pool.false_claims[:20]:
        for tone in Tone:
            message = ToneTemplater(tone, rotate=False).render(claim, 0)
            assert payload_preservation(message) == 1.0


def test_tone_arms_are_length_matched(pool: ClaimPool):
    report = length_match_report(pool.false_claims, tolerance=3)
    assert report["passes"]
    assert report["max_length_gap_tokens"] <= 3


def test_tone_stimulus_contrast_is_in_the_declared_direction():
    emp_face, emp_force = TONE_STIMULUS[Tone.EMPATHETIC]
    agg_face, agg_force = TONE_STIMULUS[Tone.AGGRESSIVE]
    assert agg_face > emp_face, "aggressive framing must threaten face more"
    assert agg_force > emp_force, "aggressive framing must assert more directly"


def test_payload_is_deterministic_per_claim(pool: ClaimPool):
    claim = pool.false_claims[0]
    assert build_payload(claim) == build_payload(claim)


def test_rotation_varies_the_surface_but_not_the_payload(pool: ClaimPool):
    claim = pool.false_claims[0]
    templater = ToneTemplater(Tone.AGGRESSIVE, rotate=True)
    messages = [templater.render(claim, r) for r in range(3)]
    assert len({m.text for m in messages}) > 1
    assert len({m.payload for m in messages}) == 1


def test_fixed_schedule_enters_on_the_declared_round():
    schedule = InterventionSchedule(Timing.EARLY)
    stances = {f"c{i}": Stance.NEUTRAL for i in range(5)}
    assert not schedule.update(0, stances)
    assert schedule.update(1, stances)  # round 2, one-indexed
    assert schedule.realised_entry_round == 1


def test_intervener_persists_once_it_has_entered():
    schedule = InterventionSchedule(Timing.EARLY)
    stances = {f"c{i}": Stance.NEUTRAL for i in range(5)}
    schedule.update(1, stances)
    assert all(schedule.update(r, stances) for r in range(2, 12))


def test_control_arm_never_activates():
    schedule = InterventionSchedule(Timing.NONE)
    stances = {f"c{i}": Stance.PRO for i in range(5)}
    assert not any(schedule.update(r, stances) for r in range(12))
    assert schedule.to_json()["ever_entered"] is False


def test_majority_trigger_fires_on_a_local_majority():
    schedule = InterventionSchedule(Timing.LATE, mode="majority")
    quiet = {f"c{i}": Stance.NEUTRAL for i in range(5)}
    assert not schedule.update(0, quiet)
    majority = {"c0": Stance.PRO, "c1": Stance.PRO, "c2": Stance.PRO,
                "c3": Stance.NEUTRAL, "c4": Stance.NEUTRAL}
    assert schedule.update(1, majority)
    assert schedule.realised_entry_round == 1


def test_majority_trigger_still_enters_when_no_majority_forms():
    """A late cell where the majority never forms must not become a control cell."""
    schedule = InterventionSchedule(Timing.LATE, mode="majority")
    quiet = {f"c{i}": Stance.NEUTRAL for i in range(5)}
    entries = [schedule.update(r, quiet) for r in range(12)]
    assert any(entries)
    assert schedule.realised_entry_round is not None


# --------------------------------------------------- norm robustness (GovSim 3.3)


def _perturbed_config(pool, library, kind, **kw):
    return EpisodeConfig(
        episode_id=f"perturb-{kind}",
        claim=pool.false_claims[0],
        motif=library.motifs[0],
        timing=Timing.NONE,
        tone=None,
        seed=7,
        arm="perturbation",
        perturbation=kind,
        perturbation_round=kw.pop("perturbation_round", 6),
        **kw,
    )


def test_perturbation_leaves_the_early_rounds_to_the_inner_policy(pool, library):
    result = asyncio.run(run_episode(_perturbed_config(pool, library, "amplifier")))
    early = [
        r for r in result.records
        if r.agent_id == "B2" and r.round_index < 6
    ]
    assert early
    assert all("perturbed" not in r.meta for r in early)


def test_amplifier_backs_the_claim_every_round_after_entry(pool, library):
    result = asyncio.run(run_episode(_perturbed_config(pool, library, "amplifier")))
    late = [r for r in result.records if r.agent_id == "B2" and r.round_index >= 6]
    assert late
    assert all(r.action is Action.ENDORSE for r in late)
    assert all(r.meta.get("perturbed") == "amplifier" for r in late)


def test_free_rider_never_pays_the_challenge_cost_after_entry(pool, library):
    result = asyncio.run(run_episode(_perturbed_config(pool, library, "free_rider")))
    late = [r for r in result.records if r.agent_id == "B2" and r.round_index >= 6]
    assert late
    assert all(r.action is not Action.CHALLENGE for r in late)
    # A free rider that would have challenged goes silent rather than switching
    # to a cheap endorsement, so its withdrawal is visible in the action log.
    assert any(r.meta.get("perturbed") == "free_rider" for r in late)


def test_unknown_perturbation_is_rejected(pool, library):
    with pytest.raises(ValueError):
        asyncio.run(run_episode(_perturbed_config(pool, library, "saboteur")))
