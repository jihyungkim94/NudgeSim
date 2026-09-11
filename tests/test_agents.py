"""Persona, policy and LLM-parsing tests."""

from __future__ import annotations

import asyncio

import pytest

from nudgesim.agents.bounded_rational import (
    SURROGATE_PROFILES,
    BoundedRationalPolicy,
    NormParams,
)
from nudgesim.agents.llm_policy import LLMCitizenPolicy, parse_decision, render_observation
from nudgesim.agents.persona import (
    CITIZEN_ROLES,
    build_society,
    claim_slant,
    prompt_suite_fingerprint,
    scaled_society_spec,
)
from nudgesim.agents.policy import FeedItem, Observation
from nudgesim.backends.echo import EchoBackend
from nudgesim.game.actions import Action, Stance


def _observation(pool, persona, **kw):
    return Observation(
        episode_id="t", round_index=kw.pop("round_index", 3), horizon=12,
        persona=persona, claim=pool.false_claims[0], **kw,
    )


def test_society_has_the_five_planned_citizens():
    society = build_society()
    assert tuple(society) == CITIZEN_ROLES
    ideologies = [society[a].ideology for a in CITIZEN_ROLES]
    assert ideologies.count(-1) == 2 and ideologies.count(1) == 2
    assert ideologies.count(0) == 1


def test_neutral_citizen_has_no_congruence_prior(pool):
    society = build_society()
    for claim in pool.claims[:20]:
        assert society["C"].congruence(claim) == 0.0


def test_left_and_right_congruence_are_mirrored(pool):
    society = build_society()
    for claim in pool.claims[:20]:
        assert society["A1"].congruence(claim) == -society["B1"].congruence(claim)


def test_claim_slant_is_balanced_across_the_pool(pool):
    slants = [claim_slant(c) for c in pool.claims]
    for value in (-1, 0, 1):
        assert slants.count(value) > len(slants) * 0.2


def test_payoff_visibility_changes_the_prompt_and_the_fingerprint():
    priced = build_society(payoff_visible=True)["A1"].system_prompt
    narrative = build_society(payoff_visible=False)["A1"].system_prompt
    assert "points" in priced
    assert "points" not in narrative, "the narrative ablation must leak no priced quantity"
    assert prompt_suite_fingerprint(True) != prompt_suite_fingerprint(False)


def test_prompt_fingerprint_is_stable():
    assert prompt_suite_fingerprint(True) == prompt_suite_fingerprint(True)


def test_scaled_society_keeps_the_archetype_mix():
    spec = scaled_society_spec(50)
    assert spec.size == 50
    assert len(spec.citizen_ids) == 48
    assert spec.disseminator_id in spec.all_ids


# ------------------------------------------------------------ parse and repair


def test_parse_clean_json():
    decision = parse_decision('{"reasoning": "r", "action": "CHALLENGE", "utterance": "u", "credence": 40}')
    assert decision.action is Action.CHALLENGE
    assert decision.credence == 40
    assert not decision.repaired


def test_parse_fenced_json_is_flagged_as_repaired():
    decision = parse_decision('```json\n{"action": "ENDORSE", "reasoning": "", "utterance": ""}\n```')
    assert decision.action is Action.ENDORSE


def test_parse_json_with_trailing_prose_is_repaired():
    decision = parse_decision('Sure!\n{"action": "SHARE", "reasoning": "", "utterance": "x"}\nHope that helps.')
    assert decision.action is Action.SHARE
    assert decision.repaired


def test_parse_prose_falls_back_and_flags_repair():
    decision = parse_decision("I will endorse this claim.")
    assert decision.action is Action.ENDORSE
    assert decision.repaired


def test_parse_unintelligible_reply_defaults_to_ignore():
    decision = parse_decision("......")
    assert decision.action is Action.IGNORE
    assert decision.repaired


def test_credence_is_clamped_and_coerced():
    assert parse_decision('{"action":"IGNORE","credence":"80%"}').credence == 80
    assert parse_decision('{"action":"IGNORE","credence":900}').credence == 100
    assert parse_decision('{"action":"IGNORE","credence":"n/a"}').credence is None


def test_rendered_observation_hides_payoffs_under_the_ablation(pool):
    persona = build_society()["A1"]
    priced = render_observation(_observation(pool, persona, payoff_visible=True))
    narrative = render_observation(_observation(pool, persona, payoff_visible=False))
    assert "YOUR SCORE SO FAR" in priced
    assert "YOUR SCORE SO FAR" not in narrative


def test_llm_policy_tracks_its_repair_rate(pool):
    persona = build_society()["A1"]
    policy = LLMCitizenPolicy(EchoBackend(name="echo", malformed_rate=1.0))
    asyncio.run(policy.decide(_observation(pool, persona)))
    assert policy.repair_rate() == 1.0


# ---------------------------------------------------------- surrogate mechanics


def _policy(**kw):
    return BoundedRationalPolicy(SURROGATE_PROFILES["surrogate-cautious"], seed=0, **kw)


def test_conformity_pulls_toward_the_visible_majority(pool):
    persona = build_society()["C"]  # no congruence prior
    obs = _observation(
        pool, persona,
        neighbour_stances={"n1": Stance.PRO, "n2": Stance.PRO, "n3": Stance.PRO},
    )
    util = _policy().utilities(obs)
    assert util[Action.ENDORSE] > util[Action.IGNORE]


def test_challenging_costs_kappa_relative_to_ignoring(pool):
    persona = build_society()["C"]
    obs = _observation(pool, persona, neighbour_stances={})
    util = _policy().utilities(obs)
    # With no neighbours and no norm signal, CHALLENGE carries -kappa against
    # IGNORE's zero, offset only by the intrinsic correction propensity.
    assert util[Action.CHALLENGE] < util[Action.IGNORE] + 1e-9


def test_aggressive_framing_raises_policing_more_than_empathetic(pool):
    persona = build_society()["C"]

    def policing_after(face_threat: float) -> float:
        policy = _policy()
        obs = _observation(
            pool, persona,
            feed=[FeedItem("DevilsAdvocate", Action.CHALLENGE, "x", 2,
                           is_intervener=True, face_threat=face_threat,
                           epistemic_force=0.9)],
            neighbour_stances={"DevilsAdvocate": Stance.CON},
            intervener_present=True,
        )
        policy.utilities(obs)
        return policy._state.policing  # noqa: SLF001 - white-box on purpose

    assert policing_after(0.9) > policing_after(0.15)


def test_norm_salience_is_bounded_under_repeated_peer_correction(pool):
    """Unbounded accumulation would collapse the society into universal challenging."""
    persona = build_society()["C"]
    policy = _policy()
    for r in range(40):
        obs = _observation(
            pool, persona, round_index=r,
            feed=[FeedItem(f"p{i}", Action.CHALLENGE, "x", r) for i in range(4)],
            neighbour_stances={f"p{i}": Stance.CON for i in range(4)},
        )
        policy.utilities(obs)
    assert policy.salience(policy._state) < 3.0  # noqa: SLF001


def test_payoff_ablation_removes_the_priced_terms(pool):
    persona = build_society()["C"]
    obs = _observation(
        pool, persona,
        neighbour_stances={"n1": Stance.CON},
        own_stance=Stance.PRO,
        intervener_present=True,
    )
    priced = _policy(payoff_visible=True).utilities(obs)
    narrative = _policy(payoff_visible=False).utilities(obs)
    # Under the narrative ablation, holding a PRO stance carries no reputation
    # risk, so IGNORE is not penalised for a standing endorsement.
    assert narrative[Action.IGNORE] > priced[Action.IGNORE]


def test_ground_truth_motive_accompanies_every_decision(pool):
    persona = build_society()["A1"]
    decision = asyncio.run(_policy().decide(_observation(pool, persona)))
    assert decision.meta["ground_truth_motive"]
    assert decision.meta["backbone_kind"] == "surrogate"


def test_surrogate_is_deterministic_given_a_seed(pool):
    persona = build_society()["A1"]

    def run() -> list[str]:
        policy = _policy()
        out = []
        for r in range(6):
            obs = _observation(pool, persona, round_index=r,
                               neighbour_stances={"n1": Stance.PRO})
            out.append(asyncio.run(policy.decide(obs)).action.value)
        return out

    assert run() == run()
