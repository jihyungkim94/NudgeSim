"""Hand-computed ledger tests -- the Month-1 gate from plan section 9.

Every number in this file was computed by hand from the payoff definition in
plan section 3.2 before the implementation was run. If the ledger drifts, the
primary outcomes (FPR, EPC, welfare) drift silently with it, so these are the
most load-bearing tests in the repository.
"""

from __future__ import annotations

import pytest

from nudgesim.game import Action, Claim, PayoffLedger, PayoffParams, Veracity
from nudgesim.game.benchmark import (
    endorse_break_even_rounds,
    myopic_best_response,
    welfare_maximising_profile,
)
from nudgesim.game.actions import Stance

FALSE_CLAIM = Claim(
    claim_id="m_false",
    text="A de-identified false statement.",
    veracity=Veracity.FALSE,
    severity="pants-fire",
    topic="health",
)
TRUE_CLAIM = Claim(
    claim_id="m_true",
    text="A de-identified true statement.",
    veracity=Veracity.TRUE,
    severity="true",
    topic="health",
)

AGENTS = ["c1", "c2", "c3"]
FULL = {a: [x for x in [*AGENTS, "D"] if x != a] for a in [*AGENTS, "D"]}


def _ledger(horizon: int = 2, **kw) -> PayoffLedger:
    params = PayoffParams(beta=1, gamma=6, kappa=2, delta=3, horizon=horizon, **kw)
    return PayoffLedger(params, scored_agents=AGENTS, observers=["D"])


def test_hand_computed_two_round_episode():
    """Worked example.

    Round 0 -- D SHARE, c1 ENDORSE, c2 ENDORSE, c3 IGNORE
      conformity: c1 sees PRO,PRO,NEUTRAL -> 2; c2 -> 2; c3 (NEUTRAL) -> 0
      no challenges: no kappa, no delta
    Round 1 -- D SHARE, c1 ENDORSE, c2 CHALLENGE, c3 CHALLENGE
      conformity: c1 (PRO) sees PRO,CON,CON -> 1; c2 (CON) -> 1; c3 (CON) -> 1
      kappa: c2 -2, c3 -2
      delta: c1 is the only exposed backer, hit by both challengers -> 2 * 3 = 6
             (c2 retracted this round, so it is not exposed to c3's challenge)
    Veracity at T: c1 and c2 each endorsed one false claim -> -6 each; c3 -> 0
    """
    led = _ledger()
    led.settle_round(
        0,
        {
            "D": (Action.SHARE, FALSE_CLAIM),
            "c1": (Action.ENDORSE, FALSE_CLAIM),
            "c2": (Action.ENDORSE, FALSE_CLAIM),
            "c3": (Action.IGNORE, FALSE_CLAIM),
        },
        FULL,
    )
    assert led.agents["c1"].entries[0].conformity_reward == 2
    assert led.agents["c2"].entries[0].conformity_reward == 2
    assert led.agents["c3"].entries[0].conformity_reward == 0
    assert led.agents["c1"].entries[0].reputation_cost == 0

    led.settle_round(
        1,
        {
            "D": (Action.SHARE, FALSE_CLAIM),
            "c1": (Action.ENDORSE, FALSE_CLAIM),
            "c2": (Action.CHALLENGE, FALSE_CLAIM),
            "c3": (Action.CHALLENGE, FALSE_CLAIM),
        },
        FULL,
    )
    e1, e2, e3 = (led.agents[a].entries[1] for a in AGENTS)
    assert (e1.conformity_reward, e2.conformity_reward, e3.conformity_reward) == (1, 1, 1)
    assert (e1.challenge_cost, e2.challenge_cost, e3.challenge_cost) == (0, 2, 2)
    assert e1.reputation_hits == 2 and e1.reputation_cost == 6
    assert e2.reputation_hits == 0 and e3.reputation_hits == 0

    assert led.agents["c1"].immediate_total == -3
    assert led.agents["c2"].immediate_total == 1
    assert led.agents["c3"].immediate_total == -1

    led.resolve({FALSE_CLAIM.claim_id: FALSE_CLAIM})
    assert led.agents["c1"].total == -9
    assert led.agents["c2"].total == -5
    assert led.agents["c3"].total == -1
    assert led.epistemic_welfare() == -15

    dec = led.welfare_decomposition()
    assert dec["conformity_component"] == 7
    assert dec["accuracy_component"] == -12
    assert dec["challenge_cost"] == 4
    assert dec["reputation_cost"] == 6
    assert (
        dec["conformity_component"]
        + dec["accuracy_component"]
        - dec["challenge_cost"]
        - dec["reputation_cost"]
        == dec["welfare_total"]
    )


def test_true_claim_endorsement_pays_gamma():
    led = _ledger(horizon=1)
    led.settle_round(
        0,
        {a: (Action.ENDORSE, TRUE_CLAIM) for a in AGENTS},
        {a: [x for x in AGENTS if x != a] for a in AGENTS},
    )
    led.resolve({TRUE_CLAIM.claim_id: TRUE_CLAIM})
    # 2 conforming neighbours * beta=1, plus gamma=6 for one true endorsement.
    for a in AGENTS:
        assert led.agents[a].total == 8


def test_repeated_endorsement_charged_once_under_unique_claim_mode():
    led = _ledger(horizon=2)
    for r in (0, 1):
        led.settle_round(r, {"c1": (Action.ENDORSE, FALSE_CLAIM)}, {"c1": []})
    led.resolve({FALSE_CLAIM.claim_id: FALSE_CLAIM})
    assert led.agents["c1"].veracity_reward == -6


def test_per_action_veracity_mode_charges_each_endorsement():
    led = _ledger(horizon=2, veracity_mode="per_action")
    for r in (0, 1):
        led.settle_round(r, {"c1": (Action.ENDORSE, FALSE_CLAIM)}, {"c1": []})
    led.resolve({FALSE_CLAIM.claim_id: FALSE_CLAIM})
    assert led.agents["c1"].veracity_reward == -12


def test_conformity_match_action_is_stricter_than_stance():
    """SHARE and ENDORSE share a stance but are different actions."""
    moves = {
        "c1": (Action.SHARE, FALSE_CLAIM),
        "c2": (Action.ENDORSE, FALSE_CLAIM),
        "c3": (Action.ENDORSE, FALSE_CLAIM),
    }
    nb = {a: [x for x in AGENTS if x != a] for a in AGENTS}

    stance_led = _ledger(horizon=1)
    stance_led.settle_round(0, moves, nb)
    assert stance_led.agents["c1"].entries[0].conformity_count == 2

    action_led = _ledger(horizon=1, conformity_match="action")
    action_led.settle_round(0, moves, nb)
    assert action_led.agents["c1"].entries[0].conformity_count == 0
    assert action_led.agents["c2"].entries[0].conformity_count == 1


def test_conformity_only_counts_visible_neighbours():
    led = _ledger(horizon=1)
    led.settle_round(
        0,
        {a: (Action.ENDORSE, FALSE_CLAIM) for a in AGENTS},
        {"c1": ["c2"], "c2": ["c1"], "c3": []},  # c3 is isolated
    )
    assert led.agents["c1"].entries[0].conformity_count == 1
    assert led.agents["c3"].entries[0].conformity_count == 0


def test_standing_endorsement_stays_exposed_in_later_rounds():
    """An agent that endorses and then goes quiet still owns its endorsement."""
    led = _ledger(horizon=2)
    led.settle_round(0, {"c1": (Action.ENDORSE, FALSE_CLAIM)}, {"c1": []})
    led.settle_round(
        1,
        {"c2": (Action.CHALLENGE, FALSE_CLAIM)},
        {"c2": []},
    )
    assert led.agents["c1"].entries[0].reputation_hits == 0
    # c1 said its piece in round 0 and went silent. It is still publicly backing
    # the claim in round 1, so c2's challenge lands on it -- a silent endorser is
    # not an exempt one.
    silent_entry = led.agents["c1"].entries[-1]
    assert silent_entry.round_index == 1 and not silent_entry.acted
    assert silent_entry.reputation_hits == 1 and silent_entry.reputation_cost == 3
    assert led.agents["c2"].entries[-1].challenge_cost == 2
    assert led.standing_stance("c1", FALSE_CLAIM.claim_id) is Stance.PRO


def test_ledger_rejects_overrun_and_double_resolution():
    led = _ledger(horizon=1)
    led.settle_round(0, {"c1": (Action.IGNORE, FALSE_CLAIM)}, {"c1": []})
    with pytest.raises(RuntimeError):
        led.settle_round(1, {"c1": (Action.IGNORE, FALSE_CLAIM)}, {"c1": []})
    led.resolve({FALSE_CLAIM.claim_id: FALSE_CLAIM})
    with pytest.raises(RuntimeError):
        led.resolve({FALSE_CLAIM.claim_id: FALSE_CLAIM})


def test_unknown_claim_at_resolution_is_an_error_not_a_zero():
    led = _ledger(horizon=1)
    led.settle_round(0, {"c1": (Action.ENDORSE, FALSE_CLAIM)}, {"c1": []})
    with pytest.raises(KeyError):
        led.resolve({})


# ----------------------------------------------------------------- benchmark


def test_myopic_agent_conforms_and_never_pays_kappa():
    params = PayoffParams()
    br = myopic_best_response([Stance.PRO, Stance.PRO, Stance.NEUTRAL], params)
    assert set(br.actions) == {Action.SHARE, Action.ENDORSE}
    assert not br.pays_kappa
    assert br.payoff == 2


def test_challenging_only_pays_myopically_with_a_large_con_majority():
    params = PayoffParams(kappa=2)
    br = myopic_best_response([Stance.CON] * 4 + [Stance.PRO], params)
    assert br.actions == (Action.CHALLENGE,)  # 4 - 2 = 2 beats the 1 PRO neighbour
    # A bare majority is not enough: kappa eats the whole margin.
    tie = myopic_best_response([Stance.CON] * 3 + [Stance.PRO], params)
    assert Action.CHALLENGE in tie.actions and Action.ENDORSE in tie.actions


def test_break_even_matches_gamma_over_beta():
    assert endorse_break_even_rounds(PayoffParams(), 1.0) == 6.0
    assert endorse_break_even_rounds(PayoffParams().with_ratio(3.0), 1.0) == pytest.approx(
        1 / 3
    )
    assert endorse_break_even_rounds(PayoffParams(), 0.0) == float("inf")


def test_uniform_ignore_beats_uniform_endorse_on_a_false_claim():
    w = welfare_maximising_profile(5, PayoffParams(), claim_is_false=True)
    assert w["welfare_all_ignore"] > w["welfare_all_challenge"] > w["welfare_all_endorse"]


# ------------------------------------------------- reputation accounting modes


def _two_challengers_on_a_standing_endorsement(mode: str, rounds: int = 3):
    """c1 endorses once, then c2 and c3 challenge it every round."""
    params = PayoffParams(
        beta=0, gamma=6, kappa=2, delta=3, horizon=rounds + 1, reputation_mode=mode
    )
    led = PayoffLedger(params, scored_agents=AGENTS)
    led.settle_round(0, {"c1": (Action.ENDORSE, FALSE_CLAIM)}, {"c1": []})
    for r in range(1, rounds + 1):
        led.settle_round(
            r,
            {
                "c2": (Action.CHALLENGE, FALSE_CLAIM),
                "c3": (Action.CHALLENGE, FALSE_CLAIM),
            },
            {"c2": [], "c3": []},
        )
    return led.agents["c1"].reputation_cost_component


def test_reputation_per_challenge_is_the_literal_plan_reading():
    # 3 rounds x 2 challengers x delta 3
    assert _two_challengers_on_a_standing_endorsement("per_challenge") == 18


def test_reputation_capped_per_round_charges_at_most_one_hit():
    # 3 rounds x 1 hit x delta 3
    assert _two_challengers_on_a_standing_endorsement("capped_per_round") == 9


def test_reputation_once_per_challenger_charges_each_critic_once():
    # 2 distinct challengers x delta 3, however long they keep it up
    assert _two_challengers_on_a_standing_endorsement("once_per_challenger") == 6
    assert _two_challengers_on_a_standing_endorsement("once_per_challenger", rounds=9) == 6
