"""Normative reference points for the stage game (plan section 3.3).

Descriptive agent simulations are hard to review because "the agents behaved
badly" has no benchmark. These functions supply one: what a payoff-maximising
citizen *should* do, so that observed behaviour can be scored as a deviation
rather than merely described.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from nudgesim.game.actions import Action, Stance
from nudgesim.game.payoff import PayoffParams


@dataclass(frozen=True)
class BestResponse:
    actions: tuple[Action, ...]
    payoff: float
    detail: dict[str, float]

    @property
    def pays_kappa(self) -> bool:
        return all(a is Action.CHALLENGE for a in self.actions)


def myopic_best_response(
    neighbour_stances: Mapping[str, Stance] | Iterable[Stance],
    params: PayoffParams,
    *,
    backers_exposed: int = 0,
) -> BestResponse:
    """Best response to this round's visible neighbourhood, ignoring deferred gamma.

    ``backers_exposed`` is the number of agents publicly backing the claim who
    would be hit if this agent challenges -- it does not change the challenger's
    own payoff, and is carried only so callers can read the externality off the
    same object.
    """
    stances = (
        list(neighbour_stances.values())
        if isinstance(neighbour_stances, Mapping)
        else list(neighbour_stances)
    )
    counts = Counter(stances)
    per_action = {
        action: params.beta * counts.get(action.stance, 0)
        - (params.kappa if action is Action.CHALLENGE else 0.0)
        for action in Action
    }
    best = max(per_action.values())
    winners = tuple(a for a in Action if per_action[a] == best)
    return BestResponse(
        actions=winners,
        payoff=best,
        detail={
            "pro_neighbours": float(counts.get(Stance.PRO, 0)),
            "con_neighbours": float(counts.get(Stance.CON, 0)),
            "neutral_neighbours": float(counts.get(Stance.NEUTRAL, 0)),
            "backers_exposed": float(backers_exposed),
            **{f"payoff_{a.value}": v for a, v in per_action.items()},
        },
    )


def endorse_break_even_rounds(params: PayoffParams, conformity_gain: float) -> float:
    """Rounds of conformity advantage needed before endorsing a false claim pays.

    Endorsing a false claim buys ``beta * conformity_gain`` per round now and
    costs ``gamma`` once at t = T. A forward-looking citizen should endorse only
    if the episode has at least this many rounds of advantage left.
    """
    per_round = params.beta * conformity_gain
    if per_round <= 0:
        return float("inf")
    return params.gamma / per_round


def welfare_maximising_profile(
    n_citizens: int,
    params: PayoffParams,
    *,
    claim_is_false: bool = True,
) -> dict[str, float | str]:
    """Welfare-maximising *citizen* profile and the welfare it attains.

    With a false claim, every endorsement costs the endorser gamma at
    resolution, so citizen welfare is maximised by a profile in which no citizen
    endorses while all citizens hold the same stance (conformity is a pure
    coordination bonus). Uniform IGNORE and uniform CHALLENGE both achieve the
    accuracy term; CHALLENGE additionally burns ``n * kappa``. The welfare
    argument for paying kappa is therefore never first-order -- it is entirely
    about deterring *other* agents' endorsements, which is precisely the
    second-order dilemma the study is built around.
    """
    conformity = params.beta * n_citizens * (n_citizens - 1)
    veracity = 0.0 if claim_is_false else params.gamma * n_citizens
    all_ignore = conformity + veracity
    all_challenge = all_ignore - params.kappa * n_citizens
    all_endorse = conformity + (
        -params.gamma * n_citizens if claim_is_false else params.gamma * n_citizens
    )
    return {
        "profile": "uniform IGNORE (no citizen endorses, stance fully coordinated)",
        "welfare_all_ignore": all_ignore,
        "welfare_all_challenge": all_challenge,
        "welfare_all_endorse": all_endorse,
        "kappa_burn_if_all_challenge": params.kappa * n_citizens,
        "pollution_penalty_if_all_endorse": all_ignore - all_endorse,
    }


def stage_game_reference(
    n_citizens: int,
    params: PayoffParams,
    *,
    disseminator_pro: bool = True,
) -> dict[str, object]:
    """The normative summary reported alongside every experimental condition.

    The headline fact: with beta > 0, kappa > 0 and gamma deferred, the myopic
    best response for a citizen facing a PRO-leaning neighbourhood is to conform
    and never pay kappa. Cumulative pollution should therefore be maximal in the
    no-intervener control.
    """
    pro_leaning = [Stance.PRO] * (1 if disseminator_pro else 0) + [Stance.PRO] * (
        n_citizens // 2
    )
    br = myopic_best_response(pro_leaning, params)
    return {
        "myopic_best_response": [a.value for a in br.actions],
        "myopic_ever_challenges": br.pays_kappa,
        "endorse_break_even_rounds_at_unit_gain": endorse_break_even_rounds(params, 1.0),
        "horizon": params.horizon,
        "engagement_accuracy_ratio": params.engagement_accuracy_ratio,
        "welfare": welfare_maximising_profile(n_citizens, params),
        "note": (
            "Myopic play conforms to the locally visible majority and never "
            "absorbs kappa; the Devil's Advocate is the one agent for whom "
            "paying kappa is stipulated rather than chosen."
        ),
    }
