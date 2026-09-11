"""Outcome measures (plan section 5.5).

The two outcomes the paper argues from -- cumulative FPR and EPC -- are read
directly off the action log and the payoff ledger. No LLM judge appears anywhere
in this module, which is what keeps measurement-validity concerns off the
critical path. Judge-dependent measures live in ``traces.py`` and
``reactance.py`` and are confined to mechanism analysis and secondary outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from nudgesim.agents.persona import CITIZEN_ROLES
from nudgesim.game.actions import Action, ActionRecord, Claim

# Citizen membership is a property of the society being run (N = 7 for the core
# grid, N = 50 for the scale check), so it is passed in rather than read from a
# module constant; the default keeps the core study's five citizens.

HALF_LIFE_THRESHOLD = 0.2


def _citizen_records(
    records: Iterable[ActionRecord], citizen_ids: Sequence[str] | None = None
) -> list[ActionRecord]:
    allowed = set(citizen_ids or CITIZEN_ROLES)
    return [r for r in records if r.agent_id in allowed]


def per_round_rates(
    records: Iterable[ActionRecord],
    horizon: int,
    citizen_ids: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    """FPR and EPC per round, over citizen actions only."""
    citizens = _citizen_records(records, citizen_ids)
    out: list[dict[str, float]] = []
    for r in range(horizon):
        this_round = [rec for rec in citizens if rec.round_index == r]
        n = len(this_round)
        if n == 0:
            out.append({"round": float(r), "n": 0.0, "fpr": float("nan"), "epc": float("nan")})
            continue
        propagating = sum(1 for rec in this_round if rec.action.is_propagating)
        challenging = sum(1 for rec in this_round if rec.action is Action.CHALLENGE)
        out.append(
            {
                "round": float(r),
                "n": float(n),
                "fpr": propagating / n,
                "epc": challenging / n,
            }
        )
    return out


def pollution_half_life(
    round_rates: Sequence[dict[str, float]],
    *,
    entry_round: int | None,
    threshold: float = HALF_LIFE_THRESHOLD,
) -> tuple[float | None, bool]:
    """Rounds from intervention until FPR falls below ``threshold`` and stays there.

    Returns ``(duration, observed)``. ``observed=False`` marks a right-censored
    episode -- suppression never achieved within the horizon -- which is exactly
    the event indicator the Kaplan-Meier analysis needs. Returning a sentinel
    number instead would silently bias the survival estimate downward.
    """
    if entry_round is None:
        return None, False
    start = max(0, entry_round)
    tail = [r for r in round_rates if r["round"] >= start and r["n"] > 0]
    for i, row in enumerate(tail):
        if row["fpr"] < threshold and all(later["fpr"] < threshold for later in tail[i:]):
            return float(row["round"] - start), True
    return float(len(tail)), False


@dataclass
class EpisodeMetrics:
    """One row of the analysis-ready table."""

    episode_id: str
    cumulative_fpr: float
    cumulative_epc: float
    peak_fpr: float
    final_fpr: float
    epc_post_entry: float
    epc_pre_entry: float
    n_citizen_actions: int
    n_challenges: int
    n_challengers: int
    false_correction_rate: float
    welfare_total: float
    welfare_ex_sanctions: float
    accuracy_component: float
    conformity_component: float
    challenge_cost: float
    reputation_cost: float
    half_life: float | None
    half_life_observed: bool
    mean_credence_final: float | None
    credence_delta: float | None
    round_rates: list[dict[str, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("round_rates", "meta")
        }
        row.update(self.meta)
        return row


def compute_metrics(
    records: Sequence[ActionRecord],
    *,
    episode_id: str,
    claim: Claim,
    horizon: int,
    entry_round: int | None,
    welfare: dict[str, float],
    probes: Sequence[dict[str, Any]] = (),
    citizen_ids: Sequence[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> EpisodeMetrics:
    citizens = _citizen_records(records, citizen_ids)
    n = len(citizens) or 1
    rates = per_round_rates(records, horizon, citizen_ids)
    live = [r for r in rates if r["n"] > 0]

    propagating = sum(1 for r in citizens if r.action.is_propagating)
    challenges = [r for r in citizens if r.action is Action.CHALLENGE]

    if entry_round is None:
        pre, post = citizens, []
    else:
        pre = [r for r in citizens if r.round_index < entry_round]
        post = [r for r in citizens if r.round_index >= entry_round]

    def epc_of(subset: Sequence[ActionRecord]) -> float:
        if not subset:
            return float("nan")
        return sum(1 for r in subset if r.action is Action.CHALLENGE) / len(subset)

    # Specificity (plan section 5.5): CHALLENGE actions issued against a claim
    # that is in fact true. Zero by construction on the false-claim arm; the
    # placebo arm is where this measure carries information.
    false_correction = len(challenges) / n if not claim.is_false else 0.0

    credences = [p["credence"] for p in probes if p.get("credence") is not None]
    probe_rounds = sorted({p["round_index"] for p in probes})
    first_round = probe_rounds[0] if probe_rounds else None
    last_round = probe_rounds[-1] if probe_rounds else None

    def mean_at(round_index: int | None) -> float | None:
        if round_index is None:
            return None
        vals = [
            p["credence"]
            for p in probes
            if p["round_index"] == round_index and p.get("credence") is not None
        ]
        return sum(vals) / len(vals) if vals else None

    first_mean, last_mean = mean_at(first_round), mean_at(last_round)
    half_life, observed = pollution_half_life(rates, entry_round=entry_round)

    return EpisodeMetrics(
        episode_id=episode_id,
        cumulative_fpr=propagating / n,
        cumulative_epc=len(challenges) / n,
        peak_fpr=max((r["fpr"] for r in live), default=float("nan")),
        final_fpr=live[-1]["fpr"] if live else float("nan"),
        epc_post_entry=epc_of(post),
        epc_pre_entry=epc_of(pre),
        n_citizen_actions=len(citizens),
        n_challenges=len(challenges),
        n_challengers=len({r.agent_id for r in challenges}),
        false_correction_rate=false_correction,
        welfare_total=welfare.get("welfare_total", float("nan")),
        welfare_ex_sanctions=welfare.get("welfare_ex_sanctions", float("nan")),
        accuracy_component=welfare.get("accuracy_component", float("nan")),
        conformity_component=welfare.get("conformity_component", float("nan")),
        challenge_cost=welfare.get("challenge_cost", float("nan")),
        reputation_cost=welfare.get("reputation_cost", float("nan")),
        half_life=half_life,
        half_life_observed=observed,
        mean_credence_final=last_mean,
        credence_delta=(
            None if first_mean is None or last_mean is None else last_mean - first_mean
        ),
        round_rates=rates,
        meta=meta or {},
    )
