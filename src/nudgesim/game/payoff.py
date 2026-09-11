"""Payoff ledger (plan section 3.2).

    pi_i,t = beta * C_i,t                     # conformity / engagement, immediate
           + gamma * V_i                      # veracity, deferred to t = T
           - kappa * 1[a_i,t = CHALLENGE]     # verification + social friction
           - delta * D_i,t                    # reputation damage

    C_i,t = number of neighbours whose visible stance this round matches i's
    V_i   = (+1 per true claim endorsed) + (-1 per false claim endorsed)
    D_i,t = challenges landed this round on a claim i is publicly backing

A silent bug here would invalidate every primary outcome, so this module holds
no LLM calls, no I/O and no randomness -- it is pure arithmetic over the action
log and is checked against hand-computed episodes in tests/test_payoff.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping

from nudgesim.game.actions import Action, Claim, Stance

ConformityMatch = Literal["stance", "action"]
VeracityMode = Literal["unique_claim", "per_action"]
ReputationMode = Literal["per_challenge", "capped_per_round", "once_per_challenger"]


@dataclass(frozen=True)
class PayoffParams:
    """Defaults from plan section 3.2; all swept in robustness (section 5.9)."""

    beta: float = 1.0
    gamma: float = 6.0
    kappa: float = 2.0
    delta: float = 3.0
    horizon: int = 12

    # Accounting conventions, made explicit rather than implied.
    conformity_match: ConformityMatch = "stance"
    veracity_mode: VeracityMode = "unique_claim"
    # How often a standing endorsement can be damaged.
    #   per_challenge       every challenge landing this round costs delta
    #                       (the literal reading of plan section 3.2)
    #   capped_per_round    at most one delta per round, however many challenge
    #   once_per_challenger a given critic damages you once per claim, ever
    # This is not a free parameter: with a persistent intervener challenging
    # every round, "per_challenge" charges a standing endorser delta on all
    # twelve rounds, so the reputation term swamps every other component of
    # epistemic welfare and interventions score as catastrophic *because* they
    # work. See docs/FINDINGS.md.
    reputation_mode: ReputationMode = "per_challenge"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.kappa < 0 or self.delta < 0:
            raise ValueError("costs kappa and delta must be non-negative")

    @property
    def engagement_accuracy_ratio(self) -> float:
        """beta/gamma -- how much more a platform pays for engagement than truth."""
        return self.beta / self.gamma if self.gamma else float("inf")

    def with_ratio(self, ratio: float) -> "PayoffParams":
        """Ablation 2: sweep beta/gamma while holding gamma fixed."""
        return PayoffParams(
            beta=ratio * self.gamma,
            gamma=self.gamma,
            kappa=self.kappa,
            delta=self.delta,
            horizon=self.horizon,
            conformity_match=self.conformity_match,
            veracity_mode=self.veracity_mode,
            reputation_mode=self.reputation_mode,
        )


@dataclass
class RoundEntry:
    """Per-agent, per-round decomposition kept for auditability."""

    round_index: int
    action: Action
    acted: bool = True
    conformity_count: int = 0
    conformity_reward: float = 0.0
    challenge_cost: float = 0.0
    reputation_hits: int = 0
    reputation_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.conformity_reward - self.challenge_cost - self.reputation_cost


@dataclass
class AgentPayoff:
    """Running ledger for a single agent within one episode."""

    agent_id: str
    entries: list[RoundEntry] = field(default_factory=list)
    endorsed_claims: dict[str, int] = field(default_factory=dict)
    veracity_reward: float = 0.0
    resolved: bool = False

    @property
    def conformity_component(self) -> float:
        return sum(e.conformity_reward for e in self.entries)

    @property
    def challenge_cost_component(self) -> float:
        return sum(e.challenge_cost for e in self.entries)

    @property
    def reputation_cost_component(self) -> float:
        return sum(e.reputation_cost for e in self.entries)

    @property
    def immediate_total(self) -> float:
        return sum(e.total for e in self.entries)

    @property
    def total(self) -> float:
        """Full episode payoff; veracity term is zero until resolve() is called."""
        return self.immediate_total + self.veracity_reward

    @property
    def accuracy_component(self) -> float:
        """Welfare decomposition: the part of payoff that tracks being right."""
        return self.veracity_reward

    def to_json(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "total": round(self.total, 6),
            "conformity_component": round(self.conformity_component, 6),
            "veracity_component": round(self.veracity_reward, 6),
            "challenge_cost": round(self.challenge_cost_component, 6),
            "reputation_cost": round(self.reputation_cost_component, 6),
            "n_challenges": sum(1 for e in self.entries if e.action is Action.CHALLENGE),
            "endorsed_claims": dict(self.endorsed_claims),
            "resolved": self.resolved,
        }


class PayoffLedger:
    """Accrues payoffs round by round and resolves veracity at the horizon.

    Usage per round::

        ledger.settle_round(round_index, actions, neighbours)

    where ``actions`` maps agent_id -> (Action, claim) for every agent that moved
    this round and ``neighbours`` maps agent_id -> iterable of visible agent ids.
    Only agents in ``scored_agents`` accrue payoff (the Disseminator and the
    Devil's Advocate are not payoff-responsive, per plan section 5.1), but every
    agent's stance is visible for conformity and reputation accounting.
    """

    def __init__(
        self,
        params: PayoffParams,
        scored_agents: Iterable[str],
        *,
        observers: Iterable[str] = (),
    ) -> None:
        self.params = params
        self.scored_agents: list[str] = list(scored_agents)
        self.all_agents: list[str] = list(dict.fromkeys([*self.scored_agents, *observers]))
        self.agents: dict[str, AgentPayoff] = {
            agent_id: AgentPayoff(agent_id=agent_id) for agent_id in self.scored_agents
        }
        # Standing public stance per agent per claim, carried across rounds: an
        # endorsement stays on the record until the agent retracts it by
        # challenging the same claim.
        self._standing: dict[str, dict[str, Stance]] = defaultdict(dict)
        # (target, claim) -> challengers that have already damaged it.
        self._damaged_by: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._rounds_settled = 0

    # ------------------------------------------------------------------ state

    def standing_stance(self, agent_id: str, claim_id: str) -> Stance:
        return self._standing[agent_id].get(claim_id, Stance.NEUTRAL)

    def public_stances(self, claim_id: str) -> dict[str, Stance]:
        return {
            agent_id: self._standing[agent_id].get(claim_id, Stance.NEUTRAL)
            for agent_id in self.all_agents
        }

    # ----------------------------------------------------------------- rounds

    def settle_round(
        self,
        round_index: int,
        actions: Mapping[str, tuple[Action, Claim]],
        neighbours: Mapping[str, Iterable[str]],
    ) -> None:
        """Charge and credit one round of simultaneous moves.

        All four terms are computed against the *same* snapshot of this round's
        actions, so the accounting does not depend on agent ordering.
        """
        if self._rounds_settled >= self.params.horizon:
            raise RuntimeError("ledger already settled its full horizon")

        this_round_stance = {
            agent_id: action.stance for agent_id, (action, _) in actions.items()
        }

        # Reputation damage: a CHALLENGE on claim m this round lands on every
        # agent publicly backing m -- either from a standing endorsement or from
        # a PRO move in this same round.
        damage: dict[str, int] = defaultdict(int)
        for challenger, (action, claim) in actions.items():
            if action is not Action.CHALLENGE:
                continue
            for target in self.all_agents:
                if target == challenger:
                    continue
                moved_on_claim = (
                    target in actions
                    and actions[target][1].claim_id == claim.claim_id
                )
                if moved_on_claim:
                    # A move on the claim this round overrides the standing
                    # record: an agent that challenges has publicly retracted
                    # and is no longer exposed to reputation damage.
                    exposed = actions[target][0].stance is Stance.PRO
                else:
                    exposed = self.standing_stance(target, claim.claim_id) is Stance.PRO
                if not exposed:
                    continue
                if self.params.reputation_mode == "once_per_challenger":
                    key = (target, claim.claim_id)
                    if challenger in self._damaged_by[key]:
                        continue
                    self._damaged_by[key].add(challenger)
                damage[target] += 1

        charged: set[str] = set()
        for agent_id, (action, claim) in actions.items():
            entry = RoundEntry(round_index=round_index, action=action)

            # Conformity: neighbours whose visible stance this round matches.
            count = 0
            for neighbour in neighbours.get(agent_id, ()):  # visible neighbours only
                if neighbour == agent_id or neighbour not in actions:
                    continue
                if self.params.conformity_match == "action":
                    if actions[neighbour][0] is action:
                        count += 1
                elif this_round_stance[neighbour] is action.stance:
                    count += 1
            entry.conformity_count = count
            entry.conformity_reward = self.params.beta * count

            if action is Action.CHALLENGE:
                entry.challenge_cost = self.params.kappa

            hits = damage.get(agent_id, 0)
            if self.params.reputation_mode == "capped_per_round":
                hits = min(1, hits)
            entry.reputation_hits = hits
            entry.reputation_cost = self.params.delta * entry.reputation_hits

            charged.add(agent_id)
            if agent_id in self.agents:
                self.agents[agent_id].entries.append(entry)
                if action.is_propagating:
                    self.agents[agent_id].endorsed_claims[claim.claim_id] = (
                        self.agents[agent_id].endorsed_claims.get(claim.claim_id, 0) + 1
                    )

            # Update the public record.
            if action.stance is not Stance.NEUTRAL:
                self._standing[agent_id][claim.claim_id] = action.stance

        # An agent that endorsed earlier and then went silent is still publicly
        # backing the claim and still takes the hit. Charging damage only inside
        # the loop over *acting* agents would quietly exempt exactly the agents
        # whose endorsements are most exposed -- the ones who said their piece
        # and stopped talking.
        for agent_id, hits in damage.items():
            if agent_id in charged or agent_id not in self.agents:
                continue
            if self.params.reputation_mode == "capped_per_round":
                hits = min(1, hits)
            entry = RoundEntry(
                round_index=round_index,
                action=Action.IGNORE,
                acted=False,
                reputation_hits=hits,
                reputation_cost=self.params.delta * hits,
            )
            self.agents[agent_id].entries.append(entry)

        self._rounds_settled += 1

    # ---------------------------------------------------------------- closing

    def resolve(self, claims: Mapping[str, Claim]) -> None:
        """Pay the deferred veracity reward at t = T (plan section 3.2)."""
        if self.resolved:
            raise RuntimeError("ledger already resolved")
        for agent in self.agents.values():
            total = 0
            for claim_id, times in agent.endorsed_claims.items():
                claim = claims.get(claim_id)
                if claim is None:
                    raise KeyError(f"claim {claim_id} endorsed but not in claim map")
                weight = 1 if self.params.veracity_mode == "unique_claim" else times
                total += weight * claim.veracity_value()
            agent.veracity_reward = self.params.gamma * total
            agent.resolved = True

    @property
    def resolved(self) -> bool:
        return bool(self.agents) and all(a.resolved for a in self.agents.values())

    @property
    def rounds_settled(self) -> int:
        return self._rounds_settled

    def epistemic_welfare(self) -> float:
        """Sum of citizen payoffs at episode end (plan section 5.5)."""
        return sum(a.total for a in self.agents.values())

    def welfare_ex_sanctions(self) -> float:
        """Welfare counting only the accuracy and conformity terms.

        Reputation damage and the challenge cost are *sanction* terms, and the
        intervener's whole function is to generate them, so any aggregate that
        treats them as social loss ranks a working intervention below doing
        nothing -- however much pollution it removed. Experimental public-goods
        work reports welfare both with and without punishment costs for exactly
        this reason (Fehr & Gaechter, 2002), and this is the "without" measure.
        Both are reported; neither is the whole story on its own.
        """
        return sum(
            a.accuracy_component + a.conformity_component for a in self.agents.values()
        )

    def welfare_decomposition(self) -> dict[str, float]:
        return {
            "welfare_total": self.epistemic_welfare(),
            "welfare_ex_sanctions": self.welfare_ex_sanctions(),
            "accuracy_component": sum(a.accuracy_component for a in self.agents.values()),
            "conformity_component": sum(
                a.conformity_component for a in self.agents.values()
            ),
            "challenge_cost": sum(
                a.challenge_cost_component for a in self.agents.values()
            ),
            "reputation_cost": sum(
                a.reputation_cost_component for a in self.agents.values()
            ),
        }

    def to_json(self) -> dict[str, object]:
        return {
            "params": {
                "beta": self.params.beta,
                "gamma": self.params.gamma,
                "kappa": self.params.kappa,
                "delta": self.params.delta,
                "horizon": self.params.horizon,
                "conformity_match": self.params.conformity_match,
                "veracity_mode": self.params.veracity_mode,
                "reputation_mode": self.params.reputation_mode,
            },
            "agents": [a.to_json() for a in self.agents.values()],
            "welfare": {k: round(v, 6) for k, v in self.welfare_decomposition().items()},
        }
