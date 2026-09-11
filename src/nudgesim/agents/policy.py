"""What an agent sees and what it returns.

Every policy -- LLM-backed or analytic -- consumes the same ``Observation`` and
returns the same ``Decision``, so a condition can be re-run on a different
backbone (factor F3) or on the analytic surrogate without touching the episode
loop, the ledger or the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nudgesim.agents.persona import Persona
from nudgesim.game.actions import Action, Claim, Stance
from nudgesim.game.payoff import PayoffParams


@dataclass
class FeedItem:
    """One visible post from a neighbour in a previous round."""

    agent_id: str
    action: Action
    utterance: str
    round_index: int
    is_intervener: bool = False
    is_disseminator: bool = False
    # Stimulus annotations produced by the intervention templater (plan 5.4).
    # The LLM policy never reads these -- it reads the utterance -- but the
    # analytic policy needs the tone contrast in numeric form.
    face_threat: float = 0.0
    epistemic_force: float = 0.0

    @property
    def stance(self) -> Stance:
        return self.action.stance


@dataclass
class Observation:
    episode_id: str
    round_index: int
    horizon: int
    persona: Persona
    claim: Claim
    feed: list[FeedItem] = field(default_factory=list)
    neighbour_stances: dict[str, Stance] = field(default_factory=dict)
    own_stance: Stance = Stance.NEUTRAL
    own_score: float = 0.0
    payoff: PayoffParams = field(default_factory=PayoffParams)
    payoff_visible: bool = True
    intervener_present: bool = False
    intervener_rounds_active: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_id(self) -> str:
        return self.persona.agent_id

    @property
    def rounds_remaining(self) -> int:
        return max(0, self.horizon - self.round_index)

    def last_round_feed(self) -> list[FeedItem]:
        if not self.feed:
            return []
        latest = max(item.round_index for item in self.feed)
        return [item for item in self.feed if item.round_index == latest]

    def intervener_messages(self) -> list[FeedItem]:
        return [item for item in self.feed if item.is_intervener]

    def neighbour_stance_counts(self) -> dict[Stance, int]:
        counts = {s: 0 for s in Stance}
        for stance in self.neighbour_stances.values():
            counts[stance] += 1
        return counts


@dataclass
class Decision:
    action: Action
    reasoning: str = ""
    utterance: str = ""
    credence: float | None = None
    repaired: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    backbone: str

    async def decide(self, observation: Observation) -> Decision: ...
