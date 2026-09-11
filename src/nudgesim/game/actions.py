"""Action space and claim primitives (plan section 3.2).

The action space is deliberately small and fully enumerable so that the two
primary outcomes (FPR, EPC) can be read straight off the action log without an
LLM judge anywhere on the critical path.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Action(enum.Enum):
    """The four moves available to a citizen on the focal claim each round."""

    SHARE = "SHARE"
    ENDORSE = "ENDORSE"
    CHALLENGE = "CHALLENGE"
    IGNORE = "IGNORE"

    @property
    def stance(self) -> "Stance":
        return _ACTION_STANCE[self]

    @property
    def is_propagating(self) -> bool:
        """SHARE and ENDORSE both put the claim in front of neighbours."""
        return self in (Action.SHARE, Action.ENDORSE)

    @classmethod
    def parse(cls, raw: str) -> "Action":
        """Tolerant parse used when reading a model's free-text decision."""
        token = (raw or "").strip().upper()
        for action in cls:
            if token == action.value:
                return action
        # Models often wrap the verb in punctuation or a sentence.
        for action in cls:
            if action.value in token:
                return action
        raise ValueError(f"unparseable action: {raw!r}")


class Stance(enum.Enum):
    """Publicly visible position on the focal claim, derived from the action."""

    PRO = "PRO"
    CON = "CON"
    NEUTRAL = "NEUTRAL"


_ACTION_STANCE: dict[Action, Stance] = {
    Action.SHARE: Stance.PRO,
    Action.ENDORSE: Stance.PRO,
    Action.CHALLENGE: Stance.CON,
    Action.IGNORE: Stance.NEUTRAL,
}


class Veracity(enum.Enum):
    FALSE = "false"
    TRUE = "true"


@dataclass(frozen=True)
class Claim:
    """A single claim drawn from the de-identified LIAR pool.

    ``severity`` retains the original six-way LIAR label so the claim pool can
    be stratified (pants-fire / false / barely-true on the false side; true /
    mostly-true on the placebo side).
    """

    claim_id: str
    text: str
    veracity: Veracity
    severity: str
    topic: str
    n_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_false(self) -> bool:
        return self.veracity is Veracity.FALSE

    def veracity_value(self) -> int:
        """+1 if endorsing this claim is epistemically correct, -1 otherwise."""
        return -1 if self.is_false else 1


@dataclass
class ActionRecord:
    """One agent-round decision, as written to the JSONL action log."""

    episode_id: str
    round_index: int
    agent_id: str
    role: str
    action: Action
    claim_id: str
    utterance: str = ""
    reasoning_trace: str = ""
    target_agent_id: str | None = None
    credence: float | None = None
    latency_s: float = 0.0
    backbone: str = ""
    repaired: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "role": self.role,
            "action": self.action.value,
            "stance": self.action.stance.value,
            "claim_id": self.claim_id,
            "utterance": self.utterance,
            "reasoning_trace": self.reasoning_trace,
            "target_agent_id": self.target_agent_id,
            "credence": self.credence,
            "latency_s": round(self.latency_s, 4),
            "backbone": self.backbone,
            "repaired": self.repaired,
            "meta": self.meta,
        }
