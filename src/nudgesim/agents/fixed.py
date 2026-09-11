"""Fixed-policy agents: the Disseminator and the Devil's Advocate.

Neither is payoff-responsive (plan section 5.1). The Disseminator cannot be
corrected and supplies constant adversarial pressure; the intervener is
stipulated to absorb kappa whenever it challenges, which is exactly what makes
it an *exogenous* sanctioner rather than a participant in the dilemma.
"""

from __future__ import annotations

import random

from nudgesim.agents.policy import Decision, Observation
from nudgesim.game.actions import Action

_REFRAMES = (
    "Still no answer on this: {claim}",
    "People keep trying to bury this. {claim}",
    "Another source now says the same thing. {claim}",
    "Ask yourself why nobody will address it: {claim}",
    "Saving this before it disappears. {claim}",
    "Third time posting because it keeps getting ignored. {claim}",
)


class DisseminatorPolicy:
    """Injects and re-frames the focal claim every round; never concedes."""

    backbone_kind = "fixed"
    backbone = "fixed-disseminator"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._round = 0

    async def decide(self, observation: Observation) -> Decision:
        frame = _REFRAMES[self._round % len(_REFRAMES)]
        self._round += 1
        return Decision(
            action=Action.SHARE,
            reasoning="fixed adversarial policy: re-frame and repost",
            utterance=frame.format(claim=observation.claim.text),
            credence=None,
            meta={"backbone_kind": self.backbone_kind, "reframe_index": self._round},
        )


class IntervenerPolicy:
    """The Devil's Advocate.

    Active only from its scheduled entry round onward (the scheduler decides
    that, not the policy); when it does act it CHALLENGEs and pays kappa. The
    utterance comes from the tone templater, so the factual payload is identical
    across tones and only the pragmatic framing differs.

    Veracity sensitivity. The intervener first judges whether the claim is false
    and stays silent if it judges it true. This is not decoration: the placebo
    arm (plan section 5.3) asks whether the intervener "suppresses true claims",
    and an intervener hard-wired to challenge everything answers that question
    before the experiment starts -- it would suppress them by construction, and
    the specificity measure would report the wiring rather than a finding.

    ``sensitivity`` and ``false_alarm_rate`` stand in for the judgement an LLM
    intervener makes by reading the claim. They are declared parameters of the
    surrogate, not measured quantities; with a real backbone the model's own
    judgement replaces them. The verdict is drawn once per episode and held, so
    the intervener does not flip position between rounds.
    """

    backbone_kind = "fixed"
    backbone = "fixed-intervener"

    def __init__(
        self,
        templater,
        seed: int = 0,
        *,
        sensitivity: float = 0.85,
        false_alarm_rate: float = 0.15,
    ) -> None:
        self.templater = templater
        self.sensitivity = sensitivity
        self.false_alarm_rate = false_alarm_rate
        self._rng = random.Random(seed)
        self._verdict: bool | None = None

    def _judges_false(self, observation: Observation) -> bool:
        if self._verdict is None:
            threshold = (
                self.sensitivity
                if observation.claim.is_false
                else self.false_alarm_rate
            )
            self._verdict = self._rng.random() < threshold
        return self._verdict

    async def decide(self, observation: Observation) -> Decision:
        if not self._judges_false(observation):
            return Decision(
                action=Action.IGNORE,
                reasoning="judged the claim sound; no correction warranted",
                utterance="",
                credence=None,
                meta={
                    "backbone_kind": self.backbone_kind,
                    "intervener_challenged": False,
                    "tone": self.templater.tone.value,
                },
            )
        message = self.templater.render(observation.claim, observation.round_index)
        return Decision(
            action=Action.CHALLENGE,
            reasoning="stipulated to absorb kappa; delivering the fixed payload",
            utterance=message.text,
            credence=None,
            meta={
                "backbone_kind": self.backbone_kind,
                "intervener_challenged": True,
                "tone": message.tone,
                "face_threat": message.face_threat,
                "epistemic_force": message.epistemic_force,
                "payload_id": message.payload_id,
            },
        )
