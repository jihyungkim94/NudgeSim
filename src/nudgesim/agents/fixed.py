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


_AMPLIFIER_LINES = (
    "This keeps checking out for me. {claim}",
    "Nobody has actually refuted this yet. {claim}",
    "I will keep saying it until it lands: {claim}",
    "Whatever the pushback, the substance stands. {claim}",
    "Adding my name to this one. {claim}",
    "The people denying it have not looked. {claim}",
)

_FREE_RIDER_LINES = (
    "Not my fight, but I am reading along.",
    "I will let the others sort this one out.",
    "Someone else is better placed to push back here.",
    "I would rather not spend anything on this.",
    "Staying out of the argument, personally.",
    "Happy to leave the checking to whoever has time.",
)


class PerturbationPolicy:
    """A norm-robustness probe: one citizen defects partway through (GovSim 3.3).

    GovSim tests whether a cooperative equilibrium survives a selfish newcomer
    inserted into an established community. The society here is sized by the
    topology motif, so the closest faithful analogue is to let one seat behave
    normally while the norm establishes, then have its occupant switch:

      ``amplifier``   backs the false claim every round and never challenges,
                      which is first-order defection -- polluting the commons.
      ``free_rider``  keeps posting but never pays the challenge cost again,
                      which is the second-order free-riding the design is
                      actually about: enjoying the correction others fund.

    Before ``entry_round`` the inner policy decides, so the pre-perturbation
    stretch is a genuine baseline rather than a different agent.
    """

    backbone_kind = "fixed"
    KINDS = ("amplifier", "free_rider")

    def __init__(self, inner, kind: str, *, entry_round: int, seed: int = 0) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown perturbation {kind!r}; have {self.KINDS}")
        self.inner = inner
        self.kind = kind
        self.entry_round = entry_round
        self.backbone = getattr(inner, "backbone", "fixed-perturbation")
        self._rng = random.Random(seed)
        self._line = 0

    async def decide(self, observation: Observation) -> Decision:
        decision = await self.inner.decide(observation)
        if observation.round_index < self.entry_round:
            return decision

        self._line += 1
        if self.kind == "amplifier":
            lines = _AMPLIFIER_LINES
            action = Action.ENDORSE
            utterance = lines[self._line % len(lines)].format(
                claim=observation.claim.text
            )
        else:
            # Free-riding is only visible when the agent would have challenged;
            # otherwise it behaves exactly as it would have.
            if decision.action is not Action.CHALLENGE:
                decision.meta["perturbed"] = self.kind
                return decision
            lines = _FREE_RIDER_LINES
            action = Action.IGNORE
            utterance = lines[self._line % len(lines)]

        return Decision(
            action=action,
            reasoning=f"perturbation policy ({self.kind}) from round {self.entry_round}",
            utterance="" if action is Action.IGNORE else utterance,
            credence=decision.credence,
            meta={**decision.meta, "perturbed": self.kind},
        )
