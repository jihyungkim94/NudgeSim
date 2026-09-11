"""Tone templating with a held-constant factual payload (plan section 5.4).

The most obvious reviewer objection to any tone study is that the tone
manipulation is confounded with informational content. The design answers it
structurally:

  * the factual PAYLOAD is generated once per claim and inserted into both tone
    wrappers verbatim;
  * wrappers are selected from variant banks so the two rendered messages are
    length-matched within a declared tolerance;
  * ``lexical_overlap`` and ``length_match_report`` are run over the whole grid
    as automated manipulation checks before any analysis.

Each message also carries two declared stimulus annotations -- ``face_threat``
and ``epistemic_force``. For LLM runs these are only the *intended* values that
the perceived-tone manipulation check validates against. For the analytic
surrogate they are the channel through which tone enters behaviour at all.
"""

from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass

from nudgesim.game.actions import Claim


class Tone(enum.Enum):
    EMPATHETIC = "empathetic_nudge"
    AGGRESSIVE = "aggressive_debunking"


TONE_MODES: tuple[Tone, ...] = (Tone.EMPATHETIC, Tone.AGGRESSIVE)

# Declared stimulus properties of each framing.
#   face_threat     -- how much the message assigns blame and threatens face
#   epistemic_force -- how directly it asserts the claim is false
# Both wrappers carry the same factual payload; they differ on these two axes.
TONE_STIMULUS: dict[Tone, tuple[float, float]] = {
    Tone.EMPATHETIC: (0.15, 0.55),
    Tone.AGGRESSIVE: (0.90, 0.90),
}

# Payload slot templates. The payload states the correction; it never contains a
# stance marker, an imperative, or an appeal -- those live in the wrappers.
_PAYLOAD_TEMPLATES = (
    "the published {topic} record does not support it",
    "the full {topic} documentation reads differently",
    "the primary {topic} source says something else",
)

_WRAPPERS: dict[Tone, tuple[str, ...]] = {
    Tone.EMPATHETIC: (
        "I can see why this feels urgent. I went and checked, and {payload}. Has anyone else looked?",
        "This is an easy one to believe. I looked it up though, and {payload}. Worth a second check?",
        "I understand why this is going round. Having read it, {payload}. Could someone verify?",
    ),
    Tone.AGGRESSIVE: (
        "This is false and it needs to stop. I went and checked: {payload}. Quit reposting it.",
        "That claim is simply wrong. I looked it up: {payload}. Stop passing it around.",
        "You are spreading something false. Having read it, {payload}. Drop it now.",
    ),
}

_WORD_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class InterventionMessage:
    text: str
    tone: str
    payload: str
    payload_id: str
    face_threat: float
    epistemic_force: float
    n_tokens: int


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def build_payload(claim: Claim) -> tuple[str, str]:
    """Deterministic factual payload for a claim, identical across tones."""
    digest = hashlib.sha256(claim.claim_id.encode("utf-8")).digest()
    template = _PAYLOAD_TEMPLATES[digest[0] % len(_PAYLOAD_TEMPLATES)]
    payload = template.format(topic=claim.topic)
    payload_id = f"pl-{digest.hex()[:8]}"
    return payload, payload_id


class ToneTemplater:
    """Renders the intervener's utterance for one tone condition.

    Wrapper choice is *not* random: for a given claim, the pair of wrappers
    (one per tone) that minimises the length difference is selected, so the two
    experimental arms are matched on length by construction rather than on
    average.
    """

    def __init__(self, tone: Tone, *, rotate: bool = True) -> None:
        self.tone = tone
        self.rotate = rotate
        self._calls = 0

    def _select_index(self, claim: Claim) -> int:
        payload, _ = build_payload(claim)
        best_index, best_gap = 0, float("inf")
        for i, wrapper in enumerate(_WRAPPERS[Tone.EMPATHETIC]):
            a = len(_tokens(wrapper.format(payload=payload)))
            b = len(_tokens(_WRAPPERS[Tone.AGGRESSIVE][i].format(payload=payload)))
            gap = abs(a - b)
            if gap < best_gap:
                best_index, best_gap = i, gap
        return best_index

    def render(self, claim: Claim, round_index: int) -> InterventionMessage:
        payload, payload_id = build_payload(claim)
        index = self._select_index(claim)
        if self.rotate and self._calls:
            # Repeating the identical sentence every round would trip the
            # repetition guard and read as a broken agent; rotate through the
            # matched variants, keeping the same index offset in both arms.
            index = (index + self._calls) % len(_WRAPPERS[self.tone])
        self._calls += 1
        text = _WRAPPERS[self.tone][index].format(payload=payload)
        face_threat, epistemic_force = TONE_STIMULUS[self.tone]
        return InterventionMessage(
            text=text,
            tone=self.tone.value,
            payload=payload,
            payload_id=payload_id,
            face_threat=face_threat,
            epistemic_force=epistemic_force,
            n_tokens=len(_tokens(text)),
        )


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words -- payload equivalence check."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def payload_preservation(message: InterventionMessage) -> float:
    """Fraction of the payload's tokens that survive into the rendered message."""
    payload_tokens = set(_tokens(message.payload))
    if not payload_tokens:
        return 1.0
    return len(payload_tokens & set(_tokens(message.text))) / len(payload_tokens)


def length_match_report(claims: list[Claim], *, tolerance: int = 3) -> dict[str, object]:
    """Automated manipulation check over a claim pool (plan section 5.4).

    Reports the length gap between the two tone arms and confirms that the
    factual payload survives intact in both. A gap outside ``tolerance`` tokens
    means the tone contrast is confounded with message length and the run should
    not proceed.
    """
    gaps: list[int] = []
    overlaps: list[float] = []
    preservations: list[float] = []
    for claim in claims:
        emp = ToneTemplater(Tone.EMPATHETIC, rotate=False).render(claim, 0)
        agg = ToneTemplater(Tone.AGGRESSIVE, rotate=False).render(claim, 0)
        gaps.append(abs(emp.n_tokens - agg.n_tokens))
        overlaps.append(lexical_overlap(emp.payload, agg.payload))
        preservations.append(min(payload_preservation(emp), payload_preservation(agg)))
    n = len(gaps) or 1
    max_gap = max(gaps) if gaps else 0
    return {
        "n_claims": len(claims),
        "mean_length_gap_tokens": sum(gaps) / n,
        "max_length_gap_tokens": max_gap,
        "tolerance_tokens": tolerance,
        "payload_lexical_overlap": sum(overlaps) / n,
        "min_payload_preservation": min(preservations) if preservations else 1.0,
        "passes": max_gap <= tolerance
        and min(preservations, default=1.0) >= 0.999
        and sum(overlaps) / n >= 0.999,
    }
