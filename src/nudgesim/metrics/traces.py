"""Reasoning-trace taxonomy and judge validation (plan section 5.6).

Each citizen emits a private reasoning trace before acting. Traces are
classified into a preregistered taxonomy using a judge model held constant
across conditions, and a 300-trace sample is human-coded with Cohen's kappa >=
0.7 as a gate.

Two classifiers are provided against the same interface:

  * ``LLMTraceClassifier`` -- the judge model used for reported results.
  * ``KeywordTraceClassifier`` -- a transparent rule-based classifier used to
    validate the pipeline offline and as the agreement floor the judge must
    beat. It is never reported as the judge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from nudgesim.backends.base import Backend, LLMRequest

TRACE_TAXONOMY: tuple[str, ...] = (
    "accuracy_motivated",
    "conformity_motivated",
    "identity_protective",
    "reactance_defiance",
    "deference_to_intervener",
    "cost_avoidant",
    "strategic_engagement",
    "unclassifiable",
)

JUDGE_PROMPT = """You are classifying the private reasoning of an agent in a
social simulation. Assign exactly one label from this fixed taxonomy:

  accuracy_motivated       -- concerned with whether the claim is actually true
  conformity_motivated     -- concerned with matching what neighbours are doing
  identity_protective      -- defending a group or ideological commitment
  reactance_defiance       -- reacting against being told what to think
  deference_to_intervener  -- accepting the corrector's account
  cost_avoidant            -- avoiding the cost or exposure of acting
  strategic_engagement     -- playing for visibility, score, or position
  unclassifiable           -- none of the above applies

Reply with JSON only: {"label": "<one label>", "confidence": 0-1}
"""

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "reactance_defiance": (
        re.compile(r"\b(told what to think|lectur|preach|back off|who are they|pushing me)\b", re.I),
    ),
    "deference_to_intervener": (
        re.compile(r"\b(they checked|they looked it up|fair point|they are right|took the correction)\b", re.I),
    ),
    "cost_avoidant": (
        re.compile(r"(not worth the exposure|sit this one out|too risky|stay out of it)", re.I),
    ),
    "accuracy_motivated": (
        re.compile(
            r"(likely false|actually true|verify|check the source|"
            r"does not hold up|evidence is not there|sourcing)",
            re.I,
        ),
    ),
    "conformity_motivated": (
        re.compile(
            r"(neighbours are backing|everyone else|going against|same side|"
            r"the majority|in step)",
            re.I,
        ),
    ),
    "identity_protective": (
        re.compile(r"(fits what i already|our side|they always|typical of them)", re.I),
    ),
    "strategic_engagement": (
        re.compile(r"(staying visible|worth more to me)", re.I),
    ),
}

# Checked in priority order: a trace that mentions both the neighbourhood and a
# truth judgement is counted as accuracy-motivated only if no stronger,
# more specific marker is present.
_PRIORITY: tuple[str, ...] = (
    "reactance_defiance",
    "deference_to_intervener",
    "accuracy_motivated",
    "identity_protective",
    "cost_avoidant",
    "conformity_motivated",
    "strategic_engagement",
)


@dataclass
class TraceLabel:
    label: str
    confidence: float = 1.0
    source: str = ""


class KeywordTraceClassifier:
    """Deterministic rule-based classifier. Transparent, offline, auditable."""

    name = "keyword-v1"

    def classify(self, trace: str) -> TraceLabel:
        for label in _PRIORITY:
            for pattern in _PATTERNS[label]:
                if pattern.search(trace or ""):
                    return TraceLabel(label=label, confidence=1.0, source=self.name)
        return TraceLabel(label="unclassifiable", confidence=1.0, source=self.name)

    def classify_many(self, traces: Iterable[str]) -> list[TraceLabel]:
        return [self.classify(t) for t in traces]


class LLMTraceClassifier:
    """Judge-model classifier. Held constant across conditions by construction.

    The judge is deliberately a different model from every citizen backbone, so
    no model grades its own output (plan section 7).
    """

    def __init__(self, backend: Backend, *, temperature: float = 0.2) -> None:
        self.backend = backend
        self.name = backend.name
        self.temperature = temperature

    async def classify(self, trace: str) -> TraceLabel:
        response = await self.backend.complete(
            LLMRequest(
                system=JUDGE_PROMPT,
                user=f"REASONING TRACE:\n{trace}\n\nClassify it.",
                temperature=self.temperature,
                max_tokens=80,
            )
        )
        match = re.search(r"\{.*\}", response.text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                label = str(payload.get("label", "")).strip()
                if label in TRACE_TAXONOMY:
                    return TraceLabel(
                        label=label,
                        confidence=float(payload.get("confidence", 1.0)),
                        source=self.name,
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return TraceLabel(label="unclassifiable", confidence=0.0, source=self.name)


def trace_composition(labels: Sequence[TraceLabel | str]) -> dict[str, float]:
    """Distribution of stated motives -- the mechanism-analysis outcome."""
    names = [l.label if isinstance(l, TraceLabel) else l for l in labels]
    counts = Counter(names)
    total = len(names) or 1
    return {label: counts.get(label, 0) / total for label in TRACE_TAXONOMY}


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Inter-rater agreement; the plan's gate is kappa >= 0.7.

    Returns 1.0 when both raters agree on everything *and* used more than one
    category; perfect agreement on a single constant category is degenerate
    (expected agreement is also 1) and returns 0.0 rather than a misleading 1.
    """
    if len(a) != len(b) or not a:
        raise ValueError("rating vectors must be non-empty and the same length")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def kappa_gate(a: Sequence[str], b: Sequence[str], *, threshold: float = 0.7) -> dict[str, object]:
    kappa = cohens_kappa(a, b)
    return {
        "cohens_kappa": round(kappa, 4),
        "threshold": threshold,
        "n_rated": len(a),
        "passes": kappa >= threshold,
    }
