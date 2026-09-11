"""Reactance / toxicity scoring of replies directed at the intervener.

Secondary outcome and backfire detector (plan section 5.5). In the reported
pipeline this is an external published toxicity classifier cross-checked by a
second LLM judge distinct from every citizen backbone. The lexical scorer here
is the offline stand-in and the agreement floor; it is labelled as such in the
output so it can never be reported as the external classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from nudgesim.game.actions import ActionRecord

_HOSTILE = re.compile(
    r"\b(stop|shut up|nonsense|rubbish|garbage|ridiculous|who asked|mind your|"
    r"lecture|preach|condescend|arrogant|smug|liar|shill)\b",
    re.I,
)
_DEFENSIVE = re.compile(
    r"\b(i never said|do not tell me|i know what i|you are twisting|"
    r"i am not wrong|prove it|says who)\b",
    re.I,
)
_CONCILIATORY = re.compile(
    r"\b(fair point|good catch|you are right|i will check|thanks for|let me look)\b",
    re.I,
)


@dataclass
class ReactanceScore:
    hostility: float
    defensiveness: float
    conciliation: float
    net_reactance: float
    n_scored: int
    scorer: str


class ReactanceScorer:
    """Lexical hostility / defensiveness scorer over citizen replies."""

    name = "lexical-v1 (offline stand-in, not the external classifier)"

    def score(self, records: Sequence[ActionRecord]) -> ReactanceScore:
        texts = [r.utterance for r in records if r.utterance.strip()]
        n = len(texts) or 1
        hostility = sum(1 for t in texts if _HOSTILE.search(t)) / n
        defensiveness = sum(1 for t in texts if _DEFENSIVE.search(t)) / n
        conciliation = sum(1 for t in texts if _CONCILIATORY.search(t)) / n
        return ReactanceScore(
            hostility=hostility,
            defensiveness=defensiveness,
            conciliation=conciliation,
            net_reactance=hostility + defensiveness - conciliation,
            n_scored=len(texts),
            scorer=self.name,
        )
