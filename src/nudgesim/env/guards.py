"""Termination guards and degeneracy statistics (plan section 12).

Degenerate loops and persona drift are reported, not suppressed: an episode that
collapsed into repetition is flagged in the log and counted in the run report so
the rate is visible per backbone and per condition.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_META_RE = re.compile(
    r"\b(as an ai|as a language model|language model|i am an ai|"
    r"in this simulation|as an assistant|i cannot roleplay)\b",
    re.I,
)
_WORD_RE = re.compile(r"[a-z']+")


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


@dataclass
class Guards:
    """Per-episode degeneracy tracking.

    ``repetition_threshold`` is the 3-gram overlap above which two consecutive
    utterances by the same agent count as a repeat; ``max_repeats`` consecutive
    repeats terminate the episode early with a flag.
    """

    repetition_threshold: float = 0.85
    max_repeats: int = 3
    _last: dict[str, str] = field(default_factory=dict)
    _streak: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    repeat_events: int = 0
    persona_breaks: int = 0
    utterances_seen: int = 0
    terminated_early: bool = False
    termination_reason: str = ""

    def observe(self, agent_id: str, utterance: str) -> None:
        if not utterance.strip():
            return
        self.utterances_seen += 1
        if _META_RE.search(utterance):
            self.persona_breaks += 1
        previous = self._last.get(agent_id)
        if previous is not None and ngram_overlap(previous, utterance) >= self.repetition_threshold:
            self._streak[agent_id] += 1
            self.repeat_events += 1
            if self._streak[agent_id] >= self.max_repeats:
                self.terminated_early = True
                self.termination_reason = f"degenerate repetition by {agent_id}"
        else:
            self._streak[agent_id] = 0
        self._last[agent_id] = utterance

    @property
    def persona_break_rate(self) -> float:
        return self.persona_breaks / self.utterances_seen if self.utterances_seen else 0.0

    @property
    def repetition_rate(self) -> float:
        return self.repeat_events / self.utterances_seen if self.utterances_seen else 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "utterances_seen": self.utterances_seen,
            "repeat_events": self.repeat_events,
            "repetition_rate": round(self.repetition_rate, 6),
            "persona_breaks": self.persona_breaks,
            "persona_break_rate": round(self.persona_break_rate, 6),
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason,
        }


def stance_drift(actions_by_round: list[str]) -> float:
    """Total-variation distance between first-third and last-third action mixes.

    A high value is not necessarily bad -- an intervention is *supposed* to move
    behaviour -- so this is reported as a descriptive statistic and read against
    the control arm, never used as a filter.
    """
    if len(actions_by_round) < 6:
        return 0.0
    k = len(actions_by_round) // 3
    early, late = actions_by_round[:k], actions_by_round[-k:]

    def dist(seq: list[str]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for item in seq:
            out[item] += 1.0 / len(seq)
        return out

    a, b = dist(early), dist(late)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k2, 0.0) - b.get(k2, 0.0)) for k2 in keys)
