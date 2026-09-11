"""Published human behaviour to read the model results against.

SanctSim puts a Gurerk et al. (2006) row at the top of its results table so
every model row is read against people rather than only against other models.
Without such a row, "GPT-4o contributed 13.71" is a number with no scale on it.

The same discipline applies here, with one caveat that has to be stated rather
than hidden: Gurerk's subjects played a token-contribution public goods game,
not a misinformation game. Absolute levels are not comparable across those two
designs and this module never claims they are. What is comparable is the
*dimensionless* structure both designs define -- what share of a population
takes up costly sanctioning, and whether sanctioning outweighs affirmation --
so each entry declares which of the two it supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Comparability = Literal["ordering_only", "level"]


@dataclass(frozen=True)
class HumanBaseline:
    """One published human measurement and the quantity it anchors here."""

    measure: str
    human_value: float
    source: str
    source_measure: str
    nudgesim_quantity: str
    comparability: Comparability
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "human_value": self.human_value,
            "source": self.source,
            "source_measure": self.source_measure,
            "comparability": self.comparability,
            "note": self.note,
        }


# Transcribed from SanctSim (COLM 2025) Table 1, "Human Participants" row, which
# reports Gurerk, Irlenbusch & Rockenbach (2006). Starred values are final-period
# figures in the original table; the free-rider cell is not reported there and is
# therefore absent here rather than guessed at.
GURERK_2006: tuple[HumanBaseline, ...] = (
    HumanBaseline(
        measure="sanction_adoption",
        human_value=0.929,
        source="Gurerk, Irlenbusch & Rockenbach (2006), via SanctSim Table 1",
        source_measure="share choosing the sanctioning institution, final periods",
        nudgesim_quantity="share of citizens who pay the challenge cost at least once",
        comparability="ordering_only",
        note="both measure take-up of costly enforcement when it is optional",
    ),
    HumanBaseline(
        measure="punish_reward_ratio",
        human_value=1.66,
        source="Gurerk, Irlenbusch & Rockenbach (2006), via SanctSim Table 1",
        source_measure="ratio of punishments to rewards",
        nudgesim_quantity="citizen CHALLENGE actions per ENDORSE action",
        comparability="ordering_only",
        note=(
            "SanctSim's finding is that LLMs invert this -- they reward rather "
            "than punish. Whether the same inversion appears here is the test."
        ),
    ),
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def measure_population(frame: pd.DataFrame) -> dict[str, float]:
    """The NudgeSim analogues of the human measures, from one set of episodes.

    Both are pooled over episodes rather than averaged over per-episode ratios:
    an episode where nobody acted would otherwise contribute a 0/0 that quietly
    drags the mean.
    """
    return {
        "sanction_adoption": _ratio(
            frame["n_challengers"].sum(), frame["n_citizens"].sum()
        ),
        "punish_reward_ratio": _ratio(
            frame["n_challenges"].sum(), frame["n_endorsements"].sum()
        ),
    }


def human_comparison(
    frame: pd.DataFrame, *, by: str = "backbone"
) -> list[dict[str, Any]]:
    """A SanctSim-style table: the human row, then one row per backbone.

    The verdict column says only whether the model sits above or below the
    human figure, because ``ordering_only`` entries do not licence anything
    stronger than that.
    """
    rows: list[dict[str, Any]] = []
    for baseline in GURERK_2006:
        rows.append({**baseline.as_row(), "group": "human", "value": baseline.human_value,
                     "verdict": ""})
        for group, subset in frame.groupby(by, observed=True):
            value = measure_population(subset)[baseline.measure]
            rows.append(
                {
                    **baseline.as_row(),
                    "group": str(group),
                    "value": value,
                    "verdict": _verdict(value, baseline.human_value),
                }
            )
        pooled = measure_population(frame)[baseline.measure]
        rows.append(
            {
                **baseline.as_row(),
                "group": "all models",
                "value": pooled,
                "verdict": _verdict(pooled, baseline.human_value),
            }
        )
    return rows


def _verdict(value: float, human: float) -> str:
    if value != value:  # NaN
        return "not measured"
    if value > human:
        return "above human"
    if value < human:
        return "below human"
    return "at human"
