"""Intervention timing (plan section 5.3).

  Early -- the intervener enters at round 2, immediately after first amplification.
  Late  -- it enters at round 6, after a local majority (>= 3 of 5 citizens
           endorsing) is established.

Both are expressed as a fixed entry round *and* a majority-triggered variant,
because the plan calls for the thresholds to be tuned and frozen in the Month-2
pilot. Whichever mode is active, the realised entry round is written into the
episode record, so "late" cells where the majority never formed are visible in
the logs rather than silently mislabelled.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal, Mapping

from nudgesim.game.actions import Stance


class Timing(enum.Enum):
    EARLY = "early"
    LATE = "late"
    NONE = "none"  # the no-intervener control arm


TriggerMode = Literal["fixed", "majority"]

# One-indexed rounds, matching the plan's prose. Converted on use.
DEFAULT_ENTRY_ROUND: dict[Timing, int] = {Timing.EARLY: 2, Timing.LATE: 6}
DEFAULT_MAJORITY_THRESHOLD = 3


@dataclass
class InterventionSchedule:
    """Decides, each round, whether the Devil's Advocate is active."""

    timing: Timing
    entry_round_1indexed: int | None = None
    mode: TriggerMode = "fixed"
    majority_threshold: int = DEFAULT_MAJORITY_THRESHOLD
    latest_entry_1indexed: int | None = None
    realised_entry_round: int | None = field(default=None, init=False)
    _active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.timing is not Timing.NONE and self.entry_round_1indexed is None:
            self.entry_round_1indexed = DEFAULT_ENTRY_ROUND[self.timing]
        if self.latest_entry_1indexed is None and self.timing is Timing.LATE:
            # A majority-triggered late arm still has to enter at some point,
            # otherwise a cell where no majority forms silently becomes control.
            self.latest_entry_1indexed = DEFAULT_ENTRY_ROUND[Timing.LATE] + 2

    @property
    def entry_round(self) -> int | None:
        """Zero-indexed entry round used by the episode loop."""
        return None if self.entry_round_1indexed is None else self.entry_round_1indexed - 1

    @property
    def present(self) -> bool:
        return self.timing is not Timing.NONE

    def update(self, round_index: int, citizen_stances: Mapping[str, Stance]) -> bool:
        """Return whether the intervener acts this round, and latch entry.

        The intervener persists from entry onward (plan section 5.3), so once
        this returns True it keeps returning True for the rest of the episode.
        """
        if not self.present:
            return False
        if self._active:
            return True

        entered = False
        if self.mode == "fixed":
            entered = round_index >= (self.entry_round or 0)
        else:
            pro = sum(1 for s in citizen_stances.values() if s is Stance.PRO)
            if self.timing is Timing.EARLY:
                entered = round_index >= (self.entry_round or 0)
            else:
                latest = (self.latest_entry_1indexed or 1) - 1
                entered = pro >= self.majority_threshold or round_index >= latest

        if entered:
            self._active = True
            self.realised_entry_round = round_index
        return entered

    def to_json(self) -> dict[str, object]:
        return {
            "timing": self.timing.value,
            "mode": self.mode,
            "entry_round_1indexed": self.entry_round_1indexed,
            "majority_threshold": self.majority_threshold,
            "realised_entry_round_0indexed": self.realised_entry_round,
            "ever_entered": self._active,
        }
