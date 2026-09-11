"""Design sensitivity: what the preregistered grid can and cannot detect.

The plan allocates 30 seeded replications per cell. Whether that is enough is an
empirical question about the effect sizes involved, and it is much cheaper to
answer before the grid runs than after. This module answers it two ways:

  * ``required_n`` -- the closed-form two-sample requirement for a given effect.
  * ``power_curve`` -- resampling power for a specific contrast, estimated by
    repeatedly drawing cells of size n from the observed episodes and counting
    how often the contrast clears alpha. This respects the actual, non-normal
    distribution of the outcome rather than assuming one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from analysis.stats_utils import cohens_d


def required_n(effect_size: float, *, power: float = 0.8, alpha: float = 0.05) -> float:
    """Per-group n for a two-sided two-sample t-test at the given effect size."""
    if not np.isfinite(effect_size) or effect_size == 0:
        return float("inf")
    return float(
        TTestIndPower().solve_power(
            effect_size=abs(effect_size), power=power, alpha=alpha, ratio=1.0
        )
    )


def resampled_power(
    a: Sequence[float],
    b: Sequence[float],
    n: int,
    *,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Fraction of resampled experiments of size n per arm that reach significance."""
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    y = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        xa = rng.choice(x, size=n, replace=True)
        yb = rng.choice(y, size=n, replace=True)
        if np.ptp(xa) == 0 and np.ptp(yb) == 0:
            continue
        if stats.mannwhitneyu(xa, yb, alternative="two-sided").pvalue < alpha:
            hits += 1
    return hits / n_sim


def power_curve(
    a: Sequence[float],
    b: Sequence[float],
    sizes: Sequence[int] = (15, 30, 60, 120, 240, 480),
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, float]]:
    return [
        {
            "n_per_cell": int(n),
            "power": resampled_power(a, b, n, alpha=alpha, seed=seed + n),
        }
        for n in sizes
    ]


def design_sensitivity_report(
    frame: pd.DataFrame, *, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Power for each preregistered primary contrast, at the observed effect sizes."""
    core = frame[(frame["arm"] == "core") & (~frame["terminated_early"])]
    treated = core[core["timing"] != "none"]
    control = core[core["timing"] == "none"]

    contrasts: dict[str, tuple[Sequence[float], Sequence[float]]] = {
        "H1_timing_on_FPR_early_vs_late": (
            treated[treated["timing"] == "early"]["cumulative_fpr"],
            treated[treated["timing"] == "late"]["cumulative_fpr"],
        ),
        "H2_tone_on_FPR_aggressive_vs_empathetic": (
            treated[treated["tone"] == "aggressive_debunking"]["cumulative_fpr"],
            treated[treated["tone"] == "empathetic_nudge"]["cumulative_fpr"],
        ),
        "H3_tone_on_EPC_aggressive_vs_empathetic": (
            treated[treated["tone"] == "aggressive_debunking"]["cumulative_epc"],
            treated[treated["tone"] == "empathetic_nudge"]["cumulative_epc"],
        ),
        "intervention_on_EPC_treated_vs_control": (
            treated["cumulative_epc"],
            control["cumulative_epc"],
        ),
        "intervention_on_FPR_treated_vs_control": (
            treated["cumulative_fpr"],
            control["cumulative_fpr"],
        ),
    }

    out: dict[str, Any] = {
        "alpha": alpha,
        "note": (
            "Per-cell n in the shipped grid is 30. A contrast whose 80%-power "
            "requirement exceeds that is not testable as preregistered, "
            "whatever its true sign."
        ),
        "contrasts": {},
    }
    for name, (a, b) in contrasts.items():
        d = cohens_d(a, b)
        out["contrasts"][name] = {
            "cohens_d": round(d, 4),
            "n_a": int(pd.Series(a).notna().sum()),
            "n_b": int(pd.Series(b).notna().sum()),
            "required_n_per_cell_80pct": (
                None if not np.isfinite(required_n(d)) else round(required_n(d), 1)
            ),
            "adequately_powered_at_30": bool(
                np.isfinite(required_n(d)) and required_n(d) <= 30
            ),
            "power_curve": power_curve(a, b, alpha=alpha, seed=seed),
        }

    # Per-backbone version of the headline contrast: pooling across backbones can
    # mask a contrast that only exists in some model families (H5).
    per_backbone = {}
    for backbone, block in treated.groupby("backbone", observed=True):
        agg = block[block["tone"] == "aggressive_debunking"]["cumulative_epc"]
        emp = block[block["tone"] == "empathetic_nudge"]["cumulative_epc"]
        d = cohens_d(agg, emp)
        per_backbone[str(backbone)] = {
            "cohens_d": round(d, 4),
            "required_n_per_cell_80pct": (
                None if not np.isfinite(required_n(d)) else round(required_n(d), 1)
            ),
            "power_at_30": resampled_power(agg, emp, 30, alpha=alpha, seed=seed),
        }
    out["h3_per_backbone"] = per_backbone
    return out
