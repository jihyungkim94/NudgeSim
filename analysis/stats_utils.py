"""Inference hygiene helpers (plan section 8).

Bootstrap 95% CIs, Holm correction within each outcome family, and effect sizes
reported alongside every p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class Estimate:
    label: str
    value: float
    ci_low: float
    ci_high: float
    n: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "label": self.label,
            "estimate": round(self.value, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "n": self.n,
        }


def bootstrap_mean(
    values: Sequence[float], *, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def bootstrap_diff(
    treat: Sequence[float],
    control: Sequence[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    a = np.asarray([v for v in treat if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in control if np.isfinite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, b.size), replace=True).mean(axis=1)
    diff = da - db
    return (
        float(a.mean() - b.mean()),
        float(np.quantile(diff, alpha / 2)),
        float(np.quantile(diff, 1 - alpha / 2)),
    )


def cohens_d(treat: Sequence[float], control: Sequence[float]) -> float:
    """Hedges-corrected standardised mean difference."""
    a = np.asarray([v for v in treat if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in control if np.isfinite(v)], dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    if pooled == 0:
        return 0.0
    d = (a.mean() - b.mean()) / pooled
    correction = 1 - 3 / (4 * (a.size + b.size) - 9)
    return float(d * correction)


def holm(pvalues: Sequence[float]) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni within one outcome family."""
    finite = [p if np.isfinite(p) else 1.0 for p in pvalues]
    if not finite:
        return [], []
    reject, adjusted, _, _ = multipletests(finite, method="holm")
    return [float(p) for p in adjusted], [bool(r) for r in reject]


def partial_eta_squared(anova_table) -> dict[str, float]:
    """Partial eta^2 per term from a statsmodels type-II ANOVA table."""
    out: dict[str, float] = {}
    if "Residual" not in anova_table.index:
        return out
    ss_resid = float(anova_table.loc["Residual", "sum_sq"])
    for term in anova_table.index:
        if term == "Residual":
            continue
        ss = float(anova_table.loc[term, "sum_sq"])
        denom = ss + ss_resid
        out[term] = float(ss / denom) if denom else float("nan")
    return out


def variance_decomposition(frame, outcome: str, factors: Sequence[str]) -> dict[str, float]:
    """Share of outcome variance attributable to each factor's group means.

    Used for H5: whether backbone choice moves the outcome more than the
    intervention design does.
    """
    total = float(frame[outcome].var(ddof=1))
    out: dict[str, float] = {"total_variance": total}
    if not np.isfinite(total) or total == 0:
        return out
    for factor in factors:
        means = frame.groupby(factor, observed=True)[outcome].mean()
        sizes = frame.groupby(factor, observed=True)[outcome].size()
        grand = frame[outcome].mean()
        between = float((sizes * (means - grand) ** 2).sum() / max(1, len(frame) - 1))
        out[f"between_{factor}"] = between
        out[f"share_{factor}"] = between / total
    return out
