"""Figure generation (plan section 9, Month 4).

Static figures for the manuscript. Design rules applied throughout: one y-axis
per panel (never a dual axis), a fixed categorical hue order assigned to
conditions and never cycled, direct labels on every series in addition to the
legend so identity is never carried by colour alone, thin marks, recessive
grid and axes, and text in ink colours rather than series colours.

The control arm is drawn in neutral grey rather than taking a categorical slot:
it is the baseline the other four are read against, not a fifth peer category.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots (light surface). Contrast for slots 3-4 sits below
# 3:1, which is why every series is directly labelled and the numeric table is
# shipped alongside as cell_means.csv.
SERIES = {
    "early|empathetic_nudge": "#2a78d6",
    "early|aggressive_debunking": "#eb6834",
    "late|empathetic_nudge": "#1baf7a",
    "late|aggressive_debunking": "#eda100",
}
CONTROL_COLOR = "#8a8a85"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e2e1dc"

LABELS = {
    "early|empathetic_nudge": "Early · empathetic",
    "early|aggressive_debunking": "Early · aggressive",
    "late|empathetic_nudge": "Late · empathetic",
    "late|aggressive_debunking": "Late · aggressive",
    "none|none": "No intervener (control)",
}

# Direct labels must be unique on their own: "empathetic" alone appears twice,
# once for each timing, and the two curves converge by the final round.
SHORT_LABELS = {
    "early|empathetic_nudge": "E·emp",
    "early|aggressive_debunking": "E·agg",
    "late|empathetic_nudge": "L·emp",
    "late|aggressive_debunking": "L·agg",
    "none|none": "control",
}


def _placed(
    points: list[tuple[float, str, str]], min_gap: float
) -> list[tuple[float, str, str]]:
    """Nudge end-of-line labels apart so converging curves stay readable."""
    out: list[tuple[float, str, str]] = []
    for y, text, color in sorted(points, key=lambda p: p[0]):
        target = y
        for taken, _, _ in out:
            if abs(target - taken) < min_gap:
                target = taken + min_gap
        out.append((target, text, color))
    return out


def _style(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _condition(frame: pd.DataFrame) -> pd.Series:
    return frame["timing"].astype(str) + "|" + frame["tone"].astype(str)


def fig_trajectories(rounds: pd.DataFrame, out: Path) -> Path:
    """FPR and EPC over rounds, by condition. Two panels, one y-axis each."""
    rounds = rounds[(rounds["arm"] == "core") & (~rounds["terminated_early"])].copy()
    rounds["condition"] = _condition(rounds)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, metric, title, ylabel in (
        (axes[0], "fpr", "False-claim propagation (FPR)", "share of citizen actions"),
        (axes[1], "epc", "Endogenous peer correction (EPC)", "share of citizen actions"),
    ):
        endpoints: list[tuple[float, str, str]] = []
        for condition, color in [("none|none", CONTROL_COLOR), *SERIES.items()]:
            block = rounds[rounds["condition"] == condition]
            if block.empty:
                continue
            curve = block.groupby("round", observed=True)[metric].mean()
            ax.plot(
                curve.index + 1,
                curve.to_numpy(),
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=LABELS[condition],
                zorder=3 if condition != "none|none" else 2,
            )
            endpoints.append((float(curve.to_numpy()[-1]), SHORT_LABELS[condition], color))
        for entry, style in ((2, "--"), (6, ":")):
            ax.axvline(entry, color=INK_MUTED, linewidth=1.0, linestyle=style, alpha=0.5)
        ax.text(2.1, ax.get_ylim()[1] * 0.97, "early entry", fontsize=7, color=INK_MUTED, va="top")
        ax.text(6.1, ax.get_ylim()[1] * 0.97, "late entry", fontsize=7, color=INK_MUTED, va="top")
        _style(ax, title=title, xlabel="round", ylabel=ylabel)
        ax.set_xlim(0.5, 13.6)
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for label_y, text, _c in _placed(endpoints, span * 0.05):
            ax.annotate(text, xy=(12.3, label_y), xytext=(6, 0), textcoords="offset points",
                        fontsize=7, color=INK_MUTED, va="center")

    axes[0].legend(
        frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower left", ncol=1
    )
    fig.tight_layout()
    path = out / "fig1_trajectories.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_epc_by_backbone(core: pd.DataFrame, out: Path) -> Path:
    """The headline outcome, per model family, with bootstrap CIs."""
    from analysis.stats_utils import bootstrap_mean

    core = core.copy()
    core["condition"] = _condition(core)
    backbones = sorted(core["backbone"].unique())
    conditions = ["none|none", *SERIES.keys()]

    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=SURFACE)
    width = 0.15
    positions = np.arange(len(backbones))
    for i, condition in enumerate(conditions):
        color = CONTROL_COLOR if condition == "none|none" else SERIES[condition]
        means, los, his = [], [], []
        for backbone in backbones:
            block = core[(core["backbone"] == backbone) & (core["condition"] == condition)]
            mean, lo, hi = bootstrap_mean(block["cumulative_epc"], n_boot=2000)
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            positions + offset,
            means,
            width=width * 0.88,  # 2px surface gap between adjacent bars
            color=color,
            label=LABELS[condition],
            zorder=3,
        )
        ax.errorbar(
            positions + offset,
            means,
            yerr=[los, his],
            fmt="none",
            ecolor=INK_MUTED,
            elinewidth=1.0,
            capsize=2,
            zorder=4,
        )
        for x, value, upper in zip(positions + offset, means, his):
            ax.annotate(
                f"{value:.2f}",
                xy=(x, value + upper),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=INK_MUTED,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([b.replace("surrogate-", "") for b in backbones])
    _style(
        ax,
        title="Endogenous peer-correction rate by backbone and condition (95% bootstrap CI)",
        xlabel="citizen backbone (surrogate profile)",
        ylabel="EPC · share of citizen actions",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper left")
    ax.set_ylim(0, max(0.35, ax.get_ylim()[1] * 1.25))
    fig.tight_layout()
    path = out / "fig2_epc_by_backbone.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_survival(core: pd.DataFrame, out: Path) -> Path:
    """Kaplan-Meier pollution half-life, by tone."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return out / "fig3_survival.SKIPPED"

    treated = core[(core["timing"] != "none") & core["half_life"].notna()]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=SURFACE)
    colors = {"empathetic_nudge": "#2a78d6", "aggressive_debunking": "#eb6834"}
    for tone, block in treated.groupby("tone", observed=True):
        kmf = KaplanMeierFitter()
        kmf.fit(block["half_life"], event_observed=block["half_life_observed"])
        curve = kmf.survival_function_
        ax.step(
            curve.index,
            curve.iloc[:, 0].to_numpy(),
            where="post",
            color=colors.get(str(tone), CONTROL_COLOR),
            linewidth=2.0,
            label=f"{str(tone).replace('_', ' ')} (n={len(block)})",
        )
        ax.annotate(
            str(tone).split("_")[0],
            xy=(curve.index[-1], curve.iloc[-1, 0]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK_MUTED,
            va="center",
        )
    _style(
        ax,
        title="Pollution half-life: rounds until FPR < 0.2 and stays there",
        xlabel="rounds since intervener entry",
        ylabel="share of episodes not yet suppressed",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")
    fig.tight_layout()
    path = out / "fig3_survival.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_power(report: dict[str, Any], out: Path) -> Path:
    """Design sensitivity: what the preregistered 30 seeds/cell can detect."""
    contrasts = report["design_sensitivity"]["contrasts"]
    keys = [
        ("intervention_on_EPC_treated_vs_control", "#2a78d6", "intervention → EPC"),
        ("H1_timing_on_FPR_early_vs_late", "#1baf7a", "H1 timing → FPR"),
        ("H3_tone_on_EPC_aggressive_vs_empathetic", "#eb6834", "H3 tone → EPC"),
        ("H2_tone_on_FPR_aggressive_vs_empathetic", "#eda100", "H2 tone → FPR"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor=SURFACE)
    endpoints: list[tuple[float, str, str]] = []
    for key, color, label in keys:
        if key not in contrasts:
            continue
        curve = contrasts[key]["power_curve"]
        xs = [p["n_per_cell"] for p in curve]
        ys = [p["power"] for p in curve]
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        endpoints.append((float(ys[-1]), label, color))
    ax.axhline(0.8, color=INK_MUTED, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(30, color=INK_MUTED, linewidth=1.0, linestyle=":", alpha=0.6)
    ax.text(31, 0.05, "plan: 30 seeds/cell", fontsize=7, color=INK_MUTED)
    ax.text(16, 0.82, "80% power", fontsize=7, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xticks([15, 30, 60, 120, 240, 480])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Log minor ticks otherwise print "2 x 10^1" on top of the chosen labels.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, title="Design sensitivity of the preregistered grid",
           xlabel="seeded replications per cell", ylabel="power to detect the observed effect")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(13, 1500)
    for label_y, text, _c in _placed(endpoints, 0.07):
        ax.annotate(text, xy=(500, label_y), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=INK_MUTED, va="center")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="center left")
    fig.tight_layout()
    path = out / "fig4_power.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_welfare(core: pd.DataFrame, out: Path) -> Path:
    """Welfare decomposition: where the payoff actually goes, by condition."""
    core = core.copy()
    core["condition"] = _condition(core)
    order = ["none|none", *SERIES.keys()]
    components = [
        ("conformity_component", "#2a78d6", "conformity reward"),
        ("accuracy_component", "#1baf7a", "veracity reward"),
        ("challenge_cost", "#eda100", "challenge cost (−κ)"),
        ("reputation_cost", "#eb6834", "reputation cost (−δ)"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=SURFACE)
    positions = np.arange(len(order))
    width = 0.2
    for i, (column, color, label) in enumerate(components):
        values = [core[core["condition"] == c][column].mean() for c in order]
        # Costs are stored as positive magnitudes; plot them as the debits they are.
        if "cost" in column:
            values = [-v for v in values]
        offset = (i - (len(components) - 1) / 2) * width
        ax.bar(positions + offset, values, width=width * 0.88, color=color, label=label, zorder=3)
        for x, value in zip(positions + offset, values):
            ax.annotate(f"{value:+.0f}", xy=(x, value), xytext=(0, 3 if value >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c].replace(" · ", "\n") for c in order], fontsize=8)
    _style(ax, title="Epistemic welfare decomposition (mean per episode, 5 citizens)",
           xlabel="", ylabel="payoff points")
    ax.legend(
        frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    fig.tight_layout()
    path = out / "fig5_welfare.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_visibility(sweep: list[dict[str, Any]], out: Path) -> Path:
    """The headline: one preprocessing choice, three answers.

    Left panel is the intervention effect with its bootstrap interval; right
    panel is the share of neighbourhoods in which no citizen can observe any
    other, which is what makes the leftmost estimate small. Drawn together
    because the point is that the two move as one.
    """
    labels = [r["label"].replace("\\emph{", "").replace("}", "").replace("\\", "")
              for r in sweep]
    y = np.arange(len(sweep))[::-1]
    ate = [r["ate"] for r in sweep]
    lo = [r["ate"] - r["ci_low"] for r in sweep]
    hi = [r["ci_high"] - r["ate"] for r in sweep]
    blind = [r["no_citizen_visibility"] for r in sweep]
    # The degenerate translation is the one being warned about, so it is the
    # only bar that takes a colour; the other two are read against it.
    colors = ["#c2382b" if b > 0.5 else CONTROL_COLOR for b in blind]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.errorbar(ate, y, xerr=[lo, hi], fmt="o", markersize=7, capsize=4,
                linewidth=1.4, color=INK, ecolor=INK_MUTED, zorder=3)
    for yi, value, color in zip(y, ate, colors):
        ax.plot([value], [yi], "o", markersize=7, color=color, zorder=4)
        ax.annotate(f"{value:+.3f}", (value, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, max(r["ci_high"] for r in sweep) * 1.22)
    _style(ax, title="Intervention → endogenous peer correction (ATE, 95% CI)",
           xlabel="change in share of citizen actions", ylabel="")

    ax = axes[1]
    ax.barh(y, blind, height=0.45, color=colors, zorder=3)
    for yi, value in zip(y, blind):
        ax.annotate(f"{value:.1%}", (value, yi), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlim(0, 1.05)
    _style(ax, title="Neighbourhoods with no citizen-to-citizen edge",
           xlabel="share of motifs", ylabel="")

    fig.tight_layout()
    path = out / "fig6_visibility.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return path


def generate_all(run_dir: str | Path, out_dir: str | Path | None = None) -> list[str]:
    run_dir = Path(run_dir)
    out = Path(out_dir or run_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(run_dir / "results.parquet")
    core = frame[(frame["arm"] == "core") & (~frame["terminated_early"])]
    report = json.loads((run_dir / "analysis" / "analysis.json").read_text(encoding="utf-8"))

    paths = [fig_epc_by_backbone(core, out), fig_survival(core, out),
             fig_power(report, out), fig_welfare(core, out)]
    rounds_path = run_dir / "rounds.parquet"
    if rounds_path.exists():
        paths.insert(0, fig_trajectories(pd.read_parquet(rounds_path), out))
    sweep_path = run_dir.parent / "visibility" / "sweep.json"
    if sweep_path.exists():
        paths.append(fig_visibility(json.loads(sweep_path.read_text()), out))
    return [str(p) for p in paths]


if __name__ == "__main__":
    import sys

    for produced in generate_all(sys.argv[1] if len(sys.argv) > 1 else "runs/main"):
        print(produced)
