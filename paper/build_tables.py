#!/usr/bin/env python3
"""Emit every table in the paper from run artefacts, so no number is typed by hand.

    .venv/bin/python paper/build_tables.py

Reads the run directories ./reproduce.sh --full produces, and runs the
visibility sweep itself, caching it, because that result is the paper's headline
and is not produced by the standard grid. Writes one .tex fragment per table
into paper/tables/.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from analysis.stats_utils import bootstrap_diff, cohens_d
from nudgesim.agents.bounded_rational import NormParams
from nudgesim.agents.persona import DEFAULT_SOCIETY
from nudgesim.data.claims import ClaimPool
from nudgesim.data.topology import MotifLibrary
from nudgesim.game.payoff import PayoffParams
from nudgesim.runner import GridSpec, expand_grid, run_grid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tables"
RUNS = ROOT.parent / "runs"
LIAR, PHEME, SEED = "data/raw/liar", "data/raw/pheme", 20260911

VISIBILITY_MODELS = (
    ("Reply tree, rooted \\emph{(as commonly specified)}", 0),
    ("Bounded attention, $w=1$ \\emph{(ours)}", 1),
    ("All siblings visible", 99),
)


def _tex(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"wrote tables/{name}.tex")


def _analysis(run: str) -> dict:
    return json.loads((RUNS / run / "analysis" / "analysis.json").read_text())


def _core(run: str) -> pd.DataFrame:
    frame = pd.read_parquet(RUNS / run / "results.parquet")
    return frame[(frame["arm"] == "core") & (~frame["terminated_early"])]


# ----------------------------------------------------- the visibility sweep


def visibility_sweep() -> list[dict]:
    """Run the core grid once per visibility model. Cached: it takes a minute."""
    cache = RUNS / "visibility" / "sweep.json"
    if cache.exists():
        return json.loads(cache.read_text())

    pool = ClaimPool.from_liar(LIAR, seed=SEED)
    spec = GridSpec(include_arms=("core",))
    rows = []
    for label, window in VISIBILITY_MODELS:
        library = MotifLibrary.from_pheme(PHEME, seed=SEED, sibling_window=window)
        configs = expand_grid(
            spec, pool, library, payoff=PayoffParams(), norms=NormParams(), seed=SEED
        )
        summaries, _ = asyncio.run(run_grid(configs, concurrency=32))
        frame = pd.DataFrame(summaries)
        frame = frame[~frame["terminated_early"]]
        treated = frame[frame["treated"]]["cumulative_epc"]
        control = frame[~frame["treated"]]["cumulative_epc"]
        ate, low, high = bootstrap_diff(treated, control)
        rows.append(
            {
                "label": label,
                "sibling_window": window,
                "no_citizen_visibility": _no_citizen_visibility(library),
                "mean_reach": float(frame["intervener_reach"].mean()),
                "epc_control": float(control.mean()),
                "epc_treated": float(treated.mean()),
                "ate": ate,
                "ci_low": low,
                "ci_high": high,
                "cohens_d": cohens_d(treated, control),
            }
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2))
    return rows


def _no_citizen_visibility(library: MotifLibrary) -> float:
    """Share of motifs in which no citizen can see any other citizen."""
    blind = 0
    for motif in library.motifs:
        assignment = motif.assign_roles(DEFAULT_SOCIETY.all_ids)
        node_of = {node: role for node, role in assignment.items()}
        citizens = {n for n, role in node_of.items() if role in DEFAULT_SOCIETY.citizen_ids}
        if not any(u in citizens and v in citizens for u, v in motif.graph.edges):
            blind += 1
    return blind / max(1, len(library.motifs))


def table_visibility() -> None:
    rows = visibility_sweep()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Visibility model & No citizen & Mean & EPC & EPC & ATE [95\% CI] \\",
        r" & visibility & reach & control & treated & \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['no_citizen_visibility']:.1%} & {r['mean_reach']:.2f} & "
            f"{r['epc_control']:.3f} & {r['epc_treated']:.3f} & "
            f"${r['ate']:+.3f}$ [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] \\\\".replace("%", r"\%")
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("visibility", "\n".join(lines))


def table_welfare() -> None:
    core = _core("main")
    g = core.groupby("treated")[
        ["welfare_total", "welfare_ex_sanctions", "accuracy_component",
         "conformity_component", "cumulative_fpr", "cumulative_epc"]
    ].mean()
    labels = [
        ("cumulative_fpr", "False-claim propagation rate (FPR)", "lower"),
        ("cumulative_epc", "Endogenous peer correction (EPC)", "higher"),
        ("welfare_total", r"$\sum \pi$ \emph{(total payoff)}", "higher"),
        ("welfare_ex_sanctions", r"$\sum \pi$ excluding sanction costs", "higher"),
        ("conformity_component", r"Conformity term $\beta C$", "higher"),
        ("accuracy_component", r"Veracity term $\gamma V$", "higher"),
    ]
    lines = [
        r"\begin{tabular}{lrrl}", r"\toprule",
        r"Measure & No intervener & Intervener & Ranks intervention \\", r"\midrule",
    ]
    for col, label, better in labels:
        ctrl, trt = float(g.loc[False, col]), float(g.loc[True, col])
        wins = (trt < ctrl) if better == "lower" else (trt > ctrl)
        verdict = r"\textbf{above} control" if wins else "below control"
        lines.append(f"{label} & {ctrl:.3f} & {trt:.3f} & {verdict} \\\\")
        if col == "cumulative_epc":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("welfare", "\n".join(lines))


def table_power() -> None:
    contrasts = _analysis("main")["design_sensitivity"]["contrasts"]
    pretty = {
        "intervention_on_EPC_treated_vs_control": "Intervention $\\to$ EPC",
        "intervention_on_FPR_treated_vs_control": "Intervention $\\to$ FPR",
        "H1_timing_on_FPR_early_vs_late": "H1 \\quad timing $\\to$ FPR",
        "H3_tone_on_EPC_aggressive_vs_empathetic": "H3 \\quad tone $\\to$ EPC \\emph{(headline)}",
        "H2_tone_on_FPR_aggressive_vs_empathetic": "H2 \\quad tone $\\to$ FPR",
    }
    lines = [
        r"\begin{tabular}{lrrc}", r"\toprule",
        r"Contrast & Cohen's $d$ & $n$/cell for 80\% power & Powered at 30? \\", r"\midrule",
    ]
    for key, label in pretty.items():
        c = contrasts[key]
        mark = r"\checkmark" if c["adequately_powered_at_30"] else "--"
        lines.append(
            f"{label} & ${c['cohens_d']:+.3f}$ & {c['required_n_per_cell_80pct']:,.0f} & {mark} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("power", "\n".join(lines))


def table_validation() -> None:
    runs = [
        ("Declared", "main", 30), ("Inflated $\\times 2.9$", "strong_dgp", 30),
        ("Set to zero", "null_dgp", 30), ("Declared", "highpower", 200),
        ("Inflated $\\times 2.9$", "strong_highpower", 200),
        ("Set to zero", "null_highpower", 200),
    ]
    lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Tone$\to$EPC channel & seeds/cell & Cohen's $d$ & $n$/cell for 80\% power \\", r"\midrule",
    ]
    for i, (label, run, seeds) in enumerate(runs):
        if i == 3:
            lines.append(r"\midrule")
        c = _analysis(run)["design_sensitivity"]["contrasts"][
            "H3_tone_on_EPC_aggressive_vs_empathetic"
        ]
        lines.append(
            f"{label} & {seeds} & ${c['cohens_d']:+.3f}$ & {c['required_n_per_cell_80pct']:,.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("validation", "\n".join(lines))


def table_ate() -> None:
    report = _analysis("main")["primary"]
    lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Outcome & Factor & ATE [95\% CI] & Cohen's $d$ \\", r"\midrule",
    ]
    for outcome, label in (("cumulative_epc", "EPC"), ("cumulative_fpr", "FPR")):
        for i, row in enumerate(report[outcome]["factor_ates"]["pooled"]):
            star = r"$^{*}$" if row["significant"] else ""
            name = f"{row['level']} vs.\\ {row['reference']}"
            lines.append(
                f"{label if i == 0 else ''} & {name} & "
                f"${row['ate']:+.3f}$ [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]{star} & "
                f"${row['cohens_d']:+.3f}$ \\\\"
            )
        lines.append(r"\midrule" if outcome == "cumulative_epc" else "")
    lines = [ln for ln in lines if ln != ""]
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("ate", "\n".join(lines))


def table_human() -> None:
    rows = _analysis("main")["human_baseline"]
    pretty = {
        "sanction_adoption": "Take-up of costly enforcement",
        "punish_reward_ratio": "Sanctions per affirmation",
    }
    lines = [
        r"\begin{tabular}{llr}", r"\toprule",
        r"Measure & Population & Value \\", r"\midrule",
    ]
    for measure, label in pretty.items():
        subset = [r for r in rows if r["measure"] == measure]
        human = next(r for r in subset if r["group"] == "human")
        pooled = next(r for r in subset if r["group"] == "all models")
        lines.append(f"{label} & Human \\citep{{guerek2006}} & \\textbf{{{human['value']:.3f}}} \\\\")
        lines.append(f" & Analytic surrogate & {pooled['value']:.3f} \\\\")
        if measure == "sanction_adoption":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("human", "\n".join(lines))


def table_cost() -> None:
    cost = json.loads((RUNS / "cost" / "cost.json").read_text())
    p, grid = cost["profile"], cost["grid"]
    rec = cost["recommended_portfolio"]
    body = "\n".join([
        r"\begin{tabular}{lr}", r"\toprule",
        r"Quantity & Measured \\", r"\midrule",
        f"Model calls per episode & {p['calls_per_episode']:.0f} \\\\",
        f"Mean input tokens per call & {p['mean_input_tokens']:.0f} \\\\",
        f"Calls per model over the grid & {grid['calls_per_model']:,} \\\\",
        f"Duplicate prompts in {p['calls']:,} calls & \\textbf{{{p['duplicate_rate']:.0%}}} \\\\".replace("%", r"\%"),
        f"Share of prompt that is a reusable prefix & {p['system_share']:.1%} \\\\".replace("%", r"\%"),
        r"\midrule",
        f"Programme cost, recommended portfolio & \\${rec['usd_programme_all_levers']:,.0f} \\\\",
        f"Programme cost, thinking on every arm & \\${cost['programme']['usd']['mixed_tiers']['thinking_output']:,.0f} \\\\",
        r"\bottomrule", r"\end{tabular}",
    ])
    _tex("cost", body)


# The manuscript's figures are produced by analysis/figures.py from the same
# runs; copied in so the paper directory is self-contained for a LaTeX build.
FIGURES = {
    "visibility": "fig6_visibility.png",
    "trajectories": "fig1_trajectories.png",
    "welfare": "fig5_welfare.png",
    "power": "fig4_power.png",
}


def copy_figures() -> None:
    import shutil

    source = RUNS / "main" / "figures"
    target = ROOT / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for name, filename in FIGURES.items():
        origin = source / filename
        if not origin.exists():
            print(f"  missing {origin} -- run analysis.figures first")
            continue
        shutil.copyfile(origin, target / f"{name}.png")
        print(f"wrote figures/{name}.png")


def table_composition() -> None:
    comp = _analysis("main")["composition"]
    lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"\multicolumn{4}{l}{\emph{Inside one mixed episode: same claim, topology and intervention}} \\",
        r"Model in the seat & EPC & FPR & seats \\", r"\midrule",
    ]
    for row in comp["within_mixed"]:
        lines.append(
            f"{row['backbone'].replace('surrogate-', '')} & {row['epc']:.3f} & "
            f"{row['fpr']:.3f} & {row['n_seats']} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{Among other models vs.\ among copies of itself}} \\",
        r"Model & among copies & among others & difference [95\% CI] \\", r"\midrule",
    ]
    for row in sorted(comp["mixed_vs_homogeneous"], key=lambda r: r["diff"]):
        star = r"$^{*}$" if row["significant"] else ""
        lines.append(
            f"{row['backbone'].replace('surrogate-', '')} & "
            f"{row['epc_among_copies']:.3f} & {row['epc_among_others']:.3f} & "
            f"${row['diff']:+.3f}$ [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]{star} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _tex("composition", "\n".join(lines))


def main() -> int:
    copy_figures()
    table_visibility()
    table_welfare()
    table_power()
    table_validation()
    table_ate()
    table_human()
    table_cost()
    table_composition()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
