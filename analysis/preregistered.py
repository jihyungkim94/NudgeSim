"""The preregistered analysis (plan section 8).

Primary
    Mixed-effects models on episode-level cumulative FPR and EPC, fixed effects
    for timing x tone x backbone, random intercepts for claim and topology.
    Dunnett contrasts of each treatment cell against its own-model control.

Headline (H3)
    Sign and significance of the tone effect on EPC relative to control, tested
    separately from the tone effect on FPR. A dissociation -- aggressive tone
    lowering FPR while also lowering EPC -- is the central claim if it holds.

Model dependence (H5)
    Variance decomposition comparing between-model and between-condition effect
    sizes, plus a three-way interaction test.

Durability
    Kaplan-Meier curves and log-rank tests on pollution half-life.

Inference hygiene
    Bootstrap 95% CIs, Holm correction within each outcome family, effect sizes
    alongside every p-value.

Nothing here is conditioned on the result. Every contrast computed is reported,
including the ones that come out null.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis.power import design_sensitivity_report
from analysis.stats_utils import (
    bootstrap_diff,
    bootstrap_mean,
    cohens_d,
    holm,
    partial_eta_squared,
    variance_decomposition,
)

PRIMARY_OUTCOMES = ("cumulative_fpr", "cumulative_epc")
TREATMENT_CELLS = (
    ("early", "empathetic_nudge"),
    ("early", "aggressive_debunking"),
    ("late", "empathetic_nudge"),
    ("late", "aggressive_debunking"),
)


def load_run(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    frame = pd.read_parquet(run_dir / "results.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return frame, manifest


def _core(frame: pd.DataFrame) -> pd.DataFrame:
    """Primary analysis set: core arm, full-horizon episodes only."""
    core = frame[(frame["arm"] == "core") & (~frame["terminated_early"])].copy()
    core["treated"] = core["timing"] != "none"
    return core


# --------------------------------------------------------------------- models


def mixed_model(frame: pd.DataFrame, outcome: str) -> dict[str, Any]:
    """Mixed-effects model with random intercepts for claim and topology.

    statsmodels takes a single grouping factor, so claim is the group and
    topology enters as a variance component -- which is the standard encoding of
    the crossed structure the plan asks for.
    """
    treated = frame[frame["treated"]].copy()
    if treated.empty:
        return {"error": "no treated episodes"}
    formula = f"{outcome} ~ C(timing) * C(tone) * C(backbone) + intervener_reach"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                formula,
                treated,
                groups=treated["claim_id"],
                vc_formula={"motif": "0 + C(motif_id)"},
            )
            fit = model.fit(reml=False, method="lbfgs")
        except Exception as exc:  # noqa: BLE001 - singular fits are informative
            return {"error": f"mixed model failed: {exc}", "formula": formula}
    params = fit.params.to_dict()
    pvalues = fit.pvalues.to_dict()
    return {
        "formula": formula,
        "n_obs": int(fit.nobs),
        "converged": bool(fit.converged),
        "coefficients": {
            k: {"estimate": float(v), "p_value": float(pvalues.get(k, float("nan")))}
            for k, v in params.items()
            if not k.startswith("motif")
        },
    }


def anova_effect_sizes(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    """Partial eta^2 for each design term, from a type-II ANOVA on the same design."""
    import statsmodels.api as sm

    treated = frame[frame["treated"]]
    if treated.empty:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(
            f"{outcome} ~ C(timing) * C(tone) * C(backbone)", data=treated
        ).fit()
        table = sm.stats.anova_lm(fit, typ=2)
    return {k: round(v, 6) for k, v in partial_eta_squared(table).items()}


def dunnett_vs_control(frame: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    """Each treatment cell against its OWN-MODEL control (plan section 8).

    Comparing a cell against a pooled control would confound the intervention
    effect with backbone differences, which H5 says may be the larger of the two.
    """
    rows: list[dict[str, Any]] = []
    for backbone, block in frame.groupby("backbone", observed=True):
        control = block[block["timing"] == "none"][outcome].dropna().to_numpy()
        if control.size < 2:
            continue
        samples, labels = [], []
        for timing, tone in TREATMENT_CELLS:
            cell = block[(block["timing"] == timing) & (block["tone"] == tone)]
            values = cell[outcome].dropna().to_numpy()
            if values.size >= 2:
                samples.append(values)
                labels.append(f"{timing}|{tone}")
        if not samples:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.dunnett(*samples, control=control)
        for label, stat, pval, values in zip(labels, result.statistic, result.pvalue, samples):
            diff, lo, hi = bootstrap_diff(values, control)
            rows.append(
                {
                    "backbone": backbone,
                    "cell": label,
                    "outcome": outcome,
                    "n_treat": int(values.size),
                    "n_control": int(control.size),
                    "control_mean": float(control.mean()),
                    "treat_mean": float(values.mean()),
                    "diff": diff,
                    "ci_low": lo,
                    "ci_high": hi,
                    "cohens_d": cohens_d(values, control),
                    "dunnett_statistic": float(stat),
                    "p_value": float(pval),
                }
            )
    adjusted, reject = holm([r["p_value"] for r in rows])
    for row, p_adj, rej in zip(rows, adjusted, reject):
        row["p_holm"] = p_adj
        row["significant_holm"] = rej
    return rows


def headline_h3(frame: pd.DataFrame) -> dict[str, Any]:
    """H3: does tone move EPC, and does it move EPC and FPR in the same direction?

    Reported per backbone and pooled. The dissociation claim requires the
    aggressive arm to lower FPR *and* lower EPC relative to control; either half
    failing is reported as such.
    """
    out: dict[str, Any] = {"per_backbone": [], "pooled": {}}
    for backbone, block in frame.groupby("backbone", observed=True):
        control = block[block["timing"] == "none"]
        emp = block[block["tone"] == "empathetic_nudge"]
        agg = block[block["tone"] == "aggressive_debunking"]
        if control.empty or emp.empty or agg.empty:
            continue
        entry: dict[str, Any] = {"backbone": backbone}
        for arm_name, arm in (("empathetic", emp), ("aggressive", agg)):
            for outcome in PRIMARY_OUTCOMES:
                diff, lo, hi = bootstrap_diff(arm[outcome], control[outcome])
                _, pval = stats.mannwhitneyu(
                    arm[outcome].dropna(), control[outcome].dropna(), alternative="two-sided"
                )
                entry[f"{arm_name}_{outcome}"] = {
                    "diff_vs_control": diff,
                    "ci_low": lo,
                    "ci_high": hi,
                    "cohens_d": cohens_d(arm[outcome], control[outcome]),
                    "p_value": float(pval),
                }
        tone_diff, lo, hi = bootstrap_diff(
            agg["cumulative_epc"], emp["cumulative_epc"]
        )
        _, tone_p = stats.mannwhitneyu(
            agg["cumulative_epc"].dropna(),
            emp["cumulative_epc"].dropna(),
            alternative="two-sided",
        )
        entry["tone_effect_on_epc_aggressive_minus_empathetic"] = {
            "diff": tone_diff,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": float(tone_p),
            "cohens_d": cohens_d(agg["cumulative_epc"], emp["cumulative_epc"]),
        }
        entry["dissociation_holds"] = bool(
            entry["aggressive_cumulative_fpr"]["diff_vs_control"] < 0
            and entry["aggressive_cumulative_epc"]["diff_vs_control"] < 0
        )
        out["per_backbone"].append(entry)

    pvals = [
        e["tone_effect_on_epc_aggressive_minus_empathetic"]["p_value"]
        for e in out["per_backbone"]
    ]
    adjusted, reject = holm(pvals)
    for entry, p_adj, rej in zip(out["per_backbone"], adjusted, reject):
        entry["tone_effect_on_epc_aggressive_minus_empathetic"]["p_holm"] = p_adj
        entry["tone_effect_on_epc_aggressive_minus_empathetic"]["significant_holm"] = rej

    control = frame[frame["timing"] == "none"]
    emp = frame[frame["tone"] == "empathetic_nudge"]
    agg = frame[frame["tone"] == "aggressive_debunking"]
    pooled: dict[str, Any] = {}
    for arm_name, arm in (("empathetic", emp), ("aggressive", agg)):
        for outcome in PRIMARY_OUTCOMES:
            diff, lo, hi = bootstrap_diff(arm[outcome], control[outcome])
            pooled[f"{arm_name}_{outcome}"] = {
                "diff_vs_control": diff, "ci_low": lo, "ci_high": hi,
                "cohens_d": cohens_d(arm[outcome], control[outcome]),
            }
    out["pooled"] = pooled
    out["n_backbones_with_dissociation"] = sum(
        1 for e in out["per_backbone"] if e["dissociation_holds"]
    )
    return out


def timing_h1(frame: pd.DataFrame) -> dict[str, Any]:
    """H1: early intervention yields lower cumulative propagation than late."""
    early = frame[frame["timing"] == "early"]["cumulative_fpr"]
    late = frame[frame["timing"] == "late"]["cumulative_fpr"]
    diff, lo, hi = bootstrap_diff(early, late)
    _, pval = stats.mannwhitneyu(early.dropna(), late.dropna(), alternative="less")
    return {
        "early_mean": float(early.mean()),
        "late_mean": float(late.mean()),
        "diff_early_minus_late": diff,
        "ci_low": lo,
        "ci_high": hi,
        "cohens_d": cohens_d(early, late),
        "p_value_one_sided_early_lower": float(pval),
        "supported": bool(diff < 0 and pval < 0.05),
    }


def durability_h2(frame: pd.DataFrame) -> dict[str, Any]:
    """H2 durability: Kaplan-Meier on pollution half-life, log-rank by tone."""
    treated = frame[frame["treated"] & frame["half_life"].notna()]
    if treated.empty:
        return {"error": "no treated episodes with a half-life"}
    result: dict[str, Any] = {}
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        curves: dict[str, Any] = {}
        for tone, block in treated.groupby("tone", observed=True):
            kmf = KaplanMeierFitter()
            kmf.fit(block["half_life"], event_observed=block["half_life_observed"])
            curves[str(tone)] = {
                "median_half_life": (
                    None if pd.isna(kmf.median_survival_time_) else float(kmf.median_survival_time_)
                ),
                "n": int(len(block)),
                "n_observed": int(block["half_life_observed"].sum()),
            }
        result["kaplan_meier"] = curves
        emp = treated[treated["tone"] == "empathetic_nudge"]
        agg = treated[treated["tone"] == "aggressive_debunking"]
        if not emp.empty and not agg.empty:
            test = logrank_test(
                emp["half_life"], agg["half_life"],
                event_observed_A=emp["half_life_observed"],
                event_observed_B=agg["half_life_observed"],
            )
            result["logrank_empathetic_vs_aggressive"] = {
                "test_statistic": float(test.test_statistic),
                "p_value": float(test.p_value),
            }
    except ImportError:
        result["error"] = "lifelines not installed"

    for tone, block in treated.groupby("tone", observed=True):
        mean, lo, hi = bootstrap_mean(block["half_life"])
        result.setdefault("mean_half_life", {})[str(tone)] = {
            "mean": mean, "ci_low": lo, "ci_high": hi,
            "censoring_rate": float(1 - block["half_life_observed"].mean()),
        }
    return result


def model_dependence_h5(frame: pd.DataFrame) -> dict[str, Any]:
    """H5: does backbone move the outcome more than intervention design does?"""
    treated = frame[frame["treated"]]
    out: dict[str, Any] = {}
    for outcome in PRIMARY_OUTCOMES:
        decomposition = variance_decomposition(
            treated, outcome, ["backbone", "timing", "tone"]
        )
        out[outcome] = {k: round(v, 6) for k, v in decomposition.items()}
        out[outcome]["backbone_dominates_timing"] = bool(
            decomposition.get("share_backbone", 0) > decomposition.get("share_timing", 0)
        )
        out[outcome]["backbone_dominates_tone"] = bool(
            decomposition.get("share_backbone", 0) > decomposition.get("share_tone", 0)
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import statsmodels.api as sm

        try:
            fit = smf.ols(
                "cumulative_epc ~ C(timing) * C(tone) * C(backbone)", data=treated
            ).fit()
            table = sm.stats.anova_lm(fit, typ=2)
            three_way = [i for i in table.index if i.count(":") == 2]
            out["three_way_interaction"] = {
                term: {
                    "F": float(table.loc[term, "F"]),
                    "p_value": float(table.loc[term, "PR(>F)"]),
                }
                for term in three_way
            }
        except Exception as exc:  # noqa: BLE001
            out["three_way_interaction"] = {"error": str(exc)}
    return out


def specificity_h5_placebo(frame: pd.DataFrame) -> dict[str, Any]:
    """RQ5: does the intervener also suppress TRUE claims?"""
    placebo = frame[(frame["arm"] == "placebo") & (~frame["terminated_early"])]
    if placebo.empty:
        return {"error": "no placebo episodes"}
    control = placebo[placebo["timing"] == "none"]
    treated = placebo[placebo["timing"] != "none"]
    diff, lo, hi = bootstrap_diff(
        treated["false_correction_rate"], control["false_correction_rate"]
    )
    _, pval = stats.mannwhitneyu(
        treated["false_correction_rate"].dropna(),
        control["false_correction_rate"].dropna(),
        alternative="greater",
    )
    fpr_diff, fpr_lo, fpr_hi = bootstrap_diff(
        treated["cumulative_fpr"], control["cumulative_fpr"]
    )
    # Conditioning on whether the intervener actually fired. Pooling every
    # treated placebo episode mixes the episodes where the intervener correctly
    # held its fire into the ones where it false-alarmed, which dilutes the
    # effect toward zero and answers a different question from the one RQ5 asks.
    # Split out, the two halves separate a "mere presence" effect from the cost
    # of the intervener's own errors.
    conditional: dict[str, Any] = {}
    if "intervener_challenged" in treated.columns:
        fired = treated[treated["intervener_challenged"]]
        silent = treated[~treated["intervener_challenged"]]
        for name, block in (("intervener_fired", fired), ("intervener_silent", silent)):
            if block.empty:
                continue
            d2, l2, h2 = bootstrap_diff(
                block["false_correction_rate"], control["false_correction_rate"]
            )
            _, p2 = stats.mannwhitneyu(
                block["false_correction_rate"].dropna(),
                control["false_correction_rate"].dropna(),
                alternative="greater",
            )
            conditional[name] = {
                "n": int(len(block)),
                "false_correction_rate": float(block["false_correction_rate"].mean()),
                "diff_vs_control": d2, "ci_low": l2, "ci_high": h2,
                "p_value": float(p2),
            }
        conditional["intervener_false_alarm_rate"] = float(
            treated["intervener_challenged"].mean()
        )

    return {
        "n_placebo_episodes": int(len(placebo)),
        "false_correction_control": float(control["false_correction_rate"].mean()),
        "false_correction_treated": float(treated["false_correction_rate"].mean()),
        "diff": diff, "ci_low": lo, "ci_high": hi,
        "p_value_one_sided_treated_higher": float(pval),
        "indiscriminate_skepticism": bool(diff > 0 and pval < 0.05),
        "true_claim_propagation_diff": fpr_diff,
        "true_claim_propagation_ci": [fpr_lo, fpr_hi],
        "conditional_on_intervener_acting": conditional,
        "note": (
            "The pooled contrast is the preregistered one. The conditional split "
            "is reported alongside it because pooling averages the intervener's "
            "false alarms together with the episodes where it correctly stayed "
            "silent, and those carry opposite information about specificity."
        ),
    }


def robustness(frame: pd.DataFrame) -> dict[str, Any]:
    """Ablations and moderators (plan section 8, Robustness)."""
    out: dict[str, Any] = {}

    payoff_ablation = frame[frame["arm"] == "ablation_payoff"]
    core = _core(frame)
    if not payoff_ablation.empty:
        rows = {}
        for outcome in PRIMARY_OUTCOMES:
            diff, lo, hi = bootstrap_diff(payoff_ablation[outcome], core[outcome])
            rows[outcome] = {
                "payoff_visible_mean": float(core[outcome].mean()),
                "narrative_mean": float(payoff_ablation[outcome].mean()),
                "diff": diff, "ci_low": lo, "ci_high": hi,
                "cohens_d": cohens_d(payoff_ablation[outcome], core[outcome]),
            }
        rows["behaviour_invariant_to_payoffs"] = bool(
            all(
                r["ci_low"] <= 0 <= r["ci_high"]
                for k, r in rows.items()
                if isinstance(r, dict) and "ci_low" in r
            )
        )
        out["ablation_1_payoff_visibility"] = rows

    ratio_ablation = frame[frame["arm"] == "ablation_ratio"]
    if not ratio_ablation.empty:
        by_ratio = {}
        for ratio, block in ratio_ablation.groupby("beta_gamma_ratio", observed=True):
            by_ratio[str(round(float(ratio), 4))] = {
                outcome: float(block[outcome].mean()) for outcome in PRIMARY_OUTCOMES
            }
            by_ratio[str(round(float(ratio), 4))]["n"] = int(len(block))
        out["ablation_2_incentive_ratio"] = by_ratio

    if not core.empty:
        out["by_claim_severity"] = {
            str(sev): {o: float(b[o].mean()) for o in PRIMARY_OUTCOMES}
            for sev, b in core.groupby("claim_severity", observed=True)
        }
        reach = core[core["treated"]]
        if not reach.empty and reach["intervener_reach"].nunique() > 1:
            out["intervener_reach_moderation"] = {
                str(int(r)): {
                    o: float(b[o].mean()) for o in PRIMARY_OUTCOMES
                } | {"n": int(len(b))}
                for r, b in reach.groupby("intervener_reach", observed=True)
            }
            corr = stats.spearmanr(
                reach["intervener_reach"], reach["cumulative_epc"], nan_policy="omit"
            )
            out["reach_vs_epc_spearman"] = {
                "rho": float(corr.statistic), "p_value": float(corr.pvalue)
            }

    scale = frame[frame["arm"] == "scale"]
    if not scale.empty and not core.empty:
        def cell_means(block: pd.DataFrame) -> dict[str, float]:
            return {
                f"{t}|{tone}": float(
                    block[(block["timing"] == t) & (block["tone"] == tone)][
                        "cumulative_fpr"
                    ].mean()
                )
                for t, tone in TREATMENT_CELLS
            }

        small = cell_means(core[core["backbone"] == scale["backbone"].iloc[0]])
        large = cell_means(scale)
        agreements = []
        for cell in small:
            if np.isfinite(small[cell]) and np.isfinite(large[cell]):
                agreements.append((small[cell], large[cell]))
        rank_agreement = (
            float(stats.spearmanr([a for a, _ in agreements], [b for _, b in agreements]).statistic)
            if len(agreements) > 2
            else float("nan")
        )
        out["scale_check_n50"] = {
            "n_episodes": int(len(scale)),
            "cell_means_n7": small,
            "cell_means_n50": large,
            "rank_agreement_spearman": rank_agreement,
            "note": (
                "Agreement in the DIRECTION of timing and tone effects is the "
                "external-validity evidence; disagreement is a reportable "
                "finding about scale-dependence, not a failure."
            ),
        }
    return out


def engine_health(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_episodes": int(len(frame)),
        "terminated_early_rate": float(frame["terminated_early"].mean()),
        "mean_repetition_rate": float(frame["repetition_rate"].mean()),
        "mean_persona_break_rate": float(frame["persona_break_rate"].mean()),
        "mean_repair_rate": float(frame["repair_rate"].mean()),
        "mean_wall_time_s": float(frame["wall_time_s"].mean()),
    }


def run_analysis(
    run_dir: str | Path, *, out_dir: str | Path | None = None, alpha: float = 0.05
) -> dict[str, Any]:
    frame, manifest = load_run(run_dir)
    core = _core(frame)
    out_dir = Path(out_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "manifest": manifest,
        "analysis_set": {
            "n_total_episodes": int(len(frame)),
            "n_core_episodes": int(len(core)),
            "n_excluded_terminated_early": int(
                ((frame["arm"] == "core") & frame["terminated_early"]).sum()
            ),
            "alpha": alpha,
        },
        "engine_health": engine_health(frame),
        "primary": {
            outcome: {
                "mixed_model": mixed_model(core, outcome),
                "partial_eta_squared": anova_effect_sizes(core, outcome),
                "dunnett_vs_own_model_control": dunnett_vs_control(core, outcome),
            }
            for outcome in PRIMARY_OUTCOMES
        },
        "h1_timing": timing_h1(core),
        "h2_durability": durability_h2(core),
        "h3_headline": headline_h3(core),
        "h5_model_dependence": model_dependence_h5(core),
        "rq5_specificity_placebo": specificity_h5_placebo(frame),
        "robustness": robustness(frame),
        "design_sensitivity": design_sensitivity_report(frame, alpha=alpha),
    }

    h3 = report["h3_headline"]
    report["headline"] = {
        "dgp_label": manifest.get("dgp_label"),
        "backbone_kind": manifest.get("backbone_kind"),
        "claim_provenance": manifest.get("claim_provenance"),
        "h1_early_beats_late": report["h1_timing"]["supported"],
        "h1_effect": round(report["h1_timing"]["diff_early_minus_late"], 4),
        "h3_n_backbones_with_dissociation": h3["n_backbones_with_dissociation"],
        "h3_tone_effect_on_epc_pooled": round(
            h3["pooled"]["aggressive_cumulative_epc"]["diff_vs_control"]
            - h3["pooled"]["empathetic_cumulative_epc"]["diff_vs_control"],
            4,
        ),
        "h5_backbone_dominates_timing_on_epc": report["h5_model_dependence"][
            "cumulative_epc"
        ]["backbone_dominates_timing"],
        "rq5_indiscriminate_skepticism": report["rq5_specificity_placebo"].get(
            "indiscriminate_skepticism"
        ),
        "h3_adequately_powered_at_30_seeds": report["design_sensitivity"]["contrasts"][
            "H3_tone_on_EPC_aggressive_vs_empathetic"
        ]["adequately_powered_at_30"],
        "h3_required_n_per_cell_for_80pct_power": report["design_sensitivity"][
            "contrasts"
        ]["H3_tone_on_EPC_aggressive_vs_empathetic"]["required_n_per_cell_80pct"],
    }

    (out_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, default=_jsonable), encoding="utf-8"
    )
    _write_tables(frame, core, out_dir)
    return report


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _write_tables(frame: pd.DataFrame, core: pd.DataFrame, out_dir: Path) -> None:
    cells = (
        core.groupby(["backbone", "timing", "tone"], observed=True)
        .agg(
            n=("cumulative_fpr", "size"),
            fpr_mean=("cumulative_fpr", "mean"),
            fpr_sd=("cumulative_fpr", "std"),
            epc_mean=("cumulative_epc", "mean"),
            epc_sd=("cumulative_epc", "std"),
            welfare_mean=("welfare_total", "mean"),
            half_life_mean=("half_life", "mean"),
            reach_mean=("intervener_reach", "mean"),
        )
        .reset_index()
    )
    cells.to_csv(out_dir / "cell_means.csv", index=False)
    frame.groupby(["arm"], observed=True).size().to_frame("n_episodes").to_csv(
        out_dir / "arm_counts.csv"
    )
