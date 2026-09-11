"""Command-line entry point.

    nudgesim calibrate  --out runs/calibration     # plan section 5.7 Go/No-Go
    nudgesim check      --out runs/checks          # manipulation + ledger checks
    nudgesim run        --out runs/main            # the full preregistered grid
    nudgesim analyze    --run runs/main            # plan section 8 analysis
    nudgesim reproduce  --out runs/repro           # all of the above, one command
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from nudgesim.agents.bounded_rational import NormParams
from nudgesim.agents.persona import DEFAULT_SOCIETY
from nudgesim.calibration import ToleranceBand, calibrate
from nudgesim.data.claims import ClaimPool
from nudgesim.data.topology import MotifLibrary
from nudgesim.env.episode import EpisodeConfig, run_episode
from nudgesim.game.benchmark import stage_game_reference
from nudgesim.game.payoff import PayoffParams
from nudgesim.intervention.scheduler import Timing
from nudgesim.intervention.tone import length_match_report
from nudgesim.runner import (
    GridSpec,
    RunManifest,
    expand_grid,
    run_grid,
    write_run,
)


def _load_data(args: argparse.Namespace) -> tuple[ClaimPool, MotifLibrary]:
    pool = (
        ClaimPool.from_liar(args.liar, seed=args.seed)
        if args.liar
        else ClaimPool.synthetic(seed=args.seed)
    )
    library = (
        MotifLibrary.from_pheme(args.pheme, seed=args.seed)
        if args.pheme
        else MotifLibrary.synthetic(seed=args.seed)
    )
    return pool, library


def _payoff(args: argparse.Namespace) -> PayoffParams:
    return PayoffParams(
        beta=args.beta, gamma=args.gamma, kappa=args.kappa, delta=args.delta,
        horizon=args.horizon,
    )


def _norms(args: argparse.Namespace) -> tuple[NormParams, str]:
    """Select the declared data-generating process for the surrogate run."""
    if args.dgp == "null":
        return (
            NormParams(omega_crowd_tone=0.0, omega_crowd_base=0.25),
            "null-tone-crowding (H3 effect set to zero; false-positive check)",
        )
    if args.dgp == "strong":
        return (
            NormParams(omega_crowd_tone=2.5),
            "strong-tone-crowding (H3 effect inflated to a detectable size; "
            "sensitivity check for the analysis pipeline)",
        )
    if args.dgp == "no-novelty":
        return (
            NormParams(novelty_engagement=0.0),
            "no-novelty (Vosoughi channel removed; demonstrates the calibration gate has teeth)",
        )
    return NormParams(), "declared-crowding (H3 effect present at declared size)"


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from nudgesim.data.fetch import fetch_liar, pheme_instructions, verify_liar

    if args.corpus in ("liar", "all"):
        print(f"[nudgesim] fetching LIAR into {args.liar_dest} ...")
        try:
            path = fetch_liar(args.liar_dest)
            report = verify_liar(path)
            print(f"[nudgesim] LIAR ok: {report['n_rows']} rows, "
                  f"{len(report['labels'])} labels, files {report['files']}")
        except (RuntimeError, ValueError) as exc:
            print(f"[nudgesim] LIAR FAILED: {exc}")
            return 1
    if args.corpus in ("pheme", "all"):
        print()
        print(pheme_instructions(args.pheme_dest))
    return 0


def _citizen_visibility(library) -> dict[str, object]:
    """How many citizens can see each other, per motif.

    A motif with no citizen-to-citizen edge is one where every citizen sees only
    the Disseminator: no local majority can form among the five, and no citizen
    can observe another paying the correction cost. Both are mechanisms the
    study measures, so a topology source that yields mostly such motifs cannot
    answer the research questions -- and neither the calibration gate nor the
    intervener-reach filter detects it.
    """
    counts: dict[int, int] = {}
    for motif in library.motifs:
        assignment = motif.assign_roles(DEFAULT_SOCIETY.all_ids)
        node_of = {role: node for node, role in assignment.items()}
        citizens = {node_of[c] for c in DEFAULT_SOCIETY.citizen_ids}
        edges = sum(1 for u, v in motif.graph.edges if u in citizens and v in citizens)
        counts[edges] = counts.get(edges, 0) + 1
    total = sum(counts.values()) or 1
    return {
        "citizen_to_citizen_edges_per_motif": dict(sorted(counts.items())),
        "share_with_no_citizen_visibility": round(counts.get(0, 0) / total, 4),
    }


def cmd_check(args: argparse.Namespace) -> int:
    pool, library = _load_data(args)
    payoff = _payoff(args)
    report = {
        "corpora": {
            "claims": pool.provenance,
            "topology": library.provenance,
            "topology_detail": {
                k: v for k, v in library.meta.items()
                if k in ("n_structure_files", "n_threads_parsed",
                         "n_threads_large_enough", "events", "n_threads")
            },
        },
        "manipulation_check_tone": length_match_report(pool.false_claims),
        "claim_pool": pool.summary(),
        "motif_library": library.summary(),
        "intervener_reach_distribution": {
            str(k): v for k, v in library.reach_distribution(DEFAULT_SOCIETY.all_ids).items()
        },
        "citizen_visibility": _citizen_visibility(library),
        "stage_game_reference": stage_game_reference(5, payoff),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["manipulation_check_tone"]["passes"] else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    pool, library = _load_data(args)
    payoff = _payoff(args)
    norms, dgp_label = _norms(args)
    core = library.with_min_reach(1, DEFAULT_SOCIETY.all_ids)
    rng = random.Random(args.seed)

    async def control_episodes(false_side: bool, n: int):
        configs = []
        drawn = pool.stratified_sample(n, rng)
        for i in range(n):
            claim = drawn[i]
            if not false_side:
                placebo = pool.placebo_for(claim)
                if placebo is None:
                    continue
                claim = placebo
            configs.append(
                EpisodeConfig(
                    episode_id=f"calib-{'false' if false_side else 'true'}-{i}",
                    claim=claim,
                    motif=core.sample(rng),
                    timing=Timing.NONE,
                    tone=None,
                    seed=1000 + i,
                    payoff=payoff,
                    backbone=args.backbone,
                    arm="calibration",
                    norms=norms,
                )
            )
        return await asyncio.gather(*(run_episode(c) for c in configs))

    false_results = asyncio.run(control_episodes(True, args.calibration_episodes))
    true_results = asyncio.run(control_episodes(False, args.calibration_episodes))
    report = calibrate(
        false_results, true_results, library.thread_stats, tolerance=ToleranceBand()
    )
    payload = report.to_json()
    payload["dgp_label"] = dgp_label
    payload["claim_provenance"] = pool.provenance
    payload["motif_provenance"] = library.provenance

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if report.passes else 2


def cmd_run(args: argparse.Namespace) -> int:
    pool, library = _load_data(args)
    payoff = _payoff(args)
    norms, dgp_label = _norms(args)
    spec = GridSpec(
        core_seeds=args.core_seeds,
        placebo_seeds=args.placebo_seeds,
        ablation_payoff_seeds=args.ablation_payoff_seeds,
        ablation_ratio_seeds=args.ablation_ratio_seeds,
        scale_seeds=args.scale_seeds,
        horizon=args.horizon,
        include_arms=tuple(args.arms.split(",")),
    )
    configs = expand_grid(spec, pool, library, payoff=payoff, norms=norms, seed=args.seed)
    print(f"[nudgesim] expanded {len(configs)} episodes: {spec.expected_episodes()}", flush=True)

    started = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(1e-9, time.time() - started)
        print(f"[nudgesim] {done}/{total} episodes ({rate:.0f}/s)", flush=True)

    rows, raw = asyncio.run(run_grid(configs, concurrency=args.concurrency, on_progress=progress))

    from nudgesim.oasis import scale_check_backend_note

    manifest = RunManifest(
        run_id=args.run_id or f"run-{int(started)}",
        grid=spec,
        claim_provenance=pool.provenance,
        motif_provenance=library.provenance,
        payoff=payoff,
        norms=norms,
        dgp_label=dgp_label,
        backbone_kind="surrogate" if not args.liar_backbones else "llm",
        scale_backend=scale_check_backend_note(),
    )
    paths = write_run(args.out, manifest, rows, raw)
    print(json.dumps({"episodes": len(rows), "elapsed_s": round(time.time() - started, 1), **paths}, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from analysis.preregistered import run_analysis

    result = run_analysis(args.run, out_dir=args.out or args.run, alpha=args.alpha)
    print(json.dumps(result["headline"], indent=2, default=str))
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    out = Path(args.out)
    for step, fn, sub in (
        ("checks", cmd_check, "checks"),
        ("calibration", cmd_calibrate, "calibration"),
        ("grid", cmd_run, "main"),
    ):
        sub_args = argparse.Namespace(**vars(args))
        sub_args.out = str(out / sub)
        print(f"\n=== {step} ===", flush=True)
        code = fn(sub_args)
        if code != 0 and step == "calibration":
            print("[nudgesim] calibration gate returned NO-GO; stopping before the grid.")
            return code
    analyze_args = argparse.Namespace(**vars(args))
    analyze_args.run = str(out / "main")
    analyze_args.out = str(out / "analysis")
    print("\n=== analysis ===", flush=True)
    return cmd_analyze(analyze_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nudgesim", description=__doc__)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--liar", default=None, help="path to the LIAR release (dir or .tsv)")
    parser.add_argument("--pheme", default=None, help="path to the PHEME-9 release directory")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--backbone", default="surrogate-cautious")
    parser.add_argument("--liar-backbones", action="store_true",
                        help="record the run as LLM-backed rather than surrogate")
    parser.add_argument(
        "--dgp",
        choices=("declared", "null", "strong", "no-novelty"),
        default="declared",
        help="surrogate data-generating process: 'declared' is the literature-"
             "anchored parameterisation, 'strong' inflates the H3 tone channel to a "
             "detectable size (sensitivity check), 'null' sets it to zero "
             "(false-positive check), 'no-novelty' removes the Vosoughi channel "
             "(calibration-gate check)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-data", help="download LIAR; explain PHEME")
    p_fetch.add_argument("--corpus", choices=("liar", "pheme", "all"), default="all")
    p_fetch.add_argument("--liar-dest", default="data/raw/liar")
    p_fetch.add_argument("--pheme-dest", default="data/raw/pheme")
    p_fetch.set_defaults(func=cmd_fetch_data)

    p_check = sub.add_parser("check", help="manipulation checks and design references")
    p_check.add_argument("--out", default="runs/checks")
    p_check.set_defaults(func=cmd_check)

    p_cal = sub.add_parser("calibrate", help="face-validity Go/No-Go gate")
    p_cal.add_argument("--out", default="runs/calibration")
    p_cal.add_argument("--calibration-episodes", type=int, default=60)
    p_cal.set_defaults(func=cmd_calibrate)

    p_run = sub.add_parser("run", help="execute the preregistered grid")
    p_run.add_argument("--out", default="runs/main")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--core-seeds", type=int, default=30)
    p_run.add_argument("--placebo-seeds", type=int, default=10)
    p_run.add_argument("--ablation-payoff-seeds", type=int, default=3)
    p_run.add_argument("--ablation-ratio-seeds", type=int, default=3)
    p_run.add_argument("--scale-seeds", type=int, default=10)
    p_run.add_argument("--arms", default="core,placebo,ablation,scale")
    p_run.add_argument("--concurrency", type=int, default=16)
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="run the preregistered analysis")
    p_an.add_argument("--run", default="runs/main")
    p_an.add_argument("--out", default=None)
    p_an.add_argument("--alpha", type=float, default=0.05)
    p_an.set_defaults(func=cmd_analyze)

    p_rep = sub.add_parser("reproduce", help="checks + calibration + grid + analysis")
    p_rep.add_argument("--out", default="runs/repro")
    p_rep.add_argument("--run-id", default=None)
    p_rep.add_argument("--calibration-episodes", type=int, default=60)
    p_rep.add_argument("--core-seeds", type=int, default=30)
    p_rep.add_argument("--placebo-seeds", type=int, default=10)
    p_rep.add_argument("--ablation-payoff-seeds", type=int, default=3)
    p_rep.add_argument("--ablation-ratio-seeds", type=int, default=3)
    p_rep.add_argument("--scale-seeds", type=int, default=10)
    p_rep.add_argument("--arms", default="core,placebo,ablation,scale")
    p_rep.add_argument("--concurrency", type=int, default=16)
    p_rep.add_argument("--alpha", type=float, default=0.05)
    p_rep.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
