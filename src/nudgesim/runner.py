"""Grid expansion, execution and logging (plan section 5.9).

Monte Carlo protocol:

    one episode = 1 claim x 1 topology x 1 condition x 1 seed

    core grid  2 timing x 2 tone x 4 backbones = 16 cells, + 4 per-model
               controls = 20 cells x 30 seeds                     600 episodes
    placebo    the same 20 cells on matched true claims x 10 seeds 200 episodes
    ablation 1 payoff visibility off, 20 cells x 3 seeds            60 episodes
    ablation 2 beta/gamma in {0.5, 1, 3}, 20 cells x 3 seeds       180 episodes
    scale      N = 50, 2 timing x 2 tone, 1 backbone x 10 seeds     40 episodes
    perturb    20 cells x 2 defector kinds x 8 seeds               320 episodes
                                                                 ------------
                                                                1400 episodes

Claims and topologies are randomised across episodes from a seeded stream;
conditions are assigned by grid. Every episode is written as one JSONL line and
the run is aggregated into a versioned Parquet table.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import random

from nudgesim import __version__
from nudgesim.agents.bounded_rational import NormParams, SURROGATE_PROFILES
from nudgesim.agents.persona import (
    DEFAULT_SOCIETY,
    PROMPT_SUITE_VERSION,
    SocietySpec,
    prompt_suite_fingerprint,
)
from nudgesim.data.claims import ClaimPool
from nudgesim.data.topology import MotifLibrary
from nudgesim.env.episode import EpisodeConfig, EpisodeResult, run_episode
from nudgesim.game.actions import Action
from nudgesim.game.payoff import PayoffParams
from nudgesim.intervention.scheduler import Timing
from nudgesim.intervention.tone import Tone
from nudgesim.metrics.outcomes import compute_metrics
from nudgesim.metrics.reactance import ReactanceScorer
from nudgesim.metrics.traces import KeywordTraceClassifier, trace_composition
from nudgesim.oasis import ScaleCheckSpec, build_scale_library, scale_check_backend_note

DEFAULT_BACKBONES: tuple[str, ...] = tuple(SURROGATE_PROFILES)
ABLATION_RATIOS: tuple[float, ...] = (0.5, 1.0, 3.0)
PERTURBATIONS: tuple[str, ...] = ("amplifier", "free_rider")


@dataclass
class GridSpec:
    """Episode counts per arm. Defaults reproduce the plan's protocol exactly."""

    core_seeds: int = 30
    placebo_seeds: int = 10
    ablation_payoff_seeds: int = 3
    ablation_ratio_seeds: int = 3
    scale_seeds: int = 10
    perturbation_seeds: int = 8
    backbones: tuple[str, ...] = DEFAULT_BACKBONES
    # Maps a backbone name to a backend spec. Empty means the analytic surrogate.
    backend_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    horizon: int = 12
    # The round a perturbed citizen defects: half-way, so the norm has had time
    # to establish and half the episode remains to show whether it holds.
    perturbation_round: int = 6
    min_intervener_reach: int = 1
    trigger_mode: str = "fixed"
    include_arms: tuple[str, ...] = (
        "core", "placebo", "ablation", "scale", "perturbation",
    )

    def expected_episodes(self) -> dict[str, int]:
        cells = 5 * len(self.backbones)  # control + 2 timings x 2 tones, per backbone
        out = {
            "core": cells * self.core_seeds,
            "placebo": cells * self.placebo_seeds,
            "ablation_payoff": cells * self.ablation_payoff_seeds,
            "ablation_ratio": cells * len(ABLATION_RATIOS) * self.ablation_ratio_seeds,
            "scale": 4 * self.scale_seeds,
            "perturbation": cells * len(PERTURBATIONS) * self.perturbation_seeds,
        }
        out = {k: v for k, v in out.items() if _arm_family(k) in self.include_arms}
        out["total"] = sum(out.values())
        return out


def _arm_family(arm: str) -> str:
    return "ablation" if arm.startswith("ablation") else arm


CONDITIONS: tuple[tuple[Timing, Tone | None], ...] = (
    (Timing.NONE, None),
    (Timing.EARLY, Tone.EMPATHETIC),
    (Timing.EARLY, Tone.AGGRESSIVE),
    (Timing.LATE, Tone.EMPATHETIC),
    (Timing.LATE, Tone.AGGRESSIVE),
)


@dataclass
class RunManifest:
    """Everything needed to tell two runs apart. Written next to the logs."""

    run_id: str
    grid: GridSpec
    claim_provenance: str
    motif_provenance: str
    payoff: PayoffParams
    norms: NormParams
    dgp_label: str
    backbone_kind: str
    scale_backend: str
    started_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "nudgesim_version": __version__,
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_at": self.started_at,
            "prompt_suite": PROMPT_SUITE_VERSION,
            "prompt_fingerprint": prompt_suite_fingerprint(True),
            "claim_provenance": self.claim_provenance,
            "motif_provenance": self.motif_provenance,
            "scale_check_backend": self.scale_backend,
            "backbone_kind": self.backbone_kind,
            "dgp_label": self.dgp_label,
            "declared_dgp_parameters": self.norms.as_dict(),
            "payoff_defaults": {
                "beta": self.payoff.beta,
                "gamma": self.payoff.gamma,
                "kappa": self.payoff.kappa,
                "delta": self.payoff.delta,
                "horizon": self.payoff.horizon,
            },
            "grid": {
                "core_seeds": self.grid.core_seeds,
                "placebo_seeds": self.grid.placebo_seeds,
                "ablation_payoff_seeds": self.grid.ablation_payoff_seeds,
                "ablation_ratio_seeds": self.grid.ablation_ratio_seeds,
                "scale_seeds": self.grid.scale_seeds,
                "backbones": list(self.grid.backbones),
                "ablation_ratios": list(ABLATION_RATIOS),
                "min_intervener_reach": self.grid.min_intervener_reach,
                "trigger_mode": self.grid.trigger_mode,
                "expected_episodes": self.grid.expected_episodes(),
            },
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def expand_grid(
    spec: GridSpec,
    pool: ClaimPool,
    library: MotifLibrary,
    *,
    payoff: PayoffParams,
    norms: NormParams,
    seed: int = 0,
) -> list[EpisodeConfig]:
    """Build every episode config for the run, with claims and topologies pinned.

    The claim/topology stream is drawn from one seeded RNG per (arm, cell) so
    that a cell's replications are reproducible independently of how many other
    cells are in the run -- adding an arm never perturbs an existing cell.
    """
    configs: list[EpisodeConfig] = []
    core_library = library.with_min_reach(
        spec.min_intervener_reach, DEFAULT_SOCIETY.all_ids
    )

    def stream(tag: str) -> random.Random:
        return random.Random(f"{seed}|{tag}")

    def add_cell(
        arm: str,
        timing: Timing,
        tone: Tone | None,
        backbone: str,
        n_seeds: int,
        *,
        placebo: bool,
        payoff_params: PayoffParams,
        payoff_visible: bool,
        tag_extra: str = "",
        perturbation: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        tag = f"{arm}|{timing.value}|{tone.value if tone else 'none'}|{backbone}{tag_extra}"
        rng = stream(tag)
        # Draw the whole cell's claims in one stratified call, so each cell is
        # balanced across severity strata rather than balanced only in
        # expectation across cells.
        drawn = pool.stratified_sample(n_seeds, rng)
        for s in range(n_seeds):
            claim = drawn[s]
            if placebo:
                placebo_claim = pool.placebo_for(claim)
                if placebo_claim is None:
                    continue
                claim = placebo_claim
            motif = core_library.sample(rng)
            configs.append(
                EpisodeConfig(
                    episode_id=f"{tag}|s{s}".replace("|", "__"),
                    claim=claim,
                    motif=motif,
                    timing=timing,
                    tone=tone,
                    seed=hash((tag, s)) % (2**31),
                    payoff=payoff_params,
                    backbone=backbone,
                    payoff_visible=payoff_visible,
                    arm=arm,
                    perturbation=perturbation,
                    perturbation_round=spec.perturbation_round,
                    trigger_mode=spec.trigger_mode,
                    norms=norms,
                    backend_spec=spec.backend_specs.get(backbone),
                    meta={"cell": tag, "replication": s, **(meta or {})},
                )
            )

    for backbone in spec.backbones:
        for timing, tone in CONDITIONS:
            if "core" in spec.include_arms:
                add_cell("core", timing, tone, backbone, spec.core_seeds,
                         placebo=False, payoff_params=payoff, payoff_visible=True)
            if "placebo" in spec.include_arms:
                add_cell("placebo", timing, tone, backbone, spec.placebo_seeds,
                         placebo=True, payoff_params=payoff, payoff_visible=True)
            if "ablation" in spec.include_arms:
                add_cell("ablation_payoff", timing, tone, backbone,
                         spec.ablation_payoff_seeds, placebo=False,
                         payoff_params=payoff, payoff_visible=False,
                         meta={"ablation": "payoff_visibility"})
                for ratio in ABLATION_RATIOS:
                    add_cell("ablation_ratio", timing, tone, backbone,
                             spec.ablation_ratio_seeds, placebo=False,
                             payoff_params=payoff.with_ratio(ratio),
                             payoff_visible=True, tag_extra=f"|r{ratio}",
                             meta={"ablation": "incentive_ratio", "beta_gamma_ratio": ratio})
            if "perturbation" in spec.include_arms:
                for kind in PERTURBATIONS:
                    add_cell("perturbation", timing, tone, backbone,
                             spec.perturbation_seeds, placebo=False,
                             payoff_params=payoff, payoff_visible=True,
                             tag_extra=f"|{kind}", perturbation=kind,
                             meta={"perturbation_kind": kind})

    if "scale" in spec.include_arms:
        scale_spec = ScaleCheckSpec(seeds=spec.scale_seeds, horizon=spec.horizon)
        scale_library = build_scale_library(scale_spec, seed=seed + 7)
        for timing, tone in CONDITIONS:
            if timing is Timing.NONE:
                continue
            tag = f"scale|{timing.value}|{tone.value if tone else 'none'}"
            rng = stream(tag)
            drawn = pool.stratified_sample(scale_spec.seeds, rng)
            for s in range(scale_spec.seeds):
                claim = drawn[s]
                configs.append(
                    EpisodeConfig(
                        episode_id=f"{tag}|s{s}".replace("|", "__"),
                        claim=claim,
                        motif=scale_library.sample(rng),
                        timing=timing,
                        tone=tone,
                        seed=hash((tag, s)) % (2**31),
                        payoff=payoff,
                        backbone=scale_spec.backbone,
                        arm="scale",
                        society=scale_spec.society,
                        backend_spec=spec.backend_specs.get(scale_spec.backbone),
                        trigger_mode=spec.trigger_mode,
                        norms=norms,
                        meta={"cell": tag, "replication": s, "n_agents": scale_spec.n_agents},
                    )
                )

    return configs


def summarise_episode(result: EpisodeResult) -> dict[str, Any]:
    """Fold one episode into a single analysis row."""
    cfg = result.config
    entry = result.schedule.realised_entry_round
    metrics = compute_metrics(
        result.records,
        episode_id=cfg.episode_id,
        claim=cfg.claim,
        horizon=cfg.payoff.horizon,
        entry_round=entry,
        welfare=result.ledger.welfare_decomposition(),
        probes=result.probes,
        citizen_ids=cfg.society.citizen_ids,
    )
    citizens = set(cfg.society.citizen_ids)
    citizen_records = [r for r in result.records if r.agent_id in citizens]

    # Rates over the citizens who are NOT the perturbation seat, computed for
    # every arm so perturbed and unperturbed episodes are measured the same way.
    # A free-riding defector stops challenging by construction; what the
    # robustness question actually asks is whether its peers change.
    peers = [r for r in citizen_records if r.agent_id != cfg.perturbed_agent]
    n_peers = len(peers) or 1
    epc_peers = sum(1 for r in peers if r.action is Action.CHALLENGE) / n_peers
    fpr_peers = sum(1 for r in peers if r.action.is_propagating) / n_peers
    classifier = KeywordTraceClassifier()
    composition = trace_composition(
        [classifier.classify(r.reasoning_trace).label for r in citizen_records]
    )
    reactance = ReactanceScorer().score(
        [
            r
            for r in citizen_records
            if r.round_index >= (entry if entry is not None else 0)
        ]
    )
    episode_json = result.to_json()

    row: dict[str, Any] = {
        **metrics.to_row(),
        "arm": cfg.arm,
        "arm_family": _arm_family(cfg.arm),
        "timing": cfg.timing.value,
        "tone": cfg.tone.value if cfg.tone else "none",
        "backbone": cfg.backbone,
        "backbone_kind": "llm" if cfg.backend_spec else "surrogate",
        "perturbation": cfg.perturbation or "none",
        "epc_peers": epc_peers,
        "fpr_peers": fpr_peers,
        "treated": cfg.timing is not Timing.NONE,
        "payoff_visible": cfg.payoff_visible,
        "beta_gamma_ratio": cfg.payoff.engagement_accuracy_ratio,
        "seed": cfg.seed,
        "replication": cfg.meta.get("replication"),
        "cell": cfg.meta.get("cell"),
        "claim_id": cfg.claim.claim_id,
        "claim_veracity": cfg.claim.veracity.value,
        "claim_severity": cfg.claim.severity,
        "claim_topic": cfg.claim.topic,
        "motif_id": cfg.motif.motif_id,
        "n_agents": cfg.society.size,
        "intervener_reach": episode_json["intervener_reach"],
        "motif_depth": episode_json["motif_stats"]["depth"],
        "motif_max_breadth": episode_json["motif_stats"]["max_breadth"],
        "motif_structural_virality": episode_json["motif_stats"]["structural_virality"],
        "entry_round": entry,
        "intervener_challenged": episode_json["intervener_challenged"],
        "terminated_early": result.guards.terminated_early,
        "repetition_rate": result.guards.repetition_rate,
        "persona_break_rate": result.guards.persona_break_rate,
        "repair_rate": result.repair_rate,
        "hostility": reactance.hostility,
        "defensiveness": reactance.defensiveness,
        "net_reactance": reactance.net_reactance,
        "wall_time_s": result.wall_time_s,
    }
    row.update({f"trace_{k}": v for k, v in composition.items()})
    row["_round_rates"] = metrics.round_rates
    return row


async def run_grid(
    configs: Sequence[EpisodeConfig],
    *,
    concurrency: int = 8,
    on_progress=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute every episode, returning (analysis rows, raw episode records)."""
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    done = 0

    async def one(config: EpisodeConfig) -> None:
        nonlocal done
        async with semaphore:
            result = await run_episode(config)
        rows.append(summarise_episode(result))
        raw.append(result.to_json())
        done += 1
        if on_progress and done % 50 == 0:
            on_progress(done, len(configs))

    await asyncio.gather(*(one(c) for c in configs))
    return rows, raw


def write_run(
    out_dir: str | Path,
    manifest: RunManifest,
    rows: Iterable[dict[str, Any]],
    raw: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Write JSONL episode logs plus a versioned Parquet results table."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    jsonl_path = out / "episodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in raw:
            fh.write(json.dumps(record, default=str) + "\n")

    frame = pd.DataFrame(rows).sort_values(["arm", "cell", "replication"]).reset_index(drop=True)

    # Round-level table, kept separate so the episode table stays one row per
    # episode: the trajectory figures and the half-life survival analysis both
    # need per-round rates, and re-deriving them from the JSONL on every plot is
    # needlessly slow.
    round_records: list[dict[str, Any]] = []
    for row in rows:
        for entry in row.get("_round_rates", []):
            round_records.append(
                {
                    "episode_id": row["episode_id"],
                    "arm": row["arm"],
                    "timing": row["timing"],
                    "tone": row["tone"],
                    "backbone": row["backbone"],
                    "entry_round": row["entry_round"],
                    "terminated_early": row["terminated_early"],
                    "round": int(entry["round"]),
                    "n": entry["n"],
                    "fpr": entry["fpr"],
                    "epc": entry["epc"],
                }
            )
    rounds_frame = pd.DataFrame(round_records)
    rounds_frame.to_parquet(out / "rounds.parquet", index=False)
    frame = frame.drop(columns=["_round_rates"])

    parquet_path = out / "results.parquet"
    frame.to_parquet(parquet_path, index=False)
    csv_path = out / "results.csv"
    frame.to_csv(csv_path, index=False)

    manifest_path = out / "manifest.json"
    payload = manifest.to_json()
    payload["n_episodes_written"] = len(rows)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "episodes_jsonl": str(jsonl_path),
        "rounds_parquet": str(out / "rounds.parquet"),
        "results_parquet": str(parquet_path),
        "results_csv": str(csv_path),
        "manifest": str(manifest_path),
    }
