"""Face-validity calibration gate (plan section 5.7).

Before the main grid runs, the no-intervener control must reproduce empirical
cascade behaviour within a pre-declared tolerance band:

  1. Cascade shape -- depth, breadth and structural virality of the simulated
     diffusion tree, compared against the source threads the topologies were
     drawn from.
  2. Diffusion asymmetry -- the false-versus-true asymmetry reported for real
     platforms (Vosoughi, Roy & Aral, 2018) reproduced in sign.

Shape is compared on *size-normalised* statistics. A 7-node induced motif cannot
reproduce the absolute depth of a 25-node thread, so comparing raw depths would
fail the gate for a reason that has nothing to do with agent behaviour. What can
be asked -- and what the gate asks -- is whether the cascade the agents actually
produce has the same shape, relative to the structure available to it, as the
real thread did.

If the baseline cannot reproduce known cascade statistics, no causal claim about
interventions inside it is worth making, so this is a hard Go/No-Go, not a
robustness appendix.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx

from nudgesim.data.topology import CascadeStats, Motif, cascade_stats
from nudgesim.env.episode import EpisodeResult
from nudgesim.game.actions import Action


@dataclass(frozen=True)
class ToleranceBand:
    """Pre-declared tolerances. Frozen before any control baseline is run."""

    depth_ratio_tol: float = 0.35
    breadth_ratio_tol: float = 0.35
    virality_ratio_tol: float = 0.40
    min_asymmetry_margin: float = 0.02  # false must exceed true by at least this

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def diffusion_tree(result: EpisodeResult) -> tuple[nx.DiGraph, str]:
    """Reconstruct who propagated the claim from whom.

    An agent joins the cascade the first time it propagates *after* one of its
    neighbours has already propagated -- that neighbour becomes its parent, ties
    broken toward whoever propagated earliest and sits closest to the source.

    Agents that propagate spontaneously in round 0, before anyone could have
    influenced them, are not yet part of the cascade; they join later if and
    when they propagate downstream of someone. Keying this off each agent's
    *first* propagation instead would permanently exclude every early
    independent poster, which shrinks measured cascades exactly in the arms
    where propagation starts fastest -- inverting the diffusion asymmetry the
    gate is trying to measure.
    """
    cfg = result.config
    society = cfg.society
    graph = cfg.motif.graph
    assignment = result.role_assignment  # role -> node
    node_of = assignment
    role_of = {node: role for role, node in assignment.items()}

    # All propagation events, in round order.
    events: list[tuple[int, str]] = sorted(
        (r.round_index, r.agent_id)
        for r in result.records
        if r.action.is_propagating
    )
    first_prop: dict[str, int] = {}
    for round_index, agent in events:
        first_prop.setdefault(agent, round_index)

    root = society.disseminator_id
    tree = nx.DiGraph()
    tree.add_node(root)
    if root not in first_prop:
        return tree, root

    hops = nx.single_source_shortest_path_length(graph, node_of[root])
    for round_index, agent in events:
        if agent in tree:
            continue
        neighbours = [role_of[n] for n in graph.neighbors(node_of[agent])]
        candidates = [
            n
            for n in neighbours
            if n in tree and first_prop.get(n, 1 << 30) < round_index
        ]
        if not candidates:
            continue
        parent = min(
            candidates, key=lambda n: (first_prop[n], hops.get(node_of[n], 99), n)
        )
        tree.add_edge(parent, agent)
    return tree, root


def normalised_shape(stats: CascadeStats) -> dict[str, float]:
    """Shape statistics divided by cascade size, so motifs and threads compare."""
    size = max(1, stats.size)
    return {
        "depth_ratio": stats.depth / size,
        "breadth_ratio": stats.max_breadth / size,
        "virality_ratio": stats.structural_virality / size,
    }


@dataclass
class CalibrationReport:
    n_false_episodes: int
    n_true_episodes: int
    simulated_shape: dict[str, float]
    reference_shape: dict[str, float]
    shape_deltas: dict[str, float]
    shape_passes: dict[str, bool]
    asymmetry: dict[str, float]
    asymmetry_passes: bool
    tolerance: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return all(self.shape_passes.values()) and self.asymmetry_passes

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": "face-validity calibration (plan section 5.7)",
            "verdict": "GO" if self.passes else "NO-GO",
            "n_false_episodes": self.n_false_episodes,
            "n_true_episodes": self.n_true_episodes,
            "simulated_shape": {k: round(v, 6) for k, v in self.simulated_shape.items()},
            "reference_shape": {k: round(v, 6) for k, v in self.reference_shape.items()},
            "shape_deltas": {k: round(v, 6) for k, v in self.shape_deltas.items()},
            "shape_passes": self.shape_passes,
            "asymmetry": {k: round(v, 6) for k, v in self.asymmetry.items()},
            "asymmetry_passes": self.asymmetry_passes,
            "tolerance": self.tolerance,
            "notes": self.notes,
        }


def calibrate(
    false_control: Sequence[EpisodeResult],
    true_control: Sequence[EpisodeResult],
    reference_threads: Sequence[CascadeStats],
    *,
    tolerance: ToleranceBand | None = None,
) -> CalibrationReport:
    tol = tolerance or ToleranceBand()

    def shape_of(results: Sequence[EpisodeResult]) -> dict[str, float]:
        rows = []
        for result in results:
            tree, root = diffusion_tree(result)
            if tree.number_of_nodes() < 2:
                continue
            rows.append(normalised_shape(cascade_stats(tree, root)))
        if not rows:
            return {"depth_ratio": 0.0, "breadth_ratio": 0.0, "virality_ratio": 0.0}
        return {k: statistics.fmean([r[k] for r in rows]) for k in rows[0]}

    simulated = shape_of(false_control)
    reference = (
        {
            k: statistics.fmean([normalised_shape(s)[k] for s in reference_threads])
            for k in ("depth_ratio", "breadth_ratio", "virality_ratio")
        }
        if reference_threads
        else {"depth_ratio": 0.0, "breadth_ratio": 0.0, "virality_ratio": 0.0}
    )

    deltas = {k: simulated[k] - reference[k] for k in simulated}
    passes = {
        "depth_ratio": abs(deltas["depth_ratio"]) <= tol.depth_ratio_tol,
        "breadth_ratio": abs(deltas["breadth_ratio"]) <= tol.breadth_ratio_tol,
        "virality_ratio": abs(deltas["virality_ratio"]) <= tol.virality_ratio_tol,
    }

    def propagation_rate(results: Sequence[EpisodeResult]) -> float:
        total = hits = 0
        for result in results:
            citizens = set(result.config.society.citizen_ids)
            for record in result.records:
                if record.agent_id in citizens:
                    total += 1
                    hits += int(record.action.is_propagating)
        return hits / total if total else 0.0

    def cascade_size(results: Sequence[EpisodeResult]) -> float:
        sizes = []
        for result in results:
            tree, _ = diffusion_tree(result)
            sizes.append(tree.number_of_nodes())
        return statistics.fmean(sizes) if sizes else 0.0

    false_rate, true_rate = propagation_rate(false_control), propagation_rate(true_control)
    false_size, true_size = cascade_size(false_control), cascade_size(true_control)
    asymmetry = {
        "false_propagation_rate": false_rate,
        "true_propagation_rate": true_rate,
        "propagation_margin": false_rate - true_rate,
        "false_cascade_size": false_size,
        "true_cascade_size": true_size,
        "cascade_size_margin": false_size - true_size,
    }
    asymmetry_ok = (false_rate - true_rate) >= tol.min_asymmetry_margin

    notes: list[str] = []
    if not asymmetry_ok:
        notes.append(
            "False claims did not out-diffuse true claims by the declared margin; "
            "remediation path (plan section 12) is to adjust neighbourhood "
            "visibility and disseminator persistence, then re-run."
        )
    if not all(passes.values()):
        notes.append(
            "Simulated cascade shape fell outside the tolerance band; the claim "
            "narrows to a within-simulation mechanism claim and network-level "
            "language is dropped."
        )

    return CalibrationReport(
        n_false_episodes=len(false_control),
        n_true_episodes=len(true_control),
        simulated_shape=simulated,
        reference_shape=reference,
        shape_deltas=deltas,
        shape_passes=passes,
        asymmetry=asymmetry,
        asymmetry_passes=asymmetry_ok,
        tolerance=tol.as_dict(),
        notes=notes,
    )
