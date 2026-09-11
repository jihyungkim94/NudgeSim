"""PHEME topology source and calibration target (plan sections 5.2 and 5.7).

Each PHEME conversation thread ships a ``structure.json`` encoding the reply
tree of a real rumour cascade. We parse those trees, compute cascade statistics,
and sample connected 7-node motifs in which the cascade root becomes the
Disseminator and downstream nodes become citizens.

Only *derived topology* is used and released -- never tweet content (plan
section 6, licensing).

As with the claim pool, an offline surrogate generator is provided so the engine
is runnable without the corpus; motif libraries carry a ``provenance`` string
that propagates into every episode record.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx


@dataclass(frozen=True)
class CascadeStats:
    """Structural summary of one reply tree (or of a simulated cascade)."""

    size: int
    depth: int
    max_breadth: int
    mean_branching: float
    structural_virality: float

    def as_dict(self) -> dict[str, float]:
        return {
            "size": float(self.size),
            "depth": float(self.depth),
            "max_breadth": float(self.max_breadth),
            "mean_branching": self.mean_branching,
            "structural_virality": self.structural_virality,
        }


def cascade_stats(tree: nx.DiGraph, root: Any) -> CascadeStats:
    """Compute the four statistics used by the calibration gate.

    ``structural_virality`` is the Wiener index normalisation of Goel et al.:
    the mean pairwise distance in the undirected cascade, which separates a
    broadcast (low) from a deep person-to-person chain (high).
    """
    depths = nx.single_source_shortest_path_length(tree, root)
    size = tree.number_of_nodes()
    depth = max(depths.values()) if depths else 0
    breadth_by_level: dict[int, int] = {}
    for d in depths.values():
        breadth_by_level[d] = breadth_by_level.get(d, 0) + 1
    max_breadth = max(breadth_by_level.values()) if breadth_by_level else 0
    internal = [n for n in tree.nodes if tree.out_degree(n) > 0]
    mean_branching = (
        statistics.fmean([tree.out_degree(n) for n in internal]) if internal else 0.0
    )
    if size > 1:
        undirected = tree.to_undirected()
        total, pairs = 0, 0
        for _, lengths in nx.all_pairs_shortest_path_length(undirected):
            total += sum(lengths.values())
            pairs += len(lengths) - 1
        virality = total / pairs if pairs else 0.0
    else:
        virality = 0.0
    return CascadeStats(
        size=size,
        depth=depth,
        max_breadth=max_breadth,
        mean_branching=round(mean_branching, 6),
        structural_virality=round(virality, 6),
    )


@dataclass
class Motif:
    """A connected 7-node subgraph used as one episode's visibility network."""

    motif_id: str
    graph: nx.Graph
    root: str
    source_thread: str = ""
    source_veracity: str = "unverified"
    provenance: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.graph.number_of_nodes()

    def neighbours(self, node: str) -> list[str]:
        return sorted(self.graph.neighbors(node))

    def assign_roles(
        self,
        roles: Sequence[str],
        *,
        root_role: str = "Disseminator",
        hub_role: str | None = "DevilsAdvocate",
        placement: str = "hub",
    ) -> dict[str, str]:
        """Map agent roles onto motif nodes (plan section 5.2).

        The Disseminator takes the cascade root. Where the intervener sits is a
        substantive design choice, not a detail: a Devil's Advocate parked on a
        leaf can barely be seen, so plain BFS order would silently weaken the
        treatment. Default ``placement="hub"`` puts it on the highest-degree
        non-root node -- the position a platform-embedded intervener would
        occupy. ``"peripheral"`` (lowest degree) and ``"bfs"`` are retained as
        robustness variants.
        """
        roles = list(roles)
        if len(roles) != self.size:
            raise ValueError(f"need {self.size} roles for motif {self.motif_id}, got {len(roles)}")
        if placement not in ("hub", "peripheral", "bfs"):
            raise ValueError(f"unknown placement {placement!r}")

        bfs_order = [self.root] + [n for n in nx.bfs_tree(self.graph, self.root) if n != self.root]
        assignment: dict[str, str] = {}
        free_nodes = list(bfs_order)
        free_roles = list(roles)

        if root_role in free_roles:
            assignment[self.root] = root_role
            free_nodes.remove(self.root)
            free_roles.remove(root_role)

        if hub_role is not None and hub_role in free_roles and placement != "bfs":
            # Deterministic tie-break on node id keeps the assignment stable
            # across runs for a given motif.
            key = (lambda n: (-self.graph.degree(n), n)) if placement == "hub" else (
                lambda n: (self.graph.degree(n), n)
            )
            pick = min(free_nodes, key=key)
            assignment[pick] = hub_role
            free_nodes.remove(pick)
            free_roles.remove(hub_role)

        for node, role in zip(free_nodes, free_roles):
            assignment[node] = role
        return assignment

    def intervener_reach(
        self,
        roles: Sequence[str],
        *,
        intervener_role: str = "DevilsAdvocate",
        citizen_roles: Sequence[str] = ("A1", "A2", "B1", "B2", "C"),
        placement: str = "hub",
    ) -> int:
        """How many citizens the intervener is adjacent to, hence can reach.

        The Devil's Advocate is a peer in a network, not a platform banner: it
        can only be seen by its neighbours. Reach is therefore a property of the
        sampled topology and a first-class moderator of any intervention effect,
        so it is logged on every episode and entered as a covariate in the
        analysis. Motifs with reach 0 are degenerate -- the treatment is never
        delivered -- and are excluded by the sampling constraint in
        ``MotifLibrary.with_min_reach``.
        """
        assignment = self.assign_roles(
            roles, hub_role=intervener_role, placement=placement
        )
        node_of = {role: node for node, role in assignment.items()}
        if intervener_role not in node_of:
            return 0
        neighbours = set(self.graph.neighbors(node_of[intervener_role]))
        citizen_nodes = {node_of[c] for c in citizen_roles if c in node_of}
        return len(neighbours & citizen_nodes)

    def to_json(self) -> dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "nodes": sorted(self.graph.nodes),
            "edges": sorted(tuple(sorted(e)) for e in self.graph.edges),
            "root": self.root,
            "source_thread": self.source_thread,
            "source_veracity": self.source_veracity,
            "provenance": self.provenance,
            "meta": self.meta,
            "stats": cascade_stats(nx.bfs_tree(self.graph, self.root), self.root).as_dict(),
        }


@dataclass
class MotifLibrary:
    """Motifs plus the cascade statistics of the threads they came from."""

    motifs: list[Motif]
    provenance: str
    thread_stats: list[CascadeStats] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.motifs)

    def sample(self, rng: random.Random) -> Motif:
        return self.motifs[rng.randrange(len(self.motifs))]

    def with_min_reach(
        self,
        min_reach: int,
        roles: Sequence[str],
        *,
        intervener_role: str = "DevilsAdvocate",
        citizen_roles: Sequence[str] | None = None,
        placement: str = "hub",
    ) -> "MotifLibrary":
        """Preregistered sampling constraint on intervener reach.

        A motif where the Devil's Advocate is adjacent to no citizen cannot
        deliver the treatment at all; including such episodes would dilute every
        cell with undelivered interventions rather than measuring a weaker
        effect. The constraint is declared up front and the number of motifs it
        removes is reported.
        """
        citizens = tuple(
            citizen_roles
            if citizen_roles is not None
            else [r for r in roles if r not in (intervener_role, roles[0])]
        )
        kept = [
            m
            for m in self.motifs
            if m.intervener_reach(
                roles,
                intervener_role=intervener_role,
                citizen_roles=citizens,
                placement=placement,
            )
            >= min_reach
        ]
        if not kept:
            raise ValueError(f"no motif reaches {min_reach} citizens")
        return MotifLibrary(
            motifs=kept,
            provenance=self.provenance,
            thread_stats=self.thread_stats,
            meta={
                **self.meta,
                "min_intervener_reach": min_reach,
                "n_motifs_before_reach_filter": len(self.motifs),
                "n_motifs_dropped_by_reach_filter": len(self.motifs) - len(kept),
            },
        )

    def reach_distribution(
        self,
        roles: Sequence[str],
        *,
        intervener_role: str = "DevilsAdvocate",
        citizen_roles: Sequence[str] | None = None,
        placement: str = "hub",
    ) -> dict[int, int]:
        citizens = tuple(
            citizen_roles
            if citizen_roles is not None
            else [r for r in roles if r not in (intervener_role, roles[0])]
        )
        out: dict[int, int] = {}
        for m in self.motifs:
            r = m.intervener_reach(
                roles,
                intervener_role=intervener_role,
                citizen_roles=citizens,
                placement=placement,
            )
            out[r] = out.get(r, 0) + 1
        return dict(sorted(out.items()))

    def reference_stats(self) -> dict[str, float]:
        """Mean of the source-thread statistics -- the calibration target."""
        if not self.thread_stats:
            return {}
        keys = self.thread_stats[0].as_dict().keys()
        return {
            k: statistics.fmean([s.as_dict()[k] for s in self.thread_stats]) for k in keys
        }

    def summary(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "n_motifs": len(self.motifs),
            "n_source_threads": len(self.thread_stats),
            "reference_stats": self.reference_stats(),
            **self.meta,
        }

    # ----------------------------------------------------------------- loaders

    @classmethod
    def from_pheme(
        cls,
        root_dir: str | Path,
        *,
        motif_size: int = 7,
        max_motifs: int = 200,
        seed: int = 0,
        sibling_window: int = 1,
    ) -> "MotifLibrary":
        """Parse a PHEME-9 release directory into motifs and thread statistics.

        Expected layout (the figshare release)::

            <root>/<event>-all-rnr-threads/{rumours,non-rumours}/<thread>/structure.json

        Only ``structure.json`` and ``annotation.json`` are read. Tweet bodies
        under ``source-tweets/`` and ``reactions/`` are never opened, and tweet
        ids are replaced by positional labels during motif extraction, so only
        derived topology reaches any artefact (plan section 6, licensing).

        Two things this deliberately does not do:

        * It does not stop parsing at ``max_motifs``. Thread statistics are the
          calibration gate's reference distribution (section 5.7), so they are
          computed over every thread in the release; the cap applies only to how
          many motifs are kept for sampling.
        * It does not take motifs in directory order. ``rglob`` yields threads
          sorted by path, so the first 200 would all come from whichever event
          sorts first -- the motif library would silently describe one news
          event rather than nine. Threads are shuffled from the run seed before
          the cap is applied.
        """
        root_dir = Path(root_dir)
        structures = sorted(root_dir.rglob("structure.json"))
        if not structures:
            raise FileNotFoundError(f"no structure.json under {root_dir}")

        rng = random.Random(seed)
        order = list(structures)
        rng.shuffle(order)

        stats: list[CascadeStats] = []
        candidates: list[tuple[Path, nx.DiGraph, Any]] = []
        events: set[str] = set()

        for path in order:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                # The release contains empty, truncated and non-UTF-8 files. One
                # bad thread must not abort a 6,000-thread parse.
                continue
            tree, tree_root = _tree_from_structure(raw)
            if tree is None or tree.number_of_nodes() < 2:
                continue
            stats.append(cascade_stats(tree, tree_root))
            events.add(_pheme_event(path))
            if tree.number_of_nodes() >= motif_size:
                candidates.append((path, tree, tree_root))

        motifs: list[Motif] = []
        for path, tree, tree_root in candidates:
            if len(motifs) >= max_motifs:
                break
            motif = _sample_motif(
                tree,
                tree_root,
                motif_size,
                rng,
                motif_id=f"pheme-{path.parent.name}",
                source_thread=path.parent.name,
                source_veracity=_pheme_veracity(path),
                provenance="PHEME-9 (Kochkina et al., 2018), derived topology only",
                sibling_window=sibling_window,
            )
            if motif is not None:
                motif.meta["event"] = _pheme_event(path)
                motifs.append(motif)

        if motif_size == 7:
            # The hand-specified motif is defined at the core study's size; a
            # scaled library has no counterpart and must not silently inherit a
            # 7-node one.
            motifs.append(canonical_motif())
        return cls(
            motifs=motifs,
            provenance="PHEME-9 (Kochkina et al., 2018), derived topology only",
            thread_stats=stats,
            meta={
                "n_structure_files": len(structures),
                "n_threads_parsed": len(stats),
                "n_threads_large_enough": len(candidates),
                "events": sorted(events),
                "motif_size": motif_size,
                "max_motifs": max_motifs,
                "sibling_window": sibling_window,
            },
        )

    @classmethod
    def synthetic(
        cls,
        *,
        n_threads: int = 120,
        motif_size: int = 7,
        seed: int = 0,
        attachment_bias: float = 0.6,
        depth_bias: float = 0.45,
        sibling_window: int = 1,
    ) -> "MotifLibrary":
        """Surrogate reply-tree generator for offline runs.

        Trees grow by a preferential-attachment reply process: a new reply
        attaches to an existing node with probability proportional to
        ``(in-degree + attachment_bias)``, tilted toward recent nodes by
        ``depth_bias`` so that both broadcast-shaped and chain-shaped cascades
        occur. This reproduces the *shape family* of rumour cascades; it is not
        PHEME and is labelled as such.
        """
        rng = random.Random(seed)
        motifs: list[Motif] = []
        stats: list[CascadeStats] = []
        for t in range(n_threads):
            size = rng.randint(motif_size + 1, motif_size * 6)
            tree = nx.DiGraph()
            tree.add_node("n0")
            recency: list[str] = ["n0"]
            for i in range(1, size):
                weights = []
                for j, node in enumerate(recency):
                    deg = tree.out_degree(node)
                    recency_w = (j + 1) / len(recency)
                    weights.append((deg + attachment_bias) * (recency_w ** depth_bias))
                parent = rng.choices(recency, weights=weights, k=1)[0]
                child = f"n{i}"
                tree.add_edge(parent, child)
                recency.append(child)
            stats.append(cascade_stats(tree, "n0"))
            motif = _sample_motif(
                tree,
                "n0",
                motif_size,
                rng,
                motif_id=f"syn-{t:03d}",
                source_thread=f"synthetic-{t:03d}",
                source_veracity="rumour" if t % 2 == 0 else "non-rumour",
                provenance="SYNTHETIC-SURROGATE topology (not PHEME)",
                sibling_window=sibling_window,
            )
            if motif is not None:
                motifs.append(motif)

        if motif_size == 7:
            motifs.append(canonical_motif())
        return cls(
            motifs=motifs,
            provenance="SYNTHETIC-SURROGATE topology (no PHEME data present; not a PHEME result)",
            thread_stats=stats,
            meta={"n_threads": n_threads, "motif_size": motif_size, "seed": seed,
                  "sibling_window": sibling_window},
        )


def _pheme_event(path: Path) -> str:
    """Event name from a thread path, e.g. 'charliehebdo-all-rnr-threads'."""
    for part in path.parts:
        if part.endswith("-all-rnr-threads"):
            return part.replace("-all-rnr-threads", "")
    # Some redistributions flatten the '-all-rnr-threads' suffix away; fall back
    # to the directory two levels above the thread (…/<event>/rumours/<thread>).
    parts = path.parts
    for i, part in enumerate(parts):
        if part in ("rumours", "non-rumours") and i > 0:
            return parts[i - 1]
    return "unknown"


def _pheme_veracity(path: Path) -> str:
    """Thread veracity from annotation.json, falling back to the folder.

    PHEME stores no veracity string. The real annotations come in two shapes:

        non-rumour: {"is_rumour": "nonrumour"}
        rumour:     {"is_rumour": "rumour", "misinformation": 0, "true": 1, ...}

    ``is_rumour`` has to be checked first: a non-rumour annotation carries
    neither flag, so reading the flags alone labels every non-rumour thread
    "unverified". Flag values are integers in the release, not strings.
    A rumour with neither flag set is genuinely unverified.
    """
    annotation = path.parent / "annotation.json"
    if annotation.exists():
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict) and payload:
            if str(payload.get("is_rumour", "")).strip().lower() in ("nonrumour", "non-rumour"):
                return "non-rumour"
            if str(payload.get("true", 0)).strip() == "1":
                return "true"
            if str(payload.get("misinformation", 0)).strip() == "1":
                return "false"
            return "unverified"
    if "non-rumours" in path.parts:
        return "non-rumour"
    if "rumours" in path.parts:
        return "unverified"
    return "n/a"


def _tree_from_structure(raw: Any) -> tuple[nx.DiGraph | None, Any]:
    """Convert PHEME's nested ``structure.json`` dict into a DiGraph."""
    if not isinstance(raw, dict) or not raw:
        return None, None
    tree = nx.DiGraph()
    roots = list(raw.keys())
    root = roots[0]

    def walk(node_id: Any, children: Any) -> None:
        tree.add_node(node_id)
        if not isinstance(children, dict):
            return
        for child_id, grandchildren in children.items():
            tree.add_edge(node_id, child_id)
            walk(child_id, grandchildren)

    for node_id in roots:
        walk(node_id, raw[node_id])
    if not nx.is_weakly_connected(tree):
        largest = max(nx.weakly_connected_components(tree), key=len)
        tree = tree.subgraph(largest).copy()
        root = next((n for n in tree.nodes if tree.in_degree(n) == 0), next(iter(tree.nodes)))
    return tree, root


def _display_order(node: Any) -> tuple[int, str]:
    """Thread display order. Tweet ids are snowflake ids, so numeric order is
    chronological; anything non-numeric falls back to lexicographic."""
    text = str(node)
    return (0, f"{int(text):030d}") if text.isdigit() else (1, text)


def _visibility_graph(
    tree: nx.DiGraph, nodes: Sequence[Any], sibling_window: int
) -> nx.Graph:
    """Who can see whom, given who replied to whom.

    ``structure.json`` is a reply tree, not a visibility graph: it records that
    B answered A, not that B and C -- who both answered A -- can read each
    other. Taking the reply tree as the visibility graph is what makes real
    PHEME motifs unusable, because real cascades are broadcast stars: a source
    tweet with many direct replies. Rooted at the disseminator, such a motif
    gives every citizen exactly one neighbour (the disseminator), so no local
    majority can form and no citizen can watch another pay the correction cost.

    Connecting *all* siblings is the opposite failure. In a star every reply is
    a sibling of every other, so the motif becomes the complete graph: everyone
    sees everyone, "local" majority becomes global, and the network structure
    the study manipulates disappears. Measured on PHEME-9, that turns 69% of
    motifs into K7.

    A thread view shows neither extreme. It is ranked and truncated, so a reader
    sees the source plus the replies near their own. ``sibling_window`` is that
    bounded attention: each reply is linked to the ``w`` siblings that follow it
    in display order. At w = 1 on PHEME-9 no motif is a star, none is complete,
    and mean density is 0.50 -- sparse enough to keep locality, dense enough for
    a majority to form.
    """
    graph = tree.to_undirected().subgraph(nodes).copy()
    if sibling_window <= 0:
        return graph

    siblings: dict[Any, list[Any]] = {}
    present = set(nodes)
    for node in nodes:
        for parent in tree.predecessors(node):
            siblings.setdefault(parent, []).append(node)

    for kids in siblings.values():
        kids = sorted(kids, key=_display_order)
        for i, node in enumerate(kids):
            for other in kids[i + 1 : i + 1 + sibling_window]:
                if node in present and other in present:
                    graph.add_edge(node, other)
    return graph


def _sample_motif(
    tree: nx.DiGraph,
    root: Any,
    size: int,
    rng: random.Random,
    *,
    motif_id: str,
    source_thread: str,
    source_veracity: str,
    provenance: str,
    sibling_window: int = 1,
) -> Motif | None:
    """Grow a connected ``size``-node motif outward from the cascade root."""
    if tree.number_of_nodes() < size:
        return None
    undirected = tree.to_undirected()
    chosen = [root]
    frontier = list(undirected.neighbors(root))
    while len(chosen) < size and frontier:
        pick = frontier.pop(rng.randrange(len(frontier)))
        if pick in chosen:
            continue
        chosen.append(pick)
        frontier.extend(n for n in undirected.neighbors(pick) if n not in chosen)
    if len(chosen) < size:
        return None

    sub = _visibility_graph(tree, chosen, sibling_window)
    relabel = {old: f"v{i}" for i, old in enumerate(chosen)}  # strips tweet ids
    sub = nx.relabel_nodes(sub, relabel)
    return Motif(
        motif_id=motif_id,
        graph=sub,
        root=relabel[root],
        source_thread=source_thread,
        source_veracity=source_veracity,
        provenance=provenance,
        meta={"sibling_window": sibling_window},
    )


def canonical_motif() -> Motif:
    """The hand-specified motif every condition is additionally run on.

    Plan section 5.2: guards against results being an artifact of any single
    sampled topology. Shape: the root broadcasts to three hubs, each hub carries
    one downstream node, and the two mid-layer hubs are cross-linked so a local
    majority can form without passing through the root.

        v0 (root/Disseminator)
        |-- v1 -- v4
        |-- v2 -- v5
        |-- v3 -- v6
        v1 -- v2  (cross-link)
        v4 -- v5  (cross-link)
    """
    g = nx.Graph()
    g.add_edges_from(
        [
            ("v0", "v1"), ("v0", "v2"), ("v0", "v3"),
            ("v1", "v4"), ("v2", "v5"), ("v3", "v6"),
            ("v1", "v2"), ("v4", "v5"),
        ]
    )
    return Motif(
        motif_id="canonical",
        graph=g,
        root="v0",
        source_thread="hand-specified",
        source_veracity="n/a",
        provenance="hand-specified canonical motif (plan section 5.2)",
    )


def edge_list(motif: Motif) -> list[tuple[str, str]]:
    return sorted(tuple(sorted(e)) for e in motif.graph.edges)
