"""Claim pool and topology tests."""

from __future__ import annotations

import pytest

from nudgesim.agents.persona import DEFAULT_SOCIETY
from nudgesim.data.claims import (
    ENGAGEMENT_BY_SEVERITY,
    FALSE_LABELS,
    TRUE_LABELS,
    ClaimPool,
    claim_engagement,
    deidentify,
)
from nudgesim.data.topology import MotifLibrary, canonical_motif, cascade_stats
import networkx as nx


def test_deidentify_removes_named_speakers():
    text = "Barack Obama said the programme was cancelled."
    out = deidentify(text, ["Barack Obama"])
    assert "Obama" not in out and "Barack" not in out
    assert out.endswith("said the programme was cancelled.")


def test_deidentify_is_stable_for_the_same_name():
    a = deidentify("Jane Roe voted against it.", ["Jane Roe"])
    b = deidentify("Jane Roe voted against it.", ["Jane Roe"])
    assert a == b


def test_deidentify_catches_title_plus_name_without_a_vocabulary():
    out = deidentify("Senator Smith introduced the bill.")
    assert "Smith" not in out


def test_synthetic_pool_is_labelled_as_not_liar(pool: ClaimPool):
    assert "SYNTHETIC" in pool.provenance
    assert "not a LIAR result" in pool.provenance


def test_pool_severity_split_excludes_half_true(pool: ClaimPool):
    severities = {c.severity for c in pool.claims}
    assert severities <= set(FALSE_LABELS) | set(TRUE_LABELS)
    assert "half-true" not in severities


def test_every_false_claim_has_a_topic_matched_placebo(pool: ClaimPool):
    for pair in pool.placebo_pairs:
        assert pair.false_claim.is_false and not pair.true_claim.is_false
    # Topic matching should succeed for essentially the whole pool.
    matched = sum(
        1 for p in pool.placebo_pairs if p.false_claim.topic == p.true_claim.topic
    )
    assert matched / len(pool.placebo_pairs) > 0.95


def test_stratified_sample_touches_every_false_stratum(pool: ClaimPool, rng):
    sample = pool.stratified_sample(30, rng)
    assert {c.severity for c in sample} == set(FALSE_LABELS)


def test_engagement_ranks_false_above_true():
    assert ENGAGEMENT_BY_SEVERITY["pants-fire"] > ENGAGEMENT_BY_SEVERITY["true"]
    pool = ClaimPool.synthetic(seed=3)
    false_mean = sum(claim_engagement(c) for c in pool.false_claims) / len(pool.false_claims)
    true_mean = sum(claim_engagement(c) for c in pool.true_claims) / len(pool.true_claims)
    assert false_mean > true_mean


def test_canonical_motif_shape_is_stable():
    motif = canonical_motif()
    assert motif.size == 7
    assert motif.root == "v0"
    assert nx.is_connected(motif.graph)


def test_role_placement_puts_the_intervener_on_a_hub():
    motif = canonical_motif()
    hub = motif.assign_roles(DEFAULT_SOCIETY.all_ids, placement="hub")
    peripheral = motif.assign_roles(DEFAULT_SOCIETY.all_ids, placement="peripheral")
    hub_node = [n for n, r in hub.items() if r == "DevilsAdvocate"][0]
    per_node = [n for n, r in peripheral.items() if r == "DevilsAdvocate"][0]
    assert motif.graph.degree(hub_node) >= motif.graph.degree(per_node)
    assert hub[motif.root] == "Disseminator"


def test_reach_filter_removes_motifs_that_cannot_deliver_the_treatment():
    lib = MotifLibrary.synthetic(seed=5)
    before = lib.reach_distribution(DEFAULT_SOCIETY.all_ids)
    filtered = lib.with_min_reach(1, DEFAULT_SOCIETY.all_ids)
    assert 0 in before
    assert 0 not in filtered.reach_distribution(DEFAULT_SOCIETY.all_ids)
    assert filtered.meta["n_motifs_dropped_by_reach_filter"] == before[0]


def test_scaled_library_has_no_seven_node_canonical_motif():
    lib = MotifLibrary.synthetic(n_threads=5, motif_size=20, seed=2)
    assert all(m.size == 20 for m in lib.motifs)


def test_cascade_stats_on_a_known_tree():
    tree = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d")])
    stats = cascade_stats(tree, "a")
    assert stats.size == 4
    assert stats.depth == 2
    assert stats.max_breadth == 2


def test_assign_roles_rejects_a_size_mismatch():
    with pytest.raises(ValueError):
        canonical_motif().assign_roles(("only", "three", "roles"))
