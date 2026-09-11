"""Claim pool and topology tests."""

from __future__ import annotations

import pathlib
import random
import re

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
    """sibling_window=0 is the old reply-tree-as-visibility model, which produces
    motifs where the intervener is adjacent to no citizen. The filter must drop
    exactly those."""
    lib = MotifLibrary.synthetic(seed=5, sibling_window=0)
    before = lib.reach_distribution(DEFAULT_SOCIETY.all_ids)
    assert before.get(0), "expected undeliverable motifs without sibling visibility"
    filtered = lib.with_min_reach(1, DEFAULT_SOCIETY.all_ids)
    assert 0 not in filtered.reach_distribution(DEFAULT_SOCIETY.all_ids)
    assert filtered.meta["n_motifs_dropped_by_reach_filter"] == before[0]


def test_bounded_attention_removes_undeliverable_motifs():
    """The default visibility model should leave nothing for the filter to drop."""
    lib = MotifLibrary.synthetic(seed=5)
    assert 0 not in lib.reach_distribution(DEFAULT_SOCIETY.all_ids)


def test_bounded_attention_does_not_collapse_a_star_into_a_clique():
    """The real PHEME failure mode, in miniature.

    A broadcast star -- one source tweet with six direct replies -- is what 69%
    of real PHEME motifs are. Connecting every sibling makes it the complete
    graph, so "local majority" becomes global and the topology stops varying at
    all. A bounded window must keep it sparse while still letting the replies
    see each other.
    """
    from nudgesim.data.topology import _visibility_graph

    star = nx.DiGraph([("r", f"c{i}") for i in range(6)])
    nodes = ["r", *[f"c{i}" for i in range(6)]]

    reply_tree = _visibility_graph(star, nodes, sibling_window=0)
    windowed = _visibility_graph(star, nodes, sibling_window=1)
    clique = _visibility_graph(star, nodes, sibling_window=99)

    # Reply tree alone: every citizen sees only the root.
    assert reply_tree.number_of_edges() == 6
    assert all(reply_tree.degree(f"c{i}") == 1 for i in range(6))

    # Unbounded siblings: the complete graph on 7 nodes.
    assert clique.number_of_edges() == 21

    # Bounded: replies can see each other, and it is not complete.
    assert 6 < windowed.number_of_edges() < 21
    assert all(windowed.degree(f"c{i}") > 1 for i in range(6))
    assert nx.is_connected(windowed)


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


# ------------------------------------------------ real-corpus loading (LIAR)

LIAR_DIR = pathlib.Path("data/raw/liar")
needs_liar = pytest.mark.skipif(
    not (LIAR_DIR / "train.tsv").exists(),
    reason="LIAR not present; run `nudgesim fetch-data`",
)


@pytest.fixture(scope="module")
def liar_pool() -> ClaimPool:
    return ClaimPool.from_liar(LIAR_DIR, max_per_label=None, seed=1)


@needs_liar
def test_liar_parses_every_published_row(liar_pool: ClaimPool):
    """12,836 statements in the release; anything less means rows were merged."""
    assert liar_pool.meta["n_rows_read"] == 12836
    # All six labels minus the 2,638 excluded half-true statements.
    assert len(liar_pool.claims) == 12836 - 2638


@needs_liar
def test_liar_statements_carry_no_embedded_field_separators(liar_pool: ClaimPool):
    """Unbalanced quotes in LIAR merge rows unless the reader uses QUOTE_NONE."""
    assert not [c for c in liar_pool.claims if "\t" in c.text or "\n" in c.text]


@needs_liar
def test_liar_label_distribution_matches_the_release(liar_pool: ClaimPool):
    counts = liar_pool.summary()["severity_counts"]
    assert counts == {
        "pants-fire": 1050, "false": 2511, "barely-true": 2108,
        "true": 2063, "mostly-true": 2466,
    }


@needs_liar
def test_no_identifiable_public_figure_survives_deidentification(liar_pool: ClaimPool):
    """Plan section 6 safeguard, enforced as a test rather than an intention."""
    named = re.compile(
        r"\b(Obama|Clinton|Trump|Bush|Romney|Biden|McCain|Rubio|Cruz|Sanders|"
        r"Pelosi|Perry|Gingrich|Kerry|Reagan|Boehner|Giuliani|Palin)\b",
        re.I,
    )
    leaks = [c.text for c in liar_pool.claims if named.search(c.text)]
    assert not leaks, f"{len(leaks)} statements leak an identifiable name: {leaks[:3]}"


@needs_liar
def test_deidentification_does_not_shred_ordinary_noun_phrases(liar_pool: ClaimPool):
    """Over-redaction is its own failure: agents must read claims, not boilerplate.

    LIAR's speaker column is roughly a third organisations, so a naive surname
    pass turns "the big Wall Street banks" into three stacked attributions.
    """
    corpus = " ".join(c.text for c in liar_pool.claims)
    assert "Wall Street" in corpus
    assert "Republican" in corpus and "Democrat" in corpus
    attribution = re.compile(
        r"a (national politician|state legislator|senior official|party spokesperson|"
        r"congressional candidate|governor)|an advocacy group|a cable news host"
    )
    total = sum(len(c.text.split()) for c in liar_pool.claims)
    replaced = sum(
        len(m.group(0).split()) for c in liar_pool.claims for m in attribution.finditer(c.text)
    )
    assert replaced / total < 0.20, "de-identification is eating the corpus"


@needs_liar
def test_real_pool_is_labelled_as_liar(liar_pool: ClaimPool):
    assert "LIAR" in liar_pool.provenance
    assert "SYNTHETIC" not in liar_pool.provenance


def test_organisation_speakers_are_excluded_from_the_surname_pass():
    from nudgesim.data.claims import build_speaker_matcher, deidentify

    matcher = build_speaker_matcher(
        ["barack-obama", "john-mccain", "wall-street-journal", "republican-party-of-texas"]
    )
    out = deidentify("The big Wall Street banks backed Obama and McCain.", matcher=matcher)
    assert "Wall Street" in out
    assert "Obama" not in out and "McCain" not in out


def test_surname_pass_ignores_lowercase_homographs():
    from nudgesim.data.claims import build_speaker_matcher, deidentify

    matcher = build_speaker_matcher(["sarah-stone", "mike-baker"])
    out = deidentify("The baker sold stone fruit to Baker.", matcher=matcher)
    assert "The baker sold stone fruit" in out
    assert "to Baker." not in out


def test_single_claim_draws_are_not_all_the_same_stratum(pool: ClaimPool):
    """Drawing one claim at a time must still hit every severity stratum.

    Regression test: a cursor starting at zero returns stratum 0 on every
    one-claim call, which ran an entire 1,080-episode grid on pants-fire claims
    without anything in the pipeline noticing.
    """
    seen = {pool.stratified_sample(1, random.Random(s))[0].severity for s in range(60)}
    assert seen == set(FALSE_LABELS)


def test_grid_cells_are_balanced_across_severity_strata(pool, library):
    from nudgesim.agents.bounded_rational import NormParams
    from nudgesim.game.payoff import PayoffParams
    from nudgesim.runner import GridSpec, expand_grid

    spec = GridSpec(core_seeds=30, placebo_seeds=1, ablation_payoff_seeds=1,
                    ablation_ratio_seeds=1, scale_seeds=1, include_arms=("core",))
    configs = expand_grid(spec, pool, library, payoff=PayoffParams(), norms=NormParams())
    counts: dict[str, int] = {}
    for config in configs:
        counts[config.claim.severity] = counts.get(config.claim.severity, 0) + 1
    assert set(counts) == set(FALSE_LABELS)
    # Even split within tolerance of the 30-per-cell rounding.
    assert max(counts.values()) - min(counts.values()) <= len(configs) * 0.1
