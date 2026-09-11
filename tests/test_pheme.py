"""PHEME-9 loader tests against fixtures that mirror the real release.

The loader is written against the figshare release layout, which this project's
build environment cannot download (figshare is blocked by egress policy). These
fixtures reproduce the format and, more importantly, its known quirks, so the
loader is exercised before anyone drops 2 GB of real data on it:

  * leaves are ``[]`` (an empty LIST), not ``{}``;
  * a thread can be a bare source tweet with no replies at all;
  * some threads are deep chains, some are flat broadcasts;
  * ``annotation.json`` carries veracity for rumour threads and is absent for
    non-rumours;
  * the release contains files that are malformed, empty, or non-UTF-8, and a
    single bad thread must not take down a 6,000-thread parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nudgesim.agents.persona import DEFAULT_SOCIETY
from nudgesim.data.topology import MotifLibrary

EVENTS = ("charliehebdo", "ferguson", "germanwings-crash", "ottawashooting")


def _chain(root: int, depth: int) -> dict:
    """A deep reply chain: each tweet answered by exactly one other."""
    node: dict | list = []
    for i in range(depth, 0, -1):
        node = {str(root + i): node}
    return {str(root): node}


def _broadcast(root: int, width: int) -> dict:
    """A flat cascade: the source tweet answered directly by many."""
    return {str(root): {str(root + i): [] for i in range(1, width + 1)}}


def _mixed(root: int, width: int, depth: int) -> dict:
    children: dict = {}
    for i in range(1, width + 1):
        sub: dict | list = []
        for j in range(depth, 0, -1):
            sub = {str(root + i * 100 + j): sub}
        children[str(root + i)] = sub
    return {str(root): children}


def _write(path: Path, payload) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "structure.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def pheme_root(tmp_path_factory) -> Path:
    """A miniature PHEME-9 release: 4 events, rumours and non-rumours."""
    root = tmp_path_factory.mktemp("pheme")
    tid = 552783238415265792

    for e, event in enumerate(EVENTS):
        base = root / f"{event}-all-rnr-threads"
        for kind in ("rumours", "non-rumours"):
            for i in range(6):
                tid += 1000
                thread = base / kind / str(tid)
                if i % 3 == 0:
                    _write(thread, _chain(tid, 9))
                elif i % 3 == 1:
                    _write(thread, _broadcast(tid, 11))
                else:
                    _write(thread, _mixed(tid, 4, 3))
                # The real release's two annotation shapes, verbatim: flags are
                # integers, and a non-rumour carries only is_rumour.
                if kind == "rumours":
                    (thread / "annotation.json").write_text(
                        json.dumps({
                            "is_rumour": "rumour",
                            "category": "a rumour category string",
                            "misinformation": 1 if i % 2 else 0,
                            "true": 0 if i % 2 else 1,
                            "links": [],
                            "is_turnaround": 0,
                        }),
                        encoding="utf-8",
                    )
                else:
                    (thread / "annotation.json").write_text(
                        json.dumps({"is_rumour": "nonrumour"}), encoding="utf-8",
                    )
                # Real threads ship the tweet bodies too; the loader must never
                # read them.
                (thread / "source-tweets").mkdir(exist_ok=True)
                (thread / "source-tweets" / f"{tid}.json").write_text(
                    json.dumps({"text": "TWEET BODY MUST NOT BE READ", "id": tid}),
                    encoding="utf-8",
                )

    # Degenerate threads that exist in the real release.
    bad = root / "charliehebdo-all-rnr-threads" / "rumours"
    _write(bad / "900000000000000001", {"900000000000000001": []})      # no replies
    (bad / "900000000000000002").mkdir(parents=True, exist_ok=True)
    (bad / "900000000000000002" / "structure.json").write_text("", encoding="utf-8")
    (bad / "900000000000000003").mkdir(parents=True, exist_ok=True)
    (bad / "900000000000000003" / "structure.json").write_text(
        "{not valid json", encoding="utf-8")
    (bad / "900000000000000004").mkdir(parents=True, exist_ok=True)
    (bad / "900000000000000004" / "structure.json").write_bytes(b"\xff\xfe\x00bad")
    _write(bad / "900000000000000005", [])                               # list, not dict
    return root


def test_loads_a_release_shaped_directory(pheme_root: Path):
    lib = MotifLibrary.from_pheme(pheme_root, seed=1)
    assert len(lib) > 10
    assert "PHEME-9" in lib.provenance
    assert "SYNTHETIC" not in lib.provenance


def test_malformed_threads_are_skipped_not_fatal(pheme_root: Path):
    """One unparseable thread must not abort a 6,000-thread release."""
    lib = MotifLibrary.from_pheme(pheme_root, seed=1)
    assert lib.meta["n_structure_files"] == 53   # 48 good + 5 degenerate
    assert lib.meta["n_threads_parsed"] < lib.meta["n_structure_files"]
    assert lib.meta["n_threads_parsed"] > 40


def test_empty_list_leaves_are_parsed_as_leaves(pheme_root: Path):
    """PHEME writes leaves as [] -- reading them as children would break depth."""
    lib = MotifLibrary.from_pheme(pheme_root, seed=1)
    ref = lib.reference_stats()
    assert ref["depth"] >= 1
    assert ref["size"] > 1


def test_motifs_span_every_event(pheme_root: Path):
    """Alphabetical order would take every motif from the first event."""
    lib = MotifLibrary.from_pheme(pheme_root, max_motifs=12, seed=3)
    assert set(lib.meta["events"]) == set(EVENTS)
    covered = {m.meta.get("event") for m in lib.motifs if m.meta.get("event")}
    assert len(covered) >= 3, f"motifs drawn from only {covered}"


def test_thread_statistics_cover_all_threads_not_just_sampled_motifs(pheme_root: Path):
    """The calibration reference must not be computed from a truncated sample."""
    few = MotifLibrary.from_pheme(pheme_root, max_motifs=3, seed=1)
    many = MotifLibrary.from_pheme(pheme_root, max_motifs=200, seed=1)
    assert len(few.motifs) < len(many.motifs)
    assert len(few.thread_stats) == len(many.thread_stats)
    assert few.reference_stats() == many.reference_stats()


def test_veracity_is_read_from_annotation_not_only_the_folder(pheme_root: Path):
    lib = MotifLibrary.from_pheme(pheme_root, max_motifs=200, seed=1)
    veracities = {m.source_veracity for m in lib.motifs}
    assert "non-rumour" in veracities
    assert veracities & {"true", "false"}, f"annotation.json unused: {veracities}"


def test_non_rumour_annotation_is_not_read_as_unverified():
    """A non-rumour carries only is_rumour; reading the flags alone mislabels it."""
    from nudgesim.data.topology import _pheme_veracity

    def verdict(tmp: Path, payload: dict | None) -> str:
        thread = tmp / "ferguson-all-rnr-threads" / "non-rumours" / "1"
        thread.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            (thread / "annotation.json").write_text(json.dumps(payload), encoding="utf-8")
        return _pheme_veracity(thread / "structure.json")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert verdict(Path(d), {"is_rumour": "nonrumour"}) == "non-rumour"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"is_rumour": "nonrumour"}, "non-rumour"),
        ({"is_rumour": "rumour", "misinformation": 0, "true": 1}, "true"),
        ({"is_rumour": "rumour", "misinformation": 1, "true": 0}, "false"),
        ({"is_rumour": "rumour", "misinformation": 0, "true": 0}, "unverified"),
        ({"is_rumour": "rumour", "misinformation": 1}, "false"),
        ({"is_rumour": "rumour", "misinformation": 0}, "unverified"),
    ],
)
def test_real_annotation_shapes(tmp_path: Path, payload: dict, expected: str):
    """Every (misinformation, true) combination present in the real release."""
    from nudgesim.data.topology import _pheme_veracity

    thread = tmp_path / "ferguson-all-rnr-threads" / "rumours" / "1"
    thread.mkdir(parents=True)
    (thread / "annotation.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _pheme_veracity(thread / "structure.json") == expected


def test_tweet_ids_never_reach_the_motif(pheme_root: Path):
    """Only derived topology is released (plan section 6, licensing)."""
    lib = MotifLibrary.from_pheme(pheme_root, max_motifs=200, seed=1)
    for motif in lib.motifs:
        for node in motif.graph.nodes:
            assert node.startswith("v"), f"raw tweet id leaked into motif: {node}"


def test_tweet_text_is_never_loaded(pheme_root: Path):
    lib = MotifLibrary.from_pheme(pheme_root, max_motifs=200, seed=1)
    blob = json.dumps([m.to_json() for m in lib.motifs])
    assert "TWEET BODY MUST NOT BE READ" not in blob


def test_library_is_usable_by_the_engine(pheme_root: Path):
    lib = MotifLibrary.from_pheme(pheme_root, max_motifs=200, seed=1)
    usable = lib.with_min_reach(1, DEFAULT_SOCIETY.all_ids)
    assert len(usable) > 0
    motif = usable.motifs[0]
    assert motif.size == 7
    roles = motif.assign_roles(DEFAULT_SOCIETY.all_ids)
    assert roles[motif.root] == DEFAULT_SOCIETY.disseminator_id


def test_missing_directory_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MotifLibrary.from_pheme(tmp_path / "nope")
