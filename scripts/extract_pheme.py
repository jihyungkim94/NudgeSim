#!/usr/bin/env python3
"""Pull just the reply trees out of a PHEME download, and bundle them for upload.

NudgeSim reads only the reply tree of each thread, so the multi-GB archive does
not need to be unpacked. Point this at whatever file figshare gave you:

    python scripts/extract_pheme.py ~/Downloads/<the-file-you-downloaded>

It handles .zip, .tar.bz2, .tar.gz and .tar, looks inside nested archives
(figshare often wraps the real one), and writes the trees to data/raw/pheme/.

Two PHEME releases are in circulation and they are not the same shape:

  PHEME-9 "all-rnr-annotated-threads" ships structure.json per thread. Those are
      copied out directly.
  PHEME-5 "pheme-rnr-dataset" has no structure.json at all. The reply tree is
      still recoverable from the tweet files, so it is rebuilt from the
      in_reply_to_status_id field. Only ids are read -- never tweet text -- and
      only the rebuilt tree is written out.

If neither is found, it prints what the archive actually contains instead of
guessing.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

WANTED = ("structure.json", "annotation.json")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".bz2", ".gz", ".tgz", ".tbz2", ".xz")


# --------------------------------------------------------------------- archives

def _members(archive: Path):
    """Yield (name, open_fn) for every regular file in a zip or tar."""
    if zipfile.is_zipfile(archive):
        zf = zipfile.ZipFile(archive)
        for info in zf.infolist():
            if not info.is_dir():
                yield info.filename, (lambda n=info.filename: zf.open(n))
        return
    if tarfile.is_tarfile(archive):
        tf = tarfile.open(archive)
        for member in tf:
            if member.isfile():
                yield member.name, (lambda m=member: tf.extractfile(m))
        return
    raise SystemExit(f"{archive.name} is not a .zip or .tar archive")


def _safe(target: Path, root: Path) -> bool:
    """Refuse paths that escape the destination (archives are untrusted input)."""
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _copy_wanted(archive: Path, dest: Path) -> int:
    """Copy out structure.json and annotation.json; return the structure count.

    Only structure.json counts as success. PHEME-5 ships annotation.json but no
    structure.json, so counting both would make the copy step look like it
    worked and skip the reconstruction that release actually needs.
    """
    written = 0
    for name, opener in _members(archive):
        if Path(name).name not in WANTED:
            continue
        target = dest / name
        if not _safe(target, dest):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        src = opener()
        if src is None:
            continue
        with src, target.open("wb") as out:
            shutil.copyfileobj(src, out)
        if Path(name).name == "structure.json":
            written += 1
    return written


def _nested_archives(archive: Path) -> list[str]:
    return [
        name for name, _ in _members(archive)
        if Path(name).suffix.lower() in ARCHIVE_SUFFIXES
    ]


# ------------------------------------------------------- PHEME-5 reconstruction

def _rebuild_from_tweets(archive: Path, dest: Path) -> int:
    """Rebuild reply trees from tweet files (PHEME-5 has no structure.json).

    Reads only `id`/`id_str` and `in_reply_to_status_id` from each tweet. Tweet
    text is never read, and only the derived tree is written out.
    """
    threads: dict[str, dict[str, str | None]] = collections.defaultdict(dict)
    roots: dict[str, str] = {}

    for name, opener in _members(archive):
        path = Path(name)
        if path.suffix != ".json" or path.name in WANTED:
            continue
        parts = path.parts
        if not any(p in ("reactions", "source-tweets", "source-tweet") for p in parts):
            continue
        # .../<event>/<rumours|non-rumours>/<thread_id>/<reactions|source-tweets>/<id>.json
        try:
            kind_at = max(
                i for i, p in enumerate(parts)
                if p in ("reactions", "source-tweets", "source-tweet")
            )
        except ValueError:
            continue
        if kind_at == 0:
            continue
        thread_key = "/".join(parts[:kind_at])
        src = opener()
        if src is None:
            continue
        try:
            with src:
                tweet = json.loads(src.read().decode("utf-8", "replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(tweet, dict):
            continue
        tid = tweet.get("id_str") or tweet.get("id")
        if tid is None:
            continue
        parent = tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id")
        threads[thread_key][str(tid)] = None if parent is None else str(parent)
        if parts[kind_at] in ("source-tweets", "source-tweet"):
            roots[thread_key] = str(tid)

    written = 0
    for thread_key, edges in threads.items():
        root = roots.get(thread_key) or next(
            (t for t, p in edges.items() if p is None or p not in edges), None
        )
        if root is None or len(edges) < 2:
            continue
        children: dict[str, list[str]] = collections.defaultdict(list)
        for tid, parent in edges.items():
            if tid != root and parent in edges:
                children[parent].append(tid)

        def build(node: str) -> dict | list:
            kids = children.get(node)
            if not kids:
                return []            # PHEME writes leaves as an empty list
            return {k: build(k) for k in sorted(kids)}

        target = dest / thread_key / "structure.json"
        if not _safe(target, dest):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({root: build(root)}), encoding="utf-8")
        written += 1
    return written


# ------------------------------------------------------------------- diagnosis

def _describe(archive: Path) -> str:
    names = [n for n, _ in _members(archive)]
    tops = sorted({Path(n).parts[0] for n in names if Path(n).parts})[:12]
    common = collections.Counter(Path(n).name for n in names).most_common(10)
    dirs = collections.Counter(
        p for n in names for p in Path(n).parts[:-1]
    ).most_common(12)
    lines = [
        f"archive contains {len(names)} files",
        "",
        "top-level entries:",
        *(f"    {t}" for t in tops),
        "",
        "most common file names:",
        *(f"    {c:>7}  {n}" for n, c in common),
        "",
        "most common directory names:",
        *(f"    {c:>7}  {n}" for n, c in dirs),
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------ main

def harvest(archive: Path, dest: Path, depth: int = 0) -> tuple[int, str]:
    """Return (threads written, how). Recurses one level into nested archives."""
    n = _copy_wanted(archive, dest)
    if n:
        return n, "structure.json"

    if depth < 2:
        nested = _nested_archives(archive)
        for name in nested:
            with tempfile.TemporaryDirectory() as tmp:
                inner = Path(tmp) / Path(name).name
                for member, opener in _members(archive):
                    if member != name:
                        continue
                    src = opener()
                    if src is None:
                        continue
                    with src, inner.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    break
                if not inner.exists():
                    continue
                print(f"  looking inside {name} ...")
                got, how = harvest(inner, dest, depth + 1)
                if got:
                    return got, how

    rebuilt = _rebuild_from_tweets(archive, dest)
    if rebuilt:
        return rebuilt, "rebuilt from reply ids (PHEME-5 layout)"
    return 0, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path, help="the file you downloaded from figshare")
    ap.add_argument("--dest", type=Path, default=Path("data/raw/pheme"))
    ap.add_argument("--bundle", type=Path, default=Path("pheme_structures.zip"))
    ap.add_argument("--no-bundle", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="only describe what is in the archive, extract nothing")
    args = ap.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"no such file: {args.archive}")

    if args.inspect:
        print(_describe(args.archive))
        return 0

    print(f"reading {args.archive} ...")
    args.dest.mkdir(parents=True, exist_ok=True)
    n, how = harvest(args.archive, args.dest)

    if n == 0:
        print("\nNo reply trees found in that archive. Here is what it holds:\n")
        print(_describe(args.archive))
        print("\nSend the output above and I can tell you which file to use.")
        return 1

    trees = list(args.dest.rglob("structure.json"))
    size = sum(f.stat().st_size for f in args.dest.rglob("*.json"))
    events = sorted({
        p.replace("-all-rnr-threads", "")
        for f in trees for p in f.parts
        if p.endswith("-all-rnr-threads")
    })
    print(f"\nsource  : {how}")
    print(f"threads : {len(trees)}")
    if events:
        print(f"events  : {len(events)} {events}")
    print(f"size    : {size / 1e6:.1f} MB")
    print(f"written : {args.dest}")

    if not args.no_bundle:
        base = args.bundle.with_suffix("")
        made = shutil.make_archive(str(base), "zip", root_dir=args.dest)
        print(f"\nupload this file: {made}  ({Path(made).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
