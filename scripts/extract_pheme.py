#!/usr/bin/env python3
"""Pull just the reply trees out of a PHEME download, and bundle them for upload.

NudgeSim reads only two small JSON files per thread, so the ~2 GB archive does
not need to be unpacked. Point this at whatever file figshare gave you:

    python scripts/extract_pheme.py ~/Downloads/<the-file-you-downloaded>

It handles .zip, .tar.bz2, .tar.gz and .tar, writes the trees to
data/raw/pheme/, and leaves a small pheme_structures.zip next to it to upload.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

WANTED = ("structure.json", "annotation.json")


def _wanted(name: str) -> bool:
    return Path(name).name in WANTED


def extract(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if _wanted(m)]
            if not members:
                return 0
            for name in members:
                target = dest / name
                if not _safe(target, dest):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
        return written

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.isfile() or not _wanted(member.name):
                    continue
                target = dest / member.name
                if not _safe(target, dest):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
        return written

    raise SystemExit(f"{archive} is not a .zip or .tar archive")


def _safe(target: Path, root: Path) -> bool:
    """Refuse paths that escape the destination (archives are untrusted input)."""
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path, help="the file you downloaded from figshare")
    ap.add_argument("--dest", type=Path, default=Path("data/raw/pheme"))
    ap.add_argument("--bundle", type=Path, default=Path("pheme_structures.zip"))
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"no such file: {args.archive}")

    print(f"reading {args.archive} ...")
    n = extract(args.archive, args.dest)
    if n == 0:
        raise SystemExit(
            "found no structure.json inside that archive.\n"
            "It may be an outer wrapper — unzip it once and run this on the "
            "archive inside."
        )

    trees = list(args.dest.rglob("structure.json"))
    size = sum(f.stat().st_size for f in args.dest.rglob("*.json"))
    events = sorted({
        p.replace("-all-rnr-threads", "")
        for f in trees for p in f.parts if p.endswith("-all-rnr-threads")
    })
    print(f"\nextracted {n} files")
    print(f"  threads : {len(trees)}")
    print(f"  events  : {len(events)} {events if events else ''}")
    print(f"  size    : {size / 1e6:.1f} MB")
    print(f"  written : {args.dest}")

    if not args.no_bundle:
        base = args.bundle.with_suffix("")
        made = shutil.make_archive(str(base), "zip", root_dir=args.dest)
        print(f"\nupload this file: {made}  ({Path(made).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
