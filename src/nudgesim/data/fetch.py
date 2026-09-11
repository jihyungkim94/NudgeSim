"""Corpus acquisition (plan section 6).

Neither corpus is redistributed with this package. This module fetches what can
be fetched automatically and tells you precisely what to do about the rest.

LIAR
    Automatable. The canonical home is the UCSB archive; the Hugging Face
    loading script is deprecated and is not used. A byte-identical copy of the
    three TSVs is mirrored on GitHub, which is what ``fetch_liar`` pulls, and
    the result is checksummed against the published row and label counts so a
    silently different mirror fails loudly rather than quietly changing results.

PHEME-9
    Not automatable. The release lives behind figshare's download endpoint and
    is a ~2 GB archive covered by Twitter's content terms, so it is a manual,
    consciously-accepted download rather than something a script does for you.
    ``pheme_instructions`` prints the steps and the expected layout.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

LIAR_MIRROR = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master"
LIAR_FILES = ("train.tsv", "valid.tsv", "test.tsv")

# Published properties of the LIAR release (Wang, 2017). A mirror that does not
# reproduce these is not LIAR, whatever it is named.
LIAR_EXPECTED_ROWS = 12836
LIAR_EXPECTED_LABELS = {
    "pants-fire": 1050,
    "false": 2511,
    "barely-true": 2108,
    "half-true": 2638,
    "mostly-true": 2466,
    "true": 2063,
}

PHEME_DOI = "10.6084/m9.figshare.6392078"
PHEME_URL = "https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078"


def fetch_liar(dest: str | Path = "data/raw/liar", *, timeout: int = 120) -> Path:
    """Download the three LIAR TSVs and verify them against the release."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in LIAR_FILES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{LIAR_MIRROR}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                target.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not download {url}: {exc}\n"
                "If your network blocks this host, download LIAR by hand from the "
                "UCSB archive and unpack train/valid/test.tsv into "
                f"{dest}/ -- the loader reads the release format directly."
            ) from exc
    verify_liar(dest)
    return dest


def verify_liar(path: str | Path) -> dict[str, object]:
    """Check a LIAR directory against the published row and label counts."""
    import csv

    path = Path(path)
    files = sorted(path.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"no LIAR .tsv files under {path}")

    rows = 0
    labels: dict[str, int] = {}
    for file in files:
        with file.open(newline="", encoding="utf-8") as fh:
            for record in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(record) != 14:
                    continue
                rows += 1
                label = record[1].strip().lower()
                labels[label] = labels.get(label, 0) + 1

    problems: list[str] = []
    if rows != LIAR_EXPECTED_ROWS:
        problems.append(f"expected {LIAR_EXPECTED_ROWS} rows, parsed {rows}")
    for label, expected in LIAR_EXPECTED_LABELS.items():
        if labels.get(label) != expected:
            problems.append(f"label {label}: expected {expected}, got {labels.get(label)}")
    if problems:
        raise ValueError("LIAR verification failed:\n  " + "\n  ".join(problems))

    return {"n_rows": rows, "labels": labels, "files": [f.name for f in files], "ok": True}


def pheme_instructions(dest: str | Path = "data/raw/pheme") -> str:
    return f"""PHEME-9 must be downloaded by hand.

  1. Open {PHEME_URL}
     (DOI {PHEME_DOI})
  2. Download the veracity archive (~2 GB).
  3. You do NOT need to unpack all of it. NudgeSim reads only each thread's
     reply tree, so extract just those (a few MB) and nothing else:

       mkdir -p {dest}
       tar -xjf PHEME_veracity.tar.bz2 -C {dest} --wildcards \\
           '*/structure.json' '*/annotation.json'

     Unpacking the whole archive also works; the loader ignores the rest.
     The resulting layout is:

       {dest}/<event>-all-rnr-threads/{{rumours,non-rumours}}/<thread>/structure.json

  4. Verify:  nudgesim --pheme {dest} check

Why this is not automated: the archive sits behind figshare's download
endpoint, and the tweet content it contains is governed by platform terms.
Fetching it is a decision the user makes, not one a script makes for them.

What is actually read. Only structure.json (the reply tree) and, where present,
annotation.json (the veracity flags). Tweet bodies under source-tweets/ and
reactions/ are never opened, and tweet ids are replaced by positional labels
during motif extraction -- so only derived topology reaches any artefact, which
is also what makes the release licensable (plan section 6).

Without PHEME the pipeline runs on the surrogate cascade generator, and every
artefact is labelled
  "SYNTHETIC-SURROGATE topology (no PHEME data present; not a PHEME result)".
The claim pool and the topology source are independent, so LIAR + surrogate
topology is a valid and clearly-labelled intermediate configuration.
"""
