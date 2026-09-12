#!/usr/bin/env python3
"""Structural check on the paper source.

No LaTeX engine is reachable from this sandbox (the network policy blocks the
TeX bundle hosts), so the document is parsed rather than compiled. This catches
what actually breaks a build of this document -- unbalanced environments, a
missing \\input, a \\cite with no bib entry, a \\ref with no label -- but it is
not a substitute for compiling before submission.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pylatexenc.latexwalker import LatexWalker, LatexWalkerParseError

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.tex"


def inlined(path: Path) -> str:
    """Splice every \\input target in, so the parser sees the whole document."""
    text = path.read_text(encoding="utf-8")

    def splice(match: re.Match[str]) -> str:
        target = ROOT / f"{match.group(1)}.tex"
        if not target.exists():
            raise SystemExit(f"missing \\input target: {target.relative_to(ROOT)}")
        return target.read_text(encoding="utf-8")

    return re.sub(r"\\input\{([^}]+)\}", splice, text)


def main() -> int:
    source = inlined(MAIN)
    problems: list[str] = []

    try:
        LatexWalker(source, tolerant_parsing=False).get_latex_nodes()
    except LatexWalkerParseError as exc:
        problems.append(f"parse error: {exc}")

    begins = re.findall(r"\\begin\{(\w+\*?)\}", source)
    ends = re.findall(r"\\end\{(\w+\*?)\}", source)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            problems.append(
                f"environment {env!r}: {begins.count(env)} begins, {ends.count(env)} ends"
            )

    bib = (ROOT / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", source)
             for k in group.split(",")}
    for missing in sorted(cited - keys):
        problems.append(f"\\cite{{{missing}}} has no entry in refs.bib")
    for unused in sorted(keys - cited):
        problems.append(f"refs.bib entry {unused!r} is never cited")

    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    for ref in sorted({r for r in re.findall(r"\\ref\{([^}]+)\}", source)} - labels):
        problems.append(f"\\ref{{{ref}}} has no \\label")

    pending = source.count(r"\pending{")
    for problem in problems:
        print(f"  FAIL  {problem}")
    print(
        f"\n{len(problems)} problem(s); {len(begins)} environments, "
        f"{len(cited)} citations, {pending} pending-result blocks."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
