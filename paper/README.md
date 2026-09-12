# Paper

`main.tex` is the paper. Every table in it is generated from run artefacts — no
number is typed by hand.

```sh
.venv/bin/python paper/build_tables.py   # regenerate tables/ from runs/
.venv/bin/python paper/check.py          # structural check on the source
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`build_tables.py` reads `runs/real_main`, `runs/real_strong`, `runs/real_null`,
`runs/hp_*` and `runs/cost/cost.json`, and runs the visibility sweep itself
(cached in `runs/visibility/sweep.json`) because that result is the paper's
headline and is not produced by the standard grid. Re-run the grid first if the
engine has changed; the tables follow from whatever is in `runs/`.

## Two things to know before reading

**No LLM run has been executed.** Sections that report model behaviour are
marked with a red `[PENDING LLM RUN]` box and are deliberately empty rather than
estimated. `check.py` counts those boxes so they cannot be lost track of.
Everything else is either a property of the environment's design or of a
declared analytic data-generating process, and is labelled as such.

**The PDF has not been compiled here.** This sandbox's network policy blocks the
TeX distribution hosts, so `check.py` parses the document (with `\input`s
spliced in) and verifies environment balance, citation keys against `refs.bib`,
and `\ref`/`\label` resolution. That is not the same as a successful build —
compile on Overleaf or locally before circulating.
