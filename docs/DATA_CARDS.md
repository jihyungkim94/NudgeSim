# Data cards

## LIAR (Wang, 2017)

**Role.** Claim pool. False-side statements are the injected misinformation;
matched true statements are the placebo controls.

**Properties.** 12.8K short PolitiFact statements, six-way veracity labels,
speaker and context metadata.

**Access.** The original UCSB archive. The Hugging Face loading script is
deprecated and is not used. **Not redistributed in this repository.**

**Stratification.** False side: `pants-fire`, `false`, `barely-true`. True side:
`true`, `mostly-true`. `half-true` is excluded from both — it is neither a clean
falsehood to inject nor a clean truth to protect.

**De-identification.** Real speaker names are replaced with generic attributions
("a state legislator", "a party spokesperson") in **all agent-visible text**, so
the simulation never generates novel false associations about identifiable
individuals. Two passes: the `speaker` column as an explicit vocabulary, then a
title-plus-name regex for names appearing only inside statement text. Replacement
is deterministic per name, so the same person maps to the same generic role
throughout.

**Placebo matching.** Each false claim is paired with a true claim matched on
topic, then on token length. Mean length ratio > 0.97 on the shipped pool.

## PHEME-9 (Kochkina, Liakata & Zubiaga, 2018)

**Role.** Topology source and calibration target. Reply trees initialise the
visibility network; cascade statistics validate the control baseline.

**Properties.** Twitter rumour threads across nine breaking-news events, with
per-thread reply trees and veracity annotations.

**Access.** figshare DOI `10.6084/m9.figshare.6392078`. Annotations are CC-BY;
tweet content is governed by platform terms. **Not redistributed here.**

**What is read.** Only `structure.json` (the reply tree) and, where present,
`annotation.json` (the veracity flags — PHEME stores no veracity string, so a
rumour thread is read as true/false/unverified from its `true` and
`misinformation` flags, and an absent annotation is itself the non-rumour
label). **Tweet text is never loaded**: `source-tweets/` and `reactions/` are
not opened. Node identifiers are stripped and replaced with positional labels
(`v0…v6`) during motif extraction, so no tweet id survives into any artefact.
Only *derived topology* is released.

**You do not need the whole archive.** Because only two small JSON files per
thread are read, the ~2 GB download can be extracted selectively — a few MB:

```bash
tar -xjf PHEME_veracity.tar.bz2 -C data/raw/pheme --wildcards \
    '*/structure.json' '*/annotation.json'
```

**Two sampling properties worth knowing**, both of which only appear with a real
release and are covered by tests in `tests/test_pheme.py`:

* Threads are shuffled from the run seed before the motif cap applies. `rglob`
  yields paths in sorted order, so taking the first N motifs would draw them all
  from whichever event sorts first — the library would describe one news event
  rather than nine.
* Cascade statistics are computed over **every** thread, not just the threads
  that became motifs. They are the calibration gate's reference distribution
  (§5.7), so truncating them at the motif cap would compare the simulation
  against a biased subsample of the real cascades.

## Surrogate generators (offline fallback)

Used when neither corpus is present. Both carry a provenance string that
propagates into every episode record, every results table and every manifest, so a
surrogate run can never be mistaken for a corpus-backed one:

- `SYNTHETIC-SURROGATE (no LIAR data present; not a LIAR result)`
- `SYNTHETIC-SURROGATE topology (no PHEME data present; not a PHEME result)`

**Claims.** Templated, non-defamatory statements carrying no real speakers or
events. They preserve only the structural properties the simulation uses —
severity stratum, topic, token length — and nothing else.

**Topologies.** Reply trees grown by a preferential-attachment reply process
tilted toward recency, reproducing the *shape family* of rumour cascades. This is
not PHEME and is labelled as such. The N = 50 scale-check topologies come from
this generator in **both** online and offline modes, because real PHEME threads
rarely reach 50 nodes — so the scale check tests scale-dependence of the
mechanism, not of PHEME-shaped cascades specifically.

## Human data

None collected. No human subjects, no live-platform deployment, no interaction
between agents and real users. The human coding described in plan §5.6 is
performed by the research team on model-generated text.
