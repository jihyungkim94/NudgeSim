# Project plan sources

`docs/Project_Plan_NudgeSim_V1.docx` is generated, not hand-edited. These are
the sources it is generated from, so a revision to the plan is a diff rather
than a re-upload of a binary.

```sh
cd docs/plan_src && npm install && npm run build
```

`helpers.js` holds the page geometry, palette and the table, callout and
heading builders. `part1.js` … `part5.js` are the document in order, presented
as the current design rather than as a changelog. Internal engineering
history (what was tried, what broke, what changed) lives in
`docs/FINDINGS.md`, not in the plan itself — keep it that way: the plan is
what a reader evaluating the research sees, the findings doc is the lab
notebook behind it.
