# Project plan sources

`docs/Project_Plan_NudgeSim_v2.4.docx` is generated, not hand-edited. These are
the sources it is generated from, so a revision to the plan is a diff rather
than a re-upload of a binary.

```sh
cd docs/plan_src && npm install && npm run build
```

`helpers.js` holds the page geometry, palette and the table, callout and
heading builders. `part1.js` … `part5.js` are the document in order; §0's
revision table lives in `part1.js` and is the first thing to update when a
finding changes the design.
