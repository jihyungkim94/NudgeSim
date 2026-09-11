# Preregistration

The analysis plan, frozen before the main grid runs (plan §8, §9 Month 2). This
document is the OSF preregistration content in repository form.

## Primary outcomes

Both read directly off the action log and the payoff ledger. **No LLM judge
appears anywhere on the critical path.**

- **Cumulative FPR** — share of citizen actions that SHARE or ENDORSE the focal
  false claim.
- **Cumulative EPC** *(headline)* — share that CHALLENGE it, i.e. citizens
  voluntarily absorbing κ.

## Secondary and exploratory

Secondary: epistemic welfare (reported decomposed, **with and without sanction
costs** — see FINDINGS.md §2), pollution half-life, false-correction rate,
reactance/toxicity. Exploratory: stated credence at rounds 0/6/12, reported as a
model self-report and never as a belief; reasoning-trace composition.

## Hypotheses

| | Statement | Test |
|---|---|---|
| H1 | Early intervention yields lower cumulative FPR than late | one-sided, early < late |
| H2 | Aggressive suppresses faster; empathetic suppresses more durably | two-sided; log-rank on half-life |
| H3 | Empathetic raises EPC above control; aggressive suppresses it below control | sign and significance of the tone effect on EPC, tested **separately** from FPR |
| H4 | Timing moderates tone | timing × tone interaction |
| H5 | Backbone effects exceed condition effects | variance decomposition + three-way interaction |
| RQ5 | The intervener does not suppress true claims | placebo arm false-correction rate |

Every hypothesis is stated over an action the agent **takes and pays for**, never
over a belief it reports.

## Models

Primary: mixed-effects on episode-level FPR and EPC, fixed effects for
timing × tone × backbone, random intercept for claim, variance component for
topology, with intervener reach as a covariate. Dunnett contrasts of each
treatment cell against **its own-model control** — pooling controls across
backbones would confound the intervention effect with backbone differences, which
H5 says may be the larger of the two.

## Inference hygiene

Bootstrap 95% CIs (5,000 draws), Holm correction **within each outcome family**,
and effect sizes (Hedges-corrected d, partial η², rank statistics) reported
alongside every p-value. α = 0.05.

## Analysis set and exclusions

Primary analyses use the core arm, full-horizon episodes only. Episodes flagged by
the degeneracy guard (`terminated_early`) are excluded from primary models and
their rate is reported per backbone and per condition — reported, not suppressed.
Motifs where the intervener reaches no citizen are excluded at sampling time by a
declared rule, and the count dropped is reported.

## Stopping rules and gates

1. **Month 1 — ledger gate.** The payoff ledger must pass hand-computed unit
   tests before any grid run. (`tests/test_payoff.py`, 16 tests.)
2. **Month 2 — calibration gate.** Hard Go/No-Go; see CALIBRATION.md. The runner
   refuses to proceed past a NO-GO.
3. **Month 2 — classifier gate.** 300 human-coded traces, Cohen's κ ≥ 0.7, before
   any trace-based mechanism claim.
4. **Manipulation checks.** The tone arms must be length-matched within 3 tokens
   with an identical factual payload, verified over the whole claim pool before
   any run.

No optional stopping: the grid size is fixed in advance and the analysis is run
once on the completed grid.

## Amendments made before freezing

Recorded here rather than silently applied. Each is justified in
[FINDINGS.md](FINDINGS.md).

1. **Welfare reported with and without sanction costs.** The original Σπ measure
   is mechanically anti-correlated with the treatment.
2. **RQ5 conditional contrast added** alongside the pooled one, conditioning on
   whether the intervener actually fired. The pooled contrast is retained as the
   preregistered primary.
3. **Intervener reach** added as a covariate and as a declared sampling
   constraint.
4. **Power.** At 30 seeds/cell only the intervention-vs-control contrast on EPC is
   adequately powered. Either `core_seeds` rises to ~250, or H2 and H3 move to the
   exploratory set. **This decision must be made before the grid runs, not after.**
5. **H2 durability** is untestable as specified (98% censoring). Either the
   horizon lengthens, the threshold becomes relative, or the survival analysis is
   replaced by area-under-FPR.

## Reproducibility

Every episode record carries: prompt-suite version and fingerprint, claim and
topology provenance, payoff parameters, the declared DGP parameters, git commit,
package version, and the seed. A mid-grid prompt edit changes the fingerprint and
is therefore visible in the logs rather than invisible in the results.
