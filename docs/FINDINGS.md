# Findings

Two kinds of result are reported here, and they are not interchangeable.

**Part 1 — problems in the plan as written.** Findings about the *design*,
established by implementing it. They hold regardless of what runs the engine,
and they are the part of this document that matters most.

**Part 2 — the design-sensitivity run.** The full preregistered grid (1,080
episodes) executed against a declared analytic data-generating process. These
numbers are properties of that DGP. They are **not evidence about LLM
behaviour**.

**Data status.** Claims are the **real LIAR corpus** (Wang, 2017 — all 12,836
statements, de-identified). Topologies are the surrogate cascade generator:
PHEME sits behind figshare, which this environment's egress policy blocks, so it
must be fetched by hand (`nudgesim fetch-data --corpus pheme`). Every artefact
records which. `runs/main` uses the surrogate claim pool, `runs/liar_main` uses
real LIAR; the two agree closely, and the LIAR run is quoted below.

---

## Part 1 — Problems in the plan, surfaced by building it

### 1. The design is under-powered — the headline hypothesis by ~9×

The plan allocates 30 seeded replications per cell. At the effect sizes this
design produces, that is enough for exactly one of the preregistered primary
contrasts.

| Contrast | Cohen's d | n/cell for 80% power | Powered at 30? |
|---|---:|---:|:--:|
| Intervention vs control → EPC | +0.80 | 26 | **yes** |
| **H1** early vs late → FPR | −0.39 | 105 | no |
| **H3** tone → EPC *(headline)* | −0.25 | 259 | no |
| **H2** tone → FPR | +0.11 | 1,205 | no |

The paper's central claim needs roughly **9× the preregistered sample**; H1,
described in the plan as the most straightforward hypothesis, needs 3.5×.

This does not say the effects are absent. It says **the grid as specified cannot
see them** — much cheaper to learn now than in Week 14.

*Recommendation.* Raise `core_seeds` to ~300 (the core grid becomes 6,000
episodes; cost scales with episodes, not cells, so this stays inside the plan's
own budget), or move H2 and H3 to the exploratory set. Decide **before** the grid
runs.

**The pipeline is validated in both directions.** Five runs, identical code,
varying only the declared DGP and the sample size:

| Run | Tone→EPC channel | seeds/cell | Backbones with a significant effect (Holm) |
|---|---|---:|:--|
| `main` | declared | 30 | 0 of 4 — under-powered |
| `highpower` | declared | 200 | **2 of 4** — detected where the effect is largest |
| `strong_highpower` | inflated ×3 | 200 | **4 of 4** — detected everywhere |
| `null_dgp` | set to zero | 30 | 0 of 4 — no false positive |
| `null_highpower` | set to zero | 200 | 0 of 4 — no false positive |

Sensitivity and specificity both demonstrated: the analysis finds the effect when
it is there and there is enough data, and never invents one when it is not.

### 2. "Epistemic welfare = Σπ" ranks every working intervention below doing nothing

The plan defines epistemic welfare as the sum of citizen payoffs. Because δ
charges a citizen every time a claim it is backing gets challenged, and the
intervener's entire function is to generate challenges, the measure is
mechanically anti-correlated with the treatment:

| Condition | FPR ↓ | EPC ↑ | Σπ (plan's measure) | Σπ excluding sanctions |
|---|---:|---:|---:|---:|
| No intervener (control) | 0.657 | 0.052 | **−26.5** | 2.9 |
| Early · empathetic | 0.560 | 0.164 | −152.7 | **11.6** |
| Early · aggressive | 0.577 | 0.126 | −134.0 | 9.6 |
| Late · empathetic | 0.614 | 0.114 | −105.9 | 6.5 |
| Late · aggressive | 0.632 | 0.102 | −107.7 | 8.3 |

Reputation damage is ~88% of all debits in treated arms. Under the plan's
definition the best-performing intervention scores six times worse than doing
nothing — *because it worked*.

This is not a ledger bug: the implementation matches §3.2 exactly and is checked
against hand-computed episodes. It is a specification problem, and sweeping the
accounting convention does not fix it — three conventions (`per_challenge`,
`capped_per_round`, `once_per_challenger`, all implemented and tested) rank every
treated arm below control, because sanction costs are deadweight in the aggregate
however they are counted.

*Recommendation.* Report welfare as experimental public-goods work does (Fehr &
Gächter, 2002): decomposed, with and without sanction costs. The engine emits
`welfare_ex_sanctions` alongside `welfare_total`; on that measure every
intervention arm beats control and the ordering is informative rather than
predetermined.

### 3. The placebo arm measures nothing unless the intervener can decline to act

§5.3 says the intervener "should not suppress" true claims. An intervener
hard-wired to CHALLENGE whatever it is pointed at suppresses them by
construction, and the specificity measure reports the wiring.

Making it veracity-sensitive turns RQ5 into a real question — but only if you
condition on whether it actually fired:

| Placebo arm | n | False-correction rate on TRUE claims | vs control |
|---|---:|---:|---|
| Control (no intervener) | 40 | 0.073 | — |
| Intervener present, stayed silent | 90 | 0.053 | −0.020, p = 0.94 |
| Intervener present, **false-alarmed** | 21 | 0.137 | **+0.065, p = 0.002** |
| *Pooled (as preregistered)* | 111 | 0.069 | −0.004, p = 0.71 |

The preregistered pooled contrast returns a **null**. The conditional split shows
why that null misleads: the intervener's mere presence induces no indiscriminate
skepticism, but each false alarm roughly **doubles** unwarranted challenges
against true claims. Pooling averages the two together.

*Recommendation.* Preregister the conditional contrast and log the intervener's
own false-alarm rate as a first-class quantity — the specificity of the
intervention is bounded by the specificity of the intervener.

### 4. The durability measure is uninformative at a 12-round horizon

Pollution half-life is defined as rounds until FPR falls below 0.2 *and stays
there*. Across 480 treated episodes, **10 reach it — 97.9% censoring**. Mean
end-of-episode FPR is 0.59, Kaplan–Meier medians are undefined in both tone arms,
and the log-rank test on H2 runs on 10 events.

The cause is structural: the Disseminator reposts every round and is by
construction incorrigible, so FPR has a floor a 12-round episode with a single
intervener cannot push through.

*Recommendation.* Lengthen the horizon, make the threshold relative (e.g. 50% of
the control arm's peak), or replace survival analysis with area-under-FPR. **As
specified, H2 cannot be tested.**

### Smaller, but worth fixing

- **Intervener network reach is an uncontrolled moderator.** The Devil's Advocate
  is a peer, so it reaches only its neighbours — 0 to 5 citizens in sampled
  7-node motifs, and **13% of motifs reach none**, making the treatment
  undeliverable. Reach correlates with EPC (Spearman ρ = 0.34, p < 1e-13). Now
  handled by a declared sampling rule plus a logged covariate, but the plan
  should state which it intends.
- **Ablation 2 is non-monotone.** EPC across the β/γ sweep runs 0.136 → 0.231 →
  0.178 (ratios 0.5, 1, 3), not monotone as the plan's framing implies: raising
  the conformity weight strengthens *herding*, which amplifies whichever side is
  locally winning rather than pushing consistently toward pollution.

## Bugs this work found

Recorded because each one silently produced plausible-looking results:

1. **Every episode in the grid ran on a `pants-fire` claim.** The §5.9 severity
   stratification is implemented as a cursor over strata, and the runner drew one
   claim per episode — so it returned stratum 0 every time. The grid looked
   healthy and the numbers looked reasonable; the stratification simply never
   happened. Fixed, with a regression test asserting a balanced three-way split.
2. **A silent endorser was never charged reputation damage.** Damage was applied
   only to agents that *acted* that round, exempting exactly the agents whose
   endorsements were most exposed — those who said their piece and went quiet.
3. **An anti-runaway fix destroyed the effect it was protecting.** Saturating the
   norm-salience channel stopped a genuine feedback explosion in *peer*
   correction, but applying the same transform to the *policing* channel
   compressed a three-fold difference in enforcement intensity into a few
   hundredths of a utility point — flattening the empathetic-versus-aggressive
   contrast the study exists to measure. Policing comes from one agent at one
   action per round, so exponential decay already bounds it; it needed
   normalising, not saturating.
4. **LIAR rows were silently merging.** LIAR statements contain unbalanced double
   quotes, and Python's default CSV quoting swallowed line breaks at them,
   merging rows and bleeding later columns into statement text. `QUOTE_NONE`
   fixes it; a test now asserts all 12,836 published rows parse.
5. **De-identification was both too weak and too strong.** The speaker column
   alone left 5.8% of statements naming public figures (statements *about*
   people, not *by* them). Adding a surname pass fixed that but shredded the
   corpus — LIAR's speaker column is ~⅓ organisations, so "the big Wall Street
   banks" became three stacked attributions. Filtering organisation-shaped
   speakers gives 0.00% leakage with 11% of tokens replaced.

---

## Part 2 — The design-sensitivity run

`runs/liar_main`: 1,080 episodes, real LIAR claims, surrogate topology, declared
DGP, surrogate backbones. **Properties of the declared DGP, not results about
language models.**

**Engine health.** 0% of episodes terminated early, 0% persona breaks, 0.3%
repetition, ~0.16 s/episode. The full grid runs in 14 s.

**Calibration gate (§5.7).** GO, with a propagation margin of +0.113 stable
across six seeds (6/6). The gate has teeth: removing the novelty channel — the
mechanism Vosoughi, Roy & Aral (2018) attribute the asymmetry to — drops the
margin to +0.002 and the gate fails 5 of 6 seeds. See [CALIBRATION.md](CALIBRATION.md).

**H1 (timing).** Supported in sign and significant: early intervention yields
lower cumulative FPR than late (d = −0.39). Under-powered as preregistered.

**H3 (the headline).** The predicted **dissociation does not hold in any
backbone**. Both tones raise EPC well above control (+0.09 empathetic, +0.06
aggressive); neither suppresses it below control. In this DGP norm demonstration
dominates crowding-out: watching *anyone* pay the correction cost makes citizens
more willing to pay it, and that outweighs the responsibility-displacement the
aggressive tone produces. The tone difference runs in the predicted direction
(aggressive < empathetic) but needs ~259 seeds/cell to detect.

Note *why* it is small: tone reaches EPC through two opposing channels — an
aggressive corrector both displaces responsibility (lowering EPC) and updates
beliefs harder (raising it). The net is a difference of two larger quantities,
which is exactly the kind of effect a 30-seed design cannot resolve.

**H5 (model dependence).** Supported, strongly. Backbone accounts for **32.6%** of
EPC variance; timing 3.4%; tone 1.5%. Backbone choice dominates intervention
design by roughly an order of magnitude. If this survives contact with real model
families, "the right way to intervene" is a per-model question and the
four-backbone benchmark is the paper's most robust contribution.

**Ablation 1 (payoff visibility).** Behaviour is *not* invariant to the payoff
structure — removing prices raises EPC from 0.113 to 0.332 (d = +2.11). The game
structure drives behaviour; agents correct freely once correcting is free.

**Ablation 2 (incentive ratio).** EPC 0.136 → 0.231 → 0.178 across β/γ ∈
{0.5, 1, 3}; non-monotone, see above.

**Scale check (N = 50).** Cell rank agreement with N = 7 is Spearman ρ = 0.60 on
real LIAR (ρ = 1.00 on the surrogate pool) — the ordering of conditions largely
survives the scale change, with late arms slightly more polluted at N = 50.

**Trace classifier.** Offline agreement with ground-truth motives is κ = 1.0, but
this is **circular** and proves only that the plumbing works: the surrogate's
traces are templated, so a keyword classifier tuned to them recovers them
perfectly. The plan's real gate — 300 human-coded LLM traces at κ ≥ 0.7 — is
untouched by this and still has to be run.
