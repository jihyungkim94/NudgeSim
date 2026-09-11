# Findings

Two kinds of result are reported here, and they are not interchangeable.

**Part 1 — problems in the plan as written.** These are findings about the
*design*, established by implementing it. They hold regardless of what runs the
engine, and they are the part of this document that matters most.

**Part 2 — the design-sensitivity run.** 1,080 episodes of the full preregistered
grid, executed against a declared analytic data-generating process. These numbers
are properties of that DGP. They are **not evidence about LLM behaviour** and
must not be reported as such.

---

## Part 1 — Four problems in the plan, surfaced by building it

### 1. The headline hypothesis is under-powered by roughly an order of magnitude

The plan allocates 30 seeded replications per cell. At the effect sizes this
design produces, that is enough for exactly one of the preregistered primary
contrasts.

| Contrast | Cohen's d | n/cell for 80% power | Powered at 30? |
|---|---:|---:|:--:|
| Intervention vs control → EPC | +0.88 | 22 | **yes** |
| Intervention vs control → FPR | −0.49 | 67 | no |
| **H1** early vs late → FPR | −0.41 | 93 | no |
| **H3** tone → EPC *(headline)* | −0.25 | 245 | no |
| **H2** tone → FPR | +0.19 | 452 | no |

The paper's central claim (H3) needs roughly **8× the preregistered sample**. H1,
described in the plan as the most straightforward hypothesis, needs about 3×.

This is not a statement that the effects are absent — the high-power control below
shows the pipeline detects them once powered. It is a statement that **the grid as
specified cannot see them**, which is much cheaper to learn now than in Week 14.

*Recommendation.* Either raise `core_seeds` to ~250 (the core grid becomes ~5,000
episodes, and at the plan's own cost model still well inside the stated budget,
since cost scales with episodes not with cells), or drop H2/H3 from the confirmatory
set and preregister them as exploratory.

**Validation that the pipeline is honest about this.** Three runs, same code:

| Run | DGP | seeds/cell | Backbones with a significant tone effect on EPC (Holm) |
|---|---|---:|:--|
| `runs/main` | H3 effect present | 30 | 0 of 4 — under-powered |
| `runs/highpower` | H3 effect present | 200 | **3 of 4** — detected |
| `runs/null_highpower` | H3 effect set to zero | 200 | **0 of 4** — no false positive |

The analysis finds the effect when it is there and enough data is available, and
does not invent one when it is not.

### 2. "Epistemic welfare = Σπ" ranks every working intervention below doing nothing

The plan defines epistemic welfare as the sum of citizen payoffs. Because δ
charges a citizen every time a claim it is backing gets challenged, and the
intervener's entire function is to generate challenges, the welfare measure is
mechanically anti-correlated with the treatment:

| Condition | FPR ↓ | EPC ↑ | Σπ (plan's measure) | Σπ excluding sanctions |
|---|---:|---:|---:|---:|
| No intervener (control) | 0.677 | 0.042 | **−22.6** | 3.2 |
| Early · empathetic | 0.577 | 0.149 | −149.5 | **12.0** |
| Early · aggressive | 0.596 | 0.114 | −130.1 | 11.4 |
| Late · empathetic | 0.626 | 0.104 | −103.7 | 6.7 |
| Late · aggressive | 0.657 | 0.092 | −109.3 | 9.4 |

Reputation damage is ~88% of all debits in treated arms. Under the plan's
definition the best-performing intervention looks seven times worse than doing
nothing, *because it worked*.

This is not a ledger bug — the implementation matches §3.2 exactly, and is checked
against hand-computed episodes in `tests/test_payoff.py`. It is a specification
problem. Sweeping the accounting convention does not fix it: three conventions
(`per_challenge`, `capped_per_round`, `once_per_challenger`, all implemented and
tested) all rank every treated arm below control, because sanction costs are
deadweight in the aggregate however they are counted.

*Recommendation.* Report welfare the way experimental public-goods work does
(Fehr & Gächter, 2002): decomposed, with and without sanction costs. The engine
now emits `welfare_ex_sanctions` alongside `welfare_total`; on that measure every
intervention arm beats control, and the ordering is informative rather than
predetermined.

### 3. The placebo arm measures nothing unless the intervener can decline to act

Plan §5.3 says the intervener "should not suppress" true claims. An intervener
hard-wired to CHALLENGE whatever it is pointed at suppresses them by construction,
and the specificity measure reports the wiring.

Making the intervener veracity-sensitive (it judges the claim first and stays
silent if it judges it sound) turns RQ5 into a real question, and the answer
separates cleanly — but only if you condition on whether it actually fired:

| Placebo arm | n | False-correction rate on TRUE claims | vs control |
|---|---:|---:|---|
| Control (no intervener) | 40 | 0.076 | — |
| Intervener present, stayed silent | 141 | 0.054 | −0.023, p = 0.87 |
| Intervener present, **false-alarmed** | 19 | 0.150 | **+0.073, p = 0.004** |
| *Pooled (as preregistered)* | 160 | 0.068 | −0.009, p = 0.78 |

The preregistered pooled contrast returns a **null**. The conditional split shows
why that null is misleading: the intervener's mere presence induces no
indiscriminate skepticism, but each false alarm roughly **doubles** unwarranted
challenges against true claims. Pooling averages the two together.

*Recommendation.* Preregister the conditional contrast, and log the intervener's
own false-alarm rate as a first-class quantity — the specificity of the
intervention is bounded by the specificity of the intervener.

### 4. The durability measure is uninformative at a 12-round horizon

Pollution half-life is defined as rounds until FPR falls below 0.2 *and stays
there*. Across 480 treated episodes, **9 reach it — 98.1% censoring**. Mean
end-of-episode FPR is 0.59. Kaplan–Meier medians are undefined (∞) in both tone
arms, and the log-rank test on H2 is consequently uninformative (p = 0.09 on 9
events).

The cause is structural: the Disseminator reposts every round and is by
construction incorrigible, so FPR has a floor that a 12-round episode with a
single intervener cannot push through.

*Recommendation.* Either lengthen the horizon, relax the threshold to a relative
one (e.g. 50% of the control arm's peak), or replace the survival analysis with
the area-under-FPR-curve measure the engine already computes. As specified, H2
cannot be tested.

### Also worth fixing (smaller)

- **Intervener network reach is an uncontrolled moderator.** The Devil's Advocate
  is a peer in a network, so it only reaches its neighbours. In sampled 7-node
  motifs it reaches 0–5 citizens; 13% of motifs reach **none**, making the
  treatment undeliverable. Reach correlates with EPC (Spearman ρ = 0.34,
  p < 1e-13). The engine now drops unreachable motifs by a declared sampling rule
  and logs reach as a covariate — but the plan should say which it intends.
- **A silent agent must still own its endorsements.** An early draft of the ledger
  charged reputation damage only to agents that *acted* that round, silently
  exempting exactly the agents whose endorsements were most exposed — those who
  said their piece and went quiet. Fixed, and now covered by a test.

---

## Part 2 — The design-sensitivity run

`runs/main`, 1,080 episodes, declared DGP, surrogate backbones, surrogate data.
**These are properties of the declared DGP, not results about language models.**

**Engine health.** 0% of episodes terminated early, 0% persona breaks, 0.3%
repetition rate, mean 0.16 s/episode. The full 1,080-episode grid runs in 14 s.

**Calibration gate (§5.7).** GO. The baseline reproduces the false-versus-true
diffusion asymmetry with a propagation margin of +0.113, stable across six seeds
(6/6 GO). The gate has teeth: removing the novelty channel — the mechanism
Vosoughi, Roy & Aral (2018) attribute the asymmetry to — drops the margin to
+0.002 and the gate fails 5 of 6 seeds. See [CALIBRATION.md](CALIBRATION.md).

**H1 (timing).** Supported in sign and significant: early intervention yields
lower cumulative FPR than late (Δ = −0.055, 95% CI [−0.078, −0.031], d = −0.41).
Under-powered as preregistered (see Part 1).

**H3 (the headline).** The predicted **dissociation does not hold in any
backbone**. Both tones raise EPC well above control (+0.11 and +0.07); neither
suppresses it. In this DGP, norm demonstration dominates crowding-out: watching
*anyone* pay the correction cost makes citizens more willing to pay it, and that
outweighs the responsibility-displacement the aggressive tone produces. The tone
difference is in the predicted direction (aggressive < empathetic on EPC) but is
not significant at 30 seeds/cell.

**H5 (model dependence).** Supported, strongly. Backbone accounts for 29% of EPC
variance; timing 3.4%; tone 1.6%. **Backbone choice dominates intervention design
by roughly an order of magnitude** — the surrogate profiles differ more from each
other than early differs from late. If this survives contact with real model
families, "the right way to intervene" is a per-model question, and the
four-backbone benchmark is the paper's most robust contribution.

**Ablation 1 (payoff visibility).** Behaviour is *not* invariant to the payoff
structure — removing prices raises EPC from 0.100 to 0.319 (d = 2.42) and lowers
FPR. The game structure drives behaviour; agents challenge freely once challenging
is free. The ablation does what it was designed to do.

**Ablation 2 (incentive ratio).** Monotone across the β/γ sweep: FPR 0.545 → 0.490
→ 0.416 and EPC 0.141 → 0.161 → 0.218 as β/γ goes 0.5 → 1 → 3. Note the direction:
raising the conformity weight *strengthens herding*, and with an intervener present
the local majority herds toward correction.

**Scale check (N = 50).** Cell rank agreement with N = 7 is Spearman ρ = 0.80 —
the ordering of timing and tone conditions largely survives the scale change, with
the late arms slightly more polluted at N = 50.

**Trace classifier.** Offline agreement with ground-truth motives is κ = 1.0, but
this is **circular** and proves only that the plumbing works: the surrogate's
traces are templated, so a keyword classifier tuned to them recovers them
perfectly. The plan's real gate — 300 human-coded LLM traces at κ ≥ 0.7 — is
untouched by this and still has to be run.
