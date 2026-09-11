# Findings

Two kinds of result are reported here, and they are not interchangeable.

**Part 1 — problems in the plan as written.** Findings about the *design*,
established by implementing it. They hold regardless of what runs the engine,
and they are the part of this document that matters most.

**Part 2 — the design-sensitivity run.** The full preregistered grid (1,080
episodes) executed against a declared analytic data-generating process. These
numbers are properties of that DGP. They are **not evidence about LLM
behaviour**.

**Data status.** Both corpora are now real: the **LIAR corpus** (Wang, 2017 —
all 12,836 statements, de-identified) and **PHEME-9** (Kochkina et al., 2018 —
all 6,425 threads, derived topology only). `runs/real_main` is the real-data
run. Earlier runs on surrogate data are kept for comparison, and every artefact
records its provenance.

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
`welfare_ex_sanctions` alongside `welfare_total`.

*Refinement, after finding 5.* Removing sanction costs is necessary but not
sufficient. Once the visibility model was corrected and neighbourhoods became
denser, the conformity term grew from ~35 to ~90 points and swamped everything
else — and the control arm maximises conformity, because an unchallenged claim
produces unanimity. So `accuracy + conformity` reverses and favours doing
nothing again:

| | accuracy | conformity | acc + conf |
|---|---:|---:|---:|
| Control | **−30.0** | 97.7 | 67.7 |
| Early · empathetic | **−27.4** | 89.6 | 62.2 |

The accuracy term is the only component whose sign is stable across both
visibility models (intervention beats control on accuracy under each). That is
the epistemically meaningful quantity: conformity is a private engagement
payoff, not a measure of how well-informed the society is. **Epistemic welfare
should be the veracity component**, with conformity and sanction costs reported
beside it as private payoff and enforcement cost. Summing them was the original
mistake; summing a subset of them is a smaller version of the same mistake.

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

### 5. Real PHEME topologies cannot carry the study's mechanism

This one only appeared when the actual corpus arrived, and it is the most
serious of the five.

Plan §5.2 samples a connected 7-node motif from a PHEME reply tree and maps the
Disseminator to the cascade root. Run against the real release, that procedure
produces a **broadcast star** 69% of the time — a source tweet with six direct
replies and no other edges. Real PHEME cascades are broad and shallow (mean
branching 4.46, mean depth 3.59), so a 7-node neighbourhood of the root is
almost always the root plus six leaves.

In a star rooted at the Disseminator, every citizen's only neighbour is the
Disseminator. The consequences are fatal to the design, not merely inconvenient:

| | surrogate topology | **real PHEME** |
|---|---:|---:|
| Motifs where the intervener reaches no citizen | 8 / 201 | **138 / 201 (69%)** |
| Motifs where no citizen can see any other citizen | 64 / 201 | **178 / 201 (89%)** |
| Mean intervener reach in the run | 2.18 | **1.09** |
| Distinct motifs usable after the reach filter | 113 | **63** |

With no citizen-to-citizen edge there is **no local majority** to form among the
five citizens, and **no citizen can ever observe another paying the correction
cost** — which is the demonstration channel EPC exists to measure. The study's
central mechanism is structurally absent from 89% of the real topologies.

The run still completes, and effects survive in attenuated form (intervention →
EPC d = 0.69 against 0.80; control FPR 0.753 against 0.657; EPC roughly a third
lower throughout), but on a motif set that has had 69% of itself discarded and
is not representative of what remains.

**Neither existing gate catches this.** The §5.7 calibration gate passes
comfortably on real PHEME — better than on surrogate topology, since the
simulated cascades are now being compared against the very threads they were
drawn from (depth-ratio delta 0.027 against 0.199). The gate tests cascade
*shape* and *diffusion asymmetry*; neither is sensitive to whether the agents
can see one another. The intervener-reach filter catches only the intervener's
half of the problem and says nothing about citizen-to-citizen visibility.

*Root cause.* `structure.json` is a **reply tree**, not a **visibility graph**.
It records who replied to whom. The plan needs who can *see* whom, and in a real
conversation thread everyone reading it sees the sibling replies too. Treating
the reply tree as the visibility graph is the error.

*This is a design decision for the author, not an implementation detail*, so the
options are recorded rather than chosen:

1. **Add sibling visibility.** Replies to the same parent are mutually visible,
   which is what a threaded conversation view actually shows. Minimal change,
   well motivated, and turns stars into the dense neighbourhoods the design
   assumes — arguably too dense.
2. **Do not root motifs at the cascade root.** Sample a connected subgraph
   anywhere in the thread and map the Disseminator to its highest-degree node.
   Keeps the reply tree as the visibility graph but abandons §5.2's "Disseminator
   at the root".
3. **Declare the constraint and filter on it.** Keep the design and require a
   minimum citizen-to-citizen edge count at sampling time, reporting how much of
   PHEME that discards. Honest, but 89% attrition makes the surviving sample hard
   to defend.

**Resolved: bounded attention.** Options 1 and 2 were both measured on the real
release before either was adopted, and neither works alone:

| visibility model | reach = 0 | no citizen visibility | complete graph K7 | density |
|---|---:|---:|---:|---:|
| Reply tree, rooted *(the plan as written)* | 68.2% | 92.3% | 0% | 0.29 |
| Option 1 — all siblings visible | 0% | 0% | **68.3%** | 0.90 |
| Option 2 — unrooted motifs | 54.8% | 77.0% | 0% | 0.29 |
| Option 1 + 2 combined | 0% | 0% | **54.1%** | 0.81 |
| **Bounded attention (w = 1)** | **0%** | **0%** | **0%** | **0.50** |

Option 1 fixes visibility by destroying locality: in a star every reply is a
sibling of every other, so the motif becomes the complete graph and "local
majority" becomes global majority. Option 2 leaves 77% of motifs with no citizen
visibility at all.

A thread view is neither extreme. It is ranked and truncated: a reader sees the
source plus the replies near their own, not all fifty. Modelling that — each
reply linked to the `w` siblings following it in display order — is one
parameter, and at `w = 1` it satisfies every criterion on the real release with
**no thread discarded**: no motif is a star, none is complete, mean density
0.50, and the reach filter becomes a no-op (201 of 201 motifs kept, against
63 before).

Effects strengthen accordingly, because the mechanism can now actually operate:

| | reply tree | bounded attention |
|---|---:|---:|
| Distinct motifs used | 63 | **176** |
| Mean intervener reach | 1.09 | **2.22** |
| Intervention → EPC | d = 0.69 | **d = 0.93** (n = 19 suffices) |
| H1 timing → FPR | d = −0.38 (n = 105) | **d = −0.61** (n = 43) |

H1 moves from needing 3.5× the preregistered sample to needing 1.4×. H3 is
unchanged in character (d = −0.17, n ≈ 580): its smallness is a property of the
two opposing channels, not of the topology.

`nudgesim check` reports citizen-to-citizen visibility alongside intervener
reach, so this class of failure is visible before a grid runs rather than after.

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

0. **The intervener's reach and the citizens' visibility were never measured.**
   The topology looked fine by every statistic the pipeline computed, while 89%
   of motifs made the study's mechanism impossible. See finding 5.
1. **Claim length decided whether an episode was truncated.** The degeneracy
   guard compared whole utterances, and an utterance that quotes the claim is
   mostly claim. On the real LIAR pool, whose statements average 18 words
   against 13.6 in the synthetic one, that flagged 30% of episodes as
   degenerate and silently truncated them — while flagging none on the shorter
   pool. The Disseminator is *specified* to repost the same claim every round,
   so the guard was firing on the behaviour the design asks for. It now judges
   the agent's own framing with the quoted claim removed.
2. **Every episode in the grid ran on a `pants-fire` claim.** The §5.9 severity
   stratification is implemented as a cursor over strata, and the runner drew one
   claim per episode — so it returned stratum 0 every time. The grid looked
   healthy and the numbers looked reasonable; the stratification simply never
   happened. Fixed, with a regression test asserting a balanced three-way split.
3. **A silent endorser was never charged reputation damage.** Damage was applied
   only to agents that *acted* that round, exempting exactly the agents whose
   endorsements were most exposed — those who said their piece and went quiet.
4. **An anti-runaway fix destroyed the effect it was protecting.** Saturating the
   norm-salience channel stopped a genuine feedback explosion in *peer*
   correction, but applying the same transform to the *policing* channel
   compressed a three-fold difference in enforcement intensity into a few
   hundredths of a utility point — flattening the empathetic-versus-aggressive
   contrast the study exists to measure. Policing comes from one agent at one
   action per round, so exponential decay already bounds it; it needed
   normalising, not saturating.
5. **LIAR rows were silently merging.** LIAR statements contain unbalanced double
   quotes, and Python's default CSV quoting swallowed line breaks at them,
   merging rows and bleeding later columns into statement text. `QUOTE_NONE`
   fixes it; a test now asserts all 12,836 published rows parse.
6. **De-identification was both too weak and too strong.** The speaker column
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

---

## The revised plan

[`Project_Plan_NudgeSim_v2.3.docx`](Project_Plan_NudgeSim_v2.3.docx) is the
project plan with all five problems fixed in place. §0 of that document carries
a revision table mapping each change to the evidence above. The substantive
edits are in §5.2 (intervener placement and reach), §5.3 (veracity-sensitive
intervener), §5.5 (welfare and durability measures), §5.9 (two-tier sampling
protocol), §8 (power statement and conditional contrasts), §9 (a Week-8 power
gate) and §12 (two new risks).

The headline change to the protocol: the grid grows from 1,080 to 2,780
episodes, but not by scaling everything 9× — the core grid stays at 30 seeds
because that already powers what it is for, and a focused 150-seed arm covers
only the two contrasts that need it. That is ~2.6× the compute rather than ~9×,
with a pre-declared reduction ladder if the budget binds.


---

## Where this sits in the GovSim / SanctSim / MoralSim / CoopEval line

Two of the findings above are not about misinformation at all. They are about
the class of study this project belongs to, and they would apply to any design
in it.

**Backbone dominates mechanism.** Which model family the citizens run on
accounts for roughly a third of the variance in peer correction; the
intervention's timing and tone account for single digits. If that survives
contact with real backbones, then in this class of study the model is not a
nuisance parameter to average over — it is the largest effect present, and a
mechanism result reported on a single backbone is one draw from a wide
distribution.

**Summing a payoff function is not a welfare measure.** Total payoff ranked
every working intervention below doing nothing, because the sanction costs it
charges are precisely what a working intervention generates. Removing sanctions
was not enough either: conformity then dominated, and conformity is maximised by
the arm in which nobody disagrees. Only the veracity term is stable in sign.
Any design in this line that reports Σπ as welfare inherits the same problem.

Plan §2.1 states the relationship in full: NudgeSim is an instance of that
programme applied to information as the public good, asking the second-order
question of an *intervention* rather than of a population — not whether agents
sanction, but whether being sanctioned *for* changes whether they sanction.
