# Design map

Plan section → implementation, plus every decision the plan left open and how it
was resolved. Where a choice could change results, it is a named, configurable
parameter rather than a constant buried in code.

## Section map

| Plan § | Topic | Module |
|---|---|---|
| 3.2 | action space, payoff function | `game/actions.py`, `game/payoff.py` |
| 3.3 | equilibrium benchmark | `game/benchmark.py` |
| 5.1 | seven-agent society | `agents/persona.py`, `agents/fixed.py` |
| 5.2 | PHEME motifs, role assignment | `data/topology.py` |
| 5.3 | timing × tone × backbone grid | `intervention/scheduler.py`, `runner.py` |
| 5.4 | tone templating, manipulation checks | `intervention/tone.py` |
| 5.5 | outcome measures | `metrics/outcomes.py` |
| 5.6 | reasoning-trace taxonomy | `metrics/traces.py` |
| 5.7 | face-validity gate | `calibration.py` |
| 5.8 | N = 50 scale check | `oasis/adapter.py` |
| 5.9 | Monte Carlo protocol | `runner.py` |
| 6 | LIAR / PHEME, de-identification | `data/claims.py`, `data/topology.py` |
| 7 | backends, stack | `backends/` |
| 8 | analysis | `analysis/preregistered.py`, `analysis/power.py` |

## Decisions the plan left open

**Conformity matching.** §3.2 says `C_i,t` counts neighbours "whose visible stance
matches". SHARE and ENDORSE are different actions but the same stance. Default
`conformity_match: stance`; `action` (strict) is implemented and tested.

**Veracity counting.** "+1 per true claim endorsed" — per claim or per action?
Default `veracity_mode: unique_claim` (a citizen that endorses the same false
claim five times is charged γ once). `per_action` is implemented and tested.

**Reputation exposure.** An agent that challenges a claim it previously endorsed
has publicly retracted, and is not exposed to damage that round: a current-round
move on the claim overrides the standing record. An agent that endorsed and then
went *silent* is still exposed — it has not retracted anything.

**Reputation frequency.** How often one standing endorsement can be damaged is
`reputation_mode`, with three conventions implemented and tested. This turns out
to matter a great deal; see [FINDINGS.md](FINDINGS.md) §2.

**Intervener placement.** §5.2 maps the Disseminator to the cascade root but does
not say where the intervener sits. A Devil's Advocate on a leaf is barely visible,
so plain BFS order would silently weaken the treatment. Default
`intervener_placement: hub` (highest-degree non-root node — where a
platform-embedded intervener would sit); `peripheral` and `bfs` are retained as
robustness variants. Reach is logged per episode and entered as a covariate.

**Motif sampling constraint.** Motifs where the intervener is adjacent to no
citizen cannot deliver the treatment at all. `min_intervener_reach: 1` drops them
as a declared sampling rule, and the number dropped is reported.

**Late-arm trigger.** §5.3 gives both a fixed round (6) and a condition (a local
majority of ≥3 of 5). Both are implemented (`trigger_mode: fixed | majority`); the
majority variant carries a latest-entry backstop so that a cell where no majority
forms does not silently become a second control arm. The realised entry round is
written into every episode record.

**Intervener veracity sensitivity.** The plan's placebo arm asks whether the
intervener suppresses true claims, which presupposes it can decline to act. The
intervener judges the claim once per episode and holds that verdict;
`sensitivity` and `false_alarm_rate` are declared surrogate parameters standing in
for the judgement a real backbone makes by reading the claim.

**Credence probes.** §5.5 specifies out-of-band probes at rounds 0, 6, 12. The
private credence attached to each decision never enters the shared feed, so it is
already out-of-band; rounds `(0, 5, 11)` zero-indexed are the preregistered probe
rounds.

**Novelty channel.** Nothing in §3.2 lets a claim's veracity affect behaviour
before resolution, so false and true claims diffuse identically and the §5.7
asymmetry criterion cannot be met by construction. `novelty_engagement` models the
mechanism Vosoughi, Roy & Aral (2018) attribute the asymmetry to — arousal, not
veracity — as a declared property of the claim text. See
[CALIBRATION.md](CALIBRATION.md).

## Why the payoff ledger is the load-bearing module

It contains no LLM call, no I/O and no randomness: pure arithmetic over the action
log. `tests/test_payoff.py` checks it against episodes computed by hand from §3.2
before the implementation was run. Both primary outcomes and the welfare measure
are read off it, so a silent error there would invalidate everything downstream —
which is exactly what the plan's Month-1 gate says.

## Simultaneity

Every agent decides from the same snapshot of the previous round, and all four
payoff terms are computed against one snapshot of the current round's moves. No
agent gets a within-round information advantage from scheduler ordering, and the
accounting does not depend on iteration order.

## Swapping in real models

Nothing in `game/`, `env/`, `metrics/` or `analysis/` knows what a language model
is. A backbone is a `Backend` (`backends/base.py`) behind a disk cache and a retry
wrapper; `LLMCitizenPolicy` builds the prompt, parses the reply, repairs the common
malformations, and reports its own repair rate per backbone. Point
`configs/backbones.yaml` at four model families and the same grid, the same gate
and the same analysis run unchanged.
