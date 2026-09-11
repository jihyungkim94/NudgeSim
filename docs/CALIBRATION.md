# The face-validity calibration gate

Plan §5.7. Before the main grid runs, the no-intervener control must reproduce
empirical cascade behaviour within a pre-declared tolerance band. If the baseline
cannot reproduce known cascade statistics, no causal claim about interventions
inside it is worth making — so this is a hard Go/No-Go, not a robustness
appendix.

## What it tests

**1. Cascade shape.** Depth, maximum breadth and structural virality of the
simulated diffusion tree, against the source threads the topologies were drawn
from.

Compared on **size-normalised** statistics. A 7-node induced motif cannot
reproduce the absolute depth of a 25-node thread, so comparing raw depths would
fail the gate for a reason that has nothing to do with agent behaviour. What can
be asked is whether the cascade the agents produce has the same shape *relative to
the structure available to it* as the real thread did.

**2. Diffusion asymmetry.** The false-versus-true asymmetry reported for real
platforms (Vosoughi, Roy & Aral, 2018), reproduced in sign, with a declared
minimum margin.

The diffusion tree is reconstructed from the action log: an agent joins the
cascade the first time it propagates *after* one of its neighbours has, that
neighbour becoming its parent. Agents that propagate spontaneously in round 0 are
not yet part of a cascade but join later if they propagate downstream of someone.

## The gate has teeth

The asymmetry criterion is the sharp one, and it is not free. Nothing in the
plan's payoff function lets a claim's veracity influence behaviour before
resolution — agents start at p(false) = 0.5 whatever the claim is — so without an
additional channel, false and true claims diffuse identically and the criterion
fails.

Vosoughi et al. attribute the real asymmetry to **novelty and the emotional
response it provokes**, not to veracity as such: nobody forwards a claim because
it is false, they forward it because it is startling. `novelty_engagement` models
exactly that, as a declared property of the claim *text* (scaled by LIAR severity
stratum) — never a leak of the veracity label.

Six seeds, 200 control episodes per side, everything else identical:

| DGP | GO verdicts | Mean propagation margin |
|---|---:|---:|
| Declared (novelty channel on) | **6 / 6** | +0.113 |
| `--dgp no-novelty` (channel off) | 1 / 6 | +0.002 |

The single GO in the bottom row is noise clearing a 0.02 threshold; the mean margin
is indistinguishable from zero. The gate rejects a baseline that cannot reproduce
the asymmetry, and accepts one that can.

## Reproducing

```bash
nudgesim calibrate --calibration-episodes 200 --out runs/calibration
nudgesim --dgp no-novelty calibrate --calibration-episodes 200 --out runs/cal_nonovelty
```

Exit code 0 is GO, 2 is NO-GO; `reproduce.sh` stops before the grid on NO-GO.

## If the gate fails with real backbones

The plan's stated remediation (§12) is to adjust neighbourhood visibility and
disseminator persistence and re-run; if it still fails, the paper narrows to a
within-simulation mechanism claim and drops network-level language. Both levers
are configuration (`memory_window`, the Disseminator's reframe policy), so the
remediation does not require a code change.
