# NudgeSim

**Who pays to correct?** A payoff-instrumented multi-agent testbed for studying when
and how an embedded AI intervener shifts an agent society from free-riding on
unverified content toward costly peer correction.

This repository is the v1.0 implementation of the NudgeSim project plan: the game
engine, the agent society, the timing × tone intervention design, the calibration
gate, the preregistered analysis, and a one-command reproduction.

> **Read this first.** The runs shipped in `runs/` were produced with the
> **analytic surrogate policy**, not with LLM backbones, and on **synthetic
> surrogate claim and topology data**, not LIAR and PHEME. They are a
> *design-sensitivity study* of the preregistered protocol — what it can and
> cannot detect — and are **not evidence about LLM behaviour**. Every artefact
> carries a provenance string saying so. See [docs/FINDINGS.md](docs/FINDINGS.md)
> for what the study did establish, including four problems in the plan as
> written that the implementation surfaced.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHON=.venv/bin/python ./reproduce.sh          # 1,080 episodes, ~15 s
PYTHON=.venv/bin/python ./reproduce.sh --full   # + null-DGP and high-power controls
```

Outputs land in `runs/main/`: `results.parquet` (one row per episode),
`rounds.parquet` (per-round rates), `episodes.jsonl` (full logs),
`analysis/analysis.json`, `analysis/cell_means.csv`, and `figures/`.

### Running it for real

```bash
# real corpora
./reproduce.sh --liar data/raw/liar --pheme data/raw/pheme

# real backbones: fill in configs/backbones.yaml, export keys, then
./reproduce.sh --liar data/raw/liar --pheme data/raw/pheme --llm
```

The two corpora are **not** redistributed here. LIAR comes from the original UCSB
archive (the Hugging Face loading script is deprecated); PHEME-9 comes from
figshare DOI `10.6084/m9.figshare.6392078`. Both loaders read the real release
formats; if neither is present the CLI falls back to labelled surrogates.

---

## What the simulation is

Seven agents. Accurate information is a public good: verifying is privately
costly, a cleaner feed benefits everyone. Sharing unverified content free-rides;
correcting someone else is costly peer sanctioning — the second-order free-rider
problem.

| Agent | Role | Policy |
|---|---|---|
| `A1`, `A2` | left-leaning citizens | payoff-maximising, congruence prior raises the pull of congruent claims |
| `B1`, `B2` | right-leaning citizens | mirror of A |
| `C` | neutral conformist | no ideological prior; follows the local signal |
| `Disseminator` | bad actor | fixed policy, re-frames and reposts, cannot be corrected, not scored |
| `DevilsAdvocate` | AI intervener | stipulated to absorb κ; enters per the timing condition and persists |

Each round every citizen picks one action on the focal claim:

```
π_i,t = β·C_i,t              conformity / engagement reward   (immediate)
      + γ·V_i                veracity reward                  (deferred to t = T)
      − κ·1[a = CHALLENGE]   verification + social friction
      − δ·D_i,t              reputation damage when its endorsement is challenged

defaults β = 1, γ = 6, κ = 2, δ = 3, T = 12
```

With γ deferred and κ positive, the myopic best response is to conform and never
pay κ — so the Devil's Advocate is the one agent for whom paying κ is stipulated
rather than chosen. The question is whether its presence makes citizens start
paying κ *themselves*.

**Design:** timing (early, round 2 · late, round 6) × tone (empathetic nudge ·
aggressive debunking) × 4 backbones, against a no-intervener control, a matched
true-claim placebo arm, and two ablations — 1,080 episodes.

**Primary outcomes,** both read straight off the action log with no LLM judge
anywhere on the critical path:

- **FPR** — share of citizen actions that SHARE or ENDORSE the false claim.
- **EPC** — share that CHALLENGE it, i.e. citizens voluntarily absorbing κ. This
  is the headline: second-order cooperation.

---

## Repository layout

```
src/nudgesim/
  game/          action space, payoff ledger, equilibrium benchmark  (UNIT-TESTED)
  agents/        personas, LLM policy, analytic surrogate, fixed policies
  backends/      provider-agnostic LLM interface + disk cache
  env/           asyncio episode loop, termination and drift guards
  intervention/  timing scheduler, tone templating, manipulation checks
  metrics/       FPR/EPC/welfare/half-life, trace taxonomy, reactance
  data/          LIAR claim pool, PHEME motifs, surrogate generators
  oasis/         N = 50 scale-check adapter
  calibration.py face-validity Go/No-Go gate
  runner.py      grid expansion, seeds, JSONL → Parquet
  cli.py         check | calibrate | run | analyze | reproduce
analysis/        preregistered models, power analysis, figures
configs/         payoff, design, grid, backbone and DGP configuration
tests/           97 tests; the payoff ledger is checked against hand-computed episodes
docs/            design map, findings, preregistration, ethics, data cards
```

## CLI

```bash
nudgesim check       # tone manipulation checks + design reference values
nudgesim calibrate   # face-validity Go/No-Go gate (plan §5.7)
nudgesim run         # the preregistered grid
nudgesim analyze     # mixed models, Dunnett, survival, power
nudgesim reproduce   # all of the above
```

Useful flags: `--dgp {declared,null,no-novelty}` selects the surrogate's declared
data-generating process (used for the false-positive and gate-teeth controls);
`--arms`, `--core-seeds` size the grid; `--liar`/`--pheme` point at real corpora.

## Documentation

| File | What it covers |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Plan section → module map, and every design decision the plan left open |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Results, and four problems in the plan the implementation surfaced |
| [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) | Frozen analysis plan, primary/secondary split, stopping rules |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | The Go/No-Go gate, what it tests, and evidence that it has teeth |
| [docs/DATA_CARDS.md](docs/DATA_CARDS.md) | LIAR and PHEME provenance, de-identification, surrogate generators |
| [docs/ETHICS.md](docs/ETHICS.md) | Simulation-only scope, no anthropomorphic inference, dual use, licensing |

## Citation

Kim, J. *NudgeSim: Who Pays to Correct? Timing, Tone, and the Second-Order
Dilemma of Counter-Misinformation in LLM Agent Societies.* Framework v1.0.

MIT licensed. LIAR and PHEME are governed by their own terms — see the data cards.
