# NudgeSim

**Who pays to correct?** A payoff-instrumented multi-agent testbed for studying when
and how an embedded AI intervener shifts an agent society from free-riding on
unverified content toward costly peer correction.

This repository is the v1.0 implementation of the NudgeSim project plan: the game
engine, the agent society, the timing × tone intervention design, the calibration
gate, the preregistered analysis, and a one-command reproduction.

> **Read this first.** The runs shipped in `runs/` were produced with the
> **analytic surrogate policy**, not with LLM backbones. Both corpora are real:
> claims from **LIAR**, topologies from **PHEME-9**. These runs are a
> *design-sensitivity study* of the preregistered protocol — what it can and
> cannot detect — and are **not evidence about LLM behaviour**. Every artefact
> carries a provenance string saying so. See
> [docs/FINDINGS.md](docs/FINDINGS.md) for what the study did establish,
> including five problems in the plan as written and seven silent bugs the
> implementation surfaced.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHON=.venv/bin/python ./reproduce.sh          # 1,550 episodes, ~15 s
PYTHON=.venv/bin/python ./reproduce.sh --full   # + the validation triad and high-power controls
```

On Windows, `reproduce.ps1` is the same steps with the same run names:

```powershell
python -m venv .venv; .venv\Scripts\pip install -e ".[dev]"
.\reproduce.ps1
.\reproduce.ps1 -Full -Liar data\raw\liar -Pheme data\raw\pheme
```

Outputs land in `runs/main/`: `results.parquet` (one row per episode),
`rounds.parquet` (per-round rates), `episodes.jsonl` (full logs),
`analysis/analysis.json`, `analysis/cell_means.csv`, and `figures/`. Only the
small reviewable artefacts are committed; the bulk logs regenerate in seconds.

## The paper

`paper/` holds the manuscript. Every table in it is generated from the run
artefacts, so no number is typed by hand:

```bash
PYTHON=.venv/bin/python ./reproduce.sh --full --liar ... --pheme ...
.venv/bin/python paper/build_tables.py    # tables/*.tex from runs/
.venv/bin/python paper/check.py           # structural check on the source
```

No LLM run has been executed yet, so the sections reporting model behaviour are
marked `[PENDING LLM RUN]` and left empty rather than estimated. See
[paper/README.md](paper/README.md).

### Getting the corpora

```bash
nudgesim fetch-data          # downloads and verifies LIAR; prints PHEME steps
```

Neither corpus is redistributed here.

- **LIAR** is fetched automatically and checked against the published release:
  12,836 rows and the exact six-way label distribution. A mirror that does not
  reproduce those fails loudly rather than quietly changing results.
- **PHEME-9** is a manual download (figshare DOI `10.6084/m9.figshare.6392078`,
  ~2 GB, covered by Twitter's content terms). Unpack it so the layout is
  `data/raw/pheme/<event>/{rumours,non-rumours}/<thread>/structure.json`. Only
  each thread's reply tree is read — tweet text is never loaded, and node ids are
  stripped during motif extraction, so only derived topology reaches any artefact.

The claim pool and the topology source are independent, so LIAR + surrogate
topology is a valid, clearly-labelled intermediate configuration.

```bash
./reproduce.sh --liar data/raw/liar                        # real claims
./reproduce.sh --liar data/raw/liar --pheme data/raw/pheme # both corpora
./reproduce.sh --liar data/raw/liar --pheme data/raw/pheme --llm  # + real models
```

Citizens run on the analytic surrogate unless `--models` is given:

```bash
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY
nudgesim --liar data/raw/liar --pheme data/raw/pheme \
  --models openai:gpt-4o-mini,anthropic:claude-haiku-4-5-20251001 \
  run --out runs/llm

# open weights behind vLLM, or any OpenAI-compatible endpoint
nudgesim --models openai:Qwen/Qwen2.5-7B-Instruct \
  --backend-url http://localhost:8000/v1 run --out runs/vllm
```

A run with `--models` is recorded as `backbone_kind: llm`; without it, as
`surrogate`. The analysis will not describe a surrogate run as a model result.

**No key? Verify the plumbing first.** `scripts/mock_llm_server.py` is a local
OpenAI-compatible endpoint that returns deliberately messy replies (fenced JSON,
prose, wrong-case action words). Everything except the model itself — prompt
assembly, HTTP transport, concurrency, parsing, repair, caching, cost
accounting — runs against it:

```bash
python scripts/mock_llm_server.py --port 8079 &
OPENAI_API_KEY=not-needed nudgesim --models openai:mock-a --backend-url http://127.0.0.1:8079/v1 \
  run --out runs/smoke --arms core --core-seeds 5
```

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
true-claim placebo arm, and two ablations — 1,550 episodes.

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
tests/           144 tests; the payoff ledger is checked against hand-computed episodes
docs/            design map, findings, preregistration, ethics, data cards
```

## CLI

```bash
nudgesim fetch-data  # download + verify LIAR; instructions for PHEME
nudgesim check       # tone manipulation checks + design reference values
nudgesim calibrate   # face-validity Go/No-Go gate (plan §5.7)
nudgesim run         # the preregistered grid
nudgesim analyze     # mixed models, Dunnett, survival, power
nudgesim reproduce   # all of the above
```

Useful flags: `--dgp {declared,strong,null,no-novelty}` selects the surrogate's
declared data-generating process (used for the sensitivity, false-positive and
gate-teeth controls);
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
