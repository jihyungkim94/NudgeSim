# NudgeSim: Who Pays to Correct?

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-149%20passing-brightgreen.svg)
![Model benchmark](https://img.shields.io/badge/model%20benchmark-not%20yet%20run-orange.svg)

![Does an AI corrector build the norm, or replace it?](imgs/nudgesim_question.png)

## What happens when an AI corrector joins a society that won't correct itself?

Accurate information is a public good: verifying costs you, a cleaner feed
benefits everyone. So sharing unverified content is free-riding, and correcting
somebody else is costly peer sanctioning — nobody wants to pay for it.

NudgeSim drops an AI intervener into exactly that dilemma and measures whether
it **catalyses or crowds out** the society's own willingness to pay. Seven
agents, one false claim from **LIAR**, twelve rounds, on network topologies
taken from real rumour cascades in **PHEME-9**. The headline outcome is not what
an agent says it believes; it is what it pays for.

## 💡 Key Discoveries

**Reply trees are not visibility graphs, and the difference is the effect
size.** Treating reply edges as visibility edges leaves 88.6% of neighbourhoods
with no citizen able to see any other — deleting the mechanism the study
measures. Across three translations of the same corpus on the same seeds, the
intervention effect runs +0.041, +0.080 and +0.095.

**Summing a payoff function is not a welfare measure under costly sanctioning.**
The intervention cuts propagation and more than triples peer correction, and
total payoff still ranks it below doing nothing — the sanction costs it charges
are exactly what a working intervention generates.

**A monoculture backbone factor cannot answer its own question.** One model per
episode cannot measure whether one model free-rides on another's correction,
because there is no other model present. A mixed-society arm seats the whole
roster in one episode and rotates which model holds which seat.

## 🛠️ Installation

```bash
git clone https://github.com/jihyungkim94/NudgeSim.git && cd NudgeSim
python -m venv .venv && .venv/bin/pip install -e ".[dev,llm]"
```

## 🔑 API Configuration

> **The model benchmark has not been run.** It needs API credentials for the
> hosted backbones and a GPU or inference host for the open-weights ones, and
> neither is in place yet. That is the only thing standing between this
> repository and the model results — the LLM path itself is implemented and
> verified end to end.

```bash
cp .env.example .env        # ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL
```

The roster is declared in [`configs/backbones.yaml`](configs/backbones.yaml) and
priced in advance by `nudgesim cost`: 60 calls an episode, ~613 input tokens a
call, 23,250 calls per backbone over the grid.

**No key?** `scripts/mock_llm_server.py` is a local OpenAI-compatible endpoint
that deliberately returns malformed replies — prose instead of JSON, markdown
fences, wrong-case action words — so prompt assembly, transport, concurrency,
parsing, repair, caching and cost accounting can all be exercised without one.
Against it the pipeline repairs ~11% malformed replies without a crash.

## 🚀 Quick Start

```bash
# the committed runs: analytic surrogate policy, no network calls, ~25 s
PYTHON=.venv/bin/python ./reproduce.sh              # Windows: .\reproduce.ps1

# real corpora
nudgesim fetch-data                                 # LIAR, verified on download
python scripts/extract_pheme.py ~/Downloads/<figshare-archive>

# language models in the seats
nudgesim --liar data/raw/liar --pheme data/raw/pheme \
  --models openai:gpt-4o,anthropic:claude-opus-5 run --out runs/llm

# open weights through the same interface
./scripts/serve_vllm.sh Qwen/Qwen3-30B-A3B
nudgesim --models openai:Qwen/Qwen3-30B-A3B \
  --backend-url http://localhost:8000/v1 run --out runs/vllm
```

## ⚙️ The design

![Corpora, society, design, outcomes](imgs/nudgesim_pipeline.png)

Five citizens with explicit per-round payoffs, one disseminator that cannot be
corrected, and one Devil's Advocate stipulated to absorb the correction cost κ:

```
π_i,t = β·C_i,t − κ·1[a = CHALLENGE] − δ·D_i,t + γ·V_i
        conformity   correction cost   reputation    veracity (deferred to T)

defaults β = 1, γ = 6, κ = 2, δ = 3, T = 12
```

With γ deferred and κ positive, the myopic best response is to conform and never
pay κ. The question is whether the intervener's presence makes citizens start
paying it *themselves*.

Timing (early · late) × tone (empathetic nudge · aggressive debunking) ×
backbone, against a no-intervener control, a true-claim placebo arm, two
ablations, a perturbation arm and the mixed-society arm — 1,550 episodes.

Two outcomes, both read off the action log with no model judge on the critical
path: **FPR**, how far the false claim spreads, and **EPC**, how often citizens
pay κ themselves.

## 📊 Output & Analysis

![Measured on the analytic surrogate over real LIAR and PHEME](imgs/nudgesim_measured.png)

Runs land in `runs/main/`: `results.parquet` (one row per episode),
`rounds.parquet`, `episodes.jsonl`, `analysis/`, and `figures/`.

The committed runs use the **analytic surrogate policy**, not LLM backbones.
Both corpora are real, and the runs are a design-sensitivity study of the
protocol — what it can and cannot detect. They are **not evidence about LLM
behaviour**: every artefact carries a provenance string, a run with `--models`
is tagged `backbone_kind: llm` and one without as `surrogate`, and the analysis
refuses to describe a surrogate run as a model result.

```bash
nudgesim check | calibrate | run | analyze | cost | reproduce
```

## 📁 Project Structure

```
src/nudgesim/
  game/          action space, payoff ledger, equilibrium benchmark (UNIT-TESTED)
  agents/        personas, LLM policy, analytic surrogate, fixed policies
  backends/      provider-agnostic LLM interface + disk cache
  env/           asyncio episode loop, termination and drift guards
  intervention/  timing scheduler, tone templating, manipulation checks
  metrics/       FPR/EPC, welfare, durability, trace taxonomy
  data/          LIAR claim pool, PHEME motifs, surrogate generators
  cost.py        prompt metering and budget forecasting
  runner.py      grid expansion, seeds, JSONL → Parquet
analysis/        preregistered models, power analysis, figures
configs/         payoff, design, grid and backbone configuration
scripts/         PHEME extractor, mock LLM server, vLLM launcher
tests/           149 tests; the payoff ledger is checked by hand-computed episodes
docs/            design map, findings, preregistration, ethics, data cards
```

Full write-ups live in [`docs/`](docs/): [DESIGN](docs/DESIGN.md),
[FINDINGS](docs/FINDINGS.md), [PREREGISTRATION](docs/PREREGISTRATION.md),
[CALIBRATION](docs/CALIBRATION.md), [DATA_CARDS](docs/DATA_CARDS.md),
[ETHICS](docs/ETHICS.md), and the research plan
[`Project_Plan_NudgeSim_V1.docx`](docs/Project_Plan_NudgeSim_V1.docx).

## 📄 Citation

```bibtex
@misc{kim2026nudgesim,
  title  = {NudgeSim: Who Pays to Correct? Timing, Tone, and the Second-Order
            Dilemma of Counter-Misinformation in LLM Agent Societies},
  author = {Kim, Jihyung},
  year   = {2026},
  url    = {https://github.com/jihyungkim94/NudgeSim}
}
```

MIT licensed. LIAR and PHEME are governed by their own terms — see the data cards.
