# Ethics and responsible release

Plan §11.

**Simulation only.** No human subjects, no live-platform deployment, no
interaction between agents and real users. IRB review is expected to be
unnecessary, to be confirmed with the host institution.

**No anthropomorphic inference.** Agent actions are outputs of statistical models
under a prompt and payoff regime. They are not evidence of belief, intent, or
moral reasoning. Results describe model behaviour under incentives and are **not**
estimates of how humans respond to correction.

This is the most likely reviewer objection to agent-based misinformation work, and
the design answers it *structurally* rather than by disclaimer: both primary
outcomes are actions the agent pays for, read off the action log and the payoff
ledger. The exploratory credence probe is reported as a model self-report and is
never described as a belief.

**Contained misinformation.** All false claims originate in established research
datasets; the pipeline authors no novel misinformation beyond controlled
paraphrase. Real speaker names are removed from all agent-visible text (see
[DATA_CARDS.md](DATA_CARDS.md)). The offline surrogate pool is templated and
non-defamatory by construction.

**On concealment.** The intervener's in-character, undisclosed-AI persona exists
only inside the sandbox. Covert deployment of persuasive AI on real platforms
raises consent and manipulation concerns; findings here are positioned as design
guidance for **transparent, platform-sanctioned** interventions, not as a
deployable persuasion policy.

**Dual use.** Evidence about persuasive timing and tone could inform influence
operations. Mitigations: report aggregate effects rather than optimised persuasion
recipes; centre the release on measurement and defence tooling; complete the venue
ethics checklist. The tone templates in this repository are deliberately generic
slot templates, not tuned persuasive copy.

**Provenance discipline.** Every artefact records whether it came from real
corpora or surrogates, and whether citizens were LLMs or the analytic policy. A
surrogate run is labelled `backbone_kind: surrogate` and the analysis will not
describe it as a model result. This is an ethics property, not just hygiene:
publishing surrogate numbers as an LLM benchmark would be the most likely way this
codebase could mislead.

**Licensing.** Code is MIT. PHEME annotations are CC-BY with tweet content
governed by platform terms — only derived topology is released. LIAR is
distributed for research use. Neither corpus is redistributed here.
