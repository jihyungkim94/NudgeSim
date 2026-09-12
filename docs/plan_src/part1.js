const H = require("./helpers.js");
const { d, p, t, mono, h, bullet, numbered, table, callout } = H;

function evidence(title, lines) {
  return callout(title, lines, { bg: "EAF1F8", color: H.ACCENT });
}

const front = [
  new d.Paragraph({
    children: [t("PROJECT PLAN", { bold: true, size: 20, color: H.MUTED })],
    spacing: { after: 60 },
  }),
  new d.Paragraph({
    children: [t("NudgeSim: Who Pays to Correct?", { bold: true, size: 40, color: H.ACCENT })],
    spacing: { after: 60 },
  }),
  new d.Paragraph({
    children: [t("Timing, Tone, and the Second-Order Dilemma of Counter-Misinformation in LLM Agent Societies",
      { bold: true, size: 26, color: H.ACCENT })],
    spacing: { after: 160 },
  }),
  p([t("A payoff-grounded, preregistered multi-agent testbed for evaluating when and how an embedded AI intervener shifts an agent society from free-riding on unverified content toward costly peer correction.",
    { italics: true, color: H.MUTED })], { after: 240 }),

  table([26, 74], [
    ["Field", "Details"],
    ["Prepared by", "Jihyung Kim — Lecturer, Dept. of AI Convergence, Sahmyook University; M.C.S., University of Illinois Urbana-Champaign (2026)"],
    ["Prepared for", "Pre-PhD Research Project application — Prof. Zhijing Jin, University of Toronto / Max Planck Institute / Vector Institute"],
    ["Proposed collaboration", "First-authored paper by the candidate, with Prof. Jin as senior/advising author"],
    ["Research area", "Multi-agent LLM safety · social dilemmas and cooperation · counter-misinformation intervention design"],
    ["Duration", "4 months (16 weeks) from project start, with a named follow-on phase"],
    ["In-scope deliverable", "Open framework v1.0 + arXiv preprint + workshop paper; main-track submission is the designated immediate follow-on, not an in-scope promise"],
    ["Status", "A reference implementation of the full design exists, passes 149 tests including hand-computed payoff-ledger checks, and reproduces the full grid — including a pilot run against the real LIAR and PHEME-9 corpora — with one command. The 16-week plan (§9) begins from that engine rather than from scratch."],
    [{ text: "Date / version", bold: true }, { text: "September 2026 · V1 (living document — comments welcome)", bold: true }],
  ]),

  h("1. Project Summary", 1),
  p("Accurate information in a social network is a public good: verifying a claim is privately costly, while a less-polluted feed benefits everyone. Sharing engaging but unverified content is therefore a form of free-riding, and correcting someone else's false claim is costly peer sanctioning — which reintroduces the classic second-order free-rider problem: everyone wants the network corrected, nobody wants to pay to correct it."),
  p("NudgeSim places an embedded AI intervener inside exactly this dilemma. We build a seven-agent LLM society — five citizens with explicit per-round payoffs, one malicious disseminator, and one \"Devil's Advocate\" intervener that is exogenously willing to absorb the verification cost — seeded with human-vetted false claims from LIAR and wired according to real rumor-cascade topologies from PHEME. Citizens choose among SHARE, ENDORSE, CHALLENGE, and IGNORE under a payoff function that pays an immediate conformity reward, a delayed veracity reward, and charges a cost for challenging."),
  p("The design crosses intervention timing (early vs. late) × intervention tone (empathetic nudge vs. aggressive debunking) × citizen backbone model (six model families across four vendors), against a no-intervener control, a true-claim placebo arm, and two theoretically motivated ablations. A mixed-society condition additionally seats the full model roster together in one episode, so that backbone effects and second-order free-riding between models can be measured directly rather than inferred from separate single-model runs. The headline outcome is not what agents say they believe. It is what they pay for: the rate at which citizens voluntarily absorb the challenge cost themselves once an intervener is present — whether an exogenous sanctioner catalyzes or crowds out endogenous peer correction."),
  p([
    t("Deliverables: (i) a comparative benchmark of how six model families respond to intervention timing and tone under identical incentives, including whether one model free-rides on another's correction in a mixed society; (ii) NudgeSim, an open, payoff-instrumented testbed released with a one-command reproduction script; and (iii) a preregistered manuscript, positioned for a social-agents workshop and an ICWSM/CSCW or ARR main-track submission."),
  ]),

  h("2. Positioning: What Is Already Done, and What Is Not", 1),
  p("This section is placed before the design because the single largest risk to the project is novelty overlap, and the plan is built around a defensible delta rather than a claim of priority."),
  p("LLM-based counter-misinformation simulation is no longer an empty field. Most directly, MADD (Qiao, Li, Zhou & Hu, Findings of EMNLP 2025; arXiv:2507.16848) simulates disinformation dissemination and correction across a 689-node LLM agent network with bot correctors, manipulating intervention timing at three levels (early / mid / late) and comparing fact-based versus narrative-based correction strategies against an unintervened control. CoSim (arXiv:2605.17353) compares four intervention types including accuracy prompts and fact checking, though it treats timing as an observed outcome rather than a manipulated factor. MOSAIC (arXiv:2504.07830) evaluates competing fact-checking regimes, and a non-cooperative opinion-polarisation game (arXiv:2502.11649) already reports that aggressive debunking can raise polarisation before it stabilises."),
  p("A framing of this project as \"no one has crossed timing with tone\" would not survive review. The delta below is what remains genuinely open."),
  table([20, 38, 42], [
    ["Dimension", "Closest prior work (MADD, CoSim, MOSAIC)", "NudgeSim contribution"],
    ["Incentive structure", "Descriptive social simulation; agents have no explicit payoff and no equilibrium benchmark", "Explicit per-round payoff function; correction is a costly sanction in a public-goods game with a stated stage-game benchmark"],
    ["Design", "Timing and strategy varied, but largely one at a time; no formal factorial or interaction test", "Full timing × tone × model factorial with preregistered interaction contrasts and Dunnett tests against control"],
    ["Headline outcome", "Propagation / belief-like endorsement of the false claim", "Endogenous peer-correction rate — whether citizens themselves start paying the challenge cost (second-order cooperation)"],
    ["Tone axis", "Fact-based vs. narrative-based (an evidence-format contrast)", "Empathetic nudge vs. aggressive debunking (a pragmatic-framing contrast, factual payload held constant)"],
    ["Model coverage", "Typically a single backbone (e.g. DeepSeek-V3)", "Six model families across four vendors benchmarked under identical incentives, both as monocultures and seated together in a mixed society; model-dependence and cross-model free-riding are first-class results"],
    ["Specificity check", "Rarely tested", "True-claim placebo arm measures whether interveners induce indiscriminate skepticism — with a veracity-sensitive intervener, so the arm can actually fail (§5.3)"],
    ["Inference", "Aggregate descriptive comparison", "OSF preregistration, mixed-effects models, bootstrap CIs, Holm correction, effect sizes, and a pre-committed power analysis (§8)"],
  ]),
  h("2.1 Relation to the GovSim / SanctSim / MoralSim / CoopEval line", 2),
  p([t("The section above positions NudgeSim against the counter-misinformation simulation literature. That is the wrong comparison to lead with for this application, because the closest intellectual parent is not MADD — it is the programme this lab has been building."),
  ]),
  p([
    t("GovSim", { bold: true }),
    t(" (Piatti, Jin et al., NeurIPS 2024) asks whether a society of LLM agents can sustain a shared resource, and finds that sustaining it is the exception rather than the rule. "),
    t("SanctSim", { bold: true }),
    t(" (Piedrahita et al. & Jin, COLM 2025) moves from the resource to the institution that protects it, placing LLM agents in a public-goods game with sanctioning and asking who pays to punish. "),
    t("MoralSim", { bold: true }),
    t(" (Backmann et al. & Jin, 2025) separates what an agent's payoff rewards from what its stated ethics require, and watches which wins. "),
    t("CoopEval", { bold: true }),
    t(" (Tewolde, Zhang, Piedrahita, Conitzer & Jin, ICML 2026) turns those designs into a benchmark of cooperation-sustaining mechanisms."),
  ]),
  p([
    t("Read together, that is a single research programme: "),
    t("give LLM agents an explicit payoff, put them in a dilemma with a known human-experimental benchmark, and measure what they do rather than what they say.", { italics: true }),
    t(" NudgeSim is not adjacent to this programme. It is an instance of it, applied to the one public good whose collapse is already a live policy problem."),
  ]),
  table([26, 37, 37], [
    ["", "That line", "NudgeSim"],
    ["The public good", "A shared resource (GovSim) or a contribution pot (SanctSim) — abstract by design, so the dilemma is clean.", "Accurate information in a feed. Verifying is privately costly; a less-polluted feed benefits everyone."],
    ["Free-riding", "Over-extraction; withholding contribution.", "Sharing engaging but unverified content: the conformity reward lands immediately, the veracity penalty is deferred."],
    ["The sanction", "Costly punishment of defectors (SanctSim).", "Costly peer correction — CHALLENGE, priced at κ."],
    ["The open question", "Who pays the sanction, and do reasoning models pay less?", "What happens when an outside agent is stipulated to pay it. Does an exogenous sanctioner build the society's own willingness to correct, or replace it?"],
    ["What is new here", "—", "The second-order question asked of an intervention rather than of the population: not whether agents sanction, but whether being sanctioned FOR changes whether they sanction — and, in a mixed society, whether one model's willingness depends on which models sit beside it."],
  ]),
  p([
    t("This framing also explains the choice of headline outcome. Propagation of the false claim is the obvious dependent variable and the one the misinformation literature reports; "),
    t("endogenous peer correction is the one this line would ask for", { bold: true }),
    t(", because it is second-order cooperation measured as an action the agent pays for. A study that lowered propagation while quietly eroding the society's own willingness to correct would look like a success on the first measure and a failure on the second, and only the second is about cooperation."),
  ]),
  evidence("Feasibility evidence: two results that generalise beyond this study", [
    "The reference implementation — run end to end on an analytic surrogate policy in place of LLM backbones, over the full grid on the real LIAR corpus and all 6,425 real PHEME-9 threads — surfaces two results that speak to the programme this line is building, not only to misinformation.",
    [t("Backbone dominates mechanism. ", { bold: true }),
     t("Across the grid, which model family the citizens run on accounts for roughly thirty per cent of the variance in peer correction, against single digits for the intervention's timing and tone. If that survives contact with real backbones, the model is not a nuisance parameter to be averaged over — it is the largest effect in the room, and a mechanism benchmark that reports a single backbone is reporting one draw from a wide distribution. §5.3 and §5.11 build the design around measuring it properly, including within a single mixed episode.")],
    [t("Welfare has to separate accuracy from conformity and sanction cost. ", { bold: true }),
     t("A naive sum of citizen payoffs ranks every working intervention below doing nothing, because the sanction costs a working intervention generates are exactly what it charges against itself, and conformity alone is maximised by the arm where nobody disagrees. §5.5 reports the veracity component as the primary welfare measure for this reason, with conformity and sanction cost shown alongside it rather than summed in — the standard treatment in experimental public-goods work (Fehr & Gächter, 2002).")],
  ]),
  p("The contribution claim is therefore stated as: prior work has observed that correction timing and framing matter; NudgeSim formalises the setting as a social dilemma, isolates the causal contribution of timing and tone under matched incentives, and asks the question the descriptive literature cannot — whether an AI intervener builds or erodes a community's own willingness to correct, and whether that willingness is a property of the model or of the company it keeps. MADD and CoSim are cited as motivation and as calibration references, not treated as absent."),
];

module.exports = front;
