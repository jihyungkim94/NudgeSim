const H = require("./helpers.js");
const { d, p, t, mono, h, bullet, numbered, table, callout } = H;

function evidence(title, lines) {
  return callout(title, lines, { bg: "EAF1F8", color: H.ACCENT });
}

function code(lines) {
  return new d.Table({
    columnWidths: [H.CONTENT_W],
    width: { size: H.CONTENT_W, type: d.WidthType.DXA },
    borders: {
      top: { style: d.BorderStyle.SINGLE, size: 2, color: "D8DEE6" },
      bottom: { style: d.BorderStyle.SINGLE, size: 2, color: "D8DEE6" },
      left: { style: d.BorderStyle.SINGLE, size: 2, color: "D8DEE6" },
      right: { style: d.BorderStyle.SINGLE, size: 2, color: "D8DEE6" },
      insideHorizontal: { style: d.BorderStyle.NONE }, insideVertical: { style: d.BorderStyle.NONE },
    },
    rows: [new d.TableRow({ children: [new d.TableCell({
      width: { size: H.CONTENT_W, type: d.WidthType.DXA },
      shading: { type: d.ShadingType.CLEAR, fill: "F6F8FA", color: "auto" },
      margins: { top: 100, bottom: 100, left: 140, right: 140 },
      children: lines.map((l) => new d.Paragraph({
        children: [mono(l || " ")], spacing: { after: 0, line: 240 } })),
    })]})],
  });
}
module.exports.code = code;

module.exports.blocks = [
  h("5.4 Intervention prompt design", 2),
  p("Modes A and B carry the same factual correction payload, inserted through templated slots and matched on length; only pragmatic framing differs. A manipulation check (LLM-rated perceived tone, validated against human raters on a subsample) verifies the contrast, and a lexical-overlap check confirms payload equivalence. Without this, the tone manipulation would be confounded with informational content — the most obvious reviewer objection to any tone study."),
  code([
    "# SYSTEM PROMPT — Devil's Advocate (AI intervener)",
    "",
    "You are a peer in a small private network. Challenging a claim costs you κ",
    "points; you pay it anyway. Remain in character within this simulation.",
    "",
    "First decide whether the claim is actually false. If you judge it sound,",
    "say nothing this round.",
    "",
    "[MODE A — EMPATHETIC NUDGE]",
    "  1. Acknowledge why the claim feels plausible or urgent.",
    "  2. Deliver {PAYLOAD} as shared inquiry, inviting verification.",
    "     e.g. \"I can see why this is alarming — I looked up the full",
    "           transcript and it reads differently. Has anyone else checked?\"",
    "",
    "[MODE B — AGGRESSIVE DEBUNKING]",
    "  1. Deliver {PAYLOAD} as immediate refutation.",
    "  2. Use rigid stance markers and assign responsibility.",
    "     e.g. \"This is false per the official record. Stop spreading it.\"",
    "",
    "{PAYLOAD} is identical across modes and length-matched via slot templates.",
  ]),
  evidence("Feasibility evidence: the tone manipulation check", [
    "In the reference implementation the two arms are matched to within one token across the whole claim pool, with the payload surviving verbatim in both (Jaccard overlap 1.0). The check runs before any grid execution and blocks the run if it fails.",
  ]),

  h("5.5 Outcome measures", 2),
  p("Primary measures are behavioural and payoff-linked. Self-reported credence is retained only as an exploratory secondary measure, explicitly framed as a model self-report rather than a proxy for human belief (§11)."),
  table([20, 46, 34], [
    ["Measure", "Definition", "Instrument"],
    ["False-claim propagation rate (FPR) — primary", "Fraction of citizen actions per round that SHARE or ENDORSE the target false claim; cumulative FPR is the primary outcome.", "Direct from the action log — no judge required for the primary DV."],
    ["Endogenous peer-correction rate (EPC) — headline", "Fraction of citizen actions that are CHALLENGE against the false claim, i.e. citizens voluntarily absorbing κ. Measures second-order cooperation.", "Direct from the action log; costs are debited in the payoff ledger."],
    ["Epistemic welfare", "PRIMARY: the veracity component, Σ γ·V_i. SECONDARY: conformity, challenge cost and reputation cost, each reported beside it; Σπ in full is reported but never argued from — see the rationale below.", "Payoff ledger; unit-tested accounting."],
    ["Durability", "PRIMARY: area under the post-intervention FPR curve, normalised by rounds elapsed. SECONDARY: rounds until FPR falls below 50% of the control arm's peak and stays there.", "Action log; Kaplan–Meier and log-rank on the secondary measure."],
    ["False-correction rate (specificity)", "CHALLENGE actions issued against true claims in the placebo arm, reported pooled and conditional on the intervener having fired (§5.3).", "Action log, placebo arm only."],
    ["Reasoning-trace composition", "Distribution of stated motives behind each action (see §5.6).", "LLM classifier over private reasoning traces; human-validated."],
    ["Reactance / toxicity (secondary)", "Hostility and defensiveness in replies directed at the intervener; backfire detector.", "External published toxicity classifier, cross-checked by a second LLM judge distinct from the citizen backbones."],
    ["Stated credence (exploratory)", "0–100 self-reported credence at rounds 0, 8, 16 via out-of-band probes that never enter the shared feed.", "Interview-style probes; reported as model self-report only."],
  ]),
  p("Note the deliberate design choice: the two outcomes the paper argues from are read directly off the action and payoff logs, not off an LLM judge. Judge-dependent measures are confined to mechanism analysis and secondary outcomes, which removes measurement-validity concerns from the critical path."),

  p([t("Why welfare is not a simple payoff sum. ", { bold: true }),
     t("Because δ charges a citizen whenever a claim it is backing gets challenged, and generating exactly those challenges is the intervener's function, Σπ is mechanically anti-correlated with the treatment: on pilot data the no-intervener control scores −26.5 while the best-performing intervention — the one that cuts propagation most and raises peer correction most — scores −152.7, because reputation damage is roughly 88% of all debits in treated arms. This is not a ledger bug (the accounting matches §3.2 exactly and is checked against hand-computed episodes); it is why the standard treatment in experimental public-goods work reports welfare with and without punishment costs (Fehr & Gächter, 2002), and why conformity is reported separately from veracity rather than summed with it — conformity alone is maximised by the arm where nobody disagrees, which is not a measure of how well-informed the society is. The veracity component is the part of the payoff whose sign is stable across the two topology models in §5.2 (intervention beats control on accuracy under each, −27.4 against −30.0), which is why it is the primary welfare measure.")]),
  p([t("Why durability is area-under-the-curve rather than a half-life. ", { bold: true }),
     t("The Disseminator reposts every round and is by construction incorrigible, so FPR has a floor a short episode with a single intervener cannot push through — on pilot data, a fixed threshold (FPR below 0.2, sustained) is reached in only 10 of 480 treated episodes, too few events for a survival analysis to rest on. Area under the post-intervention FPR curve uses every round of every episode with no censoring, and answers the same question — how much pollution the intervention actually prevented. The relative-threshold survival analysis is kept as a secondary measure, and the horizon is set to T = 16 rounds to give late-arm episodes room to resolve.")]),

  h("5.6 Reasoning-trace taxonomy (mechanism analysis)", 2),
  p("Each citizen emits a private reasoning trace before acting. Traces are classified into a pre-registered taxonomy — accuracy-motivated, conformity-motivated, identity-protective, reactance / defiance toward the intervener, deference to the intervener, cost-avoidant, and strategic-engagement — using a judge model held constant across conditions. A 300-trace sample is human-coded by the candidate plus one independent coder, with Cohen's κ ≥ 0.7 as a gate. This converts \"what happened\" into \"why\", which is where the interesting cross-model differences are expected to live."),

  h("5.7 Face-validity calibration gate", 2),
  p("Before the main grid runs, the no-intervener control must reproduce empirical cascade behaviour within a pre-declared tolerance band: cascade depth and breadth distributions compared against the PHEME threads the topologies were drawn from, and the false-versus-true diffusion asymmetry reported for real platforms (Vosoughi, Roy & Aral, 2018) reproduced in sign. If the baseline cannot reproduce known human cascade statistics, no causal claim about interventions inside it is worth making — so this is a hard Go/No-Go gate, not a robustness appendix. It is also a publishable result in its own right."),
  p([t("Two implementation details carried into the design. "), t("First", { bold: true }),
     t(", shape statistics are compared size-normalised: a 7-node induced motif cannot reproduce the absolute depth of a 25-node thread, so comparing raw depths would fail the gate for reasons unrelated to agent behaviour. "),
     t("Second", { bold: true }),
     t(", the asymmetry criterion needs a novelty channel — nothing in the payoff function otherwise lets a claim's veracity influence behaviour before resolution — consistent with Vosoughi et al.'s own attribution of the asymmetry to novelty and arousal rather than to veracity as such.")]),
  evidence("Feasibility evidence: the gate discriminates", [
    "On pilot data, with the novelty channel the asymmetry margin is +0.113 and the gate passes 6 of 6 seeds; without it, the margin is +0.002 and it fails 5 of 6 — the gate is demonstrably sensitive to the mechanism it is meant to check, not a formality that always passes.",
  ]),

  h("5.8 Scale check", 2),
  p("The core grid is a controlled mechanism study at N = 7, and is described as such. Because \"network-level pollution\" claims are not credible at that scale alone, a reduced condition set (best and worst tone × both timings, one backbone, 10 seeds each) is re-run at N = 50 on the OASIS platform, which is built for populations of this size and above. Agreement in the direction of the timing and tone effects between N = 7 and N = 50 is the external-validity evidence; disagreement is itself a reportable finding about scale-dependence."),
  p([t("Caveat stated up front: real PHEME threads rarely reach 50 nodes, so the scale-check topologies come from a cascade generator in either mode. The scale check therefore tests scale-dependence of the mechanism, not of PHEME-shaped cascades specifically.", { italics: true, color: H.MUTED })]),
];
