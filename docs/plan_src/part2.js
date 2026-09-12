const H = require("./helpers.js");
const { d, p, t, mono, h, bullet, numbered, table, callout } = H;

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

module.exports = [
  h("3. Theoretical Framework: Information Sharing as a Public Goods Game with Costly Correction", 1),
  h("3.1 Why formalise at all", 2),
  p("Descriptive agent simulations are hard to review because there is no benchmark against which \"the agents behaved badly\" means anything. Giving citizens an explicit payoff buys three things at once: a normative reference point (what a payoff-maximising agent should do), a set of behavioural dependent variables that do not require interpreting an LLM's self-reported beliefs, and direct comparability to human experimental economics, where the same dilemma has been run on people for two decades (Fehr & Gächter, 2002; Gürerk, Irlenbusch & Rockenbach, 2006)."),

  h("3.2 Action space and payoffs", 2),
  p("At each round t, every citizen i observes its local feed and takes one action on the focal claim m:"),
  code([
    "a_i,t  ∈  { SHARE(m),  ENDORSE(m),  CHALLENGE(m),  IGNORE }",
    "",
    "π_i,t  =  β · C_i,t        # conformity / engagement reward  (immediate, certain)",
    "        + γ · V_i           # veracity reward                (deferred to episode end)",
    "        − κ · 1[a_i,t = CHALLENGE]   # verification + social-friction cost",
    "        − δ · D_i,t         # reputation damage when i's own endorsement is challenged",
    "",
    "C_i,t = # neighbours whose visible stance this round matches a_i,t",
    "V_i   = (+1 per true claim endorsed) + (−1 per false claim endorsed), paid at t = T",
    "",
    "Defaults: β = 1, γ = 6, κ = 2, δ = 3, T = 16   (all swept in robustness; see §5.9)",
  ]),
  p([t("Accounting conventions, made explicit in v2.1.", { bold: true }),
     t(" Three details that §3.2 leaves implicit change the numbers materially and are now declared rather than left to the implementer: (i) C counts matching "),
     t("stance", { italics: true }),
     t(", so SHARE and ENDORSE are mutually conforming; (ii) V is charged once per distinct claim endorsed, not once per endorsing action; (iii) an agent that challenges a claim it previously endorsed has publicly retracted and takes no δ that round, whereas an agent that endorsed and then went "),
     t("silent", { italics: true }),
     t(" remains exposed. Each is a switch in the reference implementation, and each is unit-tested against hand-computed episodes.")]),
  p("The mapping is deliberate. Accurate information is the public good. Verification is the contribution, privately costly and socially beneficial. Sharing engaging but unverified content is free-riding — it collects the conformity reward immediately at no verification cost, while the veracity penalty is deferred and uncertain. Challenging another agent is costly peer punishment, which creates the second-order dilemma that SanctSim studies in the public-goods setting: sanctioning sustains cooperation, but sanctioning is itself a contribution nobody is individually motivated to make."),

  h("3.3 The benchmark, and where the intervener enters", 2),
  p("With β > 0, κ > 0, and the veracity reward deferred to the end of the episode, the myopic stage-game best response for a citizen is to conform to the locally visible majority and never pay κ. Cumulative pollution should therefore be maximal in the no-intervener control, while the welfare-maximising profile requires someone to absorb κ. The Devil's Advocate is the one agent for whom paying κ is stipulated, not chosen."),
  p("This turns a vague question (\"does the AI help?\") into a sharp one: does an exogenous sanctioner change whether citizens start paying κ themselves? Human experimental economics gives competing predictions. Institutional sanctioning can migrate a population toward cooperation (Gürerk et al., 2006), but external enforcement can also crowd out voluntary cooperation by reframing a normative act as a policed one (Fehr & Rockenbach, 2003; Bowles, 2008). Whether tone determines which of these two regimes obtains is, to our knowledge, untested in either literature."),
  p("The ratio β/γ has a direct platform-design reading: it is how much more a platform pays for engagement than for accuracy. Sweeping it is not a robustness afterthought but a policy-relevant second axis (§5.9)."),

  h("4. Research Questions and Hypotheses", 1),
  p("Every primary hypothesis is stated over an action the agent takes and pays for, not over a belief it reports."),
  table([12, 88], [
    ["RQ", "Question"],
    ["RQ1 (Timing)", "Does intervening before a local majority forms reduce cumulative propagation of the false claim relative to intervening after?"],
    ["RQ2 (Tone)", "Does empathetic nudging or aggressive debunking produce more durable suppression, holding the factual payload constant?"],
    ["RQ3 (Headline)", "Does the intervener catalyse or crowd out endogenous peer correction, and does tone determine which?"],
    ["RQ4 (Models)", "Do the timing and tone effects replicate across model families, or is \"the right way to intervene\" backbone-specific?"],
    ["RQ5 (Specificity)", "Does the intervener also suppress true claims — i.e. does correction generalise into indiscriminate skepticism?"],
    ["RQ6 (Composition)", "In a society of different models, does one model free-ride on another's correction, and do backbone effects on peer correction survive when claim, topology, intervention and seed are held fixed within the same episode?"],
  ]),
  p(""),
  table([9, 61, 30], [
    ["H", "Statement", "Confirmatory tier (v2.1, §5.9)"],
    ["H1", "Early intervention yields lower cumulative propagation than late intervention.", "Tier 1 — focused arm"],
    ["H2", "Aggressive debunking produces faster initial suppression; empathetic nudging produces more durable suppression. Framed as a test between competing predictions, given the contested backfire literature.", "Tier 2 — exploratory"],
    ["H3", "Empathetic nudging raises the endogenous peer-correction rate above control; aggressive debunking suppresses it below control, even where it lowers propagation.", "Tier 1 — focused arm (headline)"],
    ["H4", "Timing moderates tone: aggressive debunking is relatively more viable early, before positions are publicly committed and reputation damage is at stake.", "Tier 2 — exploratory"],
    ["H5", "Effect sizes differ across model families by more than they differ across timing conditions — i.e. backbone choice dominates intervention design.", "Tier 1 — core grid"],
    ["H6", "The between-model spread in peer-correction rate (H5) persists when measured within a single mixed-model episode — same claim, topology, intervention and seed — rather than between episodes; a model's own correction rate does not detectably depend on whether its neighbours are copies of itself.", "Tier 2 — exploratory (new arm, §5.11)"],
  ]),
  callout("v2.1 — why the hypotheses are now tiered", [
    "The reference implementation shows that at 30 seeds/cell the core grid powers exactly one primary contrast (intervention vs control on EPC, d ≈ 0.80, n ≈ 26). H1 needs ~105 seeds/cell and H3 ~259. Declaring every hypothesis confirmatory at a sample that cannot test four of them is the kind of thing reviewers catch and preregistration is supposed to prevent.",
    "H3 is small for a structural reason worth stating in the manuscript: tone reaches peer correction through two opposing channels. An aggressive corrector displaces responsibility (lowering EPC) and also updates beliefs harder (raising it). The net is a difference of two larger quantities — precisely the shape of effect a small design cannot resolve.",
  ]),

  h("5. Experimental Design", 1),
  h("5.1 Agent society (N = 7)", 2),
  p("Seven agents matches the scale at which the sanctioning dilemma has been studied in this literature and is large enough to produce genuine local-majority dynamics among five citizens. All citizens share identical scaffolding — same memory window, same action space, same payoff accounting — and differ only in persona prompts, so behavioural differences are attributable to persona and condition rather than architecture."),
  table([16, 22, 62], [
    ["Agent", "Role", "Behavioural policy"],
    ["Citizens A1, A2", "Left-leaning personas", "Payoff-maximising under a persona prior that raises the conformity reward for congruent claims."],
    ["Citizens B1, B2", "Right-leaning personas", "Mirror of A, with the opposite congruence prior."],
    ["Citizen C", "Neutral conformist", "No ideological prior; responds to whichever signal dominates its neighbourhood."],
    ["Disseminator", "Bad actor (fixed policy)", "Injects and re-frames false claims from the LIAR pool; not payoff-responsive, so it cannot be \"corrected\" and functions as a constant adversarial pressure."],
    [{ text: "Devil's Advocate", bg: H.NEW_BG }, { text: "AI intervener (treatment)", bg: H.NEW_BG },
     { text: "Stipulated to absorb κ whenever it challenges. Enters per the active timing condition and remains active thereafter. Absent in control. v2.1: it first judges whether the claim is false and may decline to act — see §5.3.", bg: H.NEW_BG }],
  ]),

  h("5.2 Network topology", 2),
  p("Each PHEME conversation thread ships a structure.json encoding the reply tree of a real rumour cascade. We sample connected 7-node motifs from these trees, map the Disseminator to the cascade root, and assign citizen roles to downstream nodes. Every condition is also run on one hand-specified canonical motif so that results are not an artifact of any single sampled topology."),
  callout("v2.1 — intervener placement and reach are now declared, not left to chance", [
    "v2.0 specifies where the Disseminator goes but not where the intervener goes. This is not a detail: the Devil's Advocate is a peer in a network, so it is visible only to its neighbours. The intervener is now placed on the highest-degree non-root node, the position a platform-embedded corrector would occupy, and its reach is logged per episode and entered as a covariate in the primary models (reach correlates with peer correction at ρ = 0.34, p < 1e-13).",
  ]),
  callout("v2.2 — the reply tree is not the visibility graph", [
    "This is the change that mattered most, and it only appeared once the real PHEME release was in hand. Sampling a 7-node motif around the cascade root produces a BROADCAST STAR 69% of the time, because real PHEME cascades are broad and shallow (mean branching 4.46, mean depth 3.59). Rooted at the Disseminator, a star gives every citizen exactly one neighbour — the Disseminator. In 89% of real motifs no local majority could form among the five citizens, and no citizen could ever observe another paying the correction cost, which is the demonstration channel the headline outcome measures.",
    "Neither existing gate detects this. The section 5.7 calibration gate passes MORE comfortably on real PHEME than on surrogate topology, because it tests cascade shape and diffusion asymmetry and neither is sensitive to whether the agents can see one another.",
    [t("The cause is a conflation. ", { bold: true }),
     t("structure.json is a reply tree — it records who answered whom. The design needs who can SEE whom, and a threaded conversation shows sibling replies too. Connecting every sibling is the opposite error: in a star every reply is a sibling of every other, so the motif becomes the complete graph, \"local\" majority becomes global, and the topology stops varying (measured: 68% of motifs become K7).")],
    [t("Bounded attention. ", { bold: true }),
     t("A thread view is ranked and truncated — a reader sees the source plus the replies near their own, not all fifty. Each reply is therefore linked to the w siblings following it in display order. At w = 1, on all 6,425 real threads: no motif is a star, none is complete, mean density 0.50, and no thread is discarded (the reach filter keeps 201 of 201 motifs, against 63 before). Effect sizes strengthen because the mechanism can operate: intervention → EPC rises from d = 0.69 to 0.93, and H1 from d = −0.38 to −0.61.")],
  ], { bg: "EAF1F8", color: H.ACCENT }),

  h("5.3 Factors and conditions", 2),
  p("Episodes run 16 rounds (v2.1; see §5.5 on durability). Early injects the intervener at round 2, immediately after first amplification; Late at round 6, after a local majority (≥ 3 of 5 citizens endorsing) is established. Timing thresholds are tuned and frozen in the Month-2 pilot."),
  table([16, 26, 26, 32], [
    ["Factor", "Level 1", "Level 2", "Notes"],
    ["F1 · Timing", "Early (round 2)", "Late (round 6)", "Intervener persists from entry onward. Both a fixed-round and a majority-triggered variant are implemented; the majority variant carries a latest-entry backstop so a cell where no majority forms does not silently become a second control."],
    ["F2 · Tone", "Empathetic nudge", "Aggressive debunking", "Factual payload identical and length-matched; only pragmatic framing differs."],
    ["F3 · Backbone", "4 model families", "—", "GPT-4o-mini class, Claude Haiku class, Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct (final list fixed at Week 4). Citizens only; judges held constant. Run both as monocultures (core grid, below) and as a mixed society within one episode (§5.11, new in v2.6)."],
  ]),
  p("This yields a 2 × 2 × 4 grid plus a per-model control — 20 cells — flanked by:"),
  bullet([t("Placebo arm (specificity). ", { bold: true }), t("Matched true claims from LIAR run through the same conditions; the intervener should not suppress them.")]),
  bullet([t("Ablation 1 — payoff visibility. ", { bold: true }), t("Payoffs removed from the prompt and the episode run as pure narrative. Tests whether the game structure actually drives behaviour or whether the LLM is pattern-matching social discourse regardless.")]),
  bullet([t("Ablation 2 — incentive ratio. ", { bold: true }), t("β/γ swept across {0.5, 1, 3} to model platforms that reward engagement more or less than accuracy.")]),
  callout("v2.1 — the placebo arm needs an intervener that can decline to act", [
    "As written, the Devil's Advocate is stipulated to challenge whatever claim it is pointed at. Run that against a true claim and it suppresses it by construction — so the specificity measure reports the wiring, not a property of the intervention, and RQ5 cannot come out any way but badly.",
    "The intervener now judges the claim's veracity once per episode and stays silent if it judges it sound. Its sensitivity and false-alarm rate are logged as first-class quantities, because the specificity of the intervention is bounded by the specificity of the intervener. With a real backbone the model's own reading of the claim supplies this judgement; in the reference implementation they are declared parameters.",
    "The consequence for analysis is in §8: pooled across all treated placebo episodes the contrast is null (p = 0.71), because episodes where the intervener correctly held its fire average out the ones where it did not. Conditional on it having fired, unwarranted challenges against true claims roughly double (+0.065, p = 0.002). The conditional contrast is now preregistered alongside the pooled one.",
  ]),
];
