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
    "Defaults: β = 1, γ = 6, κ = 2, δ = 3, T = 12   (all swept in robustness; see §5.9)",
  ]),
  p([t("Accounting conventions. ", { bold: true }),
     t("Three details determine the numbers and are declared rather than left to the implementer: (i) C counts matching "),
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
    ["H", "Statement", "Confirmatory tier (§5.9)"],
    ["H1", "Early intervention yields lower cumulative propagation than late intervention.", "Tier 1 — powered contrast"],
    ["H2", "Aggressive debunking produces faster initial suppression; empathetic nudging produces more durable suppression. Framed as a test between competing predictions, given the contested backfire literature.", "Tier 2 — exploratory"],
    ["H3", "Empathetic nudging raises the endogenous peer-correction rate above control; aggressive debunking suppresses it below control, even where it lowers propagation.", "Tier 2 — exploratory (headline direction; see power note below)"],
    ["H4", "Timing moderates tone: aggressive debunking is relatively more viable early, before positions are publicly committed and reputation damage is at stake.", "Tier 2 — exploratory"],
    ["H5", "Effect sizes differ across model families by more than they differ across timing conditions — i.e. backbone choice dominates intervention design.", "Tier 1 — core grid"],
    ["H6", "The between-model spread in peer-correction rate (H5) persists when measured within a single mixed-model episode — same claim, topology, intervention and seed — rather than between episodes; a model's own correction rate does not detectably depend on whether its neighbours are copies of itself.", "Tier 2 — exploratory (mixed-society arm, §5.11)"],
  ]),
  p([t("Why the hypotheses are tiered. ", { bold: true }),
     t("At the core grid's 30 seeds/cell, the design is powered for exactly one primary contrast (intervention vs control on EPC, d ≈ 0.80, n ≈ 26) plus the model-dependence contrast H5. H1 needs roughly 105 seeds/cell and H3 roughly 259 at the effect sizes the pilot produces; declaring every hypothesis confirmatory at a sample that cannot test them would not survive review, so tiering is set before the grid runs rather than after (§5.9, §8).")]),
  p([t("H3 is expected to be small for a structural reason worth stating up front. ", { italics: true, color: H.MUTED }),
     t("Tone reaches peer correction through two opposing channels: an aggressive corrector displaces responsibility (lowering EPC) and also updates beliefs harder (raising it). The net is a difference of two larger quantities — precisely the shape of effect the core grid is not sized to resolve on its own, which is why it is exploratory rather than dropped.", { italics: true, color: H.MUTED })]),

  h("5. Experimental Design", 1),
  h("5.1 Agent society (N = 7)", 2),
  p("Seven agents matches the scale at which the sanctioning dilemma has been studied in this literature and is large enough to produce genuine local-majority dynamics among five citizens. All citizens share identical scaffolding — same memory window, same action space, same payoff accounting — and differ only in persona prompts, so behavioural differences are attributable to persona and condition rather than architecture."),
  table([16, 22, 62], [
    ["Agent", "Role", "Behavioural policy"],
    ["Citizens A1, A2", "Left-leaning personas", "Payoff-maximising under a persona prior that raises the conformity reward for congruent claims."],
    ["Citizens B1, B2", "Right-leaning personas", "Mirror of A, with the opposite congruence prior."],
    ["Citizen C", "Neutral conformist", "No ideological prior; responds to whichever signal dominates its neighbourhood."],
    ["Disseminator", "Bad actor (fixed policy)", "Injects and re-frames false claims from the LIAR pool; not payoff-responsive, so it cannot be \"corrected\" and functions as a constant adversarial pressure."],
    ["Devil's Advocate", "AI intervener (treatment)", "Stipulated to absorb κ whenever it challenges. Enters per the active timing condition and remains active thereafter, absent in control. It first judges whether the claim is actually false and stays silent if it judges it sound (§5.3) — a corrector that challenges everything it is pointed at cannot test specificity."],
  ]),

  h("5.2 Network topology and visibility", 2),
  p("Each PHEME conversation thread ships a structure.json encoding the reply tree of a real rumour cascade. We sample connected 7-node motifs from these trees, map the Disseminator to the cascade root, and assign citizen roles to downstream nodes. Every condition is also run on one hand-specified canonical motif so that results are not an artifact of any single sampled topology. The Devil's Advocate is placed on the highest-degree non-root node — the position a platform-embedded corrector would occupy — and its reach (0 to 5 citizens, depending on the sampled motif) is logged per episode and entered as a covariate in the primary models."),
  p([t("Visibility is modelled as bounded attention, not as the raw reply tree. ", { bold: true }),
     t("A reply tree records who answered whom; it is not the same graph as who can "),
     t("see", { italics: true }),
     t(" whom, and the two must not be conflated. Rooted at the Disseminator, a real PHEME cascade is broad and shallow (mean branching 4.46, mean depth 3.59), so a naive reading of the reply tree as the visibility graph gives most citizens exactly one neighbour — the Disseminator — leaving no room for a local majority to form or for one citizen to observe another paying the correction cost, which is the channel the headline outcome measures. Connecting every sibling reply instead over-corrects into a near-complete graph, collapsing \"local\" into \"global\". Bounded attention resolves this the way an actual thread view works: a reader sees the source plus the replies near their own in display order, so each reply is linked to the w siblings following it. At w = 1 the model is well-behaved on all 6,425 real PHEME-9 threads: no motif degenerates into a star or a clique, mean density is 0.50, and no thread has to be discarded.")]),

  h("5.3 Factors and conditions", 2),
  p("Episodes run 12 rounds. Early injects the intervener at round 2, immediately after first amplification; Late at round 6, after a local majority (≥ 3 of 5 citizens endorsing) is established. Both a fixed-round and a majority-triggered variant of Late are implemented, the latter with a latest-entry backstop so that a cell where no majority forms does not silently become a second control. Timing thresholds are tuned and frozen in the Month-2 pilot."),
  table([16, 26, 26, 32], [
    ["Factor", "Level 1", "Level 2", "Notes"],
    ["F1 · Timing", "Early (round 2)", "Late (round 6)", "Intervener persists from entry onward."],
    ["F2 · Tone", "Empathetic nudge", "Aggressive debunking", "Factual payload identical and length-matched; only pragmatic framing differs."],
    ["F3 · Backbone", "6 model families, 4 vendors", "—", "GPT-4o, a current OpenAI reasoning model, Claude Opus 5 (thinking on and off), Llama-3.3-70B, Qwen3-30B-A3B. Citizens only; judge held constant and distinct from every citizen backbone (§7). Run both as monocultures (core grid, below) and seated together in a mixed society within one episode (§5.11)."],
  ]),
  p("This yields a 2 × 2 × 6 grid plus a per-model control — 30 cells at the six-level roster, against the 20 the four-policy reference run uses (§5.9) — flanked by:"),
  bullet([t("Placebo arm (specificity). ", { bold: true }), t("Matched true claims from LIAR run through the same conditions; the intervener should not suppress them. Because the intervener can decline to act (§5.1), this arm has a real chance of failing — it does not pass by construction.")]),
  bullet([t("Ablation 1 — payoff visibility. ", { bold: true }), t("Payoffs removed from the prompt and the episode run as pure narrative. Tests whether the game structure actually drives behaviour or whether the LLM is pattern-matching social discourse regardless.")]),
  bullet([t("Ablation 2 — incentive ratio. ", { bold: true }), t("β/γ swept across {0.5, 1, 3} to model platforms that reward engagement more or less than accuracy.")]),
  bullet([t("Mixed-society arm (composition, RQ6/H6). ", { bold: true }), t("The full model roster seated together in one episode instead of one model per episode; see §5.11.")]),
  evidence("Feasibility evidence: the placebo arm and the specificity contrast", [
    "On the reference implementation's declared parameters, the pooled contrast across all treated placebo episodes is null (p = 0.71) — episodes where the intervener correctly held its fire average out the ones where it did not. Conditional on the intervener having actually fired, unwarranted challenges against true claims roughly double (+0.065, p = 0.002). Both the pooled and the conditional contrast are preregistered (§8), because pooling alone would have hidden the second number inside the first.",
  ]),

];
