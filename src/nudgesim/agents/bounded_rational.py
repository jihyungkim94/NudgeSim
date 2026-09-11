"""Analytic citizen policy: a bounded-rational surrogate for an LLM backbone.

WHAT THIS IS FOR
----------------
The experiment in the project plan runs citizens on four LLM backbones. This
module is not a substitute for that. It exists for three jobs that do not need a
network call:

  1. Unit and integration testing of the engine, ledger and metrics.
  2. The calibration gate's shape checks, which are about the *environment*
     (topology, visibility, disseminator persistence) rather than about models.
  3. A **design-sensitivity simulation**: running the full preregistered grid
     against a declared data-generating process, to establish that the analysis
     plan in plan section 8 can actually detect effects of the assumed size at
     30 seeds per cell -- and that it returns a null when the DGP contains none.

Results produced with this policy are properties of the DGP below, NOT evidence
about LLM behaviour. Every record it writes is tagged ``backbone_kind:
"surrogate"`` and the analysis refuses to label such runs as model results.

THE MODEL
---------
Utility of each action, evaluated against the visible neighbourhood:

  conformity      + beta * (neighbours whose standing stance matches)
  congruence      + lambda_cong * g          on PRO, - lambda_cong * g on CON
  veracity        + phi * gamma * (1 - 2*p_false) on PRO, and only if the agent
                    is not already committed (an endorsement already on the
                    record has already incurred the deferred penalty, which is
                    what produces commitment lock-in)
  challenge cost  - kappa                    on CHALLENGE
  reputation      - rho * delta * E[incoming challenges]  while publicly backing
  correction norm + nu * salience            on CHALLENGE
  reactance       + xi * face_threat * commitment * max(0, g)   on PRO

with the norm salience carrying the two competing predictions from human
experimental economics that the study is built to separate:

  salience = 1 + omega_demo * peer_corrections_seen
               - omega_crowd * policing_salience

  policing_salience accumulates with intervener activity, weighted by the
  message's face threat. omega_crowd > 0 encodes crowding-out (Fehr &
  Rockenbach 2003; Bowles 2008); omega_demo > 0 encodes norm demonstration
  (Guererk et al. 2006). BOTH are free parameters, and the shipped grid is run
  under two declared DGPs -- one with a tone-dependent crowding-out term and one
  without -- precisely so the analysis can be shown to distinguish them rather
  than to confirm a foregone conclusion.

Beliefs update toward the intervener's payload in proportion to its epistemic
force, damped by reactance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from nudgesim.agents.policy import Decision, Observation
from nudgesim.data.claims import claim_engagement
from nudgesim.game.actions import Action, Stance


@dataclass(frozen=True)
class BackboneProfile:
    """Behavioural parameters standing in for one model family.

    The four shipped profiles are deliberately *not* named after real model
    families: an offline run must never read as a benchmark of GPT or Claude.
    When real backbones are configured, this class is unused.
    """

    name: str
    temperature: float = 0.9          # softmax sharpness over action utilities
    foresight: float = 0.45           # phi: weight on the deferred veracity term
    cost_sensitivity: float = 1.0     # multiplier on kappa
    risk_weight: float = 0.35         # rho: weight on expected reputation damage
    correction_propensity: float = 0.55   # nu: intrinsic pull toward correcting
    reactance: float = 0.6            # xi
    deference: float = 0.35           # alpha: belief update rate on a challenge
    conformity_weight: float = 1.0    # multiplier on beta
    congruence_weight: float = 1.2    # lambda_cong
    prior_false: float = 0.5          # p_false before any evidence
    verbosity: int = 1

    def as_dict(self) -> dict[str, float | str]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# Four surrogate profiles spanning a plausible range: a cautious/accuracy-leaning
# family, a strongly conformist one, a defiant one, and a cost-averse one. The
# spread is what makes the H4/H5 model-dependence tests meaningful in the
# design-sensitivity run.
SURROGATE_PROFILES: dict[str, BackboneProfile] = {
    "surrogate-cautious": BackboneProfile(
        name="surrogate-cautious",
        temperature=0.7,
        foresight=0.70,
        correction_propensity=0.75,
        reactance=0.35,
        deference=0.50,
        conformity_weight=0.8,
        congruence_weight=0.9,
    ),
    "surrogate-conformist": BackboneProfile(
        name="surrogate-conformist",
        temperature=1.0,
        foresight=0.25,
        correction_propensity=0.35,
        reactance=0.45,
        deference=0.40,
        conformity_weight=1.5,
        congruence_weight=1.3,
    ),
    "surrogate-defiant": BackboneProfile(
        name="surrogate-defiant",
        temperature=0.9,
        foresight=0.40,
        correction_propensity=0.60,
        reactance=1.10,
        deference=0.22,
        conformity_weight=1.0,
        congruence_weight=1.6,
    ),
    "surrogate-cost-averse": BackboneProfile(
        name="surrogate-cost-averse",
        temperature=0.8,
        foresight=0.50,
        cost_sensitivity=1.8,
        correction_propensity=0.30,
        reactance=0.50,
        deference=0.45,
        conformity_weight=1.1,
        congruence_weight=1.0,
    ),
}


@dataclass(frozen=True)
class NormParams:
    """Declared ground truth for the social-norm channel (see module docstring).

    ``omega_crowd_tone`` is the parameter that encodes H3. The design-sensitivity
    run is executed at two settings -- the declared effect and zero -- so the
    analysis is shown to detect one and not invent the other.
    """

    omega_demo_peer: float = 0.70      # a peer correcting demonstrates the norm
    omega_demo_external: float = 0.15  # an outsider doing it demonstrates less
    omega_crowd_base: float = 0.25     # any enforcer displaces some responsibility
    omega_crowd_tone: float = 0.85     # extra displacement per unit of face threat
    salience_decay: float = 0.80       # per-round memory of both channels
    min_salience: float = 0.0
    # Novelty channel (Vosoughi, Roy & Aral 2018): how much a startling claim's
    # arousal value adds to the pull of propagating it. Setting this to 0 makes
    # false and true claims diffuse identically, which fails the section 5.7
    # calibration gate -- see docs/CALIBRATION.md.
    novelty_engagement: float = 0.45

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class _AgentState:
    p_false: float
    peer_signal: float = 0.0      # corrections by peers, per visible neighbour
    external_signal: float = 0.0  # corrections by the intervener
    policing: float = 0.0         # face-threat-weighted enforcement salience
    committed: bool = False
    seen_rounds: set[int] = field(default_factory=set)


class BoundedRationalPolicy:
    """One instance per agent per episode; carries latent belief and norm state."""

    backbone_kind = "surrogate"

    def __init__(
        self,
        profile: BackboneProfile,
        *,
        seed: int,
        norms: NormParams | None = None,
        payoff_visible: bool = True,
    ) -> None:
        self.profile = profile
        self.backbone = profile.name
        self.norms = norms or NormParams()
        self.payoff_visible = payoff_visible
        self._rng = random.Random(seed)
        self._state: _AgentState | None = None
        self.last_utilities: dict[str, float] = {}

    # ------------------------------------------------------------------ state

    def _ensure_state(self, obs: Observation) -> _AgentState:
        if self._state is None:
            self._state = _AgentState(p_false=self.profile.prior_false)
        return self._state

    def _absorb_feed(self, obs: Observation, state: _AgentState) -> None:
        """Update belief and norm salience from posts not yet processed.

        Both norm channels are accumulated as *rates* -- corrections seen per
        visible neighbour, intervener activity per round -- and then passed
        through a saturating transform in ``salience``. Accumulating raw counts
        instead produces runaway positive feedback: every peer correction makes
        the next one more likely without bound, and the society collapses into
        universal challenging within a few rounds.
        """
        state.peer_signal *= self.norms.salience_decay
        state.policing *= self.norms.salience_decay

        fresh = [i for i in obs.feed if i.round_index not in state.seen_rounds]
        n_visible = max(1, len(obs.neighbour_stances))

        peer_challenges = 0
        for item in fresh:
            if item.action is not Action.CHALLENGE:
                continue
            if item.is_intervener:
                state.policing += (
                    self.norms.omega_crowd_base
                    + self.norms.omega_crowd_tone * item.face_threat
                )
                # Belief update, damped by the reactance the framing provokes.
                damping = 1.0 / (
                    1.0
                    + self.profile.reactance
                    * item.face_threat
                    * (1.0 if state.committed else 0.3)
                )
                gain = self.profile.deference * item.epistemic_force * damping
                state.p_false += gain * (1.0 - state.p_false)
                state.external_signal += 1.0
            else:
                peer_challenges += 1
                gain = 0.5 * self.profile.deference
                state.p_false += gain * (1.0 - state.p_false)

        state.peer_signal += peer_challenges / n_visible
        state.seen_rounds.update(item.round_index for item in obs.feed)
        state.p_false = min(0.99, max(0.01, state.p_false))

    @staticmethod
    def _saturate(x: float) -> float:
        """Bounded in [0, 1): diminishing returns on repeated social evidence."""
        return x / (1.0 + x) if x > 0 else 0.0

    def salience(self, state: _AgentState) -> float:
        """How much correcting feels like this agent's job right now.

        Rises with demonstrated peer correction (Guererk et al. 2006) and falls
        when an external enforcer is visibly policing the thread (Fehr &
        Rockenbach 2003; Bowles 2008). Which of the two dominates -- and whether
        tone decides it -- is the empirical question, not an assumption baked in
        here: both weights are free parameters declared in the run config.
        """
        raw = (
            1.0
            + self.norms.omega_demo_peer * self._saturate(state.peer_signal)
            + self.norms.omega_demo_external * self._saturate(state.external_signal)
            - self._saturate(state.policing)
        )
        return max(self.norms.min_salience, raw)

    # --------------------------------------------------------------- decision

    def utilities(self, obs: Observation) -> dict[Action, float]:
        state = self._ensure_state(obs)
        self._absorb_feed(obs, state)

        p = obs.payoff
        prof = self.profile
        counts = obs.neighbour_stance_counts()
        g = obs.persona.congruence(obs.claim)
        committed = obs.own_stance is Stance.PRO or state.committed

        # Expected incoming challenges next round: visible CON neighbours, plus
        # the intervener only if this agent has actually SEEN it challenge.
        # Keying this off the scheduler's "intervener is active" flag instead
        # would let a citizen anticipate criticism from an agent that has never
        # said anything -- which shows up in the placebo arm as a spurious
        # "mere presence" effect that is really just the flag.
        expected_challenges = counts[Stance.CON] + (
            1.0 if state.external_signal > 0 else 0.0
        )

        beta = p.beta * prof.conformity_weight
        lam = prof.congruence_weight
        face = max((i.face_threat for i in obs.intervener_messages()), default=0.0)

        if not self.payoff_visible:
            # Ablation 1: the payoff block is gone from the prompt. The agent
            # becomes a narrative social imitator -- conformity and congruence
            # still operate, the priced terms do not.
            beta = 1.0 * prof.conformity_weight
            phi = kappa = rho = 0.0
        else:
            phi = prof.foresight
            kappa = p.kappa * prof.cost_sensitivity
            rho = prof.risk_weight

        util: dict[Action, float] = {}
        for action in Action:
            u = beta * counts[action.stance]
            if action.stance is Stance.PRO:
                u += lam * g
                u += self.norms.novelty_engagement * claim_engagement(obs.claim)
                if not committed:
                    u += phi * p.gamma * (1.0 - 2.0 * state.p_false)
                u -= rho * p.delta * expected_challenges
                u += prof.reactance * face * (1.0 if committed else 0.3) * max(0.0, g)
            elif action.stance is Stance.CON:
                u -= lam * g
                u -= kappa
                u += prof.correction_propensity * self.salience(state) * (
                    0.5 + state.p_false
                )
            else:  # IGNORE keeps a standing endorsement exposed
                if committed:
                    u -= rho * p.delta * expected_challenges
            util[action] = u

        # SHARE is a slightly louder ENDORSE: it reaches further but commits the
        # agent no less, so it carries a small extra exposure term.
        util[Action.SHARE] -= 0.15 * rho * p.delta * expected_challenges
        return util

    async def decide(self, observation: Observation) -> Decision:
        util = self.utilities(observation)
        state = self._ensure_state(observation)
        self.last_utilities = {a.value: round(v, 4) for a, v in util.items()}

        tau = max(0.05, self.profile.temperature)
        top = max(util.values())
        weights = [math.exp((util[a] - top) / tau) for a in Action]
        action = self._rng.choices(list(Action), weights=weights, k=1)[0]

        if action.is_propagating:
            state.committed = True
        elif action is Action.CHALLENGE:
            state.committed = False

        credence = round((1.0 - state.p_false) * 100.0, 1)
        trace, motive = self._trace(action, observation, state)
        return Decision(
            action=action,
            reasoning=trace,
            utterance=self._utterance(action, observation),
            credence=credence,
            meta={
                "utilities": self.last_utilities,
                "p_false": round(state.p_false, 4),
                "salience": round(self.salience(state), 4),
                "policing": round(state.policing, 4),
                "peer_signal": round(state.peer_signal, 4),
                "backbone_kind": self.backbone_kind,
                "ground_truth_motive": motive,
            },
        )

    # --------------------------------------------------------------- surfaces

    def _trace(
        self, action: Action, obs: Observation, state: _AgentState
    ) -> tuple[str, str]:
        """Reasoning trace plus the ground-truth motive that produced it.

        The trace is generated from the same quantities that drove the choice,
        so the surrogate run carries a known motive label for every trace. That
        gives the trace-taxonomy classifier (plan section 5.6) a ground truth to
        be validated against offline -- the same role the 300 human-coded traces
        play for the reported results.
        """
        counts = obs.neighbour_stance_counts()
        g = obs.persona.congruence(obs.claim)
        if action is Action.CHALLENGE:
            if state.policing > state.peer_signal:
                return (
                    "Someone is already policing this thread, but it still seems wrong "
                    "to let it stand -- the claim does not hold up.",
                    "accuracy_motivated",
                )
            return (
                f"I put this at {state.p_false:.0%} likely false and the evidence is not "
                "there; someone has to verify it publicly.",
                "accuracy_motivated",
            )
        if action is Action.IGNORE:
            return (
                "Not worth the exposure either way this round; I will sit this one out.",
                "cost_avoidant",
            )
        if g > 0.5:
            return (
                "This fits what I already think is going on -- our side has been saying "
                "it for a while.",
                "identity_protective",
            )
        if counts[Stance.PRO] >= counts[Stance.CON]:
            return (
                f"{counts[Stance.PRO]} of my neighbours are backing it; going against "
                "the majority here is not worth it.",
                "conformity_motivated",
            )
        return (
            "I am not certain, but staying visible on this is worth more to me than "
            "being cautious.",
            "strategic_engagement",
        )

    # Paraphrase banks. The engine's repetition guard exists to catch a model
    # that has collapsed into a loop; a policy that emits one canned sentence
    # per action would trip it every episode, so the surrogate varies its
    # wording the way a live backbone would. Semantics are identical within a
    # bank -- only the surface form rotates.
    _UTTERANCES: dict[str, tuple[str, ...]] = {
        "CHALLENGE": (
            "I do not think this holds up -- has anyone actually checked the source?",
            "This does not match what I read. Where is it actually coming from?",
            "I am going to push back on this one. The sourcing is not there.",
            "Before this goes further: has anybody verified any of it?",
        ),
        "SHARE": (
            "Passing this on: {claim}",
            "Worth a look, in case you have not seen it: {claim}",
            "Adding this here: {claim}",
            "Others should see this: {claim}",
        ),
        "ENDORSE": (
            "This matches what I have been hearing. I think it is right.",
            "Lines up with everything else on this. I would say it is accurate.",
            "Consistent with what I already knew -- I will back it.",
            "Nothing here surprises me. I think it stands.",
        ),
    }

    def _utterance(self, action: Action, obs: Observation) -> str:
        if action is Action.IGNORE:
            return ""
        bank = self._UTTERANCES[action.value]
        template = bank[obs.round_index % len(bank)]
        return template.format(claim=obs.claim.text)


def profile_for(name: str) -> BackboneProfile:
    if name in SURROGATE_PROFILES:
        return SURROGATE_PROFILES[name]
    raise KeyError(f"unknown surrogate profile {name!r}; have {sorted(SURROGATE_PROFILES)}")


def with_overrides(profile: BackboneProfile, **kw: float) -> BackboneProfile:
    return replace(profile, **kw)
