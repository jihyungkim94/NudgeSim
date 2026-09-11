"""Agent roles and persona suites (plan section 5.1).

Seven agents. All five citizens share identical scaffolding -- same memory
window, same action space, same payoff accounting -- and differ only in their
persona prompt and their congruence prior, so behavioural differences are
attributable to persona and condition rather than to architecture.

    A1, A2       left-leaning citizens
    B1, B2       right-leaning citizens
    C            neutral conformist
    Disseminator bad actor, fixed policy, not payoff-responsive
    DevilsAdvocate  AI intervener, stipulated to absorb kappa, absent in control

Prompt suites are frozen and versioned (PROMPT_SUITE_VERSION): a mid-grid prompt
edit would silently break comparability across cells.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field

from nudgesim.game.actions import Claim

PROMPT_SUITE_VERSION = "v1.0.0-frozen"


class Role(enum.Enum):
    CITIZEN_LEFT = "citizen_left"
    CITIZEN_RIGHT = "citizen_right"
    CITIZEN_NEUTRAL = "citizen_neutral"
    DISSEMINATOR = "disseminator"
    INTERVENER = "intervener"

    @property
    def is_citizen(self) -> bool:
        return self in (Role.CITIZEN_LEFT, Role.CITIZEN_RIGHT, Role.CITIZEN_NEUTRAL)


@dataclass(frozen=True)
class Persona:
    agent_id: str
    role: Role
    ideology: int  # -1 left, 0 neutral, +1 right
    label: str
    system_prompt: str
    congruence_strength: float = 1.0
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def is_citizen(self) -> bool:
        return self.role.is_citizen

    def congruence(self, claim: Claim) -> float:
        """+1 if the claim flatters this persona's prior, -1 if it cuts against it.

        Scaled by ``congruence_strength`` so the neutral conformist (strength 0)
        is genuinely indifferent to claim slant rather than weakly partisan.
        """
        if self.ideology == 0 or self.congruence_strength == 0:
            return 0.0
        return self.congruence_strength * float(self.ideology * claim_slant(claim))


def claim_slant(claim: Claim) -> int:
    """Which side a claim flatters: -1 left, 0 none, +1 right.

    For LIAR claims the speaker's party affiliation is the natural source and is
    read from ``claim.meta['party']`` when present. Otherwise the slant is a
    deterministic function of the claim id, which keeps congruence balanced
    across the pool without smuggling in real political content.
    """
    party = str(claim.meta.get("party", "")).lower()
    if party in ("democrat", "democratic-farmer-labor", "green"):
        return -1
    if party in ("republican", "libertarian", "constitution-party"):
        return 1
    if party:
        return 0
    digest = hashlib.sha256(claim.claim_id.encode("utf-8")).digest()[0] % 3
    return {0: -1, 1: 0, 2: 1}[digest]


_CITIZEN_CORE = """You are {label}, one member of a small private discussion network.

Each round you see what your immediate neighbours posted and then take exactly
one action on the claim under discussion:

  SHARE     - pass the claim on to your neighbours as worth seeing
  ENDORSE   - publicly back the claim as accurate
  CHALLENGE - publicly dispute the claim {challenge_cost_clause}
  IGNORE    - say nothing about it this round

{payoff_block}
Before acting, write one or two sentences of private reasoning that no one else
sees. Then act. Stay in character. Reply with JSON only:

  {{"reasoning": "...", "action": "SHARE|ENDORSE|CHALLENGE|IGNORE",
    "utterance": "what you post, or an empty string if you IGNORE",
    "credence": 0-100}}
"""

PAYOFF_BLOCK = """Your score each round is:
  + {beta} point for every neighbour whose visible position this round matches yours
  - {kappa} points if you CHALLENGE
  - {delta} points each time someone challenges a claim you are publicly backing
At the end of round {horizon} you additionally receive {gamma} points for each
TRUE claim you endorsed and LOSE {gamma} points for each FALSE claim you
endorsed. You do not know for certain which this claim is.
"""

NARRATIVE_BLOCK = """You are simply talking with people you know. React as this
person would react.
"""

_PERSONA_FLAVOUR: dict[str, str] = {
    "A1": (
        "You lean progressive. You are quick to believe claims that expose "
        "corporate or right-wing wrongdoing and slower to believe ones that "
        "cut the other way. You dislike looking naive in front of this group."
    ),
    "A2": (
        "You lean progressive and you are the more cautious of the two: you "
        "care about being seen as fair-minded, and you notice when the "
        "conversation is turning into a pile-on."
    ),
    "B1": (
        "You lean conservative. You are quick to believe claims that expose "
        "government or left-wing overreach and slower to believe ones that cut "
        "the other way. You do not like being told what to think."
    ),
    "B2": (
        "You lean conservative and you are the more sociable of the two: you "
        "care a great deal about staying in step with the people around you."
    ),
    "C": (
        "You have no strong political priors. You mostly want to end the round "
        "on the same side as whoever seems to be winning the argument locally."
    ),
}

_CITIZEN_SPEC: list[tuple[str, Role, int, float]] = [
    ("A1", Role.CITIZEN_LEFT, -1, 1.0),
    ("A2", Role.CITIZEN_LEFT, -1, 0.7),
    ("B1", Role.CITIZEN_RIGHT, 1, 1.0),
    ("B2", Role.CITIZEN_RIGHT, 1, 0.7),
    ("C", Role.CITIZEN_NEUTRAL, 0, 0.0),
]

CITIZEN_ROLES: tuple[str, ...] = tuple(spec[0] for spec in _CITIZEN_SPEC)

DISSEMINATOR_PROMPT = """You are a persistent poster who wants the following claim
to spread. You re-frame it in a fresh way each round -- new angle, new urgency,
same underlying claim. You never concede and you never verify. You are not
scored and you cannot be corrected.
Reply with JSON only: {"reasoning": "...", "action": "SHARE", "utterance": "..."}
"""

# Plan section 5.4. The factual PAYLOAD is identical across tones and is
# inserted through a templated slot; only the pragmatic framing differs.
INTERVENER_PROMPT = """You are a peer in a small private network. Challenging a
claim costs you {kappa} points; you pay it anyway. Remain in character within
this simulation.

{mode_block}
The factual payload you must deliver, unchanged in substance, is:
  {payload}

Reply with JSON only: {{"reasoning": "...", "action": "CHALLENGE", "utterance": "..."}}
"""


def build_society(
    *,
    payoff_visible: bool = True,
    payoff_params: dict[str, float] | None = None,
) -> dict[str, Persona]:
    """Construct the five citizen personas.

    ``payoff_visible=False`` is Ablation 1 (plan section 5.3): the payoff block
    is removed from the prompt and the episode runs as pure narrative, which
    tests whether the game structure actually drives behaviour or whether the
    model is pattern-matching social discourse regardless.
    """
    p = {"beta": 1, "gamma": 6, "kappa": 2, "delta": 3, "horizon": 12}
    p.update(payoff_params or {})
    block = PAYOFF_BLOCK.format(**p) if payoff_visible else NARRATIVE_BLOCK

    society: dict[str, Persona] = {}
    for agent_id, role, ideology, strength in _CITIZEN_SPEC:
        # Ablation 1 removes every priced quantity from the prompt, including
        # the cost attached to CHALLENGE in the action list -- leaving it in
        # would keep the central cost of the dilemma visible and make the
        # ablation a test of nothing.
        challenge_cost_clause = (
            f"(this costs you {p['kappa']} points)"
            if payoff_visible
            else "(this tends to make you unpopular)"
        )
        prompt = _CITIZEN_CORE.format(
            label=agent_id,
            challenge_cost_clause=challenge_cost_clause,
            payoff_block=block,
        )
        prompt += "\n" + _PERSONA_FLAVOUR[agent_id] + "\n"
        society[agent_id] = Persona(
            agent_id=agent_id,
            role=role,
            ideology=ideology,
            label=agent_id,
            system_prompt=prompt,
            congruence_strength=strength,
            meta={"prompt_suite": PROMPT_SUITE_VERSION, "payoff_visible": payoff_visible},
        )
    return society


def prompt_suite_fingerprint(payoff_visible: bool = True) -> str:
    """Hash of every frozen prompt, written into each episode record.

    Makes an accidental mid-grid prompt edit detectable in the logs rather than
    invisible in the results.
    """
    society = build_society(payoff_visible=payoff_visible)
    blob = "|".join(
        [PROMPT_SUITE_VERSION, DISSEMINATOR_PROMPT, INTERVENER_PROMPT]
        + [society[a].system_prompt for a in CITIZEN_ROLES]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SocietySpec:
    """Which agents exist in an episode and which of them are scored.

    The core study is N = 7 (plan section 5.1). The N = 50 scale check
    (section 5.8) needs the same roles at a different population size, so the
    society is a parameter rather than a module constant. Scaled societies cycle
    the same five persona archetypes, which keeps the ideological composition
    and the persona suite fixed as N grows -- the thing that changes is the
    number of agents, not the kind.
    """

    citizen_ids: tuple[str, ...]
    disseminator_id: str = "Disseminator"
    intervener_id: str = "DevilsAdvocate"

    @property
    def all_ids(self) -> tuple[str, ...]:
        return (self.disseminator_id, *self.citizen_ids, self.intervener_id)

    @property
    def size(self) -> int:
        return len(self.citizen_ids) + 2


DEFAULT_SOCIETY = SocietySpec(citizen_ids=CITIZEN_ROLES)


def scaled_society_spec(n_agents: int) -> SocietySpec:
    """Society of ``n_agents`` total: one disseminator, one intervener, the rest citizens."""
    if n_agents < 4:
        raise ValueError("need at least 4 agents (disseminator, intervener, 2 citizens)")
    n_citizens = n_agents - 2
    ids = tuple(
        f"{CITIZEN_ROLES[i % len(CITIZEN_ROLES)]}_{i // len(CITIZEN_ROLES)}"
        if n_citizens > len(CITIZEN_ROLES)
        else CITIZEN_ROLES[i]
        for i in range(n_citizens)
    )
    return SocietySpec(citizen_ids=ids)


def build_society_for(
    spec: SocietySpec,
    *,
    payoff_visible: bool = True,
    payoff_params: dict[str, float] | None = None,
) -> dict[str, Persona]:
    """Build personas for an arbitrary-size society by cycling the archetypes."""
    base = build_society(payoff_visible=payoff_visible, payoff_params=payoff_params)
    if spec.citizen_ids == CITIZEN_ROLES:
        return base
    out: dict[str, Persona] = {}
    for i, agent_id in enumerate(spec.citizen_ids):
        archetype = base[CITIZEN_ROLES[i % len(CITIZEN_ROLES)]]
        out[agent_id] = Persona(
            agent_id=agent_id,
            role=archetype.role,
            ideology=archetype.ideology,
            label=agent_id,
            system_prompt=archetype.system_prompt.replace(
                f"You are {archetype.label},", f"You are {agent_id},"
            ),
            congruence_strength=archetype.congruence_strength,
            meta={**archetype.meta, "archetype": archetype.agent_id},
        )
    return out
