"""The episode loop (plan section 7).

One episode = 1 claim x 1 topology x 1 condition x 1 seed. Twelve rounds of
simultaneous moves over a fixed visibility network, with a payoff ledger settled
each round and veracity resolved at the horizon.

Simultaneity matters: every agent decides from the same snapshot of the previous
round, so no agent gets a within-round information advantage from the order the
scheduler happened to run them in.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nudgesim.agents.bounded_rational import (
    BackboneProfile,
    BoundedRationalPolicy,
    NormParams,
    SURROGATE_PROFILES,
)
from nudgesim.agents.fixed import DisseminatorPolicy, IntervenerPolicy
from nudgesim.agents.llm_policy import LLMCitizenPolicy
from nudgesim.agents.persona import (
    CITIZEN_ROLES,
    DEFAULT_SOCIETY,
    PROMPT_SUITE_VERSION,
    Persona,
    Role,
    SocietySpec,
    build_society_for,
    prompt_suite_fingerprint,
)
from nudgesim.agents.policy import Decision, FeedItem, Observation, Policy
import networkx as nx

from nudgesim.data.topology import Motif, cascade_stats
from nudgesim.env.guards import Guards, stance_drift
from nudgesim.game.actions import Action, ActionRecord, Claim, Stance
from nudgesim.game.payoff import PayoffLedger, PayoffParams
from nudgesim.intervention.scheduler import InterventionSchedule, Timing
from nudgesim.intervention.tone import Tone, ToneTemplater

DISSEMINATOR_ID = DEFAULT_SOCIETY.disseminator_id
INTERVENER_ID = DEFAULT_SOCIETY.intervener_id
SOCIETY_ROLES: tuple[str, ...] = DEFAULT_SOCIETY.all_ids

# Preregistered out-of-band credence probe rounds (plan section 5.5), 0-indexed.
PROBE_ROUNDS: tuple[int, ...] = (0, 5, 11)


@dataclass
class EpisodeConfig:
    episode_id: str
    claim: Claim
    motif: Motif
    timing: Timing
    tone: Tone | None
    seed: int
    payoff: PayoffParams = field(default_factory=PayoffParams)
    backbone: str = "surrogate-cautious"
    payoff_visible: bool = True
    arm: str = "core"                    # core | placebo | ablation | scale
    memory_window: int = 3
    intervener_placement: str = "hub"
    trigger_mode: str = "fixed"
    society: SocietySpec = DEFAULT_SOCIETY
    norms: NormParams = field(default_factory=NormParams)
    policy_factory: Callable[[str, Persona, int], Policy] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def condition_id(self) -> str:
        tone = self.tone.value if self.tone else "none"
        return f"{self.timing.value}|{tone}|{self.backbone}"


@dataclass
class EpisodeResult:
    config: EpisodeConfig
    records: list[ActionRecord]
    ledger: PayoffLedger
    schedule: InterventionSchedule
    guards: Guards
    role_assignment: dict[str, str]
    probes: list[dict[str, Any]]
    wall_time_s: float
    repair_rate: float = 0.0

    def to_json(self) -> dict[str, Any]:
        cfg = self.config
        by_agent: dict[str, list[str]] = {}
        for record in self.records:
            by_agent.setdefault(record.agent_id, []).append(record.action.value)
        return {
            "episode_id": cfg.episode_id,
            "arm": cfg.arm,
            "timing": cfg.timing.value,
            "tone": cfg.tone.value if cfg.tone else "none",
            "backbone": cfg.backbone,
            "payoff_visible": cfg.payoff_visible,
            "seed": cfg.seed,
            "claim_id": cfg.claim.claim_id,
            "claim_veracity": cfg.claim.veracity.value,
            "claim_severity": cfg.claim.severity,
            "claim_topic": cfg.claim.topic,
            "motif_id": cfg.motif.motif_id,
            "motif_provenance": cfg.motif.provenance,
            "n_agents": cfg.society.size,
            "intervener_reach": cfg.motif.intervener_reach(
                cfg.society.all_ids,
                intervener_role=cfg.society.intervener_id,
                citizen_roles=cfg.society.citizen_ids,
                placement=cfg.intervener_placement,
            ),
            "motif_stats": cascade_stats(
                nx.bfs_tree(cfg.motif.graph, cfg.motif.root), cfg.motif.root
            ).as_dict(),
            "prompt_suite": PROMPT_SUITE_VERSION,
            "prompt_fingerprint": prompt_suite_fingerprint(cfg.payoff_visible),
            "role_assignment": self.role_assignment,
            "schedule": self.schedule.to_json(),
            "intervener_challenged": any(
                r.agent_id == cfg.society.intervener_id and r.action is Action.CHALLENGE
                for r in self.records
            ),
            "guards": self.guards.to_json(),
            "stance_drift": {
                agent: round(stance_drift(actions), 6) for agent, actions in by_agent.items()
            },
            "ledger": self.ledger.to_json(),
            "probes": self.probes,
            "repair_rate": round(self.repair_rate, 6),
            "wall_time_s": round(self.wall_time_s, 4),
            "payoff_params": {
                "beta": cfg.payoff.beta,
                "gamma": cfg.payoff.gamma,
                "kappa": cfg.payoff.kappa,
                "delta": cfg.payoff.delta,
                "horizon": cfg.payoff.horizon,
            },
            "meta": cfg.meta,
        }


def _surrogate_factory(
    backbone: str, norms: NormParams, payoff_visible: bool
) -> Callable[[str, Persona, int], Policy]:
    profile: BackboneProfile = SURROGATE_PROFILES[backbone]

    def make(agent_id: str, persona: Persona, seed: int) -> Policy:
        return BoundedRationalPolicy(
            profile, seed=seed, norms=norms, payoff_visible=payoff_visible
        )

    return make


def build_episode(config: EpisodeConfig) -> tuple[dict[str, Persona], dict[str, Policy], dict[str, str]]:
    """Instantiate the society, assign roles to motif nodes, and build policies."""
    society = build_society_for(
        config.society,
        payoff_visible=config.payoff_visible,
        payoff_params={
            "beta": config.payoff.beta,
            "gamma": config.payoff.gamma,
            "kappa": config.payoff.kappa,
            "delta": config.payoff.delta,
            "horizon": config.payoff.horizon,
        },
    )
    node_to_role = config.motif.assign_roles(
        config.society.all_ids,
        root_role=config.society.disseminator_id,
        hub_role=config.society.intervener_id,
        placement=config.intervener_placement,
    )
    role_to_node = {role: node for node, role in node_to_role.items()}

    personas: dict[str, Persona] = dict(society)
    personas[config.society.disseminator_id] = Persona(
        agent_id=config.society.disseminator_id,
        role=Role.DISSEMINATOR,
        ideology=0,
        label=config.society.disseminator_id,
        system_prompt="(fixed adversarial policy)",
    )
    personas[config.society.intervener_id] = Persona(
        agent_id=config.society.intervener_id,
        role=Role.INTERVENER,
        ideology=0,
        label=config.society.intervener_id,
        system_prompt="(fixed intervener policy)",
    )

    factory = config.policy_factory or _surrogate_factory(
        config.backbone, config.norms, config.payoff_visible
    )
    policies: dict[str, Policy] = {}
    for offset, agent_id in enumerate(config.society.citizen_ids):
        policies[agent_id] = factory(agent_id, personas[agent_id], config.seed * 1009 + offset)
    policies[config.society.disseminator_id] = DisseminatorPolicy(seed=config.seed)
    if config.timing is not Timing.NONE:
        tone = config.tone or Tone.EMPATHETIC
        policies[config.society.intervener_id] = IntervenerPolicy(
            ToneTemplater(tone), seed=config.seed
        )

    return personas, policies, role_to_node


async def run_episode(config: EpisodeConfig) -> EpisodeResult:
    start = time.perf_counter()
    personas, policies, role_to_node = build_episode(config)
    graph = config.motif.graph
    node_of = role_to_node
    role_of = {node: role for role, node in role_to_node.items()}

    def neighbours_of(agent_id: str) -> list[str]:
        return sorted(role_of[n] for n in graph.neighbors(node_of[agent_id]))

    society = config.society
    neighbour_map = {agent: neighbours_of(agent) for agent in society.all_ids}

    ledger = PayoffLedger(
        config.payoff,
        scored_agents=list(society.citizen_ids),
        observers=[society.disseminator_id, society.intervener_id],
    )
    schedule = InterventionSchedule(
        config.timing, mode=config.trigger_mode  # type: ignore[arg-type]
    )
    guards = Guards()
    records: list[ActionRecord] = []
    probes: list[dict[str, Any]] = []
    feed_by_agent: dict[str, list[FeedItem]] = {agent: [] for agent in society.all_ids}
    rng = random.Random(config.seed)

    for round_index in range(config.payoff.horizon):
        citizen_stances = {
            agent: ledger.standing_stance(agent, config.claim.claim_id)
            for agent in society.citizen_ids
        }
        intervener_active = schedule.update(round_index, citizen_stances)

        actors = [society.disseminator_id, *society.citizen_ids]
        if intervener_active and society.intervener_id in policies:
            actors.append(society.intervener_id)

        observations: dict[str, Observation] = {}
        for agent in actors:
            window = [
                item
                for item in feed_by_agent[agent]
                if item.round_index >= round_index - config.memory_window
            ]
            observations[agent] = Observation(
                episode_id=config.episode_id,
                round_index=round_index,
                horizon=config.payoff.horizon,
                persona=personas[agent],
                claim=config.claim,
                feed=window,
                neighbour_stances={
                    n: ledger.standing_stance(n, config.claim.claim_id)
                    for n in neighbour_map[agent]
                },
                own_stance=ledger.standing_stance(agent, config.claim.claim_id),
                own_score=(
                    ledger.agents[agent].immediate_total if agent in ledger.agents else 0.0
                ),
                payoff=config.payoff,
                payoff_visible=config.payoff_visible,
                intervener_present=intervener_active,
                intervener_rounds_active=(
                    0
                    if schedule.realised_entry_round is None
                    else round_index - schedule.realised_entry_round
                ),
            )

        decisions: list[Decision] = await asyncio.gather(
            *(policies[agent].decide(observations[agent]) for agent in actors)
        )
        moves = dict(zip(actors, decisions))

        # The two fixed-policy agents are not free to deviate.
        moves[society.disseminator_id].action = Action.SHARE
        if society.intervener_id in moves:
            # The intervener may only challenge or stay silent -- it never
            # propagates -- but whether it challenges is its own judgement.
            if moves[society.intervener_id].action is not Action.IGNORE:
                moves[society.intervener_id].action = Action.CHALLENGE
            else:
                # A silent intervener must be indistinguishable from the control
                # arm, where no intervener acts at all. Leaving it in the move
                # set would make it a visible NEUTRAL neighbour and hand every
                # citizen a conformity point the control arm never gets.
                silent_intervener = moves.pop(society.intervener_id)
                records.append(
                    ActionRecord(
                        episode_id=config.episode_id,
                        round_index=round_index,
                        agent_id=society.intervener_id,
                        role=personas[society.intervener_id].role.value,
                        action=Action.IGNORE,
                        claim_id=config.claim.claim_id,
                        reasoning_trace=silent_intervener.reasoning,
                        backbone=getattr(
                            policies[society.intervener_id], "backbone", "unknown"
                        ),
                        meta=silent_intervener.meta,
                    )
                )

        ledger.settle_round(
            round_index,
            {agent: (decision.action, config.claim) for agent, decision in moves.items()},
            neighbour_map,
        )

        for agent, decision in moves.items():
            guards.observe(agent, decision.utterance, config.claim.text)
            records.append(
                ActionRecord(
                    episode_id=config.episode_id,
                    round_index=round_index,
                    agent_id=agent,
                    role=personas[agent].role.value,
                    action=decision.action,
                    claim_id=config.claim.claim_id,
                    utterance=decision.utterance,
                    reasoning_trace=decision.reasoning,
                    credence=decision.credence,
                    backbone=getattr(policies[agent], "backbone", "unknown"),
                    repaired=decision.repaired,
                    meta=decision.meta,
                )
            )
            if round_index in PROBE_ROUNDS and personas[agent].is_citizen:
                # Out-of-band: the probe reads the private credence already
                # attached to the decision and never enters the shared feed.
                probes.append(
                    {
                        "round_index": round_index,
                        "agent_id": agent,
                        "credence": decision.credence,
                    }
                )

        # Publish this round's posts to every neighbour's feed.
        for agent, decision in moves.items():
            if decision.action is Action.IGNORE:
                continue
            item = FeedItem(
                agent_id=agent,
                action=decision.action,
                utterance=decision.utterance,
                round_index=round_index,
                is_intervener=(agent == society.intervener_id),
                is_disseminator=(agent == society.disseminator_id),
                face_threat=float(decision.meta.get("face_threat", 0.0)),
                epistemic_force=float(decision.meta.get("epistemic_force", 0.0)),
            )
            for neighbour in neighbour_map[agent]:
                feed_by_agent[neighbour].append(item)

        if guards.terminated_early:
            break

    # A guard-terminated episode still resolves, so its partial payoffs stay
    # comparable; the early-termination flag travels with the record and the
    # analysis excludes flagged episodes from the primary models.
    ledger.resolve({config.claim.claim_id: config.claim})

    repair_rates = [
        p.repair_rate() for p in policies.values() if isinstance(p, LLMCitizenPolicy)
    ]
    return EpisodeResult(
        config=config,
        records=records,
        ledger=ledger,
        schedule=schedule,
        guards=guards,
        role_assignment={role: node for role, node in role_to_node.items()},
        probes=probes,
        wall_time_s=time.perf_counter() - start,
        repair_rate=sum(repair_rates) / len(repair_rates) if repair_rates else 0.0,
    )
