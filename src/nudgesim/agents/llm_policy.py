"""LLM-backed citizen policy: prompt construction, parsing, and repair.

The observation is rendered as a compact, fixed-format brief so that the only
thing varying across conditions is the content the design manipulates -- never
the layout. Decoding temperature is pinned per role (plan section 5.9).
"""

from __future__ import annotations

import json
import re

from nudgesim.agents.policy import Decision, Observation
from nudgesim.backends.base import Backend, LLMRequest
from nudgesim.game.actions import Action, Stance

_JSON_RE = re.compile(r"\{.*\}", re.S)
_CREDENCE_RE = re.compile(r"(\d{1,3})\s*(?:%|percent|/\s*100)?")

_STANCE_WORD = {
    Stance.PRO: "backing it",
    Stance.CON: "disputing it",
    Stance.NEUTRAL: "saying nothing about it",
}


def render_observation(obs: Observation) -> str:
    lines: list[str] = [
        f"ROUND {obs.round_index + 1} of {obs.horizon}.",
        "",
        f"THE CLAIM UNDER DISCUSSION: \"{obs.claim.text}\"",
        "",
    ]

    if obs.feed:
        lines.append("WHAT YOUR NEIGHBOURS POSTED (most recent last):")
        for item in sorted(obs.feed, key=lambda i: (i.round_index, i.agent_id)):
            if item.action is Action.IGNORE:
                continue
            lines.append(
                f"  [r{item.round_index + 1}] {item.agent_id} ({item.action.value}): {item.utterance}"
            )
    else:
        lines.append("WHAT YOUR NEIGHBOURS POSTED: nothing yet.")
    lines.append("")

    if obs.neighbour_stances:
        summary = ", ".join(
            f"{agent} is {_STANCE_WORD[stance]}"
            for agent, stance in sorted(obs.neighbour_stances.items())
        )
        lines.append(f"WHERE YOUR NEIGHBOURS CURRENTLY STAND: {summary}.")
    lines.append(f"YOUR OWN CURRENT PUBLIC POSITION: you are {_STANCE_WORD[obs.own_stance]}.")

    if obs.payoff_visible:
        lines += [
            "",
            f"YOUR SCORE SO FAR: {obs.own_score:+.0f} points.",
            f"ROUNDS REMAINING BEFORE TRUE/FALSE IS REVEALED: {obs.rounds_remaining}.",
        ]

    lines += [
        "",
        "Take exactly one action now. Reply with JSON only.",
    ]
    return "\n".join(lines)


def parse_decision(text: str) -> Decision:
    """Parse a model reply, repairing the common failure modes.

    Repairs, in order: fenced or trailing prose around a JSON object; a bare
    action word with no JSON at all. A decision that had to be repaired is
    flagged so the rate can be reported per backbone rather than hidden -- a
    backbone that needs repair on 30% of turns is a finding, not a nuisance.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    match = _JSON_RE.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            action = Action.parse(str(payload.get("action", "")))
            credence = payload.get("credence")
            return Decision(
                action=action,
                reasoning=str(payload.get("reasoning", "")).strip(),
                utterance=str(payload.get("utterance", "")).strip(),
                credence=_coerce_credence(credence),
                repaired=match.group(0) != raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Last resort: find an action word in prose.
    try:
        action = Action.parse(raw)
    except ValueError:
        action = Action.IGNORE
    return Decision(
        action=action,
        reasoning=raw[:400],
        utterance="" if action is Action.IGNORE else raw[:280],
        credence=None,
        repaired=True,
        meta={"parse": "prose-fallback"},
    )


def _coerce_credence(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(100.0, float(value)))
    if isinstance(value, str):
        match = _CREDENCE_RE.search(value)
        if match:
            return max(0.0, min(100.0, float(match.group(1))))
    return None


class LLMCitizenPolicy:
    """A citizen whose action is chosen by a language model."""

    backbone_kind = "llm"

    def __init__(
        self,
        backend: Backend,
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.backbone = backend.name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.n_repaired = 0
        self.n_calls = 0

    async def decide(self, observation: Observation) -> Decision:
        request = LLMRequest(
            system=observation.persona.system_prompt,
            user=render_observation(observation),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        response = await self.backend.complete(request)
        decision = parse_decision(response.text)
        self.n_calls += 1
        self.n_repaired += int(decision.repaired)
        decision.meta.update(
            {
                "backbone_kind": self.backbone_kind,
                "cached": response.cached,
                "latency_s": round(response.latency_s, 4),
                "usage": response.usage,
            }
        )
        return decision

    def repair_rate(self) -> float:
        return self.n_repaired / self.n_calls if self.n_calls else 0.0
