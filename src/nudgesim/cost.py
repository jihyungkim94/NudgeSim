"""Token accounting and grid pricing (plan section 12).

The plan asserts that swapping citizens onto real language models is
affordable. That assertion was never checked against a measured prompt, so
this module measures instead of asserting: it meters the prompts the episode
loop actually assembles, then prices the grid against the published rate card.

Two numbers drive everything and both are measured, not assumed:

  input tokens   the real system + user strings, tokenised
  call count     agents x rounds x episodes, taken from the grid spec

Output tokens cannot be measured without a key, so they are a declared
parameter with two reference points: an exemplar reply that satisfies the
response schema, and the ``max_tokens`` ceiling the plan pins. Both are
reported, so the budget is a bracket rather than a point estimate dressed up
as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nudgesim.backends.base import Backend, LLMRequest, LLMResponse

# ---------------------------------------------------------------- tokenisation

try:  # tiktoken is a proxy, not the Anthropic tokenizer; close enough to budget on.
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")
    TOKENIZER_NAME = "tiktoken/cl100k_base"
except Exception:  # noqa: BLE001 - any import or download failure falls back
    _ENCODING = None
    TOKENIZER_NAME = "chars-per-token fallback"

# Measured on this project's own prompt corpus -- 3.05M characters of assembled
# system and user blocks over 20 real episodes came to 721,884 cl100k tokens.
# The generic 4.0 rule over-counts our prompts by 6%; this is the fallback for
# an environment where the tokeniser's vocabulary cannot be fetched.
CHARS_PER_TOKEN = 4.231


def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return round(len(text) / CHARS_PER_TOKEN)


# A reply that satisfies the response schema without padding: one clause of
# reasoning, the action, one sentence of utterance, a credence. Shorter than
# this and the model is not answering the prompt; much longer and it is
# ignoring "Reply with JSON only".
EXEMPLAR_COMPLETION = (
    '{"reasoning": "Two neighbours are backing it and the sourcing looks thin, '
    'so a public correction is worth the cost to me.", "action": "CHALLENGE", '
    '"utterance": "I looked for the original report and could not find it -- '
    'worth checking before this spreads further.", "credence": 22.0}'
)

# ---------------------------------------------------------------------- pricing

@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, standard (non-batch, non-cached) rates."""

    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    tier: str


# Anthropic public list prices. Verified 2026-06-24; re-verify before drawing
# on a grant, because rate cards move and this table does not.
CATALOGUE: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice("claude-opus-5", 5.0, 25.0, "frontier"),
    "claude-sonnet-5": ModelPrice("claude-sonnet-5", 2.0, 10.0, "mid"),
    "claude-sonnet-4-6": ModelPrice("claude-sonnet-4-6", 3.0, 15.0, "mid"),
    "claude-haiku-4-5": ModelPrice("claude-haiku-4-5", 1.0, 5.0, "small"),
}

BATCH_MULTIPLIER = 0.5          # Batch API, 50% off both directions
CACHE_READ_MULTIPLIER = 0.1     # cached input token read
CACHE_WRITE_MULTIPLIER = 1.25   # writing a cache entry costs a premium once

# ------------------------------------------------------------------- metering


@dataclass
class CallRecord:
    system_tokens: int
    user_tokens: int
    cache_key: str


class MeteringBackend:
    """Tokenises every request, then delegates. Adds no network of its own."""

    def __init__(self, inner: Backend) -> None:
        self.inner = inner
        self.name = inner.name
        self.records: list[CallRecord] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.records.append(
            CallRecord(
                system_tokens=count_tokens(request.system),
                user_tokens=count_tokens(request.user),
                cache_key=request.cache_key(self.name),
            )
        )
        return await self.inner.complete(request)


@dataclass
class TokenProfile:
    """What one episode of prompts actually costs in input tokens."""

    n_calls: int
    n_distinct_prompts: int
    n_episodes: int
    system_tokens_total: int
    user_tokens_total: int
    max_input_tokens: int
    tokenizer: str = TOKENIZER_NAME
    truncated_episodes: int = 0

    @property
    def input_tokens_total(self) -> int:
        return self.system_tokens_total + self.user_tokens_total

    @property
    def mean_input_tokens(self) -> float:
        return self.input_tokens_total / max(1, self.n_calls)

    @property
    def calls_per_episode(self) -> float:
        return self.n_calls / max(1, self.n_episodes)

    @property
    def system_share(self) -> float:
        """Fraction of input that is the reusable, cacheable persona prefix."""
        return self.system_tokens_total / max(1, self.input_tokens_total)

    @property
    def duplicate_rate(self) -> float:
        """Share of calls a response cache would serve without paying at all."""
        return 1.0 - self.n_distinct_prompts / max(1, self.n_calls)

    def to_json(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer,
            "episodes_metered": self.n_episodes,
            "calls": self.n_calls,
            "calls_per_episode": round(self.calls_per_episode, 1),
            "mean_input_tokens": round(self.mean_input_tokens, 1),
            "max_input_tokens": self.max_input_tokens,
            "system_share": round(self.system_share, 3),
            "duplicate_rate": round(self.duplicate_rate, 3),
            "truncated_episodes": self.truncated_episodes,
        }


def profile(records: list[CallRecord], n_episodes: int) -> TokenProfile:
    return TokenProfile(
        n_calls=len(records),
        n_distinct_prompts=len({r.cache_key for r in records}),
        n_episodes=n_episodes,
        system_tokens_total=sum(r.system_tokens for r in records),
        user_tokens_total=sum(r.user_tokens for r in records),
        max_input_tokens=max((r.system_tokens + r.user_tokens for r in records), default=0),
    )


# ------------------------------------------------------------------- costing


@dataclass
class CostEstimate:
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    levers: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_list": round(self.usd, 2),
            **{k: round(v, 2) for k, v in self.levers.items()},
        }


def estimate(
    prof: TokenProfile,
    price: ModelPrice,
    *,
    calls: int,
    output_tokens: int,
) -> CostEstimate:
    """Price ``calls`` calls at this profile's mean prompt length.

    ``levers`` prices the same workload under each cost reduction separately,
    so the plan can state what each one buys rather than quoting one blended
    number nobody can audit.
    """
    input_tokens = round(prof.mean_input_tokens * calls)
    out_tokens = output_tokens * calls
    in_usd = input_tokens / 1e6 * price.input_usd_per_mtok
    out_usd = out_tokens / 1e6 * price.output_usd_per_mtok
    total = in_usd + out_usd

    # Prompt caching bills the persona prefix at the read rate after the first
    # write; the per-round user block is never repeated, so it stays at list.
    cached_in = prof.system_share * in_usd * CACHE_READ_MULTIPLIER + (
        1 - prof.system_share
    ) * in_usd
    deduped = total * (1 - prof.duplicate_rate)

    return CostEstimate(
        model=price.model,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=out_tokens,
        usd=total,
        levers={
            "usd_batch": total * BATCH_MULTIPLIER,
            "usd_prompt_cache": cached_in + out_usd,
            "usd_response_cache": deduped,
            "usd_all_levers": (cached_in + out_usd) * BATCH_MULTIPLIER * (
                1 - prof.duplicate_rate
            ),
        },
    )


# ------------------------------------------------------------------ metering run


# A word pool the stub samples replies from. Ordinary English of the register
# citizens write in, so token counts land where a real reply would.
_STUB_VOCAB: list[str] = (
    "claim source report checking evidence thread posted reading again unclear "
    "sourcing thin holds spreading further original context numbers figure "
    "official statement release quoted before someone flagged version earlier "
    "later timeline account outlet corrected update disputed backing agreed "
    "concerned honestly probably certainly wrong right doubtful careful seems "
    "looks matters worth sharing passing along stands check verify trust"
).split()

class _VariedStub:
    """Offline stand-in that answers with different wording every round.

    The echo backend repeats one fixed sentence per action, which trips the
    repetition guard and ends episodes at round 10 of 12 -- so metering against
    it would under-count calls by a sixth. Real models vary their wording, so
    the stub does too.
    """

    def __init__(self) -> None:
        self.name = "metering-stub"
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        import json
        import random

        self.calls += 1
        rng = random.Random(request.cache_key(self.name))
        action = rng.choice(["SHARE", "ENDORSE", "CHALLENGE", "IGNORE"])
        # Sampled wording, ~20 words, the length a real reply runs to. Fixed
        # phrasing would trip the repetition guard and cut episodes short,
        # which would under-count both calls and feed length.
        words = rng.sample(_STUB_VOCAB, 20)
        payload = {
            "reasoning": " ".join(rng.sample(_STUB_VOCAB, 18)),
            "action": action,
            "utterance": "" if action == "IGNORE" else " ".join(words) + ".",
            "credence": round(rng.random() * 100, 1),
        }
        return LLMResponse(text=json.dumps(payload), backbone=self.name)


async def meter(configs, *, concurrency: int = 8) -> TokenProfile:
    """Run real episodes with metered LLM citizens and return the token profile.

    What is measured is the prompt the episode loop assembles -- identical
    whatever answers come back -- and the number of calls, which the loop also
    fixes. Completions come from an offline stub; nothing touches the network.
    """
    import asyncio
    from dataclasses import replace

    from nudgesim.agents.llm_policy import LLMCitizenPolicy
    from nudgesim.env.episode import run_episode

    meters: list[MeteringBackend] = []

    def factory(_agent_id, _persona, seed):
        backend = MeteringBackend(_VariedStub())
        meters.append(backend)
        return LLMCitizenPolicy(backend, temperature=0.7, seed=seed)

    semaphore = asyncio.Semaphore(concurrency)
    truncated = 0

    async def one(config) -> None:
        nonlocal truncated
        async with semaphore:
            result = await run_episode(
                replace(config, policy_factory=factory, backend_spec=None)
            )
        truncated += int(result.guards.terminated_early)

    await asyncio.gather(*(one(c) for c in configs))
    prof = profile([r for m in meters for r in m.records], len(configs))
    prof.truncated_episodes = truncated
    return prof
