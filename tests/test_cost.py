"""Token accounting and grid pricing."""

from __future__ import annotations

import asyncio

import pytest

from nudgesim.cost import (
    CATALOGUE,
    CallRecord,
    MeteringBackend,
    ModelPrice,
    count_tokens,
    estimate,
    meter,
    profile,
)
from nudgesim.backends.echo import EchoBackend
from nudgesim.backends.base import LLMRequest
from nudgesim.env.episode import EpisodeConfig
from nudgesim.intervention.scheduler import Timing
from nudgesim.intervention.tone import Tone


def test_count_tokens_scales_with_text():
    assert count_tokens("") == 0
    assert count_tokens("hello world") < count_tokens("hello world " * 50)


def test_metering_backend_records_every_call_and_delegates():
    backend = MeteringBackend(EchoBackend(seed=1))
    request = LLMRequest(system="you are a citizen", user="what now?")
    response = asyncio.run(backend.complete(request))
    assert response.text
    assert len(backend.records) == 1
    assert backend.records[0].system_tokens > 0
    assert backend.records[0].user_tokens > 0


def test_profile_derives_shares_from_records():
    records = [
        CallRecord(system_tokens=60, user_tokens=40, cache_key="a"),
        CallRecord(system_tokens=60, user_tokens=40, cache_key="a"),
        CallRecord(system_tokens=60, user_tokens=140, cache_key="b"),
    ]
    prof = profile(records, n_episodes=1)
    assert prof.n_calls == 3
    assert prof.mean_input_tokens == pytest.approx(400 / 3)
    assert prof.system_share == pytest.approx(180 / 400)
    assert prof.duplicate_rate == pytest.approx(1 / 3)
    assert prof.max_input_tokens == 200


def test_estimate_prices_both_directions_and_orders_the_levers():
    prof = profile(
        [CallRecord(system_tokens=500, user_tokens=500, cache_key=str(i)) for i in range(10)],
        n_episodes=1,
    )
    price = ModelPrice("test", input_usd_per_mtok=10.0, output_usd_per_mtok=20.0, tier="mid")
    est = estimate(prof, price, calls=1_000, output_tokens=100)

    # 1,000 calls x 1,000 input tokens at $10/Mtok = $10; output $2.
    assert est.usd == pytest.approx(12.0)
    assert est.levers["usd_batch"] == pytest.approx(6.0)
    # Half the input is a cacheable system prefix, billed at 0.1x.
    assert est.levers["usd_prompt_cache"] == pytest.approx(5.0 + 0.5 + 2.0)
    # No duplicate prompts here, so a response cache buys nothing.
    assert est.levers["usd_response_cache"] == pytest.approx(12.0)


def test_catalogue_tiers_are_ordered_by_price():
    by_tier = {p.tier: p for p in CATALOGUE.values()}
    assert by_tier["small"].input_usd_per_mtok < by_tier["frontier"].input_usd_per_mtok
    assert by_tier["small"].output_usd_per_mtok < by_tier["frontier"].output_usd_per_mtok


def test_meter_runs_full_horizon_without_tripping_the_repetition_guard(pool, library):
    configs = [
        EpisodeConfig(
            episode_id=f"cost-{i}",
            claim=pool.false_claims[i],
            motif=library.motifs[i],
            timing=Timing.EARLY,
            tone=Tone.EMPATHETIC,
            seed=i,
        )
        for i in range(3)
    ]
    prof = asyncio.run(meter(configs, concurrency=2))
    horizon = configs[0].payoff.horizon
    citizens = len(configs[0].society.citizen_ids)
    assert prof.truncated_episodes == 0
    assert prof.calls_per_episode == pytest.approx(citizens * horizon)
    assert prof.mean_input_tokens > 100
