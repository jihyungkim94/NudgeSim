"""LLM backend abstraction.

The experiment code never imports a vendor SDK directly. Every citizen backbone
(plan factor F3) and the judge model reach the network through this one
interface, which gives us three things the plan asks for: response caching and
batching to hold cost down (section 12), a decoding temperature pinned per role
(section 5.9), and the ability to swap the whole grid onto a different provider
without touching the game, the metrics or the analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a completion after its retries."""


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    temperature: float = 0.7
    max_tokens: int = 400
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def cache_key(self, backbone: str) -> str:
        payload = json.dumps(
            {
                "backbone": backbone,
                "system": self.system,
                "user": self.user,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "stop": list(self.stop),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LLMResponse:
    text: str
    backbone: str
    latency_s: float = 0.0
    cached: bool = False
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Minimal contract every model family must satisfy."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class CachedBackend:
    """Disk-backed response cache in front of any backend.

    Identical (backbone, prompt, temperature, seed) tuples recur constantly
    across a 1,000-episode grid -- early rounds of the control arm especially --
    so caching is the single largest cost lever available (plan section 12).
    """

    def __init__(self, inner: Backend, cache_dir: str | Path = ".cache/llm") -> None:
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key(self.inner.name)
        path = self._path(key)
        if path.exists():
            self.hits += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=payload["text"],
                backbone=payload["backbone"],
                latency_s=0.0,
                cached=True,
                usage=payload.get("usage", {}),
            )
        self.misses += 1
        response = await self.inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"text": response.text, "backbone": response.backbone, "usage": response.usage}
            ),
            encoding="utf-8",
        )
        return response

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class RetryingBackend:
    """Bounded exponential-backoff wrapper; rate limits are expected, not fatal."""

    def __init__(self, inner: Backend, *, attempts: int = 4, base_delay: float = 1.0) -> None:
        self.inner = inner
        self.name = inner.name
        self.attempts = attempts
        self.base_delay = base_delay

    async def complete(self, request: LLMRequest) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return await self.inner.complete(request)
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last = exc
                if attempt == self.attempts - 1:
                    break
                await asyncio.sleep(self.base_delay * (2**attempt))
        raise BackendError(f"{self.name} failed after {self.attempts} attempts: {last}") from last


def build_backend(spec: dict[str, Any]) -> Backend:
    """Construct a backend from a config block.

    ``spec`` looks like::

        {provider: openai|anthropic|vllm|echo, model: <id>, base_url: ..., cache: true}

    Vendor SDKs are imported lazily so the core package installs and the offline
    test suite runs without them.
    """
    provider = str(spec.get("provider", "echo")).lower()
    model = str(spec.get("model", provider))
    backend: Backend

    if provider == "echo":
        from nudgesim.backends.echo import EchoBackend

        backend = EchoBackend(name=model, seed=int(spec.get("seed", 0)))
    elif provider in ("openai", "vllm", "openai_compat"):
        from nudgesim.backends.openai_compat import OpenAICompatBackend

        backend = OpenAICompatBackend(
            model=model,
            base_url=spec.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from nudgesim.backends.anthropic_api import AnthropicBackend

        backend = AnthropicBackend(model=model, api_key_env=spec.get("api_key_env", "ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"unknown backend provider: {provider!r}")

    if spec.get("retry", True):
        backend = RetryingBackend(backend)
    if spec.get("cache", True):
        backend = CachedBackend(backend, cache_dir=spec.get("cache_dir", ".cache/llm"))
    return backend


def timed(start: float) -> float:
    return time.perf_counter() - start
