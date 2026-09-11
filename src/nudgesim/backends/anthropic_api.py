"""Anthropic Messages API backend."""

from __future__ import annotations

import os
import time

from nudgesim.backends.base import BackendError, LLMRequest, LLMResponse


class AnthropicBackend:
    def __init__(self, model: str, *, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise BackendError(
                "anthropic package not installed; install nudgesim[llm]"
            ) from exc
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise BackendError(f"{api_key_env} is not set")
        self.name = model
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        result = await self._client.messages.create(
            model=self.model,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=list(request.stop) or None,
        )
        text = "".join(block.text for block in result.content if block.type == "text")
        return LLMResponse(
            text=text,
            backbone=self.name,
            latency_s=time.perf_counter() - start,
            usage={
                "prompt_tokens": result.usage.input_tokens,
                "completion_tokens": result.usage.output_tokens,
            },
        )
