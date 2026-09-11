"""OpenAI-compatible chat-completions backend.

Covers the hosted OpenAI API and any OpenAI-compatible server, which is how the
open-weights citizen backbones are served under vLLM (plan section 7).
"""

from __future__ import annotations

import os
import time

from nudgesim.backends.base import BackendError, LLMRequest, LLMResponse


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise BackendError(
                "openai package not installed; install nudgesim[llm]"
            ) from exc
        api_key = os.environ.get(api_key_env)
        if not api_key and not base_url:
            raise BackendError(f"{api_key_env} is not set and no base_url was given")
        self.name = model
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop:
            kwargs["stop"] = list(request.stop)
        result = await self._client.chat.completions.create(**kwargs)
        usage = getattr(result, "usage", None)
        return LLMResponse(
            text=result.choices[0].message.content or "",
            backbone=self.name,
            latency_s=time.perf_counter() - start,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        )
