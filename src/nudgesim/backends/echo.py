"""Deterministic offline backend used to exercise the LLM code path in tests.

It emits a syntactically valid decision JSON derived from a seeded RNG plus the
prompt hash, so the prompt-building, parsing and repair logic can be tested
without a network call. It is never used for a reported experimental result --
runs record the backbone id, and ``echo`` is rejected by the runner unless
``allow_echo`` is set.
"""

from __future__ import annotations

import hashlib
import json
import random

from nudgesim.backends.base import LLMRequest, LLMResponse


class EchoBackend:
    def __init__(self, name: str = "echo", seed: int = 0, malformed_rate: float = 0.0) -> None:
        self.name = name
        self.seed = seed
        self.malformed_rate = malformed_rate
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        digest = hashlib.sha256(
            f"{self.seed}|{request.cache_key(self.name)}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))

        if rng.random() < self.malformed_rate:
            # Exercise the repair path: prose instead of JSON.
            return LLMResponse(
                text="I think I will endorse this one, it matches what my neighbours say.",
                backbone=self.name,
            )

        action = rng.choice(["SHARE", "ENDORSE", "CHALLENGE", "IGNORE"])
        payload = {
            "reasoning": f"echo backend deterministic trace ({action.lower()} branch)",
            "action": action,
            "utterance": f"[{self.name}] {action.lower()} response",
            "credence": round(rng.random() * 100, 1),
        }
        return LLMResponse(text=json.dumps(payload), backbone=self.name)
