#!/usr/bin/env python3
"""A local OpenAI-compatible server, for exercising the LLM path without a key.

The LLM code path -- prompt assembly, HTTP transport, concurrency, JSON parsing,
repair, caching, cost accounting -- has never run against a real endpoint. This
serves /v1/chat/completions so all of it can be verified end to end; only the
model itself is stubbed.

    python scripts/mock_llm_server.py --port 8077 &
    nudgesim --backend-url http://127.0.0.1:8077/v1 run ...

It deliberately misbehaves the way real models do: a configurable share of
replies come back as prose instead of JSON, wrapped in markdown fences, or with
the action word in the wrong case. If the pipeline cannot survive this server it
will not survive a real one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIONS = ("SHARE", "ENDORSE", "CHALLENGE", "IGNORE")

_stats = {"requests": 0, "malformed": 0}
_lock = threading.Lock()


def _decide(system: str, user: str, seed: int, malformed_rate: float) -> str:
    """Deterministic pseudo-decision, seeded by the prompt so caching is testable."""
    digest = hashlib.sha256(f"{seed}|{system}|{user}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    # Lean on what the prompt actually says, so the transcript is not nonsense.
    pro = len(re.findall(r"backing it", user))
    con = len(re.findall(r"disputing it", user))
    weights = [2 + pro, 2 + pro, 1 + 2 * con, 2]
    action = rng.choices(ACTIONS, weights=weights, k=1)[0]

    # Vary the wording. A model that emits one fixed sentence per action trips
    # the engine's repetition guard every episode, which would make this server
    # test the guard rather than the pipeline.
    openers = ("Having read it,", "For what it is worth,", "My take:",
               "Looking at this again,", "Honestly,", "On reflection,")
    bodies = {
        "SHARE": ("worth passing on", "people should see this", "sending this along"),
        "ENDORSE": ("this looks right to me", "I think it holds up", "I would back this"),
        "CHALLENGE": ("I do not think this stands", "the sourcing is thin",
                      "someone should check this"),
    }
    utterance = "" if action == "IGNORE" else (
        f"{rng.choice(openers)} {rng.choice(bodies[action])}."
    )
    payload = {
        "reasoning": f"mock model: {pro} backing, {con} disputing in view",
        "action": action,
        "utterance": utterance,
        "credence": round(rng.random() * 100, 1),
    }
    body = json.dumps(payload)

    roll = rng.random()
    if roll < malformed_rate * 0.4:
        with _lock:
            _stats["malformed"] += 1
        return f"Sure, here is my decision:\n```json\n{body}\n```\nHope that helps."
    if roll < malformed_rate * 0.7:
        with _lock:
            _stats["malformed"] += 1
        return f"I will {action.lower()} this one."
    if roll < malformed_rate:
        with _lock:
            _stats["malformed"] += 1
        return body.replace(f'"{action}"', f'"{action.lower()}"')
    return body


class Handler(BaseHTTPRequestHandler):
    malformed_rate = 0.15
    seed = 0

    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad json")
            return

        messages = body.get("messages", [])
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        with _lock:
            _stats["requests"] += 1

        text = _decide(system, user, self.seed, self.malformed_rate)
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(system.split()) + len(user.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(system.split()) + len(user.split()) + len(text.split()),
            },
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        raw = json.dumps(_stats).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--malformed-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Handler.malformed_rate = args.malformed_rate
    Handler.seed = args.seed
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock OpenAI-compatible server on http://127.0.0.1:{args.port}/v1 "
          f"(malformed_rate={args.malformed_rate})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
