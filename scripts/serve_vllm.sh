#!/usr/bin/env bash
# Serve an open-weights citizen backbone behind an OpenAI-compatible endpoint.
#
#     ./scripts/serve_vllm.sh meta-llama/Llama-3.3-70B-Instruct
#     ./scripts/serve_vllm.sh Qwen/Qwen3-30B-A3B 8001
#
# The engine speaks one OpenAI-compatible interface, so an open-weights level is
# the same code path as a hosted one -- only the base URL differs:
#
#     nudgesim --models openai:Qwen/Qwen3-30B-A3B \
#       --backend-url http://localhost:8000/v1 run --out runs/vllm
#
# These levels cost GPU time rather than tokens, which is what makes them the
# arm that shows a result was not simply bought with spend.
set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <model-id> [port] [tensor-parallel-size]}"
PORT="${2:-8000}"
TP="${3:-1}"

command -v vllm >/dev/null 2>&1 || {
  echo "vllm not found. Install it in an environment with a matching CUDA/ROCm build:" >&2
  echo "    pip install vllm" >&2
  exit 1
}

echo "serving ${MODEL} on :${PORT} (tensor-parallel-size=${TP})"
exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --disable-log-requests
