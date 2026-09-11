#!/usr/bin/env bash
# One-command reproduction (plan section 9, Month 4 deliverable).
#
#   ./reproduce.sh            offline surrogate run: 1,080 episodes, ~15s
#   ./reproduce.sh --full     also runs the null-DGP and high-power controls
#
# With the real corpora and real backbones:
#   ./reproduce.sh --liar data/raw/liar --pheme data/raw/pheme --llm
set -euo pipefail

PY="${PYTHON:-python}"
OUT="${OUT:-runs}"
DATA_ARGS=()
FULL=0
LLM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --liar)  DATA_ARGS+=(--liar "$2"); shift 2 ;;
    --pheme) DATA_ARGS+=(--pheme "$2"); shift 2 ;;
    --llm)   LLM="--liar-backbones"; shift ;;
    --full)  FULL=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "== 0/5 tests (the payoff ledger gates everything downstream)"
$PY -m pytest tests/ -q

echo "== 1/5 manipulation checks and design references"
$PY -m nudgesim.cli "${DATA_ARGS[@]}" check --out "$OUT/checks"

echo "== 2/5 calibration gate (plan section 5.7) -- hard Go/No-Go"
$PY -m nudgesim.cli "${DATA_ARGS[@]}" calibrate --out "$OUT/calibration"

echo "== 3/5 main grid: 1,080 episodes"
$PY -m nudgesim.cli "${DATA_ARGS[@]}" $LLM run --out "$OUT/main" --run-id main-declared

echo "== 4/5 preregistered analysis"
$PY -m nudgesim.cli analyze --run "$OUT/main" --out "$OUT/main/analysis"
$PY -m analysis.figures "$OUT/main"

if [[ $FULL -eq 1 ]]; then
  echo "== 5/5 controls: null DGP and high-power confirmation"
  for spec in "null:null_dgp:30" "declared:highpower:200" "null:null_highpower:200"; do
    IFS=: read -r dgp name seeds <<<"$spec"
    $PY -m nudgesim.cli "${DATA_ARGS[@]}" --dgp "$dgp" run \
        --out "$OUT/$name" --run-id "$name" --arms core --core-seeds "$seeds"
    $PY -m nudgesim.cli analyze --run "$OUT/$name" --out "$OUT/$name/analysis"
  done
  echo "== gate teeth check: the same gate with and without the novelty channel"
  # Swept across seeds on purpose: a single seed is not evidence either way,
  # since the asymmetry margin without the novelty channel sits on top of the
  # threshold and clears it occasionally by chance. See docs/CALIBRATION.md.
  for dgp in declared no-novelty; do
    for s in 1 2 3 4 5 6; do
      $PY -m nudgesim.cli --seed "$s" --dgp "$dgp" calibrate \
          --calibration-episodes 200 --out "$OUT/gate_${dgp}_${s}" >/dev/null 2>&1 || true
    done
  done
  $PY - "$OUT" <<'PYGATE'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for dgp in ("declared", "no-novelty"):
    rows = []
    for s in range(1, 7):
        f = out / f"gate_{dgp}_{s}" / "calibration.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows.append((d["verdict"], d["asymmetry"]["propagation_margin"]))
    if rows:
        go = sum(1 for v, _ in rows if v == "GO")
        mean = sum(m for _, m in rows) / len(rows)
        print(f"  {dgp:12s} GO {go}/{len(rows)}  mean propagation margin {mean:+.4f}")
PYGATE
else
  echo "== 5/5 skipped (pass --full for the null-DGP and high-power controls)"
fi

echo
echo "done. results: $OUT/main/results.parquet · $OUT/main/analysis/analysis.json"
