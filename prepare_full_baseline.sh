#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LOOP_REPO OUTPUT_ROOT" >&2
  exit 2
fi

loop_repo=$(realpath "$1")
output_root=$(realpath -m "$2")
python_bin=${PYTHON_BIN:-python3}
plan_root="$output_root/plans"
mkdir -p "$plan_root"
cd "$loop_repo"
export PYTHONPATH="$loop_repo/src${PYTHONPATH:+:$PYTHONPATH}"

prepare() {
  local benchmark=$1
  local prompt_file=$2
  local method=$3
  shift 3
  local plan="$plan_root/${method}_${benchmark}.jsonl"
  if [[ -e "$plan" ]]; then
    echo "refusing to overwrite existing plan: $plan" >&2
    exit 1
  fi
  "$python_bin" scripts/prepare_benchmark.py \
    --config configs/scale_rae_1p5b.json \
    --prompt-file "$prompt_file" \
    --benchmark "$benchmark" \
    --output-root "$output_root/images" \
    --plan "$plan" "$@"
}

for benchmark in geneval dpgbench; do
  if [[ "$benchmark" == geneval ]]; then
    prompt_file=prompts/geneval_553.txt
  else
    prompt_file=prompts/dpgbench_1065.txt
  fi
  prepare "$benchmark" "$prompt_file" without_loop --token-loop none
  prepare "$benchmark" "$prompt_file" dense_token_loop --token-loop dense
  prepare "$benchmark" "$prompt_file" sparse_token_loop --token-loop sparse
  prepare "$benchmark" "$prompt_file" loop_guidance_dense_token_loop --token-loop dense --loop-guidance
done

"$python_bin" - "$plan_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
counts = {}
for path in sorted(root.glob("*.jsonl")):
    counts[path.name] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
expected = 4 * (553 + 1065)
actual = sum(counts.values())
if actual != expected:
    raise SystemExit(f"expected {expected} planned images, found {actual}: {counts}")
print(json.dumps({"plans": counts, "total_images": actual}, indent=2, sort_keys=True))
PY
