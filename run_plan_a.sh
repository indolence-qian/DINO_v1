#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/plan_a.json}"

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Usage: bash run_plan_a.sh [--smoke] [--no-maps]' \
        'Environment: GPU_ID, CONFIG, OUTPUT_DIR, ATTRIBUTE_BANK, DINO_WEIGHTS, DINOTXT_WEIGHTS,' \
        'LLM_MODEL, SOURCE, TARGET, SOURCE_ROOT, TARGET_ROOT, BATCH_SIZE, EPOCHS, WORKERS,' \
        'ACCUMULATION_STEPS, IMAGE_SIZE, PYTHON_BIN, INSTALL_DEPS=1.'
    exit 0
fi

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    "$PYTHON_BIN" -m pip install -r requirements-plan-a.txt
fi

# Fail early if the environment or new companion weights are missing.
llm_model="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys

import torch
from plan_a.train import check_config, load_config
from pathlib import Path

config = load_config(Path(sys.argv[1]), smoke=False)
check_config(config)
if not torch.cuda.is_available():
    raise SystemExit("Cannot see CUDA. Check GPU_ID and activate the server PyTorch environment.")
print(config["llm_model"])
PY
)"

output="${OUTPUT_DIR:-runs/plan_a/$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$output"
exec > >(tee -a "$output/run.log") 2>&1
printf 'Physical GPU selector: %s; PyTorch device: cuda:0\n' "$CUDA_VISIBLE_DEVICES"
printf 'Run directory: %s\n' "$output"
git rev-parse --short HEAD > "$output/git_commit.txt"

bank="${ATTRIBUTE_BANK:-$output/generated_attributes.json}"
if [[ -z "${ATTRIBUTE_BANK:-}" ]]; then
    if [[ ! -f "$bank" ]]; then
        "$PYTHON_BIN" -m plan_a.attributes --model "$llm_model" --output "$bank" --device cuda:0
    fi
elif [[ ! -f "$bank" ]]; then
    printf 'Attribute bank is missing: %s\n' "$bank" >&2
    exit 1
fi

"$PYTHON_BIN" -m plan_a.train --config "$CONFIG" --attribute-bank "$bank" \
    --output "$output" --device cuda:0 "$@"
