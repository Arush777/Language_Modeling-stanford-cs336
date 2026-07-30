#!/usr/bin/env bash
#BSUB -J cs336-a2-smoke
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=1]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -o logs/cs336_a2_smoke_%J.out
#BSUB -e logs/cs336_a2_smoke_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs

echo "=== host / GPU ==="
hostname
date -u
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source scripts/env.sh
uv sync
uv run python - <<'PY'
import torch
import cs336_basics
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cs336_basics OK")
PY

echo "=== smoke done ==="
