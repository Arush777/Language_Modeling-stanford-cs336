#!/usr/bin/env bash
# Part B — gradient checkpointing on 1× A100-80GB
# Submit: bsub < scripts/job_part_b.sh
#BSUB -J cs336-a2-partB
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=1]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o logs/cs336_a2_partB_%J.out
#BSUB -e logs/cs336_a2_partB_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs results/part_b

hostname
date -u
nvidia-smi -L

source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

echo "=== Part B: xl B=4 T=2048 forward_backward checkpoint sweep ==="
uv run python -m cs336_systems.checkpoint_bench \
  --batch-size 4 --context-length 2048 \
  --mode forward_backward \
  --segment-sizes 1,2,4,8,16,32 \
  --out-csv "results/part_b/checkpoint_xl_T2048_${RUN_TS}.csv" \
  --out-json "results/part_b/checkpoint_xl_T2048_${RUN_TS}.json"

# Also compare neighbors around optimum for train mode if memory allows
echo "=== Part B: train mode segment sweep (may OOM some) ==="
uv run python -m cs336_systems.checkpoint_bench \
  --batch-size 4 --context-length 2048 \
  --mode train \
  --segment-sizes 1,2,4,8,16,32 \
  --out-csv "results/part_b/checkpoint_xl_T2048_train_${RUN_TS}.csv" \
  --out-json "results/part_b/checkpoint_xl_T2048_train_${RUN_TS}.json" || true

cp -f "results/part_b/checkpoint_xl_T2048_${RUN_TS}.csv" results/part_b/checkpoint_latest.csv
echo "=== Part B done ==="
date -u
ls -lah results/part_b/
