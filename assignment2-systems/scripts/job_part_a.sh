#!/usr/bin/env bash
# CS336 A2 — Part A full suite on ONE A100-80GB (normal queue).
# Submit:  bsub < scripts/job_part_a.sh
#
# Outputs (under assignment2-systems/):
#   results/part_a/timings/*.csv
#   results/part_a/memory/*.pickle   → drag into https://pytorch.org/memory_viz → PNG for writeup
#   results/part_a/nsys/*            → open in Nsight Systems GUI
#   results/part_a/figures/*.png     → commit these to GitHub
#   results/part_a/logs/             → copy of job stdout highlights
#BSUB -J cs336-a2-partA
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=1]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -o logs/cs336_a2_partA_%J.out
#BSUB -e logs/cs336_a2_partA_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs \
  results/part_a/timings \
  results/part_a/memory \
  results/part_a/nsys \
  results/part_a/figures \
  results/part_a/logs

echo "=== Part A host / GPU ==="
hostname
date -u
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
CSV="results/part_a/timings/part_a_${RUN_TS}.csv"
CSV_LATEST="results/part_a/timings/part_a_latest.csv"

echo "=== 0) Mixed-precision accumulation + ToyModel dtypes ==="
uv run python -m cs336_systems.mixed_precision_experiments --all \
  | tee "results/part_a/logs/mixed_precision_${RUN_TS}.txt"

echo "=== 1) FP32 timing sweep (all models × modes, warmup=5, steps=10) ==="
uv run python -m cs336_systems.benchmark \
  --model all --mode all \
  --warmup 5 --steps 10 \
  --batch-size 4 --context-length 512 \
  --out-csv "$CSV" \
  --out-json "results/part_a/timings/part_a_fp32_${RUN_TS}.json" \
  --skip-oom || true

echo "=== 2) Warmup ablation (small / train only) ==="
for W in 0 1 2 5; do
  uv run python -m cs336_systems.benchmark \
    --model small --mode train \
    --warmup "$W" --steps 10 \
    --batch-size 4 --context-length 512 \
    --out-csv "$CSV" \
    --skip-oom || true
done

echo "=== 3) BF16 timing sweep ==="
uv run python -m cs336_systems.benchmark \
  --model all --mode all \
  --warmup 5 --steps 10 \
  --batch-size 4 --context-length 512 \
  --bf16 \
  --out-csv "$CSV" \
  --out-json "results/part_a/timings/part_a_bf16_${RUN_TS}.json" \
  --skip-oom || true

echo "=== 4) Memory snapshots (xl, T=128 and T=2048) ==="
for T in 128 2048; do
  for MODE in forward train; do
    for BF in "" "--bf16"; do
      TAG="xl_${MODE}_T${T}"
      if [[ -n "$BF" ]]; then TAG="${TAG}_bf16"; fi
      uv run python -m cs336_systems.benchmark \
        --model xl --mode "$MODE" \
        --warmup 2 --steps 1 \
        --batch-size 4 --context-length "$T" \
        $BF \
        --memory-snapshot "results/part_a/memory/${TAG}_${RUN_TS}.pickle" \
        --out-csv "$CSV" \
        --skip-oom || true
    done
  done
done

echo "=== 5) Nsight Systems (if nsys present) ==="
if command -v nsys >/dev/null 2>&1; then
  # Small model train step — NVTX measure range
  uv run nsys profile \
    -o "results/part_a/nsys/small_train_T512_${RUN_TS}" \
    --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    --python-backtrace=cuda \
    -- python -m cs336_systems.benchmark \
      --model small --mode train \
      --warmup 2 --steps 3 \
      --batch-size 4 --context-length 512 \
      --nvtx \
      --skip-oom

  # Medium forward long context (interesting attention)
  uv run nsys profile \
    -o "results/part_a/nsys/medium_forward_T1024_${RUN_TS}" \
    --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    -- python -m cs336_systems.benchmark \
      --model medium --mode forward \
      --warmup 2 --steps 3 \
      --batch-size 4 --context-length 1024 \
      --nvtx \
      --skip-oom
else
  echo "nsys not found on this node — skip Nsight; install/load module later if needed"
fi

cp -f "$CSV" "$CSV_LATEST"

echo "=== 6) Plot figures ==="
uv run python -m cs336_systems.plot_part_a --csv "$CSV_LATEST" --out-dir results/part_a/figures

echo "=== Part A job complete ==="
date -u
ls -lah results/part_a/timings results/part_a/figures results/part_a/memory results/part_a/nsys || true
