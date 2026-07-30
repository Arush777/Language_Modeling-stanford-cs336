#!/usr/bin/env bash
# Part C — attention benches + FlashAttention tests on 1× A100-80GB
# Submit: bsub < scripts/job_part_c.sh
#BSUB -J cs336-a2-partC
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=1]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/cs336_a2_partC_%J.out
#BSUB -e logs/cs336_a2_partC_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs results/part_c

hostname
date -u
nvidia-smi -L
source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

echo "=== pytest flash attention (pytorch + triton) ==="
uv run pytest -k "test_flash" -v --tb=short | tee "results/part_c/pytest_flash_${RUN_TS}.txt"

echo "=== naive attention + torch.compile sweep ==="
uv run python -m cs336_systems.attention_bench \
  --out-csv "results/part_c/attention_bench_${RUN_TS}.csv" \
  --iters 50 --warmup 3 || true

echo "=== flash vs naive latency (causal, batch=1) ==="
uv run python -m cs336_systems.flash_bench \
  --out-csv "results/part_c/flash_bench_${RUN_TS}.csv" \
  --seq-sizes "128,256,512,1024,2048,4096,8192" \
  --d-sizes "16,32,64,128" \
  --dtypes "float32,bfloat16" || true

cp -f "results/part_c/attention_bench_${RUN_TS}.csv" results/part_c/attention_bench_latest.csv 2>/dev/null || true
cp -f "results/part_c/flash_bench_${RUN_TS}.csv" results/part_c/flash_bench_latest.csv 2>/dev/null || true

echo "=== Part C done ==="
date -u
ls -lah results/part_c/
