#!/usr/bin/env bash
# Part C patch — flash_triton ONLY at d=128 (shared-mem tile fix)
# Submit AFTER main Part C finishes, or in parallel on another GPU:
#   bsub < scripts/job_part_c_triton_d128.sh
#BSUB -J cs336-a2-partC-triton-d128
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=1]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o logs/cs336_a2_partC_triton_d128_%J.out
#BSUB -e logs/cs336_a2_partC_triton_d128_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs results/part_c

hostname
date -u
nvidia-smi -L
source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

echo "=== pytest flash (triton) sanity ==="
uv run pytest -k "test_flash" -v --tb=short | tee "results/part_c/pytest_flash_triton_d128_${RUN_TS}.txt"

echo "=== flash_triton only d=128 (fp32+bf16) ==="
uv run python -m cs336_systems.flash_bench \
  --out-csv "results/part_c/flash_bench_triton_d128_${RUN_TS}.csv" \
  --impls "flash_triton" \
  --d-sizes "128" \
  --seq-sizes "128,256,512,1024,2048,4096,8192" \
  --dtypes "float32,bfloat16"

cp -f "results/part_c/flash_bench_triton_d128_${RUN_TS}.csv" \
  results/part_c/flash_bench_triton_d128_latest.csv

echo "=== triton d=128 patch done ==="
date -u
cat "results/part_c/flash_bench_triton_d128_${RUN_TS}.csv"
