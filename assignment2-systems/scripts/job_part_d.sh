#!/usr/bin/env bash
# Part D — DDP: pytest (gloo), all-reduce microbench, xl naive/flat/overlap bench on 2× A100-80GB
# Submit: bsub < scripts/job_part_d.sh
#BSUB -J cs336-a2-partD
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=2]"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -o logs/cs336_a2_partD_%J.out
#BSUB -e logs/cs336_a2_partD_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs results/part_d

hostname
date -u
nvidia-smi -L
source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

# 1) Correctness gate: DDP tests use the gloo backend and also pass on CPU, but we
#    run them on the GPU node. The handout recommends several repeats to shake out
#    flaky hangs — we run twice and keep going to the benches even if a run fails,
#    printing a loud warning instead of burning the 4h slot.
PYTEST_RCS=""
for run in 1 2; do
  echo "=== pytest tests/test_ddp.py (run ${run}) ==="
  set +e
  uv run pytest tests/test_ddp.py -v --tb=short 2>&1 | tee "results/part_d/pytest_ddp_run${run}_${RUN_TS}.txt"
  rc=${PIPESTATUS[0]}
  set -e
  PYTEST_RCS="${PYTEST_RCS} run${run}=${rc}"
done
echo "pytest exit codes:${PYTEST_RCS}"
if [[ "${PYTEST_RCS}" != *"=0"* ]]; then
  echo "!!! WARNING: every pytest run failed — check results/part_d/pytest_ddp_*.txt"
fi

# 2) All-reduce microbench: float32 1MB/10MB/100MB/1GB on 2 GPUs (NCCL).
#    The handout's 4- and 6-GPU points need a larger job (e.g. -gpu num=4 ...);
#    rerun there with:  --world-size 4 --master-port 29512
echo "=== all_reduce_bench (world_size=2, NCCL) ==="
uv run python -m cs336_systems.all_reduce_bench \
  --world-size 2 \
  --out-csv "results/part_d/all_reduce_bench_${RUN_TS}.csv" || true

# 3) xl LM training-step benchmark, one DDP variant per invocation so a failure
#    (e.g. flat-variant OOM) cannot take down the other measurements. Distinct
#    master ports avoid any lingering rendezvous socket between invocations.
port=29521
for variant in naive flat overlap; do
  echo "=== ddp_bench variant=${variant} (xl, 2 GPUs, NCCL) ==="
  uv run python -m cs336_systems.ddp_bench \
    --variants "${variant}" \
    --world-size 2 \
    --master-port "${port}" \
    --out-csv "results/part_d/ddp_bench_${variant}_${RUN_TS}.csv" || true
  port=$((port + 1))
done

# Optional (handout ddp_overlap (b)): nsys traces comparing naive vs overlap DDP.
# Uncomment to capture (adds a few minutes; delete the .nsys-rep files after
# extracting the screenshots to respect the home quota — see NOTES.md):
#   nsys profile -o "results/part_d/nsys_naive_${RUN_TS}" --force-overwrite true \
#     uv run python -m cs336_systems.ddp_bench --variants naive --world-size 2 \
#       --steps 5 --warmup 2 --nvtx --out-csv /tmp/ddp_nsys_naive.csv || true
#   nsys profile -o "results/part_d/nsys_overlap_${RUN_TS}" --force-overwrite true \
#     uv run python -m cs336_systems.ddp_bench --variants overlap --world-size 2 \
#       --steps 5 --warmup 2 --nvtx --out-csv /tmp/ddp_nsys_overlap.csv || true

echo "=== Part D done ==="
echo "pytest exit codes:${PYTEST_RCS}"
date -u
ls -lah results/part_d/
