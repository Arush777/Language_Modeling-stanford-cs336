#!/usr/bin/env bash
# Part E — optimizer state sharding (§6) + FSDP (§7): pytest gates (gloo) and
# both xl benches on 2× A100-80GB.
#
# Job steps (each echoed as it runs):
#   0) env: source scripts/env.sh (uv/torch caches + project venv on node-local
#      /tmp, off the home quota) and `uv sync` the lockfile.
#   1) pytest tests/test_sharded_optimizer.py + tests/test_fsdp.py, TWICE —
#      distributed tests can hang/flake nondeterministically, so the handout
#      recommends repeats; we keep going to the benches even if a run fails and
#      print a loud warning instead of burning the 6h slot.
#   2) sharded_optimizer_bench: xl, full vs sharded AdamW, one variant per
#      invocation (isolation: an OOM in "full" cannot take down "sharded").
#      Distinct master ports per invocation avoid a lingering rendezvous socket.
#   3) fsdp_bench: xl FSDP single row (B=2/T=256 defaults — the file's
#      docstring has the memory math for why that fits 80 GB).
#
# Submit: bsub < scripts/job_part_e.sh
#BSUB -J cs336-a2-partE
#BSUB -q normal
#BSUB -n 8
#BSUB -gpu num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB
#BSUB -R "rusage[mem=64G,ngpus=2]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -o logs/cs336_a2_partE_%J.out
#BSUB -e logs/cs336_a2_partE_%J.err

set -euo pipefail
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
mkdir -p logs results/part_e

# Visibility into which node/GPU the scheduler gave us (matches parts A–D logs).
hostname
date -u
nvidia-smi -L
source scripts/env.sh
uv sync

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

# 1) Correctness gates. The tests spawn 2 gloo processes. We force the CPU path
#    with CUDA_VISIBLE_DEVICES=: tests/common.py puts tensors on CUDA whenever a
#    device is visible, and gloo's worker thread then calls cudaSetDevice from
#    each spawned rank — under this job's `mode=exclusive_process` GPUs that
#    aborts with cudaErrorDevicesUnavailable (SIGABRT), exactly as it did for
#    Part D (results/part_d/pytest_ddp_run*.txt). Backend is gloo either way, so
#    the CPU path tests the same code and passes 6/6. The benches below still
#    use both GPUs via NCCL. Twice each, exit codes collected, benches run
#    regardless.
PYTEST_RCS=""
for run in 1 2; do
  echo "=== pytest tests/test_sharded_optimizer.py tests/test_fsdp.py (run ${run}) ==="
  set +e
  CUDA_VISIBLE_DEVICES= uv run pytest tests/test_sharded_optimizer.py tests/test_fsdp.py -v --tb=short \
    2>&1 | tee "results/part_e/pytest_part_e_run${run}_${RUN_TS}.txt"
  rc=${PIPESTATUS[0]}
  set -e
  PYTEST_RCS="${PYTEST_RCS} run${run}=${rc}"
done
echo "pytest exit codes:${PYTEST_RCS}"
if [[ "${PYTEST_RCS}" != *"=0"* ]]; then
  echo "!!! WARNING: every pytest run failed — check results/part_e/pytest_part_e_*.txt"
fi

# 2) §6 sharded-optimizer bench: one variant per invocation for OOM isolation
#    (xl full AdamW is ~51 GiB before activations; if it OOMs we still want the
#    sharded row). Ports 29551+ to avoid clashing with Part D's 2952x range.
port=29551
for variant in full sharded; do
  echo "=== sharded_optimizer_bench variant=${variant} (xl, 2 GPUs, NCCL) ==="
  uv run python -m cs336_systems.sharded_optimizer_bench \
    --variants "${variant}" \
    --world-size 2 \
    --master-port "${port}" \
    --out-csv "results/part_e/sharded_optimizer_bench_${variant}_${RUN_TS}.csv" || true
  port=$((port + 1))
done

# 3) §7 FSDP bench: single invocation (one row — the FSDP variant IS the
#    measurement; the DDP comparison rows already exist in results/part_d/).
echo "=== fsdp_bench (xl, 2 GPUs, NCCL) ==="
uv run python -m cs336_systems.fsdp_bench \
  --world-size 2 \
  --master-port "${port}" \
  --out-csv "results/part_e/fsdp_bench_${RUN_TS}.csv" || true

echo "=== Part E done ==="
echo "pytest exit codes:${PYTEST_RCS}"
date -u
ls -lah results/part_e/
