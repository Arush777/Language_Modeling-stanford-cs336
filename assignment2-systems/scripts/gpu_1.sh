#!/usr/bin/env bash
# Interactive 1×A100-80GB (matches the bsub that worked for jupyter).
# Usage: ./scripts/gpu_1.sh   OR   source this after killing any old pending job.
exec bsub -q interactive -Is \
  -J cs336-a2-a100 \
  -n 8 \
  -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
  -R "rusage[mem=64G,ngpus=1]" \
  -R "span[hosts=1]" \
  -W 6:00 \
  bash
