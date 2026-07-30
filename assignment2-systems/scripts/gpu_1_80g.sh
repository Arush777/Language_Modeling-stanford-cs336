#!/usr/bin/env bash
# Same as gpu_1.sh — A100-80GB interactive.
exec bsub -q interactive -Is \
  -J cs336-a2-a100-80g \
  -n 8 \
  -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
  -R "rusage[mem=64G,ngpus=1]" \
  -R "span[hosts=1]" \
  -W 6:00 \
  bash
