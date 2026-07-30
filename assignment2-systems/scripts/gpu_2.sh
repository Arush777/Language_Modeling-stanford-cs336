#!/usr/bin/env bash
# Interactive 2×A100-80GB on one host (DDP / FSDP).
exec bsub -q interactive -Is \
  -J cs336-a2-2gpu \
  -n 8 \
  -gpu "num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
  -R "rusage[mem=64G,ngpus=2]" \
  -R "span[hosts=1]" \
  -W 6:00 \
  bash
