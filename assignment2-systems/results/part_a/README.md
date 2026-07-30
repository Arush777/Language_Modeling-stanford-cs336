# Part A outputs — where things land

All paths relative to `assignment2-systems/`.

## After `bsub < scripts/job_part_a.sh`

| Path | Contents | GitHub? |
|------|----------|---------|
| `results/part_a/timings/part_a_latest.csv` | All timing / FLOPs / peak-mem rows | **Yes** (small) |
| `results/part_a/timings/*.json` | Full run dumps | Yes |
| `results/part_a/figures/*.png` | Bar charts from CSV | **Yes — push these** |
| `results/part_a/memory/*.pickle` | CUDA memory snapshots | Optional (large); export PNGs from [pytorch.org/memory_viz](https://pytorch.org/memory_viz) → save PNG under `figures/` and push PNGs only |
| `results/part_a/nsys/*.nsys-rep` | Nsight traces | **Do not push** (multi‑GB). Open in Nsight Systems GUI locally; export screenshots → `results/part_a/figures/nsys_*.png` and push PNGs |
| `results/part_a/logs/` | Mixed-precision experiment text | Yes |
| `logs/cs336_a2_partA_*.out` | LSF stdout | Optional |

## FLOPs

Analytical GEMM FLOPs are in the CSV columns `analytical_flops` and `tflops_per_s` (FLOPs / mean wall time). Formula: `cs336_systems/flops.py` (2MN K for matmuls; SDPA QKᵀ + AV; SwiGLU 3 linears; × layers; LM head; backward ≈ 2× forward).

## Hardware

Job requests **1× A100-80GB** only. OOM on xl/10B is recorded in CSV (`ok=False`) and the sweep continues.

## Push (Arush777)

When timings/figures exist, commit under author Arush777 (ask the agent to commit/push). Prefer: CSV + `figures/*.png` + code. Exclude raw `.nsys-rep` / huge `.pickle` via `.gitignore`.
