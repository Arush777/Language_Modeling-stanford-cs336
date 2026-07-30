# Assignment 2 Systems — Results index

Paths under `assignment2-systems/`.

| Part | Status | Where to look |
|------|--------|----------------|
| **A** Profiling | Done | `results/part_a/` |
| **B** Checkpointing | Done | `results/part_b/` |
| **C** Attention / Flash | Done (+ Triton d=128 patch merged) | **`results/part_c/flash_bench_merged_latest.csv`** |
| **D** DDP | Done (benches OK; pytest GPU harness flake documented) | `results/part_d/README.md` |
| **E** Sharded opt + FSDP | Done (job `1412621`) | **`results/part_e/README.md`** |
| **F** Written calcs | Done | **`writeup/PART_F.md`** |
| **G** Leaderboard | Skipped for now | Optional; needs 2×B200 |

## Push policy

Commit CSV + README + code under **Arush777**. Do not push large `.nsys-rep` / memory pickles.
