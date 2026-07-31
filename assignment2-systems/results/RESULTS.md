# Assignment 2 Systems — Results index

**How to read this folder**

| Kind | What it is | Push to GitHub? |
|------|------------|-----------------|
| `*.csv` | Raw measurements (reproducible tables) | Yes (small) |
| `figures/*.png` | **Plots for humans / writeup / README** | **Yes — this is the presentation** |
| `*.txt` | pytest logs | Optional |
| `*.nsys-rep` / `*.pickle` | Huge profiler dumps | **No** (gitignore); export screenshots if needed |

Regenerate all B–E plots from CSVs:

```bash
cd assignment2-systems
uv run python -m cs336_systems.plot_results
```

| Part | Status | Open this |
|------|--------|-----------|
| **A** Profiling | Done | [`part_a/figures/`](part_a/figures/) + [`part_a/README.md`](part_a/README.md) |
| **B** Checkpointing | Done | [`part_b/README.md`](part_b/README.md) (embeds figure) |
| **C** Attention / Flash | Done | [`part_c/README.md`](part_c/README.md) |
| **D** DDP | Done | [`part_d/README.md`](part_d/README.md) |
| **E** Sharded opt + FSDP | Done | [`part_e/README.md`](part_e/README.md) |
| **F** Written calcs | Done | [`../writeup/PART_F.md`](../writeup/PART_F.md) (equations, not plots) |
| **G** Leaderboard | Skipped | — |
