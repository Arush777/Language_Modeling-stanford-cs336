# CS336 Assignment 2 — local notes (CCC + disk)

Handout: `cs336_assignment2_systems.pdf`  
Official repo: https://github.com/stanford-cs336/assignment2-systems  
Course: https://cs336.stanford.edu/

## Honor code

Stanford CS336 + this repo’s `AGENTS.md`: you implement FlashAttention / DDP / FSDP / sharded optimizer yourself. AI may help with concepts, CCC/disk, and reviewing *your* code — not writing solutions.

## Soft quota

- Soft **90 GiB**, hard **100 GiB**.
- Keep the **venv and caches off home** (see `scripts/env.sh`). Project tree on home should stay a few MB.

## One-time / per-shell setup

```bash
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
source scripts/env.sh
# If /tmp venv missing (new node or cleaned):
uv sync
uv run python -c "import cs336_basics, torch; print(torch.__version__, torch.cuda.is_available())"
```

On a **GPU interactive job**, re-`source scripts/env.sh` and re-run `uv sync` if that node’s `/tmp/$USER/cs336-a2-venv` does not exist yet (login-node `/tmp` ≠ compute-node `/tmp`).

## GPU jobs (never on login node)

Use the same pattern that worked for Jupyter (A100-80GB, `-n 8`, `span[hosts=1]`):

```bash
# 1× A100-80GB interactive (or: ./scripts/gpu_1.sh)
bsub -q interactive -Is \
  -J cs336-a2-a100 \
  -n 8 \
  -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
  -R "rusage[mem=64G,ngpus=1]" \
  -R "span[hosts=1]" \
  -W 6:00 \
  bash

# then on the compute node:
hostname
nvidia-smi -L
source scripts/env.sh && uv sync

# 2 GPUs — DDP / FSDP
bsub -q interactive -Is \
  -J cs336-a2-2gpu \
  -n 8 \
  -gpu "num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
  -R "rusage[mem=64G,ngpus=2]" \
  -R "span[hosts=1]" \
  -W 6:00 \
  bash
```

If stuck on `<<Waiting for dispatch ...>>`, check: `bjobs -p <jobid>` — usually means no free **exclusive** GPU yet (interactive queue often full). Ctrl-C does not always kill the job; use `bkill <jobid>`.

Idle GPU jobs can be killed (~2 h). Interactive queue ~6 h wall.

## Disk hygiene

- Delete large `*.nsys-rep` / profiler dumps after extracting writeup numbers.
- Do not `pip install torch` into miniconda for A2 — use this `uv` env only.
- One profile at a time; stay ~10 GiB under soft quota on home.

## Part A (implemented)

Code: `cs336_systems/benchmark.py`, `flops.py`, `nvtx_attention.py`, `mixed_precision_experiments.py`, `plot_part_a.py`.

**Submit one job (1× A100-80GB):**
```bash
cd /u/arushh/Arush/Language_Modeling-stanford-cs336/assignment2-systems
bsub < scripts/job_part_a.sh
```

Artifacts → `results/part_a/` (see `results/part_a/README.md`). Push figures + CSV under Arush777 when ready.

## What you implement later (adapters)

Wire these in `tests/adapters.py` to your code under `cs336_systems/`:

| Adapter | Area |
|---------|------|
| `get_flashattention_autograd_function_pytorch` | FA2 PyTorch reference |
| `get_flashattention_autograd_function_triton` | FA2 Triton |
| `get_ddp` / `ddp_on_after_backward` | Overlapping DDP |
| `get_fsdp` / `fsdp_on_after_backward` / `fsdp_gather_full_params` | FSDP |
| `get_sharded_optimizer` | Optimizer state sharding |

Tests: `uv run pytest -v ./tests` (on GPU node when CUDA/Triton/DDP needed).

## Problem checklist (handout)

See `writeup/PROBLEMS.md`. Fill answers in your own `writeup.pdf` (typeset).

## Submission

```bash
./test_and_make_submission.sh   # runs tests + builds zip
# Upload writeup.pdf + code zip to Gradescope if enrolled
```
