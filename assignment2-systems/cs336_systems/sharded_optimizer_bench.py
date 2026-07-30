"""
Part E §6 — sharded-vs-full optimizer: peak memory at three checkpoints + time/step.

Question being measured (handout §6): how much memory does optimizer state
sharding actually save on the xl model, and what does the post-step parameter
broadcast cost in time?

Setup: BasicsTransformerLM with MODEL_CONFIGS["xl"] (d_model=2560, d_ff=10240,
num_layers=32, num_heads=32), vocab_size=10000, fp32, 2 GPUs, NCCL.

  xl fp32 memory math (weights ≈ 3.4B params ≈ 12.7 GiB):
    FULL AdamW   : weights 12.7 + grads 12.7 + moments 2×12.7  ≈ 50.8 GiB + activations
    SHARDED AdamW: weights 12.7 + grads 12.7 + moments 1×12.7  ≈ 38.1 GiB + activations
                   (grads stay full — §6 shards only optimizer STATE; Part §7/FSDP
                   is what shards grads+weights too)
  Both fit an A100-80GB at the default B=2/T=256 (same budget reasoning as
  ddp_bench.py); --batch-size/--context-length are knobs to probe the edge.

The three memory checkpoints (measured with torch.cuda.reset_peak_memory_stats
+ max_memory_allocated, so each is the PEAK during that phase, transients
included):
  1. init        — building the model + constructing the optimizer. AdamW
                   moments are LAZY (allocated at the first step()), so this
                   checkpoint is ~weights-only for BOTH variants. The sharded
                   win is therefore NOT visible here — it appears at (3).
  2. before_step — one fwd+bwd AFTER warmup: weights + grads resident, moments
                   already materialized. Both variants should look alike here
                   (sharding does not shrink weights or grads).
  3. after_step  — optimizer.step() (for the sharded variant this INCLUDES the
                   parameter broadcasts — see below): full variant holds
                   2×params of moments, sharded holds 2×params/world_size.

Deliberate simplification, commented for honesty: we do NOT wrap the model in
DDP, and every rank uses the SAME data seed, so gradients are identical across
ranks. Rationale: §6 isolates OPTIMIZER memory, and with identical grads the
sharded optimizer's post-step broadcast is a numerically exact no-op — the
measured memory profile equals a real DP run, while the broadcast still pays
its REAL communication time (that time is the honest price of the design and
shows up in opt_ms / step_ms). What this bench deliberately omits vs a real
run is DDP's gradient all-reduce time, which Part D already measured.

Usage:
  uv run python -m cs336_systems.sharded_optimizer_bench \
      --variants full,sharded --out-csv results/part_e/sharded_optimizer_bench.csv
  # CPU-only smoke test of the harness (gloo, tiny model — NOT a benchmark):
  uv run python -m cs336_systems.sharded_optimizer_bench --backend gloo --model small \
      --variants full,sharded --steps 2 --warmup 1 --out-csv /tmp/shopt.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import statistics
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.configs import DEFAULT_ROPE_THETA, DEFAULT_VOCAB_SIZE, MODEL_CONFIGS
from cs336_systems.sharded_optimizer import ShardedOptimizer

# "full": plain AdamW over all params (the Part D memory status quo).
# "sharded": ShardedOptimizer wrapping the SAME AdamW — moments live only for
# this rank's ~1/world_size shard; step() broadcasts updated params.
VARIANTS = ("full", "sharded")


def _gib(num_bytes: float) -> float:
    return num_bytes / 2**30


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_variant(variant: str, rank: int, world_size: int, args: argparse.Namespace, device: torch.device) -> dict:
    """Build a fresh xl replica + one of the two optimizer variants; measure the three checkpoints."""
    cfg = MODEL_CONFIGS[args.model]

    # Same seed on every rank AND same data seed per rank below: replicas and
    # gradients are identical across ranks by construction (module docstring).
    torch.manual_seed(args.seed)

    # Checkpoint 1 — INIT: peak while constructing model + optimizer. Moments
    # are lazy, so this is ~weights-only for both variants (docstring).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=DEFAULT_ROPE_THETA,
    ).to(device)
    if variant == "full":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=args.lr)
    init_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    # Identical data on every rank (see module docstring for why this is the
    # honest simplification for an optimizer-MEMORY benchmark).
    gen = torch.Generator(device="cpu").manual_seed(args.data_seed)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)

    def one_step() -> tuple[float, float, float, float]:
        """Returns (fwd_s, bwd_s, opt_s, step_s). For the sharded variant the
        post-step parameter broadcasts are inside optimizer.step(), so they are
        timed as part of opt_s — that placement is the honest accounting."""
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        t0 = timeit.default_timer()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        _sync(device)
        t1 = timeit.default_timer()
        loss.backward()
        _sync(device)
        t2 = timeit.default_timer()
        optimizer.step()
        _sync(device)
        t3 = timeit.default_timer()
        return (t1 - t0, t2 - t1, t3 - t2, t3 - t0)

    ok, error = True, None
    phase_times: list[tuple[float, float, float, float]] = []
    before_step_peak = after_step_peak = 0
    try:
        # Warmup first: the first step allocates AdamW's lazy moments, so the
        # two memory checkpoints below are taken on a step where moments
        # already exist — otherwise "after_step" would measure a one-time
        # allocation spike instead of steady state.
        for _ in range(args.warmup):
            one_step()

        # Checkpoint 2 — BEFORE STEP: peak over one fwd+bwd (weights+grads
        # resident; sharding does not shrink either, so variants look alike).
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        _sync(device)
        before_step_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        # Checkpoint 3 — AFTER STEP: peak over the optimizer step. Full AdamW
        # materializes 2×params of moments here; the sharded variant holds
        # 2×params/world_size plus its broadcast transient.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        _sync(device)
        after_step_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        # Steady-state timing.
        for _ in range(args.steps):
            phase_times.append(one_step())
    except torch.cuda.OutOfMemoryError as e:
        ok, error = False, f"OOM: {e}"
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate across ranks (same pattern as ddp_bench): cross-rank mean of
    # per-rank phase means, worst per-rank step std, and all-ranks-ok.
    n_phases = 4
    if ok and phase_times:
        per_phase_mean = [statistics.mean(t[i] for t in phase_times) for i in range(n_phases)]
        step_std = statistics.stdev(t[3] for t in phase_times) if len(phase_times) > 1 else 0.0
    else:
        per_phase_mean = [0.0] * n_phases
        step_std = 0.0
    gathered_ok: list[bool | None] = [None] * world_size
    gathered_means: list[list[float] | None] = [None] * world_size
    gathered_stds: list[float | None] = [None] * world_size
    gathered_peaks: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered_ok, ok)
    dist.all_gather_object(gathered_means, per_phase_mean)
    dist.all_gather_object(gathered_stds, step_std)
    dist.all_gather_object(gathered_peaks, [init_peak, before_step_peak, after_step_peak])
    all_ok = all(gathered_ok)
    mean_phases = [statistics.mean(g[i] for g in gathered_means if g is not None) for i in range(n_phases)]
    # Peak memory: report the MAX across ranks (a job is OOM-bound by its
    # worst rank, not its average rank).
    max_peaks = [max(g[i] for g in gathered_peaks if g is not None) for i in range(3)]

    num_params = sum(p.numel() for p in model.parameters())
    row = {
        "variant": variant,
        "model": args.model,
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "warmup": args.warmup,
        "steps": args.steps,
        "peak_init_gib": _gib(max_peaks[0]),
        "peak_before_step_gib": _gib(max_peaks[1]),
        "peak_after_step_gib": _gib(max_peaks[2]),
        "fwd_ms": mean_phases[0] * 1e3,
        "bwd_ms": mean_phases[1] * 1e3,
        "opt_ms": mean_phases[2] * 1e3,
        "step_ms": mean_phases[3] * 1e3,
        "step_ms_max_rank_std": max(s for s in gathered_stds if s is not None) * 1e3,
        "num_params": num_params,
        "ok": all_ok,
        "error": None if all_ok else (error or "OOM/failure on another rank"),
    }
    # Free the replica before the next variant (xl weights+grads+moments ≈ 51 GiB full / 38 GiB sharded).
    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    if args.backend == "nccl":
        torch.cuda.set_device(rank)  # one process per GPU
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")  # gloo = CPU debug path only
    # Short-ish timeout so a wedged collective fails the job in minutes, not at the wall limit.
    dist.init_process_group(args.backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))

    rows: list[dict] = []
    for variant in args.variants.split(","):
        variant = variant.strip()
        if rank == 0:
            print(f"=== variant={variant} ===", flush=True)
        row = _bench_variant(variant, rank, world_size, args, device)
        if rank == 0:
            rows.append(row)
            if row["ok"]:
                print(
                    f"  peak GiB init={row['peak_init_gib']:6.2f} before_step={row['peak_before_step_gib']:6.2f} "
                    f"after_step={row['peak_after_step_gib']:6.2f} | step={row['step_ms']:9.2f}ms "
                    f"fwd={row['fwd_ms']:8.2f} bwd={row['bwd_ms']:8.2f} opt={row['opt_ms']:8.2f}",
                    flush=True,
                )
            else:
                print(f"  FAILED: {row['error']}", flush=True)
        dist.barrier()  # keep ranks in lockstep between variants

    if rank == 0 and args.out_csv and rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}", flush=True)
    dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser(description="CS336 A2 Part E §6 — full vs sharded optimizer memory/time benchmark")
    p.add_argument("--variants", type=str, default="full,sharded", help="comma list of: full,sharded")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="xl")
    p.add_argument("--batch-size", type=int, default=2, help="per-rank batch (module docstring has the memory math)")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0, help="model init seed (same on all ranks)")
    p.add_argument("--data-seed", type=int, default=1234, help="SAME on all ranks by design — see module docstring")
    p.add_argument("--master-addr", type=str, default="localhost")
    p.add_argument("--master-port", type=int, default=29531)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    unknown = set(args.variants.split(",")) - set(VARIANTS)
    if unknown:
        print(f"unknown variants: {unknown} (choose from {list(VARIANTS)})")
        return 1
    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        print(f"need {args.world_size} CUDA devices, have {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        return 1

    print(
        f"sharded_optimizer_bench: variants={args.variants} backend={args.backend} world_size={args.world_size} "
        f"model={args.model} B={args.batch_size}/rank T={args.context_length} warmup={args.warmup} steps={args.steps}",
        flush=True,
    )
    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
