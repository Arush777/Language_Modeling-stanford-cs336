"""
Part E §7 — FSDP on the xl model, 2 GPUs: peak memory at three checkpoints + time/step.

Question being measured (handout §7): what does fully-sharded data parallel
training cost/benefit on the xl model in practice — peak memory at init,
before the optimizer step, after the step, and the end-to-end step time?

Setup: BasicsTransformerLM with MODEL_CONFIGS["xl"] (d_model=2560, d_ff=10240,
num_layers=32, num_heads=32), vocab_size=10000, fp32 masters, 2 GPUs, NCCL.
Unlike sharded_optimizer_bench.py, ranks use DIFFERENT data seeds — this is a
real DP run: the reduce-scatter does genuine gradient averaging every step.

  xl fp32 memory math under THIS FSDP implementation (weights ≈ 3.4B params ≈ 12.7 GiB):
    persistent shards (weights)      : 12.7 / 2  ≈  6.4 GiB per rank
    sharded gradients                :           ≈  6.4 GiB per rank
    AdamW moments (on shards, free)  : 2 × 6.4   ≈ 12.7 GiB per rank
    TRANSIENT full weights alive     : up to 12.7 GiB — see below
    activations at B=2, T=256        : a few GiB (same scale as ddp_bench)
    → peak ≈ 40 GiB, comfortably under A100-80GB.

  WHY the defaults B=2 / T=256: the same budget reasoning as ddp_bench.py —
  big enough per-step compute that comm (all-gather / reduce-scatter inside
  fwd/bwd) is a visible but not dominant cost, small enough that even the
  ~12.7 GiB transient below cannot OOM an 80 GB card. --batch-size and
  --context-length are argparse knobs to probe the edge.

  HONEST ACCOUNTING of the transient (fsdp.py module docstring, deviation #2):
  our teaching implementation keeps every layer's gathered full weight alive
  from its forward until its backward (autograd saves it for the einsum), so
  during backward the FULL model's weights are transiently resident on top of
  the shards. The handout's free-and-re-gather ideal would cut that ~12.7 GiB
  at the price of one extra all-gather per layer per step — this bench's
  peak_before_step_gib column is exactly the quantity that would shrink.

Timing phases reported per step (means over steps, aggregated over ranks):
  fwd_ms  — includes the per-layer weight ALL-GATHERS: unlike DDP, FSDP comm
            is INSIDE the forward pass, so fwd_ms is larger than a plain
            replica's forward. No separate comm column can capture it.
  bwd_ms  — includes the per-layer gradient REDUCE-SCATTERS (fired by the
            autograd Function's backward as each layer finishes).
  sync_ms — finish_gradient_synchronization(): only the tiny replicated
            RMSNorm-gradient all-reduces remain here (sharded grads already
            synced during backward), so this should be ~0. It exists to make
            the previous two bullets measurable by contrast.
  opt_ms  — AdamW on shards: less work than full AdamW (1/world_size params).
  step_ms — end-to-end; THE honest comparison metric against Part D's DDP rows.

Usage:
  uv run python -m cs336_systems.fsdp_bench --out-csv results/part_e/fsdp_bench.csv
  # CPU-only smoke test of the harness (gloo, tiny model — NOT a benchmark):
  uv run python -m cs336_systems.fsdp_bench --backend gloo --model small \
      --steps 2 --warmup 1 --out-csv /tmp/fsdp.csv
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
from cs336_systems.fsdp import FSDP


def _gib(num_bytes: float) -> float:
    return num_bytes / 2**30


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_fsdp(rank: int, world_size: int, args: argparse.Namespace, device: torch.device) -> dict:
    """Build an FSDP-wrapped xl replica and measure the three checkpoints + steady-state step time."""
    cfg = MODEL_CONFIGS[args.model]

    # Same model-init seed on every rank. (FSDP also broadcasts rank 0's
    # weights at wrap time — fsdp.py — so this seed is belt-and-braces; the
    # broadcast is what actually guarantees a consistent global model.)
    torch.manual_seed(args.seed)

    # Checkpoint 1 — INIT: peak while building the model and WRAPPING it in
    # FSDP. Wrapping frees the full Linear/Embedding weights and keeps only
    # this rank's shards, so the interesting number is how the post-wrap
    # resident memory compares to a plain replica (ddp_bench: ~12.7 GiB).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base_model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=DEFAULT_ROPE_THETA,
    ).to(device)
    # Count params BEFORE wrapping: after FSDP sharding, base_model.parameters()
    # yields only this rank's shard tensors (~1/world_size of the full count).
    num_params = sum(p.numel() for p in base_model.parameters())
    fsdp_model = FSDP(base_model)
    # AdamW over fsdp_model.parameters(): the optimizer only ever sees the
    # shard Parameters (+ tiny replicated norms), so its 2-moment state is
    # sharded across ranks FOR FREE — no ShardedOptimizer wrapper needed here.
    optimizer = AdamW(fsdp_model.parameters(), lr=args.lr)
    init_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    # DIFFERENT data per rank (seed includes rank): gradients genuinely differ,
    # so the reduce-scatter averages real information every step — a real DP
    # run, not a communication no-op.
    gen = torch.Generator(device="cpu").manual_seed(args.data_seed + rank)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)

    def one_step() -> tuple[float, float, float, float, float]:
        """Returns (fwd_s, bwd_s, sync_s, opt_s, step_s). All-gathers live in
        fwd_s; reduce-scatters in bwd_s; sync_s is only the tiny norm grads."""
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        t0 = timeit.default_timer()
        logits = fsdp_model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        _sync(device)
        t1 = timeit.default_timer()
        loss.backward()
        _sync(device)
        t2 = timeit.default_timer()
        fsdp_model.finish_gradient_synchronization()
        _sync(device)
        t3 = timeit.default_timer()
        optimizer.step()
        _sync(device)
        t4 = timeit.default_timer()
        return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)

    ok, error = True, None
    phase_times: list[tuple[float, float, float, float, float]] = []
    before_step_peak = after_step_peak = 0
    try:
        # Warmup: first step allocates AdamW's lazy (sharded) moments; the
        # checkpoints below are taken at steady state, not during the one-time
        # allocation spike.
        for _ in range(args.warmup):
            one_step()

        # Checkpoint 2 — BEFORE STEP: peak over one fwd+bwd. This is where the
        # ~12.7 GiB of gathered full weights is transiently resident (module
        # docstring), on top of shards + sharded grads.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        logits = fsdp_model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        _sync(device)
        before_step_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        # Checkpoint 3 — AFTER STEP: norm-grad sync + optimizer step; moments
        # (sharded) now resident. Compare against the FULL AdamW row of
        # sharded_optimizer_bench.py: 2×params vs 2×params/world_size.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fsdp_model.finish_gradient_synchronization()
        optimizer.step()
        _sync(device)
        after_step_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        for _ in range(args.steps):
            phase_times.append(one_step())
    except torch.cuda.OutOfMemoryError as e:
        ok, error = False, f"OOM: {e}"
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate across ranks (same pattern as ddp_bench.py): cross-rank mean of
    # per-rank phase means, worst per-rank step std, worst-rank peaks.
    n_phases = 5
    if ok and phase_times:
        per_phase_mean = [statistics.mean(t[i] for t in phase_times) for i in range(n_phases)]
        step_std = statistics.stdev(t[4] for t in phase_times) if len(phase_times) > 1 else 0.0
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
    # Peak memory: report the MAX across ranks — a job is OOM-bound by its
    # worst rank, not its average rank.
    max_peaks = [max(g[i] for g in gathered_peaks if g is not None) for i in range(3)]

    row = {
        "variant": "fsdp",
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
        "sync_ms": mean_phases[2] * 1e3,
        "opt_ms": mean_phases[3] * 1e3,
        "step_ms": mean_phases[4] * 1e3,
        "step_ms_max_rank_std": max(s for s in gathered_stds if s is not None) * 1e3,
        "num_params": num_params,
        "ok": all_ok,
        "error": None if all_ok else (error or "OOM/failure on another rank"),
    }
    del fsdp_model, base_model, optimizer, x, y
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

    row = _bench_fsdp(rank, world_size, args, device)
    if rank == 0:
        if row["ok"]:
            print(
                f"  peak GiB init={row['peak_init_gib']:6.2f} before_step={row['peak_before_step_gib']:6.2f} "
                f"after_step={row['peak_after_step_gib']:6.2f} | step={row['step_ms']:9.2f}ms "
                f"fwd={row['fwd_ms']:8.2f} bwd={row['bwd_ms']:8.2f} sync={row['sync_ms']:6.2f} opt={row['opt_ms']:8.2f}",
                flush=True,
            )
        else:
            print(f"  FAILED: {row['error']}", flush=True)
        if args.out_csv:
            out_path = Path(args.out_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerows([row])
            print(f"wrote {out_path}", flush=True)
    dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser(description="CS336 A2 Part E §7 — FSDP xl memory/time benchmark (2 GPUs)")
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
    p.add_argument("--data-seed", type=int, default=1234, help="base seed; rank's shard uses seed+rank (real DP)")
    p.add_argument("--master-addr", type=str, default="localhost")
    p.add_argument("--master-port", type=int, default=29541)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        print(f"need {args.world_size} CUDA devices, have {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        return 1

    print(
        f"fsdp_bench: backend={args.backend} world_size={args.world_size} model={args.model} "
        f"B={args.batch_size}/rank T={args.context_length} warmup={args.warmup} steps={args.steps}",
        flush=True,
    )
    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
