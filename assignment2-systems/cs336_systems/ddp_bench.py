"""
Part D §5.2–5.3 — benchmark the xl language model under naive / flat / overlap DDP.

Setup (handout: 1 node × 2 GPUs, NCCL, xl from §2.1.2):
  model          BasicsTransformerLM with MODEL_CONFIGS["xl"] (d_model=2560, d_ff=10240,
                 num_layers=32, num_heads=32), vocab_size=10000, fp32.
  context_length 256 and per-rank batch 2 by default. Rationale: xl is ~3.4B params,
                 i.e. ~12.7 GiB of fp32 weights, plus the same again for gradients and
                 2× that for AdamW moments. The flat variant additionally materializes
                 a full-size gradient copy (+12.7 GiB), so large batches risk OOM on
                 A100-80GB; B=2/T=256 leaves comfortable headroom while keeping
                 per-step compute large enough that communication hiding is visible.
                 Both are argparse knobs if you want to probe the edge.

What we measure per training step (means over steps, aggregated over ranks):
  fwd_ms / bwd_ms / comm_ms / opt_ms / step_ms.
  - For NaiveDDP and FlatDDP, ALL gradient communication happens inside
    `finish_gradient_synchronization()`, so comm_ms is exactly the handout's
    "time spent communicating gradients".
  - For the overlapping DDP, communication is issued DURING backward. Our
    device-wide torch.cuda.synchronize() after backward() absorbs any
    collectives still in flight into bwd_ms, so comm_ms is only the exposed
    tail. The honest comparison metric is therefore step_ms end-to-end.

Variants are benchmarked sequentially inside one process group; each catches
OOM and reports it as a row with ok=False instead of killing the sweep (if one
rank fails inside a collective the NCCL timeout — 5 min here, not the default
30 — bounds the damage). The LSF job additionally runs each variant as its own
invocation for extra isolation.

Usage:
  uv run python -m cs336_systems.ddp_bench --variants naive,flat,overlap --out-csv results/part_d/ddp_bench.csv
  # CPU-only smoke test of the harness (gloo, tiny model — NOT a benchmark):
  uv run python -m cs336_systems.ddp_bench --backend gloo --model small --variants naive,flat,overlap --steps 2 --warmup 1 --out-csv /tmp/ddp.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import statistics
import timeit
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.configs import (
    DEFAULT_ROPE_THETA,
    DEFAULT_VOCAB_SIZE,
    MODEL_CONFIGS,
)
from cs336_systems.ddp import DDP, FlatDDP, NaiveDDP, _unique_trainable_parameters

# The overlapping DDP is the class the tests use; naive/flat are the §5.2/§5.3.1 baselines.
VARIANTS = {"naive": NaiveDDP, "flat": FlatDDP, "overlap": DDP}


def _build_model(model_name: str, vocab_size: int, context_length: int, device: torch.device) -> BasicsTransformerLM:
    cfg = MODEL_CONFIGS[model_name]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=DEFAULT_ROPE_THETA,
    )
    return model.to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_variant(
    variant: str,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Build a fresh replica, train `warmup + steps` steps under one DDP variant, return a result row."""
    cfg = MODEL_CONFIGS[args.model]
    # Same seed on every rank: replicas start identical even before the init broadcast.
    torch.manual_seed(args.seed)
    base_model = _build_model(args.model, args.vocab_size, args.context_length, device)
    ddp_model = VARIANTS[variant](base_model)
    # AdamW matches the Part A/B benchmark setup; grads are averaged across ranks,
    # so every rank applies the identical update and replicas stay in lockstep.
    optimizer = AdamW(ddp_model.parameters(), lr=args.lr)

    # Per-rank data shard: different seed per rank so gradients genuinely differ
    # across ranks (identical shards would benchmark a communication no-op).
    gen = torch.Generator(device="cpu").manual_seed(args.data_seed + rank)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), generator=gen).to(device)

    def one_step() -> tuple[float, float, float, float, float]:
        """Returns (fwd_s, bwd_s, comm_s, opt_s, step_s). Per-phase cuda syncs cost a
        little overlap realism but give clean phase attribution; step_s is the metric
        to compare across variants."""
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        t0 = timeit.default_timer()
        with nvtx.range("ddp_fwd") if args.nvtx else nullcontext():
            logits = ddp_model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        _sync(device)
        t1 = timeit.default_timer()
        with nvtx.range("ddp_bwd") if args.nvtx else nullcontext():
            loss.backward()
        _sync(device)
        t2 = timeit.default_timer()
        with nvtx.range("ddp_comm") if args.nvtx else nullcontext():
            ddp_model.finish_gradient_synchronization()
        _sync(device)
        t3 = timeit.default_timer()
        with nvtx.range("ddp_opt") if args.nvtx else nullcontext():
            optimizer.step()
        _sync(device)
        t4 = timeit.default_timer()
        return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)

    ok, error = True, None
    phase_times: list[tuple[float, float, float, float, float]] = []
    try:
        for _ in range(args.warmup):
            one_step()
        for _ in range(args.steps):
            phase_times.append(one_step())
    except torch.cuda.OutOfMemoryError as e:
        ok, error = False, f"OOM: {e}"
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate across ranks (handout §5.1.1): report cross-rank mean of per-rank
    # phase means, and the worst per-rank step-time std.
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
    dist.all_gather_object(gathered_ok, ok)
    dist.all_gather_object(gathered_means, per_phase_mean)
    dist.all_gather_object(gathered_stds, step_std)
    all_ok = all(gathered_ok)
    mean_phases = [statistics.mean(g[i] for g in gathered_means if g is not None) for i in range(n_phases)]

    # Context numbers for the writeup: how much data moves every step.
    trainable = _unique_trainable_parameters(base_model)
    grad_bytes = sum(p.numel() * p.element_size() for p in trainable)
    num_params = sum(p.numel() for p in base_model.parameters())

    row = {
        "variant": variant,
        "model": args.model,
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "warmup": args.warmup,
        "steps": args.steps,
        "fwd_ms": mean_phases[0] * 1e3,
        "bwd_ms": mean_phases[1] * 1e3,
        "comm_ms": mean_phases[2] * 1e3,
        "opt_ms": mean_phases[3] * 1e3,
        "step_ms": mean_phases[4] * 1e3,
        "step_ms_max_rank_std": max(s for s in gathered_stds if s is not None) * 1e3,
        "num_params": num_params,
        "grad_bytes_per_step": grad_bytes,
        "ok": all_ok,
        "error": None if all_ok else (error or "OOM/failure on another rank"),
    }
    # Free the replica before the next variant (xl weights+grads+AdamW ≈ 38 GiB).
    del ddp_model, base_model, optimizer, x, y
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
                    f"  step={row['step_ms']:9.2f}ms  fwd={row['fwd_ms']:8.2f}  bwd={row['bwd_ms']:8.2f}  "
                    f"comm={row['comm_ms']:8.2f}  opt={row['opt_ms']:8.2f}  (grad {row['grad_bytes_per_step'] / 2**20:.0f} MiB/step)",
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
    p = argparse.ArgumentParser(description="CS336 A2 Part D — naive / flat / overlap DDP benchmark")
    p.add_argument("--variants", type=str, default="naive,flat,overlap", help="comma list of: naive,flat,overlap")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="xl")
    p.add_argument("--batch-size", type=int, default=2, help="per-rank batch (see module docstring for the memory math)")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0, help="model init seed (same on all ranks)")
    p.add_argument("--data-seed", type=int, default=1234, help="base seed for per-rank data shards")
    p.add_argument("--master-addr", type=str, default="localhost")
    p.add_argument("--master-port", type=int, default=29511)
    p.add_argument("--nvtx", action="store_true", help="NVTX ranges around fwd/bwd/comm/opt (for nsys traces)")
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    unknown = set(args.variants.split(",")) - VARIANTS.keys()
    if unknown:
        print(f"unknown variants: {unknown} (choose from {sorted(VARIANTS)})")
        return 1
    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        print(f"need {args.world_size} CUDA devices, have {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        return 1

    print(
        f"ddp_bench: variants={args.variants} backend={args.backend} world_size={args.world_size} "
        f"model={args.model} B={args.batch_size}/rank T={args.context_length} warmup={args.warmup} steps={args.steps}",
        flush=True,
    )
    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
