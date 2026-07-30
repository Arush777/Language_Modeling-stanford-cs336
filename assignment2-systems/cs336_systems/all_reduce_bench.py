"""
Part D §5.1 — all-reduce microbenchmark (single node, one process per GPU).

First principles
----------------
All-reduce replaces each rank's tensor with the SUM over all ranks' tensors.
It is the workhorse collective behind DDP gradient averaging, so before
benchmarking full DDP training we measure the collective itself: how does the
wall time of one all-reduce scale with tensor size (1MB → 1GB, float32) and
with world size (2 GPUs in the default job; 4/6 need a larger LSF job — see
scripts/job_part_d.sh)?

Two gotchas this script is careful about (handout §5.1.1):

- NCCL is asynchronous at the CUDA level: `dist.all_reduce(t, async_op=False)`
  returns once the communication kernel is QUEUED, not finished. We therefore
  time a loop of `iters` collectives and call `torch.cuda.synchronize()` once
  at the end — the same methodology NCCL's own tests use — after a warmup that
  lets NCCL set up its internal buffers/channels.
- Ranks don't finish at exactly the same time, so each rank measures its own
  per-op time and we aggregate across ranks with `dist.all_gather_object`
  (report mean and max).

Bandwidths: "alg_bw" = size / time (bytes each rank moves per second);
"bus_bw" = alg_bw × 2·(world_size−1)/world_size, the ring all-reduce
correction that estimates the actual link utilization and is the number
comparable across world sizes.

Usage:
  uv run python -m cs336_systems.all_reduce_bench --world-size 2 --out-csv results/part_d/all_reduce.csv
  # CPU-only smoke test of the harness mechanics (NOT a benchmark):
  uv run python -m cs336_systems.all_reduce_bench --world-size 2 --backend gloo --sizes-mb 1,10 --out-csv /tmp/ar.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BYTES_PER_FLOAT32 = 4


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    sizes_bytes: list[int],
    warmup: int,
    iters: int,
    master_addr: str,
    master_port: int,
    out_csv: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    if backend == "nccl":
        # One process per GPU (handout §5.1): pin this rank to its own device.
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        # gloo = CPU debug path only ("debug locally with Gloo, benchmark with NCCL").
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    rows: list[dict] = []
    for size in sizes_bytes:
        numel = size // BYTES_PER_FLOAT32
        data = torch.rand(numel, device=device, dtype=torch.float32)

        # Warmup: NCCL lazily creates channels/buffers on the first collectives.
        for _ in range(warmup):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()  # line the ranks up so one slow rank doesn't skew the loop

        t0 = timeit.default_timer()
        for _ in range(iters):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        # async_op=False only waits for the enqueue — synchronize to time real completion.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = timeit.default_timer()
        per_op_s = (t1 - t0) / iters

        gathered: list[float | None] = [None] * world_size
        dist.all_gather_object(gathered, per_op_s)
        if rank == 0:
            times = [t for t in gathered if t is not None]
            mean_s = statistics.mean(times)
            alg_bw = size / mean_s
            bus_bw = alg_bw * 2 * (world_size - 1) / world_size
            row = {
                "backend": backend,
                "world_size": world_size,
                "size_mb": size / 2**20,
                "numel": numel,
                "warmup": warmup,
                "iters": iters,
                "mean_ms": mean_s * 1e3,
                "max_ms": max(times) * 1e3,
                "alg_bw_gbps": alg_bw / 1e9,
                "bus_bw_gbps": bus_bw / 1e9,
            }
            rows.append(row)
            print(
                f"  size={row['size_mb']:8.1f}MB  mean={row['mean_ms']:9.3f}ms  "
                f"max={row['max_ms']:9.3f}ms  alg_bw={row['alg_bw_gbps']:7.2f}GB/s  bus_bw={row['bus_bw_gbps']:7.2f}GB/s",
                flush=True,
            )
        dist.barrier()
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0 and out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}", flush=True)
    dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser(description="CS336 A2 Part D all-reduce microbenchmark")
    p.add_argument("--world-size", type=int, default=2, help="ranks = GPUs (default job has 2; 4/6 need a larger LSF job)")
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--sizes-mb", type=str, default="1,10,100,1024", help="float32 tensor sizes in MiB (1MB..1GB per handout)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--master-addr", type=str, default="localhost")
    p.add_argument("--master-port", type=int, default=29510)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    sizes_bytes = [int(float(mb) * 2**20) for mb in args.sizes_mb.split(",")]
    if args.backend == "nccl":
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
            print(f"need {args.world_size} CUDA devices, have {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            return 1
    print(
        f"all_reduce_bench: backend={args.backend} world_size={args.world_size} "
        f"sizes_mb={args.sizes_mb} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    mp.spawn(
        _worker,
        args=(args.world_size, args.backend, sizes_bytes, args.warmup, args.iters, args.master_addr, args.master_port, args.out_csv),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
