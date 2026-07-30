"""
Part C — naive attention scaling + torch.compile (handout pytorch_attention, torch_compile).

First principles
----------------
Attention scores S are (seq, seq) per batch item. Memory for S (and for saved P in
backward) grows as O(seq^2). This script measures when that blows up on an A100.

We use a *single head* (no head dim): Q,K,V shaped (batch=8, seq, d_model) as the
handout asks.
"""

from __future__ import annotations

import argparse
import csv
import timeit
from pathlib import Path

import torch
import torch.nn.functional as F


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard SDPA: materializes (B, T, T) scores."""
    d = Q.shape[-1]
    S = torch.matmul(Q, K.transpose(-2, -1)) / (d**0.5)
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def bench_pair(
    fn,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 100,
) -> tuple[float, float, float | None]:
    """Return (fwd_ms, bwd_ms, mem_bytes_before_bwd)."""
    for _ in range(warmup):
        out = fn(Q, K, V)
        loss = out.sum()
        loss.backward()
        Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()

    # Forward timing
    torch.cuda.synchronize()
    t0 = timeit.default_timer()
    for _ in range(iters):
        out = fn(Q, K, V)
        torch.cuda.synchronize()
    t1 = timeit.default_timer()
    fwd_ms = (t1 - t0) / iters * 1e3

    # Memory before backward (after one forward, grads cleared)
    Q.grad = K.grad = V.grad = None
    torch.cuda.reset_peak_memory_stats()
    out = fn(Q, K, V)
    torch.cuda.synchronize()
    mem_before_bwd = torch.cuda.memory_allocated()

    # Backward timing
    torch.cuda.synchronize()
    t0 = timeit.default_timer()
    for _ in range(iters):
        out = fn(Q, K, V)
        out.sum().backward()
        Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()
    t1 = timeit.default_timer()
    bwd_ms = (t1 - t0) / iters * 1e3
    return fwd_ms, bwd_ms, mem_before_bwd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    device = torch.device("cuda")
    d_list = [16, 32, 64, 128]
    seq_list = [256, 1024, 4096, 8192, 16384]
    batch = 8

    rows: list[dict] = []
    for d in d_list:
        for seq in seq_list:
            for compiled in (False, True):
                label = f"d={d} seq={seq} compiled={compiled}"
                print(f"=== {label} ===", flush=True)
                try:
                    Q = torch.randn(batch, seq, d, device=device, requires_grad=True)
                    K = torch.randn(batch, seq, d, device=device, requires_grad=True)
                    V = torch.randn(batch, seq, d, device=device, requires_grad=True)
                    fn = naive_attention
                    if compiled:
                        fn = torch.compile(naive_attention)
                    fwd_ms, bwd_ms, mem = bench_pair(
                        fn, Q, K, V, warmup=args.warmup, iters=args.iters
                    )
                    row = {
                        "d_model": d,
                        "seq": seq,
                        "compiled": compiled,
                        "fwd_ms": fwd_ms,
                        "bwd_ms": bwd_ms,
                        "mem_before_bwd_gib": mem / (1024**3),
                        "ok": True,
                        "error": None,
                    }
                    print(
                        f"  fwd={fwd_ms:.3f}ms bwd={bwd_ms:.3f}ms mem={row['mem_before_bwd_gib']:.3f}GiB",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    row = {
                        "d_model": d,
                        "seq": seq,
                        "compiled": compiled,
                        "fwd_ms": None,
                        "bwd_ms": None,
                        "mem_before_bwd_gib": None,
                        "ok": False,
                        "error": "OOM",
                    }
                    print(f"  OOM: {e}", flush=True)
                rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
