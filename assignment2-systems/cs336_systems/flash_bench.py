"""
Part C — FlashAttention vs naive attention latency (handout flash_benchmarking).

Uses triton.testing.do_bench. Batch=1, causal=True. Sweep seq × d × dtype.
On A100 we cap seq below handout's 65536 if needed to avoid multi-hour runs;
the job script sets the sweep explicitly.

Memory first principles (why we preflight)
------------------------------------------
Naive / flash-recompute backward both materialize O(N²) score matrices in HBM:
  S, P, dP, dS  ≈ 4 × (B · N · N · 4 bytes) in fp32
plus a handful of (B, N, d) activations. FlashAttention's *forward* is O(N) in
extra HBM, but our educational backward still pays O(N²) (handout-allowed).
We skip a config when the estimate exceeds ~85% of free GPU memory so we never
thrash into an OOM mid-do_bench.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton.testing

from cs336_systems.flash_attention_pytorch import FlashAttentionPyTorch
from cs336_systems.flash_attention_triton import FlashAttentionTriton


def naive_causal_attention(Q, K, V):
    d = Q.shape[-1]
    S = torch.matmul(Q, K.transpose(-2, -1)) * (d**-0.5)
    n = S.shape[-1]
    mask = torch.tril(torch.ones(n, n, device=S.device, dtype=torch.bool))
    S = S.masked_fill(~mask, -1e6)
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def estimate_peak_bytes(impl: str, batch: int, seq: int, d: int, elem_size: int) -> int:
    """
    Conservative peak HBM for one fwd+bwd.

    Score tiles live as full (B, N, N) fp32 during backward for *all* three
    impls here (naive saves them; flash_* recomputes them densely).
    """
    bytes_n2 = batch * seq * seq * 4  # fp32 scores even if Q is bf16
    # ~4 live N×N tensors (S, P, dP, dS) — overlapping lifetimes in recompute
    n2_bytes = 4 * bytes_n2
    # Q,K,V,O,dO,dQ,dK,dV + float casts ≈ 12 × (B,N,d) working buffers
    act_bytes = 12 * batch * seq * d * max(elem_size, 4)
    # torch.compile / allocator fragmentation headroom
    safety = 2.0
    return int((n2_bytes + act_bytes) * safety)


def gpu_budget_bytes(frac: float = 0.85) -> int:
    """Us most of the free bytes on this device; never assume a full empty GPU."""
    free, total = torch.cuda.mem_get_info()
    # Also leave a floor so we don't plan past physical capacity
    return int(min(free, total) * frac)


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Flush every row so a wall-clock kill doesn't wipe hours of timings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--seq-sizes", default="128,256,512,1024,2048,4096,8192")
    p.add_argument("--d-sizes", default="16,32,64,128")
    p.add_argument("--dtypes", default="float32,bfloat16")
    p.add_argument(
        "--impls",
        default="naive,flash_pytorch,flash_triton",
        help="Comma-separated subset of: naive,flash_pytorch,flash_triton",
    )
    p.add_argument(
        "--mem-frac",
        type=float,
        default=0.85,
        help="Skip config if estimate > this fraction of free GPU memory",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    seqs = [int(x) for x in args.seq_sizes.split(",")]
    ds = [int(x) for x in args.d_sizes.split(",")]
    dtypes = []
    for name in args.dtypes.split(","):
        dtypes.append(getattr(torch, name))
    want_impls = {x.strip() for x in args.impls.split(",") if x.strip()}
    all_impls = [
        ("naive", naive_causal_attention),
        ("flash_pytorch", FlashAttentionPyTorch.apply),
        ("flash_triton", FlashAttentionTriton.apply),
    ]
    impls = [(n, fn) for n, fn in all_impls if n in want_impls]
    if not impls:
        raise SystemExit(f"no valid --impls in {args.impls!r}")

    fieldnames = [
        "impl",
        "dtype",
        "d",
        "seq",
        "fwd_ms",
        "bwd_ms",
        "e2e_ms",
        "ok",
        "error",
        "est_peak_gib",
    ]
    # Fresh run: clobber any partial file with the same name
    if args.out_csv.exists():
        args.out_csv.unlink()

    budget = gpu_budget_bytes(args.mem_frac)
    free, total = torch.cuda.mem_get_info()
    print(
        f"GPU mem: free={free/1024**3:.1f}GiB total={total/1024**3:.1f}GiB "
        f"skip_budget={budget/1024**3:.1f}GiB (frac={args.mem_frac})",
        flush=True,
    )

    for dtype in dtypes:
        elem = torch.tensor([], dtype=dtype).element_size()
        for d in ds:
            for seq in seqs:
                for impl_name, apply_fn in impls:
                    label = f"{impl_name} dtype={dtype} d={d} seq={seq}"
                    print(f"=== {label} ===", flush=True)
                    est = estimate_peak_bytes(impl_name, batch=1, seq=seq, d=d, elem_size=elem)
                    est_gib = est / 1024**3
                    if est > budget:
                        row = {
                            "impl": impl_name,
                            "dtype": str(dtype).replace("torch.", ""),
                            "d": d,
                            "seq": seq,
                            "fwd_ms": None,
                            "bwd_ms": None,
                            "e2e_ms": None,
                            "ok": False,
                            "error": f"skip_est_oom:{est_gib:.1f}GiB>{budget/1024**3:.1f}GiB",
                            "est_peak_gib": round(est_gib, 3),
                        }
                        print(f"  SKIP est_peak={est_gib:.2f}GiB > budget", flush=True)
                        append_row(args.out_csv, row, fieldnames)
                        continue

                    try:
                        Q = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)
                        K = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)
                        V = torch.randn(1, seq, d, device="cuda", dtype=dtype, requires_grad=True)

                        def fwd():
                            if impl_name == "naive":
                                return apply_fn(Q, K, V)
                            return apply_fn(Q, K, V, True)

                        def bwd():
                            Q.grad = K.grad = V.grad = None
                            out = fwd()
                            out.sum().backward()

                        def fwd_bwd():
                            Q.grad = K.grad = V.grad = None
                            out = fwd()
                            out.sum().backward()

                        # Warm kernels
                        fwd_bwd()
                        torch.cuda.synchronize()

                        fwd_ms = triton.testing.do_bench(fwd)
                        bwd_ms = triton.testing.do_bench(bwd)
                        e2e_ms = triton.testing.do_bench(fwd_bwd)
                        row = {
                            "impl": impl_name,
                            "dtype": str(dtype).replace("torch.", ""),
                            "d": d,
                            "seq": seq,
                            "fwd_ms": fwd_ms,
                            "bwd_ms": bwd_ms,
                            "e2e_ms": e2e_ms,
                            "ok": True,
                            "error": None,
                            "est_peak_gib": round(est_gib, 3),
                        }
                        print(f"  fwd={fwd_ms:.3f} bwd={bwd_ms:.3f} e2e={e2e_ms:.3f} ms", flush=True)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        row = {
                            "impl": impl_name,
                            "dtype": str(dtype).replace("torch.", ""),
                            "d": d,
                            "seq": seq,
                            "fwd_ms": None,
                            "bwd_ms": None,
                            "e2e_ms": None,
                            "ok": False,
                            "error": "OOM",
                            "est_peak_gib": round(est_gib, 3),
                        }
                        print("  OOM", flush=True)
                    except Exception as e:
                        torch.cuda.empty_cache()
                        row = {
                            "impl": impl_name,
                            "dtype": str(dtype).replace("torch.", ""),
                            "d": d,
                            "seq": seq,
                            "fwd_ms": None,
                            "bwd_ms": None,
                            "e2e_ms": None,
                            "ok": False,
                            "error": repr(e),
                            "est_peak_gib": round(est_gib, 3),
                        }
                        print(f"  ERR {e!r}", flush=True)
                    append_row(args.out_csv, row, fieldnames)

    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
