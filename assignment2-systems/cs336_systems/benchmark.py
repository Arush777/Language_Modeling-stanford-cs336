"""
CS336 A2 Part A — end-to-end benchmark / profile harness.

Uses staff BasicsTransformerLM + AdamW. Random token batches (handout §2.1.3).
Always run via:  source scripts/env.sh && uv run python -m cs336_systems.benchmark ...
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import timeit
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_ROPE_THETA,
    DEFAULT_VOCAB_SIZE,
    MODEL_CONFIGS,
)
from cs336_systems.flops import (
    flops_for_mode,
    residual_stream_activation_mib,
    transformer_flops,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CS336 A2 Part A benchmark")
    p.add_argument("--model", choices=[*MODEL_CONFIGS.keys(), "all"], default="small")
    p.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "train", "all"],
        default="all",
        help="forward | forward_backward (loss+bwd) | train (+AdamW) | all",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--bf16", action="store_true", help="torch.autocast bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nvtx", action="store_true", help="NVTX ranges + annotated attention")
    p.add_argument(
        "--memory-snapshot",
        type=Path,
        default=None,
        help="If set, dump CUDA memory snapshot pickle after one measured step",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Append one row per run to this CSV",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full result dict JSON",
    )
    p.add_argument(
        "--skip-oom",
        action="store_true",
        default=True,
        help="On OOM, record error and continue (default True)",
    )
    p.add_argument("--no-skip-oom", action="store_false", dest="skip_oom")
    return p.parse_args()


def _build_model(name: str, vocab_size: int, context_length: int, device: torch.device) -> BasicsTransformerLM:
    cfg = MODEL_CONFIGS[name]
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


def _random_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def _one_step(
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    autocast_ctx,
) -> torch.Tensor | None:
    if mode == "forward":
        # Inference-only: do not build autograd graph (handout forward ≠ train memory).
        with torch.inference_mode():
            with autocast_ctx:
                logits = model(x)
        return logits

    model.zero_grad(set_to_none=True)
    with autocast_ctx:
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    if mode == "train":
        assert optimizer is not None
        optimizer.step()
    return loss


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    *,
    model_name: str,
    mode: str,
    warmup: int,
    steps: int,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    bf16: bool,
    device: torch.device,
    use_nvtx: bool,
    memory_snapshot: Path | None,
) -> dict:
    cfg = MODEL_CONFIGS[model_name]
    breakdown = transformer_flops(cfg, batch_size=batch_size, seq_len=context_length, vocab_size=vocab_size)
    flops = flops_for_mode(breakdown, mode)

    model = _build_model(model_name, vocab_size, context_length, device)
    model.train(mode != "forward")
    optimizer = AdamW(model.parameters(), lr=1e-3) if mode == "train" else None
    x, y = _random_batch(batch_size, context_length, vocab_size, device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device.type == "cuda"
        else nullcontext()
    )

    # Warmup
    with nvtx.range("warmup") if use_nvtx else nullcontext():
        for _ in range(warmup):
            _one_step(model, optimizer, x, y, mode, autocast_ctx)
            _sync(device)

    # Optional memory snapshot around a single step after warmup
    if memory_snapshot is not None and device.type == "cuda":
        memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        try:
            with nvtx.range("memory_step") if use_nvtx else nullcontext():
                _one_step(model, optimizer, x, y, mode, autocast_ctx)
                _sync(device)
            torch.cuda.memory._dump_snapshot(str(memory_snapshot))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    times: list[float] = []
    peak_mem_bytes = 0
    with nvtx.range("measure") if use_nvtx else nullcontext():
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            t0 = timeit.default_timer()
            _one_step(model, optimizer, x, y, mode, autocast_ctx)
            _sync(device)
            t1 = timeit.default_timer()
            times.append(t1 - t0)
            if device.type == "cuda":
                peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))

    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    tflops = (flops / mean_s) / 1e12 if mean_s > 0 else float("nan")

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "model": model_name,
        "mode": mode,
        "warmup": warmup,
        "steps": steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "bf16": bf16,
        "device": str(device),
        "mean_s": mean_s,
        "std_s": std_s,
        "times_s": times,
        "peak_mem_bytes": peak_mem_bytes,
        "peak_mem_gib": peak_mem_bytes / (1024**3),
        "num_params": n_params,
        "analytical_flops": flops,
        "tflops_per_s": tflops,
        "residual_stream_mib_fp32": residual_stream_activation_mib(batch_size, context_length, cfg.d_model, 4),
        **breakdown.as_dict(),
        "ok": True,
        "error": None,
    }
    # Free GPU before next config in a sweep
    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "mode",
        "warmup",
        "steps",
        "batch_size",
        "context_length",
        "bf16",
        "mean_s",
        "std_s",
        "peak_mem_gib",
        "analytical_flops",
        "tflops_per_s",
        "num_params",
        "residual_stream_mib_fp32",
        "ok",
        "error",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.nvtx:
        from cs336_systems.nvtx_attention import install_annotated_attention

        install_annotated_attention()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    modes = ["forward", "forward_backward", "train"] if args.mode == "all" else [args.mode]

    print(
        f"device={device} models={models} modes={modes} "
        f"warmup={args.warmup} steps={args.steps} bf16={args.bf16} "
        f"B={args.batch_size} T={args.context_length}"
    )

    results: list[dict] = []
    for model_name in models:
        for mode in modes:
            label = f"{model_name}/{mode}/bf16={args.bf16}/T={args.context_length}/warmup={args.warmup}"
            print(f"\n=== {label} ===", flush=True)
            try:
                snap = None
                if args.memory_snapshot is not None:
                    snap = args.memory_snapshot
                    if args.model == "all" or args.mode == "all":
                        snap = args.memory_snapshot.parent / (
                            f"{args.memory_snapshot.stem}_{model_name}_{mode}"
                            f"_T{args.context_length}_bf16{int(args.bf16)}{args.memory_snapshot.suffix}"
                        )
                row = run_benchmark(
                    model_name=model_name,
                    mode=mode,
                    warmup=args.warmup,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    vocab_size=args.vocab_size,
                    bf16=args.bf16,
                    device=device,
                    use_nvtx=args.nvtx,
                    memory_snapshot=snap,
                )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                row = {
                    "model": model_name,
                    "mode": mode,
                    "warmup": args.warmup,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "context_length": args.context_length,
                    "bf16": args.bf16,
                    "mean_s": None,
                    "std_s": None,
                    "peak_mem_gib": None,
                    "analytical_flops": flops_for_mode(
                        transformer_flops(
                            MODEL_CONFIGS[model_name],
                            batch_size=args.batch_size,
                            seq_len=args.context_length,
                            vocab_size=args.vocab_size,
                        ),
                        mode,
                    ),
                    "tflops_per_s": None,
                    "num_params": None,
                    "ok": False,
                    "error": f"OOM: {e}",
                }
                print(f"OOM: {e}", flush=True)
                if not args.skip_oom:
                    raise
            except Exception as e:
                row = {
                    "model": model_name,
                    "mode": mode,
                    "warmup": args.warmup,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "context_length": args.context_length,
                    "bf16": args.bf16,
                    "mean_s": None,
                    "std_s": None,
                    "peak_mem_gib": None,
                    "analytical_flops": None,
                    "tflops_per_s": None,
                    "ok": False,
                    "error": repr(e),
                }
                print(f"ERROR: {e!r}", flush=True)
                if not args.skip_oom:
                    raise

            results.append(row)
            if row.get("ok"):
                print(
                    f"mean={row['mean_s']:.6f}s ± {row['std_s']:.6f}s | "
                    f"peak={row['peak_mem_gib']:.3f} GiB | "
                    f"FLOPs={row['analytical_flops']:.3e} | "
                    f"{row['tflops_per_s']:.2f} TFLOP/s",
                    flush=True,
                )
            if args.out_csv is not None:
                _append_csv(args.out_csv, row)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out_json}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDone: {n_ok}/{len(results)} runs OK")
    # Exit 0 so LSF sweeps with --skip-oom are not killed by set -e on partial OOMs.
    # Non-OOM failures still exit 2 when --no-skip-oom.
    if n_ok == len(results):
        return 0
    only_oom = all(
        (not r.get("ok")) and r.get("error") and str(r.get("error")).startswith("OOM")
        for r in results
        if not r.get("ok")
    )
    if args.skip_oom and only_oom:
        return 0
    return 2 if not args.skip_oom else 0


if __name__ == "__main__":
    raise SystemExit(main())
