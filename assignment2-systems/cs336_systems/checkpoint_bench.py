"""
Part B — gradient checkpointing experiments (handout gradient_checkpointing).

(a) Theory printed via theoretical_notes.
(b) xl, B=4, T=2048: sweep segment sizes (no nesting) and measure peak HBM;
    also report recursive (nested) peak for reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.checkpointing import forward_with_checkpointing, theoretical_notes
from cs336_systems.configs import DEFAULT_ROPE_THETA, MODEL_CONFIGS


def _build_xl(context_length: int, device: torch.device) -> BasicsTransformerLM:
    cfg = MODEL_CONFIGS["xl"]
    # Handout §3.2 example used num_heads=16 with d=2560; Table 1 says 32.
    # We follow Table 1 / configs.py for consistency with Part A.
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=DEFAULT_ROPE_THETA,
    )
    return model.to(device)


def measure_peak(
    *,
    strategy: str,
    segment_size: int,
    context_length: int,
    batch_size: int,
    mode: str,
    device: torch.device,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _build_xl(context_length, device)
    model.train(mode != "forward")
    x = torch.randint(0, 10_000, (batch_size, context_length), device=device)
    y = torch.randint(0, 10_000, (batch_size, context_length), device=device)
    opt = AdamW(model.parameters(), lr=1e-3) if mode == "train" else None

    def run() -> None:
        if mode == "forward":
            with torch.inference_mode():
                forward_with_checkpointing(
                    model, x, strategy=strategy, segment_size=segment_size
                )
            return
        model.zero_grad(set_to_none=True)
        logits = forward_with_checkpointing(
            model, x, strategy=strategy, segment_size=segment_size
        )
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if mode == "train":
            assert opt is not None
            opt.step()

    # warmup once
    try:
        run()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        run()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        ok, err = True, None
    except torch.cuda.OutOfMemoryError as e:
        peak, ok, err = None, False, f"OOM: {e}"
        torch.cuda.empty_cache()

    del model, opt, x, y
    torch.cuda.empty_cache()
    return {
        "strategy": strategy,
        "segment_size": segment_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "mode": mode,
        "peak_mem_bytes": peak,
        "peak_mem_gib": None if peak is None else peak / (1024**3),
        "num_layers": MODEL_CONFIGS["xl"].num_layers,
        "ok": ok,
        "error": err,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward_backward")
    p.add_argument(
        "--segment-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma list of segment sizes for non-nested sweep (part b)",
    )
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    device = torch.device(args.device)

    N = MODEL_CONFIGS["xl"].num_layers
    print("=== theory (a) ===")
    for k, v in theoretical_notes(N).items():
        print(f"{k}: {v}")

    rows: list[dict] = []

    # Baseline none (may OOM at T=2048 train — expected from Part A)
    print("\n=== baseline none ===")
    row = measure_peak(
        strategy="none",
        segment_size=1,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        device=device,
    )
    print(row)
    rows.append(row)

    sizes = [int(s) for s in args.segment_sizes.split(",") if s.strip()]
    print("\n=== segment sweep (b, no nesting) ===")
    for k in sizes:
        if k > N:
            continue
        row = measure_peak(
            strategy="segment",
            segment_size=k,
            context_length=args.context_length,
            batch_size=args.batch_size,
            mode=args.mode,
            device=device,
        )
        print(f"segment_size={k}: {row}")
        rows.append(row)

    print("\n=== recursive (nested, for writeup a) ===")
    row = measure_peak(
        strategy="recursive",
        segment_size=1,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        device=device,
    )
    print(row)
    rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Recommend best non-nested by min peak among OK rows
    ok_seg = [r for r in rows if r["ok"] and r["strategy"] == "segment"]
    if ok_seg:
        best = min(ok_seg, key=lambda r: r["peak_mem_gib"])
        print(
            f"\nBest non-nested segment_size={best['segment_size']} "
            f"peak={best['peak_mem_gib']:.3f} GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
