from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

# ---- Import YOUR implementations from cs336_basics ----
from cs336_basics.data import run_get_batch
from cs336_basics.nn import cross_entropy
from cs336_basics.optim import get_adamw_cls, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.serialization import save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM

def estimate_loss(
    model: torch.nn.Module,
    data: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    """
    Evaluate the model by averaging loss over eval_iters random batches from `data`.
    Returns a Python float (mean loss).
    """
    model.eval()  # switch to eval mode (turns off dropout if you add it later)
    losses: list[float] = []

    for _ in range(eval_iters):
        # Sample a batch of token windows from the validation stream
        x, y = run_get_batch(data, batch_size, context_length, device)

        # Forward pass: logits shape should be (B, T, V)
        logits = model(x)

        # Flatten logits/targets for cross-entropy:
        # logits_2d: (B*T, V), targets_1d: (B*T,)
        B, T, V = logits.shape
        loss = cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        losses.append(float(loss.item()))

    model.train()  # back to train mode
    return float(np.mean(losses))

def append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        
def main() -> None:
    # -------------------------
    # 1) Parse command-line args
    # -------------------------
    p = argparse.ArgumentParser()

    # Paths to encoded token streams (.bin files)
    p.add_argument("--train_bin", type=Path, required=True)
    p.add_argument("--valid_bin", type=Path, required=True)

    # Model vocab size (must match tokenizer vocab.json size)
    p.add_argument("--vocab_size", type=int, required=True)

    # Data / batching
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)

    # Model hyperparams
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparams
    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=25)

    # Optim hyperparams
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=50)
    p.add_argument("--cosine_cycle_iters", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Misc
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Checkpointing
    p.add_argument("--ckpt_path", type=Path, default=Path("checkpoint.pt"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_path", type=Path, default=None)

    args = p.parse_args()
    
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.log_path is None:
        args.log_path = args.ckpt_path.with_suffix(".trainlog.jsonl")
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    
    # -------------------------
    # 2) Reproducibility / device
    # -------------------------
    torch.manual_seed(args.seed)        # sets torch RNG seed
    np.random.seed(args.seed)           # sets numpy RNG seed
    device = args.device                # e.g. "cuda" or "cpu"

    # Optional speed tweak on NVIDIA GPUs (safe for training)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    # -------------------------
    # 3) Load datasets as memmaps
    # -------------------------
    # These are flat 1D streams of token IDs saved by scripts/encode_dataset.py
    train_data = np.memmap(args.train_bin, dtype=np.uint32, mode="r")
    valid_data = np.memmap(args.valid_bin, dtype=np.uint32, mode="r")
    
    
     # -------------------------
    # 4) Build the model
    # -------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)  # move model params to GPU/CPU
    
    AdamW = get_adamw_cls()  # returns your custom optimizer class
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -------------------------
    # 6) Optionally resume checkpoint
    # -------------------------
    start_iter = 0
    if args.resume and args.ckpt_path.exists():
        start_iter = load_checkpoint(args.ckpt_path, model, optimizer)
        print(f"Resumed from {args.ckpt_path} at iteration {start_iter}")
        
    # Save run config next to checkpoint (helps reproducibility)
    config_path = args.ckpt_path.with_suffix(".config.json")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    
    # -------------------------
    # 7) Training loop
    # -------------------------
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        # (a) Set learning rate using cosine schedule with warmup
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # (b) Get a batch: x is inputs, y is next-token labels
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)

        # (c) Forward pass -> logits (B, T, V)
        logits = model(x)
        B, T, V = logits.shape

        # (d) Loss = average cross-entropy over all tokens in batch
        loss = cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        # (e) Backprop + update
        optimizer.zero_grad(set_to_none=True)  # clears old gradients
        loss.backward()                        # compute gradients
        gradient_clipping(model.parameters(), args.grad_clip)  # prevent exploding grads
        optimizer.step()                       # update weights

        # (f) Logging
        if it % args.log_every == 0:
            elapsed_min = (time.time() - t0) / 60.0
            train_loss = float(loss.item())
            train_ppl = math.exp(train_loss)
            print(
                f"iter {it:6d} | "
                f"train_loss {train_loss:.4f} | train_ppl {train_ppl:.2f} | "
                f"lr {lr:.3e} | {elapsed_min:.1f} min"
            )
            append_log(args.log_path, {
                "iter": it,
                "split": "train",
                "loss": train_loss,
                "ppl": train_ppl,
                "lr": lr,
                "elapsed_sec": time.time() - t0,
            })

        # (g) Periodic evaluation + checkpoint
        if it > 0 and it % args.eval_every == 0:
            val_loss = estimate_loss(
                model=model,
                data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )
            val_ppl = math.exp(val_loss)
            elapsed_min = (time.time() - t0) / 60.0

            print(
                f"== eval @ iter {it:6d} | "
                f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f} | {elapsed_min:.1f} min =="
            )
            append_log(args.log_path, {
                "iter": it,
                "split": "val",
                "loss": val_loss,
                "ppl": val_ppl,
                "lr": lr,
                "elapsed_sec": time.time() - t0,
            })

            save_checkpoint(model, optimizer, it, args.ckpt_path)

    # -------------------------
    # 8) Final eval + save
    # -------------------------
    val_loss = estimate_loss(
        model=model,
        data=valid_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        eval_iters=args.eval_iters,
    )
    print(f"FINAL | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")

    save_checkpoint(model, optimizer, args.max_iters, args.ckpt_path)


if __name__ == "__main__":
    main()