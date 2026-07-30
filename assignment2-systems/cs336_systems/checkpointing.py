"""
Activation / gradient checkpointing for BasicsTransformerLM (CS336 A2 §3.2).

Strategies
----------
none       : default autograd (O(N) activation residuals)
per_layer  : checkpoint each TransformerBlock (no nesting)
segment    : checkpoint consecutive groups of `segment_size` blocks (no nesting)
recursive  : nested checkpoints (binary tree) — memory-optimal ignoring compute

Uses torch.utils.checkpoint.checkpoint(..., use_reentrant=False).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _run_blocks(blocks: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    for blk in blocks:
        x = blk(x)
    return x


def checkpoint_segment(blocks: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Run a contiguous list of blocks under a single checkpoint (no nesting)."""
    if len(blocks) == 0:
        return x
    if len(blocks) == 1:
        return checkpoint(blocks[0], x, use_reentrant=False)

    def seg(inp: torch.Tensor) -> torch.Tensor:
        return _run_blocks(blocks, inp)

    return checkpoint(seg, x, use_reentrant=False)


def checkpoint_recursive(blocks: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """
    Nested / binary checkpointing.

    Forward saves only the segment input; backward recomputes the segment,
    and within a segment of size > 1 we recurse. Peak activation memory is
    O(log N) block-residuals along the active path (plus O(log N) checkpoint
    inputs), at O(N log N) recompute FLOPs.
    """
    n = len(blocks)
    if n == 0:
        return x
    if n == 1:
        return checkpoint(blocks[0], x, use_reentrant=False)

    mid = n // 2
    left, right = blocks[:mid], blocks[mid:]

    def left_fn(inp: torch.Tensor) -> torch.Tensor:
        return checkpoint_recursive(left, inp)

    def right_fn(inp: torch.Tensor) -> torch.Tensor:
        return checkpoint_recursive(right, inp)

    x = checkpoint(left_fn, x, use_reentrant=False)
    x = checkpoint(right_fn, x, use_reentrant=False)
    return x


def forward_with_checkpointing(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    strategy: str = "none",
    segment_size: int = 1,
) -> torch.Tensor:
    """
    LM forward that optionally checkpoints the transformer stack.

    Mirrors BasicsTransformerLM.forward but replaces the layer loop.
    """
    # Local imports keep this usable if model layout changes slightly.
    from cs336_basics.model import BasicsTransformerLM

    assert isinstance(model, BasicsTransformerLM)
    blocks = list(model.layers)
    n = len(blocks)

    x = model.token_embeddings(input_ids)

    if strategy == "none":
        x = _run_blocks(blocks, x)
    elif strategy == "per_layer":
        for blk in blocks:
            x = checkpoint(blk, x, use_reentrant=False)
    elif strategy == "segment":
        if segment_size < 1:
            raise ValueError("segment_size must be >= 1")
        for i in range(0, n, segment_size):
            x = checkpoint_segment(blocks[i : i + segment_size], x)
    elif strategy == "recursive":
        x = checkpoint_recursive(blocks, x)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    x = model.ln_final(x)
    return model.lm_head(x)


def theoretical_notes(num_layers: int) -> dict[str, str]:
    """Asymptotics for writeup (a). Residuals per block = Θ(1) unit."""
    N = num_layers
    return {
        "none": f"peak_act = O({N}); compute = O({N})",
        "per_layer_no_nest": (
            f"N={N} checkpoints of residual-stream size; within each ckpt only O(1) block residuals. "
            f"If stream << block-residuals, peak ≈ O(1) block residuals (+ N small ckpts). "
            f"Compute ≈ 2 O({N}) (one recompute per block)."
        ),
        "segment_k": (
            "With S = ceil(N/k) segments: peak ≈ S * ckpt_input + k * block_residuals; "
            "compute ≈ O(N) + O(N) recompute (one pass per segment on backward)."
        ),
        "recursive": (
            f"Binary nested checkpoints: recursion depth O(log {N}); "
            f"peak_act = O(log {N}) (active path); compute = O({N} log {N}). "
            "This minimizes peak activation memory when nesting is allowed."
        ),
    }
