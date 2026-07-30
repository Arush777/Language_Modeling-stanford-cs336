"""
Fully Sharded Data Parallel (FSDP) from first principles (CS336 A2 handout §7).

First principles
----------------
DDP (Part D) replicates EVERYTHING: every rank holds the full weights, full
gradients, and full optimizer state. FSDP asks: if the sharded optimizer
(§6) can shard optimizer STATE, why not shard the weights and gradients too?

The key observation is that a layer's full weight is only needed WHILE that
layer is computing. So each rank keeps only a 1/world_size SHARD of each
weight persistently, and materializes the full weight on demand:

    FORWARD, per sharded layer:
        1. ALL-GATHER the layer's weight shards from all ranks -> full weight
        2. compute with the full weight (ordinary matmul / embedding lookup)
    BACKWARD, per sharded layer (autograd traverses layers in reverse):
        3. autograd hands us the gradient w.r.t. the FULL weight
        4. REDUCE-SCATTER it: sum across ranks (gradients from different data
           shards), keep only our own shard slice -> gradient of OUR shard
    OPTIMIZER:
        5. the optimizer only ever sees shard Parameters, so its state
           (Adam moments, etc.) is automatically sharded too — for free.

Steps 1+4 are two halves of ONE autograd.Function below: forward does the
all-gather, backward does the reduce-scatter. Writing them as one Function is
the tidy part of this design — the collective pair stays symmetric, and the
reduce-scatter overlaps backward compute for free (each layer's gradient
communication fires as soon as autograd reaches it, in reverse layer order).

What gets sharded vs replicated
-------------------------------
Following the handout, we shard the big dense weights — cs336_basics `Linear`
and `Embedding` — and keep everything else REPLICATED. RMSNorm gains are tiny
(d_model-sized vectors: sharding them would save almost nothing while adding
a collective per norm per pass), so they stay full on every rank and their
gradients are synchronized DDP-style (all-reduce + divide by world_size) in
`finish_gradient_synchronization()`. This is also why norm grads end up
bitwise identical across ranks, which the test asserts.

Sharding scheme: each weight is FLATTENED to 1-D, ZERO-PADDED at the tail so
world_size divides the length, and split into equal contiguous chunks; rank r
owns chunk r. WHY pad: `all_gather` / `reduce_scatter_tensor` move EQUAL-SIZED
chunks per rank — an unpadded 6401-element weight cannot be split evenly over
2 ranks. The pad tail is zeros in weight-space and its "gradient" is defined
as zero, so it never influences real elements; gather slices it off again
(`[:orig_numel]`), so padding is invisible outside this file.

DDP math, again: each rank's local loss is a mean over its LOCAL batch shard,
so the cross-rank SUM of local gradients is world_size × the full-batch mean
gradient. We therefore reduce with ReduceOp.SUM and divide by world_size
afterwards (same reasoning as ddp.py — SUM is supported on every backend, and
sum-then-divide keeps the numerics obvious). After that division, a shard's
gradient is exactly the matching slice of the full-batch gradient, so an
optimizer step on shards + a re-gather reproduces single-process full-batch
training (the test checks this to atol=1e-6 in fp32).

Mixed precision (`compute_dtype`): master weights stay fp32 (they are the
persistent shards). When compute_dtype is set, the gathered full weight is
cast to compute_dtype before compute, so matmuls/gradients run in that dtype;
gradients are then cast BACK to fp32 BEFORE the reduce-scatter, so reduction
and the master-weight update happen at fp32 precision (this also matches the
test's mixed-precision reference hook-for-hook). The fp32-then-cast ordering
in the all-gather is deliberate — see the comment in `_FSDPAllGather.forward`.

Two known deviations from the handout's ideal (both deliberate; the tests only
require correctness, and both are quantified in fsdp_bench.py):

1. SYNCHRONOUS all-gather, no prefetch. We gather a layer's weight at the
   moment the layer is called, blocking compute on the collective. The
   handout's ideal PREFETCHES: launch the async all-gather for layer i+1
   while layer i is still computing, hiding comm behind compute. That needs
   forward pre-hooks on the next module and a side stream — real complexity
   for a correctness-only deliverable. On NCCL this choice shows up as comm
   gaps between kernels; on gloo/CPU tests it is irrelevant.

2. Full weights stay ALIVE between forward and backward. Because the model's
   own einsum/indexing ops do the compute (outside our Function), autograd
   saves each gathered full weight for backward — so during backward, ALL
   layers' full weights are resident simultaneously: peak memory ≈ full model
   size on top of the shards. The handout's ideal FREES each full weight
   right after its forward and RE-ALL-GATHERS it at the start of that layer's
   backward (weights cannot change mid-step, so the re-gather returns
   identical values) — halving this transient at the price of one extra
   all-gather per layer per step. Implementing it requires moving the matmul
   INSIDE the autograd.Function (so the Function controls what autograd
   saves); we keep compute outside so the numerics are bit-identical to the
   non-parallel baseline, which the fp32 test's atol=1e-6 effectively demands.

Known limitation (documented, not hit by the tests): tied weights BETWEEN two
sharded modules (e.g. embedding.weight is lm_head.weight) are not handled —
each module would shard the shared tensor independently. Tied weights within
the sharded-optimizer world are handled there (see sharded_optimizer.py);
extending the same id()-dedupe here is future work.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from einops import einsum
from torch import nn

from cs336_basics.model import Embedding, Linear


def _dist_ready() -> bool:
    """True when torch.distributed is initialized, so this also works in a plain single-process debug run."""
    return dist.is_available() and dist.is_initialized()


def _detect_reduce_scatter_support(device: torch.device, world_size: int) -> bool:
    """
    Probe once (at FSDP construction, so all ranks probe at the same point)
    whether this backend implements `dist.reduce_scatter_tensor`.

    NCCL always does. Some gloo builds/dtypes do not; there we fall back to
    all_reduce + keep-my-slice (see `_FSDPAllGather.backward`), which is
    numerically identical but moves the full padded tensor instead of one
    shard's worth per rank. If the probe raises, it raises on EVERY rank at
    the SAME call (same code path), so ranks never desync on the fallback.
    """
    if world_size == 1:
        return False
    probe_in = torch.ones(world_size, device=device)
    probe_out = torch.zeros(1, device=device)
    try:
        dist.reduce_scatter_tensor(probe_out, probe_in, op=dist.ReduceOp.SUM)
    except Exception:
        return False
    return True


class _FSDPAllGather(torch.autograd.Function):
    """
    The FSDP collective pair as one autograd Function.

    forward:  shard (this rank's 1/world_size flat chunk) -> full weight
              (all-gather every rank's chunk, concat, un-pad, reshape, and
              optionally cast to compute_dtype for the actual matmul).
    backward: grad w.r.t. the full weight -> grad w.r.t. OUR shard
              (cast to fp32, pad, reduce-scatter SUM, divide by world_size).

    Because this Function's backward fires the moment autograd finishes the
    layer that used the full weight, the gradient reduce-scatter of layer i
    overlaps the backward compute of layers below i — the FSDP version of
    DDP's hook-based overlap (ddp.py), obtained here without any hooks.
    """

    @staticmethod
    def forward(
        ctx: Any,
        shard: torch.Tensor,
        orig_shape: tuple[int, ...],
        orig_numel: int,
        padded_numel: int,
        shard_numel: int,
        compute_dtype: torch.dtype | None,
        world_size: int,
        rank: int,
        use_reduce_scatter: bool,
    ) -> torch.Tensor:
        # Stash only small scalars for backward. Note we do NOT save the
        # gathered full weight ourselves — the downstream ops (einsum /
        # indexing) save whatever they need in their own autograd nodes. That
        # is exactly why the full weights stay alive until backward (module
        # docstring, deviation #2).
        ctx.orig_numel = orig_numel
        ctx.padded_numel = padded_numel
        ctx.shard_numel = shard_numel
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.use_reduce_scatter = use_reduce_scatter

        if world_size == 1:
            # Nobody to gather from; the "shard" already IS the full flat
            # weight. .clone() so the Function's output does not alias the
            # persistent Parameter's storage (autograd version-counter safety
            # when the optimizer later mutates the shard in place).
            full_flat = shard.detach().clone()
        else:
            # Classic list-based all_gather: the most portable API (gloo +
            # NCCL, CPU + CUDA). NCCL-faster fused alternative:
            # all_gather_into_tensor — same semantics, one fewer concat.
            gathered = [torch.empty_like(shard) for _ in range(world_size)]
            dist.all_gather(gathered, shard.detach())
            full_flat = torch.cat(gathered)  # (world_size * shard_numel,) == padded_numel

        # Un-pad and reshape: the pad tail is dropped here and re-created as
        # zeros on the gradient side in backward.
        full_weight = full_flat[:orig_numel].view(orig_shape)

        if compute_dtype is not None:
            # TRADEOFF: we gather in fp32 (the master dtype) and cast AFTER,
            # rather than casting shards to compute_dtype BEFORE the gather.
            # Cast-before-gather would halve the comm volume for fp16 (the
            # ideal the adapter docstring describes), but gloo's all_gather on
            # CPU fp16 tensors is not guaranteed on every build, and a cast
            # commutes with a pure copy, so the numerics are IDENTICAL either
            # way. We take the portable ordering and pay fp32 wire bytes.
            full_weight = full_weight.to(compute_dtype)
        return full_weight

    @staticmethod
    def backward(ctx: Any, grad_full: torch.Tensor):
        # grad_full: gradient w.r.t. the full weight, shaped orig_shape, in the
        # compute dtype (fp32 normally; compute_dtype under mixed precision).
        #
        # Cast to fp32 BEFORE communicating: (a) the gradient must land on the
        # fp32 master shard, (b) the test's mixed-precision reference casts
        # each fp16 gradient to fp32 before the optimizer sees it — doing the
        # same matches it exactly, and (c) the cross-rank reduction then runs
        # at fp32 precision instead of accumulating fp16 rounding.
        grad_flat = grad_full.reshape(-1).to(torch.float32)

        # Re-create the zero pad tail so the buffer matches the sharding
        # layout. Pad positions have no weight elements, so zero is the
        # correct gradient there; it only ever lands in the last rank's pad
        # tail and is sliced off again by gather_full_params.
        if grad_flat.numel() != ctx.padded_numel:
            padded = grad_flat.new_zeros(ctx.padded_numel)
            padded[: grad_flat.numel()] = grad_flat
        else:
            padded = grad_flat

        shard_grad = torch.empty(ctx.shard_numel, dtype=torch.float32, device=padded.device)
        if ctx.world_size == 1:
            shard_grad.copy_(padded)
        elif ctx.use_reduce_scatter:
            # The handout's collective: SUM across ranks, each rank receives
            # only its own shard slice of the summed gradient.
            dist.reduce_scatter_tensor(shard_grad, padded, op=dist.ReduceOp.SUM)
        else:
            # Portable fallback (see _detect_reduce_scatter_support): sum the
            # full padded gradient everywhere, then keep our slice. Same math,
            # ~world_size times more bytes on the wire than the real thing.
            dist.all_reduce(padded, op=dist.ReduceOp.SUM)
            start = ctx.rank * ctx.shard_numel
            shard_grad.copy_(padded[start : start + ctx.shard_numel])

        # SUM -> divide by world_size = full-batch AVERAGE gradient (DDP math —
        # module docstring). After this, shard_grad is exactly our slice of the
        # non-parallel full-batch gradient, so stepping the shard and
        # re-gathering reproduces single-process training.
        shard_grad.div_(ctx.world_size)

        # One return per forward input (None for every non-tensor argument).
        return shard_grad, None, None, None, None, None, None, None, None


def _make_fsdp_linear_forward(
    mod: Linear,
    meta: dict,
    compute_dtype: torch.dtype | None,
    world_size: int,
    rank: int,
    use_reduce_scatter: bool,
):
    """Replacement instance `forward` for a sharded cs336_basics Linear.

    The compute itself is the SAME einsum the unsharded Linear.forward uses
    (cs336_basics/model.py) — only the weight's PROVENANCE changes (gathered
    from shards instead of a local full tensor). Keeping the model's own op
    means forward/backward numerics match the non-parallel baseline op-for-op,
    which the fp32 correctness test's atol=1e-6 effectively requires.
    """

    def fsdp_forward(x: torch.Tensor) -> torch.Tensor:
        full_weight = _FSDPAllGather.apply(
            mod.weight,
            meta["orig_shape"],
            meta["orig_numel"],
            meta["padded_numel"],
            meta["shard_numel"],
            compute_dtype,
            world_size,
            rank,
            use_reduce_scatter,
        )
        return einsum(x, full_weight, "... d_in, d_out d_in -> ... d_out")

    return fsdp_forward


def _make_fsdp_embedding_forward(
    mod: Embedding,
    meta: dict,
    compute_dtype: torch.dtype | None,
    world_size: int,
    rank: int,
    use_reduce_scatter: bool,
):
    """Replacement instance `forward` for a sharded cs336_basics Embedding —
    same row-indexing op as Embedding.forward, on the gathered full table."""

    def fsdp_forward(token_ids: torch.Tensor) -> torch.Tensor:
        full_weight = _FSDPAllGather.apply(
            mod.weight,
            meta["orig_shape"],
            meta["orig_numel"],
            meta["padded_numel"],
            meta["shard_numel"],
            compute_dtype,
            world_size,
            rank,
            use_reduce_scatter,
        )
        return full_weight[token_ids, :]

    return fsdp_forward


class FSDP(nn.Module):
    """
    Wraps a full nn.Module; shards every cs336_basics Linear/Embedding weight
    across ranks and keeps everything else (e.g. RMSNorm) replicated.

    Container contract (mirrors the DDP wrapper and the tests):
      - `.module` reaches the wrapped model;
      - being an nn.Module, `parameters()` yields the SHARD Parameters (plus
        replicated ones) — so an optimizer built on `fsdp_model.parameters()`
        steps shards directly and its state is sharded for free;
      - call `finish_gradient_synchronization()` after `loss.backward()` and
        before `optimizer.step()` (adapter: fsdp_on_after_backward);
      - `gather_full_params()` reconstructs full fp32 weights on every rank
        (adapter: fsdp_gather_full_params).
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self._compute_dtype = compute_dtype
        if _dist_ready():
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
        else:
            self._world_size, self._rank = 1, 0

        device = next(self.module.parameters()).device
        # One tiny probe collective; all ranks construct FSDP at the same
        # point, so the probe order matches across ranks.
        self._use_reduce_scatter = _detect_reduce_scatter_support(device, self._world_size)

        # param-name (e.g. "layers.0.ff.w1.weight") -> shard layout metadata.
        self._sharded_param_meta: dict[str, dict] = {}

        self._sync_replicas_from_rank_zero(src=0)
        self._shard_all_linear_and_embedding()

    # ------------------------------------------------------------------
    # Init-time replica sync (same lesson as DDP init)
    # ------------------------------------------------------------------
    def _sync_replicas_from_rank_zero(self, src: int = 0) -> None:
        """
        Broadcast every parameter and buffer from rank 0 BEFORE sharding.

        WHY (identical to ddp.py's init broadcast): ranks may seed their RNG
        differently and build different random weights. FSDP defines the global
        model to be rank 0's model, so every rank must slice its shards out of
        rank 0's weights — otherwise the concatenation of shards across ranks
        would be a Frankenstein mix of different random inits. Replicated
        params (RMSNorm) and buffers (e.g. RoPE caches) must likewise start
        bitwise identical. This costs one full-model-size broadcast ONCE at
        construction (parameter-by-parameter, so at most one extra tensor is
        alive at a time); a production FSDP shards-then-broadcasts-shards to
        avoid materializing full weights at all, which matters only when a
        single full weight no longer fits on one rank.
        """
        if self._world_size == 1:
            return
        for param in self.module.parameters():
            dist.broadcast(param.data, src=src)
        for buf in self.module.buffers():
            dist.broadcast(buf.data, src=src)

    # ------------------------------------------------------------------
    # Sharding
    # ------------------------------------------------------------------
    def _shard_all_linear_and_embedding(self) -> None:
        # named_modules() yields every submodule with its dotted path; we shard
        # ONLY Linear/Embedding instances (module docstring: norms stay
        # replicated). Iteration order is registration order — deterministic
        # and identical on all ranks, which keeps every later collective order
        # consistent without any extra coordination.
        for mod_name, mod in self.module.named_modules():
            if isinstance(mod, Linear):
                self._shard_module_weight(mod_name, mod, _make_fsdp_linear_forward)
            elif isinstance(mod, Embedding):
                self._shard_module_weight(mod_name, mod, _make_fsdp_embedding_forward)

    def _shard_module_weight(self, mod_name: str, mod: nn.Module, forward_factory) -> None:
        """
        Replace `mod.weight` (full) with this rank's flat shard, and patch the
        instance's `forward` to all-gather the full weight on demand.

        Memory note: the shard is a `.clone()` of a slice, and the assignment
        drops the module's reference to the original full Parameter — so after
        this function returns, the full weight no longer exists on this rank.
        THIS line is where FSDP's persistent memory saving actually happens.
        """
        weight = mod.weight
        orig_shape = tuple(weight.shape)
        orig_numel = weight.numel()

        flat = weight.detach().reshape(-1)
        # Zero-pad the tail so world_size divides the flat length — all-gather
        # and reduce-scatter move equal-sized chunks per rank (module
        # docstring: "Sharding scheme").
        pad = (-orig_numel) % self._world_size
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        padded_numel = flat.numel()
        shard_numel = padded_numel // self._world_size

        start = self._rank * shard_numel
        shard = flat[start : start + shard_numel].clone()
        # Preserve requires_grad (a frozen weight would stay frozen — its shard
        # simply never gets a grad, and the collective in forward still runs
        # identically on all ranks).
        mod.weight = nn.Parameter(shard, requires_grad=weight.requires_grad)

        meta = {
            "orig_shape": orig_shape,
            "orig_numel": orig_numel,
            "padded_numel": padded_numel,
            "shard_numel": shard_numel,
        }
        self._sharded_param_meta[f"{mod_name}.weight"] = meta

        # Instance-attribute forward shadows the class method for this module
        # only; nn.Module.__call__ resolves `self.forward`, so the model's own
        # code (`self.linear1(x)`) now routes through the all-gather path
        # WITHOUT any edits to the model class.
        mod.forward = forward_factory(
            mod,
            meta,
            self._compute_dtype,
            self._world_size,
            self._rank,
            self._use_reduce_scatter,
        )

    # ------------------------------------------------------------------
    # Forward just delegates — the sharded modules patch themselves
    # ------------------------------------------------------------------
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # ------------------------------------------------------------------
    # Gradient synchronization (adapter: fsdp_on_after_backward)
    # ------------------------------------------------------------------
    def finish_gradient_synchronization(self) -> None:
        """
        DDP-style all-reduce for the REPLICATED trainable parameters only.

        The sharded weights need nothing here: their gradient reduce-scatter
        already happened INSIDE backward (the autograd Function's backward —
        see `_FSDPAllGather`), so by the time `loss.backward()` returns, each
        shard's `.grad` already holds the full-batch-averaged shard gradient
        in fp32 with the shard's shape (exactly what the gradient-sync test
        asserts).

        Replicated params (RMSNorm gains) instead get the plain DDP treatment
        after backward: all-reduce SUM + divide by world_size. We do it
        synchronously and only here — norm vectors are tiny, so overlapping
        their communication with hooks (ddp.py-style) would buy nothing.

        One-call-per-step assumption, same as ddp.py: call this exactly once
        per backward (no gradient accumulation across micro-batches).
        """
        if self._world_size == 1:
            return
        for name, param in self.module.named_parameters():
            if name in self._sharded_param_meta:
                continue  # sharded: already synchronized during backward
            if not param.requires_grad or param.grad is None:
                # Unused/frozen this step. All ranks run the same graph, so
                # this is consistent across ranks — safe to skip.
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self._world_size)

    # ------------------------------------------------------------------
    # Full-parameter reconstruction (adapter: fsdp_gather_full_params)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """
        Reconstruct the full fp32 parameter tensors on EVERY rank.

        Sharded params: all-gather the flat shards, concat, slice off the pad
        tail, reshape to the original shape. Replicated params: returned as-is
        (a defensive .clone(), so the returned dict never aliases live params
        that the next optimizer step will mutate).

        This is a COLLECTIVE for sharded params — all ranks must call it
        together (the tests do). Master-weight dtype is fp32 regardless of
        compute_dtype, because the persistent shards never left fp32.
        """
        full: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            meta = self._sharded_param_meta.get(name)
            if meta is None:
                full[name] = param.detach().clone()
                continue
            if self._world_size == 1:
                full_flat = param.detach().reshape(-1)
            else:
                gathered = [torch.empty_like(param) for _ in range(self._world_size)]
                dist.all_gather(gathered, param.detach())
                full_flat = torch.cat(gathered)
            full[name] = full_flat[: meta["orig_numel"]].view(meta["orig_shape"]).clone()
        return full
