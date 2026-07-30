"""
Distributed Data Parallel (DDP) from first principles (CS336 A2 handout §5.2–5.3).

First principles
----------------
Data parallelism: every rank holds a FULL replica of the model, and each training
batch is sharded so rank r only sees n/world_size examples. After the local
backward pass, each rank holds gradients computed from its own shard only.
Before `optimizer.step()` we ALL-REDUCE the gradients across ranks, so every
replica updates with the gradient of the FULL batch. Since all replicas start
from the same weights and apply the same averaged update every step, they never
drift apart — DDP training then matches single-process training on the full
batch (up to float rounding).

Two details that are easy to get wrong:

1. WHY broadcast at init? Ranks seed their RNG differently, so each replica
   starts with DIFFERENT random weights. Averaging gradients only keeps
   replicas in sync if they started identical, so we broadcast rank 0's
   parameters (and buffers) to everyone else before training. Note we must
   broadcast ALL parameters, not just `requires_grad=True` ones: a "fixed"
   parameter can still be randomly initialized (e.g. ToyModel's frozen
   `fc2.fc.bias`), and replicas must match exactly. The broadcast is IN-PLACE
   (`dist.broadcast` writes into `param.data`), so an optimizer constructed
   after wrapping sees the broadcast values.

2. WHY average and not sum? The single-process reference computes MSE over the
   full batch of N examples: grad_full = (1/N) Σ dL_i. A DDP rank computes MSE
   over its local n = N/world_size examples: grad_local = (1/n) Σ_local dL_i.
   Summing per-rank grads therefore gives world_size × grad_full, so we divide
   by world_size afterwards. I use ReduceOp.SUM + `/ world_size` instead of
   ReduceOp.AVG because SUM is supported on every backend (gloo historically
   has no AVG), and sum-then-divide keeps the numerics obvious.

Three implementations, in increasing sophistication. All share the broadcast
above and the same training-loop API: call `finish_gradient_synchronization()`
after `loss.backward()` and before `optimizer.step()`.

- `NaiveDDP` (§5.2): after the FULL backward pass, loop over parameters and
  synchronously all-reduce each gradient. One collective per tensor, all issued
  after compute has finished — simple, but pays per-call overhead × #params and
  hides nothing.
- `FlatDDP` (§5.3.1): flatten all gradients into ONE tensor, all-reduce with a
  single collective, copy the averaged pieces back. Per-call overhead is paid
  once instead of once per parameter, but there is still no overlap with
  compute, and it costs an extra full-size copy of all gradients.
- `DDP` (§5.3.2, the adapter default): register a post-accumulate-grad hook on
  every unique trainable parameter; the moment a gradient is finished during
  backward, launch an ASYNC all-reduce for it. Communication of already-done
  layers overlaps with backward compute of the remaining layers, and
  `finish_gradient_synchronization()` only waits for whatever is still in
  flight.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def _dist_ready() -> bool:
    """True when torch.distributed is initialized, so these wrappers also work in a plain single-process debug run."""
    return dist.is_available() and dist.is_initialized()


def _unique_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """
    Trainable parameters, deduplicated by tensor identity.

    WHY dedupe: models with TIED weights (e.g. `fc4.weight = fc2.weight`) expose
    the SAME underlying tensor under several names. Communicating it once per
    name would all-reduce the same gradient twice — after SUM + /world_size
    scaling that yields world_size × the correct average (2× too large with 2
    ranks). `module.parameters()` already removes duplicates, but I keep the
    explicit id()-set guard because hook registration keys on tensor identity,
    not names — this is the classic tied-weight DDP pitfall.
    """
    seen: set[int] = set()
    params: list[nn.Parameter] = []
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


class _DDPBase(nn.Module):
    """
    Shared scaffolding for all three variants.

    Subclassing nn.Module and assigning `self.module = module` makes the wrapper
    a proper Module container: `parameters()` / `named_parameters()` recurse into
    the child and yield the SAME tensor objects as the wrapped model (no copies),
    so `optim.SGD(ddp_model.parameters())` optimizes the wrapped model directly.
    The tests additionally reach the wrapped model through the `.module`
    attribute (`ddp_model.module.state_dict()`).
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        if not _dist_ready():
            # Single-process fallback: nobody to communicate with.
            self._world_size = 1
        else:
            self._world_size = dist.get_world_size()
            self._broadcast_from_rank_zero(src=0)

    def _broadcast_from_rank_zero(self, src: int = 0) -> None:
        # Parameters: broadcast ALL of them, including requires_grad=False ones —
        # see module docstring, point 1.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=src)
        # Buffers too: e.g. BatchNorm running statistics could legitimately
        # differ across ranks. (The RoPE cache in our transformer is derived
        # deterministically from the config, but broadcasting keeps the
        # invariant "replicas are bitwise identical at step 0" for any module.)
        for buf in self.module.buffers():
            dist.broadcast(buf.data, src=src)

    def forward(self, *inputs, **kwargs):
        # DDP must not change the model's computation — just delegate.
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Runs after `loss.backward()` and before `optimizer.step()` (variant-specific)."""
        raise NotImplementedError

    def _scale_grad_to_average(self, param: nn.Parameter) -> None:
        # Turn the all-reduce SUM into the full-batch AVERAGE (docstring point 2).
        param.grad.div_(self._world_size)


class DDP(_DDPBase):
    """
    §5.3.2 — overlap gradient communication with backward computation.

    WHY hooks: backward computes gradients incrementally, from the loss toward
    the input, so a parameter's gradient is ready (and communicable) long before
    the backward pass ends. `register_post_accumulate_grad_hook` fires the moment
    autograd has FULLY accumulated a parameter's gradient — for a tied weight
    that means after both uses have contributed, exactly matching single-process
    semantics — so the hook is the natural place to launch communication without
    changing model code.

    WHY async + wait: `dist.all_reduce(..., async_op=True)` returns a handle
    immediately instead of blocking, so backward can keep computing the next
    gradients while the collective runs on the communication stream. The only
    hard requirement is that the collective has finished before the optimizer
    READS `.grad`, so `finish_gradient_synchronization()` waits on every handle
    and only then divides by world_size — dividing before the wait would race
    with the in-flight collective that is still writing into the same tensor.

    Assumption (matching the handout training loop): exactly one `backward()`
    per `finish_gradient_synchronization()` — no gradient accumulation across
    micro-batches, which would need a no_sync-style guard.
    """

    def __init__(self, module: nn.Module):
        super().__init__(module)
        # (handle, param) pairs for all-reduces launched during this step's backward.
        self._pending: list[tuple[dist.Work, nn.Parameter]] = []
        if self._world_size == 1:
            return
        for param in _unique_trainable_parameters(self.module):
            param.register_post_accumulate_grad_hook(self._on_grad_accumulated)

    def _on_grad_accumulated(self, param: nn.Parameter) -> None:
        # SUM now (async), divide by world_size after wait() in finish_...().
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append((handle, param))

    def finish_gradient_synchronization(self) -> None:
        for handle, param in self._pending:
            handle.wait()
            self._scale_grad_to_average(param)
        self._pending.clear()


class NaiveDDP(_DDPBase):
    """
    §5.2 — the minimal implementation: run the WHOLE backward pass first, then
    loop over parameters and synchronously all-reduce each gradient. Every
    tensor is its own collective (per-call overhead × #params) and none of the
    communication overlaps with compute. This is the baseline that FlatDDP and
    DDP improve on; kept for benchmarking (see ddp_bench.py).
    """

    def finish_gradient_synchronization(self) -> None:
        if self._world_size == 1:
            return
        for param in _unique_trainable_parameters(self.module):
            if param.grad is None:
                # Parameter unused in this step's forward. All ranks run the same
                # graph, so this is consistent across ranks — safe to skip.
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            self._scale_grad_to_average(param)


class FlatDDP(_DDPBase):
    """
    §5.3.1 — batch the communication: flatten every gradient into ONE tensor,
    all-reduce it with a single collective, then copy the averaged pieces back
    (the handout suggests torch._utils._flatten_dense_tensors /
    _unflatten_dense_tensors). One collective total → per-call overhead is paid
    once instead of once per parameter. Costs an extra full-size copy of all
    gradients and still has zero overlap with backward compute.

    Assumes all gradients share one dtype/device (true for our fp32 transformer);
    a mixed-dtype model would need per-dtype buckets.
    """

    def finish_gradient_synchronization(self) -> None:
        if self._world_size == 1:
            return
        grads = [p.grad for p in _unique_trainable_parameters(self.module) if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
        flat.div_(self._world_size)
        # _unflatten_dense_tensors returns views into `flat` with the original
        # shapes; copy them back over the real .grad tensors the optimizer reads.
        for grad, averaged in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(averaged)
