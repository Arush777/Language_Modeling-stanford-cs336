"""
Optimizer state sharding from first principles (CS336 A2 handout §6).

First principles
----------------
Data parallelism (Part D) makes every rank hold a FULL replica of the model,
including the optimizer state. For Adam-family optimizers that state is large:
two fp32 moments (m and v) per parameter, i.e. 2× the parameter bytes. With W
ranks we store W identical copies of the same moments — pure waste.

The fix is embarrassingly simple once you notice that Adam's update rule is
PER-PARAMETER: the update for parameter p depends only on p's own gradient,
moments, and hyperparameters — never on any other parameter. So we can
PARTITION the parameters across ranks, let each rank's local optimizer keep
moments for (and step) only its own ~1/W slice, and then make every rank's
replica consistent again by BROADCASTING each freshly-updated parameter from
the rank that owns it:

    backward()                      # every rank has the full, averaged gradient
    local_optimizer.step()          # rank r updates only the params it owns
    for each param p:
        dist.broadcast(p, src=owner(p))   # everyone else receives the update

Memory per rank drops from (moments for ALL params) to (moments for 1/W of
them); the price is a broadcast of the full parameter set after each step —
exactly the same bytes as DDP's gradient all-reduce, just on weights instead of
gradients (both are the full parameter size on the wire, modulo reduction
factor details). We comment on this tradeoff again in `step()`.

Correctness argument: all ranks start from identical weights (same seed, or a
DDP/FSDP init broadcast) and see identical (already-averaged) gradients before
step. The owner rank's Adam update is a deterministic function of those
identical values, so the broadcast values are EXACTLY what a single-process
AdamW would have produced. Non-owner ranks simply adopt those values. Final
weights therefore match the non-sharded optimizer bit-for-bit (which the test
in tests/test_sharded_optimizer.py verifies over 10 steps, tolerances ~1e-7).

Two details that are easy to get wrong:

1. WHY dedupe tied weights by tensor id? A model with `fc4.weight = fc2.weight`
   exposes the SAME tensor under two names. If both copies were assigned to
   ranks, the tensor could be stepped twice (once per alias) or broadcast from
   two different "owners" — either way the aliases diverge from each other and
   from the single-process reference. We dedupe by id() and assign each unique
   tensor to EXACTLY ONE owner rank; broadcasting the single underlying tensor
   updates every alias at once, because all aliases share the same storage.
   (This is the same tied-weight pitfall as in ddp.py — see its docstring.)

2. WHY round-robin (param_index % world_size) ownership? It is deterministic
   (all ranks iterate parameters in the same order — same model, same
   construction — so they independently agree on who owns what WITHOUT any
   extra communication), and it balances both the moment memory and the step()
   compute to within one parameter per rank. A smarter partition could balance
   by numel instead of count; round-robin is the teaching-simple version.

API contract (tests/adapters.py):
    ShardedOptimizer(params, optimizer_cls, **optimizer_kwargs)

`params` may be a plain iterable of tensors OR optimizer param-group dicts —
we pass it through `torch.optim.Optimizer.__init__`, which normalizes both
forms by calling our `add_param_group` once per group. The subclassing of
torch.optim.Optimizer is deliberate: `zero_grad()`, `state_dict()` and friends
must work on the FULL parameter list on every rank (the training loop calls
`sharded_optimizer.zero_grad()` before each backward).
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    """True when torch.distributed is initialized, so this also works in a plain single-process debug run."""
    return dist.is_available() and dist.is_initialized()


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Wraps a local `optimizer_cls` so that optimizer STATE (and the step compute)
    is sharded across ranks, while the PARAMETERS stay fully replicated.

    Each rank builds a local optimizer (an instance of `optimizer_cls`) over
    only the ~1/world_size parameters it owns. `step()` runs the local step on
    the owned shard and then broadcasts every parameter from its owner, so all
    replicas end the step with identical weights — the same values a single
    non-sharded `optimizer_cls` would have produced.

    The broadcast happens INSIDE `step()` (not in a separate sync method)
    because a parameter is only safe to share once its update is final; tying
    the broadcast to the step keeps the training loop identical to the
    non-sharded one: backward -> (DDP grad sync) -> step().
    """

    def __init__(self, params: Iterable, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        # Rank/world bookkeeping. Single-process fallback (world_size=1) keeps
        # the code path identical — every param is "owned" by the only rank and
        # the broadcasts are skipped.
        if _dist_ready():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank, self._world_size = 0, 1

        self._optimizer_cls = optimizer_cls
        # kwargs are the LOCAL optimizer's defaults (lr, betas, eps, ...). We
        # forward them untouched so the local optimizer's update math is
        # byte-identical to the non-sharded reference.
        self._local_kwargs = dict(kwargs)
        # Param-group dicts ({"params": [owned shard], ...}) accumulated by our
        # add_param_group() while super().__init__ runs; consumed at the end of
        # __init__ to build the local optimizer in one shot.
        self._local_param_groups: list[dict] = []
        # The local (shard) optimizer. None until the end of __init__; also
        # None forever on a rank that owns no parameters at all (only possible
        # when world_size > number of unique params — degenerate, but handled).
        self._local_optimizer: torch.optim.Optimizer | None = None
        # Flat list of (param, owner_rank) for every unique param, in the
        # deterministic iteration order used for ownership — rebuilt by
        # add_param_group. step() walks this list to broadcast.
        self._owned_plan: list[tuple[torch.Tensor, int]] = []

        # torch.optim.Optimizer.__init__(params, defaults) normalizes `params`
        # (tensor-iterable or list-of-dicts) into param groups and calls
        # self.add_param_group(group) for each — OUR override below, which is
        # where the sharding happens. We pass defaults={} because the real
        # defaults live in the local optimizer (forwarded via **kwargs); giving
        # the base class the kwargs too would only duplicate them into
        # self.param_groups entries we never read for stepping.
        super().__init__(params, defaults={})

        # Now that every initial group has been sharded by add_param_group,
        # build the local optimizer over exactly our shard of each group.
        if self._local_param_groups:
            self._local_optimizer = optimizer_cls(self._local_param_groups, **self._local_kwargs)
            # Drop the staging list; later add_param_group() calls forward
            # straight into the live local optimizer.
            self._local_param_groups = []

    # ------------------------------------------------------------------
    # Sharding
    # ------------------------------------------------------------------
    def add_param_group(self, param_group: dict) -> None:
        """
        Add a param group to BOTH the full (replicated) bookkeeping and the
        local (sharded) optimizer.

        Called by torch.optim.Optimizer.__init__ for each initial group, and by
        user code when adding groups later (e.g. a newly-unfrozen layer). The
        re-sharding therefore happens automatically in both cases — a new
        group's params are assigned owners by continuing the same round-robin,
        and the local optimizer grows a matching shard group.

        WHY keep the FULL group in self.param_groups at all? Because
        torch.optim.Optimizer.zero_grad() iterates self.param_groups — and the
        training loop's `sharded_optimizer.zero_grad()` must clear gradients on
        EVERY parameter of this rank's replica, not just the owned shard.
        """
        super().add_param_group(param_group)  # validates + appends the FULL group to self.param_groups
        full_params = self.param_groups[-1]["params"]

        # Dedupe by tensor identity so a tied weight is owned (and stepped, and
        # broadcast) exactly once — docstring detail #1.
        unique_params: list[torch.Tensor] = []
        seen: set[int] = set()
        for p in full_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)

        # Continue the global round-robin so ownership stays balanced even
        # across multiple groups: the next owner index continues from the total
        # number of unique params assigned so far.
        already_assigned = len(self._owned_plan)
        local_shard: list[torch.Tensor] = []
        for i, p in enumerate(unique_params):
            owner = (already_assigned + i) % self._world_size
            self._owned_plan.append((p, owner))
            if owner == self._rank:
                local_shard.append(p)

        # Build the local optimizer's version of this group: same
        # group-specific hyperparameters (e.g. a per-group lr), but params
        # replaced by our shard. "params" itself is excluded from the copy and
        # set explicitly.
        local_group = {key: value for key, value in param_group.items() if key != "params"}
        local_group["params"] = local_shard

        if local_shard:
            if self._local_optimizer is None:
                # Still inside __init__ (the local optimizer doesn't exist
                # yet): stage the group; __init__ will construct the local
                # optimizer from all staged groups at once.
                self._local_param_groups.append(local_group)
            else:
                # Local optimizer already exists (user adding a group later):
                # re-shard immediately by forwarding the shard group.
                self._local_optimizer.add_param_group(local_group)
        # A rank owning NOTHING in this group simply skips it. If it owns
        # nothing anywhere, _local_optimizer stays None and step() becomes pure
        # broadcast (see step). This also avoids passing an empty param list to
        # optimizer_cls, which raises ValueError("empty parameter list").

    # ------------------------------------------------------------------
    # Step + broadcast
    # ------------------------------------------------------------------
    def step(self, closure=None, **kwargs):  # noqa: D102 — see class docstring
        """
        Update the owned shard locally, then broadcast every parameter from its
        owner rank so all replicas converge to the same weights.

        WHY broadcast after step (and not e.g. all-gather or all-reduce)?
        After the local step, the owner rank holds THE definitive new value of
        each of its params — computed with the owner's moments. Other ranks
        must ADOPT that value verbatim (averaging across ranks would be wrong:
        non-owners never computed an update). Broadcast is exactly "one rank
        sends, everyone copies", which is the semantics we need.

        Cost note (tradeoff): the broadcasts move the FULL parameter set over
        the network once per step — the same order of bytes as DDP's gradient
        all-reduce. So sharded-optimizer training is NOT cheaper in
        communication than plain DDP; its win is MEMORY (moments and step
        compute shrink by world_size). We issue one broadcast per tensor for
        clarity; a production implementation would co-launch them async or
        flatten+bucket them like FlatDDP to amortize per-call overhead.
        """
        if self._local_optimizer is not None:
            # closure is forwarded for API completeness (LBFGS-style optimizers
            # need it). AdamW never uses it — and note a closure would be
            # semantically tricky under sharding anyway, since a closure
            # usually needs the full loss, while each rank only steps a shard.
            self._local_optimizer.step(closure=closure, **kwargs)

        if self._world_size == 1:
            return  # nobody to broadcast to

        # Broadcast every unique param from its owner. This also covers params
        # whose owner computed no update this step (e.g. requires_grad=False
        # frozen params): broadcasting them is harmless (they are identical
        # everywhere) and keeps the collective ORDER identical on all ranks —
        # skipping a param on one rank but not another would deadlock the
        # matching broadcast pair.
        for param, owner in self._owned_plan:
            dist.broadcast(param.data, src=owner)

    # ------------------------------------------------------------------
    # State dict — checkpointing belongs to the LOCAL (shard) optimizer
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:  # noqa: D102
        # The moments only exist inside the local optimizer (that's the whole
        # point of sharding), so the meaningful state to checkpoint is the local
        # one. Each rank checkpoints its own shard; restoring mirrors this.
        # (The base-class state_dict would report empty state — self.state on
        # the wrapper is never populated.)
        if self._local_optimizer is None:
            return {"state": {}, "param_groups": []}
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: D102
        if self._local_optimizer is None:
            return
        self._local_optimizer.load_state_dict(state_dict)
