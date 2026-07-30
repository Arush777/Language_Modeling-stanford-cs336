# Assignment 2 problem tracker

Mark items as you finish. Answers go in **writeup.pdf** (typeset). Code goes in `cs336_systems/` + `tests/adapters.py`.

## Part A — Profiling (~18 pts)

- [ ] `benchmarking_script` — warmup + timed fwd/bwd/step; `torch.cuda.synchronize()`
- [ ] `nsys_profile` — Nsight on fwd/bwd/AdamW; delete huge `.nsys-rep` after notes
- [ ] `mixed_precision_accumulation`
- [ ] `benchmarking_mixed_precision`
- [ ] `memory_profiling`

## Part B — Checkpointing (~4 pts)

- [ ] `gradient_checkpointing`

## Part C — Attention / FlashAttention (~29 pts)

- [ ] `pytorch_attention`
- [ ] `torch_compile`
- [ ] `flash_forward` (PyTorch + Triton) — adapters
- [ ] `flash_backward`
- [ ] `flash_benchmarking`

## Part D — DDP (~16+ pts)

- [ ] `distributed_communication_single_node`
- [ ] `naive_ddp`
- [ ] `naive_ddp_benchmarking`
- [ ] `minimal_ddp_flat_benchmarking`
- [ ] `ddp_overlap_individual_parameters` — adapters
- [ ] `ddp_overlap_individual_parameters_benchmarking`

## Part E — Sharding / FSDP (~40 pts)

- [ ] `optimizer_state_sharding` — adapter
- [ ] `optimizer_state_sharding_accounting`
- [ ] `fsdp` — adapters
- [ ] `fsdp_accounting`

## Part F — Written calcs (~17 pts)

- [ ] `alternate_ring_all_reduce`
- [ ] `data_parallel_calcs`
- [ ] `fsdp_calcs`
- [ ] `tp_calcs`
- [ ] `fsdp_tp_calcs`

## Part G — Leaderboard (optional, 10 pts)

- [ ] `leaderboard` — 2×B200, batch 2, empty compile/Triton cache, beat ~10 s

## First-principles reminders

1. **Measure before optimize** — sync CUDA; warm up; nsys for *where* time goes.
2. **Attention** — naive `N×N` scores are HBM-bound; FlashAttention tiles + online softmax.
3. **Checkpointing** — trade FLOPs to shrink activation memory.
4. **DDP** — replicate params; average grads; overlap all-reduce with backward.
5. **Optimizer shard / FSDP** — stop replicating optimizer (and params) on every rank.
