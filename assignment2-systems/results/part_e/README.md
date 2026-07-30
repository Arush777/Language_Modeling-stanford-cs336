# Part E — Optimizer state sharding + FSDP results

**Job `1412621`** on 2× A100-80GB. Pytest **6/6 twice** (gloo/CPU via `CUDA_VISIBLE_DEVICES=`).

## Sharded optimizer (`sharded_optimizer_bench_*_latest.csv`)

xl · B=2/rank · T=256 · AdamW · peak **allocated** GiB:

| variant | init | before_step | after_step | step_ms | opt_ms |
|---------|-----:|------------:|-----------:|--------:|-------:|
| full | 12.82 | 50.93 | 51.41 | 976 | 290 |
| **sharded** | 12.82 | **38.43** | **38.91** | **901** | **215** |

**Why these numbers**

- **Init ~12.8 GiB:** ~3.41B fp32 params ≈ 12.7 GiB (weights only).
- **Full before/after ~51 GiB:** weights + grads + Adam m/v (≈ 4× weights in the usual accounting; measured peak ~51 matches “weights+grads+2 moments” with allocator overhead).
- **Sharded ~39 GiB:** each rank keeps Adam state for ~½ the params → saves ~12 GiB of moments; grads still full on each rank in this ZeRO-1-style design (we only shard optimizer state, not grads/weights).
- **Sharded step slightly faster here:** local Adam touches fewer tensors; broadcast cost did not dominate at this size.

## FSDP (`fsdp_bench_latest.csv`)

| metric | value |
|--------|------:|
| init GiB | 12.86 |
| before_step GiB | 37.71 |
| after_step GiB | **25.82** |
| step_ms | 1006 |
| fwd / bwd / sync / opt ms | 314 / 538 / 7.6 / 146 |

**Why**

- Weights are **sharded**, so steady-state param memory is ~½; all-gather temporarily rebuilds full layers during fwd/bwd (`before_step` higher).
- **After step lower:** gathered full weights freed; optimizer steps **shards** only → less opt state than full AdamW.
- `sync_ms` (~8 ms) is mostly replicated RMSNorm all-reduces (exposed); weight AG/RS largely sits inside fwd/bwd.

## First-principles ladder

```
DDP:          replicate weights + grads + opt     (Part D)
ZeRO-1 lite:  replicate weights + grads; shard opt (sharded optimizer)
FSDP:         shard weights + grads + opt          (this part)
```
