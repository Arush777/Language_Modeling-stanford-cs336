# Part D — Distributed Data Parallel results

**Hardware:** 2× A100-80GB, one node (job `1411276`).

## Correctness note

On-node `pytest tests/test_ddp.py` failed with `cudaErrorDevicesUnavailable` because the test harness puts tensors on CUDA whenever a GPU is visible, while gloo worker threads then call `cudaSetDevice` under `exclusive_process`. **DDP logic is fine** — the same tests pass on CPU (`CUDA_VISIBLE_DEVICES=`), which Part E’s job script now does. Benches below used NCCL on both GPUs and completed successfully.

## All-reduce microbench (`all_reduce_bench_*.csv`)

NCCL, world_size=2, FP32 payloads:

| size | mean_ms | alg_bw (GB/s) |
|------|---------|---------------|
| 1 MiB | 0.43 | ~2.5 |
| 10 MiB | 0.39 | ~27 |
| 100 MiB | 0.75 | ~141 |
| 1 GiB | 6.08 | ~177 |

Small messages are **latency-bound**; large ones approach link bandwidth.

## xl DDP step time (`ddp_bench_*_*.csv`)

Config: xl (~3.41B params), B=2/rank, T=256, AdamW, 5 warmup + 10 steps.

| variant | step_ms | comm_ms (exposed) | notes |
|---------|---------|-------------------|--------|
| **flat** | **1308** | 168 | Best end-to-end here |
| naive | 1535 | 210 | Many small all-reduces |
| overlap | 1608 | **24** | Comm mostly hidden in `bwd_ms` |

### Why overlap didn’t win `step_ms`

Overlap **does** hide communication: exposed `comm_ms` drops from ~210 ms → ~24 ms. Our timer calls `torch.cuda.synchronize()` after `backward()`, so overlapped NCCL work is **charged to `bwd_ms`** (973 ms vs ~600 ms). Flat wins on this xl/B=2/T=256 setup because **one big collective** beats many per-parameter launches, and compute isn’t large enough for overlap to beat flatten. For writeups: report both `step_ms` and `comm_ms`, and explain the sync artifact.

## Code map

- `cs336_systems/ddp.py` — `NaiveDDP`, `FlatDDP`, overlapping `DDP`
- Adapters: `get_ddp` → overlap; `ddp_on_after_backward` → `finish_gradient_synchronization()`
