# CS336 Assignment 2 (Systems) — what I actually did

This is my student write-up for Assignment 2. The official handout is still
[`cs336_assignment2_systems.pdf`](./cs336_assignment2_systems.pdf). Code lives in
`cs336_systems/`; numbers live under `results/`. I ran everything on IBM CCC
(**A100-80GB**, LSF), not on a laptop GPU.

I am **not** doing the optional leaderboard (Part G) for now.

---

## 1. What was the problem, and what did I do?

### The big picture (first principles)

In Assignment 1 we cared about *whether the model is correct*. Assignment 2 asks:
**why is training slow or out-of-memory, and what do we change in the *system*?**

A training step is not “one magic GPU op.” It is a pipeline:

1. **Move bytes** (HBM ↔ SRAM, GPU ↔ GPU).
2. **Do FLOPs** (matmuls, attention, Adam).
3. **Keep state** (activations, weights, gradients, optimizer moments).

If any of those three is the bottleneck, optimizing the wrong one does nothing.
So the assignment walks that ladder on purpose:

| Part | Question in plain words |
|------|-------------------------|
| **A** | Where does time and memory go in a plain Transformer step? |
| **B** | Can I trade extra compute for less activation memory (checkpointing)? |
| **C** | Why is attention specially bad for memory/bandwidth, and can FlashAttention / Triton fix the IO story? |
| **D** | How do I train with a bigger batch using **multiple GPUs** without changing the math? (DDP) |
| **E** | The model still doesn’t fit — what if I **don’t replicate** optimizer state / weights on every GPU? (ZeRO-ish + FSDP) |
| **F** | On paper: when does communication beat compute for DP / FSDP / TP / both? |

Staff give you `cs336_basics` (the A1 model). I implemented the systems pieces in
`cs336_systems/`, wired `tests/adapters.py`, and submitted LSF jobs
(`scripts/job_part_*.sh`) so timing is on real A100s.

### What I implemented (short)

- **Part A:** profiling harness, FLOPs accounting, mixed precision notes, plots under `results/part_a/`.
- **Part B:** activation checkpointing + peak-memory sweeps on **xl** (`results/part_b/`).
- **Part C:** tiled FlashAttention in PyTorch (learning version), Triton forward + compiled PyTorch backward, benches; fixed a **d=128 shared-memory** tile bug and merged those rows (`flash_bench_merged_latest.csv`).
- **Part D:** naive / flat / overlapping DDP; all-reduce microbench; xl step timings on **2 GPUs**.
- **Part E:** sharded AdamW-style optimizer + FSDP over `Linear`/`Embedding`; memory + step benches on **2 GPUs**.
- **Part F:** written answers in [`writeup/PART_F.md`](./writeup/PART_F.md).

What I did *differently* from “just call `torch.nn.parallel.DistributedDataParallel`”:
I wrote the collectives myself (broadcast, all-reduce, all-gather, reduce-scatter),
with student comments on *why* AVG vs SUM, tied weights, and when overlap helps.
Same idea for FlashAttention: first a slow-but-clear PyTorch tile loop, then a fused Triton kernel.

---

## 2. What results did I get?

Pointers: [`results/RESULTS.md`](./results/RESULTS.md). Highlights only.

### Part A — profiling

BF16 was often much faster than FP32 on the A100. Very large configs (e.g. 10B,
some xl train at long T) **OOM** — that is expected: activations grow with batch × seq × depth.
Warmup matters; zero warmup timings were noisy.

### Part B — checkpointing

On xl (B=4, T=2048), **no checkpointing OOMed** in places where segment checkpointing
fit. Best non-nested setting was aggressive segmenting (e.g. segment size 1 ≈ **~38 GiB**
fwd+bwd in our sweep). Checkpointing is literally: don’t save every activation; recompute
on backward. More FLOPs, less HBM.

### Part C — attention / Flash

- **Educational PyTorch Flash** scales like **seq²** in wall time (Python tile loop). At seq=8192 it was tens of seconds per call.
- **Triton Flash** stayed roughly **milliseconds** for the same shapes — because tiles stay in **SRAM** and you don’t write the full N×N score matrix to HBM in the forward.
- At **d=128**, first Triton launch used 64×64 tiles and hit `OutOfResources` (~181 KB asked vs ~167 KB shared mem). Cap tiles at 32×32 for large D; patch job filled those cells. Use **`results/part_c/flash_bench_merged_latest.csv`**, not the raw main CSV alone.

### Part D — DDP (2× A100, xl, B=2/rank, T=256)

| Variant | step_ms | exposed comm_ms |
|---------|--------:|----------------:|
| **flat** (one big all-reduce) | **~1308** | ~168 |
| naive (per-parameter all-reduce) | ~1535 | ~210 |
| overlap (async during backward) | ~1608 | **~24** |

All-reduce bandwidth: tiny messages are latency-bound (~few GB/s effective); 1 GiB payloads got ~**177 GB/s** algorithmic bandwidth on 2 GPUs.

**How to read overlap:** exposed `comm_ms` drops a lot, but our timer `synchronize`s after backward, so overlapped NCCL work shows up inside `bwd_ms`. For *this* xl/B/T, **flat still won end-to-end** — one fat collective beat many small ones, and there wasn’t enough compute to hide everything usefully.

### Part E — sharded optimizer + FSDP (2× A100, xl)

| Setup | peak after step (GiB) | step_ms |
|-------|----------------------:|--------:|
| Full AdamW (replicated opt state) | ~51.4 | ~976 |
| **Sharded optimizer** | **~38.9** | **~901** |
| **FSDP** | **~25.8** | ~1006 |

Init is ~12.8 GiB ≈ fp32 weights for ~3.4B params. Full Adam adds grads + two moments. Sharding moments across 2 ranks cuts that redundancy. FSDP also shards weights, so steady memory is lowest (all-gather is temporary during the layer).

Pytest for sharded opt + FSDP: **6/6**, twice.

### Part F — paper calcs

Ring all-reduce variants, and when DP / FSDP / TP / 2D become **communication-bound**
(in terms of \(B, D, D_\mathrm{FF}, C, W, N\)). Written out in `writeup/PART_F.md`.

---

## 3. What did I learn? (first principles)

### Memory is not one number

“Does it fit?” means asking **what is alive at the peak**:

- **Weights** — always there (unless sharded).
- **Activations** — grow with batch × sequence × layers; checkpointing attacks this.
- **Gradients** — roughly another copy of weights (for dense layers).
- **Optimizer state** — Adam’s m and v are often **two more fp32 copies**.

DDP fixes *throughput / batch size* but **copies all four** to every GPU. That is why xl + Adam on 2 GPUs still sits near ~50 GiB before FSDP.

### Bandwidth vs FLOPs

GPUs are good at math and often **starve on bytes**. Attention’s naïve \(N\times N\) score matrix is an HBM story: FlashAttention’s win is keeping the working set in on-chip SRAM and streaming K/V — not “a faster softmax formula.” My slow PyTorch tiled version taught the algorithm; Triton taught why fusion matters.

### Communication has a shape

- **Many small all-reduces** → launch/latency tax (naive DDP).
- **One flattened all-reduce** → less tax (flat DDP) — won my xl bench.
- **Overlap** → hide bytes behind backward FLOPs; only helps when there is enough compute *and* you measure the right thing (don’t confuse “exposed wait” with “total step”).

On paper (Part F), you become communication-bound when \(T_\mathrm{comm} > T_\mathrm{comp}\). Scaling \(N\) GPUs shrinks local FLOPs as \(1/N\) but communication often stays \(\sim (N-1)/N \approx 1\) times the payload — so beyond a point, more GPUs just wait on the network.

### Correctness is part of systems

Averaging gradients (SUM then `/ world_size`), broadcasting rank-0 weights at init, and **not double-reducing tied weights** are easy to get wrong. If those are wrong, “faster” training is training a different model.

### Practical cluster lessons

- Never run GPU work on the login node.
- Keep venv/caches on `/tmp` when home quota is tight (`scripts/env.sh`).
- A failed pytest on a GPU node can be the **harness** (gloo + visible CUDA + exclusive GPU), not your algorithm — force CPU for gloo tests when needed.

### One sentence I would keep

**Optimize the scarce resource:** if you’re HBM-bound, checkpoint or shard; if you’re bandwidth-bound in attention, fuse/tile; if you’re multi-GPU bound, reduce collective count or hide them — and always measure with warmup and CUDA sync so the story matches the timeline.

---

## Where things live

```
assignment2-systems/
├── cs336_systems/     # implementations + benches
├── scripts/job_part_*.sh
├── results/part_{a,b,c,d,e}/
├── writeup/PART_F.md
└── results/RESULTS.md
```

Setup / `uv` notes: see the original course instructions in the handout and `NOTES.md` for CCC specifics.
