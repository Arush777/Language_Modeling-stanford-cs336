# Part C — FlashAttention / compile results

**Hardware:** 1× NVIDIA A100-SXM4-80GB (job `1406758`), plus Triton d=128 patch job `1408615`.

## Canonical file for plots/writeup

**`flash_bench_merged_latest.csv`** — main sweep with all `(flash_triton, d=128)` rows replaced by the fixed-tile patch (no `OutOfResources`).

Rebuild:
```bash
uv run python -m cs336_systems.merge_part_c_flash \
  --main-csv results/part_c/flash_bench_latest.csv \
  --triton-d128 results/part_c/flash_bench_triton_d128_latest.csv \
  --out-csv results/part_c/flash_bench_merged_latest.csv
```

## What we measured

| Artifact | Meaning |
|----------|---------|
| `attention_bench_latest.csv` | Naive SDPA ± `torch.compile`, B=8, seq up to 16384 |
| `flash_bench_merged_latest.csv` | naive / educational PyTorch FA / Triton FA, causal, B=1 |
| `pytest_flash_*.txt` | FA forward+backward tests (PyTorch + Triton) — all passed |

## Takeaways (first principles)

1. **Educational `flash_pytorch` is slow at long seq** — Python tile loops, cost ∝ seq². At seq=8192, ~tens of seconds per call; Triton stays sub‑ms to a few ms.
2. **Triton wins on IO** — keeps Q/K/V tiles in SRAM with online softmax; that is the FlashAttention point.
3. **d=128 Triton originally OOM’d shared memory** — 64×64 tiles × D=128 asked ~181 KB > ~167 KB SM limit. Fix: cap tiles at 32×32 when D≥128. Patch job re-measured those cells.
4. **Naive vs Triton** — for moderate seq, cuBLAS naive can look “fine”; at long seq / memory, fused Triton scales better and avoids materializing N×N in the forward.

## Jobs

| JobID | Role | Status |
|-------|------|--------|
| 1406758 | Full Part C | Done |
| 1408615 | Triton d=128 only | Done |
