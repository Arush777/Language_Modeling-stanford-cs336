"""
Make GitHub-friendly PNG figures from Part B–E CSV results.

CSV files are the raw measurements (good for tables / re-plotting).
PNG figures are what you embed in READMEs and the writeup.

Usage (from assignment2-systems/):
  uv run python -m cs336_systems.plot_results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_part_b() -> None:
    csv = RESULTS / "part_b" / "checkpoint_latest.csv"
    if not csv.exists():
        print(f"skip part_b: missing {csv}")
        return
    df = pd.read_csv(csv)
    ok = df[df["ok"].astype(str).isin(["True", "true", "1"])].copy()
    # Prefer forward_backward mode if present
    if "mode" in ok.columns and (ok["mode"] == "forward_backward").any():
        ok = ok[ok["mode"] == "forward_backward"]
    if ok.empty:
        print("skip part_b: no successful rows")
        return

    # Label bars: strategy + segment_size
    ok["label"] = ok.apply(
        lambda r: f"{r['strategy']}"
        + (f"\nseg={int(r['segment_size'])}" if r["strategy"] == "segment" else ""),
        axis=1,
    )
    ok = ok.sort_values("peak_mem_gib")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2a9d8f" if s != "none" else "#e76f51" for s in ok["strategy"]]
    ax.bar(ok["label"], ok["peak_mem_gib"], color=colors)
    ax.set_ylabel("peak memory (GiB)")
    ax.set_title("Part B — xl checkpointing peak HBM (B=4, T=2048, fwd+bwd)")
    ax.axhline(80, color="gray", ls="--", lw=1, label="A100-80GB")
    # Note OOMs in caption via text
    n_oom = (~df["ok"].astype(str).isin(["True", "true", "1"])).sum()
    ax.text(
        0.99,
        0.98,
        f"CSV also has {n_oom} OOM row(s) (e.g. strategy=none) — not plotted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
    )
    _savefig(fig, RESULTS / "part_b" / "figures" / "checkpoint_peak_mem.png")


def plot_part_c() -> None:
    flash = RESULTS / "part_c" / "flash_bench_merged_latest.csv"
    attn = RESULTS / "part_c" / "attention_bench_latest.csv"
    out = RESULTS / "part_c" / "figures"

    if flash.exists():
        df = pd.read_csv(flash)
        df = df[df["ok"].astype(str).isin(["True", "true", "1"])].copy()
        # Focus: fp32, d=64, e2e vs seq — the story students care about
        sub = df[(df["dtype"] == "float32") & (df["d"] == 64)].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for impl, g in sub.groupby("impl"):
                g = g.sort_values("seq")
                ax.plot(g["seq"], g["e2e_ms"], marker="o", label=impl)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("sequence length")
            ax.set_ylabel("e2e time (ms, log)")
            ax.set_title("Part C — Flash vs naive (fp32, d=64, causal, B=1)")
            ax.legend()
            ax.grid(True, which="both", ls=":", alpha=0.5)
            _savefig(fig, out / "flash_e2e_vs_seq_d64_fp32.png")

        # Triton only across d for seq=4096 (show d=128 works after patch)
        tri = df[(df["impl"] == "flash_triton") & (df["dtype"] == "float32") & (df["seq"] == 4096)]
        if not tri.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            tri = tri.sort_values("d")
            ax.bar([str(d) for d in tri["d"]], tri["fwd_ms"], color="#457b9d")
            ax.set_xlabel("head dim d")
            ax.set_ylabel("forward (ms)")
            ax.set_title("Part C — Triton Flash forward @ seq=4096 (fp32)")
            _savefig(fig, out / "triton_fwd_vs_d_seq4096.png")

    if attn.exists():
        df = pd.read_csv(attn)
        df = df[df["ok"].astype(str).isin(["True", "true", "1"])].copy()
        # mem vs seq for one d
        sub = df[(df["d_model"] == 64) & (df["compiled"] == False)]
        if sub.empty:
            sub = df[df["compiled"].astype(str).isin(["False", "false", "0"])]
            sub = sub[sub["d_model"] == sub["d_model"].min()]
        if not sub.empty:
            sub = sub.sort_values("seq")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(sub["seq"], sub["mem_before_bwd_gib"], marker="o", color="#e9c46a")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("sequence length")
            ax.set_ylabel("memory before bwd (GiB)")
            ax.set_title("Part C — naive attention activation memory (B=8)")
            ax.grid(True, which="both", ls=":", alpha=0.5)
            _savefig(fig, out / "naive_attn_mem_vs_seq.png")


def plot_part_d() -> None:
    out = RESULTS / "part_d" / "figures"
    ddp = RESULTS / "part_d" / "ddp_bench_summary_latest.csv"
    ar = RESULTS / "part_d" / "all_reduce_bench_latest.csv"

    if ddp.exists():
        df = pd.read_csv(ddp)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        order = ["naive", "flat", "overlap"]
        df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
        df = df.sort_values("variant")

        axes[0].bar(df["variant"].astype(str), df["step_ms"], color=["#e76f51", "#2a9d8f", "#457b9d"])
        axes[0].set_ylabel("step time (ms)")
        axes[0].set_title("xl DDP end-to-end step")

        axes[1].bar(df["variant"].astype(str), df["comm_ms"], color=["#e76f51", "#2a9d8f", "#457b9d"])
        axes[1].set_ylabel("exposed comm wait (ms)")
        axes[1].set_title("Exposed gradient sync wait")
        fig.suptitle("Part D — DDP variants (2×A100, xl, B=2, T=256)", y=1.02)
        _savefig(fig, out / "ddp_step_and_comm.png")

        # stacked phase breakdown
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(df))
        f = df["fwd_ms"].to_numpy()
        b = df["bwd_ms"].to_numpy()
        c = df["comm_ms"].to_numpy()
        o = df["opt_ms"].to_numpy()
        ax.bar(x, f, label="fwd")
        ax.bar(x, b, bottom=f, label="bwd")
        ax.bar(x, c, bottom=f + b, label="comm (exposed)")
        ax.bar(x, o, bottom=f + b + c, label="opt")
        ax.set_xticks(x)
        ax.set_xticklabels(df["variant"].astype(str))
        ax.set_ylabel("ms")
        ax.set_title("Part D — phase breakdown (note: overlap hides comm inside bwd)")
        ax.legend()
        _savefig(fig, out / "ddp_phase_breakdown.png")

    if ar.exists():
        df = pd.read_csv(ar)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["size_mb"], df["alg_bw_gbps"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("message size (MiB)")
        ax.set_ylabel("algorithmic bandwidth (GB/s)")
        ax.set_title("Part D — NCCL all-reduce bandwidth (2 GPUs)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        _savefig(fig, out / "allreduce_bandwidth.png")


def plot_part_e() -> None:
    out = RESULTS / "part_e" / "figures"
    full = RESULTS / "part_e" / "sharded_optimizer_bench_full_latest.csv"
    shard = RESULTS / "part_e" / "sharded_optimizer_bench_sharded_latest.csv"
    fsdp = RESULTS / "part_e" / "fsdp_bench_latest.csv"

    rows = []
    for path, name in [(full, "full AdamW"), (shard, "sharded opt"), (fsdp, "FSDP")]:
        if path.exists():
            r = pd.read_csv(path).iloc[0]
            rows.append(
                {
                    "name": name,
                    "init": float(r["peak_init_gib"]),
                    "before": float(r["peak_before_step_gib"]),
                    "after": float(r["peak_after_step_gib"]),
                    "step_ms": float(r["step_ms"]),
                }
            )
    if not rows:
        print("skip part_e: no CSVs")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["init"], width=w, label="init")
    ax.bar(x, df["before"], width=w, label="before step")
    ax.bar(x + w, df["after"], width=w, label="after step")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"])
    ax.set_ylabel("peak allocated (GiB)")
    ax.set_title("Part E — memory: full AdamW vs sharded opt vs FSDP (xl, 2 GPUs)")
    ax.legend()
    _savefig(fig, out / "memory_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["name"], df["step_ms"], color=["#e76f51", "#2a9d8f", "#457b9d"])
    ax.set_ylabel("step time (ms)")
    ax.set_title("Part E — training step time")
    _savefig(fig, out / "step_time_comparison.png")


def main() -> None:
    plot_part_b()
    plot_part_c()
    plot_part_d()
    plot_part_e()
    print("done — embed PNGs in each part's README with ![caption](figures/....png)")


if __name__ == "__main__":
    main()
