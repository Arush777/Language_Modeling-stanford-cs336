"""Plot Part A timing CSVs → PNG figures under results/part_a/figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/part_a/figures"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["ok"].astype(str).isin(["True", "true", "1"])].copy()
    if df.empty:
        print("no successful rows")
        return

    # mean time by model × mode (fp32 only if bf16 column exists)
    if "bf16" in df.columns:
        df_fp = df[df["bf16"].astype(str).isin(["False", "false", "0"])]
    else:
        df_fp = df

    if not df_fp.empty:
        pivot = df_fp.pivot_table(index="model", columns="mode", values="mean_s", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(10, 5), title="Part A wall time (mean s)")
        ax.set_ylabel("seconds")
        fig = ax.get_figure()
        fig.tight_layout()
        out = args.out_dir / "timings_mean_s.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    if "bf16" in df.columns and df["bf16"].nunique() > 1:
        # compare bf16 vs fp32 for forward_backward
        sub = df[df["mode"] == "forward_backward"]
        if not sub.empty:
            pivot = sub.pivot_table(index="model", columns="bf16", values="mean_s", aggfunc="mean")
            ax = pivot.plot(kind="bar", figsize=(10, 5), title="forward_backward: FP32 vs BF16")
            ax.set_ylabel("seconds")
            fig = ax.get_figure()
            fig.tight_layout()
            out = args.out_dir / "timings_bf16_vs_fp32.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")

    if "tflops_per_s" in df.columns:
        sub = df_fp[df_fp["mode"] == "forward"] if not df_fp.empty else df
        if not sub.empty:
            ax = sub.plot(x="model", y="tflops_per_s", kind="bar", legend=False, figsize=(8, 4), title="Forward TFLOP/s (analytical FLOPs / time)")
            ax.set_ylabel("TFLOP/s")
            fig = ax.get_figure()
            fig.tight_layout()
            out = args.out_dir / "forward_tflops.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
