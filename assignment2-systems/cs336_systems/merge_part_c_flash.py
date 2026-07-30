"""
Merge Part C flash_bench results: main sweep + Triton d=128 patch.

Why this exists
---------------
The long Part C job (1406758) launched *before* we fixed Triton tile sizing for
D=128. Its flash_triton rows at d=128 (seq>=256) fail with shared-memory
OutOfResources. The short patch job (1408615) re-ran flash_triton only at d=128
with the fixed tiles and succeeded for all seq × {fp32, bf16}.

This script builds one canonical CSV for writeups/plots:
  - Prefer patch rows for (impl=flash_triton, d=128)
  - Keep everything else from the main CSV (or from parsing the LSF .out log
    if the main job has not flushed its CSV yet — the old flash_bench wrote
    only at the end)

Usage (after Part C finishes):
  uv run python -m cs336_systems.merge_part_c_flash \\
    --main-csv results/part_c/flash_bench_latest.csv \\
    --triton-d128 results/part_c/flash_bench_triton_d128_latest.csv \\
    --out-csv results/part_c/flash_bench_merged_latest.csv

Or rebuild from the running/finished LSF log + patch:
  uv run python -m cs336_systems.merge_part_c_flash \\
    --from-log logs/cs336_a2_partC_1406758.out \\
    --triton-d128 results/part_c/flash_bench_triton_d128_latest.csv \\
    --out-csv results/part_c/flash_bench_merged_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = [
    "impl",
    "dtype",
    "d",
    "seq",
    "fwd_ms",
    "bwd_ms",
    "e2e_ms",
    "ok",
    "error",
    "est_peak_gib",
    "source",  # main | log | triton_d128_patch
]


def _key(row: dict) -> tuple:
    return (row["impl"], row["dtype"], int(row["d"]), int(row["seq"]))


def _is_triton_d128(row: dict) -> bool:
    return row["impl"] == "flash_triton" and int(row["d"]) == 128


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        row = {k: r.get(k) for k in FIELDNAMES if k != "source"}
        # normalize ok to bool-ish string
        if row.get("ok") is not None:
            row["ok"] = str(row["ok"])
        out.append(row)
    return out


def parse_log(path: Path) -> list[dict]:
    """
    Parse flash_bench lines from an LSF .out file.

    Matches either a timing line or ERR/OOM/SKIP after the === header.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    # dtype in log is torch.float32 / torch.bfloat16
    pat = re.compile(
        r"=== (naive|flash_pytorch|flash_triton) dtype=torch\.(\w+) d=(\d+) seq=(\d+) ===\n"
        r"(?:  fwd=([\d.]+) bwd=([\d.]+) e2e=([\d.]+) ms|"
        r"  (OOM)|"
        r"  ERR ([^\n]+)|"
        r"  SKIP ([^\n]+))",
    )
    rows: list[dict] = []
    for m in pat.finditer(text):
        impl, dtype, d, seq = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if m.group(5) is not None:
            row = {
                "impl": impl,
                "dtype": dtype,
                "d": d,
                "seq": seq,
                "fwd_ms": float(m.group(5)),
                "bwd_ms": float(m.group(6)),
                "e2e_ms": float(m.group(7)),
                "ok": "True",
                "error": None,
                "est_peak_gib": None,
            }
        else:
            err = m.group(8) or m.group(9) or m.group(10) or "unknown"
            if m.group(8):
                err = "OOM"
            elif m.group(9):
                err = m.group(9).strip()
            else:
                err = f"skip:{m.group(10).strip()}"
            row = {
                "impl": impl,
                "dtype": dtype,
                "d": d,
                "seq": seq,
                "fwd_ms": None,
                "bwd_ms": None,
                "e2e_ms": None,
                "ok": "False",
                "error": err,
                "est_peak_gib": None,
            }
        rows.append(row)
    return rows


def merge(main_rows: list[dict], patch_rows: list[dict], main_source: str) -> list[dict]:
    """
    Build canonical table: main sweep, then overwrite flash_triton@d=128 from patch.
    """
    by_key: dict[tuple, dict] = {}
    for r in main_rows:
        rr = dict(r)
        rr["source"] = main_source
        by_key[_key(rr)] = rr

    n_replaced = 0
    n_added = 0
    for r in patch_rows:
        if not _is_triton_d128(r):
            # Patch job should only contain d=128 triton; ignore anything else.
            continue
        rr = dict(r)
        rr["source"] = "triton_d128_patch"
        # Force ok string
        rr["ok"] = "True" if str(rr.get("ok")).lower() in ("true", "1") else str(rr.get("ok"))
        k = _key(rr)
        if k in by_key:
            n_replaced += 1
        else:
            n_added += 1
        by_key[k] = rr

    # Stable sort: dtype, d, seq, impl order naive < flash_pytorch < flash_triton
    impl_order = {"naive": 0, "flash_pytorch": 1, "flash_triton": 2}
    dtype_order = {"float32": 0, "bfloat16": 1}

    def sort_key(r: dict):
        return (
            dtype_order.get(r["dtype"], 9),
            int(r["d"]),
            int(r["seq"]),
            impl_order.get(r["impl"], 9),
        )

    merged = sorted(by_key.values(), key=sort_key)
    return merged, n_replaced, n_added


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in FIELDNAMES}
            w.writerow(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-csv", type=Path, default=None, help="flash_bench CSV from the long Part C job")
    p.add_argument("--from-log", type=Path, default=None, help="Parse main sweep from LSF .out instead")
    p.add_argument(
        "--triton-d128",
        type=Path,
        default=Path("results/part_c/flash_bench_triton_d128_latest.csv"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/part_c/flash_bench_merged_latest.csv"),
    )
    args = p.parse_args()

    if not args.triton_d128.exists():
        raise SystemExit(f"missing Triton d=128 patch CSV: {args.triton_d128}")

    if args.main_csv and args.from_log:
        raise SystemExit("pass only one of --main-csv or --from-log")

    if args.from_log:
        if not args.from_log.exists():
            raise SystemExit(f"missing log: {args.from_log}")
        main_rows = parse_log(args.from_log)
        main_source = "log"
        print(f"parsed {len(main_rows)} rows from log {args.from_log}")
    elif args.main_csv:
        if not args.main_csv.exists():
            raise SystemExit(
                f"missing main CSV: {args.main_csv}\n"
                "Tip: Part C's old flash_bench writes CSV only at job end. "
                "Use --from-log on the .out file until then."
            )
        main_rows = load_csv(args.main_csv)
        main_source = "main"
        print(f"loaded {len(main_rows)} rows from {args.main_csv}")
    else:
        # Auto: prefer latest CSV, else the known Part C log.
        default_csv = Path("results/part_c/flash_bench_latest.csv")
        default_log = Path("logs/cs336_a2_partC_1406758.out")
        if default_csv.exists():
            main_rows = load_csv(default_csv)
            main_source = "main"
            print(f"auto: loaded {len(main_rows)} rows from {default_csv}")
        elif default_log.exists():
            main_rows = parse_log(default_log)
            main_source = "log"
            print(f"auto: parsed {len(main_rows)} rows from {default_log}")
        else:
            raise SystemExit("no --main-csv / --from-log and no defaults found")

    patch_rows = load_csv(args.triton_d128)
    patch_ok = [r for r in patch_rows if _is_triton_d128(r) and str(r.get("ok")).lower() in ("true", "1")]
    print(f"patch: {len(patch_ok)} ok flash_triton d=128 rows from {args.triton_d128}")

    merged, n_replaced, n_added = merge(main_rows, patch_rows, main_source)
    write_csv(args.out_csv, merged)

    # Sanity: no failed flash_triton d=128 in output
    bad = [
        r
        for r in merged
        if _is_triton_d128(r) and str(r.get("ok")).lower() not in ("true", "1")
    ]
    n_t128 = sum(1 for r in merged if _is_triton_d128(r))
    print(
        f"wrote {args.out_csv}: {len(merged)} rows "
        f"(replaced {n_replaced}, added {n_added} triton@d=128); "
        f"flash_triton d=128 count={n_t128}, still_bad={len(bad)}"
    )
    if bad:
        print("WARNING: merged file still has failed flash_triton d=128 rows:")
        for r in bad:
            print(f"  {r}")
        return 1
    if n_t128 < 14:
        print(
            f"WARNING: expected 14 flash_triton d=128 rows (7 seq × 2 dtype), got {n_t128}. "
            "Main sweep may still be running — re-run this merge when Part C finishes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
