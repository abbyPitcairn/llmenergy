"""
analyze_results.py
==================
Reads a results CSV, computes per-column averages, and prints the
top-k highest and lowest rows for each numeric column.

Usage:
    python analyze_results.py --input Results/my_results.csv --k 5
"""

import argparse
import csv
import os
from collections import defaultdict

def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def parse_numeric(rows: list[dict], skip_cols: set[str]) -> dict[str, list[tuple]]:
    """
    Returns a dict: col -> list of (value, prompt_id) for all rows
    where the value is a valid float.
    """
    col_data = defaultdict(list)
    for row in rows:
        prompt_id = row.get("prompt_id", "?")
        for col, val in row.items():
            if col in skip_cols:
                continue
            try:
                col_data[col].append((float(val), prompt_id))
            except (ValueError, TypeError):
                pass  # skip empty or non-numeric
    return col_data

def compute_averages(col_data: dict[str, list[tuple]]) -> dict[str, float]:
    return {
        col: sum(v for v, _ in pairs) / len(pairs)
        for col, pairs in col_data.items()
        if pairs
    }

def top_k(col_data: dict[str, list[tuple]], k: int):
    results = {}
    for col, pairs in col_data.items():
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        results[col] = {
            "lowest":  sorted_pairs[:k],
            "highest": sorted_pairs[-k:][::-1],
        }
    return results

def print_report(averages: dict, topk: dict, k: int):
    cols = list(averages.keys())

    # ── Averages ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COLUMN AVERAGES")
    print("="*60)
    for col in cols:
        print(f"  {col:<25}  {averages[col]:.6g}")

    # ── Top-k per column ──────────────────────────────────────────────────────
    for col in cols:
        print("\n" + "="*60)
        print(f"  {col}")
        print("="*60)

        print(f"  Top {k} HIGHEST:")
        for rank, (val, pid) in enumerate(topk[col]["highest"], 1):
            print(f"    {rank}. prompt_id={pid:<6}  {val:.6g}")

        print(f"  Top {k} LOWEST:")
        for rank, (val, pid) in enumerate(topk[col]["lowest"], 1):
            print(f"    {rank}. prompt_id={pid:<6}  {val:.6g}")

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM energy results CSV.")
    parser.add_argument("--input",  "-i", required=True, help="Path to results CSV")
    parser.add_argument("--k",      "-k", type=int, default=5, help="Top-k to display (default: 5)")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save averages as CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = load_csv(args.input)
    print(f"==> Loaded {len(rows)} rows from {args.input}")

    # prompt_id is an identifier, not a metric
    skip_cols = {"prompt_id"}

    col_data = parse_numeric(rows, skip_cols)
    averages = compute_averages(col_data)
    topk     = top_k(col_data, args.k)

    print_report(averages, topk, args.k)

    # ── Optionally save averages to CSV ───────────────────────────────────────
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["column", "average"])
            for col, avg in averages.items():
                writer.writerow([col, avg])
        print(f"\n==> Averages saved to {args.output}")

if __name__ == "__main__":
    main()