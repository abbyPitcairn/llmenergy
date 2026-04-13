"""
combine_analysis.py
====================
Combines five per-run analysis CSVs into:
  1. combined_averages.csv   — average of the per-run averages, same format as input
  2. combined_rankings.csv   — all highest/lowest entries across runs, with run number,
                               sorted best-to-worst within each metric/direction
"""

import ast
import csv
import os
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "analysis"
OUTPUT_DIR  = "analysis"
NUM_RUNS    = 5
FILE_TMPL   = "gpt2_run_{:02d}_analysis.csv"

AVG_OUT      = os.path.join(OUTPUT_DIR, "combined_averages.csv")
LOW_RANK_OUT = os.path.join(OUTPUT_DIR, "lowest_rankings.csv")
HIGH_RANK_OUT= os.path.join(OUTPUT_DIR, "highest_rankings.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_run(run_num: int) -> tuple[dict, dict]:
    """Return (averages_dict, rankings_dict) for one run file."""
    path = os.path.join(INPUT_DIR, FILE_TMPL.format(run_num))
    averages  = {}   # metric -> float
    rankings  = {}   # metric -> {'lowest': [...], 'highest': [...]}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            col = row["column"]
            val = row["average"]
            # Try to parse as float (plain average row)
            try:
                averages[col] = float(val)
            except (ValueError, TypeError):
                # Must be a rankings dict string
                try:
                    rankings[col] = ast.literal_eval(val)
                except Exception as e:
                    print(f"[WARN] run {run_num}, column '{col}': could not parse value — {e}")
    return averages, rankings


# ── 1. Load all runs ──────────────────────────────────────────────────────────
all_averages = {}   # metric -> [val_run1, val_run2, ...]
all_rankings = {}   # metric -> {run_num: {'lowest': [...], 'highest': [...]}}

for run in range(1, NUM_RUNS + 1):
    avgs, rnks = load_run(run)
    for metric, val in avgs.items():
        all_averages.setdefault(metric, []).append(val)
    for metric, data in rnks.items():
        all_rankings.setdefault(metric, {})[run] = data


# ── 2. combined_averages.csv ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(AVG_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["column", "average"])
    for metric, vals in all_averages.items():
        mean = sum(vals) / len(vals)
        writer.writerow([metric, mean])

print(f"[OK] Averages written  -> {AVG_OUT}")


# ── 3. highest_rankings.csv / lowest_rankings.csv ────────────────────────────
# For each metric + direction, deduplicate by prompt_id:
#   - keep the value from its first appearance (they should be consistent across runs)
#   - collect all run numbers that prompt_id appeared in as a sorted list
# Then sort: lowest asc, highest desc, and assign rank.

def build_deduped(runs_data: dict, direction: str) -> list[dict]:
    """
    Returns a sorted, deduplicated list of dicts:
      {metric, prompt_id, value, runs, rank}
    grouped by metric.
    """
    # pid_data[prompt_id] = {'value': float, 'runs': set}
    pid_data = {}

    for run_num, data in runs_data.items():
        for val, pid in data.get(direction, []):
            pid = str(pid)
            if pid not in pid_data:
                pid_data[pid] = {"value": val, "runs": set()}
            pid_data[pid]["runs"].add(run_num)

    # Sort and assign rank
    rows = []
    reverse = (direction == "highest")
    for pid, entry in pid_data.items():
        rows.append({"prompt_id": pid, "value": entry["value"], "runs": sorted(entry["runs"])})
    rows.sort(key=lambda x: x["value"], reverse=reverse)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_rankings(path: str, direction: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "rank", "value", "prompt_id", "runs"])

        for metric, runs_data in all_rankings.items():
            rows = build_deduped(runs_data, direction)
            for row in rows:
                runs_str = ";".join(str(r) for r in row["runs"])
                writer.writerow([metric, row["rank"], row["value"], row["prompt_id"], runs_str])

    print(f"[OK] {direction.capitalize()} rankings -> {path}")


write_rankings(LOW_RANK_OUT,  "lowest")
write_rankings(HIGH_RANK_OUT, "highest")