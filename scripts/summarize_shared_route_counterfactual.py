from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    "sparse_accuracy",
    "dense_accuracy",
    "dense_needed_rate",
    "dense_fix_rate",
    "dense_break_rate",
    "dense_lower_loss_rate",
    "branch_disagreement_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize same-model sparse/dense route counterfactuals.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["causal_k"]].append(row)

    summaries = []
    for causal_k, group_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        summary: dict[str, object] = {
            "causal_k": causal_k,
            "seeds": len({row["seed"] for row in group_rows}),
            "samples": sum(int(row["samples"]) for row in group_rows),
        }
        for metric in METRICS:
            summary[metric] = round(mean(float(row[metric]) for row in group_rows), 6)
        summaries.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, summaries)
    print(f"wrote={args.out}", flush=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
