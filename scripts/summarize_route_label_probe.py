from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    "label_rate",
    "predicted_dense_rate",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "score_mean",
    "roc_auc",
    "average_precision",
    "brier_score",
    "ece",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dense-needed route-label probe CSVs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        threshold_key = "calibrated" if "_cal_" in row["model"] else row["threshold"]
        grouped[(row["model"], threshold_key, row["feature_count"])].append(row)

    summaries = []
    for (model, threshold, feature_count), group_rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "model": model,
            "threshold": threshold,
            "feature_count": feature_count,
            "seeds": len({row["test_seed"] for row in group_rows}),
            "samples": sum(int(row["samples"]) for row in group_rows),
            "threshold_mean": round(mean(float(row["threshold"]) for row in group_rows), 6),
        }
        for metric in METRICS:
            if metric in group_rows[0]:
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
