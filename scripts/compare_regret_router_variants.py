from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


DEFAULT_INPUTS = {
    "internal": Path("docs/experiments/wpu_v2_staged_regret_hybrid_5seed.csv"),
    "physics_hidden": Path("docs/experiments/wpu_v2_physics_regret_hybrid_5seed.csv"),
    "state_only": Path("docs/experiments/wpu_v2_state_regret_hybrid_5seed.csv"),
}

METRICS = [
    "sparse_loss",
    "dense_loss",
    "routed_loss",
    "oracle_loss",
    "routed_delta_vs_sparse",
    "routed_excess_over_oracle",
    "dense_compute_ratio",
    "route_regret_eval_corr",
    "route_regret_eval_mse",
    "routed_accuracy",
    "sparse_accuracy",
    "dense_accuracy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare staged regret router variants.")
    parser.add_argument("--out-summary", type=Path, default=Path("docs/experiments/wpu_v2_regret_router_variant_summary.csv"))
    parser.add_argument("--out-paired", type=Path, default=Path("docs/experiments/wpu_v2_regret_router_variant_paired.csv"))
    args = parser.parse_args()

    rows_by_variant = {name: _read_rows(path) for name, path in DEFAULT_INPUTS.items()}
    summary_rows = _summary_rows(rows_by_variant)
    paired_rows = _paired_rows(rows_by_variant, baseline="internal")
    _write_csv(args.out_summary, summary_rows)
    _write_csv(args.out_paired, paired_rows)
    print(f"wrote={args.out_summary}")
    print(f"wrote={args.out_paired}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_rows(rows_by_variant: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant, rows in rows_by_variant.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["causal_k"]].append(row)
            grouped["overall"].append(row)
        for group, group_rows in sorted(grouped.items(), key=lambda item: _group_key(item[0])):
            record: dict[str, object] = {"variant": variant, "causal_k": group, "n": len(group_rows)}
            for metric in METRICS:
                values = [float(row[metric]) for row in group_rows]
                record[f"{metric}_mean"] = round(mean(values), 6)
                record[f"{metric}_std"] = round(pstdev(values), 6)
            output.append(record)
    return output


def _paired_rows(rows_by_variant: dict[str, list[dict[str, str]]], *, baseline: str) -> list[dict[str, object]]:
    baseline_rows = {
        (int(row["seed"]), int(row["causal_k"])): row
        for row in rows_by_variant[baseline]
    }
    output: list[dict[str, object]] = []
    for variant, rows in rows_by_variant.items():
        if variant == baseline:
            continue
        grouped_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            key = (int(row["seed"]), int(row["causal_k"]))
            base_row = baseline_rows[key]
            for metric in METRICS:
                grouped_deltas[row["causal_k"]][metric].append(float(row[metric]) - float(base_row[metric]))
                grouped_deltas["overall"][metric].append(float(row[metric]) - float(base_row[metric]))
        for group, metric_deltas in sorted(grouped_deltas.items(), key=lambda item: _group_key(item[0])):
            record: dict[str, object] = {"variant": variant, "baseline": baseline, "causal_k": group}
            for metric, values in metric_deltas.items():
                record[f"{metric}_delta_mean"] = round(mean(values), 6)
                record[f"{metric}_delta_std"] = round(pstdev(values), 6)
                record[f"{metric}_wins"] = sum(value < 0.0 for value in values)
                record[f"{metric}_n"] = len(values)
            output.append(record)
    return output


def _group_key(value: str) -> tuple[int, int]:
    return (1, 0) if value == "overall" else (0, int(value))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
