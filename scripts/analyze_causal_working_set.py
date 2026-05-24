from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize causal working set experiment CSVs.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/causal_working_set_v1/n-sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/causal_working_set_v2_results.md"))
    args = parser.parse_args()

    rows = _read_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_report(args.input, rows), encoding="utf-8")
    print(f"wrote={args.output}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _render_report(path: Path, rows: list[dict[str, str]]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    aggregated = _aggregate(ok_rows)
    lines = [
        "# Causal Working Set v2 Results",
        "",
        f"Source CSV: `{path.as_posix()}`",
        "",
        "This report tests the large-`N` claim in the stricter form:",
        "",
        "```text",
        "WPU should scale with causal working set K, not total state N, when K is small and identifiable.",
        "```",
        "",
        "Short CPU runs are pipeline checks, not paper evidence. Treat accuracy",
        "near the majority baseline as inconclusive until multi-seed, longer-step,",
        "GPU-scale runs are available.",
        "",
        "## Raw Condition Summary",
        "",
        _markdown_table(_condition_rows(ok_rows)),
        "",
        "## Aggregated By Model And N",
        "",
        _markdown_table(aggregated),
        "",
        "## Best Accuracy By N",
        "",
        _markdown_table(_best_by(aggregated, "N", "accuracy_mean")),
        "",
        "## Best Accuracy By K",
        "",
        _markdown_table(_best_by(aggregated, "K", "accuracy_mean")),
        "",
        "## Fastest Forward Latency By N",
        "",
        _markdown_table(_best_by(aggregated, "N", "ms_per_sample_mean", lower_is_better=True)),
        "",
        "## Fastest Forward Latency By K",
        "",
        _markdown_table(_best_by(aggregated, "K", "ms_per_sample_mean", lower_is_better=True)),
        "",
        "## Interpretation Checklist",
        "",
        "- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.",
        "- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.",
        "- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.",
        "- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.",
        "",
    ]
    if failed_rows:
        lines.extend(["## Failed Conditions", "", _markdown_table(failed_rows), ""])
    return "\n".join(lines)


def _condition_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for row in sorted(rows, key=lambda item: (int(float(item["total_objects_n"])), item["model"])):
        output.append(
            {
                "model": row["model"],
                "N": row["total_objects_n"],
                "K": row["causal_k"],
                "distractors": row.get("adversarial_distractors", "0"),
                "interaction": row.get("interaction_mode", "standard"),
                "params": row.get("params", ""),
                "accuracy": row.get("branch_accuracy", ""),
                "majority": row.get("majority_accuracy", ""),
                "acc-majority": round(_num(row.get("branch_accuracy")) - _num(row.get("majority_accuracy")), 6),
                "mse": row.get("mse", ""),
                "selected_K": row.get("selected_k_mean", ""),
                "causal_recall": row.get("causal_recall_mean", ""),
                "sparse_ratio": row.get("sparse_ratio", ""),
                "local_dense_ratio": row.get("local_dense_ratio", ""),
                "dense_compute_ratio": row.get("dense_compute_ratio", ""),
                "ms/sample": row.get("ms_per_sample_forward", ""),
                "cuda_mb": row.get("cuda_peak_mb", ""),
            }
        )
    return output


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model"],
                row["total_objects_n"],
                row["causal_k"],
                row.get("adversarial_distractors", "0"),
                row.get("interaction_mode", "standard"),
            )
        ].append(row)
    output = []
    for (model, n_value, k_value, distractors, interaction), group in sorted(grouped.items(), key=lambda item: (int(float(item[0][1])), int(float(item[0][3])), item[0][4], item[0][0])):
        accuracy = [_num(row.get("branch_accuracy")) for row in group]
        majority = [_num(row.get("majority_accuracy")) for row in group]
        mse = [_num(row.get("mse")) for row in group]
        latency = [_num(row.get("ms_per_sample_forward")) for row in group]
        selected_k = [_num(row.get("selected_k_mean")) for row in group]
        causal_recall = [_num(row.get("causal_recall_mean")) for row in group]
        sparse_ratio = [_num(row.get("sparse_ratio")) for row in group if row.get("sparse_ratio", "") != ""]
        local_dense_ratio = [_num(row.get("local_dense_ratio")) for row in group if row.get("local_dense_ratio", "") != ""]
        dense_compute_ratio = [_num(row.get("dense_compute_ratio")) for row in group if row.get("dense_compute_ratio", "") != ""]
        selector_confidence = [_num(row.get("selector_confidence_mean")) for row in group if row.get("selector_confidence_mean", "") != ""]
        output.append(
            {
                "model": model,
                "N": n_value,
                "K": k_value,
                "distractors": distractors,
                "interaction": interaction,
                "seeds": len(group),
                "params": group[0].get("params", ""),
                "accuracy_mean": round(mean(accuracy), 6),
                "accuracy_ci95": round(_ci95(accuracy), 6),
                "majority_mean": round(mean(majority), 6),
                "acc_minus_majority": round(mean(accuracy) - mean(majority), 6),
                "mse_mean": round(mean(mse), 6),
                "selected_K_mean": round(mean(selected_k), 6),
                "causal_recall_mean": round(mean(causal_recall), 6),
                "sparse_ratio_mean": round(mean(sparse_ratio), 6) if sparse_ratio else "",
                "local_dense_ratio_mean": round(mean(local_dense_ratio), 6) if local_dense_ratio else "",
                "dense_compute_ratio_mean": round(mean(dense_compute_ratio), 6) if dense_compute_ratio else "",
                "selector_confidence_mean": round(mean(selector_confidence), 6) if selector_confidence else "",
                "ms_per_sample_mean": round(mean(latency), 6),
                "ms_per_sample_ci95": round(_ci95(latency), 6),
            }
        )
    return output


def _best_by(
    rows: list[dict[str, object]],
    group_key: str,
    metric: str,
    lower_is_better: bool = False,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    output: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: int(float(item[0]))):
        best = min(group_rows, key=lambda row: _num(row.get(metric))) if lower_is_better else max(group_rows, key=lambda row: _num(row.get(metric)))
        output.append(
            {
                group_key: group,
                "model": best["model"],
                metric: best.get(metric, ""),
                "accuracy": best.get("accuracy_mean", ""),
                "ms/sample": best.get("ms_per_sample_mean", ""),
                "params": best.get("params", ""),
            }
        )
    return output


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _num(value: object) -> float:
    try:
        return float(value or "nan")
    except (TypeError, ValueError):
        return float("nan")


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
