from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


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
        "## Best Accuracy By N",
        "",
        _markdown_table(_best_by(ok_rows, "total_objects_n", "branch_accuracy")),
        "",
        "## Fastest Forward Latency By N",
        "",
        _markdown_table(_best_by(ok_rows, "total_objects_n", "ms_per_sample_forward", lower_is_better=True)),
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
                "params": row.get("params", ""),
                "accuracy": row.get("branch_accuracy", ""),
                "mse": row.get("mse", ""),
                "selected_K": row.get("selected_k_mean", ""),
                "ms/sample": row.get("ms_per_sample_forward", ""),
                "cuda_mb": row.get("cuda_peak_mb", ""),
            }
        )
    return output


def _best_by(
    rows: list[dict[str, str]],
    group_key: str,
    metric: str,
    lower_is_better: bool = False,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: int(float(item[0]))):
        best = min(group_rows, key=lambda row: _num(row.get(metric))) if lower_is_better else max(group_rows, key=lambda row: _num(row.get(metric)))
        output.append(
            {
                group_key: group,
                "model": best["model"],
                metric: best.get(metric, ""),
                "accuracy": best.get("branch_accuracy", ""),
                "ms/sample": best.get("ms_per_sample_forward", ""),
                "params": best.get("params", ""),
            }
        )
    return output


def _num(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
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
