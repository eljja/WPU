from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


MODELS = [
    "wpu-cws-frontier",
    "wpu-cws-oracle",
    "wpu-cws-learned",
    "wpu-cws-indexed",
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "wpu-cws-indexed-adaptive-hybrid",
    "serialized-token",
    "graph-transformer",
]
COLORS = {
    "wpu-cws-frontier": "#0f766e",
    "wpu-cws-oracle": "#15803d",
    "wpu-cws-learned": "#14b8a6",
    "wpu-cws-indexed": "#2563eb",
    "wpu-cws-indexed-sparse": "#9333ea",
    "wpu-cws-indexed-local-dense": "#0891b2",
    "wpu-cws-indexed-adaptive-hybrid": "#ea580c",
    "serialized-token": "#dc2626",
    "graph-transformer": "#7c3aed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot causal working set experiment results.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/causal_working_set_v1_cpu/n-sweep.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--prefix", default="cws")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_accuracy(rows, args.output_dir / f"{args.prefix}_accuracy.png")
    _plot_latency(rows, args.output_dir / f"{args.prefix}_latency.png")
    _plot_recall(rows, args.output_dir / f"{args.prefix}_causal_recall.png")
    _plot_accuracy_latency(rows, args.output_dir / f"{args.prefix}_accuracy_latency.png")
    print(f"wrote={args.output_dir}")


def _plot_accuracy(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _aggregate(rows, "branch_accuracy")
    majority = _aggregate(rows, "majority_accuracy")
    ns = _n_values(rows)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for model in MODELS:
        xs, ys, errs = _series(grouped, model, ns)
        if xs:
            axis.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=model, color=COLORS.get(model))
    maj_x, maj_y, _ = _series(majority, MODELS[0], ns, fallback_any_model=True)
    if maj_x:
        axis.plot(maj_x, maj_y, linestyle="--", color="#64748b", label="majority baseline")
    _format_n_axis(axis, ns)
    axis.set_ylabel("Branch accuracy")
    axis.set_title("CWS accuracy as total world size N grows")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_latency(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _aggregate(rows, "ms_per_sample_forward")
    ns = _n_values(rows)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for model in MODELS:
        xs, ys, errs = _series(grouped, model, ns)
        if xs:
            axis.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=model, color=COLORS.get(model))
    _format_n_axis(axis, ns)
    axis.set_ylabel("Forward latency (ms/sample)")
    axis.set_title("CWS latency scaling")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_recall(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _aggregate(rows, "causal_recall_mean")
    ns = _n_values(rows)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    plotted = False
    for model in [
        "wpu-cws-frontier",
        "wpu-cws-oracle",
        "wpu-cws-learned",
        "wpu-cws-indexed",
        "wpu-cws-indexed-sparse",
        "wpu-cws-indexed-local-dense",
        "wpu-cws-indexed-adaptive-hybrid",
    ]:
        xs, ys, errs = _series(grouped, model, ns)
        if xs:
            plotted = True
            axis.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=model, color=COLORS.get(model))
    _format_n_axis(axis, ns)
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Causal working-set recall")
    axis.set_title("Selector quality as N grows")
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_accuracy_latency(rows: list[dict[str, str]], output: Path) -> None:
    accuracy = _aggregate(rows, "branch_accuracy")
    latency = _aggregate(rows, "ms_per_sample_forward")
    ns = _n_values(rows)
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for model in MODELS:
        xs, ys = [], []
        for n_value in ns:
            key = (model, str(n_value), _first_distractor(rows, n_value))
            if key in accuracy and key in latency:
                xs.append(latency[key][0])
                ys.append(accuracy[key][0])
        if xs:
            axis.plot(xs, ys, marker="o", linewidth=2, label=model, color=COLORS.get(model))
    axis.set_xlabel("Forward latency (ms/sample)")
    axis.set_ylabel("Branch accuracy")
    axis.set_title("Accuracy-latency trajectory over N")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or metric not in row or row.get(metric, "") == "":
            continue
        key = (row["model"], row["total_objects_n"], row.get("adversarial_distractors", "0"))
        grouped[key].append(float(row[metric]))
    return {key: (mean(values), _ci95(values)) for key, values in grouped.items()}


def _series(
    grouped: dict[tuple[str, str, str], tuple[float, float]],
    model: str,
    ns: list[int],
    fallback_any_model: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    xs, ys, errs = [], [], []
    for n_value in ns:
        key = (model, str(n_value), "0")
        if fallback_any_model and key not in grouped:
            key = next((candidate for candidate in grouped if candidate[1] == str(n_value) and candidate[2] == "0"), key)
        if key in grouped:
            xs.append(n_value)
            ys.append(grouped[key][0])
            errs.append(grouped[key][1])
    return xs, ys, errs


def _n_values(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(float(row["total_objects_n"])) for row in rows if row.get("status") == "ok"})


def _first_distractor(rows: list[dict[str, str]], n_value: int) -> str:
    values = sorted({row.get("adversarial_distractors", "0") for row in rows if int(float(row["total_objects_n"])) == n_value})
    return values[0] if values else "0"


def _format_n_axis(axis, ns: list[int]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(ns, [str(value) for value in ns], rotation=35)
    axis.set_xlabel("Total world objects N")


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


if __name__ == "__main__":
    main()
