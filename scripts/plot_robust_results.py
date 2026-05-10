from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


KEY_MODELS = ["wpu-routed", "wpu-sparse", "wpu-hybrid", "dense-graph", "graph-transformer", "serialized-token"]
COLORS = {
    "wpu-routed": "#0f766e",
    "wpu-sparse": "#14b8a6",
    "wpu-hybrid": "#166534",
    "wpu-dense": "#94a3b8",
    "dense-graph": "#2563eb",
    "graph-transformer": "#7c3aed",
    "serialized-token": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/robust_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--max-n", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_rows = _read_rows(args.input_dir / "final_baselines.csv")
    runtime_rows = _read_rows(args.input_dir / "runtime_memory.csv")
    learning_rows = _read_rows(args.input_dir / "learning_curves.csv")
    regime_rows = _read_rows(args.input_dir / "regime_sweep.csv")
    if args.max_n is not None:
        final_rows = [row for row in final_rows if int(row["total_objects"]) <= args.max_n]
        runtime_rows = [row for row in runtime_rows if int(row["total_objects"]) <= args.max_n]
        regime_rows = [row for row in regime_rows if int(row["total_objects"]) <= args.max_n]

    _plot_accuracy_ci(final_rows, args.output_dir / "robust_accuracy_ci.png")
    _plot_accuracy_runtime(final_rows, runtime_rows, args.output_dir / "robust_accuracy_runtime.png")
    _plot_learning_curves(learning_rows, args.output_dir / "robust_learning_curves.png")
    _plot_regime_work(regime_rows, args.output_dir / "robust_regime_work.png")
    print(f"wrote={args.output_dir}")


def _plot_accuracy_ci(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _group(rows, ["model", "total_objects"], "branch_accuracy")
    ns = sorted({int(row["total_objects"]) for row in rows})
    plt.figure(figsize=(9.6, 5.0))
    for model in KEY_MODELS:
        xs, ys, errs = [], [], []
        for total_objects in ns:
            values = grouped.get((model, str(total_objects)), [])
            if values:
                xs.append(total_objects)
                ys.append(mean(values))
                errs.append(_ci95(values))
        if xs:
            plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=model, color=COLORS.get(model))
    plt.xscale("log")
    plt.xticks(ns, [str(value) for value in ns], rotation=35)
    plt.ylim(0.0, 0.85)
    plt.xlabel("Total objects N")
    plt.ylabel("Branch accuracy (mean +/- 95% CI)")
    plt.title("Robust baseline comparison over 5 seeds")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_accuracy_runtime(final_rows: list[dict[str, str]], runtime_rows: list[dict[str, str]], output: Path) -> None:
    accuracy = _aggregate(final_rows, ["model", "total_objects"], "branch_accuracy")
    runtime = _aggregate([row for row in runtime_rows if row["branches"] == "3"], ["model", "total_objects"], "ms_per_sample_forward")
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    marker_handles = []
    for total_objects, marker in [(24, "o"), (84, "s"), (204, "^")]:
        marker_handles.append(plt.Line2D([0], [0], marker=marker, color="black", linestyle="", label=f"N={total_objects}"))
        for model in KEY_MODELS:
            acc = accuracy.get((model, str(total_objects)))
            ms = runtime.get((model, str(total_objects)))
            if acc is None or ms is None:
                continue
            axis.scatter(ms[0], acc[0], marker=marker, s=80, color=COLORS.get(model), alpha=0.85)
    model_handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", label=model)
        for model, color in COLORS.items()
        if model in KEY_MODELS
    ]
    axis.set_xlabel("Forward latency (ms/sample, batch=16, CPU)")
    axis.set_ylabel("Branch accuracy")
    axis.set_title("Accuracy-runtime surface, B=3")
    axis.grid(True, alpha=0.25)
    object_legend = axis.legend(
        handles=marker_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title="Object count",
    )
    axis.add_artist(object_legend)
    axis.legend(
        handles=model_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.68),
        fontsize=7,
        title="Model",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curves(rows: list[dict[str, str]], output: Path) -> None:
    grouped = _group(rows, ["model", "step"], "branch_accuracy")
    plt.figure(figsize=(8.5, 4.8))
    for model in KEY_MODELS:
        xs, ys, errs = [], [], []
        for step in [0, 25, 50, 100, 150]:
            values = grouped.get((model, str(step)), [])
            if values:
                xs.append(step)
                ys.append(mean(values))
                errs.append(_ci95(values))
        if xs:
            plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3, label=model, color=COLORS.get(model))
    plt.xlabel("Training steps")
    plt.ylabel("Branch accuracy at N=84")
    plt.title("Learning curves over 5 seeds")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_regime_work(rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in rows if row["model"] in {"wpu-routed", "graph-transformer", "serialized-token"}]
    grouped_acc = _aggregate(selected, ["model", "total_objects", "branches"], "branch_accuracy")
    grouped_work = _aggregate(selected, ["model", "total_objects", "branches"], "routed_work_proxy")
    plt.figure(figsize=(8.5, 4.8))
    for model in ["wpu-routed", "graph-transformer", "serialized-token"]:
        xs, ys, labels = [], [], []
        for total_objects in [24, 84, 204]:
            for branches in [1, 3, 8]:
                key = (model, str(total_objects), str(branches))
                if key in grouped_acc and key in grouped_work:
                    xs.append(grouped_work[key][0])
                    ys.append(grouped_acc[key][0])
                    labels.append(f"N{total_objects}/B{branches}")
        plt.scatter(xs, ys, s=65, color=COLORS.get(model), label=model, alpha=0.85)
        for x, y, label in zip(xs, ys, labels, strict=True):
            if model == "wpu-routed":
                plt.text(x * 1.05, y, label, fontsize=7)
    plt.xscale("log")
    plt.xlabel("Routed work proxy (log)")
    plt.ylabel("Branch accuracy")
    plt.title("Regime sweep: work proxy vs accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group(rows: list[dict[str, str]], keys: list[str], metric: str) -> dict[tuple[str, ...], list[float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[metric]))
    return dict(grouped)


def _aggregate(rows: list[dict[str, str]], keys: list[str], metric: str) -> dict[tuple[str, ...], tuple[float, float]]:
    return {key: (mean(values), _ci95(values)) for key, values in _group(rows, keys, metric).items()}


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


if __name__ == "__main__":
    main()
