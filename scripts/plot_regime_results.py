from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/n_sweep_v1/regime_sweep.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--max-n", type=int, default=None)
    args = parser.parse_args()

    rows = _read(args.input)
    if args.max_n is not None:
        rows = [row for row in rows if int(row["total_objects"]) <= args.max_n]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_route_regime(rows, args.output_dir / "route_regime.png")
    _plot_accuracy_work(rows, args.output_dir / "accuracy_work_tradeoff.png")
    print(f"wrote={args.output_dir / 'route_regime.png'}")
    print(f"wrote={args.output_dir / 'accuracy_work_tradeoff.png'}")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_route_regime(rows: list[dict[str, str]], output: Path) -> None:
    routed = [row for row in rows if row["model"] == "wpu-routed"]
    branches = sorted({int(row["branches"]) for row in routed})
    fig, axes = plt.subplots(1, len(branches), figsize=(4.6 * len(branches), 3.4), sharey=True)
    if len(branches) == 1:
        axes = [axes]
    for axis, branch_count in zip(axes, branches, strict=True):
        subset = sorted(
            [row for row in routed if int(row["branches"]) == branch_count],
            key=lambda row: int(row["total_objects"]),
        )
        x = [int(row["total_objects"]) for row in subset]
        axis.plot(x, [float(row["sparse_ratio"]) for row in subset], marker="o", linewidth=2, markersize=4, label="sparse")
        axis.plot(x, [float(row["hybrid_ratio"]) for row in subset], marker="o", linewidth=2, markersize=4, label="hybrid")
        axis.plot(x, [float(row["dense_ratio"]) for row in subset], marker="o", linewidth=2, markersize=4, label="dense")
        axis.set_title(f"B={branch_count}")
        axis.set_xlabel("world objects N")
        axis.set_xscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels([str(value) for value in x], rotation=35)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("route ratio")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16)
    fig.suptitle("Routed WPU switches execution path as affected-state fraction changes")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_work(rows: list[dict[str, str]], output: Path) -> None:
    subset = [row for row in rows if int(row["branches"]) == 3]
    grouped_acc = _aggregate(subset, ["model", "total_objects"], "branch_accuracy")
    grouped_work = _aggregate(subset, ["model", "total_objects"], "routed_work_proxy")
    models = sorted({row["model"] for row in subset})
    ns = sorted({int(row["total_objects"]) for row in subset})
    colors = {
        "dense-graph": "#2563eb",
        "graph-transformer": "#f97316",
        "serialized-token": "#16a34a",
        "wpu-dense": "#dc2626",
        "wpu-hybrid": "#7c3aed",
        "wpu-routed": "#7f5539",
        "wpu-sparse": "#e879f9",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharex=True)
    for model in models:
        xs, ys, yerrs, works = [], [], [], []
        for n in ns:
            acc_key = (model, str(n))
            work_key = (model, str(n))
            if acc_key not in grouped_acc or work_key not in grouped_work:
                continue
            xs.append(n)
            ys.append(grouped_acc[acc_key][0])
            yerrs.append(grouped_acc[acc_key][1])
            works.append(grouped_work[work_key][0])
        if not xs:
            continue
        axes[0].errorbar(
            xs,
            ys,
            yerr=yerrs,
            marker="o",
            linewidth=2,
            capsize=2,
            label=model,
            color=colors.get(model),
        )
        axes[1].plot(
            xs,
            works,
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(ns)
        axis.set_xticklabels([str(value) for value in ns], rotation=35)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("world objects N")
    axes[0].set_ylim(0.0, 0.86)
    axes[0].set_ylabel("branch accuracy (mean +/- 95% CI)")
    axes[0].set_title("Accuracy at B=3")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("routed work proxy (log)")
    axes[1].set_title("Work proxy at B=3")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16)
    fig.suptitle("Accuracy and routed work across world size at B=3")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _aggregate(rows: list[dict[str, str]], keys: list[str], metric: str) -> dict[tuple[str, ...], tuple[float, float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[metric]))
    return {key: (mean(values), _ci95(values)) for key, values in grouped.items()}


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


if __name__ == "__main__":
    main()
