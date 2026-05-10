from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


MODELS = ["wpu-routed", "wpu-sparse", "wpu-hybrid", "dense-graph", "graph-transformer", "serialized-token"]
COLORS = {
    "wpu-routed": "#0f766e",
    "wpu-sparse": "#14b8a6",
    "wpu-hybrid": "#166534",
    "dense-graph": "#2563eb",
    "graph-transformer": "#7c3aed",
    "serialized-token": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/step_sweep_v1"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/step_sweep_v1_results.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(args.input_dir / "learning_curves.csv")
    acc = _aggregate(rows, ["model", "step"], "branch_accuracy")
    ece = _aggregate(rows, ["model", "step"], "branch_ece")
    mse = _aggregate(rows, ["model", "step"], "mse")
    steps = sorted({int(row["step"]) for row in rows})

    acc_fig = args.figure_dir / "step_sweep_accuracy.png"
    ece_fig = args.figure_dir / "step_sweep_ece.png"
    mse_fig = args.figure_dir / "step_sweep_mse.png"
    _plot_metric(steps, acc, "branch_accuracy", "Branch accuracy at N=84", acc_fig)
    _plot_metric(steps, ece, "branch_ece", "Branch ECE at N=84", ece_fig)
    _plot_metric(steps, mse, "mse", "Next-state MSE at N=84", mse_fig, log_y=True)

    summary_rows = _summary_rows(steps, acc)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_report(summary_rows, acc_fig, ece_fig, mse_fig), encoding="utf-8")
    print(f"wrote={args.output_md}")
    print(f"wrote={acc_fig}")
    print(f"wrote={ece_fig}")
    print(f"wrote={mse_fig}")


def _summary_rows(steps: list[int], acc: dict[tuple[str, str], tuple[float, float]]) -> list[dict[str, object]]:
    rows = []
    for model in MODELS:
        values = [(step, acc[(model, str(step))][0]) for step in steps if (model, str(step)) in acc]
        best_step, best_acc = max(values, key=lambda item: item[1])
        final_acc = dict(values)[max(steps)]
        acc_50 = dict(values).get(50, math.nan)
        acc_150 = dict(values).get(150, math.nan)
        rows.append(
            {
                "model": model,
                "best_step": best_step,
                "best_acc": round(best_acc, 6),
                "acc_at_50": round(acc_50, 6),
                "acc_at_150": round(acc_150, 6),
                "acc_at_300": round(final_acc, 6),
                "gain_150_to_300": round(final_acc - acc_150, 6),
            }
        )
    return rows


def _plot_metric(
    steps: list[int],
    grouped: dict[tuple[str, str], tuple[float, float]],
    metric: str,
    ylabel: str,
    output: Path,
    *,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(9.0, 4.8))
    for model in MODELS:
        xs, ys, errs = [], [], []
        for step in steps:
            value = grouped.get((model, str(step)))
            if value:
                xs.append(step)
                ys.append(value[0])
                errs.append(value[1])
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=2, label=model, color=COLORS.get(model))
    if log_y:
        plt.yscale("log")
    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(f"Step sweep: {metric}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _report(summary_rows: list[dict[str, object]], acc_fig: Path, ece_fig: Path, mse_fig: Path) -> str:
    return "\n".join(
        [
            "# Dense Step Sweep v1 Results",
            "",
            "Training steps: `0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300`.",
            "",
            "Evaluation: `N=84`, `B=3`, 5 seeds, 256 samples per evaluation point.",
            "",
            "## Figures",
            "",
            f"![Step sweep accuracy](../figures/{acc_fig.name})",
            "",
            f"![Step sweep ECE](../figures/{ece_fig.name})",
            "",
            f"![Step sweep MSE](../figures/{mse_fig.name})",
            "",
            "## Summary",
            "",
            _table(summary_rows),
            "",
            "## Interpretation",
            "",
            "- Most useful learning happens before 100-150 steps on this synthetic task.",
            "- Longer training does not uniformly improve branch accuracy.",
            "- This weakens any claim based on a single training duration; results should report training curves or best-validation selection.",
            "- For future papers, step budget should be treated as an axis in the accuracy-compute surface.",
            "",
        ]
    )


def _table(rows: list[dict[str, object]]) -> str:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
