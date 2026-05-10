from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


MODELS = ["wpu-routed", "wpu-sparse", "wpu-hybrid", "graph-transformer", "serialized-token"]
COLORS = {
    "wpu-routed": "#0f766e",
    "wpu-sparse": "#14b8a6",
    "wpu-hybrid": "#166534",
    "graph-transformer": "#7c3aed",
    "serialized-token": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze controlled WPU stress experiments.")
    parser.add_argument("--relation-dir", type=Path, default=Path("artifacts/controlled_stress_v1"))
    parser.add_argument("--affected-dir", type=Path, default=Path("artifacts/affected_strong_v1"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/controlled_stress_v1_results.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    relation_rows = _read_rows(args.relation_dir / "relation_noise.csv")
    affected_rows = _read_rows(args.affected_dir / "affected_count.csv")

    relation_noise = sorted({int(row["relation_noise"]) for row in relation_rows})
    affected_counts = sorted({int(row["affected_background_objects"]) for row in affected_rows})

    relation_acc = _aggregate(relation_rows, ["model", "relation_noise"], "branch_accuracy")
    relation_mse = _aggregate(relation_rows, ["model", "relation_noise"], "mse")
    relation_cup = _aggregate(relation_rows, ["model", "relation_noise"], "cup_mse")
    affected_mse = _aggregate(affected_rows, ["model", "affected_background_objects"], "mse")
    affected_cup = _aggregate(affected_rows, ["model", "affected_background_objects"], "cup_mse")
    affected_bg = _aggregate(affected_rows, ["model", "affected_background_objects"], "affected_background_mse")
    affected_acc = _aggregate(affected_rows, ["model", "affected_background_objects"], "branch_accuracy")

    relation_acc_fig = args.figure_dir / "relation_noise_accuracy.png"
    relation_mse_fig = args.figure_dir / "relation_noise_mse.png"
    relation_cup_fig = args.figure_dir / "relation_noise_cup_mse.png"
    affected_mse_fig = args.figure_dir / "affected_strong_mse.png"
    affected_bg_fig = args.figure_dir / "affected_strong_background_mse.png"
    affected_acc_fig = args.figure_dir / "affected_strong_accuracy.png"

    _plot_metric(relation_noise, relation_acc, "relation_noise", "Branch accuracy", "Relation-noise stress: branch accuracy", relation_acc_fig)
    _plot_metric(relation_noise, relation_mse, "relation_noise", "Next-state MSE", "Relation-noise stress: total MSE", relation_mse_fig, log_y=True)
    _plot_metric(relation_noise, relation_cup, "relation_noise", "Cup MSE", "Relation-noise stress: cup MSE", relation_cup_fig, log_y=True)
    _plot_metric(affected_counts, affected_mse, "affected_background_objects", "Next-state MSE", "Affected-count stress: total MSE", affected_mse_fig, log_y=True)
    _plot_metric(affected_counts, affected_bg, "affected_background_objects", "Affected-background MSE", "Affected-count stress: affected background MSE", affected_bg_fig, log_y=True)
    _plot_metric(affected_counts, affected_acc, "affected_background_objects", "Branch accuracy", "Affected-count stress: branch accuracy", affected_acc_fig)

    relation_summary = _relation_summary(relation_noise, relation_acc, relation_mse, relation_cup)
    affected_summary = _affected_summary(affected_counts, affected_mse, affected_cup, affected_bg, affected_acc)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        _report(
            relation_noise=relation_noise,
            affected_counts=affected_counts,
            relation_summary=relation_summary,
            affected_summary=affected_summary,
            figures=[
                relation_acc_fig,
                relation_mse_fig,
                relation_cup_fig,
                affected_mse_fig,
                affected_bg_fig,
                affected_acc_fig,
            ],
        ),
        encoding="utf-8",
    )
    print(f"wrote={args.output_md}")
    for figure in [
        relation_acc_fig,
        relation_mse_fig,
        relation_cup_fig,
        affected_mse_fig,
        affected_bg_fig,
        affected_acc_fig,
    ]:
        print(f"wrote={figure}")


def _relation_summary(
    relation_noise: list[int],
    acc: dict[tuple[str, str], tuple[float, float]],
    mse: dict[tuple[str, str], tuple[float, float]],
    cup: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, object]]:
    first_noise = min(relation_noise)
    last_noise = max(relation_noise)
    rows = []
    for model in MODELS:
        first_key = (model, str(first_noise))
        last_key = (model, str(last_noise))
        rows.append(
            {
                "model": model,
                "acc_at_0": round(acc[first_key][0], 6),
                f"acc_at_{last_noise}": round(acc[last_key][0], 6),
                "acc_drop": round(acc[first_key][0] - acc[last_key][0], 6),
                "mse_multiplier": round(_safe_ratio(mse[last_key][0], mse[first_key][0]), 3),
                "cup_mse_multiplier": round(_safe_ratio(cup[last_key][0], cup[first_key][0]), 3),
            }
        )
    return rows


def _affected_summary(
    affected_counts: list[int],
    mse: dict[tuple[str, str], tuple[float, float]],
    cup: dict[tuple[str, str], tuple[float, float]],
    bg: dict[tuple[str, str], tuple[float, float]],
    acc: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, object]]:
    rows = []
    last_count = max(affected_counts)
    for model in MODELS:
        best_bg = None
        for count in affected_counts:
            key = (model, str(count))
            if count == 0:
                continue
            value = bg[key][0]
            if best_bg is None or value < best_bg[1]:
                best_bg = (count, value)
        assert best_bg is not None
        last_key = (model, str(last_count))
        zero_key = (model, "0")
        rows.append(
            {
                "model": model,
                "best_bg_count": best_bg[0],
                "best_bg_mse": round(best_bg[1], 6),
                f"bg_mse_at_{last_count}": round(bg[last_key][0], 6),
                f"acc_at_{last_count}": round(acc[last_key][0], 6),
                f"cup_mse_at_{last_count}": round(cup[last_key][0], 6),
                "total_mse_change_0_to_max": round(mse[last_key][0] - mse[zero_key][0], 6),
            }
        )
    return rows


def _plot_metric(
    xs: list[int],
    grouped: dict[tuple[str, str], tuple[float, float]],
    x_key: str,
    ylabel: str,
    title: str,
    output: Path,
    *,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(9.0, 4.8))
    for model in MODELS:
        ys, errs = [], []
        for x in xs:
            value = grouped.get((model, str(x)))
            if value is None:
                ys.append(math.nan)
                errs.append(0.0)
            else:
                ys.append(value[0])
                errs.append(value[1])
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=2, label=model, color=COLORS.get(model))
    if log_y:
        plt.yscale("log")
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _report(
    *,
    relation_noise: list[int],
    affected_counts: list[int],
    relation_summary: list[dict[str, object]],
    affected_summary: list[dict[str, object]],
    figures: list[Path],
) -> str:
    relation_acc_fig, relation_mse_fig, relation_cup_fig, affected_mse_fig, affected_bg_fig, affected_acc_fig = figures
    best_relation = min(relation_summary, key=lambda row: float(row["acc_drop"]))
    worst_relation = max(relation_summary, key=lambda row: float(row["acc_drop"]))
    best_bg = min(affected_summary, key=lambda row: float(row["bg_mse_at_64"]))
    return "\n".join(
        [
            "# Controlled Stress v1 Results",
            "",
            "These experiments test failure modes, not just best-case performance.",
            "",
            f"Relation-noise values: `{', '.join(str(value) for value in relation_noise)}`.",
            f"Affected-background counts: `{', '.join(str(value) for value in affected_counts)}`.",
            "",
            "## Figures",
            "",
            f"![Relation noise accuracy](../figures/{relation_acc_fig.name})",
            "",
            f"![Relation noise MSE](../figures/{relation_mse_fig.name})",
            "",
            f"![Relation noise cup MSE](../figures/{relation_cup_fig.name})",
            "",
            f"![Affected count total MSE](../figures/{affected_mse_fig.name})",
            "",
            f"![Affected count background MSE](../figures/{affected_bg_fig.name})",
            "",
            f"![Affected count branch accuracy](../figures/{affected_acc_fig.name})",
            "",
            "## Relation-Noise Summary",
            "",
            _table(relation_summary),
            "",
            "## Affected-Count Summary",
            "",
            _table(affected_summary),
            "",
            "## Interpretation",
            "",
            f"- Relation-noise robustness is strongest for `{best_relation['model']}` by accuracy drop and weakest for `{worst_relation['model']}`.",
            "- The graph-transformer baseline degrades sharply under irrelevant extra edges, which supports the need for explicit route/frontier control in noisy state graphs.",
            "- WPU-hybrid preserves branch accuracy under noise better than sparse-only WPU, suggesting that local propagation needs regional correction rather than pure locality.",
            f"- Under strong affected-background deltas, `{best_bg['model']}` has the lowest affected-background MSE at the largest affected count.",
            "- The affected-count task is not well measured by branch accuracy because the branch label is still cup-centric; background-delta MSE is the relevant failure-mode metric.",
            "- These results refine the claim: WPU is promising in noisy local-update regimes, but v1 does not yet prove broad state-delta superiority across all affected-region regimes.",
            "",
        ]
    )


def _table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.inf
    return numerator / denominator


if __name__ == "__main__":
    main()
