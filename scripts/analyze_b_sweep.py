from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


NS = [24, 84, 132, 204]
COLORS = {24: "#0f766e", 84: "#2563eb", 132: "#f59e0b", 204: "#dc2626"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/b_sweep_v1"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/b_sweep_v1_results.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    regime_rows = _read_rows(args.input_dir / "regime_sweep.csv")
    runtime_rows = _read_rows(args.input_dir / "runtime_memory.csv")
    acc = _aggregate(regime_rows, ["model", "total_objects", "branches"], "branch_accuracy")
    work = _aggregate(regime_rows, ["model", "total_objects", "branches"], "routed_work_proxy")
    runtime = _aggregate(runtime_rows, ["model", "total_objects", "branches"], "ms_per_sample_forward")
    ratios = {
        name: _aggregate(regime_rows, ["model", "total_objects", "branches"], name)
        for name in ["sparse_ratio", "hybrid_ratio", "dense_ratio"]
    }

    route_fig = args.figure_dir / "b_sweep_routes.png"
    acc_fig = args.figure_dir / "b_sweep_accuracy.png"
    work_fig = args.figure_dir / "b_sweep_work_accuracy.png"
    runtime_fig = args.figure_dir / "b_sweep_runtime.png"
    _plot_routes(ratios, route_fig)
    _plot_accuracy(acc, acc_fig)
    _plot_work_accuracy(acc, work, work_fig)
    _plot_runtime(runtime, runtime_fig)

    rows = _summary_rows(acc, work, runtime, ratios)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_report(rows, route_fig, acc_fig, work_fig, runtime_fig), encoding="utf-8")
    print(f"wrote={args.output_md}")
    print(f"wrote={route_fig}")
    print(f"wrote={acc_fig}")
    print(f"wrote={work_fig}")
    print(f"wrote={runtime_fig}")


def _summary_rows(
    acc: dict[tuple[str, str, str], tuple[float, float]],
    work: dict[tuple[str, str, str], tuple[float, float]],
    runtime: dict[tuple[str, str, str], tuple[float, float]],
    ratios: dict[str, dict[tuple[str, str, str], tuple[float, float]]],
) -> list[dict[str, object]]:
    rows = []
    for n in NS:
        best = None
        route_changes = []
        previous_route = None
        for b in range(1, 17):
            key = ("wpu-routed", str(n), str(b))
            accuracy = acc[key][0]
            if best is None or accuracy > best["best_acc"]:
                best = {
                    "N": n,
                    "best_B": b,
                    "best_acc": round(accuracy, 6),
                    "best_runtime_ms": round(runtime[key][0], 6),
                    "best_work_proxy": round(work[key][0], 6),
                }
            sparse = ratios["sparse_ratio"][key][0]
            hybrid = ratios["hybrid_ratio"][key][0]
            dense = ratios["dense_ratio"][key][0]
            route = max([("sparse", sparse), ("hybrid", hybrid), ("dense", dense)], key=lambda item: item[1])[0]
            if route != previous_route:
                route_changes.append(f"B={b}:{route}")
                previous_route = route
        assert best is not None
        best["route_changes"] = ", ".join(route_changes)
        rows.append(best)
    return rows


def _plot_routes(ratios: dict[str, dict[tuple[str, str, str], tuple[float, float]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, sharey=True)
    for ax, n in zip(axes.flatten(), NS, strict=True):
        for name, color in [("sparse_ratio", "#14b8a6"), ("hybrid_ratio", "#f59e0b"), ("dense_ratio", "#ef4444")]:
            ys = [ratios[name][("wpu-routed", str(n), str(b))][0] for b in range(1, 17)]
            ax.plot(range(1, 17), ys, marker="o", linewidth=2, label=name.replace("_ratio", ""), color=color)
        ax.set_title(f"N={n}")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Branch pressure B")
    axes[1, 1].set_xlabel("Branch pressure B")
    axes[0, 0].set_ylabel("Route ratio")
    axes[1, 0].set_ylabel("Route ratio")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("B sweep: routed WPU path transitions")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_accuracy(acc: dict[tuple[str, str, str], tuple[float, float]], output: Path) -> None:
    plt.figure(figsize=(9.0, 4.8))
    for n in NS:
        ys = [acc[("wpu-routed", str(n), str(b))][0] for b in range(1, 17)]
        errs = [acc[("wpu-routed", str(n), str(b))][1] for b in range(1, 17)]
        plt.errorbar(range(1, 17), ys, yerr=errs, marker="o", capsize=2, linewidth=2, label=f"N={n}", color=COLORS[n])
    plt.xlabel("Branch pressure B")
    plt.ylabel("Routed WPU branch accuracy")
    plt.title("B sweep: routed WPU accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_work_accuracy(
    acc: dict[tuple[str, str, str], tuple[float, float]],
    work: dict[tuple[str, str, str], tuple[float, float]],
    output: Path,
) -> None:
    plt.figure(figsize=(9.0, 4.8))
    for n in NS:
        xs = [work[("wpu-routed", str(n), str(b))][0] for b in range(1, 17)]
        ys = [acc[("wpu-routed", str(n), str(b))][0] for b in range(1, 17)]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"N={n}", color=COLORS[n])
        for x, y, b in zip(xs, ys, range(1, 17), strict=True):
            if b in {1, 3, 8, 16}:
                plt.text(x * 1.03, y, f"B{b}", fontsize=7)
    plt.xscale("log")
    plt.xlabel("Routed work proxy (log)")
    plt.ylabel("Routed WPU branch accuracy")
    plt.title("B sweep: work-accuracy trajectory")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_runtime(runtime: dict[tuple[str, str, str], tuple[float, float]], output: Path) -> None:
    plt.figure(figsize=(9.0, 4.8))
    for n in NS:
        ys = [runtime[("wpu-routed", str(n), str(b))][0] for b in range(1, 17)]
        plt.plot(range(1, 17), ys, marker="o", linewidth=2, label=f"N={n}", color=COLORS[n])
    plt.xlabel("Branch pressure B")
    plt.ylabel("Routed WPU forward latency ms/sample")
    plt.title("B sweep: routed WPU runtime")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _report(rows: list[dict[str, object]], route_fig: Path, acc_fig: Path, work_fig: Path, runtime_fig: Path) -> str:
    return "\n".join(
        [
            "# Dense B Sweep v1 Results",
            "",
            "Branch pressure values: `B=1..16`.",
            "",
            "N values: `24, 84, 132, 204`.",
            "",
            "## Figures",
            "",
            f"![B sweep routes](../figures/{route_fig.name})",
            "",
            f"![B sweep accuracy](../figures/{acc_fig.name})",
            "",
            f"![B sweep work accuracy](../figures/{work_fig.name})",
            "",
            f"![B sweep runtime](../figures/{runtime_fig.name})",
            "",
            "## Routed WPU Summary",
            "",
            _table(rows),
            "",
            "## Interpretation",
            "",
            "- `B` is not an output class count; it is scheduler pressure.",
            "- Increasing `B` moves routed WPU toward hybrid/dense paths according to `rho = B/N`.",
            "- Accuracy is not monotonic in `B`; larger branch pressure can trigger dense routing and hurt accuracy.",
            "- The best `B` depends on `N`, which is evidence against fixed thresholds as a final scheduler.",
            "- A learned scheduler should optimize accuracy-latency jointly rather than blindly following hard `rho` thresholds.",
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
