from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


WPU_MODELS = {"wpu-routed", "wpu-sparse", "wpu-hybrid", "wpu-dense"}
NON_WPU_MODELS = {"dense-graph", "graph-transformer", "serialized-token"}
KEY_MODELS = ["wpu-routed", "wpu-sparse", "wpu-hybrid", "dense-graph", "graph-transformer", "serialized-token"]
COLORS = {
    "wpu-routed": "#0f766e",
    "wpu-sparse": "#14b8a6",
    "wpu-hybrid": "#166534",
    "dense-graph": "#2563eb",
    "graph-transformer": "#7c3aed",
    "serialized-token": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dense N sweep crossover points.")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/n_sweep_v1"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/n_sweep_v1_results.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--branch", type=int, default=3)
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    final_rows = _read_rows(args.input_dir / "final_baselines.csv")
    regime_rows = _read_rows(args.input_dir / "regime_sweep.csv")
    runtime_rows = _read_rows(args.input_dir / "runtime_memory.csv")

    accuracy = _aggregate(final_rows, ["model", "total_objects"], "branch_accuracy")
    runtime = _aggregate(
        [row for row in runtime_rows if int(row["branches"]) == args.branch],
        ["model", "total_objects"],
        "ms_per_sample_forward",
    )
    regime = _aggregate(
        [row for row in regime_rows if int(row["branches"]) == args.branch],
        ["model", "total_objects"],
        "routed_work_proxy",
    )
    route_ratios = {
        name: _aggregate(
            [row for row in regime_rows if int(row["branches"]) == args.branch],
            ["model", "total_objects"],
            name,
        )
        for name in ["sparse_ratio", "hybrid_ratio", "dense_ratio"]
    }

    ns = sorted({int(row["total_objects"]) for row in final_rows})
    crossover_rows = _crossover_rows(ns, accuracy, runtime, regime)
    route_rows = _route_rows(ns, route_ratios)

    acc_fig = args.figure_dir / "n_sweep_accuracy.png"
    delta_fig = args.figure_dir / "n_sweep_delta.png"
    runtime_fig = args.figure_dir / "n_sweep_runtime.png"
    route_fig = args.figure_dir / "n_sweep_routes.png"
    _plot_accuracy(ns, accuracy, acc_fig)
    _plot_delta(ns, accuracy, delta_fig)
    _plot_runtime(ns, runtime, runtime_fig)
    _plot_routes(ns, route_ratios, route_fig)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        _markdown_report(
            ns=ns,
            crossover_rows=crossover_rows,
            route_rows=route_rows,
            acc_fig=acc_fig,
            delta_fig=delta_fig,
            runtime_fig=runtime_fig,
            route_fig=route_fig,
            branch=args.branch,
        ),
        encoding="utf-8",
    )
    print(f"wrote={args.output_md}")
    print(f"wrote={acc_fig}")
    print(f"wrote={delta_fig}")
    print(f"wrote={runtime_fig}")
    print(f"wrote={route_fig}")


def _crossover_rows(
    ns: list[int],
    accuracy: dict[tuple[str, str], tuple[float, float]],
    runtime: dict[tuple[str, str], tuple[float, float]],
    regime: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, object]]:
    rows = []
    for n in ns:
        best_wpu = _best_for_family(n, accuracy, WPU_MODELS)
        best_non = _best_for_family(n, accuracy, NON_WPU_MODELS)
        routed_acc = accuracy.get(("wpu-routed", str(n)), (math.nan, 0.0))
        routed_runtime = runtime.get(("wpu-routed", str(n)), (math.nan, 0.0))
        token_runtime = runtime.get(("serialized-token", str(n)), (math.nan, 0.0))
        graph_runtime = runtime.get(("graph-transformer", str(n)), (math.nan, 0.0))
        rows.append(
            {
                "N": n,
                "best_wpu": best_wpu[0],
                "best_wpu_acc": round(best_wpu[1][0], 6),
                "best_non_wpu": best_non[0],
                "best_non_wpu_acc": round(best_non[1][0], 6),
                "acc_gap_wpu_minus_non": round(best_wpu[1][0] - best_non[1][0], 6),
                "wpu_routed_acc": round(routed_acc[0], 6),
                "wpu_routed_runtime_ms": round(routed_runtime[0], 6),
                "serialized_token_runtime_ms": round(token_runtime[0], 6),
                "graph_transformer_runtime_ms": round(graph_runtime[0], 6),
                "routed_work_proxy": round(regime.get(("wpu-routed", str(n)), (math.nan, 0.0))[0], 6),
            }
        )
    return rows


def _route_rows(
    ns: list[int],
    route_ratios: dict[str, dict[tuple[str, str], tuple[float, float]]],
) -> list[dict[str, object]]:
    rows = []
    for n in ns:
        sparse = route_ratios["sparse_ratio"].get(("wpu-routed", str(n)), (0.0, 0.0))[0]
        hybrid = route_ratios["hybrid_ratio"].get(("wpu-routed", str(n)), (0.0, 0.0))[0]
        dense = route_ratios["dense_ratio"].get(("wpu-routed", str(n)), (0.0, 0.0))[0]
        route = max([("sparse", sparse), ("hybrid", hybrid), ("dense", dense)], key=lambda item: item[1])[0]
        rows.append({"N": n, "route": route, "sparse": sparse, "hybrid": hybrid, "dense": dense})
    return rows


def _best_for_family(
    n: int,
    accuracy: dict[tuple[str, str], tuple[float, float]],
    family: set[str],
) -> tuple[str, tuple[float, float]]:
    candidates = [(model, values) for (model, total_objects), values in accuracy.items() if int(total_objects) == n and model in family]
    return max(candidates, key=lambda item: item[1][0])


def _plot_accuracy(ns: list[int], accuracy: dict[tuple[str, str], tuple[float, float]], output: Path) -> None:
    plt.figure(figsize=(9.0, 5.0))
    for model in KEY_MODELS:
        xs, ys, errs = [], [], []
        for n in ns:
            values = accuracy.get((model, str(n)))
            if values:
                xs.append(n)
                ys.append(values[0])
                errs.append(values[1])
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=2, label=model, color=COLORS.get(model))
    plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns], rotation=35)
    plt.ylim(0.0, 0.9)
    plt.xlabel("Total objects N")
    plt.ylabel("Branch accuracy (mean +/- 95% CI)")
    plt.title("Dense N sweep: accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_delta(ns: list[int], accuracy: dict[tuple[str, str], tuple[float, float]], output: Path) -> None:
    gaps = []
    for n in ns:
        best_wpu = _best_for_family(n, accuracy, WPU_MODELS)
        best_non = _best_for_family(n, accuracy, NON_WPU_MODELS)
        gaps.append(best_wpu[1][0] - best_non[1][0])
    plt.figure(figsize=(9.0, 4.2))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(ns, gaps, marker="o", linewidth=2, color="#0f766e")
    plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns], rotation=35)
    plt.xlabel("Total objects N")
    plt.ylabel("Best WPU accuracy - best non-WPU accuracy")
    plt.title("Crossover curve: positive means WPU-family wins")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_runtime(ns: list[int], runtime: dict[tuple[str, str], tuple[float, float]], output: Path) -> None:
    plt.figure(figsize=(9.0, 5.0))
    for model in ["wpu-routed", "wpu-sparse", "dense-graph", "graph-transformer", "serialized-token"]:
        xs, ys, errs = [], [], []
        for n in ns:
            values = runtime.get((model, str(n)))
            if values:
                xs.append(n)
                ys.append(values[0])
                errs.append(values[1])
        plt.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=2, label=model, color=COLORS.get(model))
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(ns, [str(n) for n in ns], rotation=35)
    plt.xlabel("Total objects N")
    plt.ylabel("Forward latency ms/sample (log)")
    plt.title("Dense N sweep: runtime at B=3")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_routes(
    ns: list[int],
    route_ratios: dict[str, dict[tuple[str, str], tuple[float, float]]],
    output: Path,
) -> None:
    plt.figure(figsize=(9.0, 4.2))
    for name, color in [("sparse_ratio", "#14b8a6"), ("hybrid_ratio", "#f59e0b"), ("dense_ratio", "#ef4444")]:
        ys = [route_ratios[name].get(("wpu-routed", str(n)), (0.0, 0.0))[0] for n in ns]
        plt.plot(ns, ys, marker="o", linewidth=2, label=name.replace("_ratio", ""), color=color)
    plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns], rotation=35)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Total objects N")
    plt.ylabel("Route ratio")
    plt.title("Routed WPU path transition at B=3")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _markdown_report(
    *,
    ns: list[int],
    crossover_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
    acc_fig: Path,
    delta_fig: Path,
    runtime_fig: Path,
    route_fig: Path,
    branch: int,
) -> str:
    route_changes = []
    previous = None
    for row in route_rows:
        route = row["route"]
        if previous is None or route != previous:
            route_changes.append((row["N"], route))
        previous = route
    positive = [row for row in crossover_rows if float(row["acc_gap_wpu_minus_non"]) > 0]
    negative = [row for row in crossover_rows if float(row["acc_gap_wpu_minus_non"]) <= 0]
    runtime_wins = [
        row for row in crossover_rows
        if float(row["wpu_routed_runtime_ms"]) < float(row["serialized_token_runtime_ms"])
        and float(row["wpu_routed_runtime_ms"]) < float(row["graph_transformer_runtime_ms"])
    ]
    lines = [
        "# Dense N Sweep v1 Results",
        "",
        f"Branch pressure for route/runtime analysis: `B={branch}`.",
        "",
        f"N values: `{', '.join(str(n) for n in ns)}`.",
        "",
        "## Figures",
        "",
        f"![N sweep accuracy](../figures/{acc_fig.name})",
        "",
        f"![N sweep crossover](../figures/{delta_fig.name})",
        "",
        f"![N sweep runtime](../figures/{runtime_fig.name})",
        "",
        f"![N sweep routes](../figures/{route_fig.name})",
        "",
        "## Route Change Points",
        "",
        _table([{"first_N": n, "dominant_route": route} for n, route in route_changes]),
        "",
        "## Accuracy Crossover Table",
        "",
        _table(crossover_rows),
        "",
        "## Interpretation",
        "",
        f"- WPU-family wins at N values: `{', '.join(str(row['N']) for row in positive) or 'none'}`.",
        f"- Non-WPU family wins at N values: `{', '.join(str(row['N']) for row in negative) or 'none'}`.",
        f"- Routed WPU is faster than both serialized-token and graph-transformer at N values: `{', '.join(str(row['N']) for row in runtime_wins) or 'none'}`.",
        "- The important crossover is not a single number. Accuracy, route, and runtime cross at different N ranges.",
        "- The v1 target is to move the accuracy crossover rightward while preserving the runtime crossover.",
        "",
    ]
    return "\n".join(lines)


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


if __name__ == "__main__":
    main()
