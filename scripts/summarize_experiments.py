from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline_suite.csv"))
    parser.add_argument("--regime", type=Path, default=Path("artifacts/regime_sweep_trained.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/baseline_and_regime_results.md"))
    args = parser.parse_args()

    baseline_rows = _read(args.baseline)
    regime_rows = _read(args.regime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(baseline_rows, regime_rows), encoding="utf-8")
    print(f"wrote={args.output}")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _render(baseline_rows: list[dict[str, str]], regime_rows: list[dict[str, str]]) -> str:
    lines = [
        "# Baseline and Regime Results",
        "",
        "## Baseline Suite",
        "",
        "All models were trained for 100 steps on the synthetic robot-cup task with 80 background objects, then evaluated on 128 held-out samples.",
        "",
        "| Model | Background | MSE | Branch NLL | Branch Acc. | Sparse | Hybrid | Dense |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            "| {model} | {eval_background_objects} | {next_state_mse} | {branch_nll} | {branch_accuracy} | {sparse_path_ratio} | {hybrid_path_ratio} | {dense_path_ratio} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Regime Sweep",
            "",
            "The trained checkpoints were evaluated across world size and branch count. Work columns are proxies: sparse work is proportional to branch count under fixed initial delta and fanout clamp, while dense work is proportional to total object count squared.",
            "",
            "| Model | N | B | rho | MSE | Branch Acc. | ms/sample | Sparse | Hybrid | Dense |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regime_rows:
        lines.append(
            "| {model} | {total_objects} | {branches} | {rho} | {mse} | {branch_accuracy} | {ms_per_sample} | {sparse_ratio} | {hybrid_ratio} | {dense_ratio} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The route sweep supports the WPU regime claim: routed WPU moves from dense to hybrid to sparse as `rho` decreases.",
            "- The current quality results are mixed rather than uniformly favorable to WPU. In the 100-step baseline suite, serialized-token performs strongly at large `N`, while WPU variants are competitive or stronger in some smaller/intermediate regimes.",
            "- This is useful scientifically: the paper should claim a testable regime hypothesis, not universal dominance. The next experiment must match compute, tune all baselines, and report accuracy-compute-memory tradeoffs.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
