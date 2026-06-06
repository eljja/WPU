from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    Path("docs/experiments/wpu_v2_candidate_regret_gate_summary.csv"),
    Path("docs/experiments/wpu_v2_candidate_regret_gate_perturbed_summary.csv"),
    Path("docs/experiments/wpu_v2_candidate_regret_gate_penalty_summary.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize candidate-regret safety/closure frontiers across P1 selector probes."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["direct", "perturbed", "penalty"],
    )
    parser.add_argument("--harmful-limits", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/wpu_v2_candidate_safety_frontier.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/wpu_v2_candidate_safety_frontier_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/wpu_v2_candidate_safety_frontier_results.ko.md"),
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    rows = []
    for path, label in zip(args.inputs, args.labels, strict=True):
        rows.extend(_frontier_rows(path, label, args.harmful_limits))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args.inputs, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args.inputs, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _frontier_rows(path: Path, label: str, harmful_limits: list[float]) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for limit in harmful_limits:
        feasible = [
            row
            for row in source_rows
            if float(row.get("mean_harmful_accept_rate", 1.0)) <= limit
            and float(row.get("gap_closure_fraction", 0.0)) > 0.0
        ]
        if feasible:
            best = max(feasible, key=lambda row: float(row["gap_closure_fraction"]))
            rows.append(
                {
                    "probe": label,
                    "source_csv": path.as_posix(),
                    "harmful_limit": round(limit, 6),
                    "best_policy": best["policy"],
                    "causal_k": int(float(best["causal_k"])),
                    "gap_closure_fraction": round(float(best["gap_closure_fraction"]), 6),
                    "mean_harmful_accept_rate": round(float(best.get("mean_harmful_accept_rate", 0.0)), 6),
                    "mean_accept_rate": round(float(best.get("mean_accept_rate", 0.0)), 6),
                    "feasible_policy_count": len(feasible),
                }
            )
        else:
            rows.append(
                {
                    "probe": label,
                    "source_csv": path.as_posix(),
                    "harmful_limit": round(limit, 6),
                    "best_policy": "none",
                    "causal_k": 0,
                    "gap_closure_fraction": 0.0,
                    "mean_harmful_accept_rate": 0.0,
                    "mean_accept_rate": 0.0,
                    "feasible_policy_count": 0,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], input_paths: list[Path], output_csv: Path, *, korean: bool) -> str:
    if korean:
        title = "# Candidate Safety Frontier"
        intro = (
            "이 보고서는 P1 candidate-regret selector의 closure/safety tradeoff를 "
            "harmful accept threshold별로 다시 요약한다."
        )
        interpretation = [
            "P1의 실패는 단순히 threshold를 못 찾은 문제가 아니다.",
            "높은 closure를 얻는 구간은 harmful accept가 커지고, harmful accept를 강하게 낮추면 closure가 급격히 줄어든다.",
            "따라서 다음 개선은 post-hoc threshold가 아니라 candidate scoring 자체의 ranking, no-harm, uncertainty target을 함께 바꾸어야 한다.",
        ]
    else:
        title = "# Candidate Safety Frontier"
        intro = (
            "This report re-summarizes the P1 candidate-regret selector as a "
            "closure/safety frontier over harmful-accept thresholds."
        )
        interpretation = [
            "P1 is not failing because a single threshold is missing.",
            "High closure coincides with higher harmful accepts, while strict harmful-accept limits collapse closure.",
            "The next improvement must change candidate scoring itself: ranking, no-harm, and uncertainty targets have to be learned jointly rather than tuned post hoc.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in input_paths)
    lines.extend(
        [
            "",
            "Derived CSV:",
            "",
            f"- `{output_csv.as_posix()}`",
            "",
            "| probe | harmful limit | best policy | K | closure | harmful accept | accept | feasible policies |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['probe']} | {float(row['harmful_limit']):.2f} | `{row['best_policy']}` | "
            f"{row['causal_k']} | {float(row['gap_closure_fraction']):.6f} | "
            f"{float(row['mean_harmful_accept_rate']):.6f} | {float(row['mean_accept_rate']):.6f} | "
            f"{row['feasible_policy_count']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
