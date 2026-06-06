from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUTS = [
    Path("docs/experiments/pybullet_shift_leave_family_nominal.csv"),
    Path("docs/experiments/pybullet_shift_leave_family_high_force.csv"),
    Path("docs/experiments/pybullet_shift_leave_family_edge_shift.csv"),
    Path("docs/experiments/pybullet_shift_leave_family_catch_heavy.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PyBullet leave-family-out shift/calibration results.")
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_shift_leave_family_out_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_shift_leave_family_out_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/pybullet_shift_leave_family_out_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        rows.extend(_summary_rows(path))
    summary = _summarize(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render_markdown(summary, args.inputs, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(summary, args.inputs, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("row_type") == "summary"]


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    out: list[dict[str, object]] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = [row for row in group if row["model"].startswith("wpu-")]
        baseline = [row for row in group if not row["model"].startswith("wpu-")]
        best_wpu = max(wpu, key=lambda row: float(row["branch_accuracy"]))
        best_baseline = max(baseline, key=lambda row: float(row["branch_accuracy"]))
        wpu_ece = statistics.fmean(float(row["ece"]) for row in wpu)
        baseline_ece = statistics.fmean(float(row["ece"]) for row in baseline)
        out.append(
            {
                "eval_mechanism": mechanism,
                "train_mechanism": best_wpu["train_mechanism"],
                "best_wpu": best_wpu["model"],
                "best_baseline": best_baseline["model"],
                "best_wpu_accuracy": round(float(best_wpu["branch_accuracy"]), 6),
                "best_baseline_accuracy": round(float(best_baseline["branch_accuracy"]), 6),
                "accuracy_gap": round(float(best_wpu["branch_accuracy"]) - float(best_baseline["branch_accuracy"]), 6),
                "mean_wpu_ece": round(wpu_ece, 6),
                "mean_baseline_ece": round(baseline_ece, 6),
                "ece_ratio": round(wpu_ece / max(baseline_ece, 1e-9), 6),
                "wpu_win": float(best_wpu["branch_accuracy"]) >= float(best_baseline["branch_accuracy"]),
                "seed_count": int(best_wpu["seed_count"]),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], inputs: list[Path], *, korean: bool) -> str:
    if korean:
        title = "# PyBullet Leave-Family-Out Shift 결과"
        intro = (
            "이 실험은 네 mechanism family 중 하나를 완전히 held out하고 나머지 세 family로 "
            "학습한 뒤 held-out family에서 평가한다. Nominal-only shift보다 더 엄격한 "
            "cross-mechanism generalization probe다."
        )
        interpretation = (
            "Leave-family-out 결과는 WPU가 일부 geometry-driven shift에서는 강하지만 "
            "branch-prior shift 전체를 해결하지 못한다는 점을 보여준다. Accuracy win과 "
            "ECE ratio를 함께 봐야 하며, calibration은 여전히 안정적이지 않다."
        )
    else:
        title = "# PyBullet Leave-Family-Out Shift Results"
        intro = (
            "This experiment holds out one mechanism family at a time, trains on the "
            "remaining three families, and evaluates on the held-out family. It is a "
            "stricter cross-mechanism generalization probe than nominal-only shift."
        )
        interpretation = (
            "Leave-family-out results show that WPU can help in some geometry-driven "
            "shifts, but it does not solve branch-prior shift in general. Accuracy wins "
            "must be read together with ECE ratios because calibration remains unstable."
        )
    lines = [title, "", intro, "", "Source CSVs:", ""]
    lines.extend(f"- `{path.as_posix()}`" for path in inputs)
    lines.extend(
        [
            "",
            "| held-out mechanism | train mechanisms | best WPU | best baseline | WPU acc | baseline acc | gap | WPU ECE | baseline ECE | ECE ratio | WPU win |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | `{row['train_mechanism']}` | `{row['best_wpu']}` | "
            f"`{row['best_baseline']}` | {float(row['best_wpu_accuracy']):.6f} | "
            f"{float(row['best_baseline_accuracy']):.6f} | {float(row['accuracy_gap']):.6f} | "
            f"{float(row['mean_wpu_ece']):.6f} | {float(row['mean_baseline_ece']):.6f} | "
            f"{float(row['ece_ratio']):.6f} | {row['wpu_win']} |"
        )
    win_rate = sum(1 for row in rows if row["wpu_win"]) / max(len(rows), 1)
    mean_ece_ratio = statistics.fmean(float(row["ece_ratio"]) for row in rows)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            f"WPU leave-family-out win rate: `{win_rate:.6f}`.",
            f"Mean WPU/baseline ECE ratio: `{mean_ece_ratio:.6f}`.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
