from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare composition-shift calibration variants.")
    parser.add_argument("--base", type=Path, default=Path("docs/experiments/pybullet_shift_composition_stress_summary.csv"))
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv"),
    )
    parser.add_argument("--base-label", default="temperature")
    parser.add_argument("--candidate-label", default="temperature_bias")
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_shift_calibration_comparison.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_shift_calibration_comparison_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_calibration_comparison_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _compare(args.base, args.candidate, args.base_label, args.candidate_label)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["eval_mechanism"]: row for row in csv.DictReader(handle)}


def _compare(base_path: Path, candidate_path: Path, base_label: str, candidate_label: str) -> list[dict[str, object]]:
    base = _read(base_path)
    candidate = _read(candidate_path)
    rows: list[dict[str, object]] = []
    for mechanism in sorted(base):
        b = base[mechanism]
        c = candidate[mechanism]
        rows.append(
            {
                "eval_mechanism": mechanism,
                "base_label": base_label,
                "candidate_label": candidate_label,
                "base_accuracy_delta": round(float(b["accuracy_delta"]), 6),
                "candidate_accuracy_delta": round(float(c["accuracy_delta"]), 6),
                "accuracy_delta_change": round(float(c["accuracy_delta"]) - float(b["accuracy_delta"]), 6),
                "base_ece_ratio": round(float(b["ece_ratio"]), 6),
                "candidate_ece_ratio": round(float(c["ece_ratio"]), 6),
                "ece_ratio_change": round(float(c["ece_ratio"]) - float(b["ece_ratio"]), 6),
                "base_best_wpu": b["best_wpu_model"],
                "candidate_best_wpu": c["best_wpu_model"],
                "base_best_baseline": b["best_baseline_model"],
                "candidate_best_baseline": c["best_baseline_model"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    mean_accuracy_change = statistics.fmean(float(row["accuracy_delta_change"]) for row in rows)
    mean_ece_change = statistics.fmean(float(row["ece_ratio_change"]) for row in rows)
    improved_ece = sum(1 for row in rows if float(row["ece_ratio_change"]) < 0.0)
    if korean:
        title = "# PyBullet Shift Calibration Comparison"
        intro = "이 문서는 composition-shift stress에서 temperature calibration과 temperature+bias calibration을 비교한다."
        interpretation = [
            f"평균 accuracy-delta 변화는 `{mean_accuracy_change:.6f}`이고 평균 ECE-ratio 변화는 `{mean_ece_change:.6f}`이다.",
            f"ECE ratio가 개선된 mechanism은 `{improved_ece}/{len(rows)}`개다.",
            "Branch-bias calibration은 `no_catch` calibration을 크게 개선하지만, 다른 shift에서는 accuracy 또는 ECE를 악화시킬 수 있다. 따라서 P5는 보편 해결이 아니라 mechanism-aware uncertainty/calibration 문제로 남는다.",
        ]
    else:
        title = "# PyBullet Shift Calibration Comparison"
        intro = "This report compares temperature calibration with temperature+bias calibration on the composition-shift stress probe."
        interpretation = [
            f"Mean accuracy-delta change is `{mean_accuracy_change:.6f}` and mean ECE-ratio change is `{mean_ece_change:.6f}`.",
            f"ECE ratio improves on `{improved_ece}/{len(rows)}` mechanisms.",
            "Branch-bias calibration strongly helps `no_catch`, but it can degrade accuracy or ECE on other shifts. P5 therefore remains a mechanism-aware uncertainty/calibration problem, not a solved post-hoc calibration problem.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
        "",
        f"- `{args.base.as_posix()}`",
        f"- `{args.candidate.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| mechanism | base acc delta | bias acc delta | acc change | base ECE ratio | bias ECE ratio | ECE change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | {float(row['base_accuracy_delta']):.6f} | "
            f"{float(row['candidate_accuracy_delta']):.6f} | {float(row['accuracy_delta_change']):.6f} | "
            f"{float(row['base_ece_ratio']):.6f} | {float(row['candidate_ece_ratio']):.6f} | "
            f"{float(row['ece_ratio_change']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
