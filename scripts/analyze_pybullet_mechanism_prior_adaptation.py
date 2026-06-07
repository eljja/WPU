from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


WPU_PREFIX = "wpu-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyBullet shift results before and after mechanism-prior adaptation.")
    parser.add_argument("--base", type=Path, default=Path("docs/experiments/pybullet_shift_generalization.csv"))
    parser.add_argument(
        "--adapted",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_generalization_mechanism_prior.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_prior_adaptation_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_prior_adaptation_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_prior_adaptation_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _compare(_read_summary(args.base), _read_summary(args.adapted))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = [row for row in rows if row.get("row_type") == "summary"]
    return summary or rows


def _best_pair(rows: list[dict[str, str]], mechanism: str) -> tuple[dict[str, str], dict[str, str]]:
    group = [row for row in rows if row["eval_mechanism"] == mechanism]
    wpu = [row for row in group if row["model"].startswith(WPU_PREFIX)]
    baseline = [row for row in group if not row["model"].startswith(WPU_PREFIX)]
    return (
        max(wpu, key=lambda row: float(row["branch_accuracy"])),
        max(baseline, key=lambda row: float(row["branch_accuracy"])),
    )


def _compare(base: list[dict[str, str]], adapted: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mechanisms = sorted({row["eval_mechanism"] for row in base})
    for mechanism in mechanisms:
        base_wpu, base_baseline = _best_pair(base, mechanism)
        adapted_wpu, adapted_baseline = _best_pair(adapted, mechanism)
        majority_accuracy = statistics.fmean(
            float(row["majority_accuracy"]) for row in base if row["eval_mechanism"] == mechanism
        )
        base_wpu_accuracy = float(base_wpu["branch_accuracy"])
        adapted_wpu_accuracy = float(adapted_wpu["branch_accuracy"])
        base_baseline_accuracy = float(base_baseline["branch_accuracy"])
        adapted_baseline_accuracy = float(adapted_baseline["branch_accuracy"])
        rows.append(
            {
                "eval_mechanism": mechanism,
                "base_best_wpu": base_wpu["model"],
                "adapted_best_wpu": adapted_wpu["model"],
                "base_best_baseline": base_baseline["model"],
                "adapted_best_baseline": adapted_baseline["model"],
                "base_wpu_accuracy": round(base_wpu_accuracy, 6),
                "adapted_wpu_accuracy": round(adapted_wpu_accuracy, 6),
                "wpu_accuracy_change": round(adapted_wpu_accuracy - base_wpu_accuracy, 6),
                "base_baseline_accuracy": round(base_baseline_accuracy, 6),
                "adapted_baseline_accuracy": round(adapted_baseline_accuracy, 6),
                "baseline_accuracy_change": round(adapted_baseline_accuracy - base_baseline_accuracy, 6),
                "base_wpu_minus_baseline": round(base_wpu_accuracy - base_baseline_accuracy, 6),
                "adapted_wpu_minus_baseline": round(adapted_wpu_accuracy - adapted_baseline_accuracy, 6),
                "wpu_margin_change": round(
                    (adapted_wpu_accuracy - adapted_baseline_accuracy)
                    - (base_wpu_accuracy - base_baseline_accuracy),
                    6,
                ),
                "majority_accuracy": round(majority_accuracy, 6),
                "base_majority_minus_wpu": round(majority_accuracy - base_wpu_accuracy, 6),
                "adapted_majority_minus_wpu": round(majority_accuracy - adapted_wpu_accuracy, 6),
                "base_wpu_ece": round(float(base_wpu["ece"]), 6),
                "adapted_wpu_ece": round(float(adapted_wpu["ece"]), 6),
                "wpu_ece_change": round(float(adapted_wpu["ece"]) - float(base_wpu["ece"]), 6),
                "base_wpu_brier": round(float(base_wpu["brier"]), 6),
                "adapted_wpu_brier": round(float(adapted_wpu["brier"]), 6),
                "wpu_brier_change": round(float(adapted_wpu["brier"]) - float(base_wpu["brier"]), 6),
                "base_prior_dominated": majority_accuracy > base_wpu_accuracy
                and majority_accuracy > base_baseline_accuracy,
                "adapted_prior_dominated": majority_accuracy > adapted_wpu_accuracy
                and majority_accuracy > adapted_baseline_accuracy,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    shifted = [row for row in rows if row["eval_mechanism"] != "nominal"]
    base_win_rate = statistics.fmean(
        1.0 if float(row["base_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in shifted
    )
    adapted_win_rate = statistics.fmean(
        1.0 if float(row["adapted_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in shifted
    )
    mean_wpu_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in shifted)
    mean_wpu_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in shifted)
    prior_dominated_before = sum(1 for row in shifted if row["base_prior_dominated"])
    prior_dominated_after = sum(1 for row in shifted if row["adapted_prior_dominated"])
    if korean:
        title = "# PyBullet Mechanism-Prior Adaptation"
        intro = (
            "이 실험은 held-out mechanism별 작은 calibration set으로 branch label prior를 추정하고, "
            "train prior 대비 log-prior bias를 branch logits에 더한다. 이는 test label oracle이 아니라 "
            "mechanism-aware prior adaptation의 작은 진단 실험이다."
        )
        interpretation = [
            f"Shift mechanism 기준 WPU win-rate는 `{base_win_rate:.6f}`에서 `{adapted_win_rate:.6f}`로 변했다.",
            f"Shift mechanism 기준 평균 WPU accuracy 변화는 `{mean_wpu_accuracy_change:.6f}`이다.",
            f"Shift mechanism 기준 평균 WPU ECE 변화는 `{mean_wpu_ece_change:.6f}`이다. 양수면 calibration이 악화된 것이다.",
            f"Prior-dominated shifted mechanism은 `{prior_dominated_before}`개에서 `{prior_dominated_after}`개로 줄었다.",
            "`catch_heavy`는 크게 개선되지만, 다른 shift에서는 ECE와 accuracy가 악화될 수 있다. 따라서 branch prior adaptation은 필요하지만, 단순 prior bias만으로 P4/P5가 해결되지는 않는다.",
        ]
    else:
        title = "# PyBullet Mechanism-Prior Adaptation"
        intro = (
            "This experiment estimates a branch-label prior from a small calibration set for each held-out "
            "mechanism and adds a log-prior bias relative to the training prior. It is not a test-label "
            "oracle; it is a diagnostic for mechanism-aware prior adaptation."
        )
        interpretation = [
            f"WPU win-rate over shifted mechanisms changes from `{base_win_rate:.6f}` to `{adapted_win_rate:.6f}`.",
            f"Mean WPU accuracy change over shifted mechanisms is `{mean_wpu_accuracy_change:.6f}`.",
            f"Mean WPU ECE change over shifted mechanisms is `{mean_wpu_ece_change:.6f}`; positive means worse calibration.",
            f"Prior-dominated shifted mechanisms fall from `{prior_dominated_before}` to `{prior_dominated_after}`.",
            "`catch_heavy` improves strongly, but other shifts can lose accuracy or calibration. Mechanism-aware branch priors are necessary, but a simple prior bias does not solve P4/P5.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
        "",
        f"- `{args.base.as_posix()}`",
        f"- `{args.adapted.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| mechanism | base WPU acc | adapted WPU acc | WPU acc change | base WPU-baseline | adapted WPU-baseline | majority acc | base gap | adapted gap | WPU ECE change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | {float(row['base_wpu_accuracy']):.6f} | "
            f"{float(row['adapted_wpu_accuracy']):.6f} | {float(row['wpu_accuracy_change']):.6f} | "
            f"{float(row['base_wpu_minus_baseline']):.6f} | "
            f"{float(row['adapted_wpu_minus_baseline']):.6f} | {float(row['majority_accuracy']):.6f} | "
            f"{float(row['base_majority_minus_wpu']):.6f} | "
            f"{float(row['adapted_majority_minus_wpu']):.6f} | {float(row['wpu_ece_change']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
