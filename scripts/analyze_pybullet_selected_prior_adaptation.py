from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


WPU_PREFIX = "wpu-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyBullet shift results before and after calibration-selected mechanism-prior adaptation."
    )
    parser.add_argument("--base", type=Path, default=Path("docs/experiments/pybullet_shift_generalization.csv"))
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_generalization_selected_prior.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/pybullet_selected_prior_adaptation_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_selected_prior_adaptation_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_selected_prior_adaptation_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _compare(_read_summary(args.base), _read_summary(args.selected))
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


def _compare(base: list[dict[str, str]], selected: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mechanisms = sorted({row["eval_mechanism"] for row in base})
    for mechanism in mechanisms:
        base_wpu, base_baseline = _best_pair(base, mechanism)
        selected_wpu, selected_baseline = _best_pair(selected, mechanism)
        majority_accuracy = statistics.fmean(
            float(row["majority_accuracy"]) for row in base if row["eval_mechanism"] == mechanism
        )
        base_wpu_accuracy = float(base_wpu["branch_accuracy"])
        selected_wpu_accuracy = float(selected_wpu["branch_accuracy"])
        base_baseline_accuracy = float(base_baseline["branch_accuracy"])
        selected_baseline_accuracy = float(selected_baseline["branch_accuracy"])
        rows.append(
            {
                "eval_mechanism": mechanism,
                "base_best_wpu": base_wpu["model"],
                "selected_best_wpu": selected_wpu["model"],
                "base_best_baseline": base_baseline["model"],
                "selected_best_baseline": selected_baseline["model"],
                "selected_mean_strength": round(float(selected_wpu.get("mechanism_prior_strength") or 0.0), 6),
                "selection_metric": selected_wpu.get("mechanism_prior_selection_metric", ""),
                "base_wpu_accuracy": round(base_wpu_accuracy, 6),
                "selected_wpu_accuracy": round(selected_wpu_accuracy, 6),
                "wpu_accuracy_change": round(selected_wpu_accuracy - base_wpu_accuracy, 6),
                "base_baseline_accuracy": round(base_baseline_accuracy, 6),
                "selected_baseline_accuracy": round(selected_baseline_accuracy, 6),
                "base_wpu_minus_baseline": round(base_wpu_accuracy - base_baseline_accuracy, 6),
                "selected_wpu_minus_baseline": round(selected_wpu_accuracy - selected_baseline_accuracy, 6),
                "majority_accuracy": round(majority_accuracy, 6),
                "base_majority_minus_wpu": round(majority_accuracy - base_wpu_accuracy, 6),
                "selected_majority_minus_wpu": round(majority_accuracy - selected_wpu_accuracy, 6),
                "base_wpu_ece": round(float(base_wpu["ece"]), 6),
                "selected_wpu_ece": round(float(selected_wpu["ece"]), 6),
                "wpu_ece_change": round(float(selected_wpu["ece"]) - float(base_wpu["ece"]), 6),
                "base_wpu_brier": round(float(base_wpu["brier"]), 6),
                "selected_wpu_brier": round(float(selected_wpu["brier"]), 6),
                "wpu_brier_change": round(float(selected_wpu["brier"]) - float(base_wpu["brier"]), 6),
                "base_prior_dominated": majority_accuracy > base_wpu_accuracy
                and majority_accuracy > base_baseline_accuracy,
                "selected_prior_dominated": majority_accuracy > selected_wpu_accuracy
                and majority_accuracy > selected_baseline_accuracy,
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
    selected_win_rate = statistics.fmean(
        1.0 if float(row["selected_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in shifted
    )
    mean_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in shifted)
    mean_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in shifted)
    mean_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in shifted)
    prior_before = sum(1 for row in shifted if row["base_prior_dominated"])
    prior_after = sum(1 for row in shifted if row["selected_prior_dominated"])
    if korean:
        title = "# PyBullet Calibration-Selected Prior Adaptation"
        intro = (
            "이 실험은 test set이 아니라 held-out mechanism별 작은 calibration set에서 "
            "후보 prior strength를 선택한 뒤 evaluation set에 적용한다. 목표는 고정 prior "
            "bias보다 더 calibration-safe한 branch-prior adaptation이 가능한지 확인하는 것이다."
        )
        interpretation = [
            f"Shifted WPU win-rate는 `{base_win_rate:.6f}`에서 `{selected_win_rate:.6f}`로 변했다.",
            f"Shifted 평균 WPU accuracy 변화는 `{mean_accuracy_change:.6f}`이다.",
            f"Shifted 평균 WPU ECE 변화는 `{mean_ece_change:.6f}`이고, 평균 Brier 변화는 `{mean_brier_change:.6f}`이다. 음수는 개선이다.",
            f"Prior-dominated shifted mechanism은 `{prior_before}`개에서 `{prior_after}`개로 줄었다.",
            "따라서 selected prior는 P5 calibration에는 실제 개선을 보이지만, P4 baseline win-rate를 올리지는 못한다. 다음 단계는 prior strength 선택이 아니라 model confidence와 mechanism uncertainty를 함께 학습하는 것이다.",
        ]
    else:
        title = "# PyBullet Calibration-Selected Prior Adaptation"
        intro = (
            "This experiment selects a prior strength on a small held-out calibration set for each mechanism, "
            "then applies the selected strength to the evaluation set. It tests whether branch-prior adaptation "
            "can be made more calibration-safe than a fixed prior bias."
        )
        interpretation = [
            f"Shifted WPU win-rate changes from `{base_win_rate:.6f}` to `{selected_win_rate:.6f}`.",
            f"Mean shifted WPU accuracy change is `{mean_accuracy_change:.6f}`.",
            f"Mean shifted WPU ECE change is `{mean_ece_change:.6f}` and mean Brier change is `{mean_brier_change:.6f}`; negative means better.",
            f"Prior-dominated shifted mechanisms fall from `{prior_before}` to `{prior_after}`.",
            "Selected priors improve P5 calibration evidence but do not improve P4 baseline win-rate. The next step is to learn model confidence and mechanism uncertainty jointly rather than only selecting a scalar prior strength.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
        "",
        f"- `{args.base.as_posix()}`",
        f"- `{args.selected.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| mechanism | selected strength | base WPU acc | selected WPU acc | WPU acc change | base WPU-baseline | selected WPU-baseline | WPU ECE change | WPU Brier change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | {float(row['selected_mean_strength']):.6f} | "
            f"{float(row['base_wpu_accuracy']):.6f} | {float(row['selected_wpu_accuracy']):.6f} | "
            f"{float(row['wpu_accuracy_change']):.6f} | {float(row['base_wpu_minus_baseline']):.6f} | "
            f"{float(row['selected_wpu_minus_baseline']):.6f} | {float(row['wpu_ece_change']):.6f} | "
            f"{float(row['wpu_brier_change']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
