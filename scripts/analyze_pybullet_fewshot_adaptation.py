from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


WPU_PREFIX = "wpu-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and few-shot PyBullet mechanism adaptation results.")
    parser.add_argument("--base", type=Path, default=Path("docs/experiments/pybullet_shift_generalization.csv"))
    parser.add_argument(
        "--adapted",
        type=Path,
        default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation_results.ko.md"),
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
                "base_wpu_ece": round(float(base_wpu["ece"]), 6),
                "adapted_wpu_ece": round(float(adapted_wpu["ece"]), 6),
                "wpu_ece_change": round(float(adapted_wpu["ece"]) - float(base_wpu["ece"]), 6),
                "base_wpu_brier": round(float(base_wpu["brier"]), 6),
                "adapted_wpu_brier": round(float(adapted_wpu["brier"]), 6),
                "wpu_brier_change": round(float(adapted_wpu["brier"]) - float(base_wpu["brier"]), 6),
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
    mean_baseline_accuracy_change = statistics.fmean(float(row["baseline_accuracy_change"]) for row in shifted)
    mean_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in shifted)
    mean_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in shifted)
    mean_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in shifted)
    if korean:
        title = "# PyBullet Few-Shot Mechanism Adaptation"
        intro = (
            "이 실험은 nominal로 학습한 모델을 held-out mechanism별 작은 calibration set에서 "
            "몇 step fine-tune한 뒤 evaluation set에 적용한다. Baseline에도 같은 adaptation을 "
            "적용하므로, WPU만 유리하게 만든 실험이 아니다."
        )
        interpretation = [
            f"Shifted WPU win-rate는 `{base_win_rate:.6f}`에서 `{adapted_win_rate:.6f}`로 변했다.",
            f"Shifted 평균 WPU accuracy 변화는 `{mean_wpu_accuracy_change:.6f}`이고 baseline accuracy 변화는 `{mean_baseline_accuracy_change:.6f}`이다.",
            f"Shifted 평균 WPU-baseline margin 변화는 `{mean_margin_change:.6f}`이다.",
            f"Shifted 평균 WPU ECE 변화는 `{mean_ece_change:.6f}`, Brier 변화는 `{mean_brier_change:.6f}`이다.",
            "이 실험은 branch prior만으로 부족한 mechanism shift를 모델 파라미터 적응으로 줄일 수 있는지 보는 P4 follow-up이다.",
        ]
    else:
        title = "# PyBullet Few-Shot Mechanism Adaptation"
        intro = (
            "This experiment fine-tunes nominal-trained models for a few steps on a small calibration set from "
            "each held-out mechanism, then evaluates on the same shift benchmark. The same adaptation is applied "
            "to WPU and non-WPU baselines."
        )
        interpretation = [
            f"Shifted WPU win-rate changes from `{base_win_rate:.6f}` to `{adapted_win_rate:.6f}`.",
            f"Mean shifted WPU accuracy change is `{mean_wpu_accuracy_change:.6f}`; baseline accuracy change is `{mean_baseline_accuracy_change:.6f}`.",
            f"Mean shifted WPU-baseline margin change is `{mean_margin_change:.6f}`.",
            f"Mean shifted WPU ECE change is `{mean_ece_change:.6f}` and Brier change is `{mean_brier_change:.6f}`.",
            "This is a P4 follow-up: it tests whether mechanism shift that is not solved by branch priors can be reduced by parameter adaptation.",
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
        "| mechanism | base WPU acc | adapted WPU acc | WPU acc change | baseline acc change | base WPU-baseline | adapted WPU-baseline | margin change | WPU ECE change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | {float(row['base_wpu_accuracy']):.6f} | "
            f"{float(row['adapted_wpu_accuracy']):.6f} | {float(row['wpu_accuracy_change']):.6f} | "
            f"{float(row['baseline_accuracy_change']):.6f} | {float(row['base_wpu_minus_baseline']):.6f} | "
            f"{float(row['adapted_wpu_minus_baseline']):.6f} | {float(row['wpu_margin_change']):.6f} | "
            f"{float(row['wpu_ece_change']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
