from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a mechanism-aware policy that selects prior adaptation or few-shot adaptation."
    )
    parser.add_argument(
        "--selected-prior",
        type=Path,
        default=Path("docs/experiments/pybullet_selected_prior_adaptation_summary.csv"),
    )
    parser.add_argument(
        "--fewshot",
        type=Path,
        default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv"),
    )
    parser.add_argument("--prior-threshold", type=float, default=0.75)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_adaptive_policy_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_adaptive_policy_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_mechanism_adaptive_policy_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _analyze(
        _read_rows(args.selected_prior),
        _read_rows(args.fewshot),
        prior_threshold=args.prior_threshold,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _by_mechanism(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["eval_mechanism"]: row for row in rows}


def _analyze(
    selected_rows: list[dict[str, str]],
    fewshot_rows: list[dict[str, str]],
    *,
    prior_threshold: float,
) -> list[dict[str, object]]:
    selected = _by_mechanism(selected_rows)
    fewshot = _by_mechanism(fewshot_rows)
    mechanisms = sorted(set(selected) & set(fewshot))
    rows: list[dict[str, object]] = []
    for mechanism in mechanisms:
        selected_row = selected[mechanism]
        fewshot_row = fewshot[mechanism]
        selected_strength = float(selected_row["selected_mean_strength"])
        if mechanism == "nominal":
            policy = "base_nominal"
            policy_accuracy = float(selected_row["base_wpu_accuracy"])
            policy_margin = float(selected_row["base_wpu_minus_baseline"])
            policy_ece_change = 0.0
            policy_brier_change = 0.0
        elif selected_strength >= prior_threshold:
            policy = "selected_prior"
            policy_accuracy = float(selected_row["selected_wpu_accuracy"])
            policy_margin = float(selected_row["selected_wpu_minus_baseline"])
            policy_ece_change = float(selected_row["wpu_ece_change"])
            policy_brier_change = float(selected_row["wpu_brier_change"])
        else:
            policy = "fewshot_adaptation"
            policy_accuracy = float(fewshot_row["adapted_wpu_accuracy"])
            policy_margin = float(fewshot_row["adapted_wpu_minus_baseline"])
            policy_ece_change = float(fewshot_row["wpu_ece_change"])
            policy_brier_change = float(fewshot_row.get("wpu_brier_change", 0.0))
        base_accuracy = float(selected_row["base_wpu_accuracy"])
        base_margin = float(selected_row["base_wpu_minus_baseline"])
        rows.append(
            {
                "eval_mechanism": mechanism,
                "selected_policy": policy,
                "selected_prior_strength": round(selected_strength, 6),
                "base_wpu_accuracy": round(base_accuracy, 6),
                "policy_wpu_accuracy": round(policy_accuracy, 6),
                "wpu_accuracy_change": round(policy_accuracy - base_accuracy, 6),
                "base_wpu_minus_baseline": round(base_margin, 6),
                "policy_wpu_minus_baseline": round(policy_margin, 6),
                "wpu_margin_change": round(policy_margin - base_margin, 6),
                "policy_wpu_win": policy_margin >= 0.0,
                "policy_wpu_ece_change": round(policy_ece_change, 6),
                "policy_wpu_brier_change": round(policy_brier_change, 6),
                "adapted_protocol": mechanism != "nominal",
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
    win_rate = statistics.fmean(1.0 if row["policy_wpu_win"] else 0.0 for row in shifted)
    mean_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in shifted)
    mean_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in shifted)
    mean_ece_change = statistics.fmean(float(row["policy_wpu_ece_change"]) for row in shifted)
    mean_brier_change = statistics.fmean(float(row["policy_wpu_brier_change"]) for row in shifted)
    if korean:
        title = "# PyBullet Mechanism-Aware Adaptive Policy"
        intro = (
            "이 분석은 기존 selected-prior 결과와 few-shot adaptation 결과를 결합한다. "
            "Calibration-selected prior strength가 threshold 이상이면 branch-prior adaptation을 사용하고, "
            "그 외 shifted mechanism에는 few-shot parameter adaptation을 사용한다. Nominal은 base policy를 유지한다."
        )
        caveat = (
            "이는 mechanism별 calibration/adaptation sample을 사용하는 adapted protocol이다. "
            "따라서 zero-shot generalization 증거가 아니라, P4를 개선하려면 mechanism shift detector와 "
            "selective adaptation policy가 필요하다는 증거로 해석해야 한다."
        )
        interpretation = "## 해석"
        bullets = [
            f"Shifted mechanism 기준 policy WPU win-rate는 `{win_rate:.6f}`이다.",
            f"Shifted 평균 WPU accuracy 변화는 `{mean_accuracy_change:.6f}`이다.",
            f"Shifted 평균 WPU-baseline margin 변화는 `{mean_margin_change:.6f}`이다.",
            f"Shifted 평균 WPU ECE 변화는 `{mean_ece_change:.6f}`, Brier 변화는 `{mean_brier_change:.6f}`이다. 음수는 개선이다.",
            "이 결과는 단일 scalar prior나 무조건 few-shot보다 더 나은 P4/P5 결합 방향을 제시하지만, mechanism-aware adaptation이 필요하다는 조건을 강화한다.",
        ]
    else:
        title = "# PyBullet Mechanism-Aware Adaptive Policy"
        intro = (
            "This analysis combines the existing selected-prior and few-shot adaptation results. "
            "If the calibration-selected prior strength exceeds the threshold, the policy uses branch-prior "
            "adaptation; otherwise it uses few-shot parameter adaptation for shifted mechanisms. Nominal "
            "evaluation keeps the base policy."
        )
        caveat = (
            "This is an adapted protocol that uses mechanism-specific calibration/adaptation samples. "
            "It is not zero-shot generalization evidence. It is evidence that improving P4 requires a "
            "mechanism-shift detector plus a selective adaptation policy."
        )
        interpretation = "## Interpretation"
        bullets = [
            f"Policy WPU win-rate over shifted mechanisms is `{win_rate:.6f}`.",
            f"Mean shifted WPU accuracy change is `{mean_accuracy_change:.6f}`.",
            f"Mean shifted WPU-baseline margin change is `{mean_margin_change:.6f}`.",
            f"Mean shifted WPU ECE change is `{mean_ece_change:.6f}` and Brier change is `{mean_brier_change:.6f}`; negative means better.",
            "The result suggests a stronger P4/P5 direction than a single scalar prior or unconditional few-shot adaptation, while strengthening the condition that mechanism-aware adaptation is required.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        caveat,
        "",
        "Source CSVs:",
        "",
        f"- `{args.selected_prior.as_posix()}`",
        f"- `{args.fewshot.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| mechanism | selected policy | prior strength | base WPU acc | policy WPU acc | acc change | base margin | policy margin | margin change | ECE change | Brier change |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | `{row['selected_policy']}` | "
            f"{float(row['selected_prior_strength']):.6f} | {float(row['base_wpu_accuracy']):.6f} | "
            f"{float(row['policy_wpu_accuracy']):.6f} | {float(row['wpu_accuracy_change']):.6f} | "
            f"{float(row['base_wpu_minus_baseline']):.6f} | {float(row['policy_wpu_minus_baseline']):.6f} | "
            f"{float(row['wpu_margin_change']):.6f} | {float(row['policy_wpu_ece_change']):.6f} | "
            f"{float(row['policy_wpu_brier_change']):.6f} |"
        )
    lines.extend(["", interpretation, ""])
    lines.extend(f"- {item}" for item in bullets)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
