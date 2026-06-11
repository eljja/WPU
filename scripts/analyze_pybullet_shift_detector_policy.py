from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_SELECTED = Path("docs/experiments/pybullet_selected_prior_adaptation_summary.csv")
DEFAULT_FEWSHOT = Path("docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/pybullet_shift_detector_policy.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_shift_detector_policy_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_shift_detector_policy_results.ko.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a calibration-statistic detector for selecting PyBullet mechanism adaptation policies."
    )
    parser.add_argument("--selected-prior", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--fewshot", type=Path, default=DEFAULT_FEWSHOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    parser.add_argument(
        "--shift-ece-thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--prior-gap-thresholds",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10, 0.20, 0.30],
    )
    args = parser.parse_args()

    rows = _sweep(
        _read_rows(args.selected_prior),
        _read_rows(args.fewshot),
        shift_ece_thresholds=args.shift_ece_thresholds,
        prior_gap_thresholds=args.prior_gap_thresholds,
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


def _sweep(
    selected_rows: list[dict[str, str]],
    fewshot_rows: list[dict[str, str]],
    *,
    shift_ece_thresholds: list[float],
    prior_gap_thresholds: list[float],
) -> list[dict[str, object]]:
    selected = _by_mechanism(selected_rows)
    fewshot = _by_mechanism(fewshot_rows)
    mechanisms = sorted(set(selected) & set(fewshot))
    out: list[dict[str, object]] = []
    for shift_threshold in shift_ece_thresholds:
        for prior_threshold in prior_gap_thresholds:
            decisions = [
                _decision(
                    mechanism,
                    selected[mechanism],
                    fewshot[mechanism],
                    shift_ece_threshold=shift_threshold,
                    prior_gap_threshold=prior_threshold,
                )
                for mechanism in mechanisms
            ]
            shifted = [row for row in decisions if row["eval_mechanism"] != "nominal"]
            nominal = next((row for row in decisions if row["eval_mechanism"] == "nominal"), None)
            mean_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in shifted)
            mean_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in shifted)
            mean_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in shifted)
            mean_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in shifted)
            shifted_win_rate = statistics.fmean(1.0 if row["policy_wpu_win"] else 0.0 for row in shifted)
            nominal_false_adaptation = 1 if nominal and nominal["selected_policy"] != "base" else 0
            out.append(
                {
                    "row_type": "summary",
                    "shift_ece_threshold": round(shift_threshold, 6),
                    "prior_gap_threshold": round(prior_threshold, 6),
                    "shifted_wpu_win_rate": round(shifted_win_rate, 6),
                    "mean_shifted_accuracy_change": round(mean_accuracy_change, 6),
                    "mean_shifted_margin_change": round(mean_margin_change, 6),
                    "mean_shifted_ece_change": round(mean_ece_change, 6),
                    "mean_shifted_brier_change": round(mean_brier_change, 6),
                    "nominal_false_adaptation": nominal_false_adaptation,
                    "selected_prior_count": sum(1 for row in shifted if row["selected_policy"] == "selected_prior"),
                    "fewshot_count": sum(1 for row in shifted if row["selected_policy"] == "fewshot_adaptation"),
                    "base_count": sum(1 for row in shifted if row["selected_policy"] == "base"),
                    "detector_score": round(
                        shifted_win_rate + mean_margin_change + 0.25 * max(0.0, -mean_ece_change) - nominal_false_adaptation,
                        6,
                    ),
                    "decisions": "; ".join(
                        f"{row['eval_mechanism']}={row['selected_policy']}" for row in decisions
                    ),
                }
            )
            out.extend(decisions)
    return out


def _decision(
    mechanism: str,
    selected: dict[str, str],
    fewshot: dict[str, str],
    *,
    shift_ece_threshold: float,
    prior_gap_threshold: float,
) -> dict[str, object]:
    base_accuracy = float(selected["base_wpu_accuracy"])
    base_margin = float(selected["base_wpu_minus_baseline"])
    base_ece = float(selected["base_wpu_ece"])
    base_brier = float(selected["base_wpu_brier"])
    prior_gap = float(selected["base_majority_minus_wpu"])

    if base_ece < shift_ece_threshold:
        policy = "base"
        accuracy = base_accuracy
        margin = base_margin
        ece = base_ece
        brier = base_brier
    elif prior_gap >= prior_gap_threshold:
        policy = "selected_prior"
        accuracy = float(selected["selected_wpu_accuracy"])
        margin = float(selected["selected_wpu_minus_baseline"])
        ece = float(selected["selected_wpu_ece"])
        brier = float(selected["selected_wpu_brier"])
    else:
        policy = "fewshot_adaptation"
        accuracy = float(fewshot["adapted_wpu_accuracy"])
        margin = float(fewshot["adapted_wpu_minus_baseline"])
        ece = float(fewshot["adapted_wpu_ece"])
        brier = float(fewshot["adapted_wpu_brier"])

    return {
        "row_type": "decision",
        "shift_ece_threshold": round(shift_ece_threshold, 6),
        "prior_gap_threshold": round(prior_gap_threshold, 6),
        "eval_mechanism": mechanism,
        "base_wpu_ece": round(base_ece, 6),
        "base_majority_minus_wpu": round(prior_gap, 6),
        "selected_policy": policy,
        "base_wpu_accuracy": round(base_accuracy, 6),
        "policy_wpu_accuracy": round(accuracy, 6),
        "wpu_accuracy_change": round(accuracy - base_accuracy, 6),
        "base_wpu_minus_baseline": round(base_margin, 6),
        "policy_wpu_minus_baseline": round(margin, 6),
        "wpu_margin_change": round(margin - base_margin, 6),
        "base_wpu_ece_metric": round(base_ece, 6),
        "policy_wpu_ece": round(ece, 6),
        "wpu_ece_change": round(ece - base_ece, 6),
        "policy_wpu_brier": round(brier, 6),
        "wpu_brier_change": round(brier - base_brier, 6),
        "policy_wpu_win": margin >= 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "row_type",
        "shift_ece_threshold",
        "prior_gap_threshold",
        "eval_mechanism",
        "base_wpu_ece",
        "base_majority_minus_wpu",
        "selected_policy",
        "base_wpu_accuracy",
        "policy_wpu_accuracy",
        "wpu_accuracy_change",
        "base_wpu_minus_baseline",
        "policy_wpu_minus_baseline",
        "wpu_margin_change",
        "base_wpu_ece_metric",
        "policy_wpu_ece",
        "wpu_ece_change",
        "policy_wpu_brier",
        "wpu_brier_change",
        "policy_wpu_win",
        "shifted_wpu_win_rate",
        "mean_shifted_accuracy_change",
        "mean_shifted_margin_change",
        "mean_shifted_ece_change",
        "mean_shifted_brier_change",
        "nominal_false_adaptation",
        "selected_prior_count",
        "fewshot_count",
        "base_count",
        "detector_score",
        "decisions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    summary = [row for row in rows if row["row_type"] == "summary"]
    safe = [row for row in summary if int(row["nominal_false_adaptation"]) == 0]
    best = max(summary, key=lambda row: float(row["detector_score"]))
    best_safe = max(safe, key=lambda row: float(row["detector_score"])) if safe else None
    best_key = (best["shift_ece_threshold"], best["prior_gap_threshold"])
    best_decisions = [
        row
        for row in rows
        if row["row_type"] == "decision"
        and row["shift_ece_threshold"] == best_key[0]
        and row["prior_gap_threshold"] == best_key[1]
    ]
    if korean:
        title = "# PyBullet Shift-Detector Adaptive Policy"
        intro = (
            "이 audit는 mechanism 이름을 직접 사용하지 않고 calibration statistics로 adaptation policy를 "
            "선택할 수 있는지 검사한다. Base WPU ECE가 threshold보다 낮으면 base policy를 유지하고, "
            "shift로 판단되면 majority-prior gap이 큰 경우 selected-prior를, 그 외에는 few-shot "
            "adaptation을 사용한다."
        )
        caveat = (
            "이 결과도 zero-shot은 아니다. Calibration labels와 mechanism-specific adaptation sample을 "
            "사용한다. 다만 기존 mechanism-aware policy보다 엄격하게 mechanism identity 대신 observable "
            "statistics로 detect-and-adapt 결정을 만든다."
        )
        interpretation = [
            f"Best detector score는 threshold `(ECE={float(best['shift_ece_threshold']):.2f}, gap={float(best['prior_gap_threshold']):.2f})`에서 `{float(best['detector_score']):.6f}`이다.",
            f"Best shifted win-rate는 `{float(best['shifted_wpu_win_rate']):.6f}`, mean accuracy change는 `{float(best['mean_shifted_accuracy_change']):.6f}`, mean ECE change는 `{float(best['mean_shifted_ece_change']):.6f}`이다.",
            (
                f"Nominal false adaptation이 없는 best-safe policy도 detector score `{float(best_safe['detector_score']):.6f}`를 달성한다."
                if best_safe is not None
                else "Nominal false adaptation이 없는 safe policy는 없다."
            ),
            "이는 P4의 다음 방향이 mechanism 이름 oracle이 아니라 calibration-statistic detector와 selective adaptation임을 지지한다.",
        ]
        heading = "## 해석"
    else:
        title = "# PyBullet Shift-Detector Adaptive Policy"
        intro = (
            "This audit tests whether adaptation can be selected from calibration statistics instead of "
            "direct mechanism identity. If base WPU ECE is below a threshold, the detector keeps the base "
            "policy. Otherwise it selects branch-prior adaptation when the majority-prior gap is large and "
            "few-shot adaptation otherwise."
        )
        caveat = (
            "This is still not zero-shot: it uses calibration labels and mechanism-specific adaptation samples. "
            "It is stricter than the previous mechanism-aware policy because the decision is made from observable "
            "statistics rather than the mechanism name."
        )
        interpretation = [
            f"Best detector score occurs at thresholds `(ECE={float(best['shift_ece_threshold']):.2f}, gap={float(best['prior_gap_threshold']):.2f})` with score `{float(best['detector_score']):.6f}`.",
            f"Its shifted win-rate is `{float(best['shifted_wpu_win_rate']):.6f}`, mean accuracy change is `{float(best['mean_shifted_accuracy_change']):.6f}`, and mean ECE change is `{float(best['mean_shifted_ece_change']):.6f}`.",
            (
                f"The best-safe policy with no nominal false adaptation also reaches detector score `{float(best_safe['detector_score']):.6f}`."
                if best_safe is not None
                else "No safe policy avoids nominal false adaptation."
            ),
            "This supports the next P4 direction: calibration-statistic detection plus selective adaptation, not a mechanism-name oracle.",
        ]
        heading = "## Interpretation"

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
        "| ECE threshold | prior-gap threshold | shifted win-rate | acc change | margin change | ECE change | Brier change | nominal false adaptation | score | decisions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summary, key=lambda item: float(item["detector_score"]), reverse=True)[:12]:
        lines.append(
            f"| {float(row['shift_ece_threshold']):.2f} | {float(row['prior_gap_threshold']):.2f} | "
            f"{float(row['shifted_wpu_win_rate']):.6f} | {float(row['mean_shifted_accuracy_change']):.6f} | "
            f"{float(row['mean_shifted_margin_change']):.6f} | {float(row['mean_shifted_ece_change']):.6f} | "
            f"{float(row['mean_shifted_brier_change']):.6f} | {row['nominal_false_adaptation']} | "
            f"{float(row['detector_score']):.6f} | `{row['decisions']}` |"
        )
    decision_heading = "## 최선 결정" if korean else "## Best Decisions"
    lines.extend(["", decision_heading, ""])
    lines.extend(
        f"- `{row['eval_mechanism']}`: `{row['selected_policy']}` "
        f"(base ECE `{float(row['base_wpu_ece']):.6f}`, prior gap `{float(row['base_majority_minus_wpu']):.6f}`)"
        for row in best_decisions
    )
    lines.extend(["", heading, ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
