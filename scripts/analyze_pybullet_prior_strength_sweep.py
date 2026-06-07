from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


WPU_PREFIX = "wpu-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze mechanism-prior strength sweeps for accuracy/calibration tradeoffs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_generalization_prior_strength_sweep.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/pybullet_prior_strength_sweep_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_prior_strength_sweep_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_prior_strength_sweep_results.ko.md"),
    )
    args = parser.parse_args()

    details, aggregates = _analyze(_read_summary(args.input))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, details + aggregates)
    args.out_md.write_text(_render(details, aggregates, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(details, aggregates, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = [row for row in rows if row.get("row_type") == "summary"]
    if summary:
        return summary
    return _summarize_seed_rows([row for row in rows if row.get("row_type") == "seed"])


def _summarize_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row["model"],
                row["train_mechanism"],
                row["eval_mechanism"],
                _strength(row),
            ),
            [],
        ).append(row)
    numeric_fields = [
        "branch_accuracy",
        "majority_accuracy",
        "mse",
        "nll",
        "brier",
        "ece",
        "selected_k_mean",
        "causal_recall_mean",
        "dense_compute_ratio",
        "temperature",
        "mechanism_prior_strength",
    ]
    output: list[dict[str, str]] = []
    for (model, train_mechanism, eval_mechanism, strength), group in sorted(grouped.items()):
        row = {
            "row_type": "summary",
            "model": model,
            "seed": "all",
            "train_mechanism": train_mechanism,
            "eval_mechanism": eval_mechanism,
            "calibration_mode": group[0].get("calibration_mode", "none"),
            "calibration_bias": "",
            "background_objects": group[0].get("background_objects", ""),
            "total_objects_n": group[0].get("total_objects_n", ""),
            "samples": group[0].get("samples", ""),
            "seed_count": str(len(group)),
        }
        for field in numeric_fields:
            row[field] = f"{statistics.fmean(float(item.get(field) or 0.0) for item in group):.6f}"
        row["mechanism_prior_strength"] = f"{strength:.6f}"
        output.append(row)
    return output


def _strength(row: dict[str, str]) -> float:
    return round(float(row.get("mechanism_prior_strength") or 0.0), 6)


def _best_pair(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    wpu = [row for row in rows if row["model"].startswith(WPU_PREFIX)]
    baseline = [row for row in rows if not row["model"].startswith(WPU_PREFIX)]
    return (
        max(wpu, key=lambda row: float(row["branch_accuracy"])),
        max(baseline, key=lambda row: float(row["branch_accuracy"])),
    )


def _analyze(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    strengths = sorted({_strength(row) for row in rows})
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    shifted = [mechanism for mechanism in mechanisms if mechanism != "nominal"]
    details: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    for strength in strengths:
        strength_rows = [row for row in rows if _strength(row) == strength]
        strength_details: list[dict[str, object]] = []
        for mechanism in mechanisms:
            group = [row for row in strength_rows if row["eval_mechanism"] == mechanism]
            if not group:
                continue
            best_wpu, best_baseline = _best_pair(group)
            majority_accuracy = statistics.fmean(float(row["majority_accuracy"]) for row in group)
            wpu_accuracy = float(best_wpu["branch_accuracy"])
            baseline_accuracy = float(best_baseline["branch_accuracy"])
            detail = {
                "row_type": "mechanism",
                "mechanism_prior_strength": strength,
                "eval_mechanism": mechanism,
                "best_wpu": best_wpu["model"],
                "best_baseline": best_baseline["model"],
                "wpu_accuracy": round(wpu_accuracy, 6),
                "baseline_accuracy": round(baseline_accuracy, 6),
                "wpu_minus_baseline": round(wpu_accuracy - baseline_accuracy, 6),
                "majority_accuracy": round(majority_accuracy, 6),
                "majority_minus_wpu": round(majority_accuracy - wpu_accuracy, 6),
                "wpu_ece": round(float(best_wpu["ece"]), 6),
                "baseline_ece": round(float(best_baseline["ece"]), 6),
                "wpu_brier": round(float(best_wpu["brier"]), 6),
                "baseline_brier": round(float(best_baseline["brier"]), 6),
                "wpu_win": wpu_accuracy >= baseline_accuracy,
                "prior_dominated": majority_accuracy > wpu_accuracy and majority_accuracy > baseline_accuracy,
                "shifted": mechanism != "nominal",
            }
            details.append(detail)
            if mechanism != "nominal":
                strength_details.append(detail)
        aggregates.append(_aggregate(strength, strength_details))
    return details, aggregates


def _aggregate(strength: float, details: list[dict[str, object]]) -> dict[str, object]:
    return {
        "row_type": "aggregate",
        "mechanism_prior_strength": strength,
        "eval_mechanism": "shifted",
        "best_wpu": "",
        "best_baseline": "",
        "wpu_accuracy": round(statistics.fmean(float(row["wpu_accuracy"]) for row in details), 6),
        "baseline_accuracy": round(statistics.fmean(float(row["baseline_accuracy"]) for row in details), 6),
        "wpu_minus_baseline": round(statistics.fmean(float(row["wpu_minus_baseline"]) for row in details), 6),
        "majority_accuracy": round(statistics.fmean(float(row["majority_accuracy"]) for row in details), 6),
        "majority_minus_wpu": round(statistics.fmean(float(row["majority_minus_wpu"]) for row in details), 6),
        "wpu_ece": round(statistics.fmean(float(row["wpu_ece"]) for row in details), 6),
        "baseline_ece": round(statistics.fmean(float(row["baseline_ece"]) for row in details), 6),
        "wpu_brier": round(statistics.fmean(float(row["wpu_brier"]) for row in details), 6),
        "baseline_brier": round(statistics.fmean(float(row["baseline_brier"]) for row in details), 6),
        "wpu_win": "",
        "prior_dominated": sum(1 for row in details if row["prior_dominated"]),
        "shifted": True,
        "shifted_wpu_win_rate": round(statistics.fmean(1.0 if row["wpu_win"] else 0.0 for row in details), 6),
        "shifted_mechanism_count": len(details),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "row_type",
        "mechanism_prior_strength",
        "eval_mechanism",
        "best_wpu",
        "best_baseline",
        "wpu_accuracy",
        "baseline_accuracy",
        "wpu_minus_baseline",
        "majority_accuracy",
        "majority_minus_wpu",
        "wpu_ece",
        "baseline_ece",
        "wpu_brier",
        "baseline_brier",
        "wpu_win",
        "prior_dominated",
        "shifted",
        "shifted_wpu_win_rate",
        "shifted_mechanism_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render(
    details: list[dict[str, object]],
    aggregates: list[dict[str, object]],
    args: argparse.Namespace,
    *,
    korean: bool,
) -> str:
    zero = min(aggregates, key=lambda row: abs(float(row["mechanism_prior_strength"])))
    safe = [
        row
        for row in aggregates
        if float(row["mechanism_prior_strength"]) > 0.0
        and float(row["shifted_wpu_win_rate"]) >= float(zero["shifted_wpu_win_rate"])
        and float(row["wpu_ece"]) <= float(zero["wpu_ece"])
    ]
    best_accuracy = max(
        aggregates,
        key=lambda row: (
            float(row["shifted_wpu_win_rate"]),
            float(row["wpu_accuracy"]),
            -float(row["wpu_ece"]),
        ),
    )
    best_safe = (
        max(
            safe,
            key=lambda row: (
                float(row["shifted_wpu_win_rate"]),
                float(row["wpu_accuracy"]),
                -float(row["wpu_ece"]),
            ),
        )
        if safe
        else None
    )
    if korean:
        title = "# PyBullet Prior-Strength Sweep"
        intro = (
            "이 실험은 mechanism-aware branch prior를 한 번 학습한 모델에 여러 강도로 적용한다. "
            "목표는 prior adaptation이 정확도 개선뿐 아니라 calibration-safe하게 적용될 수 있는지 확인하는 것이다."
        )
        safe_text = (
            f"`strength={float(best_safe['mechanism_prior_strength']):.2f}`가 `strength=0` 대비 "
            f"win-rate를 유지/개선하면서 ECE를 악화시키지 않는다."
            if best_safe is not None
            else "`strength=0` 대비 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 강도는 발견되지 않았다."
        )
        interpretation = [
            f"정확도 기준 best strength는 `{float(best_accuracy['mechanism_prior_strength']):.2f}`이며 shifted WPU win-rate는 `{float(best_accuracy['shifted_wpu_win_rate']):.6f}`이다.",
            f"Calibration-safe 판정: {safe_text}",
            "따라서 v2의 다음 개선점은 단순 prior bias가 아니라 confidence-aware strength selection, mechanism uncertainty, 또는 per-class calibration이다.",
        ]
    else:
        title = "# PyBullet Prior-Strength Sweep"
        intro = (
            "This experiment applies mechanism-aware branch priors at multiple strengths to the same trained "
            "models. The goal is to test whether prior adaptation can improve accuracy without degrading "
            "calibration relative to a zero-strength prior."
        )
        safe_text = (
            f"`strength={float(best_safe['mechanism_prior_strength']):.2f}` preserves or improves win-rate "
            f"relative to `strength=0` without increasing ECE."
            if best_safe is not None
            else "No nonzero strength preserves or improves win-rate relative to `strength=0` without increasing ECE."
        )
        interpretation = [
            f"The accuracy-best strength is `{float(best_accuracy['mechanism_prior_strength']):.2f}` with shifted WPU win-rate `{float(best_accuracy['shifted_wpu_win_rate']):.6f}`.",
            f"Calibration-safe result: {safe_text}",
            "The v2 fix should therefore move beyond a fixed prior bias toward confidence-aware strength selection, mechanism uncertainty, or per-class calibration.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSV:",
        "",
        f"- `{args.input.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| strength | shifted WPU win-rate | WPU acc | baseline acc | WPU-baseline | WPU ECE | WPU Brier | prior-dominated shifts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {float(row['mechanism_prior_strength']):.2f} | "
            f"{float(row['shifted_wpu_win_rate']):.6f} | {float(row['wpu_accuracy']):.6f} | "
            f"{float(row['baseline_accuracy']):.6f} | {float(row['wpu_minus_baseline']):.6f} | "
            f"{float(row['wpu_ece']):.6f} | {float(row['wpu_brier']):.6f} | "
            f"{int(row['prior_dominated'])} |"
        )
    lines.extend(["", "## Shift Detail", ""])
    lines.extend(
        [
            "| strength | mechanism | best WPU | best baseline | WPU acc | baseline acc | WPU-baseline | WPU ECE | prior-dominated |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in details:
        if row["eval_mechanism"] == "nominal":
            continue
        lines.append(
            f"| {float(row['mechanism_prior_strength']):.2f} | {row['eval_mechanism']} | "
            f"{row['best_wpu']} | {row['best_baseline']} | {float(row['wpu_accuracy']):.6f} | "
            f"{float(row['baseline_accuracy']):.6f} | {float(row['wpu_minus_baseline']):.6f} | "
            f"{float(row['wpu_ece']):.6f} | {row['prior_dominated']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
