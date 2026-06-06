from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PyBullet composition-shift stress results.")
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/pybullet_shift_composition_stress.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_shift_composition_stress_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_shift_composition_stress_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_shift_composition_stress_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _summary(_read_summary_rows(args.input))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args.input, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args.input, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("row_type") == "summary"]


def _summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = [row for row in group if row["model"].startswith("wpu-")]
        baseline = [row for row in group if not row["model"].startswith("wpu-")]
        best_wpu = max(wpu, key=lambda row: float(row["branch_accuracy"]))
        best_baseline = max(baseline, key=lambda row: float(row["branch_accuracy"]))
        output.append(
            {
                "eval_mechanism": mechanism,
                "train_mechanism": group[0]["train_mechanism"],
                "best_wpu_model": best_wpu["model"],
                "best_baseline_model": best_baseline["model"],
                "best_wpu_accuracy": round(float(best_wpu["branch_accuracy"]), 6),
                "best_baseline_accuracy": round(float(best_baseline["branch_accuracy"]), 6),
                "accuracy_delta": round(float(best_wpu["branch_accuracy"]) - float(best_baseline["branch_accuracy"]), 6),
                "wpu_win": float(best_wpu["branch_accuracy"]) >= float(best_baseline["branch_accuracy"]),
                "best_wpu_ece": round(float(best_wpu["ece"]), 6),
                "best_baseline_ece": round(float(best_baseline["ece"]), 6),
                "ece_ratio": round(float(best_wpu["ece"]) / max(float(best_baseline["ece"]), 1e-8), 6),
                "best_wpu_brier": round(float(best_wpu["brier"]), 6),
                "best_baseline_brier": round(float(best_baseline["brier"]), 6),
                "best_wpu_nll": round(float(best_wpu["nll"]), 6),
                "best_baseline_nll": round(float(best_baseline["nll"]), 6),
                "seed_count": int(float(group[0]["seed_count"])),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], input_csv: Path, output_csv: Path, *, korean: bool) -> str:
    win_rate = statistics.fmean(1.0 if row["wpu_win"] else 0.0 for row in rows)
    mean_delta = statistics.fmean(float(row["accuracy_delta"]) for row in rows)
    mean_ece_ratio = statistics.fmean(float(row["ece_ratio"]) for row in rows)
    if korean:
        title = "# PyBullet Composition-Shift Stress"
        intro = (
            "이 실험은 nominal/high_force/edge_shift/catch_heavy로 학습한 모델을 "
            "새로운 조합형 mechanism에서 평가한다."
        )
        interpretation = [
            f"WPU win-rate는 `{win_rate:.6f}`, 평균 accuracy delta는 `{mean_delta:.6f}`다.",
            f"평균 ECE ratio는 `{mean_ece_ratio:.6f}`이며, 1보다 작으면 best WPU의 ECE가 best baseline보다 낮다는 뜻이다.",
            "이 stress test는 단일 held-out family보다 어렵다. compound shift에서 지면 WPU 주장은 local-state regime으로 더 좁혀야 한다.",
        ]
    else:
        title = "# PyBullet Composition-Shift Stress"
        intro = (
            "This experiment trains on nominal/high_force/edge_shift/catch_heavy "
            "and evaluates on unseen composition mechanisms."
        )
        interpretation = [
            f"WPU win-rate is `{win_rate:.6f}` and mean accuracy delta is `{mean_delta:.6f}`.",
            f"Mean ECE ratio is `{mean_ece_ratio:.6f}`; values below 1 mean the best WPU has lower ECE than the best baseline.",
            "This stress test is harder than single held-out families. Failures here narrow the WPU claim to local-state regimes rather than broad shift generalization.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSV:",
        "",
        f"- `{input_csv.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{output_csv.as_posix()}`",
        "",
        "| mechanism | best WPU | best baseline | WPU acc | baseline acc | delta | WPU ECE | baseline ECE | ECE ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | `{row['best_wpu_model']}` | `{row['best_baseline_model']}` | "
            f"{float(row['best_wpu_accuracy']):.6f} | {float(row['best_baseline_accuracy']):.6f} | "
            f"{float(row['accuracy_delta']):.6f} | {float(row['best_wpu_ece']):.6f} | "
            f"{float(row['best_baseline_ece']):.6f} | {float(row['ece_ratio']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
