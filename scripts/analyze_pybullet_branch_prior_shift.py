from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


WPU_PREFIX = "wpu-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether PyBullet mechanism shifts are dominated by branch-prior shift."
    )
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/pybullet_shift_generalization.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_branch_prior_shift.csv"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_branch_prior_shift_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_branch_prior_shift_results.ko.md"),
    )
    args = parser.parse_args()

    rows = _summarize(_read_rows(args.input))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args.input, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args.input, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = [row for row in rows if row.get("row_type") == "summary"]
    if summary:
        return summary
    return rows


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for mechanism in sorted({row["eval_mechanism"] for row in rows}):
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = [row for row in group if row["model"].startswith(WPU_PREFIX)]
        baseline = [row for row in group if not row["model"].startswith(WPU_PREFIX)]
        best_wpu = max(wpu, key=lambda row: float(row["branch_accuracy"]))
        best_baseline = max(baseline, key=lambda row: float(row["branch_accuracy"]))
        majority_accuracy = statistics.fmean(float(row["majority_accuracy"]) for row in group)
        best_wpu_accuracy = float(best_wpu["branch_accuracy"])
        best_baseline_accuracy = float(best_baseline["branch_accuracy"])
        majority_minus_best_wpu = majority_accuracy - best_wpu_accuracy
        majority_minus_best_baseline = majority_accuracy - best_baseline_accuracy
        output.append(
            {
                "eval_mechanism": mechanism,
                "best_wpu_model": best_wpu["model"],
                "best_baseline_model": best_baseline["model"],
                "best_wpu_accuracy": round(best_wpu_accuracy, 6),
                "best_baseline_accuracy": round(best_baseline_accuracy, 6),
                "majority_accuracy": round(majority_accuracy, 6),
                "wpu_minus_baseline": round(best_wpu_accuracy - best_baseline_accuracy, 6),
                "majority_minus_best_wpu": round(majority_minus_best_wpu, 6),
                "majority_minus_best_baseline": round(majority_minus_best_baseline, 6),
                "branch_prior_dominates": majority_accuracy > best_wpu_accuracy
                and majority_accuracy > best_baseline_accuracy,
                "best_wpu_ece": round(float(best_wpu["ece"]), 6),
                "best_baseline_ece": round(float(best_baseline["ece"]), 6),
                "best_wpu_brier": round(float(best_wpu["brier"]), 6),
                "best_baseline_brier": round(float(best_baseline["brier"]), 6),
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
    prior_dominated = [row for row in rows if row["branch_prior_dominates"]]
    shifted = [row for row in rows if row["eval_mechanism"] != "nominal"]
    shifted_prior_dominated = [row for row in shifted if row["branch_prior_dominates"]]
    mean_wpu_delta = statistics.fmean(float(row["wpu_minus_baseline"]) for row in shifted)
    mean_prior_gap = statistics.fmean(float(row["majority_minus_best_wpu"]) for row in shifted)

    if korean:
        title = "# PyBullet Branch-Prior Shift Audit"
        intro = (
            "이 분석은 PyBullet mechanism-family shift에서 실패 원인이 relation/propagation 구조인지, "
            "아니면 branch label prior 변화인지 분리한다. `majority_accuracy`는 해당 eval mechanism에서 "
            "가장 흔한 branch만 항상 예측하는 비학습 prior baseline이다."
        )
        interpretation = [
            f"Shift mechanism 기준 평균 WPU-baseline accuracy delta는 `{mean_wpu_delta:.6f}`다.",
            f"Shift mechanism 기준 majority-prior와 best WPU의 평균 gap은 `{mean_prior_gap:.6f}`다.",
            f"Majority prior가 best WPU와 best baseline을 모두 이기는 prior-dominated mechanism은 `{len(shifted_prior_dominated)}/{len(shifted)}`개다.",
            "Prior-dominated mechanism에서는 더 큰 propagation block보다 mechanism-aware branch prior, branch-frequency shift detector, uncertainty-gated recompute가 먼저 필요하다.",
            "이 결과는 WPU 주장을 좁힌다. 객체화와 sparse propagation이 충분해도, branch prior가 바뀌면 state processor는 명시적인 prior adaptation 없이는 실패할 수 있다.",
        ]
    else:
        title = "# PyBullet Branch-Prior Shift Audit"
        intro = (
            "This analysis separates relation/propagation failure from branch-label prior shift in the "
            "PyBullet mechanism-family benchmark. `majority_accuracy` is the non-learned baseline that "
            "always predicts the most frequent branch in the evaluation mechanism."
        )
        interpretation = [
            f"Mean WPU-baseline accuracy delta over shifted mechanisms is `{mean_wpu_delta:.6f}`.",
            f"Mean majority-prior gap over the best WPU on shifted mechanisms is `{mean_prior_gap:.6f}`.",
            f"`{len(shifted_prior_dominated)}/{len(shifted)}` shifted mechanisms are prior-dominated, meaning the majority prior beats both the best WPU and the best non-WPU baseline.",
            "In prior-dominated mechanisms, mechanism-aware branch priors, branch-frequency shift detection, and uncertainty-gated recompute are higher-priority fixes than simply enlarging the propagation block.",
            "This narrows the WPU claim: objectification and sparse propagation are not enough when the branch prior itself shifts without explicit prior adaptation.",
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
        "| mechanism | best WPU | best baseline | WPU acc | baseline acc | majority acc | WPU-baseline | majority-WPU | prior dominated |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eval_mechanism']} | `{row['best_wpu_model']}` | `{row['best_baseline_model']}` | "
            f"{float(row['best_wpu_accuracy']):.6f} | {float(row['best_baseline_accuracy']):.6f} | "
            f"{float(row['majority_accuracy']):.6f} | {float(row['wpu_minus_baseline']):.6f} | "
            f"{float(row['majority_minus_best_wpu']):.6f} | {row['branch_prior_dominates']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    if prior_dominated:
        lines.extend(["", "## Mechanism Consequence", ""])
        if korean:
            lines.append(
                "Prior-dominated mechanism은 "
                + ", ".join(f"`{row['eval_mechanism']}`" for row in prior_dominated)
                + "이다. 이 구간은 WPU v2에서 P4/P5의 핵심 반례로 유지해야 한다."
            )
        else:
            lines.append(
                "The prior-dominated mechanisms are "
                + ", ".join(f"`{row['eval_mechanism']}`" for row in prior_dominated)
                + ". These should remain explicit P4/P5 counterexamples in WPU v2."
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
