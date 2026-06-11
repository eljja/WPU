from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("docs/experiments/pybullet_state_integrity_audit.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/pybullet_correction_trigger_frontier.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_correction_trigger_frontier_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_correction_trigger_frontier_results.ko.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the P2 correction-trigger frontier from PyBullet state-integrity audits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    parser.add_argument("--model", default="wpu-cws-indexed-sparse")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--integrity-target", type=float, default=0.8)
    parser.add_argument("--low-correction-target", type=float, default=0.25)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    frontier = _frontier(rows, args.model, args.horizon, args.integrity_target, args.low_correction_target)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, frontier)
    args.out_md.write_text(_render(frontier, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(frontier, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _frontier(
    rows: list[dict[str, str]],
    model: str,
    horizon: int,
    integrity_target: float,
    low_correction_target: float,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and int(row["horizon"]) == horizon
        and (
            row["run_label"].startswith("selective_corrected")
            or row["run_label"] in {"finite_corrected", "finite_clamped", "guarded", "rollback"}
        )
    ]
    out = []
    for row in sorted(selected, key=lambda item: (float(item["correction_rate"]), item["run_label"])):
        integrity = float(row["state_integrity_score"])
        correction = float(row["correction_rate"])
        low_disruption = float(row["low_disruption_integrity_score"])
        out.append(
            {
                "run_label": row["run_label"],
                "model": row["model"],
                "horizon": int(row["horizon"]),
                "policy_family": _policy_family(row["run_label"]),
                "state_integrity_score": round(integrity, 6),
                "low_disruption_integrity_score": round(low_disruption, 6),
                "constraint_violations_per_step": round(float(row["constraint_violations_per_step"]), 6),
                "correction_rate": round(correction, 6),
                "corrected_object_fraction": round(float(row["corrected_object_fraction"]), 6),
                "rollback_rate": round(float(row["rollback_rate"]), 6),
                "escalation_rate": round(float(row["escalation_rate"]), 6),
                "meets_integrity_target": int(integrity >= integrity_target),
                "meets_low_correction_target": int(correction <= low_correction_target),
                "meets_joint_target": int(
                    _policy_family(row["run_label"]) == "correction_trigger"
                    and integrity >= integrity_target
                    and correction <= low_correction_target
                ),
            }
        )
    return out


def _policy_family(run_label: str) -> str:
    if run_label in {"guarded", "rollback"}:
        return "safety_baseline"
    if run_label == "finite_clamped":
        return "no_correction_baseline"
    return "correction_trigger"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no frontier rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    best_integrity = max(rows, key=lambda row: float(row["state_integrity_score"]))
    best_low_disruption = max(rows, key=lambda row: float(row["low_disruption_integrity_score"]))
    trigger_rows = [row for row in rows if row["policy_family"] == "correction_trigger"]
    low_correction_rows = [
        row
        for row in trigger_rows
        if float(row["correction_rate"]) <= args.low_correction_target
    ]
    best_low_correction = (
        max(low_correction_rows, key=lambda row: float(row["state_integrity_score"]))
        if low_correction_rows
        else None
    )
    joint_rows = [row for row in trigger_rows if int(row["meets_joint_target"]) == 1]
    if korean:
        title = "# PyBullet Correction-Trigger Frontier 결과"
        intro = (
            "이 감사는 P2의 남은 병목을 분리한다. Selective correction은 corrected-object fraction을 "
            "낮췄지만 correction trigger frequency는 높게 남아 있다. 이 표는 trigger를 줄이면 "
            "state integrity가 유지되는지 검사한다."
        )
        conclusion = (
            f"최고 integrity는 `{float(best_integrity['state_integrity_score']):.6f}` "
            f"(`{best_integrity['run_label']}`)이고 correction rate는 "
            f"`{float(best_integrity['correction_rate']):.6f}`이다. 최고 low-disruption score는 "
            f"`{float(best_low_disruption['low_disruption_integrity_score']):.6f}` "
            f"(`{best_low_disruption['run_label']}`)이다. "
            + (
                f"Correction-trigger 계열에서 correction rate <= `{args.low_correction_target:.2f}` 조건의 최고 integrity는 "
                f"`{float(best_low_correction['state_integrity_score']):.6f}` "
                f"(`{best_low_correction['run_label']}`)이다."
                if best_low_correction is not None
                else f"Correction rate <= `{args.low_correction_target:.2f}` 조건을 만족하는 row가 없다."
            )
            + f" Joint target을 만족한 row 수는 `{len(joint_rows)}`이다."
        )
        interpretation = [
            "Entropy gate는 correction rate를 낮추지만 integrity target을 유지하지 못한다.",
            "Raw-delta threshold는 이 설정에서 유효한 trigger가 아니며 correction을 거의 제거해 실패한다.",
            "P2의 다음 단계는 더 보수적인 threshold가 아니라 transition model 자체의 안정화 또는 learned trigger다.",
        ]
    else:
        title = "# PyBullet Correction-Trigger Frontier Results"
        intro = (
            "This audit isolates the remaining P2 bottleneck. Selective correction reduces "
            "the corrected-object fraction, but correction trigger frequency remains high. "
            "The frontier tests whether lower trigger frequency preserves state integrity."
        )
        conclusion = (
            f"The best integrity is `{float(best_integrity['state_integrity_score']):.6f}` "
            f"(`{best_integrity['run_label']}`) at correction rate "
            f"`{float(best_integrity['correction_rate']):.6f}`. The best low-disruption score is "
            f"`{float(best_low_disruption['low_disruption_integrity_score']):.6f}` "
            f"(`{best_low_disruption['run_label']}`). "
            + (
                f"Among correction-trigger policies under correction rate <= "
                f"`{args.low_correction_target:.2f}`, the best integrity is "
                f"`{float(best_low_correction['state_integrity_score']):.6f}` "
                f"(`{best_low_correction['run_label']}`)."
                if best_low_correction is not None
                else f"No correction-trigger row satisfies correction rate <= `{args.low_correction_target:.2f}`."
            )
            + f" Rows meeting the joint target: `{len(joint_rows)}`."
        )
        interpretation = [
            "Entropy gating lowers correction rate but does not preserve the integrity target.",
            "Raw-delta thresholding is not a useful trigger in this setting; it removes correction and fails.",
            "The next P2 step is not a stricter threshold, but a more stable transition model or a learned trigger.",
        ]

    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{args.input.as_posix()}`",
        "",
        f"Derived CSV: `{args.out_csv.as_posix()}`",
        "",
        conclusion,
        "",
        "| run | family | integrity | low-disruption | violations/step | correction | corrected objects | rollback | escalation | joint target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_label']}` | `{row['policy_family']}` | {float(row['state_integrity_score']):.6f} | "
            f"{float(row['low_disruption_integrity_score']):.6f} | "
            f"{float(row['constraint_violations_per_step']):.6f} | "
            f"{float(row['correction_rate']):.6f} | "
            f"{float(row['corrected_object_fraction']):.6f} | "
            f"{float(row['rollback_rate']):.6f} | {float(row['escalation_rate']):.6f} | "
            f"{row['meets_joint_target']} |"
        )
    lines.extend(["", "## Interpretation" if not korean else "## 해석", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
