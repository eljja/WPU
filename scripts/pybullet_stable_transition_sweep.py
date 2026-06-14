from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from scripts.analyze_state_integrity import _integrity_score, _low_disruption_integrity_score
from scripts.pybullet_closed_loop_rollout import _rollout_condition, _train_model


DEFAULT_OUT = Path("docs/experiments/pybullet_stable_transition_sweep.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_stable_transition_sweep_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_stable_transition_sweep_results.ko.md")


@dataclass(frozen=True)
class StabilityConfig:
    name: str
    delta_norm_penalty: float
    delta_target_norm_slack: float
    rollout_consistency_penalty: float
    rollout_consistency_slack: float
    state_validity_penalty: float


DEFAULT_CONFIGS = [
    StabilityConfig("baseline_finite", 0.0, 0.5, 0.0, 0.5, 0.0),
    StabilityConfig("delta_norm_mid", 0.20, 0.25, 0.0, 0.5, 0.0),
    StabilityConfig("delta_norm_strong", 0.50, 0.00, 0.0, 0.5, 0.0),
    StabilityConfig("validity_mid", 0.0, 0.5, 0.0, 0.5, 0.50),
    StabilityConfig("validity_strong", 0.0, 0.5, 0.0, 0.5, 2.00),
    StabilityConfig("combined_mid", 0.20, 0.25, 0.02, 0.25, 0.50),
    StabilityConfig("combined_strong", 0.50, 0.00, 0.05, 0.00, 2.00),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep transition-stability training objectives for the P2 long-horizon "
            "state-integrity bottleneck."
        )
    )
    parser.add_argument("--model", default="wpu-cws-indexed-sparse")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--delta-clip", type=float, default=0.25)
    parser.add_argument("--finite-delta-clamp", type=float, default=1.0)
    parser.add_argument("--max-position-norm", type=float, default=25.0)
    parser.add_argument("--max-velocity-norm", type=float, default=25.0)
    parser.add_argument("--min-cup-z", type=float, default=-0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.report_only:
        rows = _read_csv(args.out)
        args.out_md.write_text(_render_report(rows, args.out, korean=False), encoding="utf-8")
        args.out_ko_md.write_text(_render_report(rows, args.out, korean=True), encoding="utf-8")
        print(f"wrote={args.out_md}", flush=True)
        print(f"wrote={args.out_ko_md}", flush=True)
        return

    rows: list[dict[str, object]] = []
    for config in DEFAULT_CONFIGS:
        train_args = argparse.Namespace(**vars(args))
        train_args.delta_norm_penalty = config.delta_norm_penalty
        train_args.delta_target_norm_slack = config.delta_target_norm_slack
        train_args.rollout_consistency_penalty = config.rollout_consistency_penalty
        train_args.rollout_consistency_slack = config.rollout_consistency_slack
        train_args.state_validity_penalty = config.state_validity_penalty
        for seed in args.seeds:
            print(f"train config={config.name} model={args.model} seed={seed}", flush=True)
            model = _train_model(args.model, seed, train_args)
            rows.extend(_evaluate_config_seed(model, config, seed, train_args))
            _write_csv(args.out, rows)

    summary = _summary_rows(rows)
    rows.extend(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(rows, args.out, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(rows, args.out, korean=True), encoding="utf-8")
    print(f"wrote={args.out}", flush=True)
    print(f"wrote={args.out_md}", flush=True)
    print(f"wrote={args.out_ko_md}", flush=True)


def _evaluate_config_seed(
    model: torch.nn.Module,
    config: StabilityConfig,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    eval_modes = [
        ("raw_finite_clamped", False),
        ("selective_corrected", True),
    ]
    rows: list[dict[str, object]] = []
    for eval_mode, correct in eval_modes:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.horizons = [args.horizon]
        eval_args.delta_clip = args.delta_clip
        eval_args.finite_delta_clamp = args.finite_delta_clamp
        eval_args.correct_on_violation = correct
        eval_args.rollback_on_violation = False
        eval_args.escalation_model = ""
        eval_args.selective_correction = correct
        eval_args.correction_violation_margin = 0
        eval_args.correction_stride = 1
        eval_args.correction_entropy_threshold = 0.0
        eval_args.correction_raw_delta_threshold = 0.0
        eval_args.unsafe_delta_reject_norm = 0.0
        eval_args.integrity_projection = False
        row = _rollout_condition(
            model,
            args.model,
            seed,
            args.horizon,
            eval_args,
        )
        row.update(
            {
                "row_type": "seed",
                "config": config.name,
                "eval_mode": eval_mode,
                "delta_norm_penalty": config.delta_norm_penalty,
                "delta_target_norm_slack": config.delta_target_norm_slack,
                "rollout_consistency_penalty": config.rollout_consistency_penalty,
                "rollout_consistency_slack": config.rollout_consistency_slack,
                "state_validity_penalty": config.state_validity_penalty,
            }
        )
        row.update(_score_row(row))
        rows.append(row)
    return rows


def _score_row(row: dict[str, object]) -> dict[str, object]:
    violations = float(row["constraint_violations_per_step"])
    delta_norm = float(row["delta_norm_mean"])
    flip_rate = float(row["branch_flip_rate"])
    rejection_rate = float(row.get("unsafe_delta_rejection_rate", 0.0))
    correction_rate = float(row.get("correction_rate", 0.0))
    corrected_object_fraction = float(row.get("corrected_object_fraction", 0.0))
    rollback_rate = float(row.get("rollback_rate", 0.0))
    escalation_rate = float(row.get("escalation_rate", 0.0))
    integrity = _integrity_score(violations, delta_norm, flip_rate)
    low_disruption = _low_disruption_integrity_score(
        integrity,
        rejection_rate,
        correction_rate,
        corrected_object_fraction,
        rollback_rate,
        escalation_rate,
    )
    return {
        "state_integrity_score": round(integrity, 6),
        "low_disruption_integrity_score": round(low_disruption, 6),
        "meets_joint_target": int(integrity >= 0.8 and correction_rate <= 0.25),
    }


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seed_rows = [row for row in rows if row.get("row_type") == "seed"]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in seed_rows:
        groups.setdefault((str(row["config"]), str(row["eval_mode"])), []).append(row)
    summary: list[dict[str, object]] = []
    for (config, eval_mode), group in sorted(groups.items()):
        base = dict(group[0])
        for key in [
            "seed",
            "samples",
            "branch_flip_rate",
            "constraint_violations_per_step",
            "branch_entropy_mean",
            "delta_norm_mean",
            "raw_delta_norm_mean",
            "unsafe_delta_rejection_rate",
            "correction_rate",
            "corrected_object_fraction",
            "rollback_rate",
            "escalation_rate",
            "escalation_success_rate",
            "selected_k_mean",
            "final_majority_branch_ratio",
            "state_integrity_score",
            "low_disruption_integrity_score",
            "meets_joint_target",
        ]:
            base.pop(key, None)
        base.update(
            {
                "row_type": "summary",
                "seed": "summary",
                "seed_count": len({row["seed"] for row in group}),
                "samples": sum(int(row["samples"]) for row in group),
            }
        )
        numeric_keys = [
            "branch_flip_rate",
            "constraint_violations_per_step",
            "branch_entropy_mean",
            "delta_norm_mean",
            "raw_delta_norm_mean",
            "unsafe_delta_rejection_rate",
            "correction_rate",
            "corrected_object_fraction",
            "rollback_rate",
            "escalation_rate",
            "escalation_success_rate",
            "selected_k_mean",
            "final_majority_branch_ratio",
            "state_integrity_score",
            "low_disruption_integrity_score",
        ]
        for key in numeric_keys:
            base[key] = round(sum(float(row.get(key, 0.0)) for row in group) / len(group), 6)
        base["meets_joint_target"] = int(
            float(base["state_integrity_score"]) >= 0.8 and float(base["correction_rate"]) <= 0.25
        )
        summary.append(base)
    return summary


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    summary = [row for row in rows if row.get("row_type") == "summary"]
    if not summary:
        summary = rows
    raw_rows = [row for row in summary if row.get("eval_mode") == "raw_finite_clamped"]
    corrected_rows = [row for row in summary if row.get("eval_mode") == "selective_corrected"]
    best_raw = max(raw_rows, key=lambda row: float(row["state_integrity_score"])) if raw_rows else None
    best_corrected = (
        max(corrected_rows, key=lambda row: float(row["low_disruption_integrity_score"])) if corrected_rows else None
    )
    joint_rows = [row for row in summary if int(float(row.get("meets_joint_target", 0))) == 1]
    if korean:
        title = "# PyBullet Stable Transition Sweep 결과"
        intro = (
            "이 실험은 P2의 다음 병목인 transition model 안정화를 직접 검사한다. "
            "delta-norm, rollout-consistency, state-validity loss 조합으로 WPU sparse "
            "transition을 다시 학습한 뒤, correction 없는 finite-clamped rollout과 "
            "selective correction rollout을 동시에 평가한다."
        )
        conclusion = _ko_conclusion(best_raw, best_corrected, joint_rows)
        interpretation = [
            "raw finite-clamped integrity가 오르면 transition 자체가 더 안정해졌다는 증거다.",
            "selective correction의 correction rate가 낮아지면서 integrity가 유지되면 P2 병목이 실제로 완화된 것이다.",
            "둘 다 실패하면 P2는 손실 가중치 조합이 아니라 모델 구조, multi-step supervision, 또는 simulator-resynchronized training이 필요하다.",
        ]
    else:
        title = "# PyBullet Stable Transition Sweep Results"
        intro = (
            "This experiment tests the next P2 bottleneck directly: whether transition "
            "training objectives can stabilize sparse WPU rollouts before correction. "
            "It trains WPU sparse with combinations of delta-norm, rollout-consistency, "
            "and state-validity losses, then evaluates both raw finite-clamped rollout "
            "and selective-correction rollout."
        )
        conclusion = _en_conclusion(best_raw, best_corrected, joint_rows)
        interpretation = [
            "Higher raw finite-clamped integrity is evidence that the transition model itself is more stable.",
            "Lower selective-correction rate at preserved integrity would mean the P2 correction-frequency bottleneck is reduced.",
            "If both fail, P2 needs architecture or multi-step/simulator-resynchronized training rather than more loss-weight tuning.",
        ]
    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{source_csv.as_posix()}`",
        "",
        conclusion,
        "",
        "| config | eval mode | integrity | low-disruption | violations/step | correction | corrected objects | delta norm | joint target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        summary,
        key=lambda item: (
            str(item.get("eval_mode")),
            -float(item.get("low_disruption_integrity_score", 0.0)),
            -float(item.get("state_integrity_score", 0.0)),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('config')}`",
                    f"`{row.get('eval_mode')}`",
                    f"{float(row.get('state_integrity_score', 0.0)):.6f}",
                    f"{float(row.get('low_disruption_integrity_score', 0.0)):.6f}",
                    f"{float(row.get('constraint_violations_per_step', 0.0)):.6f}",
                    f"{float(row.get('correction_rate', 0.0)):.6f}",
                    f"{float(row.get('corrected_object_fraction', 0.0)):.6f}",
                    f"{float(row.get('delta_norm_mean', 0.0)):.6f}",
                    str(int(float(row.get("meets_joint_target", 0)))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    lines.append("")
    return "\n".join(lines)


def _en_conclusion(
    best_raw: dict[str, object] | None,
    best_corrected: dict[str, object] | None,
    joint_rows: list[dict[str, object]],
) -> str:
    parts = []
    if best_raw is not None:
        parts.append(
            "Best raw finite-clamped integrity is "
            f"`{float(best_raw['state_integrity_score']):.6f}` "
            f"(`{best_raw['config']}`), with violations/step "
            f"`{float(best_raw['constraint_violations_per_step']):.6f}`."
        )
    if best_corrected is not None:
        parts.append(
            "Best selective-correction low-disruption score is "
            f"`{float(best_corrected['low_disruption_integrity_score']):.6f}` "
            f"(`{best_corrected['config']}`), with integrity "
            f"`{float(best_corrected['state_integrity_score']):.6f}` and correction rate "
            f"`{float(best_corrected['correction_rate']):.6f}`."
        )
    parts.append(
        f"Rows meeting the joint target (integrity >= 0.8 and correction_rate <= 0.25): `{len(joint_rows)}`."
    )
    return " ".join(parts)


def _ko_conclusion(
    best_raw: dict[str, object] | None,
    best_corrected: dict[str, object] | None,
    joint_rows: list[dict[str, object]],
) -> str:
    parts = []
    if best_raw is not None:
        parts.append(
            "최고 raw finite-clamped integrity는 "
            f"`{float(best_raw['state_integrity_score']):.6f}`"
            f"(`{best_raw['config']}`)이고 violations/step은 "
            f"`{float(best_raw['constraint_violations_per_step']):.6f}`다."
        )
    if best_corrected is not None:
        parts.append(
            "최고 selective-correction low-disruption score는 "
            f"`{float(best_corrected['low_disruption_integrity_score']):.6f}`"
            f"(`{best_corrected['config']}`)이고 integrity는 "
            f"`{float(best_corrected['state_integrity_score']):.6f}`, correction rate는 "
            f"`{float(best_corrected['correction_rate']):.6f}`다."
        )
    parts.append(
        f"joint target(integrity >= 0.8 및 correction_rate <= 0.25)을 만족한 row는 `{len(joint_rows)}`개다."
    )
    return " ".join(parts)


if __name__ == "__main__":
    main()
