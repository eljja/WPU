from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


ROOT = Path("docs/experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative WPU v2 priority dashboard from existing experiment CSVs."
    )
    parser.add_argument("--out-csv", type=Path, default=ROOT / "wpu_v2_priority_dashboard.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "wpu_v2_priority_dashboard.md")
    parser.add_argument("--out-ko-md", type=Path, default=ROOT / "wpu_v2_priority_dashboard.ko.md")
    args = parser.parse_args()

    rows = [
        _priority_candidate_oracle_gap(),
        _priority_state_integrity(),
        _priority_simulator_grounding(),
        _priority_shift_generalization(),
        _priority_calibration(),
        _priority_systems_profile(),
        _priority_objectification_quality(),
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render_markdown(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _priority_candidate_oracle_gap() -> dict[str, object]:
    path = ROOT / "wpu_v2_candidate_oracle_gap_v2.csv"
    rows = _read_rows(path)
    target_rows = [
        row
        for row in rows
        if row["feature_variant"] == "role_geometry_family"
        and row["policy"] == "risk_adjusted_selected_mechanism"
    ]
    aggregate_best = max(float(row["gap_closure_fraction"]) for row in target_rows)
    mean = statistics.fmean(float(row["gap_closure_fraction"]) for row in target_rows)
    noharm_path = ROOT / "wpu_v2_candidate_noharm_gate_summary.csv"
    noharm_best = None
    if noharm_path.exists():
        noharm_rows = _read_rows(noharm_path)
        noharm_best = max(float(row["gap_closure_fraction"]) for row in noharm_rows)
    noharm_note = (
        f" Sample-level no-harm/margin gates were also audited; best closure is {noharm_best:.6f}, so margin gating is not the missing fix."
        if noharm_best is not None
        else ""
    )
    regret_path = ROOT / "wpu_v2_candidate_regret_gate_summary.csv"
    penalty_path = ROOT / "wpu_v2_candidate_regret_gate_penalty_summary.csv"
    perturb_path = ROOT / "wpu_v2_candidate_regret_gate_perturbed_summary.csv"
    safety_path = ROOT / "wpu_v2_candidate_safety_gate_summary.csv"
    crossfit_path = ROOT / "wpu_v2_candidate_regret_crossfit_summary.csv"
    invariant_path = ROOT / "wpu_v2_candidate_invariant_gate_summary.csv"
    joint_path = ROOT / "wpu_v2_candidate_joint_gate_summary.csv"
    joint_regression_path = ROOT / "wpu_v2_candidate_joint_gate_regression_heavy_k16_summary.csv"
    regret_best = None
    regret_safe_best = None
    regret_unconstrained_best = None
    regret_best_harmful_accept = None
    regret_train_selected_best = None
    regret_train_selected_harmful_accept = None
    if regret_path.exists():
        regret_rows = _read_rows(regret_path)
        regret_unconstrained = max(regret_rows, key=lambda row: float(row["gap_closure_fraction"]))
        safe_regret_rows = [row for row in regret_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25]
        train_selected_rows = [row for row in regret_rows if row["policy"] == "train_selected_candidate_regret_gate"]
        train_selected_safe_rows = [
            row for row in train_selected_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        regret_best_row = None
        if train_selected_safe_rows:
            regret_best_row = max(train_selected_safe_rows, key=lambda row: float(row["gap_closure_fraction"]))
        elif train_selected_rows:
            regret_best_row = max(train_selected_rows, key=lambda row: float(row["gap_closure_fraction"]))
        elif safe_regret_rows:
            regret_best_row = max(safe_regret_rows, key=lambda row: float(row["gap_closure_fraction"]))
        else:
            regret_best_row = regret_unconstrained
        regret_best = float(regret_best_row["gap_closure_fraction"])
        regret_safe_best = max(float(row["gap_closure_fraction"]) for row in safe_regret_rows) if safe_regret_rows else None
        regret_unconstrained_best = float(regret_unconstrained["gap_closure_fraction"])
        regret_best_harmful_accept = float(regret_best_row.get("mean_harmful_accept_rate", 0.0))
        if train_selected_rows:
            train_selected_best_row = max(train_selected_rows, key=lambda row: float(row["gap_closure_fraction"]))
            regret_train_selected_best = float(train_selected_best_row["gap_closure_fraction"])
            regret_train_selected_harmful_accept = float(train_selected_best_row.get("mean_harmful_accept_rate", 0.0))
    if regret_best is not None:
        safe_text = (
            f"{regret_safe_best:.6f} under harmful-accept <= 0.25"
            if regret_safe_best is not None
            else "no safe deployed candidate under harmful-accept <= 0.25"
        )
        regret_note = (
            f" Direct candidate-regret gating reaches {regret_unconstrained_best:.6f} unconstrained and "
            f"{safe_text}; the selected deployment harmful-accept rate is {regret_best_harmful_accept:.6f}."
        )
        if regret_train_selected_best is not None:
            regret_note = (
                f" Direct candidate-regret gating reaches {regret_unconstrained_best:.6f} unconstrained and "
                f"{safe_text}; train-selected deployment reaches {regret_train_selected_best:.6f} "
                f"with harmful-accept {regret_train_selected_harmful_accept:.6f}. "
                f"The selected deployment harmful-accept rate is {regret_best_harmful_accept:.6f}."
            )
    else:
        regret_note = ""
    penalty_note = ""
    if penalty_path.exists():
        penalty_rows = _read_rows(penalty_path)
        penalty_train_selected = [
            row for row in penalty_rows if row["policy"] == "train_selected_candidate_regret_gate"
        ]
        if penalty_train_selected:
            penalty_best = max(penalty_train_selected, key=lambda row: float(row["gap_closure_fraction"]))
            penalty_note = (
                f" Harmful-accept/ranking-penalty training is safer but weaker: train-selected closure "
                f"{float(penalty_best['gap_closure_fraction']):.6f} with harmful-accept "
                f"{float(penalty_best.get('mean_harmful_accept_rate', 0.0)):.6f}."
            )
    perturb_note = ""
    if perturb_path.exists():
        perturb_rows = _read_rows(perturb_path)
        perturb_unconstrained = max(perturb_rows, key=lambda row: float(row["gap_closure_fraction"]))
        perturb_safe = [
            row for row in perturb_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        perturb_train = [
            row for row in perturb_rows if row["policy"] == "train_selected_candidate_regret_gate"
        ]
        if perturb_safe and perturb_train:
            safe_best = max(perturb_safe, key=lambda row: float(row["gap_closure_fraction"]))
            train_best = max(perturb_train, key=lambda row: float(row["gap_closure_fraction"]))
            perturb_note = (
                f" Feature perturbation improves test-sweep closure to "
                f"{float(perturb_unconstrained['gap_closure_fraction']):.6f} unconstrained and "
                f"{float(safe_best['gap_closure_fraction']):.6f} under harmful-accept <= 0.25, "
                f"but train-selected closure is {float(train_best['gap_closure_fraction']):.6f}."
            )
    safety_best = None
    safety_note = ""
    if safety_path.exists():
        safety_rows = _read_rows(safety_path)
        safety_best_row = max(safety_rows, key=lambda row: float(row["gap_closure_fraction"]))
        safety_best = float(safety_best_row["gap_closure_fraction"])
        safety_safe_rows = [
            row for row in safety_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        safety_train_rows = [
            row for row in safety_rows if row["policy"] == "train_selected_safety_utility_gate"
        ]
        safety_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in safety_safe_rows)
            if safety_safe_rows
            else None
        )
        safety_train_best = (
            max(float(row["gap_closure_fraction"]) for row in safety_train_rows)
            if safety_train_rows
            else None
        )
        safety_note = (
            f" A separate safety/utility head is a negative result: best closure is {safety_best:.6f}"
            + (
                f", safe best is {safety_safe_best:.6f}"
                if safety_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {safety_train_best:.6f}."
                if safety_train_best is not None
                else "."
            )
        )
    crossfit_best = None
    crossfit_note = ""
    if crossfit_path.exists():
        crossfit_rows = _read_rows(crossfit_path)
        crossfit_unconstrained = max(crossfit_rows, key=lambda row: float(row["gap_closure_fraction"]))
        crossfit_safe_rows = [
            row for row in crossfit_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        crossfit_selected_rows = [
            row for row in crossfit_rows if row["policy"] == "crossfit_selected_candidate_regret_gate"
        ]
        crossfit_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in crossfit_safe_rows)
            if crossfit_safe_rows
            else None
        )
        crossfit_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in crossfit_selected_rows)
            if crossfit_selected_rows
            else None
        )
        crossfit_best = crossfit_selected_best or float(crossfit_unconstrained["gap_closure_fraction"])
        crossfit_note = (
            f" Cross-fit ensemble regret gating is also a negative result for P1 improvement: "
            f"best closure is {float(crossfit_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {crossfit_safe_best:.6f}"
                if crossfit_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and cross-fit selected closure is {crossfit_selected_best:.6f}."
                if crossfit_selected_best is not None
                else "."
            )
        )
    invariant_best = None
    invariant_note = ""
    if invariant_path.exists():
        invariant_rows = _read_rows(invariant_path)
        invariant_unconstrained = max(invariant_rows, key=lambda row: float(row["gap_closure_fraction"]))
        invariant_safe_rows = [
            row for row in invariant_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        invariant_selected_rows = [
            row for row in invariant_rows if row["policy"] == "train_selected_invariant_gate"
        ]
        invariant_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in invariant_safe_rows)
            if invariant_safe_rows
            else None
        )
        invariant_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in invariant_selected_rows)
            if invariant_selected_rows
            else None
        )
        invariant_best = invariant_selected_best or float(invariant_unconstrained["gap_closure_fraction"])
        invariant_note = (
            f" Descriptor standardization plus group-DRO no-harm training remains insufficient: "
            f"best closure is {float(invariant_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {invariant_safe_best:.6f}"
                if invariant_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {invariant_selected_best:.6f}."
                if invariant_selected_best is not None
                else "."
            )
        )
    joint_best = None
    joint_note = ""
    if joint_path.exists():
        joint_rows = _read_rows(joint_path)
        joint_unconstrained = max(joint_rows, key=lambda row: float(row["gap_closure_fraction"]))
        joint_safe_rows = [
            row for row in joint_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        joint_selected_rows = [
            row for row in joint_rows if row["policy"] == "train_selected_joint_candidate_gate"
        ]
        joint_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in joint_safe_rows)
            if joint_safe_rows
            else None
        )
        joint_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in joint_selected_rows)
            if joint_selected_rows
            else None
        )
        joint_corr = statistics.fmean(
            float(row.get("mean_regret_corr", 0.0))
            for row in joint_rows
            if row["policy"] == joint_unconstrained["policy"]
        )
        joint_best = joint_selected_best or float(joint_unconstrained["gap_closure_fraction"])
        joint_note = (
            f" Joint object-set candidate gating is a negative result for P1: "
            f"best closure is {float(joint_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {joint_safe_best:.6f}"
                if joint_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", train-selected closure is {joint_selected_best:.6f}"
                if joint_selected_best is not None
                else ""
            )
            + f", and mean regret correlation is {joint_corr:.6f}."
        )
    joint_regression_note = ""
    if joint_regression_path.exists():
        joint_regression_rows = _read_rows(joint_regression_path)
        regression_best = max(joint_regression_rows, key=lambda row: float(row["gap_closure_fraction"]))
        regression_selected_rows = [
            row for row in joint_regression_rows if row["policy"] == "train_selected_joint_candidate_gate"
        ]
        regression_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in regression_selected_rows)
            if regression_selected_rows
            else None
        )
        joint_regression_note = (
            f" A regression-heavy joint gate ablation at K=16 is also negative: "
            f"best closure {float(regression_best['gap_closure_fraction']):.6f}"
            + (
                f", train-selected closure {regression_selected_best:.6f}."
                if regression_selected_best is not None
                else "."
            )
        )
    best = max(
        value
        for value in [
            aggregate_best,
            noharm_best,
            regret_best,
            safety_best,
            crossfit_best,
            invariant_best,
            joint_best,
        ]
        if value is not None
    )
    source = (
        joint_path
        if joint_best == best
        else invariant_path
        if invariant_best == best
        else regret_path
        if regret_best == best
        else path
    )
    return _row(
        1,
        "Candidate-oracle gap",
        "fail" if best < 0.5 else "partial",
        best,
        0.5,
        "gap_closure_fraction",
        source,
        f"Best deployed closure is {best:.6f}; previous aggregate-policy best is {aggregate_best:.6f} and mean aggregate closure is {mean:.6f}.{noharm_note}{regret_note}{penalty_note}{perturb_note}{safety_note}{crossfit_note}{invariant_note}{joint_note}{joint_regression_note}",
        "Move beyond post-hoc and object-set-only gates: jointly train retrieval, candidate generation, and propagation with calibrated no-harm objectives.",
    )


def _priority_state_integrity() -> dict[str, object]:
    path = ROOT / "pybullet_state_integrity_audit.csv"
    rows = _read_rows(path)
    wpu_h25 = [
        row
        for row in rows
        if row["model"].startswith("wpu-") and int(row["horizon"]) == 25
    ]
    best = max(float(row["state_integrity_score"]) for row in wpu_h25)
    sparse_clipped = next(
        float(row["state_integrity_score"])
        for row in wpu_h25
        if row["run_label"] == "clipped" and row["model"] == "wpu-cws-indexed-sparse"
    )
    sparse_guarded = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "guarded" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_regularized = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "regularized" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_rejected = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "rejected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_rejection_rate = next(
        (
            float(row.get("unsafe_delta_rejection_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "rejected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_consistency = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "consistency" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_validity = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "validity" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_validity_strong = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "validity_strong" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_rollback = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_rollback_rate = next(
        (
            float(row.get("rollback_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_corrected = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_correction_rate = next(
        (
            float(row.get("correction_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_corrected_rollback_rate = next(
        (
            float(row.get("rollback_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_escalated = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "escalated_corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_escalated_correction_rate = next(
        (
            float(row.get("correction_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "escalated_corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_escalated_rollback_rate = next(
        (
            float(row.get("rollback_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "escalated_corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_escalated_escalation_rate = next(
        (
            float(row.get("escalation_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "escalated_corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_escalated_success_rate = next(
        (
            float(row.get("escalation_success_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "escalated_corrected_rollback" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_finite_clamped = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "finite_clamped" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_finite_clamped_delta = next(
        (
            float(row.get("delta_norm_mean", 0.0))
            for row in wpu_h25
            if row["run_label"] == "finite_clamped" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_finite_corrected = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "finite_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_finite_corrected_correction_rate = next(
        (
            float(row.get("correction_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "finite_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_finite_corrected_rollback_rate = next(
        (
            float(row.get("rollback_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "finite_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_corrected = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "selective_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_corrected_low_disruption = next(
        (
            float(row.get("low_disruption_integrity_score", 0.0))
            for row in wpu_h25
            if row["run_label"] == "selective_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_corrected_correction_rate = next(
        (
            float(row.get("correction_rate", 0.0))
            for row in wpu_h25
            if row["run_label"] == "selective_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_corrected_object_fraction = next(
        (
            float(row.get("corrected_object_fraction", 0.0))
            for row in wpu_h25
            if row["run_label"] == "selective_corrected" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_stride2 = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "selective_corrected_stride2" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_selective_margin1 = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "selective_corrected_margin1" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    correction_note = (
        f" Corrected rollback sparse is {sparse_corrected:.6f} with correction rate "
        f"{sparse_correction_rate:.6f} and rollback rate {sparse_corrected_rollback_rate:.6f}."
        if sparse_corrected > 0.0
        else ""
    )
    escalation_note = (
        f" Escalated corrected rollback sparse is {sparse_escalated:.6f} with correction rate "
        f"{sparse_escalated_correction_rate:.6f}, rollback rate {sparse_escalated_rollback_rate:.6f}, "
        f"escalation rate {sparse_escalated_escalation_rate:.6f}, and escalation success "
        f"{sparse_escalated_success_rate:.6f}."
        if sparse_escalated > 0.0
        else ""
    )
    finite_note = (
        f" Finite-clamped sparse is {sparse_finite_clamped:.6f} with delta norm "
        f"{sparse_finite_clamped_delta:.6f}; finite-corrected sparse is "
        f"{sparse_finite_corrected:.6f} with correction rate "
        f"{sparse_finite_corrected_correction_rate:.6f} and rollback rate "
        f"{sparse_finite_corrected_rollback_rate:.6f}."
        if sparse_finite_corrected > 0.0
        else ""
    )
    selective_note = (
        f" Selective correction preserves sparse integrity {sparse_selective_corrected:.6f} "
        f"and improves low-disruption integrity to {sparse_selective_corrected_low_disruption:.6f} "
        f"by correcting only {sparse_selective_corrected_object_fraction:.6f} of objects when correction fires, "
        f"but the correction trigger rate remains {sparse_selective_corrected_correction_rate:.6f}. "
        f"Stride-2 and margin-1 correction gates collapse sparse integrity to "
        f"{sparse_selective_stride2:.6f} and {sparse_selective_margin1:.6f}, respectively."
        if sparse_selective_corrected > 0.0
        else ""
    )
    trigger_frontier_note = ""
    trigger_frontier_path = ROOT / "pybullet_correction_trigger_frontier.csv"
    if trigger_frontier_path.exists():
        trigger_rows = [
            row
            for row in _read_rows(trigger_frontier_path)
            if row.get("policy_family") == "correction_trigger"
        ]
        low_correction_rows = [
            row for row in trigger_rows if float(row.get("correction_rate", 1.0)) <= 0.25
        ]
        joint_rows = [row for row in trigger_rows if row.get("meets_joint_target") == "1"]
        if low_correction_rows:
            best_low_correction = max(
                low_correction_rows,
                key=lambda row: float(row["state_integrity_score"]),
            )
            trigger_frontier_note = (
                f" Correction-trigger frontier audit finds {len(joint_rows)} trigger policies meeting "
                f"integrity>=0.8 and correction_rate<=0.25; the best low-correction trigger is "
                f"{best_low_correction['run_label']} with integrity "
                f"{float(best_low_correction['state_integrity_score']):.6f} and correction rate "
                f"{float(best_low_correction['correction_rate']):.6f}."
            )
    return _row(
        2,
        "Long-horizon state integrity",
        "fail" if best < 0.8 else "partial",
        best,
        0.8,
        "best_wpu_h25_integrity",
        path,
        f"Best WPU H=25 integrity is {best:.6f}; guarded sparse is {sparse_guarded:.6f}, clipped sparse is {sparse_clipped:.6f}, regularized raw sparse is {sparse_regularized:.6f}, rollout-consistency sparse is {sparse_consistency:.6f}, validity sparse is {sparse_validity:.6f}, strong-validity sparse is {sparse_validity_strong:.6f}, unsafe-delta rejected sparse is {sparse_rejected:.6f} with rejection rate {sparse_rejection_rate:.6f}, and rollback sparse is {sparse_rollback:.6f} with rollback rate {sparse_rollback_rate:.6f}.{correction_note}{escalation_note}{finite_note}{selective_note}{trigger_frontier_note}",
        "Selective correction lowers correction disruption but not correction trigger frequency; the next step is a learned stable transition/correction trigger that preserves integrity without correcting most sparse updates.",
    )


def _priority_simulator_grounding() -> dict[str, object]:
    path = ROOT / "pybullet_cup_benchmark_7seed.csv"
    if not path.exists():
        path = ROOT / "pybullet_cup_benchmark.csv"
    rows = _read_rows(path)
    seed_count = len({row["seed"] for row in rows})
    max_background = max(int(float(row["background_objects"])) for row in rows)
    source_path = path
    coverage_note = ""
    coverage_path = ROOT / "pybullet_simulator_coverage.csv"
    if coverage_path.exists():
        coverage_rows = _read_rows(coverage_path)
        source_path = coverage_path
        axis_count = len({row["axis"] for row in coverage_rows})
        coverage_max_background = max(int(float(row["background_max"])) for row in coverage_rows)
        coverage_max_total_n = max(int(float(row["total_n_max"])) for row in coverage_rows)
        coverage_max_horizon = max(int(float(row["horizon_max"])) for row in coverage_rows)
        coverage_max_corruptions = max(int(float(row["corruption_count"])) for row in coverage_rows)
        complete_rows = [
            row
            for row in coverage_rows
            if str(row.get("baseline_complete", "")).lower() == "true"
            and int(float(row.get("model_count", 0))) > 0
        ]
        complete_max_background = max(int(float(row["background_max"])) for row in complete_rows)
        complete_max_total_n = max(int(float(row["total_n_max"])) for row in complete_rows)
        screen_row = next((row for row in coverage_rows if row["axis"] == "cup_n256_baseline_screen"), None)
        screen_note = (
            " The N_bg=256 cup screen is baseline-complete at total N=261, but it is a low-training feasibility screen rather than a strong accuracy-superiority result."
            if screen_row is not None
            else ""
        )
        medium_note = ""
        medium_path = ROOT / "pybullet_cup_benchmark_n256_medium.csv"
        if medium_path.exists():
            medium_rows = _read_rows(medium_path)
            medium_groups: dict[str, list[dict[str, str]]] = {}
            for row in medium_rows:
                medium_groups.setdefault(row["model"], []).append(row)
            medium_summary = [
                {
                    "model": model,
                    "accuracy": statistics.fmean(float(row["branch_accuracy"]) for row in group),
                    "latency": statistics.fmean(float(row["ms_per_sample_forward"]) for row in group),
                }
                for model, group in medium_groups.items()
            ]
            wpu_medium = [row for row in medium_summary if row["model"].startswith("wpu-")]
            baseline_medium = [row for row in medium_summary if not row["model"].startswith("wpu-")]
            if wpu_medium and baseline_medium:
                best_wpu = max(wpu_medium, key=lambda row: row["accuracy"])
                best_baseline = max(baseline_medium, key=lambda row: row["accuracy"])
                speedup = best_baseline["latency"] / max(best_wpu["latency"], 1e-9)
                medium_note = (
                    f" A medium-training N_bg=256 baseline-complete run raises evidence quality: "
                    f"best WPU `{best_wpu['model']}` reaches accuracy {best_wpu['accuracy']:.6f} "
                    f"versus best baseline `{best_baseline['model']}` at {best_baseline['accuracy']:.6f}, "
                    f"with WPU {speedup:.6f}x faster than that best-accuracy baseline."
                )
        mechanism_axis = next(
            (row for row in coverage_rows if row["axis"] == "mechanism_shift_generalization"),
            None,
        )
        mechanism_count = int(float(mechanism_axis["mechanism_count"])) if mechanism_axis else 1
        incomplete_axes = [
            row["axis"]
            for row in coverage_rows
            if str(row.get("baseline_complete", "")).lower() != "true"
        ]
        incomplete_note = (
            f" Baseline-incomplete large-state axes are explicitly flagged: {', '.join(incomplete_axes)}."
            if incomplete_axes
            else ""
        )
        coverage_note = (
            f" Coverage audit spans {axis_count} PyBullet axes with max N_bg={coverage_max_background}, "
            f"max total N={coverage_max_total_n}, max horizon={coverage_max_horizon}, "
            f"{mechanism_count} mechanisms, and {coverage_max_corruptions} objectification corruptions. "
            f"Baseline-complete coverage reaches N_bg={complete_max_background}, total N={complete_max_total_n}."
            f"{screen_note}"
            f"{medium_note}"
            f"{incomplete_note}"
        )
    status = "partial" if seed_count >= 2 and max_background >= 128 else "fail"
    return _row(
        3,
        "Simulator-backed benchmark",
        status,
        float(seed_count),
        5.0,
        "seed_count",
        source_path,
        f"PyBullet benchmark exists with {seed_count} seeds and background up to N_bg={max_background}; the 7-seed extension is still small but less seed-fragile.{coverage_note}",
        "Increase seeds, mechanisms, training scale, and long-horizon simulator rollouts.",
    )


def _priority_shift_generalization() -> dict[str, object]:
    path = ROOT / "pybullet_shift_generalization.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    shifts = sorted({row["eval_mechanism"] for row in rows if row["eval_mechanism"] != "nominal"})
    wins = 0
    notes: list[str] = []
    for mechanism in shifts:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        best_wpu = max(float(row["branch_accuracy"]) for row in group if row["model"].startswith("wpu-"))
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        if best_wpu >= best_baseline:
            wins += 1
        notes.append(f"{mechanism}: WPU {best_wpu:.6f} vs baseline {best_baseline:.6f}")
    win_rate = wins / max(1, len(shifts))
    observed_win_rate = win_rate
    used_adapted_protocol = False
    source_path = path
    mixture_path = ROOT / "pybullet_shift_generalization_mixture_calibrated.csv"
    if mixture_path.exists():
        mixture_rows = _rows_of_type(_read_rows(mixture_path), "summary")
        mixture_notes: list[str] = []
        for mechanism in shifts:
            group = [row for row in mixture_rows if row["eval_mechanism"] == mechanism]
            if not group:
                continue
            best_wpu = max(float(row["branch_accuracy"]) for row in group if row["model"].startswith("wpu-"))
            best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
            mixture_notes.append(f"mixture {mechanism}: WPU {best_wpu:.6f} vs baseline {best_baseline:.6f}")
        if mixture_notes:
            notes.append("3-seed calibrated mixture probe: " + "; ".join(mixture_notes))
    leave_path = ROOT / "pybullet_shift_leave_family_out_summary.csv"
    if leave_path.exists():
        leave_rows = _read_rows(leave_path)
        leave_win_rate = sum(1 for row in leave_rows if row["wpu_win"] == "True") / max(len(leave_rows), 1)
        leave_notes = [
            f"leave-family {row['eval_mechanism']}: WPU {float(row['best_wpu_accuracy']):.6f} vs baseline {float(row['best_baseline_accuracy']):.6f}"
            for row in leave_rows
        ]
        notes.append(f"3-seed leave-family-out win-rate {leave_win_rate:.6f}: " + "; ".join(leave_notes))
    stress_path = ROOT / "pybullet_shift_composition_stress_7seed_summary.csv"
    stress_seed_label = "7-seed"
    if not stress_path.exists():
        stress_path = ROOT / "pybullet_shift_composition_stress_summary.csv"
        stress_seed_label = "3-seed"
    if stress_path.exists():
        stress_rows = _read_rows(stress_path)
        stress_win_rate = sum(1 for row in stress_rows if row["wpu_win"] == "True") / max(len(stress_rows), 1)
        stress_delta = statistics.fmean(float(row["accuracy_delta"]) for row in stress_rows)
        stress_notes = [
            f"composition {row['eval_mechanism']}: WPU {float(row['best_wpu_accuracy']):.6f} vs baseline {float(row['best_baseline_accuracy']):.6f}"
            for row in stress_rows
        ]
        notes.append(
            f"{stress_seed_label} composition-shift stress win-rate {stress_win_rate:.6f}, "
            f"mean accuracy delta {stress_delta:.6f}: " + "; ".join(stress_notes)
        )
    prior_path = ROOT / "pybullet_branch_prior_shift.csv"
    if prior_path.exists():
        prior_rows = _read_rows(prior_path)
        shifted_prior_rows = [
            row
            for row in prior_rows
            if row["eval_mechanism"] != "nominal" and row["branch_prior_dominates"] == "True"
        ]
        prior_notes = [
            f"{row['eval_mechanism']}: majority {float(row['majority_accuracy']):.6f} "
            f"vs best WPU {float(row['best_wpu_accuracy']):.6f}"
            for row in shifted_prior_rows
        ]
        if prior_notes:
            notes.append("branch-prior audit flags prior-dominated shifts: " + "; ".join(prior_notes))
    adaptation_path = ROOT / "pybullet_mechanism_prior_adaptation_summary.csv"
    if adaptation_path.exists():
        adaptation_rows = [row for row in _read_rows(adaptation_path) if row["eval_mechanism"] != "nominal"]
        adapted_win_rate = statistics.fmean(
            1.0 if float(row["adapted_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in adaptation_rows
        )
        base_adaptation_win_rate = statistics.fmean(
            1.0 if float(row["base_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in adaptation_rows
        )
        mean_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in adaptation_rows)
        mean_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in adaptation_rows)
        prior_before = sum(1 for row in adaptation_rows if row["base_prior_dominated"] == "True")
        prior_after = sum(1 for row in adaptation_rows if row["adapted_prior_dominated"] == "True")
        observed_win_rate = max(observed_win_rate, adapted_win_rate)
        used_adapted_protocol = True
        source_path = adaptation_path if adapted_win_rate >= win_rate else path
        notes.append(
            f"7-seed mechanism-prior adaptation changes shifted WPU win-rate "
            f"{base_adaptation_win_rate:.6f}->{adapted_win_rate:.6f}, mean WPU accuracy "
            f"change {mean_accuracy_change:.6f}, mean WPU ECE change {mean_ece_change:.6f}, "
            f"prior-dominated shifts {prior_before}->{prior_after}"
        )
    strength_path = ROOT / "pybullet_prior_strength_sweep_summary.csv"
    if strength_path.exists():
        strength_rows = [row for row in _read_rows(strength_path) if row["row_type"] == "aggregate"]
        if strength_rows:
            best_strength = max(
                strength_rows,
                key=lambda row: (
                    float(row["shifted_wpu_win_rate"]),
                    float(row["wpu_accuracy"]),
                    -float(row["wpu_ece"]),
                ),
            )
            observed_win_rate = max(observed_win_rate, float(best_strength["shifted_wpu_win_rate"]))
            used_adapted_protocol = used_adapted_protocol or float(best_strength["mechanism_prior_strength"]) > 0.0
            if float(best_strength["shifted_wpu_win_rate"]) >= win_rate:
                source_path = strength_path
            notes.append(
                f"prior-strength sweep best strength {float(best_strength['mechanism_prior_strength']):.2f} "
                f"reaches shifted WPU win-rate {float(best_strength['shifted_wpu_win_rate']):.6f}, "
                f"mean WPU accuracy {float(best_strength['wpu_accuracy']):.6f}, and mean WPU ECE "
                f"{float(best_strength['wpu_ece']):.6f}"
            )
    selected_path = ROOT / "pybullet_selected_prior_adaptation_summary.csv"
    if selected_path.exists():
        selected_rows = [row for row in _read_rows(selected_path) if row["eval_mechanism"] != "nominal"]
        selected_win_rate = statistics.fmean(
            1.0 if float(row["selected_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in selected_rows
        )
        selected_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in selected_rows)
        selected_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in selected_rows)
        selected_prior_before = sum(1 for row in selected_rows if row["base_prior_dominated"] == "True")
        selected_prior_after = sum(1 for row in selected_rows if row["selected_prior_dominated"] == "True")
        observed_win_rate = max(observed_win_rate, selected_win_rate)
        used_adapted_protocol = True
        notes.append(
            f"calibration-selected prior keeps shifted WPU win-rate {selected_win_rate:.6f}, "
            f"mean WPU accuracy change {selected_accuracy_change:.6f}, mean WPU ECE change "
            f"{selected_ece_change:.6f}, and prior-dominated shifts "
            f"{selected_prior_before}->{selected_prior_after}"
        )
    fewshot_path = ROOT / "pybullet_fewshot_mechanism_adaptation_summary.csv"
    if fewshot_path.exists():
        fewshot_rows = [row for row in _read_rows(fewshot_path) if row["eval_mechanism"] != "nominal"]
        fewshot_win_rate = statistics.fmean(
            1.0 if float(row["adapted_wpu_minus_baseline"]) >= 0.0 else 0.0 for row in fewshot_rows
        )
        fewshot_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in fewshot_rows)
        fewshot_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in fewshot_rows)
        fewshot_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in fewshot_rows)
        observed_win_rate = max(observed_win_rate, fewshot_win_rate)
        used_adapted_protocol = True
        notes.append(
            f"few-shot mechanism adaptation reaches shifted WPU win-rate {fewshot_win_rate:.6f}, "
            f"mean WPU accuracy change {fewshot_accuracy_change:.6f}, mean WPU-baseline margin "
            f"change {fewshot_margin_change:.6f}, and mean WPU ECE change {fewshot_ece_change:.6f}; "
            "this is an adapted protocol, not zero-shot generalization"
        )
    adaptive_path = ROOT / "pybullet_mechanism_adaptive_policy_summary.csv"
    if adaptive_path.exists():
        adaptive_rows = [row for row in _read_rows(adaptive_path) if row["eval_mechanism"] != "nominal"]
        adaptive_win_rate = statistics.fmean(
            1.0 if row["policy_wpu_win"] == "True" else 0.0 for row in adaptive_rows
        )
        adaptive_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in adaptive_rows)
        adaptive_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in adaptive_rows)
        adaptive_ece_change = statistics.fmean(float(row["policy_wpu_ece_change"]) for row in adaptive_rows)
        adaptive_brier_change = statistics.fmean(float(row["policy_wpu_brier_change"]) for row in adaptive_rows)
        observed_win_rate = max(observed_win_rate, adaptive_win_rate)
        used_adapted_protocol = True
        source_path = adaptive_path if adaptive_win_rate >= observed_win_rate else source_path
        notes.append(
            f"mechanism-aware adaptive policy reaches shifted WPU win-rate {adaptive_win_rate:.6f}, "
            f"mean WPU accuracy change {adaptive_accuracy_change:.6f}, mean WPU-baseline margin "
            f"change {adaptive_margin_change:.6f}, mean WPU ECE change {adaptive_ece_change:.6f}, "
            f"and mean WPU Brier change {adaptive_brier_change:.6f}; this is a detect-and-adapt protocol, "
            "not zero-shot generalization"
        )
    detector_path = ROOT / "pybullet_shift_detector_policy.csv"
    if detector_path.exists():
        detector_rows = _read_rows(detector_path)
        safe_summary_rows = [
            row
            for row in detector_rows
            if row["row_type"] == "summary" and int(row["nominal_false_adaptation"]) == 0
        ]
        if safe_summary_rows:
            best_detector = max(safe_summary_rows, key=lambda row: float(row["detector_score"]))
            observed_win_rate = max(observed_win_rate, float(best_detector["shifted_wpu_win_rate"]))
            used_adapted_protocol = True
            if float(best_detector["shifted_wpu_win_rate"]) >= observed_win_rate:
                source_path = detector_path
            notes.append(
                f"calibration-statistic shift detector recovers the adapted policy without mechanism-name routing: "
                f"best-safe detector score {float(best_detector['detector_score']):.6f}, shifted WPU win-rate "
                f"{float(best_detector['shifted_wpu_win_rate']):.6f}, mean WPU accuracy change "
                f"{float(best_detector['mean_shifted_accuracy_change']):.6f}, mean WPU ECE change "
                f"{float(best_detector['mean_shifted_ece_change']):.6f}, nominal false adaptation "
                f"{int(best_detector['nominal_false_adaptation'])}, decisions {best_detector['decisions']}; "
                "this still uses calibration labels and adaptation samples"
            )
    status = "partial" if observed_win_rate > 0.0 else "fail"
    if observed_win_rate == 1.0 and not used_adapted_protocol and win_rate == 1.0:
        status = "pass"
    return _row(
        4,
        "Mechanism-family shift generalization",
        status,
        observed_win_rate,
        1.0,
        "wpu_shift_win_rate",
        source_path,
        "; ".join(notes),
        "Train an explicit mechanism-shift detector and evaluate selective adaptation on harder held-out mechanisms.",
    )


def _priority_calibration() -> dict[str, object]:
    path = ROOT / "pybullet_shift_generalization.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    wpu_ece = statistics.fmean(float(row["ece"]) for row in rows if row["model"].startswith("wpu-"))
    baseline_ece = statistics.fmean(float(row["ece"]) for row in rows if not row["model"].startswith("wpu-"))
    ratio = wpu_ece / baseline_ece if baseline_ece > 0 else float("inf")
    mixture_note = ""
    mixture_path = ROOT / "pybullet_shift_generalization_mixture_calibrated.csv"
    if mixture_path.exists():
        mixture_rows = _rows_of_type(_read_rows(mixture_path), "summary")
        mixture_wpu_ece = statistics.fmean(float(row["ece"]) for row in mixture_rows if row["model"].startswith("wpu-"))
        mixture_baseline_ece = statistics.fmean(float(row["ece"]) for row in mixture_rows if not row["model"].startswith("wpu-"))
        mixture_ratio = mixture_wpu_ece / mixture_baseline_ece if mixture_baseline_ece > 0 else float("inf")
        mixture_note = (
            f" A 3-seed calibrated mixture probe gives WPU ECE {mixture_wpu_ece:.6f}, "
            f"baseline ECE {mixture_baseline_ece:.6f}, ratio {mixture_ratio:.6f}."
        )
    leave_path = ROOT / "pybullet_shift_leave_family_out_summary.csv"
    if leave_path.exists():
        leave_rows = _read_rows(leave_path)
        leave_ratio = statistics.fmean(float(row["ece_ratio"]) for row in leave_rows)
        mixture_note += f" A 3-seed leave-family-out probe gives mean ECE ratio {leave_ratio:.6f}."
    stress_path = ROOT / "pybullet_shift_composition_stress_7seed_summary.csv"
    stress_seed_label = "7-seed"
    if not stress_path.exists():
        stress_path = ROOT / "pybullet_shift_composition_stress_summary.csv"
        stress_seed_label = "3-seed"
    if stress_path.exists():
        stress_rows = _read_rows(stress_path)
        stress_ratio = statistics.fmean(float(row["ece_ratio"]) for row in stress_rows)
        worst = max(stress_rows, key=lambda row: float(row["ece_ratio"]))
        mixture_note += (
            f" A {stress_seed_label} composition-shift stress probe gives mean ECE ratio {stress_ratio:.6f}; "
            f"worst is {worst['eval_mechanism']} at {float(worst['ece_ratio']):.6f}."
        )
    calibration_compare_path = ROOT / "pybullet_shift_calibration_comparison.csv"
    if calibration_compare_path.exists():
        comparison_rows = _read_rows(calibration_compare_path)
        mean_ece_change = statistics.fmean(float(row["ece_ratio_change"]) for row in comparison_rows)
        improved = sum(1 for row in comparison_rows if float(row["ece_ratio_change"]) < 0.0)
        mixture_note += (
            f" Temperature+bias calibration changes mean ECE ratio by {mean_ece_change:.6f} "
            f"and improves {improved}/{len(comparison_rows)} composition mechanisms."
            )
    prior_path = ROOT / "pybullet_branch_prior_shift.csv"
    if prior_path.exists():
        prior_rows = _read_rows(prior_path)
        shifted = [row for row in prior_rows if row["eval_mechanism"] != "nominal"]
        prior_dominated = [row for row in shifted if row["branch_prior_dominates"] == "True"]
        if prior_dominated:
            mean_prior_gap = statistics.fmean(float(row["majority_minus_best_wpu"]) for row in shifted)
            worst = max(prior_dominated, key=lambda row: float(row["majority_minus_best_wpu"]))
            mixture_note += (
                f" Branch-prior audit finds {len(prior_dominated)}/{len(shifted)} shifted mechanisms "
                f"are prior-dominated; worst is {worst['eval_mechanism']} where majority accuracy "
                f"{float(worst['majority_accuracy']):.6f} exceeds best WPU "
                f"{float(worst['best_wpu_accuracy']):.6f}. Mean shifted majority-WPU gap is "
                f"{mean_prior_gap:.6f}."
            )
    adaptation_path = ROOT / "pybullet_mechanism_prior_adaptation_summary.csv"
    if adaptation_path.exists():
        adaptation_rows = [row for row in _read_rows(adaptation_path) if row["eval_mechanism"] != "nominal"]
        mean_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in adaptation_rows)
        mean_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in adaptation_rows)
        mixture_note += (
            f" Mechanism-prior adaptation improves shifted win-rate but changes mean WPU ECE by "
            f"{mean_ece_change:.6f} and mean Brier by {mean_brier_change:.6f}; "
            "simple prior bias is therefore not calibration-safe."
        )
    strength_path = ROOT / "pybullet_prior_strength_sweep_summary.csv"
    if strength_path.exists():
        strength_rows = [row for row in _read_rows(strength_path) if row["row_type"] == "aggregate"]
        if strength_rows:
            zero = min(strength_rows, key=lambda row: abs(float(row["mechanism_prior_strength"])))
            safe_rows = [
                row
                for row in strength_rows
                if float(row["mechanism_prior_strength"]) > 0.0
                and float(row["shifted_wpu_win_rate"]) >= float(zero["shifted_wpu_win_rate"])
                and float(row["wpu_ece"]) <= float(zero["wpu_ece"])
            ]
            best_strength = max(
                strength_rows,
                key=lambda row: (
                    float(row["shifted_wpu_win_rate"]),
                    float(row["wpu_accuracy"]),
                    -float(row["wpu_ece"]),
                ),
            )
            safe_note = (
                f"calibration-safe nonzero strength exists at {float(safe_rows[0]['mechanism_prior_strength']):.2f}"
                if safe_rows
                else "no nonzero strength preserves/improves win-rate without increasing ECE"
            )
            mixture_note += (
                f" Prior-strength sweep finds best accuracy strength "
                f"{float(best_strength['mechanism_prior_strength']):.2f} with shifted WPU win-rate "
                f"{float(best_strength['shifted_wpu_win_rate']):.6f}, but {safe_note}; "
                f"zero-strength WPU ECE is {float(zero['wpu_ece']):.6f} versus best-strength WPU ECE "
                f"{float(best_strength['wpu_ece']):.6f}."
            )
    selected_path = ROOT / "pybullet_selected_prior_adaptation_summary.csv"
    if selected_path.exists():
        selected_rows = [row for row in _read_rows(selected_path) if row["eval_mechanism"] != "nominal"]
        selected_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in selected_rows)
        selected_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in selected_rows)
        selected_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in selected_rows)
        mixture_note += (
            f" Calibration-selected prior improves shifted mean WPU accuracy by "
            f"{selected_accuracy_change:.6f}, mean ECE by {selected_ece_change:.6f}, and mean Brier "
            f"by {selected_brier_change:.6f}, but it does not improve shifted WPU-vs-baseline win-rate."
        )
    fewshot_path = ROOT / "pybullet_fewshot_mechanism_adaptation_summary.csv"
    if fewshot_path.exists():
        fewshot_rows = [row for row in _read_rows(fewshot_path) if row["eval_mechanism"] != "nominal"]
        fewshot_ece_change = statistics.fmean(float(row["wpu_ece_change"]) for row in fewshot_rows)
        fewshot_brier_change = statistics.fmean(float(row["wpu_brier_change"]) for row in fewshot_rows)
        mixture_note += (
            f" Few-shot mechanism adaptation also improves shifted mean WPU ECE by "
            f"{fewshot_ece_change:.6f} and Brier by {fewshot_brier_change:.6f}, but it uses "
            "mechanism-specific parameter adaptation."
        )
    adaptive_path = ROOT / "pybullet_mechanism_adaptive_policy_summary.csv"
    if adaptive_path.exists():
        adaptive_rows = [row for row in _read_rows(adaptive_path) if row["eval_mechanism"] != "nominal"]
        adaptive_ece_change = statistics.fmean(float(row["policy_wpu_ece_change"]) for row in adaptive_rows)
        adaptive_brier_change = statistics.fmean(float(row["policy_wpu_brier_change"]) for row in adaptive_rows)
        adaptive_accuracy_change = statistics.fmean(float(row["wpu_accuracy_change"]) for row in adaptive_rows)
        adaptive_margin_change = statistics.fmean(float(row["wpu_margin_change"]) for row in adaptive_rows)
        mixture_note += (
            f" A mechanism-aware adaptive policy improves shifted WPU accuracy by "
            f"{adaptive_accuracy_change:.6f}, margin by {adaptive_margin_change:.6f}, ECE by "
            f"{adaptive_ece_change:.6f}, and Brier by {adaptive_brier_change:.6f}; this is positive "
            "for detect-and-adapt calibration, not zero-shot calibration."
        )
    detector_path = ROOT / "pybullet_shift_detector_policy.csv"
    if detector_path.exists():
        detector_rows = _read_rows(detector_path)
        safe_summary_rows = [
            row
            for row in detector_rows
            if row["row_type"] == "summary" and int(row["nominal_false_adaptation"]) == 0
        ]
        if safe_summary_rows:
            best_detector = max(safe_summary_rows, key=lambda row: float(row["detector_score"]))
            mixture_note += (
                f" A calibration-statistic shift detector keeps nominal false adaptation at "
                f"{int(best_detector['nominal_false_adaptation'])} while changing shifted WPU accuracy by "
                f"{float(best_detector['mean_shifted_accuracy_change']):.6f}, margin by "
                f"{float(best_detector['mean_shifted_margin_change']):.6f}, ECE by "
                f"{float(best_detector['mean_shifted_ece_change']):.6f}, and Brier by "
                f"{float(best_detector['mean_shifted_brier_change']):.6f}; it reduces mechanism-name oracle usage "
                "but still depends on calibration labels and adaptation samples."
            )
    gated_path = ROOT / "pybullet_uncertainty_gated_recompute.csv"
    if gated_path.exists():
        gated_rows = [
            row
            for row in _read_rows(gated_path)
            if row["row_type"] == "summary" and row["eval_mechanism"] == "aggregate"
        ]
        sparse = next(row for row in gated_rows if row["policy"] == "wpu_sparse")
        gated = [
            row
            for row in gated_rows
            if row["policy"].startswith("wpu_gated")
            and float(row["branch_accuracy"]) >= float(sparse["branch_accuracy"])
            and float(row["ece"]) <= float(sparse["ece"])
        ]
        best_gated = min(
            gated,
            key=lambda row: (
                float(row["ece"]),
                -float(row["branch_accuracy"]),
                float(row["dense_recompute_rate"]),
            ),
        )
        low_cost = [
            row
            for row in gated_rows
            if row["policy"].startswith("wpu_gated") and float(row["dense_recompute_rate"]) <= 0.25
        ]
        best_low_cost = min(
            low_cost,
            key=lambda row: (
                float(row["ece"]),
                -float(row["branch_accuracy"]),
                float(row["dense_recompute_rate"]),
            ),
        )
        mixture_note += (
            f" Uncertainty-gated local-dense recompute improves aggregate WPU accuracy by "
            f"{float(best_gated['branch_accuracy']) - float(sparse['branch_accuracy']):.6f} and ECE by "
            f"{float(best_gated['ece']) - float(sparse['ece']):.6f}, but it uses dense recompute rate "
            f"{float(best_gated['dense_recompute_rate']):.6f}. The low-cost gate uses rate "
            f"{float(best_low_cost['dense_recompute_rate']):.6f}, changes accuracy by "
            f"{float(best_low_cost['branch_accuracy']) - float(sparse['branch_accuracy']):.6f}, and changes ECE by "
            f"{float(best_low_cost['ece']) - float(sparse['ece']):.6f}; selective low-cost uncertainty routing is therefore not solved."
        )
    learned_gate_path = ROOT / "pybullet_learned_uncertainty_gate.csv"
    if learned_gate_path.exists():
        learned_rows = [
            row
            for row in _read_rows(learned_gate_path)
            if row["row_type"] == "summary" and row["eval_mechanism"] == "aggregate"
        ]
        sparse = next(row for row in learned_rows if row["policy"] == "wpu_sparse")
        source_candidates = [row for row in learned_rows if row["gate_kind"] == "source"]
        fewshot_candidates = [row for row in learned_rows if row["gate_kind"] == "fewshot"]
        source_low = min(
            [row for row in source_candidates if float(row["dense_recompute_rate"]) <= 0.25],
            key=lambda row: (
                float(row["ece"]),
                -float(row["branch_accuracy"]),
                float(row["dense_recompute_rate"]),
            ),
        )
        fewshot_low_pool = [row for row in fewshot_candidates if float(row["dense_recompute_rate"]) <= 0.25]
        fewshot_low = (
            min(
                fewshot_low_pool,
                key=lambda row: (
                    float(row["ece"]),
                    -float(row["branch_accuracy"]),
                    float(row["dense_recompute_rate"]),
                ),
            )
            if fewshot_low_pool
            else min(fewshot_candidates, key=lambda row: float(row["dense_recompute_rate"]))
        )
        fewshot_low_note = (
            "within the 0.25 recompute budget"
            if fewshot_low_pool
            else "above the 0.25 recompute budget"
        )
        mixture_note += (
            f" A learned sparse-output benefit gate improves source low-cost accuracy by "
            f"{float(source_low['branch_accuracy']) - float(sparse['branch_accuracy']):.6f} at recompute rate "
            f"{float(source_low['dense_recompute_rate']):.6f}, but changes ECE by "
            f"{float(source_low['ece']) - float(sparse['ece']):.6f}. Few-shot gating changes accuracy by "
            f"{float(fewshot_low['branch_accuracy']) - float(sparse['branch_accuracy']):.6f} and ECE by "
            f"{float(fewshot_low['ece']) - float(sparse['ece']):.6f} at rate "
            f"{float(fewshot_low['dense_recompute_rate']):.6f} ({fewshot_low_note}). Thus learned sparse-output routing improves accuracy, not calibration-safe low-cost routing."
        )
    frontier_path = ROOT / "pybullet_calibration_cost_frontier.csv"
    if frontier_path.exists():
        frontier_rows = _read_rows(frontier_path)
        policy_rows = [row for row in frontier_rows if row["protocol"] != "sparse_reference"]
        low_cost_safe = [
            row
            for row in policy_rows
            if row["low_cost"] == "True" and row["calibration_safe"] == "True"
        ]
        low_cost_rows = [row for row in policy_rows if row["low_cost"] == "True"]
        safe_rows = [row for row in policy_rows if row["calibration_safe"] == "True"]
        best_low_cost = max(low_cost_rows, key=lambda row: float(row["accuracy_delta"]))
        cheapest_safe = min(safe_rows, key=lambda row: float(row["cost_proxy"])) if safe_rows else None
        frontier_note = (
            f" Calibration-cost frontier audit finds {len(low_cost_safe)} non-reference "
            f"calibration-safe policies under cost_proxy<=0.25. The best low-cost accuracy policy is "
            f"{best_low_cost['policy']} with accuracy delta {float(best_low_cost['accuracy_delta']):.6f}, "
            f"ECE delta {float(best_low_cost['ece_delta']):.6f}, and cost {float(best_low_cost['cost_proxy']):.6f}."
        )
        if cheapest_safe is not None:
            frontier_note += (
                f" The cheapest non-reference calibration-safe policy is {cheapest_safe['policy']} "
                f"at cost {float(cheapest_safe['cost_proxy']):.6f}."
            )
        mixture_note += frontier_note
    status = "partial" if ratio <= 1.1 else "fail"
    return _row(
        5,
        "Calibration and uncertainty",
        status,
        ratio,
        1.0,
        "wpu_ece_over_baseline_ece",
        path,
        f"Mean WPU ECE is {wpu_ece:.6f}; mean baseline ECE is {baseline_ece:.6f}; ratio is {ratio:.6f}.{mixture_note}",
        "Move beyond sparse-output gates: train calibration-aware mechanism uncertainty, branch calibration loss, and multi-step ECE/Brier/NLL.",
    )


def _priority_systems_profile() -> dict[str, object]:
    path = ROOT / "pybullet_system_profile.csv"
    cuda_path = ROOT / "pybullet_system_profile_cuda.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    max_reduction = max(float(row["tensor_byte_reduction"]) for row in rows)
    max_latency_reduction = max(float(row.get("tensorize_latency_reduction", 0.0)) for row in rows)
    max_forward_reduction = max(float(row.get("sparse_forward_latency_reduction", 0.0)) for row in rows)
    max_n = max(float(row["total_objects"]) for row in rows)
    status = "partial" if max_reduction >= 0.95 else "fail"
    cuda_note = ""
    if cuda_path.exists():
        cuda_rows = _rows_of_type(_read_rows(cuda_path), "summary")
        cuda_max_forward = max(float(row.get("sparse_forward_latency_reduction", 0.0)) for row in cuda_rows)
        cuda_max_memory = max(float(row.get("sparse_peak_memory_reduction", 0.0)) for row in cuda_rows)
        cuda_max_n = max(float(row["total_objects"]) for row in cuda_rows)
        cuda_note = (
            f" CUDA random-model sparse-forward latency reduction reaches {cuda_max_forward:.6f} "
            f"and sparse peak-memory reduction reaches {cuda_max_memory:.6f} at mean total objects {cuda_max_n:.1f}."
        )
    matched_path = ROOT / "pybullet_matched_speedup_audit.csv"
    if matched_path.exists():
        matched_rows = _read_rows(matched_path)
        matched_notes = []
        for row in matched_rows:
            matched_notes.append(
                f"N={row['total_objects_n']}: matched={row['matched_accuracy']} speedup={float(row['matched_speedup']):.6f}"
            )
        cuda_note += " Matched-or-better audit: " + "; ".join(matched_notes) + "."
    pareto_path = ROOT / "pybullet_pareto_frontier.csv"
    if pareto_path.exists():
        pareto_rows = _read_rows(pareto_path)
        wpu_frontier_ns = sorted(
            {
                int(float(row["total_objects_n"]))
                for row in pareto_rows
                if row.get("is_wpu") == "True" and row.get("pareto_frontier") == "True"
            }
        )
        wpu_dominated_ns = sorted(
            {
                int(float(row["total_objects_n"]))
                for row in pareto_rows
                if row.get("is_wpu") == "True" and row.get("pareto_frontier") != "True"
            }
        )
        cuda_note += (
            f" Pareto audit places WPU on the accuracy-latency frontier at N={wpu_frontier_ns} "
            f"and dominated at N={wpu_dominated_ns}."
        )
    energy_proxy_path = ROOT / "pybullet_system_energy_proxy.csv"
    if energy_proxy_path.exists():
        proxy_rows = _read_rows(energy_proxy_path)
        best_proxy = max(proxy_rows, key=lambda row: float(row["proxy_reduction"]))
        cuda_proxy_rows = [row for row in proxy_rows if row["profile"] == "cuda_forward_screening"]
        if cuda_proxy_rows:
            best_cuda_proxy = max(cuda_proxy_rows, key=lambda row: float(row["proxy_reduction"]))
            cuda_note += (
                f" Screening-only energy proxy max is {float(best_proxy['proxy_reduction']):.6f}; "
                f"CUDA forward proxy max is {float(best_cuda_proxy['proxy_reduction']):.6f}."
            )
    boundary_path = ROOT / "pybullet_system_claim_boundary.csv"
    if boundary_path.exists():
        boundary_rows = _read_rows(boundary_path)
        supported = [row for row in boundary_rows if row["status"].startswith("supported")]
        partial = [row for row in boundary_rows if row["status"].startswith("partial")]
        missing = [row for row in boundary_rows if row["status"] == "not_measured"]
        branch_row = next((row for row in boundary_rows if row["axis"] == "branch_overlay_memory"), None)
        peak_row = next((row for row in boundary_rows if row["axis"] == "random_cuda_peak_memory"), None)
        boundary_note = (
            f" Systems claim-boundary audit separates {len(supported)} supported proxy axes, "
            f"{len(partial)} partial trained axes, and {len(missing)} unmeasured hardware "
            f"{'axis' if len(missing) == 1 else 'axes'}."
        )
        if branch_row is not None:
            boundary_note += (
                f" Branch-overlay memory proxy reduction reaches {float(branch_row['observed']):.6f}."
            )
        if peak_row is not None:
            boundary_note += (
                f" CUDA peak-memory proxy remains weak at {float(peak_row['observed']):.6f}."
            )
        cuda_note += boundary_note
    return _row(
        6,
        "Systems profile and memory traffic",
        status,
        max_reduction,
        0.95,
        "max_tensor_byte_reduction",
        path,
        f"Tensor-byte reduction reaches {max_reduction:.6f} at mean total objects {max_n:.1f}; CPU tensorization latency reduction reaches {max_latency_reduction:.6f}; random-model CPU sparse-forward latency reduction reaches {max_forward_reduction:.6f}.{cuda_note} Real energy and sparse-kernel behavior remain unproven.",
        "Measure energy, allocator traffic, sparse-kernel behavior, Pareto frontiers, and trained matched-or-better speedups.",
    )


def _priority_objectification_quality() -> dict[str, object]:
    path = ROOT / "pybullet_objectification_quality.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    combined = [row for row in rows if row["corruption"] == "combined"]
    clean = [row for row in rows if row["corruption"] == "clean"]
    combined_score = statistics.fmean(float(row["objectification_score"]) for row in combined)
    clean_score = statistics.fmean(float(row["objectification_score"]) for row in clean)
    frontier = statistics.fmean(float(row["frontier_recall"]) for row in combined)
    coupling_note = ""
    coupling_path = ROOT / "pybullet_objectification_loss_coupling.csv"
    if coupling_path.exists():
        coupling_rows = _read_rows(coupling_path)
        summaries = [
            row
            for row in coupling_rows
            if row["row_type"] == "corruption_summary" and row["corruption"] != "clean"
        ]
        correlations = [row for row in coupling_rows if row["row_type"] == "predictor_correlation"]
        worst_acc = max(summaries, key=lambda row: float(row["accuracy_drop"]))
        worst_mse = max(summaries, key=lambda row: float(row["mse_increase"]))
        best_mse_predictor = max(
            correlations,
            key=lambda row: float(row["abs_pearson_mse_increase"] or 0.0),
        )
        best_acc_predictor = max(
            correlations,
            key=lambda row: float(row["abs_pearson_accuracy_drop"] or 0.0),
        )
        coupling_note = (
            f" Loss-coupling audit finds worst mean accuracy drop "
            f"{float(worst_acc['accuracy_drop']):.6f} for {worst_acc['model']}/{worst_acc['corruption']}, "
            f"worst mean MSE increase {float(worst_mse['mse_increase']):.6f} for "
            f"{worst_mse['model']}/{worst_mse['corruption']}. The strongest MSE predictor is "
            f"{best_mse_predictor['predictor']} with |r|={float(best_mse_predictor['abs_pearson_mse_increase']):.6f}; "
            f"the strongest accuracy predictor is {best_acc_predictor['predictor']} with "
            f"|r|={float(best_acc_predictor['abs_pearson_accuracy_drop']):.6f}."
        )
    status = "partial" if combined_score < clean_score and frontier < 1.0 else "fail"
    return _row(
        7,
        "Objectification quality to propagation loss",
        status,
        combined_score,
        clean_score,
        "combined_objectification_score",
        path,
        f"Clean score {clean_score:.6f}, combined-corruption score {combined_score:.6f}, combined frontier recall {frontier:.6f}.{coupling_note}",
        "Run stronger closed-loop or multi-horizon corruption experiments and connect objectification components to downstream loss under larger branch/rollout effects.",
    )


def _row(
    priority: int,
    name: str,
    status: str,
    observed: float,
    target: float,
    metric: str,
    source: Path,
    interpretation: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "priority": priority,
        "name": name,
        "status": status,
        "observed": round(observed, 6),
        "target": round(target, 6),
        "metric": metric,
        "source": source.as_posix(),
        "interpretation": interpretation,
        "next_action": next_action,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_of_type(rows: list[dict[str, str]], row_type: str) -> list[dict[str, str]]:
    if not rows or "row_type" not in rows[0]:
        return rows
    typed = [row for row in rows if row.get("row_type") == row_type]
    return typed or rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], *, korean: bool) -> str:
    if korean:
        title = "# WPU v2 우선순위 대시보드"
        intro = (
            "이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 "
            "보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 "
            "않도록 만드는 것이다."
        )
        headers = "| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |"
        sep = "|---:|---|---|---:|---:|---|"
        interp = "## 해석"
        next_title = "## 다음 조치"
        boundary = (
            "현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 "
            "아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, "
            "objectified state에서 작은 causal working set K를 tensorization 전에 "
            "식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 "
            "조건부 주장이다."
        )
    else:
        title = "# WPU v2 Priority Dashboard"
        intro = (
            "This dashboard conservatively recomputes the current status of v2 "
            "priorities 1-7 from existing experiment CSVs. Its purpose is to keep "
            "WPU claims aligned with evidence."
        )
        headers = "| Priority | Item | Status | Observed | Target | Metric |"
        sep = "|---:|---|---|---:|---:|---|"
        interp = "## Interpretation"
        next_title = "## Next Actions"
        boundary = (
            "The dashboard shows that WPU v2 is promising but not a completed "
            "superiority claim. The strongest claim remains conditional: WPU can "
            "reduce compute and memory when objectified state exposes a small "
            "causal working set K before tensorization. Large N alone is not enough."
        )

    lines = [title, "", intro, "", headers, sep]
    for row in rows:
        name = _ko_name(int(row["priority"])) if korean else str(row["name"])
        status = _ko_status(str(row["status"])) if korean else str(row["status"])
        lines.append(
            f"| {row['priority']} | {name} | {status} | "
            f"{float(row['observed']):.6f} | {float(row['target']):.6f} | `{row['metric']}` |"
        )
    lines.extend(["", interp, "", boundary, ""])
    for row in rows:
        if korean:
            lines.append(f"- P{row['priority']} {_ko_name(int(row['priority']))}: {_ko_interpretation(int(row['priority']))}")
        else:
            lines.append(f"- P{row['priority']} {row['name']}: {row['interpretation']}")
    lines.extend(["", next_title, ""])
    for row in rows:
        if korean:
            lines.append(f"- P{row['priority']}: {_ko_next_action(int(row['priority']))}")
        else:
            lines.append(f"- P{row['priority']}: {row['next_action']}")
    return "\n".join(lines) + "\n"


def _ko_name(priority: int) -> str:
    return {
        1: "Candidate-oracle gap",
        2: "장기 state integrity",
        3: "Simulator-backed benchmark",
        4: "Mechanism-family shift generalization",
        5: "Calibration과 uncertainty",
        6: "Systems profile과 memory traffic",
        7: "Objectification quality와 propagation loss",
    }[priority]


def _ko_status(status: str) -> str:
    return {
        "pass": "pass",
        "partial": "partial",
        "fail": "fail",
    }[status]


def _ko_interpretation(priority: int) -> str:
    return {
        1: "Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 안전하지만 closure가 0.081253으로 떨어지고, feature perturbation은 test-sweep safe closure를 0.329756까지 조금 올리지만 train-selected closure는 0.312586에 머문다. 별도 safety/utility head도 negative result다. Best closure는 0.147450, safe best는 0.090719, train-selected closure는 0.144863에 그친다. Cross-fit ensemble regret gate도 train-selected overfit 가설을 부정하는 negative result다. 최고 closure는 0.287268, safe best는 0.279738, cross-fit selected closure는 0.270989로 direct regret gate보다 낮다. Descriptor standardization과 group-DRO no-harm training도 standalone 해결책이 아니며 train-selected closure는 0.093863이다. 새 joint object-set candidate gate도 negative result다. Best closure는 0.101454, safe best는 0.101454, train-selected closure는 0.072167이고 mean regret correlation은 -0.000180에 가깝다. Regression-heavy ablation도 K=16에서 best closure 0.034751, train-selected closure -0.003089에 그친다. 따라서 P1 병목은 object-set feature 부재만이 아니라 cross-seed candidate regret target의 안정성 부족이다.",
        2: "Rollback-only memory layer는 sparse WPU H=25 integrity를 0.988647까지 올리지만 rollback rate가 0.812500으로 매우 높다. Corrected rollback은 rollback rate를 0.564167까지 낮추지만 integrity가 0.900288로 떨어진다. Escalated corrected rollback은 local-dense fallback을 사용해 integrity를 0.914831로 올리고 rollback rate를 0.000000으로 낮춘다. Finite-corrected run은 finite-safe delta clipping과 correction-only projection으로 integrity 0.958735, rollback rate 0.000000, escalation rate 0.000000을 달성하지만 correction rate가 0.784166으로 높다. 새 selective correction은 같은 integrity 0.958735를 유지하면서 corrected object fraction을 0.027461로 낮추고 low-disruption integrity를 0.758574까지 올린다. 그러나 correction trigger rate는 여전히 0.784166이다. Correction-trigger frontier audit은 integrity>=0.8 및 correction_rate<=0.25를 동시에 만족하는 trigger policy가 0개임을 보인다. 최고 low-correction trigger는 selective_corrected_entropy035이고 integrity 0.653668, correction rate 0.230000에 그친다. 따라서 P2는 memory-layer disruption은 줄였지만 raw delta stability와 low-frequency correction-trigger 학습은 아직 해결되지 않았다.",
        3: "PyBullet benchmark는 7개 seed와 background N_bg=128까지 본훈련 baseline-complete로 확장됐고, N_bg=256 screen은 total N=261에서 WPU/graph/token baseline을 모두 완료했다. 새 medium-training N_bg=256 baseline-complete run은 evidence quality를 올린다. Best WPU인 wpu-cws-indexed-local-dense는 accuracy 0.466667이고 best baseline인 graph-transformer는 0.450000이며, best WPU는 해당 best-accuracy baseline보다 forward latency 기준 60.629526x 빠르다. 단, margin이 작고 단일 cup family이므로 broad superiority claim은 아니다. Coverage audit는 9개 PyBullet 축을 추적하고, WPU-only large-state extension은 N_bg=512, total N=517까지 실행됐지만 graph-transformer baseline이 같은 protocol에서 완료되지 않았으므로 accuracy superiority evidence가 아니라 systems feasibility evidence로만 취급한다. Simulator-backed evidence는 강화됐지만 mechanism 다양성, baseline-complete large-N comparison, perception/state adapter가 아직 부족하다.",
        4: "7-seed nominal-shift benchmark는 mixed이고, 3-seed leave-family-out probe는 win-rate 0.750000을 보인다. 7-seed composition-shift stress에서는 WPU가 accuracy 기준 3/3에서 baseline 이상이며 평균 accuracy delta가 0.071428이다. Branch-prior audit은 catch_heavy가 prior-dominated shift임을 보인다. Mechanism-prior adaptation은 shifted WPU win-rate를 0.333333에서 0.666667로 올리고 prior-dominated shift를 1개에서 0개로 줄인다. Prior-strength sweep의 accuracy-best 설정은 strength=0.75, mean WPU accuracy 0.601852지만 shifted win-rate는 0.666667에 머문다. Calibration-selected prior는 mean accuracy/ECE를 개선하지만 shifted win-rate는 0.333333에 머문다. Few-shot mechanism adaptation은 shifted WPU win-rate 1.000000, mean margin change 0.050264까지 도달하지만 mechanism별 calibration set을 쓰는 adapted protocol이다. Mechanism-aware adaptive policy는 selected-prior와 few-shot adaptation을 선택적으로 결합해 shifted win-rate 1.000000, mean accuracy change 0.198412, margin change 0.058201, ECE change -0.099347, Brier change -0.155443에 도달한다. 새 calibration-statistic shift detector는 mechanism 이름 대신 base ECE와 majority-prior gap으로 같은 정책을 복원하며 nominal false adaptation 0, shifted win-rate 1.000000을 달성한다. 그러나 calibration label과 adaptation sample을 쓰므로 detect-and-adapt protocol이지 zero-shot generalization은 아니다. 따라서 P4는 adapted regime에서 강화됐지만 single-family zero-shot solved는 아니다.",
        5: "7-seed 평균 WPU ECE ratio는 0.963449이고, leave-family-out 평균 ECE ratio는 0.972745로 양호하지만, calibrated mixture probe에서는 1.133834로 악화된다. 7-seed composition-shift stress의 평균 ECE ratio는 1.014879이고 no_catch에서 1.166073까지 악화된다. 이는 3-seed stress보다 안정적이지만 여전히 calibration 우위는 아니다. Temperature+bias calibration은 no_catch를 개선하지만 3개 mechanism 중 1개만 ECE ratio가 개선되어 보편 해결책은 아니다. Branch-prior audit은 catch_heavy에서 majority prior 0.753968이 best WPU 0.408730을 크게 앞선다는 점을 보여준다. Mechanism-prior adaptation은 accuracy를 개선하지만 shifted mean ECE를 0.024819 악화시킨다. Prior-strength sweep에서도 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 strength가 없었다. Calibration-selected prior는 shifted mean ECE를 -0.046204, Brier를 -0.105470 개선하지만 baseline win-rate는 올리지 못한다. Few-shot mechanism adaptation도 ECE를 -0.055342 개선한다. Mechanism-aware adaptive policy와 calibration-statistic shift detector는 shifted accuracy를 +0.198412, margin을 +0.058201, ECE를 -0.099347, Brier를 -0.155443 개선해 detect-and-adapt calibration에는 긍정적이다. Uncertainty-gated local-dense recompute는 aggregate accuracy를 +0.071428, ECE를 -0.016396 개선하지만 dense recompute rate가 0.985450으로 거의 full recompute다. Static low-cost gate는 recompute rate 0.025132에서 accuracy를 +0.009260 올리지만 ECE를 +0.005395 악화시킨다. Learned sparse-output benefit gate는 source low-cost에서 accuracy를 +0.052910 올리지만 ECE를 +0.010769 악화시킨다. 새 mechanism-selective calibration gate는 cost_proxy<=0.25에서 non-reference calibration-safe policy 1개를 만든다. Best safe policy는 cost 0.247355, accuracy delta +0.029100, ECE delta -0.001652, Brier delta -0.030758이다. 따라서 P5는 전역/zero-shot gate로는 미해결이지만, mechanism-aware adapted routing에서는 약한 positive sub-regime이 확인됐다.",
        6: "Tensor-byte reduction은 0.997454, CPU sparse-forward reduction은 0.996975, CUDA sparse-forward reduction은 0.996216까지 관측됐다. Screening-only energy proxy도 추가됐지만 실제 전력 측정은 아니다. Matched-speedup audit의 판정 기준을 corrected matched-or-better로 고치면 N=133에서는 best-accuracy non-WPU baseline 대비 WPU가 더 정확하고 더 빠르다. Pareto audit에서도 WPU는 N=133에서 frontier에 올라가지만 N=5에서는 token에 지배된다. Systems claim-boundary audit은 supported proxy 축 4개, partial trained 축 2개, real-power/sparse-kernel 미측정 축 1개를 분리한다. Branch-overlay memory proxy reduction은 0.874128이지만 CUDA peak-memory proxy reduction은 0.304080에 그친다. Real energy와 sparse-kernel behavior는 아직 미해결이다.",
        7: "Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. 새 loss-coupling audit은 worst mean accuracy drop이 wpu-cws-indexed-local-dense/combined에서 0.027778, worst mean MSE increase가 wpu-cws-indexed-sparse/drop_relations_heavy에서 0.087356임을 보인다. MSE degradation과 가장 강하게 연결된 component deficit은 selected_k_mean(|r|=0.481851)이고 accuracy degradation과 가장 강하게 연결된 component deficit은 relation_confidence(|r|=0.352431)이다. 따라서 objectification metric과 downstream loss의 연결은 시작됐지만 branch accuracy 변화가 작아 closed-loop/multi-horizon 검증이 필요하다.",
    }[priority]


def _ko_next_action(priority: int) -> str:
    return {
        1: "Post-hoc gate나 object-set-only gate를 더 튜닝하기보다 retrieval, candidate generation, propagation을 joint objective로 묶고, no-harm/calibration target을 cross-seed 전이에 맞게 학습한다.",
        2: "Selective correction으로 수정 범위는 줄였지만 correction trigger rate는 줄이지 못했다. 다음은 transition model 자체를 안정화하고, correction trigger를 learned uncertainty/state-validity objective로 학습해 대부분의 sparse update를 수정하지 않아도 integrity가 유지되게 만든다.",
        3: "N_bg=256 이상에서 training budget을 키운 baseline-complete run, 더 많은 mechanism, long-horizon simulator rollout, perception/state adapter를 추가한다.",
        4: "명시적인 mechanism-shift detector를 학습하고, 더 어려운 held-out mechanism에서 selective adaptation policy를 평가한다.",
        5: "Sparse-output gate를 넘어 calibration-aware mechanism uncertainty, branch calibration loss, multi-step ECE/Brier/NLL을 학습한다.",
        6: "Energy, allocator traffic, sparse-kernel behavior, Pareto frontier, trained matched-or-better speedup을 측정한다.",
        7: "더 강한 closed-loop 또는 multi-horizon corruption 실험을 수행하고, objectification component가 큰 branch/rollout loss를 설명하는지 검증한다.",
    }[priority]


if __name__ == "__main__":
    main()
