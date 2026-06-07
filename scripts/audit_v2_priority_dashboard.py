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
    best = max(value for value in [aggregate_best, noharm_best, regret_best, safety_best] if value is not None)
    source = regret_path if regret_best == best else path
    return _row(
        1,
        "Candidate-oracle gap",
        "fail" if best < 0.5 else "partial",
        best,
        0.5,
        "gap_closure_fraction",
        source,
        f"Best deployed closure is {best:.6f}; previous aggregate-policy best is {aggregate_best:.6f} and mean aggregate closure is {mean:.6f}.{noharm_note}{regret_note}{penalty_note}{perturb_note}{safety_note}",
        "Strengthen candidate-regret training with calibrated uncertainty, harmful-accept penalties, and cross-seed perturbations.",
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
    return _row(
        2,
        "Long-horizon state integrity",
        "fail" if best < 0.8 else "partial",
        best,
        0.8,
        "best_wpu_h25_integrity",
        path,
        f"Best WPU H=25 integrity is {best:.6f}; guarded sparse is {sparse_guarded:.6f}, clipped sparse is {sparse_clipped:.6f}, regularized raw sparse is {sparse_regularized:.6f}, rollout-consistency sparse is {sparse_consistency:.6f}, validity sparse is {sparse_validity:.6f}, strong-validity sparse is {sparse_validity_strong:.6f}, unsafe-delta rejected sparse is {sparse_rejected:.6f} with rejection rate {sparse_rejection_rate:.6f}, and rollback sparse is {sparse_rollback:.6f} with rollback rate {sparse_rollback_rate:.6f}.{correction_note}{escalation_note}",
        "Simple delta-norm, rollout-consistency, and validity regularization are insufficient; add rollback, correction, and uncertainty escalation.",
    )


def _priority_simulator_grounding() -> dict[str, object]:
    path = ROOT / "pybullet_cup_benchmark_7seed.csv"
    if not path.exists():
        path = ROOT / "pybullet_cup_benchmark.csv"
    rows = _read_rows(path)
    seed_count = len({row["seed"] for row in rows})
    max_background = max(int(float(row["background_objects"])) for row in rows)
    status = "partial" if seed_count >= 2 and max_background >= 128 else "fail"
    return _row(
        3,
        "Simulator-backed benchmark",
        status,
        float(seed_count),
        5.0,
        "seed_count",
        path,
        f"PyBullet benchmark exists with {seed_count} seeds and background up to N_bg={max_background}; the 7-seed extension is still small but less seed-fragile.",
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
    stress_path = ROOT / "pybullet_shift_composition_stress_summary.csv"
    if stress_path.exists():
        stress_rows = _read_rows(stress_path)
        stress_win_rate = sum(1 for row in stress_rows if row["wpu_win"] == "True") / max(len(stress_rows), 1)
        stress_delta = statistics.fmean(float(row["accuracy_delta"]) for row in stress_rows)
        stress_notes = [
            f"composition {row['eval_mechanism']}: WPU {float(row['best_wpu_accuracy']):.6f} vs baseline {float(row['best_baseline_accuracy']):.6f}"
            for row in stress_rows
        ]
        notes.append(
            f"3-seed composition-shift stress win-rate {stress_win_rate:.6f}, "
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
        source_path = adaptation_path if adapted_win_rate >= win_rate else path
        notes.append(
            f"7-seed mechanism-prior adaptation changes shifted WPU win-rate "
            f"{base_adaptation_win_rate:.6f}->{adapted_win_rate:.6f}, mean WPU accuracy "
            f"change {mean_accuracy_change:.6f}, mean WPU ECE change {mean_ece_change:.6f}, "
            f"prior-dominated shifts {prior_before}->{prior_after}"
        )
    status = "partial" if 0.0 < observed_win_rate < 1.0 else ("pass" if observed_win_rate == 1.0 else "fail")
    return _row(
        4,
        "Mechanism-family shift generalization",
        status,
        observed_win_rate,
        1.0,
        "wpu_shift_win_rate",
        source_path,
        "; ".join(notes),
        "Add leave-family-out training, harder shifts, and calibration-safe mechanism-aware branch priors.",
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
    stress_path = ROOT / "pybullet_shift_composition_stress_summary.csv"
    if stress_path.exists():
        stress_rows = _read_rows(stress_path)
        stress_ratio = statistics.fmean(float(row["ece_ratio"]) for row in stress_rows)
        worst = max(stress_rows, key=lambda row: float(row["ece_ratio"]))
        mixture_note += (
            f" A 3-seed composition-shift stress probe gives mean ECE ratio {stress_ratio:.6f}; "
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
        "Add temperature heads, branch calibration loss, multi-step ECE/Brier/NLL, and uncertainty-gated recompute.",
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
    status = "partial" if combined_score < clean_score and frontier < 1.0 else "fail"
    return _row(
        7,
        "Objectification quality to propagation loss",
        status,
        combined_score,
        clean_score,
        "combined_objectification_score",
        path,
        f"Clean score {clean_score:.6f}, combined-corruption score {combined_score:.6f}, combined frontier recall {frontier:.6f}. Metrics exist, but downstream loss coupling is incomplete.",
        "Train/evaluate propagation under controlled objectification corruption and regress loss against report components.",
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
        1: "Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 안전하지만 closure가 0.081253으로 떨어지고, feature perturbation은 test-sweep safe closure를 0.329756까지 조금 올리지만 train-selected closure는 0.312586에 머문다. 별도 safety/utility head도 negative result다. Best closure는 0.147450, safe best는 0.090719, train-selected closure는 0.144863에 그친다.",
        2: "Rollback-only memory layer는 sparse WPU H=25 integrity를 0.988647까지 올리지만 rollback rate가 0.812500으로 매우 높다. Corrected rollback은 rollback rate를 0.564167까지 낮추지만 integrity가 0.900288로 떨어진다. Escalated corrected rollback은 local-dense fallback을 사용해 integrity를 0.914831로 올리고 rollback rate를 0.000000으로 낮춘다. 따라서 P2는 sparse-first/dense-when-needed safety layer가 유효할 수 있음을 보이지만, raw delta stability가 해결된 것은 아니다.",
        3: "PyBullet benchmark는 7개 seed와 background N_bg=128까지 확장됐다. N=133에서 WPU sparse accuracy가 0.547619로 serialized-token 0.539683보다 약간 높지만, serialized-token은 여전히 가장 빠르다. Simulator-backed evidence는 강화됐지만 규모와 mechanism 다양성은 아직 부족하다.",
        4: "7-seed nominal-shift benchmark는 mixed이고, 3-seed leave-family-out probe는 win-rate 0.750000을 보인다. 새 composition-shift stress에서는 WPU가 accuracy 기준 3/3에서 baseline 이상이며 평균 accuracy delta가 0.123457이다. Branch-prior audit은 catch_heavy가 prior-dominated shift임을 보인다. Mechanism-prior adaptation은 shifted WPU win-rate를 0.333333에서 0.666667로 올리고 prior-dominated shift를 1개에서 0개로 줄인다. 따라서 P4는 개선됐지만 아직 solved가 아니다.",
        5: "7-seed 평균 WPU ECE ratio는 0.963449이고, leave-family-out 평균 ECE ratio는 0.972745로 양호하지만, calibrated mixture probe에서는 1.133834로 악화된다. Composition-shift stress의 평균 ECE ratio는 1.327702이고 no_catch에서 2.362081까지 악화된다. Temperature+bias calibration은 no_catch를 개선하지만 3개 mechanism 중 1개만 ECE ratio가 개선되어 보편 해결책은 아니다. Branch-prior audit은 catch_heavy에서 majority prior 0.753968이 best WPU 0.408730을 크게 앞선다는 점을 보여준다. Mechanism-prior adaptation은 accuracy를 개선하지만 shifted mean ECE를 0.024819 악화시키므로, branch probability와 prior adaptation은 아직 안정적이지 않다.",
        6: "Tensor-byte reduction은 0.997454, CPU sparse-forward reduction은 0.996975, CUDA sparse-forward reduction은 0.996216까지 관측됐다. Screening-only energy proxy도 추가됐지만 실제 전력 측정은 아니다. Matched-speedup audit의 판정 기준을 corrected matched-or-better로 고치면 N=133에서는 best-accuracy non-WPU baseline 대비 WPU가 더 정확하고 더 빠르다. Pareto audit에서도 WPU는 N=133에서 frontier에 올라가지만 N=5에서는 token에 지배된다. Real energy와 sparse-kernel behavior는 아직 미해결이다.",
        7: "Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.",
    }[priority]


def _ko_next_action(priority: int) -> str:
    return {
        1: "Candidate-regret 학습에 calibrated uncertainty, harmful-accept penalty, cross-seed perturbation을 더 강하게 넣는다.",
        2: "단순 delta-norm, rollout-consistency, state-validity regularization은 부족하다. Guarded state-store projection을 유지하되 rollback/correction과 uncertainty escalation을 모델-메모리 계층에 넣는다.",
        3: "더 많은 mechanism, long-horizon simulator rollout, parameter-matched 7-seed benchmark를 추가한다.",
        4: "Catch-heavy류 branch-prior shift를 겨냥한 mechanism-aware branch prior와 uncertainty-gated fallback을 추가한다.",
        5: "Post-hoc temperature가 아니라 학습 가능한 calibration head, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.",
        6: "Energy, allocator traffic, sparse-kernel behavior, Pareto frontier, trained matched-or-better speedup을 측정한다.",
        7: "Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.",
    }[priority]


if __name__ == "__main__":
    main()
