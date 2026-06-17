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
    end_to_end_path = ROOT / "wpu_v2_end_to_end_candidate_selector_summary.csv"
    joint_generator_path = ROOT / "wpu_v2_joint_candidate_generator_summary.csv"
    verified_controller_path = ROOT / "wpu_v2_verified_candidate_controller_summary.csv"
    joint_adapter_path = ROOT / "wpu_v2_joint_propagation_adapter_summary.csv"
    joint_utility_path = ROOT / "wpu_v2_joint_utility_verifier_summary.csv"
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
    end_to_end_best = None
    end_to_end_note = ""
    if end_to_end_path.exists():
        end_to_end_rows = _read_rows(end_to_end_path)
        end_to_end_unconstrained = max(end_to_end_rows, key=lambda row: float(row["gap_closure_fraction"]))
        end_to_end_safe_rows = [
            row for row in end_to_end_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        end_to_end_selected_rows = [
            row for row in end_to_end_rows if row["policy"] == "train_selected_end_to_end_candidate_selector"
        ]
        end_to_end_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in end_to_end_safe_rows)
            if end_to_end_safe_rows
            else None
        )
        end_to_end_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in end_to_end_selected_rows)
            if end_to_end_selected_rows
            else None
        )
        end_to_end_best = end_to_end_selected_best or float(end_to_end_unconstrained["gap_closure_fraction"])
        end_to_end_note = (
            f" Fixed-candidate/fixed-propagator downstream-loss selector training is also a negative result for P1: "
            f"best closure is {float(end_to_end_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {end_to_end_safe_best:.6f}"
                if end_to_end_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {end_to_end_selected_best:.6f}."
                if end_to_end_selected_best is not None
                else "."
            )
        )
    joint_generator_best = None
    joint_generator_note = ""
    if joint_generator_path.exists():
        generator_rows = _read_rows(joint_generator_path)
        best_generator_oracle = max(generator_rows, key=lambda row: float(row["learned_generator_oracle_closure"]))
        best_generator_evaluator = max(generator_rows, key=lambda row: float(row["evaluator_gap_closure"]))
        joint_generator_best = float(best_generator_evaluator["evaluator_gap_closure"])
        joint_generator_note = (
            f" Joint candidate generation is a new negative diagnostic: learned-generated oracle closure reaches "
            f"{float(best_generator_oracle['learned_generator_oracle_closure']):.6f} at K={best_generator_oracle['causal_k']}, "
            f"but deployed evaluator closure reaches only {joint_generator_best:.6f} at K={best_generator_evaluator['causal_k']}. "
            f"This shows candidate generation creates headroom but does not make it deployable without joint verification."
        )
    verified_best = None
    verified_note = ""
    if verified_controller_path.exists():
        verified_rows = _read_rows(verified_controller_path)
        verified_unconstrained = max(verified_rows, key=lambda row: float(row["gap_closure_fraction"]))
        verified_safe_rows = [
            row for row in verified_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        verified_selected_rows = [
            row for row in verified_rows if row["policy"] == "train_selected_verified_candidate_controller"
        ]
        verified_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in verified_safe_rows)
            if verified_safe_rows
            else None
        )
        verified_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in verified_selected_rows)
            if verified_selected_rows
            else None
        )
        verified_best = verified_selected_best or float(verified_unconstrained["gap_closure_fraction"])
        verified_note = (
            f" Label-free sparse/local-dense verification signatures are also negative as a standalone P1 fix: "
            f"best closure is {float(verified_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {verified_safe_best:.6f}"
                if verified_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {verified_selected_best:.6f}."
                if verified_selected_best is not None
                else "."
            )
        )
    joint_adapter_best = None
    joint_adapter_note = ""
    if joint_adapter_path.exists():
        adapter_rows = _read_rows(joint_adapter_path)
        adapter_unconstrained = max(adapter_rows, key=lambda row: float(row["gap_closure_fraction"]))
        adapter_safe_rows = [
            row for row in adapter_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        adapter_selected_rows = [
            row for row in adapter_rows if row["policy"] == "train_selected_joint_propagation_adapter"
        ]
        adapter_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in adapter_safe_rows)
            if adapter_safe_rows
            else None
        )
        adapter_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in adapter_selected_rows)
            if adapter_selected_rows
            else None
        )
        joint_adapter_best = adapter_selected_best or float(adapter_unconstrained["gap_closure_fraction"])
        joint_adapter_note = (
            f" A candidate-aware branch-logit propagation adapter is now tested as a shallow joint step: "
            f"best closure is {float(adapter_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {adapter_safe_best:.6f}"
                if adapter_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {adapter_selected_best:.6f}."
                if adapter_selected_best is not None
                else "."
            )
        )
    joint_utility_best = None
    joint_utility_note = ""
    if joint_utility_path.exists():
        utility_rows = _read_rows(joint_utility_path)
        utility_unconstrained = max(utility_rows, key=lambda row: float(row["gap_closure_fraction"]))
        utility_safe_rows = [
            row for row in utility_rows if float(row.get("mean_harmful_accept_rate", 1.0)) <= 0.25
        ]
        utility_selected_rows = [
            row for row in utility_rows if row["policy"] == "train_selected_joint_utility_verifier"
        ]
        utility_safe_best = (
            max(float(row["gap_closure_fraction"]) for row in utility_safe_rows)
            if utility_safe_rows
            else None
        )
        utility_selected_best = (
            max(float(row["gap_closure_fraction"]) for row in utility_selected_rows)
            if utility_selected_rows
            else None
        )
        joint_utility_best = utility_selected_best or float(utility_unconstrained["gap_closure_fraction"])
        joint_utility_note = (
            f" A joint utility verifier that combines object-set tensors, verification signatures, "
            f"uncertainty, and no-harm safety is also negative: best closure is "
            f"{float(utility_unconstrained['gap_closure_fraction']):.6f}"
            + (
                f", safe best is {utility_safe_best:.6f}"
                if utility_safe_best is not None
                else ", with no harmful-accept <= 0.25 deployment"
            )
            + (
                f", and train-selected closure is {utility_selected_best:.6f}."
                if utility_selected_best is not None
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
            end_to_end_best,
            joint_generator_best,
            verified_best,
            joint_adapter_best,
            joint_utility_best,
        ]
        if value is not None
    )
    source = (
        end_to_end_path
        if end_to_end_best == best
        else joint_path
        if joint_best == best
        else invariant_path
        if invariant_best == best
        else regret_path
        if regret_best == best
        else verified_controller_path
        if verified_best == best
        else joint_adapter_path
        if joint_adapter_best == best
        else joint_utility_path
        if joint_utility_best == best
        else joint_generator_path
        if joint_generator_best == best
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
        f"Best deployed closure is {best:.6f}; previous aggregate-policy best is {aggregate_best:.6f} and mean aggregate closure is {mean:.6f}.{noharm_note}{regret_note}{penalty_note}{perturb_note}{safety_note}{crossfit_note}{invariant_note}{joint_note}{joint_regression_note}{end_to_end_note}{joint_generator_note}{verified_note}{joint_adapter_note}{joint_utility_note}",
        "Move beyond post-hoc, object-set-only, selector-loss-only, generator-only, verification-feature-only, shallow branch-logit-adapter, and fixed-propagator utility/safety verifier probes: train candidate generation, retrieval, propagation dynamics, propagation verification, and calibrated no-harm objectives as one held-out-seed objective.",
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
    learned_trigger_note = ""
    learned_trigger_path = ROOT / "pybullet_learned_correction_trigger.csv"
    if learned_trigger_path.exists():
        learned_rows = [
            row
            for row in _read_rows(learned_trigger_path)
            if row.get("row_type") == "summary"
        ]
        if learned_rows:
            learned_joint_rows = [
                row for row in learned_rows if str(row.get("meets_joint_target", "0")) == "1"
            ]
            learned_best = max(learned_rows, key=lambda row: float(row["state_integrity_score"]))
            learned_policy_rows = [
                row for row in learned_rows if str(row.get("policy", "")).startswith("learned_")
            ]
            learned_policy_best = (
                max(learned_policy_rows, key=lambda row: float(row["state_integrity_score"]))
                if learned_policy_rows
                else None
            )
            learned_low_correction = [
                row for row in learned_rows if float(row.get("correction_rate", 1.0)) <= 0.25
            ]
            learned_best_low = (
                max(learned_low_correction, key=lambda row: float(row["state_integrity_score"]))
                if learned_low_correction
                else None
            )
            learned_trigger_note = (
                f" Learned correction-trigger audit on hard held-out seeds finds {len(learned_joint_rows)} "
                f"summary policies meeting integrity>=0.8 and correction_rate<=0.25. Best integrity is "
                f"{float(learned_best['state_integrity_score']):.6f} with correction rate "
                f"{float(learned_best['correction_rate']):.6f}; "
                + (
                    f"best learned-trigger integrity is "
                    f"{float(learned_policy_best['state_integrity_score']):.6f} at correction rate "
                    f"{float(learned_policy_best['correction_rate']):.6f}; "
                    if learned_policy_best is not None
                    else ""
                )
                + (
                    f"the best low-correction policy reaches integrity "
                    f"{float(learned_best_low['state_integrity_score']):.6f} at correction rate "
                    f"{float(learned_best_low['correction_rate']):.6f}."
                    if learned_best_low is not None
                    else "no learned policy satisfies correction_rate<=0.25."
                )
            )
    stable_transition_note = ""
    stable_transition_path = ROOT / "pybullet_stable_transition_sweep.csv"
    if stable_transition_path.exists():
        stable_rows = [
            row
            for row in _read_rows(stable_transition_path)
            if row.get("row_type") == "summary"
        ]
        raw_rows = [row for row in stable_rows if row.get("eval_mode") == "raw_finite_clamped"]
        corrected_rows = [row for row in stable_rows if row.get("eval_mode") == "selective_corrected"]
        joint_rows = [row for row in stable_rows if str(row.get("meets_joint_target", "0")) == "1"]
        if raw_rows and corrected_rows:
            best_raw = max(raw_rows, key=lambda row: float(row["state_integrity_score"]))
            best_corrected = max(corrected_rows, key=lambda row: float(row["low_disruption_integrity_score"]))
            stable_transition_note = (
                f" Stable-transition sweep gives a partial positive result: best raw finite-clamped "
                f"integrity rises to {float(best_raw['state_integrity_score']):.6f} "
                f"({best_raw['config']}) from the finite-clamped baseline {sparse_finite_clamped:.6f}; "
                f"best selective-correction low-disruption score rises to "
                f"{float(best_corrected['low_disruption_integrity_score']):.6f} "
                f"with integrity {float(best_corrected['state_integrity_score']):.6f} and correction rate "
                f"{float(best_corrected['correction_rate']):.6f}. Joint target rows remain {len(joint_rows)}."
            )
    return _row(
        2,
        "Long-horizon state integrity",
        "fail" if best < 0.8 else "partial",
        best,
        0.8,
        "best_wpu_h25_integrity",
        path,
        f"Best WPU H=25 integrity is {best:.6f}; guarded sparse is {sparse_guarded:.6f}, clipped sparse is {sparse_clipped:.6f}, regularized raw sparse is {sparse_regularized:.6f}, rollout-consistency sparse is {sparse_consistency:.6f}, validity sparse is {sparse_validity:.6f}, strong-validity sparse is {sparse_validity_strong:.6f}, unsafe-delta rejected sparse is {sparse_rejected:.6f} with rejection rate {sparse_rejection_rate:.6f}, and rollback sparse is {sparse_rollback:.6f} with rollback rate {sparse_rollback_rate:.6f}.{correction_note}{escalation_note}{finite_note}{selective_note}{trigger_frontier_note}{learned_trigger_note}{stable_transition_note}",
        "Stable-transition loss sweeps reduce correction frequency and improve low-disruption integrity but still miss the low-correction joint target; next step is multi-step/simulator-resynchronized transition training or architecture changes, not another trigger threshold.",
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
        micro_note = ""
        micro_path = ROOT / "pybullet_cup_benchmark_n512_baseline_micro.csv"
        if micro_path.exists():
            micro_rows = _read_rows(micro_path)
            micro_groups: dict[str, list[dict[str, str]]] = {}
            for row in micro_rows:
                micro_groups.setdefault(row["model"], []).append(row)
            micro_summary = [
                {
                    "model": model,
                    "accuracy": statistics.fmean(float(row["branch_accuracy"]) for row in group),
                    "latency": statistics.fmean(float(row["ms_per_sample_forward"]) for row in group),
                }
                for model, group in micro_groups.items()
            ]
            wpu_micro = [row for row in micro_summary if row["model"].startswith("wpu-")]
            baseline_micro = [row for row in micro_summary if not row["model"].startswith("wpu-")]
            if wpu_micro and baseline_micro:
                best_wpu_micro = max(wpu_micro, key=lambda row: row["accuracy"])
                best_baseline_micro = max(baseline_micro, key=lambda row: row["accuracy"])
                micro_speedup = best_baseline_micro["latency"] / max(best_wpu_micro["latency"], 1e-9)
                micro_note = (
                    f" A low-training N_bg=512 baseline-complete micro-screen now reaches total N=517: "
                    f"best WPU `{best_wpu_micro['model']}` reaches accuracy {best_wpu_micro['accuracy']:.6f} "
                    f"versus best baseline `{best_baseline_micro['model']}` at {best_baseline_micro['accuracy']:.6f}, "
                    f"with WPU {micro_speedup:.6f}x faster than that best-accuracy baseline. "
                    "Because it uses only 3 seeds, 2 training steps, and 8 samples, it is coverage evidence rather than a strong accuracy-superiority result."
                )
        n512_medium_note = ""
        n512_medium_path = ROOT / "pybullet_cup_benchmark_n512_medium.csv"
        if n512_medium_path.exists():
            n512_medium_rows = _read_rows(n512_medium_path)
            n512_medium_groups: dict[str, list[dict[str, str]]] = {}
            for row in n512_medium_rows:
                n512_medium_groups.setdefault(row["model"], []).append(row)
            n512_medium_summary = [
                {
                    "model": model,
                    "accuracy": statistics.fmean(float(row["branch_accuracy"]) for row in group),
                    "latency": statistics.fmean(float(row["ms_per_sample_forward"]) for row in group),
                }
                for model, group in n512_medium_groups.items()
            ]
            wpu_n512_medium = [row for row in n512_medium_summary if row["model"].startswith("wpu-")]
            baseline_n512_medium = [row for row in n512_medium_summary if not row["model"].startswith("wpu-")]
            if wpu_n512_medium and baseline_n512_medium:
                best_wpu_n512 = max(wpu_n512_medium, key=lambda row: row["accuracy"])
                best_baseline_n512 = max(baseline_n512_medium, key=lambda row: row["accuracy"])
                n512_speedup = best_baseline_n512["latency"] / max(best_wpu_n512["latency"], 1e-9)
                n512_medium_note = (
                    f" A 5-seed N_bg=512 baseline-complete medium run strengthens that evidence: "
                    f"best WPU `{best_wpu_n512['model']}` reaches accuracy {best_wpu_n512['accuracy']:.6f} "
                    f"versus best baseline `{best_baseline_n512['model']}` at {best_baseline_n512['accuracy']:.6f}, "
                    f"with WPU {n512_speedup:.6f}x faster than that best-accuracy baseline. "
                    "This is stronger than the micro-screen but remains a single cup-family, one-step, small-margin result."
                )
        n512_high_note = ""
        n512_high_path = ROOT / "pybullet_cup_benchmark_n512_high_budget.csv"
        if n512_high_path.exists():
            n512_high_rows = _read_rows(n512_high_path)
            n512_high_groups: dict[str, list[dict[str, str]]] = {}
            for row in n512_high_rows:
                n512_high_groups.setdefault(row["model"], []).append(row)
            n512_high_summary = [
                {
                    "model": model,
                    "accuracy": statistics.fmean(float(row["branch_accuracy"]) for row in group),
                    "latency": statistics.fmean(float(row["ms_per_sample_forward"]) for row in group),
                }
                for model, group in n512_high_groups.items()
            ]
            wpu_n512_high = [row for row in n512_high_summary if row["model"].startswith("wpu-")]
            baseline_n512_high = [row for row in n512_high_summary if not row["model"].startswith("wpu-")]
            if wpu_n512_high and baseline_n512_high:
                best_wpu_n512_high = max(wpu_n512_high, key=lambda row: row["accuracy"])
                best_baseline_n512_high = max(baseline_n512_high, key=lambda row: row["accuracy"])
                n512_high_speedup = best_baseline_n512_high["latency"] / max(
                    best_wpu_n512_high["latency"], 1e-9
                )
                n512_high_note = (
                    f" A higher-budget N_bg=512 baseline-complete run keeps the conditional edge: "
                    f"best WPU `{best_wpu_n512_high['model']}` reaches accuracy {best_wpu_n512_high['accuracy']:.6f} "
                    f"versus best baseline `{best_baseline_n512_high['model']}` at {best_baseline_n512_high['accuracy']:.6f}, "
                    f"with WPU {n512_high_speedup:.6f}x faster than that best-accuracy baseline. "
                    "The margin shrinks at higher budget, so this is not a broad superiority result."
                )
        mechanism_count = max(int(float(row["mechanism_count"])) for row in coverage_rows)
        n512_shift_note = _n512_shift_screen_note()
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
            f"{micro_note}"
            f"{n512_medium_note}"
            f"{n512_high_note}"
            f"{n512_shift_note}"
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
        "Add mechanism-aware propagation and long-horizon simulator rollouts; higher-budget N_bg=512 remains small-margin and N_bg=512 mechanism screens are mixed/negative.",
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
    n512_shift_note = _n512_shift_screen_note()
    if n512_shift_note:
        notes.append(n512_shift_note.strip())
    adapted_route_regret_note = _n512_route_regret_adapted_note()
    if adapted_route_regret_note:
        notes.append(adapted_route_regret_note.strip())
    mechanism_conditioned_note = _n512_mechanism_conditioned_note()
    if mechanism_conditioned_note:
        notes.append(mechanism_conditioned_note.strip())
    mechanism_adapter_note = _n512_mechanism_adapter_multitrain_note()
    if mechanism_adapter_note:
        notes.append(mechanism_adapter_note.strip())
    factorized_shuffled_note = _n512_mechanism_factorized_shuffled_note()
    if factorized_shuffled_note:
        notes.append(factorized_shuffled_note.strip())
    target_local_note = _n512_target_local_loss_note()
    if target_local_note:
        notes.append(target_local_note.strip())
    mechanism_branch_note = _n512_mechanism_branch_note()
    if mechanism_branch_note:
        notes.append(mechanism_branch_note.strip())
    mechanism_branch_stress_note = _n512_mechanism_branch_stress_note()
    if mechanism_branch_stress_note:
        notes.append(mechanism_branch_stress_note.strip())
    branch_expert_note = _n512_branch_expert_note()
    if branch_expert_note:
        notes.append(branch_expert_note.strip())
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
        "Train mechanism-aware propagation, explicit shift detection, and selective adaptation on harder large-N held-out mechanisms.",
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


def _n512_shift_screen_note() -> str:
    parts: list[str] = []
    for path, label in [
        (ROOT / "pybullet_shift_generalization_n512_screen.csv", "N_bg=512 nominal-train mechanism screen"),
        (ROOT / "pybullet_shift_generalization_n512_multimech.csv", "N_bg=512 multi-mechanism-train screen"),
        (
            ROOT / "pybullet_shift_generalization_n512_event_physical_screen.csv",
            "N_bg=512 event+physical-state nominal-train mechanism screen",
        ),
        (
            ROOT / "pybullet_shift_generalization_n512_event_physical_multimech.csv",
            "N_bg=512 event+physical-state multi-mechanism-train screen",
        ),
        (
            ROOT / "pybullet_shift_generalization_n512_route_regret_selected.csv",
            "N_bg=512 selected route-regret nominal-train mechanism screen",
        ),
    ]:
        if not path.exists():
            continue
        rows = _rows_of_type(_read_rows(path), "summary")
        if not rows:
            continue
        mechanisms = sorted({row["eval_mechanism"] for row in rows})
        wins = 0
        ties = 0
        losses = 0
        deltas: list[float] = []
        wpu_models = sorted({row["model"] for row in rows if row["model"].startswith("wpu-")})
        baseline_models = sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
        for mechanism in mechanisms:
            group = [row for row in rows if row["eval_mechanism"] == mechanism]
            best_wpu = max(float(row["branch_accuracy"]) for row in group if row["model"].startswith("wpu-"))
            best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
            delta = best_wpu - best_baseline
            deltas.append(delta)
            if delta > 1e-9:
                wins += 1
            elif delta < -1e-9:
                losses += 1
            else:
                ties += 1
        total_n = max(int(float(row["total_objects_n"])) for row in rows if row.get("total_objects_n", ""))
        seed_count = max(int(float(row["seed_count"])) for row in rows if row.get("seed_count", ""))
        macro_wpu = max(
            statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
            for model in wpu_models
        )
        macro_baseline = max(
            statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
            for model in baseline_models
        )
        parts.append(
            f"{label} at total N={total_n} covers {len(mechanisms)} mechanisms over {seed_count} seeds: "
            f"WPU win/tie/loss is {wins}/{ties}/{losses}, mean best-WPU-minus-best-baseline margin is "
            f"{statistics.fmean(deltas):+.6f}, and best macro WPU/baseline accuracies are "
            f"{macro_wpu:.6f}/{macro_baseline:.6f}."
        )
    if not parts:
        return ""
    return (
        " " + " ".join(parts) + " These large-N mechanism screens expand coverage but are mixed/negative: "
        "preserving action and physical state scalars can recover the nominal-train shift screen, but small "
        "identifiable K still does not by itself solve multi-mechanism law learning."
    )


def _n512_route_regret_adapted_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_route_regret_adapted_screen.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    if not rows:
        return ""
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    wpu_rows = [row for row in rows if row["model"] == "wpu-cws-indexed-physics-regret-hybrid"]
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == "wpu-cws-indexed-physics-regret-hybrid")
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in wpu_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in wpu_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    return (
        f"N_bg=512 selected route-regret plus matched mechanism-prior adaptation covers {len(mechanisms)} "
        f"shifted mechanisms: route-regret WPU win/tie/loss versus best baseline is {wins}/{ties}/{losses}, "
        f"mean margin is {statistics.fmean(deltas):+.6f}, macro WPU/baseline accuracy is "
        f"{macro_wpu:.6f}/{best_macro_baseline:.6f}, and dense compute is {macro_dense:.6f}. "
        "Thus matched prior adaptation does not rescue route-regret WPU; mechanism state must condition "
        "propagation dynamics, not only post-hoc priors or thresholds."
    )


def _n512_mechanism_conditioned_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_mechanism_conditioned_screen.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    if not rows:
        return ""
    target_model = "wpu-cws-indexed-mechanism-conditioned"
    target_rows = [row for row in rows if row["model"] == target_model]
    if not target_rows:
        return ""
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == target_model)
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_ece = statistics.fmean(float(row["ece"]) for row in target_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in target_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    worst_delta = min(deltas)
    return (
        f"N_bg=512 mechanism-conditioned propagation screen covers {len(mechanisms)} shifted mechanisms: "
        f"mechanism-conditioned WPU win/tie/loss versus best baseline is {wins}/{ties}/{losses}, "
        f"mean margin is {statistics.fmean(deltas):+.6f}, macro WPU/baseline accuracy is "
        f"{macro_wpu:.6f}/{best_macro_baseline:.6f}, ECE is {macro_ece:.6f}, and dense compute is "
        f"{macro_dense:.6f}. This is the first positive screen after route-regret adaptation failure, "
        f"but the worst mechanism remains negative at {worst_delta:+.6f}; larger sweeps are required."
    )


def _n512_mechanism_adapter_multitrain_note() -> str:
    nominal_path = ROOT / "pybullet_shift_generalization_n512_mechanism_conditioned_5seed.csv"
    multitrain_path = ROOT / "pybullet_shift_generalization_n512_mechanism_adapter_multitrain_5seed.csv"
    if not multitrain_path.exists():
        return ""
    parts: list[str] = []
    if nominal_path.exists():
        nominal_rows = _rows_of_type(_read_rows(nominal_path), "summary")
        target_model = "wpu-cws-indexed-mechanism-conditioned"
        target_rows = [row for row in nominal_rows if row["model"] == target_model]
        if target_rows:
            mechanisms = sorted({row["eval_mechanism"] for row in nominal_rows})
            wins = 0
            ties = 0
            losses = 0
            deltas: list[float] = []
            for mechanism in mechanisms:
                group = [row for row in nominal_rows if row["eval_mechanism"] == mechanism]
                wpu = next(row for row in group if row["model"] == target_model)
                best_baseline = max(
                    float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-")
                )
                delta = float(wpu["branch_accuracy"]) - best_baseline
                deltas.append(delta)
                if delta > 1e-9:
                    wins += 1
                elif delta < -1e-9:
                    losses += 1
                else:
                    ties += 1
            macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
            best_macro_baseline = max(
                statistics.fmean(float(row["branch_accuracy"]) for row in nominal_rows if row["model"] == model)
                for model in sorted({row["model"] for row in nominal_rows if not row["model"].startswith("wpu-")})
            )
            parts.append(
                f"Nominal-only 5-seed mechanism-conditioned expansion is negative: win/tie/loss "
                f"{wins}/{ties}/{losses}, mean margin {statistics.fmean(deltas):+.6f}, and macro "
                f"WPU/baseline accuracy {macro_wpu:.6f}/{best_macro_baseline:.6f}."
            )
    rows = _rows_of_type(_read_rows(multitrain_path), "summary")
    target_model = "wpu-cws-indexed-mechanism-adapter"
    target_rows = [row for row in rows if row["model"] == target_model]
    if not target_rows:
        return " ".join(parts)
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == target_model)
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_ece = statistics.fmean(float(row["ece"]) for row in target_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in target_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    parts.append(
        f"Primitive multi-mechanism training with an object-wise sparse mechanism adapter is conditionally "
        f"positive: win/tie/loss {wins}/{ties}/{losses}, mean margin {statistics.fmean(deltas):+.6f}, "
        f"macro WPU/baseline accuracy {macro_wpu:.6f}/{best_macro_baseline:.6f}, ECE {macro_ece:.6f}, "
        f"and dense compute {macro_dense:.6f}. Remaining failures are edge-composition and no_catch shifts, "
        "so this is not broad zero-shot mechanism generalization."
    )
    return " ".join(parts)


def _n512_mechanism_factorized_shuffled_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    target_model = "wpu-cws-indexed-mechanism-factorized"
    target_rows = [row for row in rows if row["model"] == target_model]
    if not target_rows:
        return ""
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == target_model)
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in target_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    return (
        "Corrected shuffled multi-mechanism training downgrades the previous adapter screen: "
        f"factorized sparse WPU win/tie/loss is {wins}/{ties}/{losses}, mean margin is "
        f"{statistics.fmean(deltas):+.6f}, macro WPU/baseline accuracy is "
        f"{macro_wpu:.6f}/{best_macro_baseline:.6f}, and dense compute is {macro_dense:.6f}. "
        "The earlier unshuffled multi-mechanism positive should be treated as order-sensitive; "
        "edge-conditioned composition remains unsolved."
    )


def _n512_target_local_loss_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_target_local_loss_multitrain_5seed.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    target_model = "wpu-cws-indexed-mechanism-factorized"
    target_rows = [row for row in rows if row["model"] == target_model]
    if not target_rows:
        return ""
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == target_model)
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_target_mse = statistics.fmean(float(row["target_mse"]) for row in target_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    return (
        "Target-local delta supervision is now audited as a direct fix for large-N loss dilution. "
        f"At weight 1.0 the factorized sparse WPU reaches win/tie/loss {wins}/{ties}/{losses}, "
        f"mean margin {statistics.fmean(deltas):+.6f}, macro WPU/baseline accuracy "
        f"{macro_wpu:.6f}/{best_macro_baseline:.6f}, and target-object MSE {macro_target_mse:.6f}. "
        "Lower weights 0.25 and 0.5 also reduce neither the edge-conditioned branch failures nor the "
        "macro gap. The result is a useful negative diagnostic: local state-delta supervision can expose "
        "state prediction quality, but branch-composition needs branch-conditioned or mechanism-specific "
        "transition dynamics rather than a scalar loss reweighting."
    )


def _n512_mechanism_branch_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_mechanism_branch_multitrain_5seed.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    target_model = "wpu-cws-indexed-mechanism-branch"
    target_rows = [row for row in rows if row["model"] == target_model]
    if not target_rows:
        return ""
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    wins = 0
    ties = 0
    losses = 0
    deltas: list[float] = []
    for mechanism in mechanisms:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        wpu = next(row for row in group if row["model"] == target_model)
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        delta = float(wpu["branch_accuracy"]) - best_baseline
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_ece = statistics.fmean(float(row["ece"]) for row in target_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in target_rows)
    best_macro_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in rows if row["model"] == model)
        for model in sorted({row["model"] for row in rows if not row["model"].startswith("wpu-")})
    )
    return (
        "Mechanism-conditioned branch transition is the first positive large-N follow-up after the shuffled "
        "factorized and target-local negative diagnostics: WPU win/tie/loss is "
        f"{wins}/{ties}/{losses}, mean margin {statistics.fmean(deltas):+.6f}, macro WPU/baseline "
        f"accuracy {macro_wpu:.6f}/{best_macro_baseline:.6f}, ECE {macro_ece:.6f}, and dense compute "
        f"{macro_dense:.6f}. This supports branch-conditioned transition dynamics as the next WPU direction, "
        "but it is still a positive screen rather than broad superiority because three mechanisms remain below "
        "the best dense baseline."
    )


def _n512_mechanism_branch_stress_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv"
    wpu_h64_path = ROOT / "pybullet_shift_generalization_n512_mechanism_branch_h64_trainpool40_steps16_samples40_3seed.csv"
    baseline_h64_path = ROOT / "pybullet_shift_generalization_n512_baselines_h64_trainpool40_steps16_samples40_3seed.csv"
    if not path.exists() or not wpu_h64_path.exists() or not baseline_h64_path.exists():
        return ""
    h32_rows = _rows_of_type(_read_rows(path), "summary")
    h64_rows = _rows_of_type(_read_rows(wpu_h64_path), "summary") + _rows_of_type(_read_rows(baseline_h64_path), "summary")
    h32_wpu = [row for row in h32_rows if row["model"] == "wpu-cws-indexed-mechanism-branch"]
    h64_wpu = [row for row in h64_rows if row["model"] == "wpu-cws-indexed-mechanism-branch"]
    if not h32_wpu or not h64_wpu:
        return ""
    h32_macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in h32_wpu)
    h32_best_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in h32_rows if row["model"] == model)
        for model in sorted({row["model"] for row in h32_rows if not row["model"].startswith("wpu-")})
    )
    h64_macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in h64_wpu)
    h64_best_baseline = max(
        statistics.fmean(float(row["branch_accuracy"]) for row in h64_rows if row["model"] == model)
        for model in sorted({row["model"] for row in h64_rows if not row["model"].startswith("wpu-")})
    )
    return (
        "A step/sample stress audit downgrades the mechanism-branch result from robust accuracy evidence to "
        "a short-budget sparse-efficiency screen. With explicit train pool control, h32 WPU reaches "
        f"{h32_macro_wpu:.6f} versus {h32_best_baseline:.6f} for the best baseline, and in a fair h64 "
        f"capacity check WPU reaches {h64_macro_wpu:.6f} versus {h64_best_baseline:.6f}. Dense compute remains "
        "0.000000, but accuracy does not beat the stronger dense/token baselines. The next fix must improve "
        "transition-head expressivity and optimization."
    )


def _n512_branch_expert_note() -> str:
    path = ROOT / "pybullet_shift_generalization_n512_mechanism_branch_expert_trainpool40_steps16_samples40_3seed.csv"
    if not path.exists():
        return ""
    rows = _rows_of_type(_read_rows(path), "summary")
    target_rows = [row for row in rows if row["model"] == "wpu-cws-indexed-mechanism-branch-expert"]
    if not target_rows:
        return ""
    macro_wpu = statistics.fmean(float(row["branch_accuracy"]) for row in target_rows)
    macro_ece = statistics.fmean(float(row["ece"]) for row in target_rows)
    macro_dense = statistics.fmean(float(row["dense_compute_ratio"]) for row in target_rows)
    return (
        "Branch-specific output experts are also a negative standalone fix under the h32 stress protocol: "
        f"macro accuracy is {macro_wpu:.6f}, ECE {macro_ece:.6f}, and dense compute {macro_dense:.6f}. "
        "They improve some edge/catch composed cases but lose general mechanism accuracy, so the next "
        "architecture step should condition the sparse propagation messages on relation type rather than "
        "only adding branch-logit experts."
    )


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
    interpretations = {
        1: "Candidate-regret deployment sweep은 margin-only gate보다 강하지만, 논문용 observed 값은 test-best sweep이 아니라 train-selected deployment를 우선 사용한다. 현재 train-selected closure는 0.328025로 목표 0.5에 못 미치고 harmful accept도 0.251111로 threshold 근처에 남아 있어 P1은 fail이다. Harmful-accept/ranking penalty 학습은 안전하지만 closure가 0.081253으로 떨어지고, feature perturbation은 test-sweep safe closure를 0.329756까지 조금 올리지만 train-selected closure는 0.312586에 머문다. 별도 safety/utility head도 negative result다. Best closure는 0.147450, safe best는 0.090719, train-selected closure는 0.144863에 그친다. Cross-fit ensemble regret gate도 train-selected overfit 가설을 부정하는 negative result다. 최고 closure는 0.287268, safe best는 0.279738, cross-fit selected closure는 0.270989로 direct regret gate보다 낮다. Descriptor standardization과 group-DRO no-harm training도 standalone 해결책이 아니며 train-selected closure는 0.093863이다. 새 joint object-set candidate gate도 negative result다. Best closure는 0.101454, safe best는 0.101454, train-selected closure는 0.072167이고 mean regret correlation은 -0.000180에 가깝다. Regression-heavy ablation도 K=16에서 best closure 0.034751, train-selected closure -0.003089에 그친다. Fixed-candidate/fixed-propagator downstream-loss selector도 negative result다. Best closure는 0.106927이고 harmful-accept <= 0.25를 만족하는 deployment가 없으며 train-selected closure는 0.096833에 그친다. 새 joint candidate generator도 learned-generated oracle closure는 K=16에서 0.361251까지 만들지만 deployed evaluator closure는 0.042951에 그친다. Label-free sparse/local-dense verification signature도 standalone 해결책이 아니다. Best closure는 0.024989, safe best는 0.023029, train-selected closure는 0.024989로 direct regret gate보다 낮다. Shallow candidate-aware branch-logit propagation adapter도 standalone 해결책이 아니다. Best/safe closure는 0.092185, train-selected closure는 0.069911로 direct regret gate보다 낮다. 새 joint utility verifier는 object-set tensor, verification signature, uncertainty, no-harm safety를 같이 쓰지만 best/safe closure 0.097845, train-selected closure 0.077781에 그쳐 direct regret gate보다 낮다. 따라서 P1 병목은 object-set feature 부재, selector-loss 교체, 후보 생성 단독, verification feature 단독, 작은 output adapter, fixed-propagator utility/safety head의 문제가 아니라 cross-seed candidate regret target, 후보 생성, retrieval, 전파 검증, propagation dynamics를 함께 안정화해야 하는 문제다.",
        2: "Rollback-only memory layer는 sparse WPU H=25 integrity를 0.988647까지 올리지만 rollback rate가 0.812500으로 매우 높다. Corrected rollback은 rollback rate를 0.564167까지 낮추지만 integrity가 0.900288로 떨어진다. Escalated corrected rollback은 local-dense fallback을 사용해 integrity를 0.914831로 올리고 rollback rate를 0.000000으로 낮춘다. Finite-corrected run은 finite-safe delta clipping과 correction-only projection으로 integrity 0.958735, rollback rate 0.000000, escalation rate 0.000000을 달성하지만 correction rate가 0.784166으로 높다. 새 selective correction은 같은 integrity 0.958735를 유지하면서 corrected object fraction을 0.027461로 낮추고 low-disruption integrity를 0.758574까지 올린다. 그러나 correction trigger rate는 여전히 0.784166이다. Correction-trigger frontier audit은 integrity>=0.8 및 correction_rate<=0.25를 동시에 만족하는 trigger policy가 0개임을 보인다. 최고 low-correction trigger는 selective_corrected_entropy035이고 integrity 0.653668, correction rate 0.230000에 그친다. 새 learned correction-trigger hard-seed audit도 joint target을 만족한 summary policy가 0개다. 최고 learned trigger integrity는 0.958931이지만 correction rate가 0.791667이고, correction_rate<=0.25 조건의 최고 integrity는 0.523279에 그친다. Stable-transition sweep은 partial positive다. delta_norm_strong은 raw finite-clamped integrity를 0.633398까지 올리고, selective-correction low-disruption score를 0.809071까지 올리며 correction rate를 0.598333까지 낮춘다. 하지만 joint target row는 여전히 0개다. 따라서 P2는 memory-layer disruption과 correction frequency를 일부 줄였지만 raw delta stability와 low-frequency correction-trigger 학습은 아직 해결되지 않았다.",
        3: "PyBullet benchmark는 7개 seed와 background N_bg=128까지 본훈련 baseline-complete로 확장됐고, N_bg=256 screen은 total N=261에서 WPU/graph/token baseline을 모두 완료했다. Medium-training N_bg=256 baseline-complete run은 evidence quality를 올린다. Best WPU인 wpu-cws-indexed-local-dense는 accuracy 0.466667이고 best baseline인 graph-transformer는 0.450000이며, best WPU는 해당 best-accuracy baseline보다 forward latency 기준 60.629526x 빠르다. 단, margin이 작고 단일 cup family이므로 broad superiority claim은 아니다. N_bg=512 baseline-complete micro-screen은 total N=517에서 WPU/graph/token baseline을 모두 포함하고 best WPU accuracy 0.375000 vs best baseline 0.333333을 보였지만, 3 seeds, 2 steps, 8 samples의 coverage evidence로만 해석한다. 5-seed N_bg=512 baseline-complete medium run은 best WPU 0.387500 vs best baseline 0.362500, speedup 67.400400x로 evidence를 강화했다. 새 higher-budget N_bg=512 run은 best WPU wpu-cws-indexed-local-dense accuracy 0.433333 vs best baseline graph-transformer 0.425000, speedup 57.595711x를 보인다. Edge는 유지되지만 margin은 더 작아졌으므로 조건부 evidence이지 broad superiority claim은 아니다. N_bg=512 mechanism screens는 total N=517에서 7개 mechanism을 다루며 coverage를 넓혔다. 원본 screen은 mixed/negative였고, action event와 physical object-state scalar를 보존한 후 nominal-train shift screen은 WPU win/tie/loss 4/0/3, mean margin +0.002976으로 회복됐지만 multi-mechanism-train screen은 2/2/3, mean margin -0.032738로 여전히 mixed/negative다. 따라서 large-N 계산 이점은 보존되지만 mechanism-law learning은 별도 문제다. Coverage audit는 PyBullet 축을 추적하고, 별도의 WPU-only large-state extension도 N_bg=512, total N=517까지 실행됐지만 dense graph baseline이 같은 higher-budget protocol에서 완료되지 않았으므로 systems feasibility evidence로만 취급한다. Simulator-backed evidence는 강화됐지만 mechanism-aware propagation, long-horizon rollout, perception/state adapter가 아직 부족하다.",
        4: "7-seed nominal-shift benchmark는 mixed이고, 3-seed leave-family-out probe는 win-rate 0.750000을 보인다. 7-seed composition-shift stress에서는 WPU가 accuracy 기준 3/3에서 baseline 이상이며 평균 accuracy delta가 0.071428이다. Branch-prior audit은 catch_heavy가 prior-dominated shift임을 보인다. Mechanism-prior adaptation은 shifted WPU win-rate를 0.333333에서 0.666667로 올리고 prior-dominated shift를 1개에서 0개로 줄인다. Prior-strength sweep의 accuracy-best 설정은 strength=0.75, mean WPU accuracy 0.601852지만 shifted win-rate는 0.666667에 머문다. Calibration-selected prior는 mean accuracy/ECE를 개선하지만 shifted win-rate는 0.333333에 머문다. Few-shot mechanism adaptation은 shifted WPU win-rate 1.000000, mean margin change 0.050264까지 도달하지만 mechanism별 calibration set을 쓰는 adapted protocol이다. Mechanism-aware adaptive policy는 selected-prior와 few-shot adaptation을 선택적으로 결합해 shifted win-rate 1.000000, mean accuracy change 0.198412, margin change 0.058201, ECE change -0.099347, Brier change -0.155443에 도달한다. 새 calibration-statistic shift detector는 mechanism 이름 대신 base ECE와 majority-prior gap으로 같은 정책을 복원하며 nominal false adaptation 0, shifted win-rate 1.000000을 달성한다. 그러나 calibration label과 adaptation sample을 쓰므로 detect-and-adapt protocol이지 zero-shot generalization은 아니다. N_bg=512 원본 nominal-train screen은 7개 mechanism에서 WPU win/tie/loss 2/1/4, 평균 margin -0.047619이고 multi-mechanism-train screen도 2/0/5와 -0.095238로 약했다. 이 과정에서 action-conditioned event와 physical object-state scalar가 tensorization에서 빠져 있음을 발견해 수정했다. 수정 후 nominal-train screen은 4/0/3, mean margin +0.002976으로 회복됐지만 multi-mechanism-train screen은 2/2/3, mean margin -0.032738에 머문다. 따라서 large-N에서 작은 K가 식별되고 물리 state scalar를 보존해도 mechanism law를 안정적으로 학습하지 못하면 WPU 정확도 우위는 사라진다. P4는 adapted regime에서 강화됐지만 large-N zero-shot/multi-mechanism law generalization은 solved가 아니다.",
        5: "7-seed 평균 WPU ECE ratio는 0.963449이고, leave-family-out 평균 ECE ratio는 0.972745로 양호하지만, calibrated mixture probe에서는 1.133834로 악화된다. 7-seed composition-shift stress의 평균 ECE ratio는 1.014879이고 no_catch에서 1.166073까지 악화된다. 이는 3-seed stress보다 안정적이지만 여전히 calibration 우위는 아니다. Temperature+bias calibration은 no_catch를 개선하지만 3개 mechanism 중 1개만 ECE ratio가 개선되어 보편 해결책은 아니다. Branch-prior audit은 catch_heavy에서 majority prior 0.753968이 best WPU 0.408730을 크게 앞선다는 점을 보여준다. Mechanism-prior adaptation은 accuracy를 개선하지만 shifted mean ECE를 0.024819 악화시킨다. Prior-strength sweep에서도 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 strength가 없었다. Calibration-selected prior는 shifted mean ECE를 -0.046204, Brier를 -0.105470 개선하지만 baseline win-rate는 올리지 못한다. Few-shot mechanism adaptation도 ECE를 -0.055342 개선한다. Mechanism-aware adaptive policy와 calibration-statistic shift detector는 shifted accuracy를 +0.198412, margin을 +0.058201, ECE를 -0.099347, Brier를 -0.155443 개선해 detect-and-adapt calibration에는 긍정적이다. Uncertainty-gated local-dense recompute는 aggregate accuracy를 +0.071428, ECE를 -0.016396 개선하지만 dense recompute rate가 0.985450으로 거의 full recompute다. Static low-cost gate는 recompute rate 0.025132에서 accuracy를 +0.009260 올리지만 ECE를 +0.005395 악화시킨다. Learned sparse-output benefit gate는 source low-cost에서 accuracy를 +0.052910 올리지만 ECE를 +0.010769 악화시킨다. 새 mechanism-selective calibration gate는 cost_proxy<=0.25에서 non-reference calibration-safe policy 1개를 만든다. Best safe policy는 cost 0.247355, accuracy delta +0.029100, ECE delta -0.001652, Brier delta -0.030758이다. 따라서 P5는 전역/zero-shot gate로는 미해결이지만, mechanism-aware adapted routing에서는 약한 positive sub-regime이 확인됐다.",
        6: "Tensor-byte reduction은 0.997454, CPU sparse-forward reduction은 0.996975, CUDA sparse-forward reduction은 0.996216까지 관측됐다. Screening-only energy proxy도 추가됐지만 실제 전력 측정은 아니다. Matched-speedup audit의 판정 기준을 corrected matched-or-better로 고치면 N=133에서는 best-accuracy non-WPU baseline 대비 WPU가 더 정확하고 더 빠르다. Pareto audit에서도 WPU는 N=133에서 frontier에 올라가지만 N=5에서는 token에 지배된다. Systems claim-boundary audit은 supported proxy 축 4개, partial trained 축 2개, real-power/sparse-kernel 미측정 축 1개를 분리한다. Branch-overlay memory proxy reduction은 0.874128이지만 CUDA peak-memory proxy reduction은 0.304080에 그친다. Real energy와 sparse-kernel behavior는 아직 미해결이다.",
        7: "Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. 새 loss-coupling audit은 worst mean accuracy drop이 wpu-cws-indexed-local-dense/combined에서 0.027778, worst mean MSE increase가 wpu-cws-indexed-sparse/drop_relations_heavy에서 0.087356임을 보인다. MSE degradation과 가장 강하게 연결된 component deficit은 selected_k_mean(|r|=0.481851)이고 accuracy degradation과 가장 강하게 연결된 component deficit은 relation_confidence(|r|=0.352431)이다. 따라서 objectification metric과 downstream loss의 연결은 시작됐지만 branch accuracy 변화가 작아 closed-loop/multi-horizon 검증이 필요하다.",
    }
    text = interpretations[priority]
    if priority in {3, 4} and (ROOT / "pybullet_shift_generalization_n512_route_regret_selected.csv").exists():
        text += (
            " 새 N_bg=512 selected route-regret nominal-train screen도 mixed/negative다. "
            "Selected route-regret WPU는 dense compute를 0.071429로 낮게 유지하지만 "
            "best-WPU 대 best-baseline win/tie/loss는 2/1/4이고, best macro WPU/baseline "
            "accuracy는 0.377976/0.508929에 그친다. 따라서 route-regret threshold selection은 "
            "계산량 제어에는 도움이 되지만 mechanism-law shift를 해결하지 못한다."
        )
    if priority in {4, 5} and (ROOT / "pybullet_shift_generalization_n512_route_regret_adapted_screen.csv").exists():
        text += (
            " Matched mechanism-prior adaptation screen도 route-regret WPU에는 negative다. "
            "4개 shifted mechanism에서 best baseline 대비 win/tie/loss는 0/0/4이고, "
            "macro WPU/baseline accuracy는 0.312500/0.527778이다. 이는 post-hoc prior나 "
            "threshold가 아니라 mechanism-conditioned propagation dynamics가 필요함을 보여준다."
        )
    if priority in {4, 5} and (ROOT / "pybullet_shift_generalization_n512_mechanism_conditioned_screen.csv").exists():
        text += (
            " 새 N_bg=512 mechanism-conditioned propagation screen은 부분적인 positive result다. "
            "Dense fallback 없이 macro WPU/baseline accuracy는 0.541667/0.500000이고, "
            "best baseline 대비 win/tie/loss는 1/2/1이며 dense compute는 0.000000이다. "
            "다만 edge_shift는 여전히 negative이므로 이는 solved zero-shot mechanism generalization이 "
            "아니라 mechanism-conditioned propagation을 더 큰 sweep으로 확장해야 한다는 증거다."
        )
    if priority in {4, 5} and (ROOT / "pybullet_shift_generalization_n512_mechanism_adapter_multitrain_5seed.csv").exists():
        text += (
            " 더 큰 follow-up은 조건을 좁힌다. Nominal-only 5-seed/7-mechanism 확장은 "
            "negative이고, object-wise adapter도 nominal-only training에서는 negative다. "
            "하지만 primitive mechanisms로 학습한 object-wise sparse mechanism adapter는 "
            "N_bg=512 5-seed에서 macro WPU/baseline accuracy 0.497143/0.472857, "
            "win/tie/loss 3/1/3, dense compute 0.000000을 달성한다. 따라서 P4/P5의 "
            "현실적인 다음 주장은 broad zero-shot이 아니라 primitive mechanism variation을 "
            "학습한 sparse local-law composition이다."
        )
    if priority in {4, 5} and (
        ROOT / "pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv"
    ).exists():
        text += (
            " 이후 training DataLoader shuffle 누락을 발견해 seed-fixed shuffle으로 수정했다. "
            "보정된 5-seed factorized adapter 결과는 negative다. Macro WPU/baseline accuracy는 "
            "0.497143/0.548571이고, win/tie/loss는 2/1/4이며 dense compute는 0.000000이다. "
            "따라서 이전 multi-mechanism positive는 order-sensitive screen으로 낮춰야 하고, "
            "edge-conditioned composition에는 explicit local-law/composition supervision이 필요하다."
        )
    return text


def _ko_next_action(priority: int) -> str:
    return {
        1: "Post-hoc gate, object-set-only gate, selector-loss-only gate, generator-only probe, verification-feature-only probe, shallow output-adapter probe, fixed-propagator utility/safety verifier를 더 튜닝하기보다 candidate generation, retrieval, propagation verification, propagation dynamics를 하나의 joint objective로 묶고, no-harm/calibration target을 held-out seed 전이에 맞게 학습한다.",
        2: "Stable-transition loss sweep은 correction frequency와 low-disruption integrity를 일부 개선했지만 low-correction joint target은 여전히 만족하지 못한다. 다음은 trigger threshold가 아니라 multi-step/simulator-resynchronized transition training 또는 구조 변경이다.",
        3: "N_bg=512 higher-budget run에서도 margin이 작고 N_bg=512 mechanism screens가 mixed/negative이므로, mechanism-aware propagation, long-horizon simulator rollout, perception/state adapter를 추가한다.",
        4: "Mechanism-aware propagation, 명시적인 mechanism-shift detector, selective adaptation을 더 어려운 large-N held-out mechanism에서 평가한다.",
        5: "Sparse-output gate를 넘어 calibration-aware mechanism uncertainty, branch calibration loss, multi-step ECE/Brier/NLL을 학습한다.",
        6: "Energy, allocator traffic, sparse-kernel behavior, Pareto frontier, trained matched-or-better speedup을 측정한다.",
        7: "더 강한 closed-loop 또는 multi-horizon corruption 실험을 수행하고, objectification component가 큰 branch/rollout loss를 설명하는지 검증한다.",
    }[priority]


if __name__ == "__main__":
    main()
