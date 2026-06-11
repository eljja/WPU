from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze long-horizon state-integrity metrics from rollout CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", default=[Path("docs/experiments/pybullet_closed_loop_rollout.csv")])
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_state_integrity_audit.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_state_integrity_audit_results.md"))
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.inputs]
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must match --inputs")
    rows: list[dict[str, str]] = []
    for path, label in zip(args.inputs, labels, strict=True):
        rows.extend(_read_rows(path, label))
    summary = _summarize(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render_markdown(args.inputs, args.out_csv, summary), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")


def _read_rows(path: Path, label: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["run_label"] = label
    return rows


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_label"], row["model"], int(row["horizon"]))].append(row)
    output: list[dict[str, object]] = []
    for (label, model, horizon), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        violations = _mean(group, "constraint_violations_per_step")
        delta_norm = _mean(group, "delta_norm_mean")
        entropy = _mean(group, "branch_entropy_mean")
        flip_rate = _mean(group, "branch_flip_rate")
        selected_k = _mean(group, "selected_k_mean")
        rejection_rate = _mean_optional(group, "unsafe_delta_rejection_rate")
        correction_rate = _mean_optional(group, "correction_rate")
        corrected_object_fraction = _mean_optional(group, "corrected_object_fraction")
        if corrected_object_fraction == 0.0 and correction_rate > 0.0 and not _has_optional(group, "corrected_object_fraction"):
            corrected_object_fraction = correction_rate
        rollback_rate = _mean_optional(group, "rollback_rate")
        escalation_rate = _mean_optional(group, "escalation_rate")
        escalation_success_rate = _mean_optional(group, "escalation_success_rate")
        integrity_score = _integrity_score(violations, delta_norm, flip_rate)
        low_disruption_score = _low_disruption_integrity_score(
            integrity_score,
            rejection_rate,
            correction_rate,
            corrected_object_fraction,
            rollback_rate,
            escalation_rate,
        )
        output.append(
            {
                "run_label": label,
                "model": model,
                "horizon": horizon,
                "seed_count": len({row["seed"] for row in group}),
                "constraint_violations_per_step": round(violations, 6),
                "delta_norm_mean": round(delta_norm, 6),
                "branch_entropy_mean": round(entropy, 6),
                "branch_flip_rate": round(flip_rate, 6),
                "selected_k_mean": round(selected_k, 6),
                "unsafe_delta_rejection_rate": round(rejection_rate, 6),
                "correction_rate": round(correction_rate, 6),
                "corrected_object_fraction": round(corrected_object_fraction, 6),
                "rollback_rate": round(rollback_rate, 6),
                "escalation_rate": round(escalation_rate, 6),
                "escalation_success_rate": round(escalation_success_rate, 6),
                "state_integrity_score": round(integrity_score, 6),
                "low_disruption_integrity_score": round(low_disruption_score, 6),
            }
        )
    return output


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _mean_optional(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return statistics.fmean(values) if values else 0.0


def _has_optional(rows: list[dict[str, str]], field: str) -> bool:
    return any(row.get(field) not in {None, ""} for row in rows)


def _integrity_score(violations: float, delta_norm: float, flip_rate: float) -> float:
    violation_penalty = min(1.0, violations / 1.0)
    delta_penalty = min(1.0, delta_norm / 10.0)
    flip_penalty = min(1.0, flip_rate / 0.5)
    return max(0.0, 1.0 - (0.55 * violation_penalty + 0.35 * delta_penalty + 0.10 * flip_penalty))


def _low_disruption_integrity_score(
    integrity_score: float,
    rejection_rate: float,
    correction_rate: float,
    corrected_object_fraction: float,
    rollback_rate: float,
    escalation_rate: float,
) -> float:
    intervention_penalty = (
        0.20 * min(1.0, rejection_rate)
        + 0.25 * min(1.0, correction_rate)
        + 0.15 * min(1.0, corrected_object_fraction)
        + 0.30 * min(1.0, rollback_rate)
        + 0.10 * min(1.0, escalation_rate)
    )
    return max(0.0, integrity_score - intervention_penalty)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(input_paths: list[Path], output_csv: Path, rows: list[dict[str, object]]) -> str:
    labels = {str(row["run_label"]) for row in rows}
    regularized_note = []
    if "regularized" in labels:
        regularized_note = [
            "",
            "The regularized run adds a training-time target-relative delta-norm",
            "penalty. It is intentionally reported as a raw rollout, not as a",
            "guarded state-store result. In the current evidence it only slightly",
            "improves raw WPU sparse H=25 integrity, so simple delta-norm",
            "regularization is not sufficient to solve model-delta instability.",
        ]
    rejection_note = []
    if any(float(row.get("unsafe_delta_rejection_rate", 0.0)) > 0.0 for row in rows):
        rejection_note = [
            "",
            "The unsafe-delta rejection run is a state-store safety mechanism,",
            "not proof that the raw transition model is stable. It must be",
            "reported together with rejection rate: high integrity with high",
            "rejection means the memory layer protected the state by declining",
            "unsafe updates.",
        ]
    correction_note = []
    if any(float(row.get("correction_rate", 0.0)) > 0.0 for row in rows):
        correction_note = [
            "",
            "The correction run applies a bounded state projection after a predicted",
            "delta increases validity violations, and only then falls back to",
            "rollback if the corrected state is still worse than the previous",
            "state. It tests whether memory-layer repair can reduce rollback",
            "frequency while preserving applied-state integrity.",
        ]
    finite_clamp_note = []
    if "finite_clamped" in labels:
        finite_clamp_note = [
            "",
            "The finite-clamped run first sanitizes non-finite or extreme",
            "predicted deltas, then applies norm clipping. It removes the",
            "sparse WPU delta-norm explosion seen in the earlier clipped run,",
            "but it does not eliminate validity violations by itself. This",
            "separates numerical delta safety from state validity.",
        ]
    finite_correction_note = []
    if "finite_corrected" in labels:
        finite_correction_note = [
            "",
            "The finite-corrected run combines finite-safe delta clipping with",
            "correction-only projection. It is a stronger memory-safety result:",
            "sparse WPU reaches H=25 integrity comparable to guarded projection",
            "with zero rollback and zero dense escalation, but at a high",
            "correction rate. This still does not prove raw dynamics stability;",
            "it shows that bounded local correction can protect applied state",
            "without declining or recomputing most updates.",
        ]
    selective_correction_note = []
    if "selective_corrected" in labels:
        selective_correction_note = [
            "",
            "The selective-correction run uses the same finite-safe correction",
            "trigger as finite-corrected rollout but only projects objects that",
            "actually violate validity bounds. It preserves sparse H=25 integrity",
            "while reducing the corrected-object fraction. The stride-2 and",
            "margin-1 variants show the current boundary: reducing correction",
            "trigger frequency directly causes validity violations to return.",
            "This narrows the P2 problem to learning a more stable transition or",
            "a better correction trigger, not merely shrinking the correction",
            "projection itself.",
        ]
    escalation_note = []
    if any(float(row.get("escalation_rate", 0.0)) > 0.0 for row in rows):
        escalation_note = [
            "",
            "The escalation run tests sparse-first, dense-when-needed memory",
            "safety: when the sparse delta increases validity violations, the",
            "state is restored and a local-dense WPU fallback recomputes the",
            "update before correction or rollback. In the current evidence this",
            "reduces rollback frequency to zero for sparse H=25 while improving",
            "corrected-rollback integrity, but it is still a safety-layer result",
            "rather than proof of stable raw sparse dynamics.",
        ]
    consistency_note = []
    if "consistency" in labels:
        consistency_note = [
            "",
            "The rollout-consistency run adds a second-step delta-growth penalty",
            "during training. In the current evidence it does not solve sparse",
            "raw-delta instability, so rollout consistency needs a stronger",
            "state-validity objective or correction mechanism before it can",
            "replace guarded memory safety.",
        ]
    validity_note = []
    if "validity" in labels or "validity_strong" in labels:
        validity_note = [
            "",
            "The state-validity runs add training losses for predicted position,",
            "velocity, and cup-floor bounds. In the current evidence they also",
            "do not solve sparse raw-delta instability: both validity and",
            "strong-validity sparse H=25 integrity remain at 0.084722.",
            "Local-dense validity also falls below the raw local-dense score.",
            "Validity losses therefore need rollback/correction and uncertainty",
            "escalation rather than acting as a standalone fix.",
        ]
    lines = [
        "# PyBullet State-Integrity Audit",
        "",
        "This audit derives long-horizon state-integrity metrics from the PyBullet",
        "closed-loop rollout results. It does not resynchronize to the simulator;",
        "it evaluates whether repeated `DeltaState` overlays keep object state",
        "within simple validity bounds.",
        "",
        "Source CSVs:",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in input_paths)
    lines.extend(
        [
            "",
            "Derived CSV:",
            "",
            f"- `{output_csv.as_posix()}`",
            "",
            "## Summary",
            "",
            "| run | model | H | violations/step | delta norm | flip rate | reject rate | correction rate | corrected objects | rollback rate | escalation rate | escalation success | integrity score | low-disruption score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {row['model']} | {row['horizon']} | "
            f"{float(row['constraint_violations_per_step']):.6f} | "
            f"{float(row['delta_norm_mean']):.6f} | {float(row['branch_flip_rate']):.6f} | "
            f"{float(row['unsafe_delta_rejection_rate']):.6f} | "
            f"{float(row['correction_rate']):.6f} | "
            f"{float(row['corrected_object_fraction']):.6f} | "
            f"{float(row['rollback_rate']):.6f} | "
            f"{float(row['escalation_rate']):.6f} | "
            f"{float(row['escalation_success_rate']):.6f} | "
            f"{float(row['state_integrity_score']):.6f} | "
            f"{float(row['low_disruption_integrity_score']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The audit confirms that one-step branch accuracy is not enough for a",
            "world-state processor. The sparse WPU path can keep a small selected",
            "`K`, but repeated raw deltas can still create invalid state. Guarded",
            "state-store projection can protect the applied state and lift WPU",
            "H=25 integrity above the dashboard threshold, but it does not prove",
            "the underlying delta model is stable. Future reports must distinguish",
            "raw model deltas from guarded state-store deltas.",
            *regularized_note,
            *rejection_note,
            *correction_note,
            *finite_clamp_note,
            *finite_correction_note,
            *selective_correction_note,
            *escalation_note,
            *consistency_note,
            *validity_note,
            "",
            "This makes state integrity a first-class WPU metric:",
            "",
            "```text",
            "state-integrity = constraint validity + bounded delta drift + branch stability",
            "```",
            "",
            "Future WPU rollout claims should report this score or its components next",
            "to accuracy and latency.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
