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
        integrity_score = _integrity_score(violations, delta_norm, flip_rate)
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
                "state_integrity_score": round(integrity_score, 6),
            }
        )
    return output


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _mean_optional(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return statistics.fmean(values) if values else 0.0


def _integrity_score(violations: float, delta_norm: float, flip_rate: float) -> float:
    violation_penalty = min(1.0, violations / 1.0)
    delta_penalty = min(1.0, delta_norm / 10.0)
    flip_penalty = min(1.0, flip_rate / 0.5)
    return max(0.0, 1.0 - (0.55 * violation_penalty + 0.35 * delta_penalty + 0.10 * flip_penalty))


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
            "| run | model | H | violations/step | delta norm | flip rate | reject rate | integrity score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {row['model']} | {row['horizon']} | "
            f"{float(row['constraint_violations_per_step']):.6f} | "
            f"{float(row['delta_norm_mean']):.6f} | {float(row['branch_flip_rate']):.6f} | "
            f"{float(row['unsafe_delta_rejection_rate']):.6f} | "
            f"{float(row['state_integrity_score']):.6f} |"
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
