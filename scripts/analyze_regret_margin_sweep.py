from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    "routed_accuracy",
    "routed_loss",
    "sparse_loss",
    "routed_delta_vs_sparse",
    "routed_excess_over_oracle",
    "dense_compute_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WPU regret-margin policies with leave-one-seed-out selection.")
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/wpu_v2_staged_regret_margin_sweep.csv"))
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_staged_regret_margin_policy_rows.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_staged_regret_margin_policy_summary.csv"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_staged_regret_margin_policy_results.md"),
    )
    args = parser.parse_args()

    rows = _read_rows(args.input)
    policy_rows = _build_policy_rows(rows)
    summary_rows = _summarize(policy_rows)
    _write_csv(args.rows_out, policy_rows)
    _write_csv(args.summary_out, summary_rows)
    _write_markdown(args.markdown_out, args.input, summary_rows)
    print(f"wrote={args.rows_out}")
    print(f"wrote={args.summary_out}")
    print(f"wrote={args.markdown_out}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_policy_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    margins = [row for row in rows if row["policy"].startswith("margin_")]
    calibrated = [row for row in rows if row["policy"] == "calibrated"]

    for row in calibrated:
        output.append(_copy_policy_row(row, "validation_calibrated", row["policy"], "per_seed_validation"))

    fixed_global_policy = _select_policy(margins)
    for row in margins:
        if row["policy"] == fixed_global_policy:
            output.append(_copy_policy_row(row, "fixed_global_margin", fixed_global_policy, "all_seed_retrospective"))

    seeds = sorted({int(row["seed"]) for row in rows})
    causal_ks = sorted({int(row["causal_k"]) for row in rows})
    by_key = {(int(row["causal_k"]), int(row["seed"]), row["policy"]): row for row in margins}
    for heldout_seed in seeds:
        train_rows = [row for row in margins if int(row["seed"]) != heldout_seed]
        global_policy = _select_policy(train_rows)
        for causal_k in causal_ks:
            output.append(
                _copy_policy_row(
                    by_key[(causal_k, heldout_seed, global_policy)],
                    "loso_global_margin",
                    global_policy,
                    f"heldout_seed_{heldout_seed}",
                )
            )
            train_k_rows = [row for row in train_rows if int(row["causal_k"]) == causal_k]
            k_policy = _select_policy(train_k_rows)
            output.append(
                _copy_policy_row(
                    by_key[(causal_k, heldout_seed, k_policy)],
                    "loso_k_conditioned_margin",
                    k_policy,
                    f"heldout_seed_{heldout_seed}",
                )
            )
    return output


def _select_policy(rows: list[dict[str, str]]) -> str:
    by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    candidates = []
    for policy, group in by_policy.items():
        candidates.append(
            (
                mean(float(row["routed_loss"]) for row in group),
                mean(float(row["dense_compute_ratio"]) for row in group),
                policy,
            )
        )
    candidates.sort()
    return candidates[0][2]


def _copy_policy_row(row: dict[str, str], eval_policy: str, selected_policy: str, selection_scope: str) -> dict[str, object]:
    copied: dict[str, object] = {
        "eval_policy": eval_policy,
        "selected_policy": selected_policy,
        "selection_scope": selection_scope,
        "causal_k": int(row["causal_k"]),
        "seed": int(row["seed"]),
    }
    for field in [
        "total_objects_n",
        "samples",
        "compute_cost",
        "route_threshold",
        "sparse_accuracy",
        "dense_accuracy",
        "routed_accuracy",
        "sparse_loss",
        "dense_loss",
        "routed_loss",
        "oracle_loss",
        "routed_delta_vs_sparse",
        "routed_excess_over_oracle",
        "dense_compute_ratio",
        "oracle_dense_compute_ratio",
        "route_regret_eval_corr",
    ]:
        copied[field] = float(row[field])
    return copied


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["eval_policy"]), "all")].append(row)
        grouped[(str(row["eval_policy"]), f"K={row['causal_k']}")].append(row)
    summary: list[dict[str, object]] = []
    for (policy, group_name), group in sorted(grouped.items()):
        item: dict[str, object] = {"eval_policy": policy, "group": group_name, "rows": len(group)}
        selected = sorted({str(row["selected_policy"]) for row in group})
        item["selected_policies"] = ";".join(selected)
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            metric_mean = mean(values)
            metric_ci = _ci95(values)
            item[f"{metric}_mean"] = round(metric_mean, 6)
            item[f"{metric}_ci95"] = round(metric_ci, 6)
        summary.append(item)
    return summary


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, source: Path, summary_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for row in summary_rows if row["group"] == "all"]
    k_rows = [row for row in summary_rows if row["group"] != "all"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# WPU V2 Regret-Margin Policy Selection\n\n")
        handle.write(f"Source CSV: `{source.as_posix()}`\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "This analysis tests whether the K-dependence observed in the fixed-margin sweep can be used as a "
            "more stable scheduler policy. Margin selection is evaluated with leave-one-seed-out selection: "
            "margins are chosen on four seeds and evaluated on the held-out seed.\n\n"
        )
        handle.write("## Overall Results\n\n")
        handle.write(
            "| policy | selected policies | routed loss | sparse loss | loss delta | dense compute | routed acc | oracle excess |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in all_rows:
            handle.write(_summary_table_line(row))
        handle.write("\n## K-Specific Results\n\n")
        handle.write(
            "| policy | group | selected policies | routed loss | loss delta | dense compute | oracle excess |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in k_rows:
            handle.write(
                f"| {row['eval_policy']} | {row['group']} | {row['selected_policies']} | "
                f"{row['routed_loss_mean']:.3f} +/- {row['routed_loss_ci95']:.3f} | "
                f"{row['routed_delta_vs_sparse_mean']:.3f} +/- {row['routed_delta_vs_sparse_ci95']:.3f} | "
                f"{row['dense_compute_ratio_mean']:.3f} +/- {row['dense_compute_ratio_ci95']:.3f} | "
                f"{row['routed_excess_over_oracle_mean']:.3f} +/- {row['routed_excess_over_oracle_ci95']:.3f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Leave-one-seed-out K-conditioned margin selection is a stricter test than post-hoc K-specific best "
            "margin selection. If it improves over a fixed global margin, the result supports a regime-conditioned "
            "scheduler. If it does not, the margin must be learned from richer state evidence rather than K alone.\n"
        )
        handle.write(
            "\nIn the current sweep, K-only margin selection does not improve over the fixed global margin overall. "
            "This is a useful negative result: K is a regime descriptor, not a sufficient scheduler state. The next "
            "scheduler should condition margin on K plus selector confidence, interaction density, regret "
            "uncertainty, sparse entropy, rollout drift, and compute budget.\n"
        )


def _summary_table_line(row: dict[str, object]) -> str:
    return (
        f"| {row['eval_policy']} | {row['selected_policies']} | "
        f"{row['routed_loss_mean']:.3f} +/- {row['routed_loss_ci95']:.3f} | "
        f"{row['sparse_loss_mean']:.3f} +/- {row['sparse_loss_ci95']:.3f} | "
        f"{row['routed_delta_vs_sparse_mean']:.3f} +/- {row['routed_delta_vs_sparse_ci95']:.3f} | "
        f"{row['dense_compute_ratio_mean']:.3f} +/- {row['dense_compute_ratio_ci95']:.3f} | "
        f"{row['routed_accuracy_mean']:.3f} +/- {row['routed_accuracy_ci95']:.3f} | "
        f"{row['routed_excess_over_oracle_mean']:.3f} +/- {row['routed_excess_over_oracle_ci95']:.3f} |\n"
    )


if __name__ == "__main__":
    main()
