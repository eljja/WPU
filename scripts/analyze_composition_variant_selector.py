from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


COMPOSITION_POLICIES = (
    "composition_regret_argmax",
    "composition_regret_expected",
    "composition_regret_count_only",
)
BASELINE_POLICY = "static_learned_interaction"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out-safe selection among composition-regret policies.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/experiments/wpu_v2_retriever_cross_seed_composition_regret.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_composition_variant_selector_summary.csv"),
    )
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = _summarize(rows)
    _write_csv(args.out, summary)
    print(f"wrote={args.out}", flush=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key = {
        (int(row["causal_k"]), int(row["seed"]), row["policy"]): row
        for row in rows
    }
    seeds_by_k: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        seeds_by_k[int(row["causal_k"])].add(int(row["seed"]))

    output: list[dict[str, object]] = []
    for criterion in ("lowest_other_seed_loss", "highest_other_seed_accuracy", "lowest_other_seed_excess"):
        for causal_k in sorted(seeds_by_k):
            selected_rows = []
            selected_policies = []
            for seed in sorted(seeds_by_k[causal_k]):
                other_seeds = [other for other in sorted(seeds_by_k[causal_k]) if other != seed]
                candidates = []
                for policy in COMPOSITION_POLICIES:
                    other_rows = [by_key[(causal_k, other, policy)] for other in other_seeds]
                    if criterion == "lowest_other_seed_loss":
                        key = (mean(float(row["loss"]) for row in other_rows), policy)
                    elif criterion == "highest_other_seed_accuracy":
                        key = (-mean(float(row["accuracy"]) for row in other_rows), policy)
                    else:
                        key = (mean(float(row["excess_over_generated_oracle"]) for row in other_rows), policy)
                    candidates.append((key, policy))
                _, selected_policy = min(candidates, key=lambda item: item[0])
                selected_rows.append(by_key[(causal_k, seed, selected_policy)])
                selected_policies.append(selected_policy)
            baseline_rows = [by_key[(causal_k, seed, BASELINE_POLICY)] for seed in sorted(seeds_by_k[causal_k])]
            output.append(
                {
                    "criterion": criterion,
                    "causal_k": causal_k,
                    "n": len(selected_rows),
                    "loss": round(mean(float(row["loss"]) for row in selected_rows), 6),
                    "accuracy": round(mean(float(row["accuracy"]) for row in selected_rows), 6),
                    "excess_over_generated_oracle": round(
                        mean(float(row["excess_over_generated_oracle"]) for row in selected_rows),
                        6,
                    ),
                    "delta_loss_vs_static_learned": round(
                        mean(float(row["loss"]) for row in selected_rows)
                        - mean(float(row["loss"]) for row in baseline_rows),
                        6,
                    ),
                    "selected_obstacles": round(mean(float(row["selected_obstacles"]) for row in selected_rows), 6),
                    "selected_hand_rate": round(mean(float(row["selected_hand_rate"]) for row in selected_rows), 6),
                    "selected_policies": ";".join(selected_policies),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
