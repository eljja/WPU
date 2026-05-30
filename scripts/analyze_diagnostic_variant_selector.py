from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


POLICY = "diagnostic_cross_seed_generated_reranker"
BASELINE_POLICY = "diagnostic_cross_seed_static_base_choice"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze deployment-safe context-variant selection for diagnostic rerankers.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/experiments/wpu_v2_retriever_cross_seed_diagnostic_reranker.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_diagnostic_variant_selector_summary.csv"),
    )
    args = parser.parse_args()

    rows = _read_rows(args.input)
    output = _summary_rows(rows)
    _write_csv(args.out, output)
    print(f"wrote={args.out}", flush=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_condition: dict[tuple[int, int, str, str], dict[str, str]] = {
        (int(row["causal_k"]), int(row["seed"]), row["context_variant"], row["policy"]): row
        for row in rows
    }
    seeds_by_k: dict[int, set[int]] = defaultdict(set)
    variants_by_k: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        seeds_by_k[int(row["causal_k"])].add(int(row["seed"]))
        variants_by_k[int(row["causal_k"])].add(row["context_variant"])

    output: list[dict[str, object]] = []
    for criterion in ("min_cv_delta", "max_cv_win_then_delta", "best_train_loss_delta"):
        for causal_k in sorted(seeds_by_k):
            selected_rows = []
            selected_variants = []
            for seed in sorted(seeds_by_k[causal_k]):
                candidates = []
                for variant in sorted(variants_by_k[causal_k]):
                    row = by_condition[(causal_k, seed, variant, POLICY)]
                    cv_delta = float(row["cv_mean_delta"])
                    cv_win = float(row["cv_win_rate"])
                    train_delta = float(row["train_reranker_loss"]) - float(row["train_static_loss"])
                    if criterion == "min_cv_delta":
                        key = (cv_delta, -cv_win, variant)
                    elif criterion == "max_cv_win_then_delta":
                        key = (-cv_win, cv_delta, variant)
                    else:
                        key = (train_delta, cv_delta, variant)
                    candidates.append((key, variant, row))
                _, variant, selected = min(candidates, key=lambda item: item[0])
                selected_rows.append(selected)
                selected_variants.append(variant)

            static_rows = [by_condition[(causal_k, seed, sorted(variants_by_k[causal_k])[0], BASELINE_POLICY)] for seed in seeds_by_k[causal_k]]
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
                    "delta_loss_vs_static_base": round(
                        mean(float(row["loss"]) for row in selected_rows)
                        - mean(float(row["loss"]) for row in static_rows),
                        6,
                    ),
                    "selected_variants": ";".join(selected_variants),
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
