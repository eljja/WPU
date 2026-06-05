from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


DEPLOYED_POLICIES = (
    "static_learned_interaction",
    "invariant_set_scorer",
    "risk_adjusted_selected_mechanism",
    "generated_plus_composition_oracle",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze WPU v2 candidate-oracle gap closure from retriever policy CSVs."
    )
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/wpu_v2_candidate_oracle_gap_v2.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/wpu_v2_candidate_oracle_gap_v2_results.md"))
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = _summarize(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render_markdown(args.input, args.out_csv, summary), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        policy = row.get("policy", "")
        if policy not in DEPLOYED_POLICIES:
            continue
        grouped[(row.get("feature_variant", "all"), int(row["causal_k"]))].append(row)

    output: list[dict[str, object]] = []
    for (variant, causal_k), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_policy[row["policy"]].append(row)
        if "static_learned_interaction" not in by_policy or "generated_plus_composition_oracle" not in by_policy:
            continue
        static = _policy_stats(by_policy["static_learned_interaction"])
        oracle = _policy_stats(by_policy["generated_plus_composition_oracle"])
        oracle_gain = max(0.0, static["loss"] - oracle["loss"])
        for policy in DEPLOYED_POLICIES:
            if policy not in by_policy:
                continue
            stats = _policy_stats(by_policy[policy])
            deployed_gain = static["loss"] - stats["loss"]
            closure = deployed_gain / oracle_gain if oracle_gain > 1e-12 else 0.0
            output.append(
                {
                    "feature_variant": variant,
                    "causal_k": causal_k,
                    "policy": policy,
                    "loss": round(stats["loss"], 6),
                    "accuracy": round(stats["accuracy"], 6),
                    "oracle_match_rate": round(stats["oracle_match_rate"], 6),
                    "candidate_oracle_loss": round(oracle["loss"], 6),
                    "candidate_oracle_gain_over_static": round(oracle_gain, 6),
                    "deployed_gain_over_static": round(deployed_gain, 6),
                    "remaining_gap": round(max(0.0, stats["loss"] - oracle["loss"]), 6),
                    "gap_closure_fraction": round(closure, 6),
                    "seed_count": len({row["heldout_seed"] for row in by_policy[policy]}),
                }
            )
    return output


def _policy_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "loss": statistics.fmean(float(row["loss"]) for row in rows),
        "accuracy": statistics.fmean(float(row["accuracy"]) for row in rows),
        "oracle_match_rate": statistics.fmean(float(row["oracle_match_rate"]) for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(input_csv: Path, output_csv: Path, rows: list[dict[str, object]]) -> str:
    risk_rows = [
        row
        for row in rows
        if row["feature_variant"] == "role_geometry_family"
        and row["policy"] in {"static_learned_interaction", "risk_adjusted_selected_mechanism", "generated_plus_composition_oracle"}
    ]
    lines = [
        "# WPU v2 Candidate-Oracle Gap Audit",
        "",
        "This audit recomputes the current candidate-oracle gap from the latest",
        "cross-seed invariant-scorer experiment. It measures how much of the",
        "available candidate-pool gain is recovered by deployed policy selection.",
        "",
        "Source CSV:",
        "",
        f"- `{input_csv.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{output_csv.as_posix()}`",
        "",
        "## Key Table",
        "",
        "| K | policy | loss | accuracy | candidate-oracle gain | deployed gain | remaining gap | gap closure | oracle match |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(risk_rows, key=lambda item: (int(item["causal_k"]), str(item["policy"]))):
        policy = str(row["policy"]).replace("_", " ")
        lines.append(
            f"| {row['causal_k']} | {policy} | {float(row['loss']):.6f} | "
            f"{float(row['accuracy']):.6f} | {float(row['candidate_oracle_gain_over_static']):.6f} | "
            f"{float(row['deployed_gain_over_static']):.6f} | {float(row['remaining_gap']):.6f} | "
            f"{float(row['gap_closure_fraction']):.6f} | {float(row['oracle_match_rate']):.6f} |"
        )
    best_closure = max(
        (
            float(row["gap_closure_fraction"])
            for row in rows
            if row["policy"] == "risk_adjusted_selected_mechanism"
            and row["feature_variant"] == "role_geometry_family"
        ),
        default=0.0,
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The candidate pool still contains substantially better working sets than",
            "the deployed selector usually chooses. Risk-adjusted mechanism routing",
            "recovers part of the oracle gain at `K=8` and `K=16`, but only a small",
            "fraction at `K=32`. The best current closure fraction is "
            f"`{best_closure:.6f}`.",
            "",
            "This means priority 1 is not solved. The current positive result is",
            "narrower: explicit state descriptors and risk-adjusted mechanism routing",
            "reduce the candidate-oracle gap without returning to token processing,",
            "but candidate scoring still leaves most oracle headroom unused.",
            "",
            "## Next Technical Target",
            "",
            "- Train candidate scoring from downstream regret with cross-seed",
            "  perturbations rather than only candidate descriptors.",
            "- Add uncertainty on the selector itself, so low-confidence selection can",
            "  expand `K` or invoke verification rather than choosing a bad candidate.",
            "- Report gap-closure fraction as a required metric for all future",
            "  working-set-control experiments.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
