from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUT = Path("docs/experiments/wpu_v2_joint_candidate_generator.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/wpu_v2_joint_candidate_generator_summary.csv")
DEFAULT_OUT_MD = Path("docs/experiments/wpu_v2_joint_candidate_generator_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/wpu_v2_joint_candidate_generator_results.ko.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the P1 joint candidate-generator probe without treating oracle rows as deployed policies."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = _summarize(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render(summary, args.input, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(summary, args.input, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for causal_k in sorted({int(row["causal_k"]) for row in rows}):
        group = [row for row in rows if int(row["causal_k"]) == causal_k]
        by_policy = {policy: [row for row in group if row["policy"] == policy] for policy in {row["policy"] for row in group}}
        required = {
            "static_learned_interaction",
            "generated_plus_composition_oracle",
            "learned_generated_oracle",
            "joint_candidate_generator_evaluator",
        }
        missing = sorted(required - set(by_policy))
        if missing:
            raise ValueError(f"missing policies for K={causal_k}: {missing}")
        static = _policy_stats(by_policy["static_learned_interaction"])
        full_oracle = _policy_stats(by_policy["generated_plus_composition_oracle"])
        learned_oracle = _policy_stats(by_policy["learned_generated_oracle"])
        evaluator = _policy_stats(by_policy["joint_candidate_generator_evaluator"])
        full_gain = static["loss"] - full_oracle["loss"]
        learned_gain = static["loss"] - learned_oracle["loss"]
        evaluator_gain = static["loss"] - evaluator["loss"]
        out.append(
            {
                "total_objects_n": int(float(group[0]["total_objects_n"])),
                "causal_k": causal_k,
                "seed_count": len({int(row["seed"]) for row in group}),
                "static_loss": round(static["loss"], 6),
                "static_accuracy": round(static["accuracy"], 6),
                "full_candidate_oracle_loss": round(full_oracle["loss"], 6),
                "full_candidate_oracle_accuracy": round(full_oracle["accuracy"], 6),
                "full_candidate_oracle_gain": round(full_gain, 6),
                "learned_generator_oracle_loss": round(learned_oracle["loss"], 6),
                "learned_generator_oracle_accuracy": round(learned_oracle["accuracy"], 6),
                "learned_generator_oracle_gain": round(learned_gain, 6),
                "learned_generator_oracle_closure": round(_safe_div(learned_gain, full_gain), 6),
                "learned_generator_oracle_match": round(learned_oracle["oracle_match_rate"], 6),
                "learned_generator_oracle_selected_rate": round(learned_oracle["selected_learned_generated_rate"], 6),
                "evaluator_loss": round(evaluator["loss"], 6),
                "evaluator_accuracy": round(evaluator["accuracy"], 6),
                "evaluator_gain": round(evaluator_gain, 6),
                "evaluator_gap_closure": round(_safe_div(evaluator_gain, full_gain), 6),
                "evaluator_oracle_match": round(evaluator["oracle_match_rate"], 6),
                "evaluator_selected_learned_generated_rate": round(evaluator["selected_learned_generated_rate"], 6),
                "verdict": _verdict(_safe_div(evaluator_gain, full_gain), _safe_div(learned_gain, full_gain)),
            }
        )
    return out


def _policy_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "loss": statistics.fmean(float(row["loss"]) for row in rows),
        "accuracy": statistics.fmean(float(row["accuracy"]) for row in rows),
        "oracle_match_rate": statistics.fmean(float(row.get("oracle_match_rate", 0.0)) for row in rows),
        "selected_learned_generated_rate": statistics.fmean(
            float(row.get("selected_learned_generated_rate", 0.0)) for row in rows
        ),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 1e-12 else 0.0


def _verdict(evaluator_closure: float, learned_oracle_closure: float) -> str:
    if evaluator_closure >= 0.5:
        return "passes_p1_candidate_generator_gate"
    if learned_oracle_closure > evaluator_closure + 0.1:
        return "generator_headroom_not_deployable"
    if learned_oracle_closure <= 0.1:
        return "generator_does_not_expand_useful_headroom"
    return "partial_but_insufficient"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], source: Path, *, korean: bool) -> str:
    best_evaluator = max(rows, key=lambda row: float(row["evaluator_gap_closure"]))
    best_oracle = max(rows, key=lambda row: float(row["learned_generator_oracle_closure"]))
    if korean:
        lines = [
            "# Joint Candidate Generator 결과",
            "",
            "이 문서는 P1의 다음 단계인 downstream-regret 기반 learned candidate generator를 요약한다. "
            "중요한 구분은 learned-generated oracle headroom과 실제 deployed evaluator 성능을 분리하는 것이다.",
            "",
            f"Source CSV: `{source.as_posix()}`",
            "",
            f"최고 learned-generator oracle closure는 `{float(best_oracle['learned_generator_oracle_closure']):.6f}` "
            f"(`K={best_oracle['causal_k']}`)다. 하지만 최고 deployed evaluator closure는 "
            f"`{float(best_evaluator['evaluator_gap_closure']):.6f}` (`K={best_evaluator['causal_k']}`)에 그친다.",
            "",
            "| K | Static loss | Full oracle loss | Learned-generator oracle closure | Evaluator closure | Evaluator accuracy | Learned-generated selected rate | Verdict |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['causal_k']} | {float(row['static_loss']):.6f} | "
                f"{float(row['full_candidate_oracle_loss']):.6f} | "
                f"{float(row['learned_generator_oracle_closure']):.6f} | "
                f"{float(row['evaluator_gap_closure']):.6f} | "
                f"{float(row['evaluator_accuracy']):.6f} | "
                f"{float(row['evaluator_selected_learned_generated_rate']):.6f} | `{row['verdict']}` |"
            )
        lines.extend(
            [
                "",
                "## 해석",
                "",
                "- Learned generator는 후보 pool의 oracle headroom을 일부 만든다.",
                "- 하지만 evaluator가 그 후보를 held-out seed에서 안전하게 선택하지 못한다.",
                "- 따라서 P1 병목은 후보 생성 단독이 아니라 후보 생성, 선택, propagation verification을 함께 학습해야 하는 문제다.",
            ]
        )
    else:
        lines = [
            "# Joint Candidate Generator Results",
            "",
            "This report summarizes a P1 probe that trains a learned candidate generator from downstream-regret object membership. "
            "It separates learned-generated oracle headroom from deployed evaluator performance.",
            "",
            f"Source CSV: `{source.as_posix()}`",
            "",
            f"The best learned-generator oracle closure is `{float(best_oracle['learned_generator_oracle_closure']):.6f}` "
            f"(`K={best_oracle['causal_k']}`). The best deployed evaluator closure is only "
            f"`{float(best_evaluator['evaluator_gap_closure']):.6f}` (`K={best_evaluator['causal_k']}`).",
            "",
            "| K | Static loss | Full oracle loss | Learned-generator oracle closure | Evaluator closure | Evaluator accuracy | Learned-generated selected rate | Verdict |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['causal_k']} | {float(row['static_loss']):.6f} | "
                f"{float(row['full_candidate_oracle_loss']):.6f} | "
                f"{float(row['learned_generator_oracle_closure']):.6f} | "
                f"{float(row['evaluator_gap_closure']):.6f} | "
                f"{float(row['evaluator_accuracy']):.6f} | "
                f"{float(row['evaluator_selected_learned_generated_rate']):.6f} | `{row['verdict']}` |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- The learned generator creates some candidate-pool oracle headroom.",
                "- The evaluator does not reliably deploy that headroom on held-out seeds.",
                "- P1 is therefore not solved by candidate generation alone; candidate generation, selection, and propagation verification need to be learned together.",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
