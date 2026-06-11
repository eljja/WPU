from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUT = Path("docs/experiments/wpu_v2_candidate_regret_gate.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/wpu_v2_candidate_regret_gate_summary.csv")
DEFAULT_OUT_MD = Path("docs/experiments/wpu_v2_candidate_regret_gate_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/wpu_v2_candidate_regret_gate_results.ko.md")


NON_DEPLOYED_POLICIES = {"static_learned_interaction", "generated_plus_composition_oracle"}
MAX_MARKDOWN_ROWS = 18


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize candidate-regret gate closure for WPU v2 priority-1 tracking."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = _summarize(rows, args.input)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render_markdown(summary, args.input, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(summary, args.input, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]], source: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for causal_k in sorted({int(row["causal_k"]) for row in rows}):
        group = [row for row in rows if int(row["causal_k"]) == causal_k]
        policies = sorted({row["policy"] for row in group})
        losses = {
            policy: statistics.fmean(float(row["loss"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        accuracies = {
            policy: statistics.fmean(float(row["accuracy"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        matches = {
            policy: statistics.fmean(float(row["oracle_match_rate"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        static_loss = losses["static_learned_interaction"]
        oracle_loss = losses["generated_plus_composition_oracle"]
        oracle_gap = static_loss - oracle_loss
        deployed_policies = [policy for policy in policies if policy not in NON_DEPLOYED_POLICIES]
        for policy in deployed_policies:
            policy_rows = [row for row in group if row["policy"] == policy]
            deployed_gain = static_loss - losses[policy]
            gap_closure = deployed_gain / oracle_gap if oracle_gap > 0 else 0.0
            out.append(
                {
                    "source": source.as_posix(),
                    "total_objects_n": int(float(policy_rows[0]["total_objects_n"])),
                    "causal_k": causal_k,
                    "policy": policy,
                    "static_loss": round(static_loss, 6),
                    "candidate_oracle_loss": round(oracle_loss, 6),
                    "candidate_oracle_gain_over_static": round(oracle_gap, 6),
                    "policy_loss": round(losses[policy], 6),
                    "policy_accuracy": round(accuracies[policy], 6),
                    "oracle_match_rate": round(matches[policy], 6),
                    "deployed_gain_over_static": round(deployed_gain, 6),
                    "gap_closure_fraction": round(gap_closure, 6),
                    "remaining_gap": round(losses[policy] - oracle_loss, 6),
                    "mean_accept_rate": _mean_optional(policy_rows, "accept_rate"),
                    "mean_harmful_accept_rate": _mean_optional(policy_rows, "harmful_accept_rate"),
                    "mean_regret_corr": _mean_optional(policy_rows, "regret_corr"),
                    "mean_predicted_sigma": _mean_optional(policy_rows, "predicted_sigma_mean"),
                    "mean_selection_train_gap_closure": _mean_optional(policy_rows, "selection_train_gap_closure"),
                    "mean_selection_train_harmful_accept_rate": _mean_optional(policy_rows, "selection_train_harmful_accept_rate"),
                    "seed_count": len({int(row["seed"]) for row in policy_rows}),
                    "failure_mode": _failure_mode(gap_closure, _mean_optional(policy_rows, "harmful_accept_rate")),
                }
            )
    return out


def _mean_optional(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return round(statistics.fmean(values), 6) if values else 0.0


def _failure_mode(gap_closure: float, harmful_accept_rate: float) -> str:
    if gap_closure < 0.0:
        return "harmful_transfer"
    if harmful_accept_rate > 0.25:
        return "insufficient_no_harm_rejection"
    if gap_closure < 0.5:
        return "partial_but_insufficient_gap_closure"
    return "passes_current_p1_threshold"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], source: Path, *, korean: bool) -> str:
    is_safety_gate = "candidate_safety_gate" in source.name
    is_invariant_gate = "candidate_invariant_gate" in source.name
    best = max(rows, key=lambda row: float(row["gap_closure_fraction"]))
    safe_rows = [row for row in rows if float(row["mean_harmful_accept_rate"]) <= 0.25]
    safe_best = max(safe_rows, key=lambda row: float(row["gap_closure_fraction"])) if safe_rows else None
    train_selected_rows = [row for row in rows if str(row["policy"]).startswith("train_selected_")]
    train_selected_best = (
        max(train_selected_rows, key=lambda row: float(row["gap_closure_fraction"]))
        if train_selected_rows
        else None
    )
    best_by_k = []
    for causal_k in sorted({int(row["causal_k"]) for row in rows}):
        group = [row for row in rows if int(row["causal_k"]) == causal_k]
        best_by_k.append(max(group, key=lambda row: float(row["gap_closure_fraction"])))
    table_rows = _top_rows(rows, best_by_k)
    if korean:
        title = (
            "# Candidate Invariant Gate 결과"
            if is_invariant_gate
            else "# Candidate Safety/Utility Gate 결과"
            if is_safety_gate
            else "# Candidate Regret Gate 결과"
        )
        if is_invariant_gate:
            intro = (
                "이 문서는 candidate descriptor를 train split에서 표준화하고, "
                "train seed별 worst-group loss와 no-harm objective를 함께 줄이는 P1 probe를 요약한다."
            )
        else:
            intro = (
            "이 문서는 candidate별 safe probability와 utility를 별도로 예측하고, "
            "예측 utility와 안전 확률이 충분할 때만 baseline 대신 선택하는 P1 probe를 요약한다."
            if is_safety_gate
            else "이 문서는 candidate별 `candidate_loss - learned_loss`를 직접 예측하고, "
            "예측 regret이 충분히 낮을 때만 baseline 대신 선택하는 P1 probe를 요약한다."
            )
        conclusion = (
            f"최고 closure는 `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`)다. P1 목표 `0.5`를 기준으로 "
            f"{'invariant-gate deployment' if is_invariant_gate else 'safety/utility deployment' if is_safety_gate else 'candidate-regret deployment'}가 candidate-oracle gap을 충분히 닫는지와 "
            "harmful accept를 억제하는지를 동시에 본다."
            + (
                f" Harmful accept <= `0.25` 조건의 conservative best는 "
                f"`{float(safe_best['gap_closure_fraction']):.6f}` "
                f"(`{safe_best['policy']}`)다."
                if safe_best is not None
                else " Harmful accept <= `0.25` 조건을 만족하는 deployed policy는 없다."
            )
            + (
                f" Train-selected deployed best는 `{float(train_selected_best['gap_closure_fraction']):.6f}` "
                f"(`K={train_selected_best['causal_k']}`)다."
                if train_selected_best is not None
                else ""
            )
        )
        notes_title = "## 해석"
        notes = [
            "CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.",
            "아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.",
            "좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.",
        ]
    else:
        title = (
            "# Candidate Invariant Gate Results"
            if is_invariant_gate
            else "# Candidate Safety/Utility Gate Results"
            if is_safety_gate
            else "# Candidate Regret Gate Results"
        )
        if is_invariant_gate:
            intro = (
                "This report summarizes a P1 probe that standardizes candidate descriptors "
                "on the training split and jointly minimizes no-harm utility loss, "
                "worst-source-seed loss, and cross-source variance."
            )
        else:
            intro = (
            "This report summarizes a P1 probe that predicts safe probability "
            "and utility separately, then deploys a candidate only when the "
            "predicted utility and safety probability are sufficiently favorable."
            if is_safety_gate
            else "This report summarizes a P1 probe that directly predicts "
            "`candidate_loss - learned_loss` and deploys a candidate only when "
            "predicted regret is sufficiently favorable."
            )
        conclusion = (
            f"The best closure is `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`). P1 evaluates whether "
            f"{'invariant-gate deployment' if is_invariant_gate else 'safety/utility deployment' if is_safety_gate else 'candidate-regret deployment'} closes the candidate-oracle gap while "
            "controlling harmful accepts."
            + (
                f" The conservative best under harmful-accept <= `0.25` is "
                f"`{float(safe_best['gap_closure_fraction']):.6f}` "
                f"(`{safe_best['policy']}`)."
                if safe_best is not None
                else " No deployed policy satisfies harmful-accept <= `0.25`."
            )
            + (
                f" The train-selected deployed best is `{float(train_selected_best['gap_closure_fraction']):.6f}` "
                f"(`K={train_selected_best['causal_k']}`)."
                if train_selected_best is not None
                else ""
            )
        )
        notes_title = "## Interpretation"
        notes = [
            "The CSV keeps all reject-margin/risk-penalty deployment sweep points.",
            "The table below shows the best policy per K and the strongest overall policies.",
            "A useful deployed policy needs both high closure and low harmful accepts.",
        ]

    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{source.as_posix()}`",
        "",
        conclusion,
        "",
        "| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['causal_k']} | `{row['policy']}` | {float(row['policy_loss']):.6f} | "
            f"{float(row['policy_accuracy']):.6f} | {float(row['candidate_oracle_gain_over_static']):.6f} | "
            f"{float(row['deployed_gain_over_static']):.6f} | {float(row['gap_closure_fraction']):.6f} | "
            f"{float(row['mean_accept_rate']):.6f} | {float(row['mean_harmful_accept_rate']):.6f} | "
            f"{float(row['mean_regret_corr']):.6f} | `{row['failure_mode']}` |"
        )
    lines.extend(["", notes_title, ""])
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def _top_rows(rows: list[dict[str, object]], best_by_k: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[int, str]] = set()
    for row in [*best_by_k, *sorted(rows, key=lambda item: float(item["gap_closure_fraction"]), reverse=True)]:
        key = (int(row["causal_k"]), str(row["policy"]))
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
        if len(selected) >= MAX_MARKDOWN_ROWS:
            break
    return selected


if __name__ == "__main__":
    main()
