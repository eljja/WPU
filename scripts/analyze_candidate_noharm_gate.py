from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUT = Path("docs/experiments/wpu_v2_retriever_conservative_set_evaluator.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/wpu_v2_candidate_noharm_gate_summary.csv")
DEFAULT_OUT_MD = Path("docs/experiments/wpu_v2_candidate_noharm_gate_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/wpu_v2_candidate_noharm_gate_results.ko.md")


DEPLOYED_POLICIES = [
    "set_evaluator",
    "conservative_margin_gate",
    "robust_per_seed_margin_gate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize sample-level no-harm gate closure for candidate-oracle gap analysis."
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
        for policy in DEPLOYED_POLICIES:
            policy_rows = [row for row in group if row["policy"] == policy]
            use_rate_col = "test_robust_use_rate" if policy == "robust_per_seed_margin_gate" else "test_evaluator_use_rate"
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
                    "mean_gate_use_rate": round(
                        statistics.fmean(float(row[use_rate_col]) for row in policy_rows),
                        6,
                    ),
                    "mean_test_margin": round(
                        statistics.fmean(float(row["test_mean_margin"]) for row in policy_rows),
                        6,
                    ),
                    "seed_count": len({int(row["seed"]) for row in policy_rows}),
                    "failure_mode": _failure_mode(gap_closure),
                }
            )
    return out


def _failure_mode(gap_closure: float) -> str:
    if gap_closure < 0.0:
        return "harmful_gate_transfer"
    if gap_closure < 0.1:
        return "weak_sample_level_selection_signal"
    if gap_closure < 0.5:
        return "partial_but_insufficient_gap_closure"
    return "passes_current_p1_threshold"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], source: Path, *, korean: bool) -> str:
    best = max(rows, key=lambda row: float(row["gap_closure_fraction"]))
    harmful = sum(1 for row in rows if float(row["gap_closure_fraction"]) < 0.0)
    if korean:
        title = "# Candidate No-Harm Gate 결과"
        intro = (
            "이 문서는 conservative set-evaluator 실험을 P1 candidate-oracle gap 관점에서 "
            "다시 요약한다. 목적은 sample-level no-harm/margin gate가 aggregate selector "
            "실패를 해결하는지 확인하는 것이다."
        )
        conclusion = (
            f"최고 closure는 `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`)이며, "
            f"음수 closure 조건은 `{harmful}`개다. 따라서 현재 margin 기반 no-harm gate는 "
            "P1을 해결하지 못한다. 실패 원인은 단순 threshold 부재가 아니라 candidate별 "
            "uncertainty/regret signal이 held-out seed에서 충분히 transfer되지 않는 데 있다."
        )
        headers = "| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Gate use | Failure mode |"
        notes_title = "## 해석"
        notes = [
            "Gate가 사용률을 낮추지 못하거나 잘못된 candidate를 계속 선택하면 no-harm 조건은 held-out seed에서 깨진다.",
            "K=8/16의 음수 closure는 margin confidence가 실제 downstream regret과 정렬되지 않음을 의미한다.",
            "다음 P1 개선은 threshold 조정보다 per-candidate uncertainty, calibrated regret target, no-harm rejection loss로 내려가야 한다.",
        ]
    else:
        title = "# Candidate No-Harm Gate Results"
        intro = (
            "This report re-summarizes the conservative set-evaluator experiment "
            "through the priority-1 candidate-oracle gap. It asks whether a "
            "sample-level no-harm/margin gate fixes aggregate selector failure."
        )
        conclusion = (
            f"The best closure is `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`), and `{harmful}` "
            "conditions have negative closure. The current margin-based no-harm "
            "gate therefore does not solve P1. The failure is not merely a missing "
            "threshold; candidate-level uncertainty/regret signals do not yet "
            "transfer reliably to held-out seeds."
        )
        headers = "| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Gate use | Failure mode |"
        notes_title = "## Interpretation"
        notes = [
            "A no-harm gate fails when its use-rate remains high or it keeps choosing harmful candidates on held-out seeds.",
            "Negative closure at K=8/16 means margin confidence is not aligned with downstream regret.",
            "The next P1 improvement must move below threshold selection toward per-candidate uncertainty, calibrated regret targets, and no-harm rejection losses.",
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
        headers,
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['causal_k']} | `{row['policy']}` | {float(row['policy_loss']):.6f} | "
            f"{float(row['policy_accuracy']):.6f} | {float(row['candidate_oracle_gain_over_static']):.6f} | "
            f"{float(row['deployed_gain_over_static']):.6f} | {float(row['gap_closure_fraction']):.6f} | "
            f"{float(row['mean_gate_use_rate']):.6f} | `{row['failure_mode']}` |"
        )
    lines.extend(["", notes_title, ""])
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
