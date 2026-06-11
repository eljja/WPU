from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path


DEFAULT_INPUT = Path("docs/experiments/pybullet_learned_uncertainty_gate.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/pybullet_mechanism_selective_calibration_gate.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_mechanism_selective_calibration_gate_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_mechanism_selective_calibration_gate_results.ko.md")


METRICS = ["branch_accuracy", "ece", "brier", "nll", "dense_recompute_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether mechanism-selective WPU recompute policies can satisfy "
            "accuracy, calibration, and low-cost constraints."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    parser.add_argument("--cost-budget", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = [row for row in rows if row["row_type"] == "summary"]
    audit_rows = _audit(summary, cost_budget=args.cost_budget)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, audit_rows)
    args.out_md.write_text(_render(audit_rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(audit_rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _audit(rows: list[dict[str, str]], *, cost_budget: float) -> list[dict[str, object]]:
    rows = [row for row in rows if row["eval_mechanism"] != "aggregate"]
    mechanisms = sorted({row["eval_mechanism"] for row in rows})
    by_mechanism = {mechanism: [row for row in rows if row["eval_mechanism"] == mechanism] for mechanism in mechanisms}
    sparse_by_mechanism = {
        mechanism: next(row for row in candidates if row["policy"] == "wpu_sparse")
        for mechanism, candidates in by_mechanism.items()
    }
    sparse_reference = _aggregate(list(sparse_by_mechanism.values()))

    mechanism_best_rows: list[dict[str, object]] = []
    candidate_lists: list[list[dict[str, str]]] = []
    for mechanism in mechanisms:
        candidates = by_mechanism[mechanism]
        sparse = sparse_by_mechanism[mechanism]
        non_reference = [row for row in candidates if row["policy"] != "wpu_sparse"]
        safe_candidates = [
            row
            for row in non_reference
            if _to_float(row, "branch_accuracy") >= _to_float(sparse, "branch_accuracy")
            and _to_float(row, "ece") <= _to_float(sparse, "ece")
            and _to_float(row, "brier") <= _to_float(sparse, "brier")
        ]
        best_safe = max(
            safe_candidates,
            key=lambda row: (
                _to_float(row, "branch_accuracy") - _to_float(sparse, "branch_accuracy"),
                _to_float(sparse, "ece") - _to_float(row, "ece"),
                -_to_float(row, "dense_recompute_rate"),
            ),
        ) if safe_candidates else None
        if best_safe is not None:
            mechanism_best_rows.append(_mechanism_row(mechanism, sparse, best_safe, "mechanism_safe_best"))
        else:
            mechanism_best_rows.append(_mechanism_row(mechanism, sparse, sparse, "no_safe_candidate"))
        candidate_lists.append(candidates)

    policy_rows: list[dict[str, object]] = []
    for combo in product(*candidate_lists):
        aggregate = _aggregate(list(combo))
        deltas = _deltas(aggregate, sparse_reference)
        low_cost = aggregate["dense_recompute_rate"] <= cost_budget
        calibration_safe = deltas["ece_delta"] <= 0.0 and deltas["brier_delta"] <= 0.0
        accuracy_safe = deltas["accuracy_delta"] >= 0.0
        policy_rows.append(
            {
                "row_type": "policy",
                "policy": "; ".join(f"{row['eval_mechanism']}={row['policy']}" for row in combo),
                "cost_budget": round(cost_budget, 6),
                "dense_recompute_rate": round(aggregate["dense_recompute_rate"], 6),
                "branch_accuracy": round(aggregate["branch_accuracy"], 6),
                "ece": round(aggregate["ece"], 6),
                "brier": round(aggregate["brier"], 6),
                "nll": round(aggregate["nll"], 6),
                "accuracy_delta": round(deltas["accuracy_delta"], 6),
                "ece_delta": round(deltas["ece_delta"], 6),
                "brier_delta": round(deltas["brier_delta"], 6),
                "nll_delta": round(deltas["nll_delta"], 6),
                "low_cost": low_cost,
                "accuracy_safe": accuracy_safe,
                "calibration_safe": calibration_safe,
                "all_safe": low_cost and accuracy_safe and calibration_safe,
            }
        )

    safe_rows = [row for row in policy_rows if row["all_safe"] and _is_non_reference_policy(str(row["policy"]))]
    best_safe = max(
        safe_rows,
        key=lambda row: (
            float(row["accuracy_delta"]),
            -float(row["ece_delta"]),
            -float(row["dense_recompute_rate"]),
        ),
    ) if safe_rows else None
    best_low_cost = max(
        [row for row in policy_rows if row["low_cost"]],
        key=lambda row: (
            float(row["accuracy_delta"]),
            -float(row["ece_delta"]),
            -float(row["brier_delta"]),
        ),
    )
    lowest_cost_safe = min(safe_rows, key=lambda row: float(row["dense_recompute_rate"])) if safe_rows else None
    summary = [
        {
            "row_type": "summary",
            "policy": "sparse_reference",
            "cost_budget": round(cost_budget, 6),
            "dense_recompute_rate": round(sparse_reference["dense_recompute_rate"], 6),
            "branch_accuracy": round(sparse_reference["branch_accuracy"], 6),
            "ece": round(sparse_reference["ece"], 6),
            "brier": round(sparse_reference["brier"], 6),
            "nll": round(sparse_reference["nll"], 6),
            "safe_policy_count": len(safe_rows),
            "best_safe_policy": best_safe["policy"] if best_safe else "",
            "best_safe_accuracy_delta": best_safe["accuracy_delta"] if best_safe else "",
            "best_safe_ece_delta": best_safe["ece_delta"] if best_safe else "",
            "best_safe_brier_delta": best_safe["brier_delta"] if best_safe else "",
            "best_safe_cost": best_safe["dense_recompute_rate"] if best_safe else "",
            "lowest_cost_safe_policy": lowest_cost_safe["policy"] if lowest_cost_safe else "",
            "lowest_cost_safe_cost": lowest_cost_safe["dense_recompute_rate"] if lowest_cost_safe else "",
            "best_low_cost_policy": best_low_cost["policy"],
            "best_low_cost_accuracy_delta": best_low_cost["accuracy_delta"],
            "best_low_cost_ece_delta": best_low_cost["ece_delta"],
            "best_low_cost_brier_delta": best_low_cost["brier_delta"],
            "best_low_cost_cost": best_low_cost["dense_recompute_rate"],
        }
    ]
    top_policy_rows = sorted(
        policy_rows,
        key=lambda row: (
            not row["all_safe"],
            not row["low_cost"],
            -float(row["accuracy_delta"]),
            float(row["ece_delta"]),
            float(row["dense_recompute_rate"]),
        ),
    )[:200]
    return summary + mechanism_best_rows + top_policy_rows


def _mechanism_row(
    mechanism: str,
    sparse: dict[str, str],
    policy: dict[str, str],
    row_type: str,
) -> dict[str, object]:
    return {
        "row_type": row_type,
        "mechanism": mechanism,
        "policy": policy["policy"],
        "dense_recompute_rate": round(_to_float(policy, "dense_recompute_rate"), 6),
        "branch_accuracy": round(_to_float(policy, "branch_accuracy"), 6),
        "ece": round(_to_float(policy, "ece"), 6),
        "brier": round(_to_float(policy, "brier"), 6),
        "accuracy_delta": round(_to_float(policy, "branch_accuracy") - _to_float(sparse, "branch_accuracy"), 6),
        "ece_delta": round(_to_float(policy, "ece") - _to_float(sparse, "ece"), 6),
        "brier_delta": round(_to_float(policy, "brier") - _to_float(sparse, "brier"), 6),
    }


def _is_non_reference_policy(policy: str) -> bool:
    return any(part.split("=", 1)[1] != "wpu_sparse" for part in policy.split("; "))


def _aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    return {metric: sum(_to_float(row, metric) for row in rows) / len(rows) for metric in METRICS}


def _deltas(policy: dict[str, float], sparse: dict[str, float]) -> dict[str, float]:
    return {
        "accuracy_delta": policy["branch_accuracy"] - sparse["branch_accuracy"],
        "ece_delta": policy["ece"] - sparse["ece"],
        "brier_delta": policy["brier"] - sparse["brier"],
        "nll_delta": policy["nll"] - sparse["nll"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    summary = next(row for row in rows if row["row_type"] == "summary")
    mechanism_rows = [row for row in rows if row["row_type"] in {"mechanism_safe_best", "no_safe_candidate"}]
    policy_rows = [row for row in rows if row["row_type"] == "policy"]
    safe_policy_rows = [row for row in policy_rows if row["all_safe"]]
    top_rows = sorted(
        policy_rows,
        key=lambda row: (
            not row["all_safe"],
            not row["low_cost"],
            -float(row["accuracy_delta"]),
            float(row["ece_delta"]),
            float(row["dense_recompute_rate"]),
        ),
    )[: args.top_k]

    if korean:
        title = "# PyBullet Mechanism-Selective Calibration Gate"
        intro = (
            "이 audit는 하나의 전역 threshold가 아니라 mechanism별 WPU recompute policy를 선택하면 "
            "low-cost 예산 안에서 accuracy와 calibration을 동시에 개선할 수 있는지 검사한다."
        )
        caveat = (
            "이 결과는 zero-shot routing이 아니다. Mechanism 또는 mechanism-level detector가 이미 "
            "식별되었다는 adapted setting이다. 목적은 P5의 다음 방향이 전역 confidence gate가 아니라 "
            "mechanism-aware calibration routing인지 확인하는 것이다."
        )
        interpretation_heading = "## 해석"
        mechanism_heading = "## Mechanism별 안전 후보"
        top_heading = "## 상위 조합"
        if int(summary["safe_policy_count"]) > 0:
            interpretation = [
                f"Low-cost(`cost <= {float(summary['cost_budget']):.2f}`), accuracy-safe, calibration-safe non-reference 조합은 `{int(summary['safe_policy_count'])}`개다.",
                f"Best safe policy의 accuracy delta는 `{float(summary['best_safe_accuracy_delta']):.6f}`, ECE delta는 `{float(summary['best_safe_ece_delta']):.6f}`, Brier delta는 `{float(summary['best_safe_brier_delta']):.6f}`, cost는 `{float(summary['best_safe_cost']):.6f}`이다.",
                "이는 P5가 불가능한 문제가 아니라, mechanism-aware selective routing으로 풀어야 하는 문제임을 지지한다.",
                "단, mechanism 식별과 calibration sample 의존성이 남아 있으므로 zero-shot calibration-safe routing은 아직 아니다.",
            ]
        else:
            interpretation = [
                f"Low-cost(`cost <= {float(summary['cost_budget']):.2f}`), accuracy-safe, calibration-safe non-reference 조합은 `0`개다.",
                "따라서 현재 후보군으로는 mechanism-selective routing도 P5를 해결하지 못한다.",
            ]
    else:
        title = "# PyBullet Mechanism-Selective Calibration Gate"
        intro = (
            "This audit tests whether selecting different WPU recompute policies by mechanism can satisfy "
            "accuracy, calibration, and low-cost constraints when a single global threshold cannot."
        )
        caveat = (
            "This is not zero-shot routing. It assumes a mechanism or mechanism-level detector has already "
            "identified the shifted family. The purpose is to test whether P5 should move from global confidence "
            "gates toward mechanism-aware calibration routing."
        )
        interpretation_heading = "## Interpretation"
        mechanism_heading = "## Per-Mechanism Safe Candidates"
        top_heading = "## Top Combinations"
        if int(summary["safe_policy_count"]) > 0:
            interpretation = [
                f"There are `{int(summary['safe_policy_count'])}` low-cost (`cost <= {float(summary['cost_budget']):.2f}`), accuracy-safe, calibration-safe non-reference combinations.",
                f"The best safe policy has accuracy delta `{float(summary['best_safe_accuracy_delta']):.6f}`, ECE delta `{float(summary['best_safe_ece_delta']):.6f}`, Brier delta `{float(summary['best_safe_brier_delta']):.6f}`, and cost `{float(summary['best_safe_cost']):.6f}`.",
                "This suggests P5 is not impossible; the next path is mechanism-aware selective routing rather than another global confidence threshold.",
                "The caveat is decisive: mechanism identification and calibration samples remain required, so this is not zero-shot calibration-safe routing.",
            ]
        else:
            interpretation = [
                f"There are `0` low-cost (`cost <= {float(summary['cost_budget']):.2f}`), accuracy-safe, calibration-safe non-reference combinations.",
                "Thus mechanism-selective routing also fails with the current candidate set.",
            ]

    lines = [
        title,
        "",
        intro,
        "",
        caveat,
        "",
        f"Source CSV: `{args.input.as_posix()}`",
        "",
        f"Derived CSV: `{args.out_csv.as_posix()}`",
        "",
        interpretation_heading,
        "",
    ]
    lines.extend(f"- {item}" for item in interpretation)
    lines.extend(
        [
            "",
            mechanism_heading,
            "",
            "| Mechanism | Selected policy | Cost | Accuracy delta | ECE delta | Brier delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in mechanism_rows:
        lines.append(
            f"| `{row['mechanism']}` | `{row['policy']}` | {float(row['dense_recompute_rate']):.6f} | "
            f"{float(row['accuracy_delta']):.6f} | {float(row['ece_delta']):.6f} | "
            f"{float(row['brier_delta']):.6f} |"
        )
    lines.extend(
        [
            "",
            top_heading,
            "",
            "| Policy | Cost | Accuracy delta | ECE delta | Brier delta | Safe |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| `{row['policy']}` | {float(row['dense_recompute_rate']):.6f} | "
            f"{float(row['accuracy_delta']):.6f} | {float(row['ece_delta']):.6f} | "
            f"{float(row['brier_delta']):.6f} | {row['all_safe']} |"
        )
    if safe_policy_rows:
        lines.extend(["", "## Best Safe Policy", "", f"`{summary['best_safe_policy']}`"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
