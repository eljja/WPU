from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("docs/experiments")
LOW_COST_BUDGET = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the accuracy-calibration-cost frontier of WPU selective recompute/adaptation policies."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "pybullet_calibration_cost_frontier.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "pybullet_calibration_cost_frontier_results.md")
    parser.add_argument("--out-ko-md", type=Path, default=ROOT / "pybullet_calibration_cost_frontier_results.ko.md")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    rows.extend(_uncertainty_gated_rows(ROOT / "pybullet_uncertainty_gated_recompute.csv"))
    rows.extend(_learned_gate_rows(ROOT / "pybullet_learned_uncertainty_gate.csv"))
    rows.extend(_adaptive_policy_rows(ROOT / "pybullet_mechanism_adaptive_policy_summary.csv"))

    for row in rows:
        row["calibration_safe"] = _is_calibration_safe(row)
        row["low_cost"] = float(row["cost_proxy"]) <= LOW_COST_BUDGET
        row["calibration_safe_low_cost"] = bool(row["calibration_safe"]) and bool(row["low_cost"])

    pareto_names = _pareto_policy_names(rows)
    for row in rows:
        row["pareto_efficient"] = row["policy"] in pareto_names

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _uncertainty_gated_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = [
        row
        for row in _read_rows(path)
        if row["row_type"] == "summary" and row["eval_mechanism"] == "aggregate"
    ]
    sparse = next(row for row in rows if row["policy"] == "wpu_sparse")
    result = [_baseline_row("uncertainty_threshold", "wpu_sparse_uncertainty_probe", sparse, path)]
    for row in rows:
        if not row["policy"].startswith("wpu_gated"):
            continue
        result.append(
            _policy_row(
                family="uncertainty_threshold",
                policy=row["policy"],
                protocol="zero_shot_threshold_gate",
                row=row,
                baseline=sparse,
                cost_proxy=float(row["dense_recompute_rate"]),
                dense_recompute_rate=float(row["dense_recompute_rate"]),
                adaptation_required=False,
                source=path,
            )
        )
    return result


def _learned_gate_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = [
        row
        for row in _read_rows(path)
        if row["row_type"] == "summary" and row["eval_mechanism"] == "aggregate"
    ]
    sparse = next(row for row in rows if row["policy"] == "wpu_sparse")
    result = [_baseline_row("learned_sparse_output_gate", "wpu_sparse_learned_gate_probe", sparse, path)]
    for row in rows:
        if row["policy"] in {"wpu_sparse", "wpu_local_dense"}:
            continue
        result.append(
            _policy_row(
                family="learned_sparse_output_gate",
                policy=row["policy"],
                protocol=f"{row['gate_kind']}_learned_gate",
                row=row,
                baseline=sparse,
                cost_proxy=float(row["dense_recompute_rate"]),
                dense_recompute_rate=float(row["dense_recompute_rate"]),
                adaptation_required=row["gate_kind"] == "fewshot",
                source=path,
            )
        )
    local_dense = next((row for row in rows if row["policy"] == "wpu_local_dense"), None)
    if local_dense is not None:
        result.append(
            _policy_row(
                family="learned_sparse_output_gate",
                policy="wpu_local_dense_learned_probe",
                protocol="full_local_dense_reference",
                row=local_dense,
                baseline=sparse,
                cost_proxy=1.0,
                dense_recompute_rate=1.0,
                adaptation_required=False,
                source=path,
            )
        )
    return result


def _adaptive_policy_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    shifted = [row for row in _read_rows(path) if row["eval_mechanism"] != "nominal"]
    if not shifted:
        return []
    accuracy_delta = _mean(float(row["wpu_accuracy_change"]) for row in shifted)
    margin_delta = _mean(float(row["wpu_margin_change"]) for row in shifted)
    ece_delta = _mean(float(row["policy_wpu_ece_change"]) for row in shifted)
    brier_delta = _mean(float(row["policy_wpu_brier_change"]) for row in shifted)
    win_rate = _mean(1.0 if row["policy_wpu_win"] == "True" else 0.0 for row in shifted)
    return [
        {
            "family": "mechanism_adaptive_policy",
            "policy": "mechanism_aware_adaptive_policy",
            "protocol": "detect_and_adapt",
            "cost_proxy": 1.0,
            "dense_recompute_rate": "",
            "adaptation_required": True,
            "branch_accuracy": _mean(float(row["policy_wpu_accuracy"]) for row in shifted),
            "ece": "",
            "brier": "",
            "accuracy_delta": accuracy_delta,
            "margin_delta": margin_delta,
            "ece_delta": ece_delta,
            "brier_delta": brier_delta,
            "wpu_win_rate": win_rate,
            "source": str(path),
        }
    ]


def _baseline_row(family: str, policy: str, row: dict[str, str], source: Path) -> dict[str, object]:
    return {
        "family": family,
        "policy": policy,
        "protocol": "sparse_reference",
        "cost_proxy": 0.0,
        "dense_recompute_rate": 0.0,
        "adaptation_required": False,
        "branch_accuracy": float(row["branch_accuracy"]),
        "ece": float(row["ece"]),
        "brier": float(row["brier"]),
        "accuracy_delta": 0.0,
        "margin_delta": "",
        "ece_delta": 0.0,
        "brier_delta": 0.0,
        "wpu_win_rate": "",
        "source": str(source),
    }


def _policy_row(
    *,
    family: str,
    policy: str,
    protocol: str,
    row: dict[str, str],
    baseline: dict[str, str],
    cost_proxy: float,
    dense_recompute_rate: float,
    adaptation_required: bool,
    source: Path,
) -> dict[str, object]:
    return {
        "family": family,
        "policy": policy,
        "protocol": protocol,
        "cost_proxy": cost_proxy,
        "dense_recompute_rate": dense_recompute_rate,
        "adaptation_required": adaptation_required,
        "branch_accuracy": float(row["branch_accuracy"]),
        "ece": float(row["ece"]),
        "brier": float(row["brier"]),
        "accuracy_delta": float(row["branch_accuracy"]) - float(baseline["branch_accuracy"]),
        "margin_delta": "",
        "ece_delta": float(row["ece"]) - float(baseline["ece"]),
        "brier_delta": float(row["brier"]) - float(baseline["brier"]),
        "wpu_win_rate": "",
        "source": str(source),
    }


def _is_calibration_safe(row: dict[str, object]) -> bool:
    return float(row["accuracy_delta"]) >= 0.0 and float(row["ece_delta"]) <= 0.0 and float(row["brier_delta"]) <= 0.0


def _pareto_policy_names(rows: list[dict[str, object]]) -> set[str]:
    efficient: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if row is other:
                continue
            better_or_equal = (
                float(other["cost_proxy"]) <= float(row["cost_proxy"])
                and float(other["ece_delta"]) <= float(row["ece_delta"])
                and float(other["accuracy_delta"]) >= float(row["accuracy_delta"])
            )
            strictly_better = (
                float(other["cost_proxy"]) < float(row["cost_proxy"])
                or float(other["ece_delta"]) < float(row["ece_delta"])
                or float(other["accuracy_delta"]) > float(row["accuracy_delta"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            efficient.add(str(row["policy"]))
    return efficient


def _render_report(rows: list[dict[str, object]], *, korean: bool) -> str:
    policy_rows = [row for row in rows if row["protocol"] != "sparse_reference"]
    low_cost_rows = [row for row in policy_rows if bool(row["low_cost"])]
    safe_rows = [row for row in policy_rows if bool(row["calibration_safe"])]
    safe_low_cost_rows = [row for row in policy_rows if bool(row["calibration_safe_low_cost"])]
    pareto_rows = [row for row in rows if bool(row["pareto_efficient"])]

    best_low_cost_accuracy = max(low_cost_rows, key=lambda row: float(row["accuracy_delta"]))
    best_safe = min(safe_rows, key=lambda row: (float(row["cost_proxy"]), float(row["ece_delta"]))) if safe_rows else None
    best_ece = min(policy_rows, key=lambda row: float(row["ece_delta"]))

    if korean:
        title = "# PyBullet Calibration-Cost Frontier Audit\n"
        summary = (
            "이 파생 감사는 기존 uncertainty-gated recompute, learned gate, mechanism-adaptive "
            "policy 결과를 같은 축으로 정규화한다. 목표는 P5를 성공처럼 포장하는 것이 아니라, "
            "정확도 개선, calibration 개선, 낮은 추가 계산이 동시에 가능한지를 검증하는 것이다.\n"
        )
        safe_text = (
            f"- Low-cost budget(`cost_proxy <= {LOW_COST_BUDGET}`) 안에서 non-reference calibration-safe policy는 "
            f"{len(safe_low_cost_rows)}개다.\n"
            f"- 가장 정확한 low-cost policy는 `{best_low_cost_accuracy['policy']}`이며 accuracy delta "
            f"`{float(best_low_cost_accuracy['accuracy_delta']):.6f}`, ECE delta "
            f"`{float(best_low_cost_accuracy['ece_delta']):.6f}`, cost proxy "
            f"`{float(best_low_cost_accuracy['cost_proxy']):.6f}`이다.\n"
            f"- 가장 큰 ECE 개선은 `{best_ece['policy']}`이며 ECE delta "
            f"`{float(best_ece['ece_delta']):.6f}`, accuracy delta "
            f"`{float(best_ece['accuracy_delta']):.6f}`, cost proxy "
            f"`{float(best_ece['cost_proxy']):.6f}`이다.\n"
        )
        if best_safe is not None:
            safe_text += (
                f"- 최저 비용 non-reference calibration-safe policy는 `{best_safe['policy']}`이며 cost proxy "
                f"`{float(best_safe['cost_proxy']):.6f}`이다.\n"
            )
        interpretation = (
            "\n## Interpretation\n\n"
            "현재 증거에서 WPU는 selective state processing으로 계산량을 줄일 수 있지만, "
            "low-cost와 calibration-safe를 동시에 만족하는 정책은 아직 확보되지 않았다. "
            "학습된 sparse-output gate는 낮은 recompute rate에서 accuracy를 개선하지만 ECE가 "
            "악화되고, calibration-safe 개선은 거의 full dense recompute 또는 mechanism-specific "
            "detect-and-adapt 정책에 의존한다. 따라서 P5의 다음 연구 목표는 branch-calibration "
            "loss, mechanism uncertainty, shift detector를 결합한 저비용 calibration-aware routing이다.\n"
        )
    else:
        title = "# PyBullet Calibration-Cost Frontier Audit\n"
        summary = (
            "This derived audit normalizes existing uncertainty-gated recompute, learned gate, "
            "and mechanism-adaptive policy results onto common accuracy, calibration, and cost axes. "
            "The purpose is not to claim P5 solved, but to test whether accuracy improvement, "
            "calibration improvement, and low added compute are simultaneously achieved.\n"
        )
        safe_text = (
            f"- Non-reference calibration-safe policies within the low-cost budget (`cost_proxy <= {LOW_COST_BUDGET}`): "
            f"`{len(safe_low_cost_rows)}`.\n"
            f"- The most accurate low-cost policy is `{best_low_cost_accuracy['policy']}` with accuracy delta "
            f"`{float(best_low_cost_accuracy['accuracy_delta']):.6f}`, ECE delta "
            f"`{float(best_low_cost_accuracy['ece_delta']):.6f}`, and cost proxy "
            f"`{float(best_low_cost_accuracy['cost_proxy']):.6f}`.\n"
            f"- The strongest ECE improvement is `{best_ece['policy']}` with ECE delta "
            f"`{float(best_ece['ece_delta']):.6f}`, accuracy delta "
            f"`{float(best_ece['accuracy_delta']):.6f}`, and cost proxy "
            f"`{float(best_ece['cost_proxy']):.6f}`.\n"
        )
        if best_safe is not None:
            safe_text += (
                f"- The lowest-cost non-reference calibration-safe policy is `{best_safe['policy']}` with cost proxy "
                f"`{float(best_safe['cost_proxy']):.6f}`.\n"
            )
        interpretation = (
            "\n## Interpretation\n\n"
            "The current evidence supports WPU as a way to reduce computation through selective state "
            "processing, but it does not yet show a low-cost calibration-safe router. Learned sparse-output "
            "gates improve accuracy at low recompute rate while worsening ECE, whereas calibration-safe "
            "improvements mostly require near-full dense recompute or mechanism-specific detect-and-adapt "
            "policies. P5 therefore remains a concrete research target: train low-cost calibration-aware "
            "routing with branch-calibration losses, mechanism uncertainty, and shift detection.\n"
        )

    table_header = (
        "\n## Frontier Rows\n\n"
        "| family | policy | protocol | cost | acc_delta | ece_delta | brier_delta | safe | low_cost | pareto |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
    )
    table_rows = []
    for row in sorted(rows, key=lambda item: (float(item["cost_proxy"]), str(item["family"]), str(item["policy"]))):
        table_rows.append(
            f"| {row['family']} | {row['policy']} | {row['protocol']} | "
            f"{float(row['cost_proxy']):.6f} | {float(row['accuracy_delta']):.6f} | "
            f"{float(row['ece_delta']):.6f} | {float(row['brier_delta']):.6f} | "
            f"{row['calibration_safe']} | {row['low_cost']} | {row['pareto_efficient']} |"
        )
    pareto_text = ", ".join(str(row["policy"]) for row in sorted(pareto_rows, key=lambda item: float(item["cost_proxy"])))
    source_text = "\nSource CSV: `docs/experiments/pybullet_calibration_cost_frontier.csv`\n"
    return (
        title
        + "\n"
        + summary
        + source_text
        + "\n## Summary\n\n"
        + safe_text
        + f"- Pareto-efficient policies: {pareto_text}.\n"
        + interpretation
        + table_header
        + "\n".join(table_rows)
        + "\n"
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "family",
        "policy",
        "protocol",
        "cost_proxy",
        "dense_recompute_rate",
        "adaptation_required",
        "branch_accuracy",
        "ece",
        "brier",
        "accuracy_delta",
        "margin_delta",
        "ece_delta",
        "brier_delta",
        "wpu_win_rate",
        "calibration_safe",
        "low_cost",
        "calibration_safe_low_cost",
        "pareto_efficient",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
