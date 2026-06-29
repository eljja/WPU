from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe adaptive bounded observation budgets for WPU world-copy correction."
    )
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--escape-rate", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--dual-omission", type=float, default=0.75)
    parser.add_argument("--contamination", type=int, default=128)
    parser.add_argument("--fixed-budget", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--cost-lambda", type=float, default=0.015)
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_adaptive_observation_budget_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_adaptive_observation_budget_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_adaptive_observation_budget_probe_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    root_rng = random.Random(args.seed)
    for n in args.world_sizes:
        for escape_rate in args.escape_rate:
            for mode in ("wpu-neighbor-only", "wpu-fixed-observation", "wpu-adaptive-observation", "dense-state-copy"):
                trials = [
                    run_stream(
                        n=n,
                        k=args.k_ref,
                        contamination=args.contamination,
                        dual_omission=args.dual_omission,
                        escape_rate=escape_rate,
                        fixed_budget=args.fixed_budget,
                        max_budget=args.max_budget,
                        cost_lambda=args.cost_lambda,
                        horizon=args.horizon,
                        mode=mode,
                        rng=random.Random(root_rng.randrange(2**31)),
                    )
                    for _ in range(args.streams)
                ]
                rows.append({
                    "mode": mode,
                    "total_n": n,
                    "escape_rate": escape_rate,
                    "dual_omission": args.dual_omission,
                    "horizon": args.horizon,
                    "streams": args.streams,
                    "mean_selected_k": mean(trials, "selected_k"),
                    "max_selected_k": max(t["selected_k"] for t in trials),
                    "mean_observation_budget": mean(trials, "observation_budget"),
                    "causal_recall": mean(trials, "recall"),
                    "causal_precision": mean(trials, "precision"),
                    "trajectory_mse": mean(trials, "mse"),
                    "state_integrity": mean(trials, "integrity"),
                    "observation_hit_rate": mean(trials, "observation_hit_rate"),
                    "observation_cost": mean(trials, "observation_cost"),
                    "objective": mean(trials, "objective"),
                    "work_proxy": mean(trials, "work"),
                    "bytes_proxy": mean(trials, "bytes"),
                })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_md.write_text(report(rows, args.out_csv, ko=False), encoding="utf-8")
    args.out_ko_md.write_text(report(rows, args.out_csv, ko=True), encoding="utf-8")
    print(f"wrote={args.out_csv}\nwrote={args.out_md}\nwrote={args.out_ko_md}")


def run_stream(
    n: int,
    k: int,
    contamination: int,
    dual_omission: float,
    escape_rate: float,
    fixed_budget: int,
    max_budget: int,
    cost_lambda: float,
    horizon: int,
    mode: str,
    rng: random.Random,
) -> dict[str, float]:
    causal = [f"causal_{i}" for i in range(k)]
    background = [f"bg_{i}" for i in range(n - k)]
    causal_set = set(causal)

    active: set[str] = set()
    relation_frontier: set[str] = set()
    neighbor_pool: set[str] = set()
    hidden_unknown: set[str] = set()
    for object_id in causal:
        if rng.random() < dual_omission:
            if rng.random() < escape_rate:
                hidden_unknown.add(object_id)
            else:
                neighbor_pool.add(object_id)
            continue
        active.add(object_id)
        if rng.random() >= 0.25:
            relation_frontier.add(object_id)

    distractor_count = min(contamination, len(background))
    active.update(rng.sample(background, distractor_count))
    remaining_background = [oid for oid in background if oid not in active]
    neighbor_pool.update(rng.sample(remaining_background, min(contamination, len(remaining_background))))

    role = {object_id: "affected" for object_id in causal}
    confidence = {object_id: rng.uniform(0.88, 1.0) for object_id in causal}
    anomaly = {object_id: rng.uniform(0.85, 1.0) for object_id in hidden_unknown}
    for object_id in background:
        role[object_id] = "affected" if rng.random() < 0.2 else "background"
        confidence[object_id] = rng.uniform(0.35, 0.78)
        anomaly[object_id] = rng.uniform(0.1, 0.72)

    base_candidates = active | relation_frontier
    observed_support = sum(
        1
        for object_id in base_candidates
        if role[object_id] == "affected" and confidence[object_id] >= 0.85
    )
    support_deficit = max(k - observed_support, 0)

    if mode == "wpu-neighbor-only":
        observation_budget = 0
    elif mode == "wpu-fixed-observation":
        observation_budget = fixed_budget if support_deficit > 0 else 0
    elif mode == "wpu-adaptive-observation":
        # A support deficit alone is not enough: it can be caused by harmless
        # local index noise. The adaptive WPU policy spends observation only
        # when a cheap uncertainty/anomaly monitor indicates unresolved objects.
        uncertainty_signal_count = sum(1 for object_id in hidden_unknown if anomaly[object_id] >= 0.85)
        observation_budget = min(max_budget, support_deficit, uncertainty_signal_count)
    elif mode == "dense-state-copy":
        observation_budget = 0
    else:
        raise ValueError(mode)

    observed: set[str] = set()
    candidates = set(base_candidates) | neighbor_pool
    if observation_budget > 0:
        observation_candidates = hidden_unknown | set(rng.sample(background, min(contamination, len(background))))
        observed = set(sorted(observation_candidates, key=lambda oid: anomaly.get(oid, 0.0), reverse=True)[:observation_budget])
        candidates.update(observed)

    if mode == "dense-state-copy":
        selected = set(causal)
    else:
        cap = 3 * k if observation_budget == 0 else 4 * k
        selected = score_and_cap(candidates, relation_frontier | neighbor_pool | observed, role, confidence, cap)

    causal_selected = selected & causal_set
    false_selected = selected - causal_set
    recall = len(causal_selected) / k
    precision = len(causal_selected) / max(len(selected), 1)
    observation_hit_rate = len(observed & hidden_unknown) / max(len(hidden_unknown), 1)

    missed_error = 0.0
    false_error = 0.0
    for step in range(horizon):
        force = 0.5 + 0.04 * step
        missed_error += sum(
            (force * (0.6 + 0.05 * (i % 3))) ** 2
            for i, object_id in enumerate(causal)
            if object_id not in selected
        ) / k
        false_error += sum((force * confidence[object_id] * 0.25) ** 2 for object_id in false_selected) / k
    mse = (missed_error + false_error) / horizon
    work = n if mode == "dense-state-copy" else len(selected)
    observation_cost = float(observation_budget)
    return {
        "selected_k": float(len(selected)),
        "observation_budget": float(observation_budget),
        "recall": recall,
        "precision": precision,
        "mse": mse,
        "integrity": 1.0 / (1.0 + mse),
        "observation_hit_rate": observation_hit_rate,
        "observation_cost": observation_cost,
        "objective": mse + cost_lambda * observation_cost,
        "work": float(work),
        "bytes": float(work * 9 * 4),
    }


def score_and_cap(
    candidates: set[str],
    relation_frontier: set[str],
    role: dict[str, str],
    confidence: dict[str, float],
    cap: int,
) -> set[str]:
    scored = sorted(
        candidates,
        key=lambda oid: (role[oid] == "affected", oid in relation_frontier, confidence[oid]),
        reverse=True,
    )
    return set(scored[:cap])


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6)


def report(rows: list[dict[str, object]], source: Path, ko: bool) -> str:
    if ko:
        lines = [
            "# Adaptive Observation Budget Probe",
            "",
            "이 probe는 WPU가 uncertainty/support deficit으로 bounded observation budget을 선택해 고정 budget 대비 비용-정확도 tradeoff를 개선할 수 있는지 검증한다.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | escape | mean K | max K | budget | recall | MSE | integrity | obs hit | obs cost | objective | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Adaptive Observation Budget Probe",
            "",
            "This probe tests whether WPU can choose a bounded observation budget from uncertainty/support deficit and improve the cost-accuracy tradeoff over a fixed budget.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | escape | mean K | max K | budget | recall | MSE | integrity | obs hit | obs cost | objective | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['total_n']} | {float(row['escape_rate']):.2f} | "
            f"{float(row['mean_selected_k']):.3f} | {row['max_selected_k']} | "
            f"{float(row['mean_observation_budget']):.3f} | {float(row['causal_recall']):.3f} | "
            f"{float(row['trajectory_mse']):.4f} | {float(row['state_integrity']):.3f} | "
            f"{float(row['observation_hit_rate']):.3f} | {float(row['observation_cost']):.3f} | "
            f"{float(row['objective']):.4f} | {float(row['work_proxy']):.1f} | {float(row['bytes_proxy']):.1f} |"
        )
    if ko:
        lines += [
            "",
            "## 해석",
            "",
            "- Adaptive policy는 local support deficit을 observation budget으로 변환하며, budget은 `max_budget`으로 제한된다.",
            "- 낮은 escape regime에서는 고정 budget보다 observation cost를 줄이는 것이 목표이고, 높은 escape regime에서는 fixed-budget 정확도에 접근하는 것이 목표다.",
            "- Dense state copy는 raw accuracy에서 계속 상한이지만 O(N) work/bytes를 사용한다.",
            "- 이 결과는 learned policy의 최종 형태가 아니라, budget 선택을 WPU correction loop의 native decision으로 분리한 substrate다.",
        ]
    else:
        lines += [
            "",
            "## Interpretation",
            "",
            "- The adaptive policy maps local support deficit to observation budget and caps it by `max_budget`.",
            "- In low-escape regimes it should spend less observation than a fixed budget; in high-escape regimes it should approach fixed-budget accuracy.",
            "- Dense state copy remains the raw-accuracy upper bound but uses O(N) work/bytes.",
            "- This is not the final learned policy; it isolates budget choice as a native WPU correction-loop decision.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
