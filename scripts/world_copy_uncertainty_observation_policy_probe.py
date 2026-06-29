from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe bounded uncertainty-triggered observation for causal objects missing from local WPU indexes."
    )
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--escape-rate", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--observation-budget", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--dual-omission", type=float, default=0.75)
    parser.add_argument("--contamination", type=int, default=128)
    parser.add_argument("--streams", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_uncertainty_observation_policy_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_uncertainty_observation_policy_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_uncertainty_observation_policy_probe_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    root_rng = random.Random(args.seed)
    for n in args.world_sizes:
        for escape_rate in args.escape_rate:
            for observation_budget in args.observation_budget:
                for mode in ("wpu-neighbor-only", "wpu-uncertainty-observation", "dense-state-copy"):
                    trials = [
                        run_stream(
                            n=n,
                            k=args.k_ref,
                            contamination=args.contamination,
                            dual_omission=args.dual_omission,
                            escape_rate=escape_rate,
                            observation_budget=observation_budget,
                            horizon=args.horizon,
                            mode=mode,
                            rng=random.Random(root_rng.randrange(2**31)),
                        )
                        for _ in range(args.streams)
                    ]
                    rows.append({
                        "mode": mode,
                        "total_n": n,
                        "dual_omission": args.dual_omission,
                        "escape_rate": escape_rate,
                        "observation_budget": observation_budget,
                        "horizon": args.horizon,
                        "streams": args.streams,
                        "mean_selected_k": mean(trials, "selected_k"),
                        "max_selected_k": max(t["selected_k"] for t in trials),
                        "causal_recall": mean(trials, "recall"),
                        "causal_precision": mean(trials, "precision"),
                        "trajectory_mse": mean(trials, "mse"),
                        "state_integrity": mean(trials, "integrity"),
                        "false_updates": mean(trials, "false_updates"),
                        "observation_hit_rate": mean(trials, "observation_hit_rate"),
                        "observation_cost": mean(trials, "observation_cost"),
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
    observation_budget: int,
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
    should_observe = observed_support < k

    if mode == "wpu-neighbor-only":
        candidates = base_candidates | neighbor_pool
        selected = score_and_cap(candidates, relation_frontier | neighbor_pool, role, confidence, 3 * k)
        observed = set()
        observation_cost = 0.0
    elif mode == "wpu-uncertainty-observation":
        candidates = set(base_candidates) | neighbor_pool
        observed = set()
        observation_cost = 0.0
        cap = 3 * k
        if should_observe and observation_budget > 0:
            observation_candidates = hidden_unknown | set(rng.sample(background, min(contamination, len(background))))
            observed = set(sorted(observation_candidates, key=lambda oid: anomaly.get(oid, 0.0), reverse=True)[:observation_budget])
            candidates.update(observed)
            observation_cost = float(len(observed))
            cap = 4 * k
        selected = score_and_cap(candidates, relation_frontier | neighbor_pool | observed, role, confidence, cap)
    elif mode == "dense-state-copy":
        selected = set(causal)
        observed = set()
        observation_cost = 0.0
    else:
        raise ValueError(mode)

    causal_selected = selected & causal_set
    false_selected = selected - causal_set
    recall = len(causal_selected) / k
    precision = len(causal_selected) / max(len(selected), 1)
    obs_hits = len(observed & hidden_unknown)
    observation_hit_rate = obs_hits / max(len(hidden_unknown), 1)

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
    return {
        "selected_k": float(len(selected)),
        "recall": recall,
        "precision": precision,
        "mse": mse,
        "integrity": 1.0 / (1.0 + mse),
        "false_updates": float(len(false_selected)),
        "observation_hit_rate": observation_hit_rate,
        "observation_cost": observation_cost,
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
            "# Uncertainty Observation Policy Probe",
            "",
            "이 probe는 causal object가 active region, relation frontier, adjacent pool에서 모두 빠질 때 WPU가 제한된 external observation probe로 복구할 수 있는지 검증한다.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | escape | obs budget | mean K | max K | recall | precision | MSE | integrity | obs hit | obs cost | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Uncertainty Observation Policy Probe",
            "",
            "This probe tests whether WPU can recover causal objects missing from the active region, relation frontier, and adjacent pool through a bounded external observation probe.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | escape | obs budget | mean K | max K | recall | precision | MSE | integrity | obs hit | obs cost | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['total_n']} | {float(row['escape_rate']):.2f} | "
            f"{row['observation_budget']} | {float(row['mean_selected_k']):.3f} | "
            f"{row['max_selected_k']} | {float(row['causal_recall']):.3f} | "
            f"{float(row['causal_precision']):.3f} | {float(row['trajectory_mse']):.4f} | "
            f"{float(row['state_integrity']):.3f} | {float(row['observation_hit_rate']):.3f} | "
            f"{float(row['observation_cost']):.3f} | {float(row['work_proxy']):.1f} | "
            f"{float(row['bytes_proxy']):.1f} |"
        )
    if ko:
        lines += [
            "",
            "## 해석",
            "",
            "- `wpu-uncertainty-observation`은 지원 evidence가 부족할 때만 bounded external observation을 요청한다.",
            "- Observation budget이 hidden causal object 수에 충분히 가까우면 neighbor-only보다 recall과 MSE가 개선된다.",
            "- 이 개선은 무료가 아니다. Observation cost와 selected K가 증가하며, observation ranking이 실패하면 dense-state-copy가 raw accuracy에서 계속 이긴다.",
            "- 핵심 조건은 K가 N이 아니라 observation budget과 local correction pool에 의해 제한된다는 점이다.",
        ]
    else:
        lines += [
            "",
            "## Interpretation",
            "",
            "- `wpu-uncertainty-observation` asks for bounded external observation only when local support evidence is insufficient.",
            "- If the observation budget is close enough to the hidden causal count, recall and MSE improve over neighbor-only correction.",
            "- The improvement is not free: observation cost and selected K increase, and dense-state-copy still wins raw accuracy when observation ranking misses causal objects.",
            "- The key condition is that K is bounded by observation budget and local correction pool rather than total N.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
