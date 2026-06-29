from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Scene:
    n: int
    k: int
    causal: list[str]
    background: list[str]
    active: set[str]
    relation_frontier: set[str]
    neighbor_pool: set[str]
    hidden_unknown: set[str]
    role: dict[str, str]
    confidence: dict[str, float]
    anomaly: dict[str, float]
    support_deficit: int


class BudgetPolicy(nn.Module):
    def __init__(self, max_budget: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 24),
            nn.GELU(),
            nn.Linear(24, max_budget + 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test a learned bounded WPU observation-budget policy.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--escape-rate", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--eval-shifts", type=str, nargs="+", default=["clean", "noisy_anomaly", "weak_anomaly"])
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--dual-omission", type=float, default=0.75)
    parser.add_argument("--contamination", type=int, default=128)
    parser.add_argument("--fixed-budget", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--cost-lambda", type=float, default=0.015)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_learned_observation_policy_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_learned_observation_policy_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_learned_observation_policy_probe_results.ko.md"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = train_policy(args, rng)

    rows: list[dict[str, object]] = []
    for shift in args.eval_shifts:
        for n in args.world_sizes:
            for escape_rate in args.escape_rate:
                for mode in (
                    "wpu-neighbor-only",
                    "wpu-fixed-observation",
                    "wpu-adaptive-observation",
                    "wpu-learned-observation",
                    "dense-state-copy",
                ):
                    trials = [
                        evaluate_scene(
                            make_scene(args, n, escape_rate, shift, random.Random(rng.randrange(2**31))),
                            mode=mode,
                            policy=policy,
                            fixed_budget=args.fixed_budget,
                            max_budget=args.max_budget,
                            cost_lambda=args.cost_lambda,
                            horizon=args.horizon,
                        )
                        for _ in range(args.eval_samples)
                    ]
                    rows.append({
                        "mode": mode,
                        "shift": shift,
                        "total_n": n,
                        "escape_rate": escape_rate,
                        "mean_selected_k": mean(trials, "selected_k"),
                        "max_selected_k": max(t["selected_k"] for t in trials),
                        "mean_observation_budget": mean(trials, "observation_budget"),
                        "causal_recall": mean(trials, "recall"),
                        "trajectory_mse": mean(trials, "mse"),
                        "state_integrity": mean(trials, "integrity"),
                        "observation_hit_rate": mean(trials, "observation_hit_rate"),
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


def train_policy(args: argparse.Namespace, rng: random.Random) -> BudgetPolicy:
    policy = BudgetPolicy(args.max_budget)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3, weight_decay=1e-4)
    features: list[list[float]] = []
    labels: list[int] = []
    for _ in range(args.train_samples):
        n = rng.choice(args.world_sizes)
        escape_rate = rng.choice(args.escape_rate)
        scene = make_scene(args, n, escape_rate, "clean", random.Random(rng.randrange(2**31)))
        features.append(scene_features(scene, args.max_budget))
        labels.append(rule_budget(scene, args.max_budget))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    for _ in range(args.train_steps):
        logits = policy(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy.eval()


def make_scene(args: argparse.Namespace, n: int, escape_rate: float, shift: str, rng: random.Random) -> Scene:
    causal = [f"causal_{i}" for i in range(args.k_ref)]
    background = [f"bg_{i}" for i in range(n - args.k_ref)]
    active: set[str] = set()
    relation_frontier: set[str] = set()
    neighbor_pool: set[str] = set()
    hidden_unknown: set[str] = set()
    for object_id in causal:
        if rng.random() < args.dual_omission:
            if rng.random() < escape_rate:
                hidden_unknown.add(object_id)
            else:
                neighbor_pool.add(object_id)
            continue
        active.add(object_id)
        if rng.random() >= 0.25:
            relation_frontier.add(object_id)

    distractor_count = min(args.contamination, len(background))
    active.update(rng.sample(background, distractor_count))
    remaining_background = [oid for oid in background if oid not in active]
    neighbor_pool.update(rng.sample(remaining_background, min(args.contamination, len(remaining_background))))

    role = {object_id: "affected" for object_id in causal}
    confidence = {object_id: rng.uniform(0.88, 1.0) for object_id in causal}
    anomaly: dict[str, float] = {}
    for object_id in hidden_unknown:
        if shift == "weak_anomaly":
            anomaly[object_id] = rng.uniform(0.65, 0.86)
        else:
            anomaly[object_id] = rng.uniform(0.85, 1.0)
    for object_id in background:
        role[object_id] = "affected" if rng.random() < 0.2 else "background"
        confidence[object_id] = rng.uniform(0.35, 0.78)
        if shift == "noisy_anomaly":
            anomaly[object_id] = rng.uniform(0.1, 0.92)
        else:
            anomaly[object_id] = rng.uniform(0.1, 0.72)

    base_candidates = active | relation_frontier
    observed_support = sum(
        1 for object_id in base_candidates if role[object_id] == "affected" and confidence[object_id] >= 0.85
    )
    support_deficit = max(args.k_ref - observed_support, 0)
    return Scene(
        n=n,
        k=args.k_ref,
        causal=causal,
        background=background,
        active=active,
        relation_frontier=relation_frontier,
        neighbor_pool=neighbor_pool,
        hidden_unknown=hidden_unknown,
        role=role,
        confidence=confidence,
        anomaly=anomaly,
        support_deficit=support_deficit,
    )


def evaluate_scene(
    scene: Scene,
    *,
    mode: str,
    policy: BudgetPolicy,
    fixed_budget: int,
    max_budget: int,
    cost_lambda: float,
    horizon: int,
) -> dict[str, float]:
    if mode == "wpu-neighbor-only":
        observation_budget = 0
    elif mode == "wpu-fixed-observation":
        observation_budget = fixed_budget if scene.support_deficit > 0 else 0
    elif mode == "wpu-adaptive-observation":
        observation_budget = rule_budget(scene, max_budget)
    elif mode == "wpu-learned-observation":
        with torch.no_grad():
            logits = policy(torch.tensor([scene_features(scene, max_budget)], dtype=torch.float32))
            observation_budget = int(logits.argmax(dim=-1).item())
    elif mode == "dense-state-copy":
        observation_budget = 0
    else:
        raise ValueError(mode)

    candidates = set(scene.active) | scene.relation_frontier | scene.neighbor_pool
    observed: set[str] = set()
    if observation_budget > 0:
        observation_candidates = scene.hidden_unknown | set(scene.background[: min(128, len(scene.background))])
        observed = set(
            sorted(observation_candidates, key=lambda oid: scene.anomaly.get(oid, 0.0), reverse=True)[:observation_budget]
        )
        candidates.update(observed)

    if mode == "dense-state-copy":
        selected = set(scene.causal)
    else:
        cap = 3 * scene.k if observation_budget == 0 else 4 * scene.k
        selected = score_and_cap(candidates, scene.relation_frontier | scene.neighbor_pool | observed, scene.role, scene.confidence, cap)

    causal_set = set(scene.causal)
    causal_selected = selected & causal_set
    false_selected = selected - causal_set
    recall = len(causal_selected) / scene.k
    observation_hit_rate = len(observed & scene.hidden_unknown) / max(len(scene.hidden_unknown), 1)
    missed_error = 0.0
    false_error = 0.0
    for step in range(horizon):
        force = 0.5 + 0.04 * step
        missed_error += sum(
            (force * (0.6 + 0.05 * (i % 3))) ** 2
            for i, object_id in enumerate(scene.causal)
            if object_id not in selected
        ) / scene.k
        false_error += sum((force * scene.confidence[object_id] * 0.25) ** 2 for object_id in false_selected) / scene.k
    mse = (missed_error + false_error) / horizon
    work = scene.n if mode == "dense-state-copy" else len(selected)
    return {
        "selected_k": float(len(selected)),
        "observation_budget": float(observation_budget),
        "recall": recall,
        "mse": mse,
        "integrity": 1.0 / (1.0 + mse),
        "observation_hit_rate": observation_hit_rate,
        "objective": mse + cost_lambda * float(observation_budget),
        "work": float(work),
        "bytes": float(work * 9 * 4),
    }


def rule_budget(scene: Scene, max_budget: int) -> int:
    anomaly_count = sum(1 for object_id in scene.hidden_unknown if scene.anomaly.get(object_id, 0.0) >= 0.85)
    return min(max_budget, scene.support_deficit, anomaly_count)


def scene_features(scene: Scene, max_budget: int) -> list[float]:
    candidate_anomalies = [scene.anomaly.get(oid, 0.0) for oid in scene.hidden_unknown | scene.neighbor_pool]
    max_anomaly = max(candidate_anomalies, default=0.0)
    mean_top_anomaly = sum(sorted(candidate_anomalies, reverse=True)[:max_budget]) / max(max_budget, 1)
    anomaly_count = sum(1 for value in candidate_anomalies if value >= 0.85)
    return [
        scene.support_deficit / max(scene.k, 1),
        anomaly_count / max(max_budget, 1),
        max_anomaly,
        mean_top_anomaly,
        min(len(scene.neighbor_pool), 128) / 128.0,
    ]


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
    title = "# Learned Observation Budget Policy Probe"
    intro = (
        "이 probe는 hand-specified observation budget rule을 작은 learned policy로 대체할 수 있는지, 그리고 anomaly shift에서 어디서 실패하는지 검증한다."
        if ko
        else "This probe tests whether a small learned policy can replace the hand-specified observation-budget rule, and where it fails under anomaly shift."
    )
    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{source.as_posix()}`.",
        "",
        "| mode | shift | N | escape | mean K | max K | budget | recall | MSE | integrity | obs hit | objective | work | bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['shift']} | {row['total_n']} | {float(row['escape_rate']):.2f} | "
            f"{float(row['mean_selected_k']):.3f} | {row['max_selected_k']} | "
            f"{float(row['mean_observation_budget']):.3f} | {float(row['causal_recall']):.3f} | "
            f"{float(row['trajectory_mse']):.4f} | {float(row['state_integrity']):.3f} | "
            f"{float(row['observation_hit_rate']):.3f} | {float(row['objective']):.4f} | "
            f"{float(row['work_proxy']):.1f} | {float(row['bytes_proxy']):.1f} |"
        )
    if ko:
        lines += [
            "",
            "## 해석",
            "",
            "- Learned policy는 WPU correction loop의 budget decision을 학습 가능한 형태로 분리한다.",
            "- Clean setting에서는 hand adaptive rule과 비슷한 bounded K/objective를 목표로 한다.",
            "- Noisy 또는 weak anomaly shift에서 learned policy가 과소/과대 관측하면 이것이 다음 calibration 실패다.",
            "- Dense state copy는 여전히 raw accuracy 상한이지만 O(N) work/bytes를 지불한다.",
        ]
    else:
        lines += [
            "",
            "## Interpretation",
            "",
            "- The learned policy separates observation-budget choice as a trainable WPU correction-loop decision.",
            "- In the clean setting it should approach the hand adaptive rule with bounded K/objective.",
            "- Under noisy or weak anomaly shift, under- or over-observation is the next calibration failure.",
            "- Dense state copy remains the raw-accuracy upper bound but pays O(N) work/bytes.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
