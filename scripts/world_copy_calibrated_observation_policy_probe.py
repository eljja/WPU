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


@dataclass(frozen=True)
class Calibration:
    scale: float = 1.0
    offset: float = 0.0


class BudgetPolicy(nn.Module):
    def __init__(self, max_budget: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 24), nn.GELU(), nn.Linear(24, max_budget + 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe calibrated learned WPU observation-budget policy under anomaly shift.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--escape-rate", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--eval-shifts", type=str, nargs="+", default=["clean", "noisy_anomaly", "weak_anomaly"])
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--dual-omission", type=float, default=0.75)
    parser.add_argument("--contamination", type=int, default=128)
    parser.add_argument("--fixed-budget", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--cost-lambda", type=float, default=0.015)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_calibrated_observation_policy_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_calibrated_observation_policy_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_calibrated_observation_policy_probe_results.ko.md"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = train_policy(args, rng)
    calibrations = {
        shift: calibrate_shift(args, shift, random.Random(rng.randrange(2**31)))
        for shift in args.eval_shifts
    }

    rows: list[dict[str, object]] = []
    for shift in args.eval_shifts:
        for n in args.world_sizes:
            for escape_rate in args.escape_rate:
                for mode in (
                    "wpu-hand-adaptive",
                    "wpu-learned-observation",
                    "wpu-calibrated-learned-observation",
                    "wpu-fixed-observation",
                    "dense-state-copy",
                ):
                    trials = [
                        evaluate_scene(
                            make_scene(args, n, escape_rate, shift, random.Random(rng.randrange(2**31))),
                            mode=mode,
                            policy=policy,
                            calibration=calibrations[shift],
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
                        "calibration_scale": round(calibrations[shift].scale, 6),
                        "calibration_offset": round(calibrations[shift].offset, 6),
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
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for _ in range(args.train_samples):
        scene = make_scene(
            args,
            rng.choice(args.world_sizes),
            rng.choice(args.escape_rate),
            "clean",
            random.Random(rng.randrange(2**31)),
        )
        x_rows.append(scene_features(scene, args.max_budget, Calibration()))
        y_rows.append(rule_budget(scene, args.max_budget))
    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    for _ in range(args.train_steps):
        loss = F.cross_entropy(policy(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy.eval()


def calibrate_shift(args: argparse.Namespace, shift: str, rng: random.Random) -> Calibration:
    if shift == "clean":
        return Calibration()
    causal_values: list[float] = []
    background_values: list[float] = []
    for _ in range(args.calibration_samples):
        scene = make_scene(
            args,
            rng.choice(args.world_sizes),
            rng.choice(args.escape_rate),
            shift,
            random.Random(rng.randrange(2**31)),
        )
        causal_values.extend(scene.anomaly[oid] for oid in scene.hidden_unknown)
        background_values.extend(scene.anomaly[oid] for oid in scene.background[: min(128, len(scene.background))])
    if not causal_values:
        return Calibration()
    causal_mean = sum(causal_values) / len(causal_values)
    background_mean = sum(background_values) / max(len(background_values), 1)
    # Map shifted causal/background means toward the clean separation used by the policy.
    target_causal = 0.925
    target_background = 0.41
    denom = max(causal_mean - background_mean, 1e-6)
    scale = max(0.1, min(4.0, (target_causal - target_background) / denom))
    offset = target_causal - scale * causal_mean
    return Calibration(scale=scale, offset=offset)


def make_scene(args: argparse.Namespace, n: int, escape_rate: float, shift: str, rng: random.Random) -> Scene:
    causal = [f"causal_{i}" for i in range(args.k_ref)]
    background = [f"bg_{i}" for i in range(n - args.k_ref)]
    active: set[str] = set()
    relation_frontier: set[str] = set()
    neighbor_pool: set[str] = set()
    hidden_unknown: set[str] = set()
    for oid in causal:
        if rng.random() < args.dual_omission:
            if rng.random() < escape_rate:
                hidden_unknown.add(oid)
            else:
                neighbor_pool.add(oid)
            continue
        active.add(oid)
        if rng.random() >= 0.25:
            relation_frontier.add(oid)

    distractor_count = min(args.contamination, len(background))
    active.update(rng.sample(background, distractor_count))
    remaining_background = [oid for oid in background if oid not in active]
    neighbor_pool.update(rng.sample(remaining_background, min(args.contamination, len(remaining_background))))

    role = {oid: "affected" for oid in causal}
    confidence = {oid: rng.uniform(0.88, 1.0) for oid in causal}
    anomaly: dict[str, float] = {}
    for oid in hidden_unknown:
        anomaly[oid] = rng.uniform(0.65, 0.86) if shift == "weak_anomaly" else rng.uniform(0.85, 1.0)
    for oid in background:
        role[oid] = "affected" if rng.random() < 0.2 else "background"
        confidence[oid] = rng.uniform(0.35, 0.78)
        anomaly[oid] = rng.uniform(0.1, 0.92) if shift == "noisy_anomaly" else rng.uniform(0.1, 0.72)

    base_candidates = active | relation_frontier
    observed_support = sum(1 for oid in base_candidates if role[oid] == "affected" and confidence[oid] >= 0.85)
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
        support_deficit=max(args.k_ref - observed_support, 0),
    )


def evaluate_scene(
    scene: Scene,
    *,
    mode: str,
    policy: BudgetPolicy,
    calibration: Calibration,
    fixed_budget: int,
    max_budget: int,
    cost_lambda: float,
    horizon: int,
) -> dict[str, float]:
    if mode == "wpu-hand-adaptive":
        budget = rule_budget(scene, max_budget)
    elif mode == "wpu-learned-observation":
        budget = predict_budget(scene, policy, max_budget, Calibration())
    elif mode == "wpu-calibrated-learned-observation":
        budget = max(0, predict_budget(scene, policy, max_budget, calibration) - neighbor_support_credit(scene))
    elif mode == "wpu-fixed-observation":
        budget = fixed_budget if scene.support_deficit > 0 else 0
    elif mode == "dense-state-copy":
        budget = 0
    else:
        raise ValueError(mode)

    candidates = set(scene.active) | scene.relation_frontier | scene.neighbor_pool
    observed: set[str] = set()
    if budget > 0:
        observation_candidates = scene.hidden_unknown | set(scene.background[: min(128, len(scene.background))])
        def observation_score(oid: str) -> float:
            score = calibrated_anomaly(scene.anomaly.get(oid, 0.0), calibration)
            if mode == "wpu-calibrated-learned-observation":
                score *= 0.5 + 0.5 * scene.confidence.get(oid, 0.0)
            return score

        observed = set(
            sorted(observation_candidates, key=observation_score, reverse=True)[:budget]
        )
        candidates.update(observed)

    if mode == "dense-state-copy":
        selected = set(scene.causal)
    else:
        cap = 3 * scene.k if budget == 0 else 4 * scene.k
        selected = score_and_cap(candidates, scene.relation_frontier | scene.neighbor_pool | observed, scene.role, scene.confidence, cap)

    causal_set = set(scene.causal)
    false_selected = selected - causal_set
    recall = len(selected & causal_set) / scene.k
    observation_hit_rate = len(observed & scene.hidden_unknown) / max(len(scene.hidden_unknown), 1)
    missed_error = 0.0
    false_error = 0.0
    for step in range(horizon):
        force = 0.5 + 0.04 * step
        missed_error += sum(
            (force * (0.6 + 0.05 * (i % 3))) ** 2
            for i, oid in enumerate(scene.causal)
            if oid not in selected
        ) / scene.k
        false_error += sum((force * scene.confidence[oid] * 0.25) ** 2 for oid in false_selected) / scene.k
    mse = (missed_error + false_error) / horizon
    work = scene.n if mode == "dense-state-copy" else len(selected)
    return {
        "selected_k": float(len(selected)),
        "observation_budget": float(budget),
        "recall": recall,
        "mse": mse,
        "integrity": 1.0 / (1.0 + mse),
        "observation_hit_rate": observation_hit_rate,
        "objective": mse + cost_lambda * float(budget),
        "work": float(work),
        "bytes": float(work * 9 * 4),
    }


def predict_budget(scene: Scene, policy: BudgetPolicy, max_budget: int, calibration: Calibration) -> int:
    with torch.no_grad():
        logits = policy(torch.tensor([scene_features(scene, max_budget, calibration)], dtype=torch.float32))
    return int(logits.argmax(dim=-1).item())


def neighbor_support_credit(scene: Scene) -> int:
    return sum(
        1
        for oid in scene.neighbor_pool
        if scene.role.get(oid) == "affected" and scene.confidence.get(oid, 0.0) >= 0.85
    )


def rule_budget(scene: Scene, max_budget: int) -> int:
    anomaly_count = sum(1 for oid in scene.hidden_unknown if scene.anomaly.get(oid, 0.0) >= 0.85)
    return min(max_budget, scene.support_deficit, anomaly_count)


def scene_features(scene: Scene, max_budget: int, calibration: Calibration) -> list[float]:
    values = [calibrated_anomaly(scene.anomaly.get(oid, 0.0), calibration) for oid in scene.hidden_unknown | scene.neighbor_pool]
    max_anomaly = max(values, default=0.0)
    mean_top = sum(sorted(values, reverse=True)[:max_budget]) / max(max_budget, 1)
    anomaly_count = sum(1 for value in values if value >= 0.85)
    return [
        scene.support_deficit / max(scene.k, 1),
        anomaly_count / max(max_budget, 1),
        max_anomaly,
        mean_top,
        min(len(scene.neighbor_pool), 128) / 128.0,
    ]


def calibrated_anomaly(value: float, calibration: Calibration) -> float:
    return max(0.0, min(1.0, value * calibration.scale + calibration.offset))


def score_and_cap(candidates: set[str], relation_frontier: set[str], role: dict[str, str], confidence: dict[str, float], cap: int) -> set[str]:
    scored = sorted(candidates, key=lambda oid: (role[oid] == "affected", oid in relation_frontier, confidence[oid]), reverse=True)
    return set(scored[:cap])


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6)


def report(rows: list[dict[str, object]], source: Path, ko: bool) -> str:
    intro = (
        "이 probe는 shifted anomaly signal에서 learned observation-budget policy를 작은 calibration set으로 보정할 수 있는지 검증한다."
        if ko
        else "This probe tests whether a small calibration set can repair a learned observation-budget policy under shifted anomaly signals."
    )
    lines = [
        "# Calibrated Observation Budget Policy Probe",
        "",
        intro,
        "",
        f"Source CSV: `{source.as_posix()}`.",
        "",
        "| mode | shift | N | escape | scale | offset | mean K | budget | recall | MSE | objective | work | bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['shift']} | {row['total_n']} | {float(row['escape_rate']):.2f} | "
            f"{float(row['calibration_scale']):.3f} | {float(row['calibration_offset']):.3f} | "
            f"{float(row['mean_selected_k']):.3f} | {float(row['mean_observation_budget']):.3f} | "
            f"{float(row['causal_recall']):.3f} | {float(row['trajectory_mse']):.4f} | "
            f"{float(row['objective']):.4f} | {float(row['work_proxy']):.1f} | {float(row['bytes_proxy']):.1f} |"
        )
    if ko:
        lines += [
            "",
            "## 해석",
            "",
            "- Calibration은 anomaly score를 clean-policy 입력 분포로 맞추는 bounded correction이다.",
            "- Weak anomaly shift에서는 calibration이 under-observation을 줄일 수 있는지 검증한다.",
            "- Noisy anomaly shift에서는 false anomaly가 많아 calibration만으로는 과대 관측을 완전히 없애기 어렵다.",
            "- Dense state copy는 raw accuracy 상한이지만 O(N) work/bytes를 사용한다.",
        ]
    else:
        lines += [
            "",
            "## Interpretation",
            "",
            "- Calibration is a bounded correction that maps anomaly scores back toward the clean policy input distribution.",
            "- Under weak anomaly shift, the test checks whether calibration reduces under-observation.",
            "- Under noisy anomaly shift, many false anomalies mean calibration alone may not remove over-observation.",
            "- Dense state copy remains the raw-accuracy upper bound but uses O(N) work/bytes.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
