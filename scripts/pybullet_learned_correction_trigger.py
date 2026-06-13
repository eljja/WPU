from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn

from wpu.data.pybullet_cup import PyBulletCupDataset, PyBulletCupSample

from scripts.analyze_state_integrity import _integrity_score, _low_disruption_integrity_score
from scripts.pybullet_closed_loop_rollout import (
    _apply_predicted_delta,
    _collate_fn,
    _constraint_violations,
    _entropy,
    _mean,
    _move_batch,
    _project_sample_state,
    _selected_k,
    _state_delta_norm,
    _train_model,
)


DEFAULT_OUT = Path("docs/experiments/pybullet_learned_correction_trigger.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_learned_correction_trigger_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_learned_correction_trigger_results.ko.md")


class CorrectionTrigger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a learned low-frequency correction trigger for "
            "PyBullet WPU state-integrity rollouts."
        )
    )
    parser.add_argument("--model", default="wpu-cws-indexed-sparse")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[19, 23])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--eval-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delta-norm-penalty", type=float, default=0.0)
    parser.add_argument("--delta-target-norm-slack", type=float, default=0.5)
    parser.add_argument("--rollout-consistency-penalty", type=float, default=0.0)
    parser.add_argument("--rollout-consistency-slack", type=float, default=0.5)
    parser.add_argument("--state-validity-penalty", type=float, default=0.0)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-steps", type=int, default=300)
    parser.add_argument(
        "--trigger-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975],
    )
    parser.add_argument(
        "--top-rates",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
        help="Evaluate top-probability trigger budgets as correction-rate targets.",
    )
    parser.add_argument("--delta-clip", type=float, default=0.25)
    parser.add_argument("--finite-delta-clamp", type=float, default=1.0)
    parser.add_argument("--max-position-norm", type=float, default=25.0)
    parser.add_argument("--max-velocity-norm", type=float, default=25.0)
    parser.add_argument("--min-cup-z", type=float, default=-0.2)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.report_only:
        rows = _read_csv(args.out)
        args.out_md.write_text(_render_report(rows, args, korean=False), encoding="utf-8")
        args.out_ko_md.write_text(_render_report(rows, args, korean=True), encoding="utf-8")
        print(f"wrote={args.out_md}", flush=True)
        print(f"wrote={args.out_ko_md}", flush=True)
        return

    model_cache: dict[int, torch.nn.Module] = {}
    train_features: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    train_severities: list[torch.Tensor] = []
    for seed in args.train_seeds:
        print(f"collect train trigger examples seed={seed}", flush=True)
        model = _cached_model(model_cache, seed, args)
        features, labels, severities = _collect_trigger_examples(
            model,
            seed=seed,
            sample_seed=seed + 30_000,
            samples=args.train_samples,
            args=args,
        )
        train_features.append(features)
        train_labels.append(labels)
        train_severities.append(severities)

    features = torch.cat(train_features, dim=0)
    labels = torch.cat(train_labels, dim=0)
    severities = torch.cat(train_severities, dim=0)
    gate_bundle = _train_gate(features, labels, severities, args)
    _fit_top_rate_thresholds(gate_bundle, features, args.top_rates)

    rows: list[dict[str, object]] = []
    train_need_rate = float(labels.float().mean().item())
    for seed in args.eval_seeds:
        print(f"eval learned correction trigger seed={seed}", flush=True)
        model = _cached_model(model_cache, seed, args)
        rows.extend(_evaluate_seed(model, gate_bundle, seed, args, train_need_rate))
        _write_csv(args.out, rows)

    summary = _summary(rows)
    rows.extend(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out}", flush=True)
    print(f"wrote={args.out_md}", flush=True)
    print(f"wrote={args.out_ko_md}", flush=True)


def _cached_model(cache: dict[int, torch.nn.Module], seed: int, args: argparse.Namespace) -> torch.nn.Module:
    if seed not in cache:
        cache[seed] = _train_model(args.model, seed, args)
    return cache[seed]


def _collect_trigger_examples(
    model: torch.nn.Module,
    *,
    seed: int,
    sample_seed: int,
    samples: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = _rollout_steps(
        model,
        seed=seed,
        sample_seed=sample_seed,
        samples=samples,
        args=args,
        policy=("no_correction", 0.0),
        gate_bundle=None,
        collect_examples=True,
    )
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([int(row["needs_correction"]) for row in rows], dtype=torch.float32)
    severities = torch.tensor([float(row["violation_delta"]) for row in rows], dtype=torch.float32).clamp_min(0.0)
    return features, labels, severities


def _train_gate(
    features: torch.Tensor,
    labels: torch.Tensor,
    severities: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(10_003)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    normalized = (features - mean) / std
    gate = CorrectionTrigger(normalized.size(1), args.gate_hidden_dim)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    pos_weight = ((1.0 - labels.mean()) / labels.mean().clamp_min(1e-6)).clamp(min=0.25, max=8.0)
    sample_weight = 1.0 + severities.clamp(max=4.0)
    gate.train()
    for _ in range(args.gate_steps):
        logits = gate(normalized)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        loss = (bce * sample_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    with torch.no_grad():
        probs = torch.sigmoid(gate(normalized))
        predictions = probs >= 0.5
    return {
        "model": gate,
        "mean": mean,
        "std": std,
        "train_need_rate": float(labels.mean().item()),
        "train_accuracy": float((predictions.float() == labels).float().mean().item()),
        "train_positive_rate": float(predictions.float().mean().item()),
        "train_loss": float(F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), labels).item()),
    }


def _evaluate_seed(
    model: torch.nn.Module,
    gate_bundle: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    train_need_rate: float,
) -> list[dict[str, object]]:
    policies: list[tuple[str, float]] = [("no_correction", 0.0), ("always_selective", 0.0)]
    policies.extend((f"learned_threshold_{threshold:g}", threshold) for threshold in args.trigger_thresholds)
    policies.extend((f"learned_top_rate_{rate:g}", rate) for rate in args.top_rates)
    rows = []
    for policy in policies:
        step_rows = _rollout_steps(
            model,
            seed=seed,
            sample_seed=seed + 40_000,
            samples=args.eval_samples,
            args=args,
            policy=policy,
            gate_bundle=gate_bundle,
            collect_examples=False,
        )
        rows.append(_metric_row(step_rows, policy[0], policy[1], seed, args, train_need_rate, gate_bundle))
    return rows


def _rollout_steps(
    model: torch.nn.Module,
    *,
    seed: int,
    sample_seed: int,
    samples: int,
    args: argparse.Namespace,
    policy: tuple[str, float],
    gate_bundle: dict[str, object] | None,
    collect_examples: bool,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    dataset = PyBulletCupDataset(
        size=samples,
        seed=sample_seed,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    raw_step_rows: list[dict[str, object]] = []
    model.eval()
    previous_branch_by_sample: dict[int, int] = {}
    with torch.no_grad():
        for sample_index in range(samples):
            sample = dataset[sample_index]
            current = PyBulletCupSample(
                state=sample.state,
                event=sample.event,
                target_object_delta=sample.target_object_delta,
                branch_label=sample.branch_label,
                causal_working_set_size=sample.causal_working_set_size,
                simulator_metadata=dict(sample.simulator_metadata),
            )
            for step_index in range(args.horizon):
                before_state_json = current.state.to_json()
                before_state = type(current.state).from_json(before_state_json)  # type: ignore[arg-type]
                before_event_time = current.event.time
                before_violations = _constraint_violations(current)
                batch, _, _, _ = _collate_fn(args, args.model)([current])
                batch = _move_batch(batch, device)
                prediction = model(batch, num_branches=3, route_branches=3)
                probabilities = prediction.branch_probabilities[0]
                entropy = _entropy(probabilities)
                branch = int(probabilities.argmax().detach().cpu().item())
                previous_branch = previous_branch_by_sample.get(sample_index)
                branch_flipped = int(previous_branch is not None and branch != previous_branch)
                previous_branch_by_sample[sample_index] = branch
                delta = prediction.object_delta[0].detach().cpu()
                raw_delta_norm = float(delta.norm().item())
                object_ids = batch.object_ids[0] if batch.object_ids is not None else list(current.state.objects)
                applied_delta_norm = _apply_predicted_delta(
                    current,
                    object_ids,
                    delta,
                    time_step=sample.event.time,
                    delta_clip=args.delta_clip,
                    integrity_projection=False,
                    finite_delta_clamp=args.finite_delta_clamp,
                    max_position_norm=args.max_position_norm,
                    max_velocity_norm=args.max_velocity_norm,
                    min_cup_z=args.min_cup_z,
                )
                after_violations = _constraint_violations(current)
                features = _trigger_features(
                    current,
                    step_index=step_index,
                    horizon=args.horizon,
                    entropy=entropy,
                    probabilities=probabilities,
                    raw_delta_norm=raw_delta_norm,
                    applied_delta_norm=applied_delta_norm,
                    selected_k=_selected_k(model, batch),
                    before_violations=before_violations,
                    after_violations=after_violations,
                    branch=branch,
                    branch_flipped=branch_flipped,
                    background_objects=args.background_objects,
                )
                needs_correction = int(after_violations > before_violations)
                trigger = _policy_triggers(policy, features, gate_bundle) if not collect_examples else False
                corrected_objects = 0
                if trigger:
                    corrected_objects = _project_sample_state(
                        current,
                        max_position_norm=args.max_position_norm,
                        max_velocity_norm=args.max_velocity_norm,
                        min_cup_z=args.min_cup_z,
                        selective=True,
                    )
                    after_violations = _constraint_violations(current)
                    applied_delta_norm = _state_delta_norm(before_state, current.state)
                raw_step_rows.append(
                    {
                        "seed": seed,
                        "sample_index": sample_index,
                        "step_index": step_index,
                        "features": features,
                        "needs_correction": needs_correction,
                        "violation_delta": max(0, after_violations - before_violations)
                        if not trigger
                        else max(0, needs_correction),
                        "constraint_violations": after_violations,
                        "delta_norm": applied_delta_norm,
                        "branch_flipped": branch_flipped,
                        "triggered": int(trigger),
                        "corrected_objects": corrected_objects,
                        "object_count": len(current.state.objects),
                    }
                )
                if collect_examples:
                    current.state = type(current.state).from_json(before_state_json)  # type: ignore[arg-type]
                    current.event.time = before_event_time
                    _apply_predicted_delta(
                        current,
                        object_ids,
                        delta,
                        time_step=sample.event.time,
                        delta_clip=args.delta_clip,
                        integrity_projection=False,
                        finite_delta_clamp=args.finite_delta_clamp,
                        max_position_norm=args.max_position_norm,
                        max_velocity_norm=args.max_velocity_norm,
                        min_cup_z=args.min_cup_z,
                    )
    return raw_step_rows


def _trigger_features(
    sample: PyBulletCupSample,
    *,
    step_index: int,
    horizon: int,
    entropy: float,
    probabilities: torch.Tensor,
    raw_delta_norm: float,
    applied_delta_norm: float,
    selected_k: float,
    before_violations: int,
    after_violations: int,
    branch: int,
    branch_flipped: int,
    background_objects: int,
) -> list[float]:
    probs = probabilities.detach().cpu()
    sorted_probs = torch.sort(probs, descending=True).values
    cup = sample.state.objects.get("cup_001")
    cup_z = 0.0
    cup_speed = 0.0
    if cup is not None:
        position = cup.attributes.get("position", [0.0, 0.0, 0.0])
        velocity = cup.attributes.get("velocity", [0.0, 0.0, 0.0])
        if isinstance(position, list) and len(position) >= 3:
            cup_z = float(position[2])
        if isinstance(velocity, list):
            cup_speed = math.sqrt(sum(float(item) ** 2 for item in velocity[:3]))
    object_count = max(len(sample.state.objects), 1)
    return [
        float(step_index) / max(float(horizon), 1.0),
        entropy,
        float(sorted_probs[0].item()),
        float((sorted_probs[0] - sorted_probs[1]).item()) if sorted_probs.numel() > 1 else 0.0,
        math.log1p(max(raw_delta_norm, 0.0)),
        math.log1p(max(applied_delta_norm, 0.0)),
        float(selected_k) / max(float(object_count), 1.0),
        float(before_violations),
        float(after_violations),
        float(after_violations - before_violations),
        cup_z,
        math.log1p(max(cup_speed, 0.0)),
        float(branch == 0),
        float(branch == 1),
        float(branch == 2),
        float(branch_flipped),
        float(background_objects),
    ]


def _policy_triggers(
    policy: tuple[str, float],
    features: list[float],
    gate_bundle: dict[str, object] | None,
) -> bool:
    name, value = policy
    if name == "no_correction":
        return False
    if name == "always_selective":
        return True
    if gate_bundle is None:
        return False
    probability = _predict_probability(gate_bundle, features)
    if name.startswith("learned_threshold_"):
        return probability >= value
    if name.startswith("learned_top_rate_"):
        threshold = float(gate_bundle.get(f"top_threshold_{value:g}", 1.0))
        return probability >= threshold
    return False


def _predict_probability(gate_bundle: dict[str, object], features: list[float]) -> float:
    gate = gate_bundle["model"]
    mean = gate_bundle["mean"]
    std = gate_bundle["std"]
    assert isinstance(gate, CorrectionTrigger)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    x = (torch.tensor(features, dtype=torch.float32).unsqueeze(0) - mean) / std
    with torch.no_grad():
        return float(torch.sigmoid(gate(x)).item())


def _fit_top_rate_thresholds(gate_bundle: dict[str, object], features: torch.Tensor, rates: list[float]) -> None:
    gate = gate_bundle["model"]
    mean = gate_bundle["mean"]
    std = gate_bundle["std"]
    assert isinstance(gate, CorrectionTrigger)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    with torch.no_grad():
        probs = torch.sigmoid(gate((features - mean) / std))
    for rate in rates:
        quantile = max(0.0, min(1.0, 1.0 - float(rate)))
        gate_bundle[f"top_threshold_{rate:g}"] = float(torch.quantile(probs, quantile).item())


def _metric_row(
    step_rows: list[dict[str, object]],
    policy: str,
    policy_value: float,
    seed: int,
    args: argparse.Namespace,
    train_need_rate: float,
    gate_bundle: dict[str, object],
) -> dict[str, object]:
    violations = _mean([float(row["constraint_violations"]) for row in step_rows])
    delta_norm = _mean([float(row["delta_norm"]) for row in step_rows])
    flip_rate = sum(int(row["branch_flipped"]) for row in step_rows) / max(len(step_rows) - args.eval_samples, 1)
    correction_rate = sum(int(row["triggered"]) for row in step_rows) / max(len(step_rows), 1)
    corrected_object_fraction = sum(float(row["corrected_objects"]) for row in step_rows) / max(
        sum(float(row["object_count"]) for row in step_rows if int(row["triggered"])), 1.0
    )
    integrity = _integrity_score(violations, delta_norm, flip_rate)
    low_disruption = _low_disruption_integrity_score(
        integrity,
        rejection_rate=0.0,
        correction_rate=correction_rate,
        corrected_object_fraction=corrected_object_fraction,
        rollback_rate=0.0,
        escalation_rate=0.0,
    )
    return {
        "row_type": "seed",
        "policy": policy,
        "policy_value": round(policy_value, 6),
        "model": args.model,
        "seed": seed,
        "horizon": args.horizon,
        "background_objects": args.background_objects,
        "samples": args.eval_samples,
        "train_need_rate": round(train_need_rate, 6),
        "gate_train_accuracy": round(float(gate_bundle["train_accuracy"]), 6),
        "gate_train_positive_rate": round(float(gate_bundle["train_positive_rate"]), 6),
        "constraint_violations_per_step": round(violations, 6),
        "delta_norm_mean": round(delta_norm, 6),
        "branch_flip_rate": round(flip_rate, 6),
        "correction_rate": round(correction_rate, 6),
        "corrected_object_fraction": round(corrected_object_fraction, 6),
        "state_integrity_score": round(integrity, 6),
        "low_disruption_integrity_score": round(low_disruption, 6),
        "meets_joint_target": int(integrity >= 0.8 and correction_rate <= 0.25),
    }


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    policies = sorted({str(row["policy"]) for row in rows if row["row_type"] == "seed"})
    out = []
    for policy in policies:
        group = [row for row in rows if row["row_type"] == "seed" and row["policy"] == policy]
        out.append(
            {
                "row_type": "summary",
                "policy": policy,
                "policy_value": _mean([float(row["policy_value"]) for row in group]),
                "model": group[0]["model"],
                "seed": "all",
                "horizon": group[0]["horizon"],
                "background_objects": group[0]["background_objects"],
                "samples": sum(int(row["samples"]) for row in group),
                "train_need_rate": _mean([float(row["train_need_rate"]) for row in group]),
                "gate_train_accuracy": _mean([float(row["gate_train_accuracy"]) for row in group]),
                "gate_train_positive_rate": _mean([float(row["gate_train_positive_rate"]) for row in group]),
                "constraint_violations_per_step": _mean([float(row["constraint_violations_per_step"]) for row in group]),
                "delta_norm_mean": _mean([float(row["delta_norm_mean"]) for row in group]),
                "branch_flip_rate": _mean([float(row["branch_flip_rate"]) for row in group]),
                "correction_rate": _mean([float(row["correction_rate"]) for row in group]),
                "corrected_object_fraction": _mean([float(row["corrected_object_fraction"]) for row in group]),
                "state_integrity_score": _mean([float(row["state_integrity_score"]) for row in group]),
                "low_disruption_integrity_score": _mean([float(row["low_disruption_integrity_score"]) for row in group]),
                "meets_joint_target": int(
                    _mean([float(row["state_integrity_score"]) for row in group]) >= 0.8
                    and _mean([float(row["correction_rate"]) for row in group]) <= 0.25
                ),
            }
        )
    return out


def _render_report(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    summary = [row for row in rows if row["row_type"] == "summary"]
    if not summary:
        summary = _summary([row for row in rows if row["row_type"] == "seed"])
    best_joint = [row for row in summary if int(float(row["meets_joint_target"])) == 1]
    best_integrity = max(summary, key=lambda row: float(row["state_integrity_score"]))
    best_low_correction = max(
        [row for row in summary if float(row["correction_rate"]) <= 0.25],
        key=lambda row: float(row["state_integrity_score"]),
        default=None,
    )
    learned_rows = [row for row in summary if str(row["policy"]).startswith("learned_")]
    best_learned = (
        max(learned_rows, key=lambda row: float(row["state_integrity_score"]))
        if learned_rows
        else None
    )
    best_low_disruption = max(summary, key=lambda row: float(row["low_disruption_integrity_score"]))
    if korean:
        title = "# PyBullet Learned Correction Trigger 결과"
        intro = (
            "이 실험은 P2 correction-trigger 병목을 hand-coded threshold가 아니라 held-out seed로 "
            "전이되는 작은 MLP trigger로 검사한다. Trigger는 sparse delta 적용 후의 confidence, "
            "delta norm, violation 변화, cup state를 보고 selective correction을 실행할지 결정한다."
        )
        conclusion = (
            f"Joint target(integrity >= 0.8, correction_rate <= 0.25)을 만족한 summary policy는 "
            f"`{len(best_joint)}`개다. 최고 integrity는 `{float(best_integrity['state_integrity_score']):.6f}` "
            f"(`{best_integrity['policy']}`)이고 correction rate는 `{float(best_integrity['correction_rate']):.6f}`이다. "
            + (
                f"Correction rate <= 0.25 조건의 최고 integrity는 `{float(best_low_correction['state_integrity_score']):.6f}` "
                f"(`{best_low_correction['policy']}`)이다. "
                if best_low_correction is not None
                else "Correction rate <= 0.25 조건을 만족한 policy가 없다. "
            )
            + (
                f"최고 learned trigger integrity는 `{float(best_learned['state_integrity_score']):.6f}` "
                f"(`{best_learned['policy']}`)이고 correction rate는 `{float(best_learned['correction_rate']):.6f}`이다. "
                if best_learned is not None
                else ""
            )
            + f"최고 low-disruption score는 `{float(best_low_disruption['low_disruption_integrity_score']):.6f}` "
            f"(`{best_low_disruption['policy']}`)이다."
        )
        interpretation = [
            "Hard-seed learned trigger는 P2를 해결하지 못했다. 높은 integrity는 correction을 대부분 실행할 때만 유지된다.",
            "항상 또는 고빈도 selective correction은 applied state를 보호하지만, 이는 stable raw dynamics가 아니라 memory-safety layer다.",
            "다음 단계는 trigger threshold 확장이 아니라 transition loss, state-validity loss, correction objective를 묶은 안정화 학습이다.",
        ]
        section = "## 해석"
    else:
        title = "# PyBullet Learned Correction Trigger Results"
        intro = (
            "This experiment tests the remaining P2 correction-trigger bottleneck with a "
            "small MLP trigger that transfers across held-out seeds rather than another "
            "hand-coded threshold. The trigger observes sparse-delta confidence, delta "
            "norm, violation change, and cup state before deciding whether to run "
            "selective correction."
        )
        conclusion = (
            f"Summary policies meeting the joint target (integrity >= 0.8 and correction_rate <= 0.25): "
            f"`{len(best_joint)}`. The best integrity is `{float(best_integrity['state_integrity_score']):.6f}` "
            f"(`{best_integrity['policy']}`) at correction rate `{float(best_integrity['correction_rate']):.6f}`. "
            + (
                f"Under correction rate <= 0.25, the best integrity is `{float(best_low_correction['state_integrity_score']):.6f}` "
                f"(`{best_low_correction['policy']}`). "
                if best_low_correction is not None
                else "No policy satisfies correction rate <= 0.25. "
            )
            + (
                f"The best learned-trigger integrity is `{float(best_learned['state_integrity_score']):.6f}` "
                f"(`{best_learned['policy']}`) at correction rate `{float(best_learned['correction_rate']):.6f}`. "
                if best_learned is not None
                else ""
            )
            + f"The best low-disruption score is `{float(best_low_disruption['low_disruption_integrity_score']):.6f}` "
            f"(`{best_low_disruption['policy']}`)."
        )
        interpretation = [
            "The hard-seed learned trigger does not solve P2: high integrity is preserved only when most sparse updates are corrected.",
            "Always-on or high-frequency selective correction protects applied state, but this is a memory-safety layer rather than stable raw dynamics.",
            "The next step is stable transition training with transition losses, state-validity losses, and correction objectives, not more threshold tuning.",
        ]
        section = "## Interpretation"
    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{args.out.as_posix()}`",
        "",
        conclusion,
        "",
        "| policy | integrity | low-disruption | violations/step | correction | corrected objects | joint target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (float(item["correction_rate"]), str(item["policy"]))):
        lines.append(
            f"| `{row['policy']}` | {float(row['state_integrity_score']):.6f} | "
            f"{float(row['low_disruption_integrity_score']):.6f} | "
            f"{float(row['constraint_violations_per_step']):.6f} | "
            f"{float(row['correction_rate']):.6f} | "
            f"{float(row['corrected_object_fraction']):.6f} | {int(float(row['meets_joint_target']))} |"
        )
    lines.extend(["", section, ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
