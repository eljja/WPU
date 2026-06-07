from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.pybullet_cup import (
    PyBulletCupDataset,
    PyBulletCupSample,
    collate_indexed_pybullet_cup_samples,
    collate_pybullet_cup_samples,
)
from wpu.models.batch import OBJECT_FEATURE_DIM, StateGraphBatch
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.factory import create_model


DEFAULT_MODELS = [
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "graph-transformer",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop state rollout diagnostic on PyBullet-derived WorldState.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--delta-clip", type=float, default=0.0)
    parser.add_argument("--finite-delta-clamp", type=float, default=0.0)
    parser.add_argument("--delta-norm-penalty", type=float, default=0.0)
    parser.add_argument("--delta-target-norm-slack", type=float, default=0.5)
    parser.add_argument("--rollout-consistency-penalty", type=float, default=0.0)
    parser.add_argument("--rollout-consistency-slack", type=float, default=0.5)
    parser.add_argument("--state-validity-penalty", type=float, default=0.0)
    parser.add_argument("--unsafe-delta-reject-norm", type=float, default=0.0)
    parser.add_argument("--correct-on-violation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollback-on-violation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--escalation-model", default="")
    parser.add_argument("--integrity-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-position-norm", type=float, default=25.0)
    parser.add_argument("--max-velocity-norm", type=float, default=25.0)
    parser.add_argument("--min-cup-z", type=float, default=-0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_closed_loop_rollout.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    model_cache: dict[tuple[str, int], torch.nn.Module] = {}
    for model_name in args.models:
        for seed in args.seeds:
            print(f"train model={model_name} seed={seed}", flush=True)
            model = _cached_model(model_cache, model_name, seed, args)
            fallback_model = None
            if args.escalation_model and args.escalation_model != model_name:
                print(f"train escalation_model={args.escalation_model} seed={seed}", flush=True)
                fallback_model = _cached_model(model_cache, args.escalation_model, seed, args)
            for horizon in args.horizons:
                print(f"rollout model={model_name} seed={seed} horizon={horizon}", flush=True)
                rows.append(
                    _rollout_condition(
                        model,
                        model_name,
                        seed,
                        horizon,
                        args,
                        fallback_model=fallback_model,
                        fallback_model_name=args.escalation_model if fallback_model is not None else "",
                    )
                )
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _cached_model(
    cache: dict[tuple[str, int], torch.nn.Module],
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> torch.nn.Module:
    key = (model_name, seed)
    if key not in cache:
        cache[key] = _train_model(model_name, seed, args)
    return cache[key]


def _train_model(model_name: str, seed: int, args: argparse.Namespace) -> torch.nn.Module:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataset = PyBulletCupDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    class_weights = _class_weights(dataset).to(device) if args.class_weights else None
    model.train()
    for step, (batch, target_delta, labels, _) in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        prediction = model(batch, num_branches=3, route_branches=3)
        loss = F.cross_entropy(prediction.branch_logits, labels, weight=class_weights)
        loss = loss + 0.1 * F.mse_loss(prediction.object_delta, target_delta)
        if args.delta_norm_penalty > 0.0:
            loss = loss + args.delta_norm_penalty * _delta_norm_excess_loss(
                prediction.object_delta,
                target_delta,
                slack=args.delta_target_norm_slack,
            )
        if args.rollout_consistency_penalty > 0.0:
            loss = loss + args.rollout_consistency_penalty * _rollout_consistency_loss(
                model,
                batch,
                prediction.object_delta,
                target_delta,
                slack=args.rollout_consistency_slack,
            )
        if args.state_validity_penalty > 0.0:
            loss = loss + args.state_validity_penalty * _state_validity_loss(
                batch,
                prediction.object_delta,
                max_position_norm=args.max_position_norm,
                max_velocity_norm=args.max_velocity_norm,
                min_cup_z=args.min_cup_z,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= args.steps:
            break
    return model


def _rollout_condition(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    horizon: int,
    args: argparse.Namespace,
    *,
    fallback_model: torch.nn.Module | None = None,
    fallback_model_name: str = "",
) -> dict[str, object]:
    device = torch.device(args.device)
    dataset = PyBulletCupDataset(
        size=args.samples,
        seed=seed + 20_000,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    samples = [dataset[index] for index in range(args.samples)]
    model.eval()
    flip_count = 0
    total_transitions = 0
    violation_count = 0
    entropy_values: list[float] = []
    delta_norm_values: list[float] = []
    raw_delta_norm_values: list[float] = []
    rejected_delta_count = 0
    correction_count = 0
    rollback_count = 0
    escalation_count = 0
    escalation_success_count = 0
    total_delta_count = 0
    selected_k_values: list[float] = []
    final_branch_counts: Counter[int] = Counter()
    with torch.no_grad():
        for sample in samples:
            current = PyBulletCupSample(
                state=sample.state,
                event=sample.event,
                target_object_delta=sample.target_object_delta,
                branch_label=sample.branch_label,
                causal_working_set_size=sample.causal_working_set_size,
                simulator_metadata=dict(sample.simulator_metadata),
            )
            previous_branch: int | None = None
            for step_index in range(horizon):
                batch, _, _, _ = _collate_fn(args, model_name)([current])
                batch = _move_batch(batch, device)
                prediction = model(batch, num_branches=3, route_branches=3)
                probabilities = prediction.branch_probabilities[0]
                branch = int(probabilities.argmax().detach().cpu().item())
                entropy_values.append(_entropy(probabilities))
                if previous_branch is not None:
                    flip_count += int(branch != previous_branch)
                    total_transitions += 1
                previous_branch = branch
                final_branch_counts[branch] += int(step_index == horizon - 1)
                selected_k_values.append(_selected_k(model, batch))
                delta = prediction.object_delta[0].detach().cpu()
                raw_delta_norm = float(delta.norm().item())
                raw_delta_norm_values.append(raw_delta_norm)
                total_delta_count += 1
                if args.unsafe_delta_reject_norm > 0.0 and raw_delta_norm > args.unsafe_delta_reject_norm:
                    delta = torch.zeros_like(delta)
                    rejected_delta_count += 1
                object_ids = batch.object_ids[0] if batch.object_ids is not None else list(current.state.objects)
                needs_memory_guard = args.rollback_on_violation or args.correct_on_violation
                before_state = current.state.to_json() if needs_memory_guard else None
                before_event_time = current.event.time if needs_memory_guard else 0.0
                before_violations = _constraint_violations(current) if needs_memory_guard else 0
                applied_delta_norm = _apply_predicted_delta(
                    current,
                    object_ids,
                    delta,
                    time_step=sample.event.time,
                    delta_clip=args.delta_clip,
                    integrity_projection=args.integrity_projection,
                    finite_delta_clamp=args.finite_delta_clamp,
                    max_position_norm=args.max_position_norm,
                    max_velocity_norm=args.max_velocity_norm,
                    min_cup_z=args.min_cup_z,
                )
                if needs_memory_guard:
                    before_world_state = type(current.state).from_json(before_state)  # type: ignore[arg-type]
                    after_violations = _constraint_violations(current)
                    if fallback_model is not None and after_violations > before_violations:
                        current.state = type(current.state).from_json(before_state)  # type: ignore[arg-type]
                        current.event.time = before_event_time
                        fallback_batch, _, _, _ = _collate_fn(args, fallback_model_name)([current])
                        fallback_batch = _move_batch(fallback_batch, device)
                        fallback_prediction = fallback_model(fallback_batch, num_branches=3, route_branches=3)
                        fallback_delta = fallback_prediction.object_delta[0].detach().cpu()
                        fallback_object_ids = (
                            fallback_batch.object_ids[0]
                            if fallback_batch.object_ids is not None
                            else list(current.state.objects)
                        )
                        applied_delta_norm = _apply_predicted_delta(
                            current,
                            fallback_object_ids,
                            fallback_delta,
                            time_step=sample.event.time,
                            delta_clip=args.delta_clip,
                            integrity_projection=args.integrity_projection,
                            finite_delta_clamp=args.finite_delta_clamp,
                            max_position_norm=args.max_position_norm,
                            max_velocity_norm=args.max_velocity_norm,
                            min_cup_z=args.min_cup_z,
                        )
                        escalation_count += 1
                        after_violations = _constraint_violations(current)
                        escalation_success_count += int(after_violations <= before_violations)
                    if args.correct_on_violation and after_violations > before_violations:
                        _project_sample_state(
                            current,
                            max_position_norm=args.max_position_norm,
                            max_velocity_norm=args.max_velocity_norm,
                            min_cup_z=args.min_cup_z,
                        )
                        correction_count += 1
                        after_violations = _constraint_violations(current)
                        applied_delta_norm = _state_delta_norm(before_world_state, current.state)
                    if args.rollback_on_violation and after_violations > before_violations:
                        current.state = before_world_state
                        current.event.time = before_event_time
                        rollback_count += 1
                        applied_delta_norm = 0.0
                delta_norm_values.append(applied_delta_norm)
                violation_count += _constraint_violations(current)
    model.train()
    return {
        "model": model_name,
        "seed": seed,
        "horizon": horizon,
        "background_objects": args.background_objects,
        "samples": args.samples,
        "branch_flip_rate": round(flip_count / max(total_transitions, 1), 6),
        "constraint_violations_per_step": round(violation_count / max(args.samples * horizon, 1), 6),
        "branch_entropy_mean": round(_mean(entropy_values), 6),
        "delta_norm_mean": round(_mean(delta_norm_values), 6),
        "raw_delta_norm_mean": round(_mean(raw_delta_norm_values), 6),
        "unsafe_delta_rejection_rate": round(rejected_delta_count / max(total_delta_count, 1), 6),
        "correction_rate": round(correction_count / max(total_delta_count, 1), 6),
        "rollback_rate": round(rollback_count / max(total_delta_count, 1), 6),
        "escalation_rate": round(escalation_count / max(total_delta_count, 1), 6),
        "escalation_success_rate": round(escalation_success_count / max(escalation_count, 1), 6),
        "escalation_model": fallback_model_name,
        "selected_k_mean": round(_mean(selected_k_values), 6),
        "final_majority_branch_ratio": round(max(final_branch_counts.values(), default=0) / max(args.samples, 1), 6),
        "delta_clip": args.delta_clip,
        "finite_delta_clamp": args.finite_delta_clamp,
        "delta_norm_penalty": args.delta_norm_penalty,
        "delta_target_norm_slack": args.delta_target_norm_slack,
        "state_validity_penalty": args.state_validity_penalty,
        "unsafe_delta_reject_norm": args.unsafe_delta_reject_norm,
        "correct_on_violation": bool(args.correct_on_violation),
        "rollback_on_violation": bool(args.rollback_on_violation),
        "integrity_projection": bool(args.integrity_projection),
    }


def _delta_norm_excess_loss(prediction_delta: torch.Tensor, target_delta: torch.Tensor, *, slack: float) -> torch.Tensor:
    prediction_norm = prediction_delta[..., 1:7].norm(dim=-1)
    target_norm = target_delta[..., 1:7].norm(dim=-1)
    return F.relu(prediction_norm - target_norm - slack).pow(2).mean()


def _rollout_consistency_loss(
    model: torch.nn.Module,
    batch: StateGraphBatch,
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    slack: float,
) -> torch.Tensor:
    next_features = batch.object_features.clone()
    next_features[..., 1:7] = next_features[..., 1:7] + prediction_delta[..., 1:7]
    next_batch = StateGraphBatch(
        object_features=next_features,
        relation_indices=batch.relation_indices,
        relation_features=batch.relation_features,
        event_features=batch.event_features,
        object_mask=batch.object_mask,
        relation_mask=batch.relation_mask,
        target_indices=batch.target_indices,
        time_features=batch.time_features,
        scheduler_metrics=batch.scheduler_metrics,
        object_ids=batch.object_ids,
    )
    second_prediction = model(next_batch, num_branches=3, route_branches=3)
    second_norm = second_prediction.object_delta[..., 1:7].norm(dim=-1)
    target_norm = target_delta[..., 1:7].norm(dim=-1).detach()
    return F.relu(second_norm - target_norm - slack).pow(2).mean()


def _state_validity_loss(
    batch: StateGraphBatch,
    prediction_delta: torch.Tensor,
    *,
    max_position_norm: float,
    max_velocity_norm: float,
    min_cup_z: float,
) -> torch.Tensor:
    next_features = batch.object_features + prediction_delta
    object_mask = batch.object_mask.float()
    position_norm = next_features[..., 1:4].norm(dim=-1)
    velocity_norm = next_features[..., 4:7].norm(dim=-1)
    position_excess = F.relu(position_norm - max_position_norm).pow(2) * object_mask
    velocity_excess = F.relu(velocity_norm - max_velocity_norm).pow(2) * object_mask
    target_index = batch.target_indices.clamp(min=0, max=next_features.size(1) - 1)
    batch_index = torch.arange(next_features.size(0), device=next_features.device)
    target_z = next_features[batch_index, target_index, 3]
    cup_floor = F.relu(min_cup_z - target_z).pow(2)
    return position_excess.mean() + velocity_excess.mean() + cup_floor.mean()


def _apply_predicted_delta(
    sample: PyBulletCupSample,
    object_ids: list[str],
    delta: torch.Tensor,
    *,
    time_step: float,
    delta_clip: float,
    integrity_projection: bool,
    finite_delta_clamp: float,
    max_position_norm: float,
    max_velocity_norm: float,
    min_cup_z: float,
) -> float:
    applied_sq_norm = 0.0
    for index, object_id in enumerate(object_ids):
        if object_id not in sample.state.objects or index >= delta.size(0):
            continue
        obj = sample.state.objects[object_id]
        position_delta = _clip_vector(delta[index, 1:4], delta_clip, finite_delta_clamp).tolist()
        velocity_delta = _clip_vector(delta[index, 4:7], delta_clip, finite_delta_clamp).tolist()
        position = obj.attributes.get("position", [0.0, 0.0, 0.0])
        velocity = obj.attributes.get("velocity", [0.0, 0.0, 0.0])
        if isinstance(position, list) and len(position) >= 3:
            projected_position = [float(position[axis]) + float(position_delta[axis]) for axis in range(3)]
            if integrity_projection:
                projected_position = _project_vector(projected_position, max_position_norm)
                if object_id == "cup_001":
                    projected_position[2] = max(float(projected_position[2]), min_cup_z)
            obj.attributes["position"] = projected_position
            applied_sq_norm += sum(float(position_delta[axis]) ** 2 for axis in range(3))
        if isinstance(velocity, list) and len(velocity) >= 3:
            projected_velocity = [float(velocity[axis]) + float(velocity_delta[axis]) for axis in range(3)]
            if integrity_projection:
                projected_velocity = _project_vector(projected_velocity, max_velocity_norm)
            obj.attributes["velocity"] = projected_velocity
            applied_sq_norm += sum(float(velocity_delta[axis]) ** 2 for axis in range(3))
    sample.state.time += time_step
    sample.event.time = sample.state.time + time_step
    return math.sqrt(applied_sq_norm)


def _constraint_violations(sample: PyBulletCupSample) -> int:
    violations = 0
    for obj in sample.state.objects.values():
        position = obj.attributes.get("position", [0.0, 0.0, 0.0])
        velocity = obj.attributes.get("velocity", [0.0, 0.0, 0.0])
        if not _finite_vector(position) or not _finite_vector(velocity):
            violations += 1
            continue
        if _norm(position) > 25.0 or _norm(velocity) > 25.0:
            violations += 1
    cup = sample.state.objects.get("cup_001")
    if cup is not None:
        position = cup.attributes.get("position", [0.0, 0.0, 0.0])
        if isinstance(position, list) and len(position) >= 3 and float(position[2]) < -0.2:
            violations += 1
    return violations


def _project_sample_state(
    sample: PyBulletCupSample,
    *,
    max_position_norm: float,
    max_velocity_norm: float,
    min_cup_z: float,
) -> None:
    for object_id, obj in sample.state.objects.items():
        position = obj.attributes.get("position", [0.0, 0.0, 0.0])
        velocity = obj.attributes.get("velocity", [0.0, 0.0, 0.0])
        if _finite_vector(position):
            projected_position = _project_vector([float(item) for item in position], max_position_norm)
            if object_id == "cup_001":
                projected_position[2] = max(float(projected_position[2]), min_cup_z)
            obj.attributes["position"] = projected_position
        if _finite_vector(velocity):
            obj.attributes["velocity"] = _project_vector([float(item) for item in velocity], max_velocity_norm)


def _state_delta_norm(before_state: object, after_state: object) -> float:
    before_objects = getattr(before_state, "objects", {})
    after_objects = getattr(after_state, "objects", {})
    total = 0.0
    for object_id, after_obj in after_objects.items():
        before_obj = before_objects.get(object_id)
        if before_obj is None:
            continue
        for attribute in ("position", "velocity"):
            before_value = before_obj.attributes.get(attribute, [0.0, 0.0, 0.0])
            after_value = after_obj.attributes.get(attribute, [0.0, 0.0, 0.0])
            if not (_finite_vector(before_value) and _finite_vector(after_value)):
                continue
            for before_item, after_item in zip(before_value, after_value, strict=False):
                total += (float(after_item) - float(before_item)) ** 2
    return math.sqrt(total)


def _clip_vector(values: torch.Tensor, max_norm: float, finite_clamp: float = 0.0) -> torch.Tensor:
    if finite_clamp > 0.0:
        values = torch.nan_to_num(values, nan=0.0, posinf=finite_clamp, neginf=-finite_clamp)
        values = values.clamp(min=-finite_clamp, max=finite_clamp)
    if max_norm <= 0.0:
        return values
    norm = values.norm().clamp_min(1e-8)
    if not torch.isfinite(norm):
        return torch.zeros_like(values)
    scale = min(1.0, float(max_norm) / float(norm.item()))
    return values * scale


def _project_vector(values: list[float], max_norm: float) -> list[float]:
    if max_norm <= 0.0:
        return values
    norm = math.sqrt(sum(float(item) ** 2 for item in values))
    if not math.isfinite(norm) or norm <= max_norm:
        return values
    scale = max_norm / max(norm, 1e-8)
    return [float(item) * scale for item in values]


def _class_weights(dataset: PyBulletCupDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _collate_fn(args: argparse.Namespace, model_name: str):
    if not _uses_pre_tensor_index(args, model_name):
        return collate_pybullet_cup_samples

    def collate(samples: list[PyBulletCupSample]):
        return collate_indexed_pybullet_cup_samples(
            samples,
            max_nodes=args.working_set_size,
            max_depth=args.index_depth,
        )

    return collate


def _uses_pre_tensor_index(args: argparse.Namespace, model_name: str) -> bool:
    return bool(args.pre_tensor_indexed and model_name.startswith("wpu-cws-indexed"))


def _move_batch(batch: StateGraphBatch, device: torch.device) -> StateGraphBatch:
    batch.object_features = batch.object_features.to(device)
    batch.relation_indices = batch.relation_indices.to(device)
    batch.relation_features = batch.relation_features.to(device)
    batch.event_features = batch.event_features.to(device)
    batch.object_mask = batch.object_mask.to(device)
    batch.relation_mask = batch.relation_mask.to(device)
    batch.target_indices = batch.target_indices.to(device)
    batch.time_features = batch.time_features.to(device)
    return batch


def _selected_k(model: torch.nn.Module, batch: StateGraphBatch) -> float:
    if isinstance(model, CausalWorkingSetProcessor) and model.last_working_set_stats is not None:
        return model.last_working_set_stats.mean_selected
    return float(batch.object_mask.sum(dim=1).float().mean().detach().cpu().item())


def _entropy(probabilities: torch.Tensor) -> float:
    probs = probabilities.detach().clamp_min(1e-8)
    return float((-(probs * probs.log()).sum()).cpu().item())


def _finite_vector(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(math.isfinite(float(item)) for item in value)


def _norm(value: object) -> float:
    if not isinstance(value, list):
        return float("inf")
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
