from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from wpu.core.state import DeltaState, Event, WorldState
from wpu.data.working_set_physics import create_causal_working_set_state
from wpu.models.batch import StateGraphBatch
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop WPU CWS rollout using BaseState + DeltaState overlays.")
    parser.add_argument("--models", nargs="+", default=["wpu-cws-oracle", "wpu-cws-learned", "wpu-cws-indexed"])
    parser.add_argument("--background-objects", type=int, default=4088)
    parser.add_argument("--causal-obstacles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--working-set-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cws_closed_loop/closed_loop.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        rows.append(_rollout_model(model_name, args))
    _write_csv(args.output, rows)
    print(f"wrote={args.output}")


def _rollout_model(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    model.eval()
    state = create_causal_working_set_state(
        cup_x=0.72,
        hand_x=0.62,
        fall_risk=0.12,
        background_objects=args.background_objects,
        causal_obstacles=args.causal_obstacles,
    )

    entropy_total = 0.0
    changed_total = 0
    delta_norm_total = 0.0
    recall_total = 0.0
    recall_count = 0
    violation_total = 0
    branch_switches = 0
    previous_branch: int | None = None

    for step in range(args.horizon):
        event = _event_for_step(step)
        batch = StateGraphBatch.from_world_states([state], [event])
        batch = _move_batch(batch, device)
        with torch.no_grad():
            prediction = model(batch, num_branches=3, route_branches=3)
        probabilities = prediction.branch_probabilities[0].detach().cpu()
        branch = int(probabilities.argmax().item())
        if previous_branch is not None and previous_branch != branch:
            branch_switches += 1
        previous_branch = branch
        entropy_total += _entropy(probabilities)

        delta = _prediction_to_delta(state, prediction.object_delta[0].detach().cpu(), step + 1)
        changed_total += len(delta.changed_objects)
        delta_norm_total += _delta_norm(delta)
        state = state.apply_delta(delta)
        violation_total += _constraint_violations(state)

        stats = getattr(model, "last_working_set_stats", None)
        if stats is not None:
            recall_total += float(stats.mean_causal_recall)
            recall_count += 1

    return {
        "model": model_name,
        "total_objects_n": len(state.objects),
        "causal_k": args.causal_obstacles + 4,
        "horizon": args.horizon,
        "changed_objects_per_step": round(changed_total / max(args.horizon, 1), 6),
        "delta_norm_per_step": round(delta_norm_total / max(args.horizon, 1), 6),
        "branch_entropy_mean": round(entropy_total / max(args.horizon, 1), 6),
        "branch_switch_rate": round(branch_switches / max(args.horizon - 1, 1), 6),
        "constraint_violations_per_step": round(violation_total / max(args.horizon, 1), 6),
        "causal_recall_mean": round(recall_total / max(recall_count, 1), 6),
    }


def _event_for_step(step: int) -> Event:
    force = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(step * 0.37))
    return Event(
        type="touch",
        target="cup_001",
        delta={"force": force, "position": [0.01 * math.sin(step * 0.19), 0.0, 0.0]},
        confidence=0.94,
        time=float(step),
    )


def _prediction_to_delta(state: WorldState, object_delta: torch.Tensor, time_value: float) -> DeltaState:
    delta = DeltaState(time=time_value)
    object_ids = list(state.objects)
    for index, object_id in enumerate(object_ids):
        if index >= object_delta.size(0):
            break
        row = object_delta[index]
        if float(row.norm().item()) < 1e-4:
            continue
        obj = state.objects[object_id]
        position = list(obj.attributes.get("position", [0.0, 0.0, 0.0]))
        velocity = list(obj.attributes.get("velocity", [0.0, 0.0, 0.0]))
        next_position = [float(position[i]) + float(row[1 + i].item()) for i in range(3)]
        next_velocity = [float(velocity[i]) + float(row[4 + i].item()) for i in range(3)]
        next_confidence = min(1.0, max(0.0, float(obj.confidence) + float(row[7].item())))
        delta.record_object(
            object_id,
            {
                "position": next_position,
                "velocity": next_velocity,
                "confidence": next_confidence,
            },
        )
    return delta


def _delta_norm(delta: DeltaState) -> float:
    total = 0.0
    for updates in delta.object_updates.values():
        position = updates.get("position", [0.0, 0.0, 0.0])
        velocity = updates.get("velocity", [0.0, 0.0, 0.0])
        total += sum(float(value) ** 2 for value in position)
        total += sum(float(value) ** 2 for value in velocity)
    return math.sqrt(total)


def _constraint_violations(state: WorldState) -> int:
    violations = 0
    for obj in state.objects.values():
        position = obj.attributes.get("position", [0.0, 0.0, 0.0])
        if not isinstance(position, list) or len(position) < 3:
            continue
        if any(abs(float(value)) > 1000.0 for value in position):
            violations += 1
        if float(position[2]) < -10.0:
            violations += 1
    return violations


def _entropy(probabilities: torch.Tensor) -> float:
    probs = probabilities.clamp_min(1e-8)
    return float(-(probs * probs.log()).sum().item())


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
