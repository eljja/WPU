from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from wpu.core.state import Event, WorldState  # noqa: E402
from wpu.data.working_set_physics import (  # noqa: E402
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
    collate_interaction_working_set_samples,
    collate_proximity_working_set_samples,
)


FEATURE_DIM = 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether interaction-density retrieval can be learned from state features.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[19, 23])
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--test-samples", type=int, default=160)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_learned_retriever_probe.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            print(f"learned-retriever N={n_value} K={k_value}", flush=True)
            train_samples = _make_samples(n_value, k_value, args.train_seeds, args.train_samples)
            test_samples = _make_samples(n_value, k_value, args.test_seeds, args.test_samples)
            model = _train_model(train_samples, args.budget, args.steps, args.hidden_dim, args.lr)
            rows.extend(_evaluate_modes(test_samples, n_value, k_value, args.budget, model))
            _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _make_samples(n_value: int, k_value: int, seeds: list[int], samples_per_seed: int):
    samples = []
    causal_obstacles = max(0, k_value - 4)
    background_objects = max(0, n_value - 4 - causal_obstacles)
    for seed in seeds:
        dataset = WorkingSetPhysicsDataset(
            size=samples_per_seed,
            seed=seed + n_value * 17 + k_value,
            background_objects=background_objects,
            causal_obstacles=causal_obstacles,
            balanced_labels=False,
            interaction_mode="pairwise",
        )
        samples.extend(dataset[index] for index in range(len(dataset)))
    return samples


def _train_model(samples, budget: int, steps: int, hidden_dim: int, lr: float) -> nn.Module:
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for sample in samples:
        teacher_ids = _selected_ids(sample, "interaction", budget)
        target = sample.event.target
        for object_id in _candidate_ids(sample.state, sample.event):
            if object_id == target:
                continue
            features.append(_candidate_features(sample.state, sample.event, object_id))
            labels.append(float(object_id in teacher_ids))
    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    model = nn.Sequential(
        nn.LayerNorm(FEATURE_DIM),
        nn.Linear(FEATURE_DIM, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    positive = label_tensor.sum().clamp_min(1.0)
    negative = (1.0 - label_tensor).sum().clamp_min(1.0)
    pos_weight = negative / positive
    for _ in range(steps):
        logits = model(feature_tensor).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label_tensor, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def _evaluate_modes(samples, n_value: int, k_value: int, budget: int, model: nn.Module) -> list[dict[str, object]]:
    rows = []
    for mode in ["indexed", "proximity", "interaction", "learned"]:
        selected_sets = []
        for sample in samples:
            selected_sets.append(_selected_ids(sample, mode, budget, model))
        rows.append(_summary_row(samples, selected_sets, n_value, k_value, budget, mode))
    return rows


def _summary_row(samples, selected_sets: list[list[str]], n_value: int, k_value: int, budget: int, mode: str) -> dict[str, object]:
    causal_ids = {"cup_001", "table_001", "hand_001", "edge_001"}
    causal_ids.update(f"obstacle_{index:03d}" for index in range(max(0, k_value - 4)))
    obstacle_ids = {object_id for object_id in causal_ids if object_id.startswith("obstacle_")}
    teacher_sets = [_selected_ids(sample, "interaction", budget) for sample in samples]
    return {
        "total_objects_n": n_value,
        "causal_k": k_value,
        "budget": budget,
        "selection_mode": mode,
        "samples": len(samples),
        "causal_recall": round(mean(len(set(ids) & causal_ids) / max(len(causal_ids), 1) for ids in selected_sets), 6),
        "obstacle_recall": round(mean(len(set(ids) & obstacle_ids) / max(len(obstacle_ids), 1) for ids in selected_sets), 6),
        "selected_obstacles": round(mean(sum(object_id.startswith("obstacle_") for object_id in ids) for ids in selected_sets), 6),
        "hand_hit_rate": round(mean(float("hand_001" in ids) for ids in selected_sets), 6),
        "teacher_overlap": round(
            mean(len(set(ids) & set(teacher)) / max(len(set(ids) | set(teacher)), 1) for ids, teacher in zip(selected_sets, teacher_sets, strict=True)),
            6,
        ),
        "selected_obstacle_pair_density": round(
            mean(_selected_pair_density(sample.state, [object_id for object_id in ids if object_id.startswith("obstacle_")]) for sample, ids in zip(samples, selected_sets, strict=True)),
            6,
        ),
    }


def _selected_ids(sample, mode: str, budget: int, model: nn.Module | None = None) -> list[str]:
    if mode == "learned":
        if model is None:
            raise ValueError("learned mode requires a model")
        return _learned_selected_ids(sample.state, sample.event, budget, model)
    collate_fn = {
        "indexed": collate_indexed_working_set_samples,
        "proximity": collate_proximity_working_set_samples,
        "interaction": collate_interaction_working_set_samples,
    }[mode]
    batch, _, _, _ = collate_fn([sample], max_nodes=budget, max_depth=1)
    return list(batch.object_ids[0]) if batch.object_ids is not None else []


def _learned_selected_ids(state: WorldState, event: Event, budget: int, model: nn.Module) -> list[str]:
    if event.target not in state.objects:
        return list(state.objects)[:budget]
    candidates = [object_id for object_id in _candidate_ids(state, event) if object_id != event.target]
    if not candidates:
        return [event.target]
    features = torch.stack([_candidate_features(state, event, object_id) for object_id in candidates])
    with torch.no_grad():
        scores = model(features).squeeze(-1)
    ranked = [object_id for _, object_id in sorted(zip(scores.tolist(), candidates, strict=True), reverse=True)]
    selected = [event.target]
    for object_id in ranked:
        if object_id not in selected:
            selected.append(object_id)
        if len(selected) >= budget:
            break
    return selected


def _candidate_ids(state: WorldState, event: Event) -> list[str]:
    if event.target not in state.objects:
        return list(state.objects)
    selected: list[str] = []
    for relation in state.relations_for(event.target):
        other = relation.other(event.target)
        if other is not None and other in state.objects and other not in selected:
            selected.append(other)
    return [event.target, *selected]


def _candidate_features(state: WorldState, event: Event, object_id: str) -> torch.Tensor:
    obj = state.objects[object_id]
    target_xy = _object_xy(state, event.target)
    object_xy = _object_xy(state, object_id)
    dx = object_xy[0] - target_xy[0]
    dy = object_xy[1] - target_xy[1]
    distance = (dx * dx + dy * dy) ** 0.5
    obstacle_ids = [candidate for candidate in _candidate_ids(state, event) if state.objects[candidate].type == "obstacle"]
    candidate_ids = _candidate_ids(state, event)
    local_density = _local_obstacle_density(state, object_id, obstacle_ids) if obj.type == "obstacle" else 0.0
    axis_alignment = 1.0 if abs(dx) < 0.05 else 0.0
    relation_strength, relation_confidence, relation_near, relation_support = _relation_features_to_target(state, event.target, object_id)
    type_flags = [
        float(obj.type == "robot_hand"),
        float(obj.type == "obstacle"),
        float(obj.type == "table_edge"),
        float(obj.type == "table"),
    ]
    force = float(event.delta.get("force", 0.0))
    return torch.tensor(
        [
            *type_flags,
            dx,
            dy,
            distance,
            local_density,
            axis_alignment,
            relation_strength,
            relation_confidence,
            relation_near,
            relation_support,
            force,
            len(candidate_ids) / 64.0,
            len(obstacle_ids) / 64.0,
        ],
        dtype=torch.float32,
    )


def _relation_features_to_target(state: WorldState, target_id: str, object_id: str) -> tuple[float, float, float, float]:
    for relation in state.relations_for(target_id):
        if relation.other(target_id) == object_id:
            return (
                float(relation.strength),
                float(relation.confidence),
                float(relation.type == "near"),
                float(relation.type == "on_top_of"),
            )
    return 0.0, 0.0, 0.0, 0.0


def _local_obstacle_density(state: WorldState, object_id: str, obstacle_ids: list[str]) -> float:
    object_xy = _object_xy(state, object_id)
    density = 0.0
    for other_id in obstacle_ids:
        if other_id == object_id:
            continue
        distance = _distance_xy(object_xy, _object_xy(state, other_id))
        density += max(0.0, 0.075 - distance) / 0.075
    return density


def _selected_pair_density(state: WorldState, obstacle_ids: list[str]) -> float:
    if len(obstacle_ids) < 2:
        return 0.0
    close_pairs = 0
    possible_pairs = 0
    for left_index, left_id in enumerate(obstacle_ids):
        for right_id in obstacle_ids[left_index + 1 :]:
            possible_pairs += 1
            close_pairs += int(_distance_xy(_object_xy(state, left_id), _object_xy(state, right_id)) < 0.075)
    return close_pairs / max(possible_pairs, 1)


def _object_xy(state: WorldState, object_id: str) -> tuple[float, float]:
    position = state.objects[object_id].attributes.get("position", [0.0, 0.0, 0.0])
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return 0.0, 0.0
    return float(position[0]), float(position[1])


def _distance_xy(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
