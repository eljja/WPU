from __future__ import annotations

from dataclasses import dataclass

import torch

from wpu.core.objectification import evaluate_objectification
from wpu.core.state import Event, Relation, WorldState
from wpu.engines.scheduler import SchedulerMetrics

OBJECT_FEATURE_DIM = 8
RELATION_FEATURE_DIM = 4
EVENT_FEATURE_DIM = 8


@dataclass(slots=True)
class StateGraphBatch:
    object_features: torch.Tensor
    relation_indices: torch.Tensor
    relation_features: torch.Tensor
    event_features: torch.Tensor
    object_mask: torch.Tensor
    relation_mask: torch.Tensor
    target_indices: torch.Tensor
    time_features: torch.Tensor
    scheduler_metrics: list[SchedulerMetrics] | None = None
    object_ids: list[list[str]] | None = None

    @classmethod
    def from_world_states(cls, states: list[WorldState], events: list[Event]) -> "StateGraphBatch":
        if len(states) != len(events):
            raise ValueError("states and events must have the same length")

        max_objects = max((len(state.objects) for state in states), default=1)
        max_relations = max((len(state.relations) for state in states), default=1)
        object_features = torch.zeros((len(states), max_objects, OBJECT_FEATURE_DIM), dtype=torch.float32)
        relation_indices = torch.zeros((len(states), max_relations, 2), dtype=torch.long)
        relation_features = torch.zeros((len(states), max_relations, RELATION_FEATURE_DIM), dtype=torch.float32)
        event_features = torch.zeros((len(states), EVENT_FEATURE_DIM), dtype=torch.float32)
        object_mask = torch.zeros((len(states), max_objects), dtype=torch.bool)
        relation_mask = torch.zeros((len(states), max_relations), dtype=torch.bool)
        target_indices = torch.zeros((len(states),), dtype=torch.long)
        time_features = torch.zeros((len(states), 1), dtype=torch.float32)
        scheduler_metrics: list[SchedulerMetrics] = []
        object_ids_batch: list[list[str]] = []

        for batch_index, (state, event) in enumerate(zip(states, events, strict=True)):
            object_ids = list(state.objects)
            object_ids_batch.append(object_ids)
            object_index = {object_id: index for index, object_id in enumerate(object_ids)}
            object_mask[batch_index, : len(object_ids)] = True
            time_features[batch_index, 0] = float(state.time)

            for index, object_id in enumerate(object_ids):
                object_features[batch_index, index] = _encode_object(state, object_id)

            for relation_index, relation in enumerate(state.relations):
                if relation_index >= max_relations:
                    break
                relation_indices[batch_index, relation_index, 0] = object_index.get(relation.src, 0)
                relation_indices[batch_index, relation_index, 1] = object_index.get(relation.dst, 0)
                relation_features[batch_index, relation_index] = _encode_relation(relation)
                relation_mask[batch_index, relation_index] = True

            target_indices[batch_index] = object_index.get(event.target, 0)
            event_features[batch_index] = _encode_event(event, target_indices[batch_index].item())
            fanout = len(state.relations) / max(len(state.objects), 1)
            scheduler_metrics.append(
                SchedulerMetrics(
                    delta_n=1,
                    fanout=max(fanout, 1.0),
                    depth=3,
                    branches=3,
                    total_n=max(len(state.objects), 1),
                    uncertainty_growth=max(0.0, 1.0 - event.confidence),
                    objectification_score=evaluate_objectification(state).contract_score,
                )
            )

        return cls(
            object_features=object_features,
            relation_indices=relation_indices,
            relation_features=relation_features,
            event_features=event_features,
            object_mask=object_mask,
            relation_mask=relation_mask,
            target_indices=target_indices,
            time_features=time_features,
            scheduler_metrics=scheduler_metrics,
            object_ids=object_ids_batch,
        )


def _encode_object(state: WorldState, object_id: str) -> torch.Tensor:
    obj = state.objects[object_id]
    position = _vector(obj.attributes.get("position", [0.0, 0.0, 0.0]), 3)
    velocity = _vector(obj.attributes.get("velocity", [0.0, 0.0, 0.0]), 3)
    return torch.tensor([_stable_hash(obj.type), *position, *velocity, float(obj.confidence)], dtype=torch.float32)


def _encode_relation(relation: Relation) -> torch.Tensor:
    return torch.tensor(
        [
            _stable_hash(relation.type),
            float(relation.strength),
            float(relation.confidence),
            float(relation.last_updated),
        ],
        dtype=torch.float32,
    )


def _encode_event(event: Event, target_index: int) -> torch.Tensor:
    position_delta = _vector(event.delta.get("position", [0.0, 0.0, 0.0]), 3)
    force = float(event.delta.get("force", 0.0))
    return torch.tensor(
        [
            _stable_hash(event.type),
            float(target_index),
            *position_delta,
            force,
            float(event.confidence),
            float(event.time),
        ],
        dtype=torch.float32,
    )


def _vector(value: object, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [0.0] * size
    result = [float(item) for item in value[:size]]
    return result + [0.0] * (size - len(result))


def _stable_hash(value: str) -> float:
    return float(sum(ord(ch) for ch in value) % 997) / 997.0
