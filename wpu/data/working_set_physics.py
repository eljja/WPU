from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import Dataset

from wpu.core.state import Event, Relation, WorldObject, WorldState
from wpu.data.object_physics import BRANCH_LABELS, create_touch_event
from wpu.models.batch import OBJECT_FEATURE_DIM, StateGraphBatch


def create_causal_working_set_state(
    *,
    cup_x: float,
    hand_x: float,
    edge_x: float = 1.0,
    fall_risk: float,
    background_objects: int,
    causal_obstacles: int = 0,
    adversarial_distractors: int = 0,
) -> WorldState:
    """Create a world with a fixed causal core and scalable distractor state."""

    state = WorldState(time=0.0, metadata={"scenario": "causal_working_set"})
    state.add_object(
        WorldObject(
            id="cup_001",
            type="cup",
            attributes={"position": [cup_x, 0.0, 0.82], "velocity": [0.0, 0.0, 0.0], "fall_risk": fall_risk},
            confidence=0.96,
        )
    )
    state.add_object(
        WorldObject(
            id="table_001",
            type="table",
            attributes={"position": [0.5, 0.0, 0.75], "velocity": [0.0, 0.0, 0.0]},
            confidence=0.99,
        )
    )
    state.add_object(
        WorldObject(
            id="hand_001",
            type="robot_hand",
            attributes={"position": [hand_x, 0.0, 0.9], "velocity": [0.1, 0.0, 0.0]},
            confidence=0.93,
        )
    )
    state.add_object(
        WorldObject(
            id="edge_001",
            type="table_edge",
            attributes={"position": [edge_x, 0.0, 0.75], "velocity": [0.0, 0.0, 0.0]},
            confidence=0.98,
        )
    )
    state.add_relation(Relation("cup_001", "table_001", "on_top_of", strength=0.92, confidence=0.94))
    state.add_relation(Relation("hand_001", "cup_001", "near", strength=0.78, confidence=0.86))
    state.add_relation(Relation("cup_001", "edge_001", "near", strength=0.62, confidence=0.82))

    for index in range(causal_obstacles):
        obstacle_id = f"obstacle_{index:03d}"
        obstacle_x = cup_x + 0.03 * ((index % 3) - 1)
        state.add_object(
            WorldObject(
                id=obstacle_id,
                type="obstacle",
                attributes={"position": [obstacle_x, 0.04 * index, 0.82], "velocity": [0.0, 0.0, 0.0]},
                confidence=0.9,
            )
        )
        state.add_relation(Relation("cup_001", obstacle_id, "near", strength=0.55, confidence=0.82))

    for index in range(adversarial_distractors):
        fake_id = f"fake_cup_{index:03d}"
        state.add_object(
            WorldObject(
                id=fake_id,
                type="cup",
                attributes={"position": [0.1 + 0.01 * index, 2.0, 0.82], "velocity": [0.0, 0.0, 0.0]},
                confidence=0.88,
            )
        )
        state.add_relation(Relation(fake_id, "table_001", "on_top_of", strength=0.85, confidence=0.8))

    for index in range(background_objects):
        state.add_object(
            WorldObject(
                id=f"context_{index:05d}",
                type="background_object",
                attributes={
                    "position": [float(index % 64), float(index // 64), 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                },
                confidence=0.75,
            )
        )
    return state


@dataclass(slots=True)
class WorkingSetPhysicsSample:
    state: WorldState
    event: Event
    target_object_delta: torch.Tensor
    branch_label: int
    causal_working_set_size: int


class WorkingSetPhysicsDataset(Dataset):
    """Synthetic large-N task with independently controlled N and causal K."""

    def __init__(
        self,
        size: int = 256,
        seed: int = 7,
        background_objects: int = 512,
        causal_obstacles: int = 0,
        adversarial_distractors: int = 0,
        balanced_labels: bool = False,
    ) -> None:
        self.size = size
        self.seed = seed
        self.background_objects = background_objects
        self.causal_obstacles = causal_obstacles
        self.adversarial_distractors = adversarial_distractors
        self.balanced_labels = balanced_labels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> WorkingSetPhysicsSample:
        if self.balanced_labels:
            target_label = index % len(BRANCH_LABELS)
            for attempt in range(256):
                sample = self._make_sample(index * 997 + attempt)
                if sample.branch_label == target_label:
                    return sample
        return self._make_sample(index)

    def _make_sample(self, index: int) -> WorkingSetPhysicsSample:
        rng = random.Random(self.seed + index)
        cup_x = rng.uniform(0.45, 0.98)
        hand_x = rng.uniform(0.2, 0.8)
        force = rng.uniform(0.05, 1.2)
        fall_risk = rng.uniform(0.0, 0.35)
        state = create_causal_working_set_state(
            cup_x=cup_x,
            hand_x=hand_x,
            fall_risk=fall_risk,
            background_objects=self.background_objects,
            causal_obstacles=self.causal_obstacles,
            adversarial_distractors=self.adversarial_distractors,
        )
        event = create_touch_event(force=force)
        obstacle_penalty = min(0.25, self.causal_obstacles * 0.015)
        edge_distance = max(0.0, 1.0 - cup_x)
        fall_score = force * 0.65 + fall_risk + obstacle_penalty - edge_distance
        catch_score = max(0.0, force - 0.55) * max(0.0, 0.85 - abs(hand_x - cup_x) - obstacle_penalty)
        if fall_score > 0.45 and catch_score < 0.18:
            label = 1
        elif fall_score > 0.35 and catch_score >= 0.18:
            label = 2
        else:
            label = 0

        target = torch.zeros((len(state.objects), OBJECT_FEATURE_DIM), dtype=torch.float32)
        target[0, 1] = force * 0.08
        target[0, 7] = -0.1 if label == 1 else 0.02
        for offset in range(self.causal_obstacles):
            object_index = 4 + offset
            if object_index < target.size(0):
                target[object_index, 1] = force * 0.01
        causal_k = 4 + self.causal_obstacles
        return WorkingSetPhysicsSample(state, event, target, label, causal_k)


def collate_working_set_samples(
    samples: list[WorkingSetPhysicsSample],
) -> tuple[StateGraphBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = StateGraphBatch.from_world_states(
        [sample.state for sample in samples],
        [sample.event for sample in samples],
    )
    max_objects = batch.object_features.size(1)
    targets = torch.zeros((len(samples), max_objects, OBJECT_FEATURE_DIM), dtype=torch.float32)
    labels = torch.tensor([sample.branch_label for sample in samples], dtype=torch.long)
    causal_k = torch.tensor([sample.causal_working_set_size for sample in samples], dtype=torch.long)
    for index, sample in enumerate(samples):
        targets[index, : sample.target_object_delta.size(0)] = sample.target_object_delta
    return batch, targets, labels, causal_k


def collate_indexed_working_set_samples(
    samples: list[WorkingSetPhysicsSample],
    *,
    max_nodes: int = 16,
    max_depth: int = 1,
) -> tuple[StateGraphBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate only the indexed event-local subgraph before tensorization.

    This is the v2 pre-tensor retrieval path. It avoids building tensors for
    every background object and instead projects each `WorldState` to the
    event-target relation frontier first.
    """

    projected_states: list[WorldState] = []
    projected_targets: list[torch.Tensor] = []
    for sample in samples:
        original_ids = list(sample.state.objects)
        selected_ids = _indexed_object_ids(sample.state, sample.event, max_nodes=max_nodes, max_depth=max_depth)
        projected_states.append(_project_state(sample.state, selected_ids))
        selected_target = torch.zeros((len(selected_ids), OBJECT_FEATURE_DIM), dtype=torch.float32)
        original_index = {object_id: index for index, object_id in enumerate(original_ids)}
        for selected_index, object_id in enumerate(selected_ids):
            source_index = original_index.get(object_id)
            if source_index is not None and source_index < sample.target_object_delta.size(0):
                selected_target[selected_index] = sample.target_object_delta[source_index]
        projected_targets.append(selected_target)

    batch = StateGraphBatch.from_world_states(projected_states, [sample.event for sample in samples])
    max_objects = batch.object_features.size(1)
    targets = torch.zeros((len(samples), max_objects, OBJECT_FEATURE_DIM), dtype=torch.float32)
    labels = torch.tensor([sample.branch_label for sample in samples], dtype=torch.long)
    causal_k = torch.tensor([sample.causal_working_set_size for sample in samples], dtype=torch.long)
    for index, target in enumerate(projected_targets):
        targets[index, : target.size(0)] = target
    return batch, targets, labels, causal_k


def _indexed_object_ids(state: WorldState, event: Event, *, max_nodes: int, max_depth: int) -> list[str]:
    if event.target not in state.objects:
        return list(state.objects)[:max_nodes]
    selected = [event.target]
    visited = {event.target}
    frontier = [(event.target, 0)]
    while frontier and len(selected) < max_nodes:
        object_id, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for relation in state.relations_for(object_id):
            other = relation.other(object_id)
            if other is None or other in visited or other not in state.objects:
                continue
            visited.add(other)
            selected.append(other)
            if len(selected) >= max_nodes:
                break
            frontier.append((other, depth + 1))
    return selected


def _project_state(state: WorldState, object_ids: list[str]) -> WorldState:
    selected = set(object_ids)
    projected = WorldState(time=state.time, metadata={**state.metadata, "pre_tensor_indexed": True})
    for object_id in object_ids:
        projected.add_object(state.objects[object_id])
    for relation in state.relations:
        if relation.src in selected and relation.dst in selected:
            projected.add_relation(relation)
    return projected


__all__ = [
    "BRANCH_LABELS",
    "WorkingSetPhysicsDataset",
    "WorkingSetPhysicsSample",
    "collate_indexed_working_set_samples",
    "collate_working_set_samples",
    "create_causal_working_set_state",
]
