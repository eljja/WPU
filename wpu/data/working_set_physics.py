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
    ) -> None:
        self.size = size
        self.seed = seed
        self.background_objects = background_objects
        self.causal_obstacles = causal_obstacles
        self.adversarial_distractors = adversarial_distractors

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> WorkingSetPhysicsSample:
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


__all__ = [
    "BRANCH_LABELS",
    "WorkingSetPhysicsDataset",
    "WorkingSetPhysicsSample",
    "collate_working_set_samples",
    "create_causal_working_set_state",
]
