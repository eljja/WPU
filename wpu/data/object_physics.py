from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import Dataset

from wpu.core.state import Event, Relation, WorldObject, WorldState
from wpu.models.batch import OBJECT_FEATURE_DIM, StateGraphBatch

BRANCH_LABELS = ["stable", "falls", "caught"]


def create_robot_cup_state(
    *,
    cup_x: float = 0.72,
    hand_x: float = 0.58,
    edge_x: float = 1.0,
    fall_risk: float = 0.15,
    background_objects: int = 80,
) -> WorldState:
    state = WorldState(time=0.0, metadata={"scenario": "robot_cup"})
    state.add_object(
        WorldObject(
            id="cup_001",
            type="cup",
            attributes={
                "position": [cup_x, 0.0, 0.82],
                "velocity": [0.0, 0.0, 0.0],
                "fall_risk": fall_risk,
            },
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
    for index in range(background_objects):
        state.add_object(
            WorldObject(
                id=f"context_{index:03d}",
                type="background_object",
                attributes={
                    "position": [float(index % 10), float(index // 10), 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                },
                confidence=0.75,
            )
        )
    return state


def create_touch_event(force: float = 0.35, time: float = 1.0) -> Event:
    return Event(
        type="hand_touched_cup",
        target="cup_001",
        delta={"position": [force * 0.08, 0.0, 0.0], "force": force},
        confidence=0.95,
        time=time,
    )


@dataclass(slots=True)
class PhysicsSample:
    state: WorldState
    event: Event
    target_object_delta: torch.Tensor
    branch_label: int


class ObjectPhysicsDataset(Dataset):
    def __init__(
        self,
        size: int = 256,
        seed: int = 7,
        background_objects: int = 80,
        relation_noise: int = 0,
        affected_background_objects: int = 0,
        background_delta_scale: float = 0.01,
    ) -> None:
        self.size = size
        self.seed = seed
        self.background_objects = background_objects
        self.relation_noise = relation_noise
        self.affected_background_objects = affected_background_objects
        self.background_delta_scale = background_delta_scale

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> PhysicsSample:
        rng = random.Random(self.seed + index)
        cup_x = rng.uniform(0.45, 0.98)
        hand_x = rng.uniform(0.2, 0.8)
        force = rng.uniform(0.05, 1.2)
        fall_risk = rng.uniform(0.0, 0.35)
        state = create_robot_cup_state(
            cup_x=cup_x,
            hand_x=hand_x,
            fall_risk=fall_risk,
            background_objects=self.background_objects,
        )
        for noise_index in range(min(self.relation_noise, self.background_objects)):
            relation_type = "near" if noise_index % 2 == 0 else "occludes"
            src = "cup_001" if noise_index % 3 else f"context_{noise_index:03d}"
            dst = f"context_{noise_index:03d}" if src == "cup_001" else "table_001"
            state.add_relation(
                Relation(
                    src,
                    dst,
                    relation_type,
                    strength=rng.uniform(0.05, 0.35),
                    confidence=rng.uniform(0.4, 0.75),
                )
            )
        event = create_touch_event(force=force)
        edge_distance = max(0.0, 1.0 - cup_x)
        fall_score = force * 0.65 + fall_risk - edge_distance
        catch_score = max(0.0, force - 0.55) * max(0.0, 0.85 - abs(hand_x - cup_x))
        if fall_score > 0.45 and catch_score < 0.18:
            label = 1
        elif fall_score > 0.35 and catch_score >= 0.18:
            label = 2
        else:
            label = 0

        target = torch.zeros((len(state.objects), OBJECT_FEATURE_DIM), dtype=torch.float32)
        target[0, 1] = force * 0.08
        target[0, 7] = -0.1 if label == 1 else 0.02
        affected_count = min(self.affected_background_objects, self.background_objects)
        for offset in range(affected_count):
            object_index = 4 + offset
            if object_index < target.size(0):
                target[object_index, 1] = force * self.background_delta_scale
                target[object_index, 7] = 0.01
        return PhysicsSample(state=state, event=event, target_object_delta=target, branch_label=label)


def collate_physics_samples(samples: list[PhysicsSample]) -> tuple[StateGraphBatch, torch.Tensor, torch.Tensor]:
    batch = StateGraphBatch.from_world_states(
        [sample.state for sample in samples],
        [sample.event for sample in samples],
    )
    max_objects = batch.object_features.size(1)
    targets = torch.zeros((len(samples), max_objects, OBJECT_FEATURE_DIM), dtype=torch.float32)
    labels = torch.tensor([sample.branch_label for sample in samples], dtype=torch.long)
    for index, sample in enumerate(samples):
        targets[index, : sample.target_object_delta.size(0)] = sample.target_object_delta
    return batch, targets, labels
