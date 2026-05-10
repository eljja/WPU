from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from typing import Any


@dataclass(slots=True)
class WorldObject:
    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_updated: float = 0.0

    def with_updates(self, updates: dict[str, Any], event_time: float | None = None) -> "WorldObject":
        next_obj = copy.deepcopy(self)
        for key, value in updates.items():
            if key == "confidence":
                next_obj.confidence = float(value)
            elif key == "type":
                next_obj.type = str(value)
            else:
                next_obj.attributes[key] = copy.deepcopy(value)
        if event_time is not None:
            next_obj.last_updated = float(event_time)
        return next_obj


@dataclass(slots=True)
class Relation:
    src: str
    dst: str
    type: str
    strength: float = 1.0
    confidence: float = 1.0
    last_updated: float = 0.0

    def touches(self, object_id: str) -> bool:
        return self.src == object_id or self.dst == object_id

    def other(self, object_id: str) -> str | None:
        if self.src == object_id:
            return self.dst
        if self.dst == object_id:
            return self.src
        return None


@dataclass(slots=True)
class Event:
    type: str
    target: str
    delta: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeltaState:
    object_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    relation_updates: list[Relation] = field(default_factory=list)
    changed_objects: set[str] = field(default_factory=set)
    time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_object(self, object_id: str, updates: dict[str, Any]) -> None:
        self.object_updates.setdefault(object_id, {}).update(copy.deepcopy(updates))
        self.changed_objects.add(object_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_updates": copy.deepcopy(self.object_updates),
            "relation_updates": [asdict(relation) for relation in self.relation_updates],
            "changed_objects": sorted(self.changed_objects),
            "time": self.time,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeltaState":
        return cls(
            object_updates=copy.deepcopy(data.get("object_updates", {})),
            relation_updates=[Relation(**item) for item in data.get("relation_updates", [])],
            changed_objects=set(data.get("changed_objects", [])),
            time=float(data.get("time", 0.0)),
            metadata=copy.deepcopy(data.get("metadata", {})),
        )


@dataclass(slots=True)
class Branch:
    id: str
    parent_id: str | None
    probability: float
    delta_state: DeltaState = field(default_factory=DeltaState)
    time: float = 0.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "probability": self.probability,
            "delta_state": self.delta_state.to_dict(),
            "time": self.time,
            "label": self.label,
        }


@dataclass(slots=True)
class WorldState:
    objects: dict[str, WorldObject] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_object(self, obj: WorldObject) -> None:
        self.objects[obj.id] = obj

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def neighbors(self, object_id: str) -> list[str]:
        neighbors: list[str] = []
        for relation in self.relations:
            other = relation.other(object_id)
            if other is not None:
                neighbors.append(other)
        return neighbors

    def relations_for(self, object_id: str) -> list[Relation]:
        return [relation for relation in self.relations if relation.touches(object_id)]

    def apply_delta(self, delta: DeltaState) -> "WorldState":
        next_state = copy.deepcopy(self)
        for object_id, updates in delta.object_updates.items():
            if object_id in next_state.objects:
                next_state.objects[object_id] = next_state.objects[object_id].with_updates(
                    updates,
                    delta.time,
                )
        if delta.relation_updates:
            next_state.relations = _merge_relation_updates(next_state.relations, delta.relation_updates)
        next_state.time = max(next_state.time, delta.time)
        return next_state

    def overlay_branch(self, branch: Branch) -> "WorldState":
        return self.apply_delta(branch.delta_state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": {object_id: asdict(obj) for object_id, obj in self.objects.items()},
            "relations": [asdict(relation) for relation in self.relations],
            "time": self.time,
            "metadata": copy.deepcopy(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        return cls(
            objects={
                object_id: WorldObject(**obj_data)
                for object_id, obj_data in data.get("objects", {}).items()
            },
            relations=[Relation(**item) for item in data.get("relations", [])],
            time=float(data.get("time", 0.0)),
            metadata=copy.deepcopy(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "WorldState":
        return cls.from_dict(json.loads(payload))


def _merge_relation_updates(base: list[Relation], updates: list[Relation]) -> list[Relation]:
    merged = list(base)
    for update in updates:
        replaced = False
        for index, relation in enumerate(merged):
            if (relation.src, relation.dst, relation.type) == (update.src, update.dst, update.type):
                merged[index] = update
                replaced = True
                break
        if not replaced:
            merged.append(update)
    return merged
