from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from wpu.core.state import Event, Relation, WorldObject, WorldState


@dataclass(slots=True)
class Region:
    id: str
    parent_id: str | None = None
    object_ids: set[str] = field(default_factory=set)
    child_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorldCausalQuery:
    event: Event
    max_objects: int = 32
    relation_depth: int = 1
    spatial_radius: float = 0.5
    include_uncertain: bool = True
    include_recent: bool = True


@dataclass(slots=True)
class WorldCausalSlice:
    object_ids: list[str]
    reason_by_object: dict[str, set[str]]
    total_objects: int

    @property
    def causal_working_set_size(self) -> int:
        return len(self.object_ids)

    @property
    def affected_fraction(self) -> float:
        return self.causal_working_set_size / max(self.total_objects, 1)


class HierarchicalWorldState:
    """Hierarchical state wrapper for large persistent worlds.

    The base `WorldState` stays the source of truth. This wrapper adds region
    membership and coarse-to-fine lookup without changing the JSON contract of
    the existing state model.
    """

    def __init__(self, state: WorldState) -> None:
        self.state = state
        self.regions: dict[str, Region] = {}
        self.object_to_region: dict[str, str] = {}

    def add_region(self, region_id: str, *, parent_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        region = self.regions.setdefault(region_id, Region(id=region_id))
        region.parent_id = parent_id
        if metadata:
            region.metadata.update(metadata)
        if parent_id is not None:
            parent = self.regions.setdefault(parent_id, Region(id=parent_id))
            parent.child_ids.add(region_id)

    def assign_object(self, object_id: str, region_id: str) -> None:
        if object_id not in self.state.objects:
            raise KeyError(f"unknown object_id: {object_id}")
        region = self.regions.setdefault(region_id, Region(id=region_id))
        previous = self.object_to_region.get(object_id)
        if previous is not None and previous in self.regions:
            self.regions[previous].object_ids.discard(object_id)
        region.object_ids.add(object_id)
        self.object_to_region[object_id] = region_id

    def region_objects(self, region_id: str, *, recursive: bool = False) -> list[str]:
        region = self.regions.get(region_id)
        if region is None:
            return []
        object_ids = set(region.object_ids)
        if recursive:
            for child_id in region.child_ids:
                object_ids.update(self.region_objects(child_id, recursive=True))
        return sorted(object_ids)


class WorldCausalIndex:
    """Multi-signal causal retrieval for persistent world state.

    This is the v3 index primitive. It combines identity, typed relations,
    spatial proximity, uncertainty, recency, and optional hierarchy. It is not a
    learned selector; it is an auditable candidate generator for event-local WPU
    propagation.
    """

    def __init__(self, state: WorldState, hierarchy: HierarchicalWorldState | None = None) -> None:
        self.state = state
        self.hierarchy = hierarchy

    def query(self, query: WorldCausalQuery) -> WorldCausalSlice:
        target = query.event.target if query.event.target in self.state.objects else _first_object_id(self.state)
        reasons: dict[str, set[str]] = {}
        ordered: list[str] = []

        def add(object_id: str, reason: str) -> None:
            if object_id not in self.state.objects:
                return
            if object_id not in reasons:
                reasons[object_id] = set()
                ordered.append(object_id)
            reasons[object_id].add(reason)

        add(target, "event_target")
        for object_id in self._relation_frontier(target, max_depth=query.relation_depth):
            add(object_id, "relation_frontier")
        for object_id in self._spatial_neighbors(target, radius=query.spatial_radius):
            add(object_id, "spatial_neighbor")
        if self.hierarchy is not None:
            region_id = self.hierarchy.object_to_region.get(target)
            if region_id is not None:
                for object_id in self.hierarchy.region_objects(region_id):
                    add(object_id, "same_region")
        if query.include_uncertain:
            for object_id, obj in self.state.objects.items():
                if obj.confidence < 0.55:
                    add(object_id, "uncertainty_hotspot")
        if query.include_recent:
            event_time = float(query.event.time)
            for object_id, obj in self.state.objects.items():
                if obj.last_updated > 0.0 and event_time >= obj.last_updated and event_time - obj.last_updated <= 1.0:
                    add(object_id, "recent_change")

        ranked = sorted(
            ordered,
            key=lambda object_id: self._rank_key(object_id, target, reasons[object_id]),
        )
        return WorldCausalSlice(
            object_ids=ranked[: query.max_objects],
            reason_by_object={object_id: reasons[object_id] for object_id in ranked[: query.max_objects]},
            total_objects=len(self.state.objects),
        )

    def _relation_frontier(self, target: str, *, max_depth: int) -> list[str]:
        visited = {target}
        frontier = [(target, 0)]
        result: list[str] = []
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for relation in self.state.relations_for(current):
                other = relation.other(current)
                if other is None or other in visited:
                    continue
                visited.add(other)
                result.append(other)
                frontier.append((other, depth + 1))
        return result

    def _spatial_neighbors(self, target: str, *, radius: float) -> list[str]:
        target_position = _position(self.state.objects[target])
        if target_position is None:
            return []
        neighbors = []
        for object_id, obj in self.state.objects.items():
            if object_id == target:
                continue
            position = _position(obj)
            if position is None:
                continue
            if _distance(target_position, position) <= radius:
                neighbors.append(object_id)
        return neighbors

    def _rank_key(self, object_id: str, target: str, reasons: set[str]) -> tuple[int, float, str]:
        priority = min(_REASON_PRIORITY.get(reason, 99) for reason in reasons)
        target_position = _position(self.state.objects[target])
        object_position = _position(self.state.objects[object_id])
        distance = _distance(target_position, object_position) if target_position and object_position else 1e9
        return (priority, distance, object_id)


_REASON_PRIORITY = {
    "event_target": 0,
    "relation_frontier": 1,
    "spatial_neighbor": 2,
    "same_region": 3,
    "uncertainty_hotspot": 4,
    "recent_change": 5,
}


def _first_object_id(state: WorldState) -> str:
    if not state.objects:
        raise ValueError("cannot query an empty WorldState")
    return next(iter(state.objects))


def _position(obj: WorldObject) -> tuple[float, float, float] | None:
    value = obj.attributes.get("position")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


__all__ = [
    "HierarchicalWorldState",
    "Region",
    "WorldCausalIndex",
    "WorldCausalQuery",
    "WorldCausalSlice",
]
