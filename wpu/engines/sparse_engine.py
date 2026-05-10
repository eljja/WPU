from __future__ import annotations

from dataclasses import dataclass, field

from wpu.core.state import DeltaState, Event, Relation, WorldState
from wpu.core.uncertainty import multiply_confidence


@dataclass(slots=True)
class PropagationResult:
    delta: DeltaState
    frontier_trace: list[list[str]] = field(default_factory=list)
    affected_objects: set[str] = field(default_factory=set)


class SparsePropagationEngine:
    def __init__(self, max_depth: int = 3, relation_threshold: float = 0.1) -> None:
        self.max_depth = max_depth
        self.relation_threshold = relation_threshold

    def sparse_propagate(self, state: WorldState, event: Event) -> PropagationResult:
        frontier = [event.target]
        visited: set[str] = set()
        result = PropagationResult(delta=DeltaState(time=event.time, metadata={"event_type": event.type}))

        for depth in range(self.max_depth):
            if not frontier:
                break
            result.frontier_trace.append(list(frontier))
            next_frontier: list[str] = []

            for object_id in frontier:
                if object_id in visited or object_id not in state.objects:
                    continue
                visited.add(object_id)
                result.affected_objects.add(object_id)

                updates = self._local_object_update(state, object_id, event, depth)
                if updates:
                    result.delta.record_object(object_id, updates)
                result.delta.relation_updates.extend(self._relation_updates(state, object_id, event))

                for relation in state.relations_for(object_id):
                    if relation.strength >= self.relation_threshold:
                        neighbor = relation.other(object_id)
                        if neighbor is not None and neighbor not in visited:
                            next_frontier.append(neighbor)

            frontier = _dedupe(next_frontier)

        return result

    def _local_object_update(self, state: WorldState, object_id: str, event: Event, depth: int) -> dict:
        obj = state.objects[object_id]
        updates: dict = {}
        if object_id == event.target:
            updates.update(event.delta)
            updates["confidence"] = multiply_confidence(obj.confidence, event.confidence)
        elif event.type in {"hand_touched_cup", "object_moved"}:
            decay = 0.5**depth
            updates["influence"] = round(float(event.confidence) * decay, 4)
            if obj.type == "cup":
                updates["fall_risk"] = min(1.0, float(obj.attributes.get("fall_risk", 0.0)) + 0.2 * decay)
        return updates

    def _relation_updates(self, state: WorldState, object_id: str, event: Event) -> list[Relation]:
        updates: list[Relation] = []
        for relation in state.relations_for(object_id):
            if event.type == "hand_touched_cup" and relation.type in {"on_top_of", "near"}:
                updates.append(
                    Relation(
                        src=relation.src,
                        dst=relation.dst,
                        type=relation.type,
                        strength=max(0.0, relation.strength - 0.1),
                        confidence=max(0.0, relation.confidence * event.confidence),
                        last_updated=event.time,
                    )
                )
        return updates


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
