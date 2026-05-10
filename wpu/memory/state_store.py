from __future__ import annotations

from dataclasses import dataclass, field

from wpu.core.state import Branch, DeltaState, Event, Relation, WorldObject, WorldState
from wpu.core.uncertainty import multiply_confidence
from wpu.memory.delta_store import DeltaStore
from wpu.memory.hierarchy import MemoryEstimate, estimate_memory


@dataclass(slots=True)
class StateStore:
    state: WorldState
    delta_store: DeltaStore = field(default_factory=DeltaStore)
    branches: dict[str, Branch] = field(default_factory=dict)

    def get_object(self, object_id: str) -> WorldObject:
        return self.state.objects[object_id]

    def relations_for(self, object_id: str) -> list[Relation]:
        return self.state.relations_for(object_id)

    def apply_event(self, event: Event) -> DeltaState:
        delta = DeltaState(time=event.time, metadata={"event_type": event.type, "target": event.target})
        if event.target not in self.state.objects:
            return delta

        obj = self.state.objects[event.target]
        updates = dict(event.delta)
        updates["confidence"] = multiply_confidence(obj.confidence, event.confidence)
        delta.record_object(event.target, updates)
        self.delta_store.append(delta)
        return delta

    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.id] = branch

    def memory_estimate(self) -> MemoryEstimate:
        return estimate_memory(self.state, self.delta_store.deltas, list(self.branches.values()))
