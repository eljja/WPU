from __future__ import annotations

from dataclasses import dataclass
import sys

from wpu.core.state import Branch, DeltaState, WorldState


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    object_memory: int
    relation_memory: int
    delta_memory: int
    branch_memory: int

    @property
    def total(self) -> int:
        return self.object_memory + self.relation_memory + self.delta_memory + self.branch_memory


def estimate_memory(
    state: WorldState,
    deltas: list[DeltaState] | None = None,
    branches: list[Branch] | None = None,
) -> MemoryEstimate:
    deltas = deltas or []
    branches = branches or []
    object_memory = sum(sys.getsizeof(obj.to_dict() if hasattr(obj, "to_dict") else obj) for obj in state.objects.values())
    relation_memory = sum(sys.getsizeof(relation) for relation in state.relations)
    delta_memory = sum(sys.getsizeof(delta.to_dict()) for delta in deltas)
    branch_memory = sum(sys.getsizeof(branch.to_dict()) for branch in branches)
    return MemoryEstimate(
        object_memory=object_memory,
        relation_memory=relation_memory,
        delta_memory=delta_memory,
        branch_memory=branch_memory,
    )
