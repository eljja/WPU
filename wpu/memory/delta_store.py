from __future__ import annotations

from dataclasses import dataclass, field

from wpu.core.state import DeltaState


@dataclass(slots=True)
class DeltaStore:
    deltas: list[DeltaState] = field(default_factory=list)

    def append(self, delta: DeltaState) -> None:
        self.deltas.append(delta)

    def latest(self) -> DeltaState | None:
        return self.deltas[-1] if self.deltas else None

    def changed_object_count(self) -> int:
        return sum(len(delta.changed_objects) for delta in self.deltas)
