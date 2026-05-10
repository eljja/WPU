from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wpu.core.state import DeltaState, WorldState


@dataclass(slots=True)
class DenseRecomputeResult:
    delta: DeltaState
    tensor: np.ndarray
    region: list[str] | None


class DenseRecomputeEngine:
    def project_state_to_tensor(self, state: WorldState, region: list[str] | None = None) -> np.ndarray:
        object_ids = region or list(state.objects)
        rows: list[list[float]] = []
        for object_id in object_ids:
            obj = state.objects[object_id]
            position = _vector(obj.attributes.get("position", [0.0, 0.0, 0.0]), 3)
            velocity = _vector(obj.attributes.get("velocity", [0.0, 0.0, 0.0]), 3)
            rows.append([_stable_hash(obj.type), *position, *velocity, obj.confidence])
        return np.array(rows, dtype=np.float32)

    def dense_recompute(self, state: WorldState, region: list[str] | None = None) -> DenseRecomputeResult:
        tensor = self.project_state_to_tensor(state, region)
        updated = self._placeholder_dense_update(tensor)
        object_ids = region or list(state.objects)
        delta = DeltaState(time=state.time, metadata={"engine": "dense"})
        for row_index, object_id in enumerate(object_ids):
            delta.record_object(
                object_id,
                {
                    "dense_score": float(updated[row_index, -1]),
                    "confidence": float(np.clip(updated[row_index, -1], 0.0, 1.0)),
                },
            )
        return DenseRecomputeResult(delta=delta, tensor=updated, region=region)

    def _placeholder_dense_update(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.size == 0:
            return tensor
        updated = tensor.copy()
        global_confidence = float(np.mean(updated[:, -1]))
        updated[:, -1] = 0.75 * updated[:, -1] + 0.25 * global_confidence
        return updated


def _vector(value: object, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [0.0] * size
    result = [float(item) for item in value[:size]]
    return result + [0.0] * (size - len(result))


def _stable_hash(value: str) -> float:
    return float(sum(ord(ch) for ch in value) % 997) / 997.0
