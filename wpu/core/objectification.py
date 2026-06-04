from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from wpu.core.state import DeltaState, WorldState


@dataclass(frozen=True, slots=True)
class ObjectificationReport:
    """Quality report for an objectified `WorldState`.

    The report is intentionally small and deterministic so it can be used in
    tests, demos, and experiment logs without pulling in model code.
    """

    object_count: int
    relation_count: int
    invalid_relation_count: int
    invalid_delta_count: int
    identity_coverage: float
    relation_validity: float
    object_confidence: float
    relation_confidence: float
    delta_validity: float
    delta_locality: float | None
    contract_score: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "object_count": self.object_count,
            "relation_count": self.relation_count,
            "invalid_relation_count": self.invalid_relation_count,
            "invalid_delta_count": self.invalid_delta_count,
            "identity_coverage": self.identity_coverage,
            "relation_validity": self.relation_validity,
            "object_confidence": self.object_confidence,
            "relation_confidence": self.relation_confidence,
            "delta_validity": self.delta_validity,
            "delta_locality": self.delta_locality,
            "contract_score": self.contract_score,
        }


def evaluate_objectification(
    state: WorldState,
    *,
    delta: DeltaState | None = None,
    expected_working_set: set[str] | None = None,
) -> ObjectificationReport:
    """Evaluate whether a `WorldState` satisfies the WPU object contract.

    This does not judge whether perception found the "right" objects. It checks
    whether the supplied objectified state is usable by WPU: stable identities,
    valid relation endpoints, bounded confidence, valid deltas, and optional
    locality against an expected causal working set.
    """

    object_ids = set(state.objects)
    object_count = len(state.objects)
    relation_count = len(state.relations)

    identity_coverage = _safe_ratio(
        sum(1 for object_id, obj in state.objects.items() if object_id and obj.id == object_id),
        object_count,
        empty_value=1.0,
    )
    invalid_relation_count = sum(
        1
        for relation in state.relations
        if relation.src not in object_ids or relation.dst not in object_ids or not relation.type
    )
    relation_validity = 1.0 - _safe_ratio(invalid_relation_count, relation_count, empty_value=0.0)
    object_confidence = _mean_confidence_quality([obj.confidence for obj in state.objects.values()])
    relation_confidence = _mean_confidence_quality([relation.confidence for relation in state.relations])

    invalid_delta_count = 0
    delta_locality: float | None = None
    if delta is not None:
        changed = set(delta.object_updates) | set(delta.changed_objects)
        invalid_delta_count = sum(1 for object_id in changed if object_id not in object_ids)
        if expected_working_set is not None:
            delta_locality = _safe_ratio(
                len(changed & expected_working_set),
                len(changed),
                empty_value=1.0,
            )
    delta_validity = 1.0 - _safe_ratio(
        invalid_delta_count,
        len(set(delta.object_updates) | set(delta.changed_objects)) if delta is not None else 0,
        empty_value=0.0,
    )

    score_parts = [
        identity_coverage,
        relation_validity,
        object_confidence,
        relation_confidence,
        delta_validity,
    ]
    if delta_locality is not None:
        score_parts.append(delta_locality)

    return ObjectificationReport(
        object_count=object_count,
        relation_count=relation_count,
        invalid_relation_count=invalid_relation_count,
        invalid_delta_count=invalid_delta_count,
        identity_coverage=identity_coverage,
        relation_validity=relation_validity,
        object_confidence=object_confidence,
        relation_confidence=relation_confidence,
        delta_validity=delta_validity,
        delta_locality=delta_locality,
        contract_score=fmean(score_parts),
    )


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return float(numerator) / float(denominator)


def _mean_confidence_quality(values: list[float]) -> float:
    if not values:
        return 1.0
    return fmean(float(value) if 0.0 <= float(value) <= 1.0 else 0.0 for value in values)
