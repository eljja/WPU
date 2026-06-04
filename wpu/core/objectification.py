from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from wpu.core.state import DeltaState, Relation, WorldState


SYMMETRIC_RELATIONS = {"near", "touching"}


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


@dataclass(frozen=True, slots=True)
class ObjectificationRepairReport:
    """Report for deterministic objectification repair heuristics."""

    added_relation_count: int
    added_relation_types: dict[str, int]
    candidate_pair_count: int
    skipped_pair_count: int = 0

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "added_relation_count": self.added_relation_count,
            "added_relation_types": dict(self.added_relation_types),
            "candidate_pair_count": self.candidate_pair_count,
            "skipped_pair_count": self.skipped_pair_count,
        }


@dataclass(frozen=True, slots=True)
class LocalLawHypothesis:
    """Interpretable local rule attached to objectified relations.

    This is a metadata contract, not a physics solver. It records the current
    best local rule, the object/relation variables it depends on, and the
    evidence accumulated so far so experiments can revise or reject the rule.
    """

    name: str
    relation_type: str
    expression: str
    input_fields: tuple[str, ...]
    parameters: dict[str, float]
    evidence: dict[str, float]
    status: str = "candidate"

    def to_dict(self) -> dict[str, str | tuple[str, ...] | dict[str, float]]:
        return {
            "name": self.name,
            "relation_type": self.relation_type,
            "expression": self.expression,
            "input_fields": self.input_fields,
            "parameters": dict(self.parameters),
            "evidence": dict(self.evidence),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class LawRevisionReport:
    """Report for stress-driven local-law revision over objectified relations."""

    base_error: float
    revised_error: float
    relative_improvement: float
    decision: str
    selected_hypothesis: LocalLawHypothesis
    calibration_samples: int
    oracle_relation_error: float | None = None
    relation_selection_gap: float | None = None
    law_residual_gap: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None | dict[str, object]]:
        return {
            "base_error": self.base_error,
            "revised_error": self.revised_error,
            "relative_improvement": self.relative_improvement,
            "decision": self.decision,
            "selected_hypothesis": self.selected_hypothesis.to_dict(),
            "calibration_samples": self.calibration_samples,
            "oracle_relation_error": self.oracle_relation_error,
            "relation_selection_gap": self.relation_selection_gap,
            "law_residual_gap": self.law_residual_gap,
        }


def evaluate_law_revision(
    *,
    base_error: float,
    revised_error: float,
    selected_hypothesis: LocalLawHypothesis,
    calibration_samples: int,
    oracle_relation_error: float | None = None,
    min_relative_improvement: float = 0.10,
) -> LawRevisionReport:
    """Evaluate whether a local-law revision should be accepted.

    `base_error` and `revised_error` are downstream errors measured on the same
    held-out objectified state distribution. If `oracle_relation_error` is
    provided, the report separates error caused by imperfect relation selection
    from error left by the candidate law family itself.
    """

    if base_error < 0.0 or revised_error < 0.0:
        raise ValueError("errors must be non-negative")
    if oracle_relation_error is not None and oracle_relation_error < 0.0:
        raise ValueError("oracle_relation_error must be non-negative")
    if calibration_samples < 0:
        raise ValueError("calibration_samples must be non-negative")

    improvement = max(0.0, base_error - revised_error)
    relative_improvement = improvement / base_error if base_error > 0.0 else 0.0
    decision = "accept_revision" if relative_improvement >= min_relative_improvement else "keep_base_or_collect_data"

    relation_selection_gap: float | None = None
    law_residual_gap: float | None = None
    if oracle_relation_error is not None:
        relation_selection_gap = max(0.0, revised_error - oracle_relation_error)
        law_residual_gap = oracle_relation_error

    return LawRevisionReport(
        base_error=base_error,
        revised_error=revised_error,
        relative_improvement=relative_improvement,
        decision=decision,
        selected_hypothesis=selected_hypothesis,
        calibration_samples=calibration_samples,
        oracle_relation_error=oracle_relation_error,
        relation_selection_gap=relation_selection_gap,
        law_residual_gap=law_residual_gap,
    )


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


def infer_missing_relations(
    state: WorldState,
    *,
    near_distance: float = 0.35,
    contact_distance: float = 0.08,
    confidence: float = 0.55,
    allowed_type_pairs: set[tuple[str, str]] | None = None,
) -> tuple[DeltaState, ObjectificationRepairReport]:
    """Infer conservative relation patches from object attributes.

    This is a deterministic objectification-repair primitive, not a learned
    physics solver. It adds weak `near` and `touching` candidate relations when
    typed object positions make the relation locally plausible. The intended use
    is to recover sparse frontier connectivity when object identity exists but
    relation extraction missed an edge. `allowed_type_pairs` is a type-aware
    safety gate for avoiding geometry-only edges to distractor objects.
    """

    if near_distance <= 0.0:
        raise ValueError("near_distance must be positive")
    if contact_distance < 0.0:
        raise ValueError("contact_distance must be non-negative")

    existing = {_relation_key(relation.src, relation.dst, relation.type) for relation in state.relations}
    object_ids = list(state.objects)
    delta = DeltaState(time=state.time, metadata={"objectification_repair": "geometry"})
    added_types: dict[str, int] = {}
    candidate_pair_count = 0
    skipped_pair_count = 0
    normalized_allowed = (
        {_type_pair_key(left, right) for left, right in allowed_type_pairs}
        if allowed_type_pairs is not None
        else None
    )

    for left_index, left_id in enumerate(object_ids):
        left_position = _position3(state, left_id)
        if left_position is None:
            continue
        for right_id in object_ids[left_index + 1 :]:
            if normalized_allowed is not None:
                type_pair = _type_pair_key(state.objects[left_id].type, state.objects[right_id].type)
                if type_pair not in normalized_allowed:
                    skipped_pair_count += 1
                    continue
            right_position = _position3(state, right_id)
            if right_position is None:
                continue
            candidate_pair_count += 1
            distance = _distance3(left_position, right_position)
            if distance <= near_distance:
                strength = max(0.05, 1.0 - distance / near_distance)
                _append_relation_patch(
                    delta,
                    existing,
                    added_types,
                    Relation(left_id, right_id, "near", strength=strength, confidence=confidence),
                )
            if distance <= contact_distance:
                strength = max(0.05, 1.0 - distance / max(contact_distance, 1e-6))
                _append_relation_patch(
                    delta,
                    existing,
                    added_types,
                    Relation(left_id, right_id, "touching", strength=strength, confidence=confidence),
                )

    return delta, ObjectificationRepairReport(
        added_relation_count=len(delta.relation_updates),
        added_relation_types=added_types,
        candidate_pair_count=candidate_pair_count,
        skipped_pair_count=skipped_pair_count,
    )


def repair_objectification_relations(
    state: WorldState,
    *,
    near_distance: float = 0.35,
    contact_distance: float = 0.08,
    confidence: float = 0.55,
    allowed_type_pairs: set[tuple[str, str]] | None = None,
) -> tuple[WorldState, ObjectificationRepairReport]:
    """Return a state with conservative geometry-inferred relation patches."""

    delta, report = infer_missing_relations(
        state,
        near_distance=near_distance,
        contact_distance=contact_distance,
        confidence=confidence,
        allowed_type_pairs=allowed_type_pairs,
    )
    return state.apply_delta(delta), report


def _safe_ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return float(numerator) / float(denominator)


def _mean_confidence_quality(values: list[float]) -> float:
    if not values:
        return 1.0
    return fmean(float(value) if 0.0 <= float(value) <= 1.0 else 0.0 for value in values)


def _append_relation_patch(
    delta: DeltaState,
    existing: set[tuple[str, str, str]],
    added_types: dict[str, int],
    relation: Relation,
) -> None:
    key = _relation_key(relation.src, relation.dst, relation.type)
    if key in existing:
        return
    existing.add(key)
    delta.relation_updates.append(relation)
    added_types[relation.type] = added_types.get(relation.type, 0) + 1


def _relation_key(src: str, dst: str, relation_type: str) -> tuple[str, str, str]:
    if relation_type in SYMMETRIC_RELATIONS and dst < src:
        return dst, src, relation_type
    return src, dst, relation_type


def _type_pair_key(left_type: str, right_type: str) -> tuple[str, str]:
    return (left_type, right_type) if left_type <= right_type else (right_type, left_type)


def _position3(state: WorldState, object_id: str) -> tuple[float, float, float] | None:
    position = state.objects[object_id].attributes.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    return float(position[0]), float(position[1]), float(position[2])


def _distance3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2) ** 0.5
