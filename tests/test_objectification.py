import pytest

import wpu
from wpu.core.state import DeltaState, Relation, WorldObject, WorldState
from wpu.data.object_physics import create_robot_cup_state
from wpu.engines.sparse_engine import SparsePropagationEngine


def test_objectification_report_scores_valid_state_contract() -> None:
    state = create_robot_cup_state()
    delta = DeltaState(time=1.0)
    delta.record_object("cup_001", {"fall_risk": 0.2})

    report = wpu.evaluate_objectification(
        state,
        delta=delta,
        expected_working_set={"cup_001", "hand_001", "table_001", "edge_001"},
    )

    assert report.object_count >= 4
    assert report.invalid_relation_count == 0
    assert report.invalid_delta_count == 0
    assert report.identity_coverage == 1.0
    assert report.relation_validity == 1.0
    assert report.delta_validity == 1.0
    assert report.delta_locality == 1.0
    assert report.contract_score > 0.9
    assert report.to_dict()["contract_score"] == pytest.approx(report.contract_score)


def test_objectification_report_measures_frontier_and_semantic_consistency() -> None:
    reference = WorldState()
    reference.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 0.0]}, confidence=0.9))
    reference.add_object(WorldObject("hand_001", "robot_hand", {"position": [0.1, 0.0, 0.0]}, confidence=0.9))
    reference.add_object(WorldObject("edge_001", "table_edge", {"position": [0.4, 0.0, 0.0]}, confidence=0.9))
    reference.add_relation(Relation("cup_001", "hand_001", "near", confidence=0.9))
    reference.add_relation(Relation("cup_001", "edge_001", "near_edge", confidence=0.9))

    corrupted = WorldState()
    corrupted.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 0.0]}, confidence=0.9))
    corrupted.add_object(WorldObject("hand_001", "robot_hand", {"position": [0.1, 0.0, 0.0]}, confidence=0.9))
    corrupted.add_object(WorldObject("edge_001", "background_object", {"position": [4.0, 0.0, 0.0]}, confidence=0.9))
    corrupted.add_relation(Relation("cup_001", "hand_001", "near", confidence=0.9))

    report = wpu.evaluate_objectification(
        corrupted,
        expected_working_set={"cup_001", "hand_001", "edge_001"},
        event_target="cup_001",
        reference_state=reference,
    )

    assert report.frontier_completeness == pytest.approx(2 / 3)
    assert report.semantic_identity_consistency == pytest.approx(2 / 3)
    assert report.to_dict()["frontier_completeness"] == pytest.approx(2 / 3)
    assert report.to_dict()["semantic_identity_consistency"] == pytest.approx(2 / 3)


def test_objectification_report_penalizes_broken_identity_relation_and_delta() -> None:
    state = WorldState()
    state.objects["cup_001"] = WorldObject("different_id", "cup", confidence=1.4)
    state.add_relation(Relation("cup_001", "missing_table", "on_top_of", confidence=-0.2))
    delta = DeltaState(time=1.0)
    delta.record_object("ghost_001", {"position": [0.0, 0.0, 0.0]})

    report = wpu.evaluate_objectification(
        state,
        delta=delta,
        expected_working_set={"cup_001"},
    )

    assert report.identity_coverage == 0.0
    assert report.invalid_relation_count == 1
    assert report.relation_validity == 0.0
    assert report.invalid_delta_count == 1
    assert report.delta_validity == 0.0
    assert report.delta_locality == 0.0
    assert report.contract_score < 0.35


def test_geometry_repair_adds_missing_relation_frontier_edges() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 0.0]}, confidence=0.9))
    state.add_object(WorldObject("hand_001", "robot_hand", {"position": [0.12, 0.0, 0.0]}, confidence=0.9))
    event = wpu.Event("hand_touched_cup", "cup_001", {"force": 0.4}, confidence=0.9)

    before = SparsePropagationEngine(max_depth=2).sparse_propagate(state, event)
    repaired, repair_report = wpu.repair_objectification_relations(state, near_distance=0.2)
    after = SparsePropagationEngine(max_depth=2).sparse_propagate(repaired, event)

    assert before.affected_objects == {"cup_001"}
    assert repair_report.added_relation_count == 1
    assert repair_report.added_relation_types == {"near": 1}
    assert "hand_001" in after.affected_objects
    assert repair_report.to_dict()["added_relation_count"] == 1


def test_type_gated_repair_rejects_near_background_distractors() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 0.0]}, confidence=0.9))
    state.add_object(WorldObject("hand_001", "robot_hand", {"position": [0.12, 0.0, 0.0]}, confidence=0.9))
    state.add_object(WorldObject("context_0001", "background_object", {"position": [0.05, 0.0, 0.0]}, confidence=0.7))

    ungated_delta, ungated = wpu.infer_missing_relations(state, near_distance=0.2)
    gated_delta, gated = wpu.infer_missing_relations(
        state,
        near_distance=0.2,
        allowed_type_pairs={("cup", "robot_hand")},
    )

    assert ungated.added_relation_count == 5
    assert ungated.added_relation_types == {"near": 3, "touching": 2}
    assert {relation.dst for relation in ungated_delta.relation_updates if relation.src == "cup_001"} >= {
        "hand_001",
        "context_0001",
    }
    assert gated.added_relation_count == 1
    assert gated.skipped_pair_count == 2
    assert [(relation.src, relation.dst, relation.type) for relation in gated_delta.relation_updates] == [
        ("cup_001", "hand_001", "near")
    ]


def test_law_revision_report_accepts_useful_objectified_rule_revision() -> None:
    hypothesis = wpu.LocalLawHypothesis(
        name="inverse_distance_revision",
        relation_type="influences",
        expression="gain * impulse / (distance^2 + c)",
        input_fields=("current_impulse", "distance", "relation_history"),
        parameters={"gain": 0.9, "c": 0.2},
        evidence={"calibration_mse": 0.001},
        status="revised",
    )

    report = wpu.evaluate_law_revision(
        base_error=0.115978,
        revised_error=0.000342,
        oracle_relation_error=0.000229,
        selected_hypothesis=hypothesis,
        calibration_samples=64,
    )

    assert report.decision == "accept_revision"
    assert report.relative_improvement > 0.99
    assert report.relation_selection_gap == pytest.approx(0.000113)
    assert report.law_residual_gap == pytest.approx(0.000229)
    assert report.to_dict()["selected_hypothesis"]["status"] == "revised"


def test_law_revision_report_rejects_weak_revision() -> None:
    hypothesis = wpu.LocalLawHypothesis(
        name="weak_revision",
        relation_type="influences",
        expression="gain * impulse",
        input_fields=("current_impulse",),
        parameters={"gain": 0.1},
        evidence={},
    )

    report = wpu.evaluate_law_revision(
        base_error=1.0,
        revised_error=0.95,
        selected_hypothesis=hypothesis,
        calibration_samples=8,
        min_relative_improvement=0.10,
    )

    assert report.decision == "keep_base_or_collect_data"
    assert report.relative_improvement == pytest.approx(0.05)
