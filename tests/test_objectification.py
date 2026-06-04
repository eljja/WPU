import pytest

import wpu
from wpu.core.state import DeltaState, Relation, WorldObject, WorldState
from wpu.data.object_physics import create_robot_cup_state


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
