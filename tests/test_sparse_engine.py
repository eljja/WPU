from wpu.data.object_physics import create_robot_cup_state, create_touch_event
from wpu.engines.sparse_engine import SparsePropagationEngine


def test_sparse_propagation_reaches_related_objects() -> None:
    state = create_robot_cup_state()
    event = create_touch_event(force=0.4)
    result = SparsePropagationEngine(max_depth=2).sparse_propagate(state, event)

    assert "cup_001" in result.affected_objects
    assert "table_001" in result.affected_objects
    assert "hand_001" in result.affected_objects
    assert result.delta.relation_updates
