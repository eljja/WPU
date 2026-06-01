import pytest

from wpu.core.state import WorldObject, WorldState
from wpu.engines.dense_engine import DenseRecomputeEngine


def test_dense_recompute_limits_updates_to_requested_region() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 1.0]}, 0.9))
    state.add_object(WorldObject("table_001", "table", {"position": [0.0, 0.0, 0.0]}, 0.5))
    state.add_object(WorldObject("hand_001", "robot_hand", {"position": [0.2, 0.0, 1.0]}, 0.1))

    result = DenseRecomputeEngine().dense_recompute(state, region=["cup_001", "hand_001"])

    assert result.region == ["cup_001", "hand_001"]
    assert set(result.delta.object_updates) == {"cup_001", "hand_001"}
    assert "table_001" not in result.delta.object_updates
    assert result.delta.object_updates["cup_001"]["confidence"] == pytest.approx(0.8)
    assert result.delta.object_updates["hand_001"]["confidence"] == pytest.approx(0.2)
    assert result.tensor.shape == (2, 8)
