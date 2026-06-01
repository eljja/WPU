import pytest
import wpu
from wpu.data.object_physics import create_robot_cup_state, create_touch_event


def test_root_package_exports_core_public_api() -> None:
    exported = set(wpu.__all__)
    expected = {
        "Branch",
        "DeltaState",
        "DenseRecomputeEngine",
        "Event",
        "ExecutionPath",
        "Scheduler",
        "SchedulerMetrics",
        "SparsePropagationEngine",
        "StateGraphBatch",
        "StateStore",
        "WorldState",
        "WorldStateProcessor",
        "rollout",
    }

    assert expected <= exported


def test_public_api_supports_minimal_state_processing_flow() -> None:
    state = create_robot_cup_state()
    event = create_touch_event()
    store = wpu.StateStore(state)

    delta = store.apply_event(event)
    sparse = wpu.SparsePropagationEngine(max_depth=1).sparse_propagate(state, event)
    dense = wpu.DenseRecomputeEngine().dense_recompute(state, region=["cup_001"])

    assert delta.object_updates["cup_001"]["force"] == 0.35
    assert delta.object_updates["cup_001"]["confidence"] == pytest.approx(
        state.objects["cup_001"].confidence * event.confidence
    )
    assert "cup_001" in sparse.affected_objects
    assert set(dense.delta.object_updates) == {"cup_001"}
