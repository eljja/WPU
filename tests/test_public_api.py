import pytest
import wpu
from pathlib import Path

from wpu.data.object_physics import create_robot_cup_state, create_touch_event


ROOT = Path(__file__).resolve().parents[1]


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


def test_readme_public_api_snippets_execute() -> None:
    for readme_name, heading in (
        ("README.md", "## Minimal Public API"),
        ("README.ko.md", "## 최소 Public API"),
    ):
        text = (ROOT / readme_name).read_text(encoding="utf-8")
        section = text.split(heading, 1)[1]
        snippet = section.split("```python", 1)[1].split("```", 1)[0]

        namespace: dict[str, object] = {}
        exec(snippet, namespace)
