import pytest
import wpu
from pathlib import Path

from wpu.data.object_physics import create_robot_cup_state, create_touch_event


ROOT = Path(__file__).resolve().parents[1]


def test_root_package_exports_core_public_api() -> None:
    exported = set(wpu.__all__)
    expected = {
        "Branch",
        "CausalWorkingSetProcessor",
        "DeltaState",
        "DeltaStore",
        "DenseRecomputeEngine",
        "Event",
        "ExecutionPath",
        "MODEL_NAMES",
        "MemoryEstimate",
        "ObjectificationReport",
        "Scheduler",
        "SchedulerMetrics",
        "SparsePropagationEngine",
        "StateGraphBatch",
        "StateStore",
        "WorldState",
        "WorldStateProcessor",
        "create_model",
        "evaluate_objectification",
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


def test_root_package_exposes_current_v2_model_factory() -> None:
    model = wpu.create_model(
        "wpu-cws-indexed",
        hidden_dim=16,
        num_heads=4,
        layers=1,
        working_set_size=4,
    )

    assert "wpu-cws-indexed" in wpu.MODEL_NAMES
    assert isinstance(model, wpu.CausalWorkingSetProcessor)


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


def test_readme_v2_factory_snippets_execute() -> None:
    for readme_name, heading in (
        ("README.md", "The current v2 working-set models"),
        ("README.ko.md", "현재 v2 working-set 모델"),
    ):
        text = (ROOT / readme_name).read_text(encoding="utf-8")
        section = text.split(heading, 1)[1]
        snippet = section.split("```python", 1)[1].split("```", 1)[0]

        namespace: dict[str, object] = {}
        exec(snippet, namespace)

        assert isinstance(namespace["model"], wpu.CausalWorkingSetProcessor)
