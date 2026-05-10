from wpu.data.object_physics import create_robot_cup_state, create_touch_event
from wpu.engines.rollout_engine import rollout
from wpu.models.world_state_processor import WorldStateProcessor


def test_rollout_normalizes_branch_probabilities() -> None:
    steps = rollout(WorldStateProcessor(hidden_dim=32), create_robot_cup_state(), [create_touch_event()], horizon=2)

    assert len(steps) == 2
    for step in steps:
        total = sum(branch.probability for branch in step.branches)
        assert abs(total - 1.0) < 1e-6
