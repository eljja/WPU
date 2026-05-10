from __future__ import annotations

from wpu.core.state import Branch, DeltaState
from wpu.data.object_physics import BRANCH_LABELS, create_robot_cup_state, create_touch_event
from wpu.engines.scheduler import Scheduler, SchedulerMetrics
from wpu.engines.sparse_engine import SparsePropagationEngine
from wpu.memory.state_store import StateStore
from wpu.models.batch import StateGraphBatch
from wpu.models.world_state_processor import WorldStateProcessor


def main() -> None:
    state = create_robot_cup_state()
    event = create_touch_event(force=0.55)
    store = StateStore(state)
    event_delta = store.apply_event(event)

    metrics = SchedulerMetrics(
        delta_n=max(len(event_delta.changed_objects), 1),
        fanout=len(state.relations) / max(len(state.objects), 1),
        depth=3,
        branches=3,
        total_n=len(state.objects),
        uncertainty_growth=1.0 - event.confidence,
    )
    decision = Scheduler().choose_path(metrics)
    sparse_result = SparsePropagationEngine(max_depth=2).sparse_propagate(state, event)

    model = WorldStateProcessor(hidden_dim=32)
    batch = StateGraphBatch.from_world_states([state], [event])
    prediction = model(batch, num_branches=3)

    branches: list[Branch] = []
    for index, probability in enumerate(prediction.branch_probabilities.detach().cpu()[0].tolist()):
        delta = DeltaState(time=event.time, metadata={"label": BRANCH_LABELS[index]})
        branch = Branch(
            id=f"branch_{BRANCH_LABELS[index]}",
            parent_id="base",
            probability=float(probability),
            delta_state=delta,
            time=event.time,
            label=BRANCH_LABELS[index],
        )
        branches.append(branch)
        store.add_branch(branch)

    memory = store.memory_estimate()

    print("Event: robot hand touched cup")
    print(f"Initial frontier: {event.target}")
    print(f"Scheduler path: {decision.path.value} ({decision.reason})")
    print(f"Model path: {prediction.selected_paths[0].value}")
    print(f"Frontier trace: {sparse_result.frontier_trace}")
    print(f"Changed objects: {sorted(sparse_result.affected_objects)}")
    print("Branches:")
    for branch in branches:
        print(f"  {branch.id}: {branch.label}, p={branch.probability:.3f}")
    print("Memory estimate:")
    print(f"  object memory: {memory.object_memory}")
    print(f"  relation memory: {memory.relation_memory}")
    print(f"  delta memory: {memory.delta_memory}")
    print(f"  branch memory: {memory.branch_memory}")
    print(f"  total: {memory.total}")


if __name__ == "__main__":
    main()
