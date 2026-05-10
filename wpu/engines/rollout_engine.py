from __future__ import annotations

from dataclasses import dataclass

import torch

from wpu.core.state import Branch, DeltaState, Event, WorldState
from wpu.models.batch import StateGraphBatch


@dataclass(slots=True)
class RolloutStep:
    step: int
    branches: list[Branch]


def rollout(
    model: torch.nn.Module,
    state: WorldState,
    actions: list[Event],
    horizon: int,
    num_branches: int = 3,
) -> list[RolloutStep]:
    active_states = [state]
    steps: list[RolloutStep] = []

    for step in range(horizon):
        new_branches: list[Branch] = []
        for state_index, active_state in enumerate(active_states):
            action = actions[min(step, len(actions) - 1)]
            batch = StateGraphBatch.from_world_states([active_state], [action])
            prediction = model(batch, horizon=1, num_branches=num_branches)
            probs = prediction.branch_probabilities.detach().cpu()[0]
            for branch_index, probability in enumerate(probs.tolist()):
                delta = DeltaState(time=action.time + step, metadata={"rollout_step": step})
                new_branches.append(
                    Branch(
                        id=f"step{step}_state{state_index}_branch{branch_index}",
                        parent_id=None if step == 0 else f"step{step - 1}_state{state_index}",
                        probability=float(probability),
                        delta_state=delta,
                        time=action.time + step,
                        label=f"branch_{branch_index}",
                    )
                )
        total = sum(branch.probability for branch in new_branches) or 1.0
        for branch in new_branches:
            branch.probability /= total
        steps.append(RolloutStep(step=step, branches=new_branches))
        active_states = [state.overlay_branch(branch) for branch in new_branches[:num_branches]]

    return steps
