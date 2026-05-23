import torch

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples, create_robot_cup_state, create_touch_event
from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_working_set_samples
from wpu.models.factory import create_model
from wpu.models.batch import StateGraphBatch
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.world_state_processor import WorldStateProcessor


def test_world_state_processor_forward_shapes() -> None:
    state = create_robot_cup_state()
    batch = StateGraphBatch.from_world_states([state], [create_touch_event()])
    model = WorldStateProcessor(hidden_dim=32)

    prediction = model(batch, num_branches=3)

    assert prediction.object_delta.shape[:2] == batch.object_features.shape[:2]
    assert prediction.relation_logits.shape[:2] == batch.relation_features.shape[:2]
    assert prediction.branch_probabilities.shape == (1, 3)
    assert torch.allclose(prediction.branch_probabilities.sum(dim=-1), torch.ones(1))


def test_training_smoke_backward() -> None:
    samples = [ObjectPhysicsDataset(size=2)[0], ObjectPhysicsDataset(size=2)[1]]
    batch, target_delta, labels = collate_physics_samples(samples)
    model = WorldStateProcessor(hidden_dim=32)
    prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_causal_working_set_processor_large_n_shapes() -> None:
    samples = [WorkingSetPhysicsDataset(size=2, background_objects=64, causal_obstacles=4)[0]]
    batch, target_delta, labels, _ = collate_working_set_samples(samples)
    model = CausalWorkingSetProcessor(hidden_dim=32, num_heads=4, working_set_size=8, selector="frontier")

    prediction = model(batch, num_branches=3)

    assert prediction.object_delta.shape == target_delta.shape
    assert prediction.branch_probabilities.shape == (1, 3)
    assert torch.allclose(prediction.branch_probabilities.sum(dim=-1), torch.ones(1))
    assert model.last_working_set_stats is not None
    assert model.last_working_set_stats.max_selected <= 8


def test_causal_working_set_training_smoke_backward() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, background_objects=32, causal_obstacles=2)
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = CausalWorkingSetProcessor(hidden_dim=32, num_heads=4, working_set_size=8, selector="oracle")
    prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_causal_working_set_selector_loss_backward() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, background_objects=32, causal_obstacles=2)
    batch, _, _, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = CausalWorkingSetProcessor(hidden_dim=32, num_heads=4, working_set_size=8, selector="learned")
    model(batch, num_branches=3)

    loss = model.selector_loss()
    loss.backward()

    assert loss.item() > 0.0
    assert any(param.grad is not None for param in model.relevance_scorer.parameters())


def test_working_set_dataset_can_balance_branch_labels() -> None:
    dataset = WorkingSetPhysicsDataset(size=9, seed=3, balanced_labels=True)

    labels = [dataset[index].branch_label for index in range(len(dataset))]

    assert labels.count(0) == 3
    assert labels.count(1) == 3
    assert labels.count(2) == 3


def test_indexed_cws_model_uses_relation_frontier() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=5, background_objects=64, causal_obstacles=4)
    batch, target_delta, _, _ = collate_working_set_samples([dataset[0]])
    model = create_model("wpu-cws-indexed", hidden_dim=32, num_heads=4, layers=1, working_set_size=8)

    prediction = model(batch, num_branches=3)

    assert prediction.object_delta.shape == target_delta.shape
    assert model.last_working_set_stats is not None
    assert model.last_working_set_stats.max_selected <= 8
    assert model.last_working_set_stats.mean_causal_recall > 0.0


def test_indexed_sparse_cws_model_disables_local_dense_block() -> None:
    model = create_model("wpu-cws-indexed-sparse", hidden_dim=32, num_heads=4, layers=1, working_set_size=8)

    assert isinstance(model, CausalWorkingSetProcessor)
    assert model.local_dense is False
