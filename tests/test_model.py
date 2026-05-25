import torch

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples, create_robot_cup_state, create_touch_event
from wpu.data.working_set_physics import (
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
    collate_working_set_samples,
)
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


def test_causal_working_set_branch_loss_trains_delta_head() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, background_objects=32, causal_obstacles=2)
    batch, _, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = CausalWorkingSetProcessor(hidden_dim=32, num_heads=4, working_set_size=8, selector="oracle")
    prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss.backward()

    assert model.object_delta_head.weight.grad is not None
    assert model.object_delta_head.weight.grad.norm().item() > 0.0


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


def test_pairwise_interaction_mode_requires_causal_obstacles() -> None:
    dataset = WorkingSetPhysicsDataset(size=12, seed=9, causal_obstacles=8, balanced_labels=True, interaction_mode="pairwise")

    labels = [dataset[index].branch_label for index in range(len(dataset))]

    assert labels.count(0) == 4
    assert labels.count(1) == 4
    assert labels.count(2) == 4


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


def test_adaptive_hybrid_routes_by_working_set_pressure() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=5, background_objects=64, causal_obstacles=4)
    batch, _, _, _ = collate_working_set_samples([dataset[0]])
    sparse_model = CausalWorkingSetProcessor(
        hidden_dim=32,
        num_heads=4,
        layers=1,
        working_set_size=8,
        selector="target",
        adaptive_hybrid=True,
        adaptive_confidence_threshold=-1.0,
    )
    dense_model = create_model("wpu-cws-indexed-adaptive-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=8)

    sparse_prediction = sparse_model(batch, num_branches=3)
    dense_prediction = dense_model(batch, num_branches=3)

    assert sparse_model.last_working_set_stats is not None
    assert dense_model.last_working_set_stats is not None
    assert sparse_model.last_working_set_stats.sparse_ratio == 1.0
    assert dense_model.last_working_set_stats.local_dense_ratio == 1.0
    assert sparse_prediction.selected_paths[0].name == "SPARSE"
    assert dense_prediction.selected_paths[0].name == "HYBRID"


def test_learned_hybrid_gate_is_trainable() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, background_objects=32, causal_obstacles=4)
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model("wpu-cws-indexed-learned-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=8)
    prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss.backward()

    assert model.last_working_set_stats is not None
    assert 0.0 < model.last_working_set_stats.local_dense_ratio < 1.0
    assert model.route_gate[-1].weight.grad is not None
    assert model.route_gate[-1].weight.grad.norm().item() > 0.0


def test_interaction_hybrid_uses_state_geometry_for_route() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, _, _, _ = collate_working_set_samples([dataset[0]])
    model = create_model("wpu-cws-indexed-interaction-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    prediction = model(batch, num_branches=3)

    assert prediction.branch_probabilities.shape == (1, 3)
    assert model.last_working_set_stats is not None
    assert 0.0 <= model.last_working_set_stats.local_dense_ratio <= 1.0
    assert model.last_working_set_stats.dense_compute_ratio == 1.0
    assert model.last_working_set_stats.sparse_ratio < 1.0


def test_selective_interaction_hybrid_reports_actual_dense_compute() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, _, _, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model(
        "wpu-cws-indexed-selective-interaction-hybrid",
        hidden_dim=32,
        num_heads=4,
        layers=1,
        working_set_size=12,
    )

    prediction = model(batch, num_branches=3)

    assert prediction.branch_probabilities.shape == (2, 3)
    assert model.last_working_set_stats is not None
    assert 0.0 <= model.last_working_set_stats.dense_compute_ratio <= 1.0
    assert 0.0 <= model.last_working_set_stats.local_dense_ratio <= model.last_working_set_stats.dense_compute_ratio


def test_selective_interaction_threshold_is_configurable() -> None:
    strict_model = create_model(
        "wpu-cws-indexed-selective-interaction-hybrid",
        hidden_dim=32,
        num_heads=4,
        layers=1,
        working_set_size=12,
        interaction_dense_threshold=0.99,
    )

    assert isinstance(strict_model, CausalWorkingSetProcessor)
    assert strict_model.interaction_dense_threshold == 0.99


def test_learned_selective_route_has_compute_loss_and_eval_hard_route() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model(
        "wpu-cws-indexed-learned-selective-hybrid",
        hidden_dim=32,
        num_heads=4,
        layers=1,
        working_set_size=12,
        interaction_dense_threshold=0.5,
    )

    prediction = model(batch, num_branches=3)
    loss = torch.nn.functional.mse_loss(prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss = loss + 0.01 * model.route_compute_loss()
    loss = loss + model.route_distillation_loss()
    loss.backward()

    assert model.interaction_route_gate[-1].weight.grad is not None
    assert model.interaction_route_gate[-1].weight.grad.norm().item() > 0.0
    assert model.route_compute_loss().item() >= 0.0
    assert model.route_distillation_loss().item() >= 0.0

    model.eval()
    with torch.no_grad():
        model(batch, num_branches=3)
    assert model.last_working_set_stats is not None
    assert 0.0 <= model.last_working_set_stats.dense_compute_ratio <= 1.0


def test_geometry_hybrid_adds_state_geometry_without_dense_compute() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, _, _, _ = collate_working_set_samples([dataset[0]])
    model = create_model("wpu-cws-indexed-geometry-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    prediction = model(batch, num_branches=3)

    assert prediction.branch_probabilities.shape == (1, 3)
    assert model.last_working_set_stats is not None
    assert model.last_working_set_stats.local_dense_ratio == 0.0
    assert model.last_working_set_stats.dense_compute_ratio == 0.0
    assert model.last_working_set_stats.sparse_ratio == 1.0


def test_pre_tensor_indexed_collate_projects_state_before_tensorization() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=7, background_objects=128, causal_obstacles=4)

    batch, target_delta, labels, causal_k = collate_indexed_working_set_samples(
        [dataset[0]],
        max_nodes=8,
        max_depth=1,
    )

    assert batch.object_features.shape[1] == 8
    assert target_delta.shape[1] == batch.object_features.shape[1]
    assert labels.tolist() in ([0], [1], [2])
    assert causal_k.tolist() == [8]
    assert batch.object_ids is not None
    assert "context_00000" not in batch.object_ids[0]
