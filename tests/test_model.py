import torch

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples, create_robot_cup_state, create_touch_event
from wpu.data.working_set_physics import (
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
    collate_interaction_working_set_samples,
    collate_proximity_working_set_samples,
    collate_selected_working_set_samples,
    collate_working_set_samples,
)
from wpu.models.factory import create_model
from wpu.models.factory import MODEL_NAMES
from wpu.core.state import Event
from wpu.models.batch import EVENT_FEATURE_DIM, StateGraphBatch
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


def test_event_encoder_preserves_action_condition() -> None:
    state = create_robot_cup_state()
    base_delta = {"position": [0.1, 0.0, 0.0], "force": 1.0}
    no_catch = Event(
        type="simulated_hand_impulse",
        target="cup_001",
        delta={**base_delta, "catch_action": 0.0},
        confidence=0.98,
        time=1.0,
    )
    catch = Event(
        type="simulated_hand_impulse",
        target="cup_001",
        delta={**base_delta, "catch_action": 1.0},
        confidence=0.98,
        time=1.0,
    )

    batch = StateGraphBatch.from_world_states([state, state], [no_catch, catch])

    assert batch.event_features.shape[-1] == EVENT_FEATURE_DIM
    assert batch.event_features[0, 6].item() == 0.0
    assert batch.event_features[1, 6].item() == 1.0


def test_object_encoder_preserves_physical_state_scalars() -> None:
    state = create_robot_cup_state()
    cup = state.objects["cup_001"]
    cup.attributes["edge_distance"] = 0.23
    cup.attributes["hand_distance"] = 0.41
    cup.attributes["fall_risk"] = 0.17
    cup.attributes["angular_velocity"] = [0.0, 3.0, 4.0]

    batch = StateGraphBatch.from_world_states([state], [create_touch_event()])

    assert batch.object_features.shape[-1] == 12
    assert torch.isclose(batch.object_features[0, 0, 8], torch.tensor(0.23))
    assert torch.isclose(batch.object_features[0, 0, 9], torch.tensor(0.41))
    assert torch.isclose(batch.object_features[0, 0, 10], torch.tensor(0.17))
    assert torch.isclose(batch.object_features[0, 0, 11], torch.tensor(5.0))


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


def test_factory_selects_valid_attention_heads_for_small_hidden_dims() -> None:
    state_batch, _, _, _ = collate_working_set_samples(
        [WorkingSetPhysicsDataset(size=1, seed=5, background_objects=8, causal_obstacles=2)[0]]
    )
    names = [
        "wpu-routed",
        "wpu-cws-indexed",
        "dense-graph",
        "graph-transformer",
        "serialized-token",
    ]

    for name in names:
        model = create_model(name, hidden_dim=10, layers=1, working_set_size=4)
        prediction = model(state_batch, num_branches=3)
        assert prediction.branch_probabilities.shape == (1, 3)


def test_all_declared_model_names_are_factory_creatable() -> None:
    for name in MODEL_NAMES:
        model = create_model(name, hidden_dim=16, layers=1, working_set_size=4)
        assert model is not None, name


def test_factory_rejects_invalid_explicit_attention_heads() -> None:
    try:
        create_model("wpu-cws-indexed", hidden_dim=10, num_heads=4)
    except ValueError as error:
        assert "must be divisible" in str(error)
    else:
        raise AssertionError("factory accepted incompatible hidden_dim/num_heads")


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


def test_causal_working_set_can_force_counterfactual_routes() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, _, _, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model("wpu-cws-indexed-local-dense", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    sparse_prediction = model(batch, num_branches=3, force_route="sparse")
    sparse_stats = model.last_working_set_stats
    dense_prediction = model(batch, num_branches=3, force_route="local_dense")
    dense_stats = model.last_working_set_stats

    assert sparse_prediction.branch_probabilities.shape == dense_prediction.branch_probabilities.shape
    assert sparse_stats is not None
    assert dense_stats is not None
    assert sparse_stats.dense_compute_ratio == 0.0
    assert sparse_stats.local_dense_ratio == 0.0
    assert dense_stats.dense_compute_ratio == 1.0
    assert dense_stats.local_dense_ratio == 1.0


def test_forced_sparse_route_skips_dense_encoder_compute() -> None:
    class CountingEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, values, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            return values + 1.0

    dataset = WorkingSetPhysicsDataset(size=1, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, _, _, _ = collate_working_set_samples([dataset[0]])
    model = create_model("wpu-cws-indexed-local-dense", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)
    encoder = CountingEncoder()
    model.working_set_encoder = encoder

    model(batch, num_branches=3, force_route="sparse")
    assert encoder.calls == 0

    model(batch, num_branches=3, force_route="local_dense")
    assert encoder.calls == 1


def test_regret_hybrid_head_trains_from_counterfactual_losses() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model("wpu-cws-indexed-regret-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    sparse_prediction = model(batch, num_branches=3, force_route="sparse")
    sparse_loss = torch.nn.functional.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
    dense_prediction = model(batch, num_branches=3, force_route="local_dense")
    dense_loss = torch.nn.functional.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
    target_regret = (dense_loss - sparse_loss).detach()
    routed_prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(routed_prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(routed_prediction.branch_logits, labels)
    loss = loss + model.route_regret_loss(target_regret)
    loss.backward()

    assert model.route_regret_prediction().shape == labels.shape
    assert model.route_regret_head[-1].weight.grad is not None
    assert model.route_regret_head[-1].weight.grad.norm().item() > 0.0
    assert model.last_working_set_stats is not None
    assert 0.0 <= model.last_working_set_stats.dense_compute_ratio <= 1.0


def test_physics_regret_hybrid_head_uses_state_context() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model("wpu-cws-indexed-physics-regret-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    sparse_prediction = model(batch, num_branches=3, force_route="sparse")
    sparse_loss = torch.nn.functional.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
    dense_prediction = model(batch, num_branches=3, force_route="local_dense")
    dense_loss = torch.nn.functional.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
    target_regret = (dense_loss - sparse_loss).detach()
    routed_prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(routed_prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(routed_prediction.branch_logits, labels)
    loss = loss + model.route_regret_loss(target_regret)
    loss.backward()

    assert model.route_regret_prediction().shape == labels.shape
    assert model.route_regret_head[0].normalized_shape == (40,)
    assert model.route_regret_head[-1].weight.grad is not None
    assert model.route_regret_head[-1].weight.grad.norm().item() > 0.0


def test_state_regret_hybrid_head_excludes_hidden_summary() -> None:
    dataset = WorkingSetPhysicsDataset(size=2, seed=9, background_objects=32, causal_obstacles=8, interaction_mode="pairwise")
    batch, target_delta, labels, _ = collate_working_set_samples([dataset[0], dataset[1]])
    model = create_model("wpu-cws-indexed-state-regret-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=12)

    sparse_prediction = model(batch, num_branches=3, force_route="sparse")
    sparse_loss = torch.nn.functional.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
    dense_prediction = model(batch, num_branches=3, force_route="local_dense")
    dense_loss = torch.nn.functional.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
    target_regret = (dense_loss - sparse_loss).detach()
    routed_prediction = model(batch, num_branches=3)

    loss = torch.nn.functional.mse_loss(routed_prediction.object_delta, target_delta)
    loss = loss + torch.nn.functional.cross_entropy(routed_prediction.branch_logits, labels)
    loss = loss + model.route_regret_loss(target_regret)
    loss.backward()

    assert model.route_regret_prediction().shape == labels.shape
    assert model.route_regret_head[0].normalized_shape == (7,)
    assert model.route_regret_head[-1].weight.grad is not None
    assert model.route_regret_head[-1].weight.grad.norm().item() > 0.0


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


def test_pre_tensor_proximity_collate_prioritizes_physical_frontier() -> None:
    dataset = WorkingSetPhysicsDataset(
        size=1,
        seed=11,
        background_objects=128,
        causal_obstacles=16,
        interaction_mode="pairwise",
    )

    batch, target_delta, labels, causal_k = collate_proximity_working_set_samples(
        [dataset[0]],
        max_nodes=8,
        max_depth=1,
    )
    object_ids = batch.object_ids[0] if batch.object_ids is not None else []

    assert batch.object_features.shape[1] == 8
    assert target_delta.shape[1] == batch.object_features.shape[1]
    assert labels.tolist() in ([0], [1], [2])
    assert causal_k.tolist() == [20]
    assert object_ids[0] == "cup_001"
    assert any(object_id.startswith("obstacle_") for object_id in object_ids)
    assert "context_00000" not in object_ids

    model = create_model("wpu-cws-indexed-regret-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=8)
    prediction = model(batch, num_branches=3)
    assert prediction.branch_probabilities.shape == (1, 3)


def test_pre_tensor_interaction_collate_prioritizes_obstacle_pairs() -> None:
    dataset = WorkingSetPhysicsDataset(
        size=1,
        seed=11,
        background_objects=128,
        causal_obstacles=16,
        interaction_mode="pairwise",
    )

    batch, target_delta, labels, causal_k = collate_interaction_working_set_samples(
        [dataset[0]],
        max_nodes=4,
        max_depth=1,
    )
    object_ids = batch.object_ids[0] if batch.object_ids is not None else []

    assert batch.object_features.shape[1] == 4
    assert target_delta.shape[1] == batch.object_features.shape[1]
    assert labels.tolist() in ([0], [1], [2])
    assert causal_k.tolist() == [20]
    assert object_ids[0] == "cup_001"
    assert "hand_001" in object_ids
    assert sum(object_id.startswith("obstacle_") for object_id in object_ids) >= 2

    model = create_model("wpu-cws-indexed-regret-hybrid", hidden_dim=32, num_heads=4, layers=1, working_set_size=4)
    prediction = model(batch, num_branches=3)
    assert prediction.branch_probabilities.shape == (1, 3)


def test_pre_tensor_selected_collate_uses_explicit_object_ids() -> None:
    dataset = WorkingSetPhysicsDataset(size=1, seed=11, background_objects=128, causal_obstacles=16)
    selected_ids = [["cup_001", "hand_001", "obstacle_000", "obstacle_001"]]

    batch, target_delta, labels, causal_k = collate_selected_working_set_samples([dataset[0]], selected_ids)

    assert batch.object_ids == selected_ids
    assert batch.object_features.shape[1] == 4
    assert target_delta.shape[1] == 4
    assert labels.tolist() in ([0], [1], [2])
    assert causal_k.tolist() == [20]
