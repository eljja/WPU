from __future__ import annotations

from torch import nn

from wpu.engines.scheduler import ExecutionPath
from wpu.models.baselines import DenseGraphProcessor, GraphTransformerProcessor, SerializedTokenProcessor
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.world_state_processor import WorldStateProcessor


MODEL_NAMES = [
    "wpu-routed",
    "wpu-routed-target",
    "wpu-routed-frontier",
    "wpu-sparse",
    "wpu-sparse-target",
    "wpu-sparse-frontier",
    "wpu-hybrid",
    "wpu-hybrid-frontier",
    "wpu-dense",
    "wpu-cws-learned",
    "wpu-cws-target",
    "wpu-cws-frontier",
    "wpu-cws-indexed",
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "wpu-cws-indexed-adaptive-hybrid",
    "wpu-cws-indexed-learned-hybrid",
    "wpu-cws-indexed-learned-selective-hybrid",
    "wpu-cws-indexed-interaction-hybrid",
    "wpu-cws-indexed-selective-interaction-hybrid",
    "wpu-cws-indexed-geometry-hybrid",
    "wpu-cws-indexed-mechanism-conditioned",
    "wpu-cws-indexed-mechanism-adapter",
    "wpu-cws-indexed-mechanism-factorized",
    "wpu-cws-indexed-mechanism-branch",
    "wpu-cws-indexed-mechanism-branch-expert",
    "wpu-cws-indexed-mechanism-relation",
    "wpu-cws-indexed-mechanism-target",
    "wpu-cws-indexed-mechanism-target-constrained",
    "wpu-cws-indexed-regret-hybrid",
    "wpu-cws-indexed-physics-regret-hybrid",
    "wpu-cws-indexed-state-regret-hybrid",
    "wpu-cws-oracle",
    "dense-graph",
    "graph-transformer",
    "serialized-token",
]


def create_model(name: str, hidden_dim: int = 64, **kwargs: object) -> nn.Module:
    num_heads = _num_heads_for(hidden_dim, kwargs.get("num_heads"))
    if name == "wpu-routed":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads)
    if name == "wpu-routed-target":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads, branch_pooling="target")
    if name == "wpu-routed-frontier":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads, branch_pooling="frontier")
    if name == "wpu-sparse":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads, forced_path=ExecutionPath.SPARSE)
    if name == "wpu-sparse-target":
        return WorldStateProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            forced_path=ExecutionPath.SPARSE,
            branch_pooling="target",
        )
    if name == "wpu-sparse-frontier":
        return WorldStateProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            forced_path=ExecutionPath.SPARSE,
            branch_pooling="frontier",
        )
    if name == "wpu-hybrid":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads, forced_path=ExecutionPath.HYBRID)
    if name == "wpu-hybrid-frontier":
        return WorldStateProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            forced_path=ExecutionPath.HYBRID,
            branch_pooling="frontier",
        )
    if name == "wpu-dense":
        return WorldStateProcessor(hidden_dim=hidden_dim, num_heads=num_heads, forced_path=ExecutionPath.DENSE)
    if name in {
        "wpu-cws-indexed",
        "wpu-cws-indexed-sparse",
        "wpu-cws-indexed-local-dense",
        "wpu-cws-indexed-adaptive-hybrid",
        "wpu-cws-indexed-learned-hybrid",
        "wpu-cws-indexed-learned-selective-hybrid",
        "wpu-cws-indexed-interaction-hybrid",
        "wpu-cws-indexed-selective-interaction-hybrid",
        "wpu-cws-indexed-geometry-hybrid",
        "wpu-cws-indexed-mechanism-conditioned",
        "wpu-cws-indexed-mechanism-adapter",
        "wpu-cws-indexed-mechanism-factorized",
        "wpu-cws-indexed-mechanism-branch",
        "wpu-cws-indexed-mechanism-branch-expert",
        "wpu-cws-indexed-mechanism-relation",
        "wpu-cws-indexed-mechanism-target",
        "wpu-cws-indexed-mechanism-target-constrained",
        "wpu-cws-indexed-regret-hybrid",
        "wpu-cws-indexed-physics-regret-hybrid",
        "wpu-cws-indexed-state-regret-hybrid",
    }:
        working_set_size = int(kwargs.get("working_set_size", 16))
        layers = int(kwargs.get("layers", 2))
        interaction_dense_threshold = float(kwargs.get("interaction_dense_threshold", 0.15))
        route_regret_threshold = float(kwargs.get("route_regret_threshold", 0.0))
        bounded_delta_max = float(kwargs.get("bounded_delta_max", 0.0))
        bounded_delta_position_max = float(kwargs.get("bounded_delta_position_max", 0.0))
        bounded_delta_velocity_max = float(kwargs.get("bounded_delta_velocity_max", 0.0))
        adaptive_delta_bounds = bool(kwargs.get("adaptive_delta_bounds", False))
        adaptive_delta_min = float(kwargs.get("adaptive_delta_min", 0.01))
        adaptive_delta_max = float(kwargs.get("adaptive_delta_max", 0.5))
        return CausalWorkingSetProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            working_set_size=working_set_size,
            selector="indexed",
            local_dense=name in {"wpu-cws-indexed", "wpu-cws-indexed-local-dense"},
            adaptive_hybrid=name
            in {
                "wpu-cws-indexed-adaptive-hybrid",
                "wpu-cws-indexed-learned-hybrid",
                "wpu-cws-indexed-learned-selective-hybrid",
                "wpu-cws-indexed-interaction-hybrid",
                "wpu-cws-indexed-selective-interaction-hybrid",
                "wpu-cws-indexed-geometry-hybrid",
                "wpu-cws-indexed-mechanism-conditioned",
                "wpu-cws-indexed-mechanism-adapter",
                "wpu-cws-indexed-mechanism-factorized",
                "wpu-cws-indexed-mechanism-branch",
                "wpu-cws-indexed-mechanism-branch-expert",
                "wpu-cws-indexed-mechanism-relation",
                "wpu-cws-indexed-mechanism-target",
                "wpu-cws-indexed-mechanism-target-constrained",
                "wpu-cws-indexed-regret-hybrid",
                "wpu-cws-indexed-physics-regret-hybrid",
                "wpu-cws-indexed-state-regret-hybrid",
            },
            adaptive_route=_adaptive_route(name),
            interaction_dense_threshold=interaction_dense_threshold,
            route_regret_threshold=route_regret_threshold,
            bounded_delta_max=bounded_delta_max,
            bounded_delta_position_max=bounded_delta_position_max,
            bounded_delta_velocity_max=bounded_delta_velocity_max,
            adaptive_delta_bounds=adaptive_delta_bounds,
            adaptive_delta_min=adaptive_delta_min,
            adaptive_delta_max=adaptive_delta_max,
        )
    if name.startswith("wpu-cws-"):
        selector = name.removeprefix("wpu-cws-")
        working_set_size = int(kwargs.get("working_set_size", 16))
        layers = int(kwargs.get("layers", 2))
        return CausalWorkingSetProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            working_set_size=working_set_size,
            selector=selector,
            local_dense=True,
        )
    if name == "dense-graph":
        return DenseGraphProcessor(hidden_dim=hidden_dim, num_heads=num_heads)
    if name == "graph-transformer":
        return GraphTransformerProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=int(kwargs.get("layers", 2)),
        )
    if name == "serialized-token":
        return SerializedTokenProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=int(kwargs.get("layers", 2)),
        )
    raise ValueError(f"unknown model: {name}")


def _adaptive_route(name: str) -> str:
    if name == "wpu-cws-indexed-learned-hybrid":
        return "learned"
    if name == "wpu-cws-indexed-learned-selective-hybrid":
        return "learned_selective"
    if name == "wpu-cws-indexed-interaction-hybrid":
        return "interaction"
    if name == "wpu-cws-indexed-selective-interaction-hybrid":
        return "selective_interaction"
    if name == "wpu-cws-indexed-geometry-hybrid":
        return "geometry"
    if name == "wpu-cws-indexed-mechanism-conditioned":
        return "mechanism"
    if name == "wpu-cws-indexed-mechanism-adapter":
        return "mechanism_adapter"
    if name == "wpu-cws-indexed-mechanism-factorized":
        return "mechanism_factorized"
    if name == "wpu-cws-indexed-mechanism-branch":
        return "mechanism_branch"
    if name == "wpu-cws-indexed-mechanism-branch-expert":
        return "mechanism_branch_expert"
    if name == "wpu-cws-indexed-mechanism-relation":
        return "mechanism_relation"
    if name == "wpu-cws-indexed-mechanism-target":
        return "mechanism_target"
    if name == "wpu-cws-indexed-mechanism-target-constrained":
        return "mechanism_target_constrained"
    if name == "wpu-cws-indexed-regret-hybrid":
        return "regret"
    if name == "wpu-cws-indexed-physics-regret-hybrid":
        return "physics_regret"
    if name == "wpu-cws-indexed-state-regret-hybrid":
        return "state_regret"
    return "hard"


def _num_heads_for(hidden_dim: int, requested: object | None) -> int:
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if requested is not None:
        num_heads = int(requested)
        if num_heads < 1:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        return num_heads
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1
