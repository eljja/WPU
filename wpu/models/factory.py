from __future__ import annotations

from torch import nn

from wpu.engines.scheduler import ExecutionPath
from wpu.models.baselines import DenseGraphProcessor, GraphTransformerProcessor, SerializedTokenProcessor
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.world_state_processor import WorldStateProcessor


MODEL_NAMES = [
    "wpu-routed",
    "wpu-sparse",
    "wpu-hybrid",
    "wpu-dense",
    "wpu-cws-learned",
    "wpu-cws-target",
    "wpu-cws-frontier",
    "wpu-cws-indexed",
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "wpu-cws-oracle",
    "dense-graph",
    "graph-transformer",
    "serialized-token",
]


def create_model(name: str, hidden_dim: int = 64, **kwargs: object) -> nn.Module:
    if name == "wpu-routed":
        return WorldStateProcessor(hidden_dim=hidden_dim)
    if name == "wpu-sparse":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.SPARSE)
    if name == "wpu-hybrid":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.HYBRID)
    if name == "wpu-dense":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.DENSE)
    if name in {"wpu-cws-indexed", "wpu-cws-indexed-sparse", "wpu-cws-indexed-local-dense"}:
        working_set_size = int(kwargs.get("working_set_size", 16))
        layers = int(kwargs.get("layers", 2))
        num_heads = int(kwargs.get("num_heads", 8 if hidden_dim % 8 == 0 else 4))
        return CausalWorkingSetProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            working_set_size=working_set_size,
            selector="indexed",
            local_dense=name != "wpu-cws-indexed-sparse",
        )
    if name.startswith("wpu-cws-"):
        selector = name.removeprefix("wpu-cws-")
        working_set_size = int(kwargs.get("working_set_size", 16))
        layers = int(kwargs.get("layers", 2))
        num_heads = int(kwargs.get("num_heads", 8 if hidden_dim % 8 == 0 else 4))
        return CausalWorkingSetProcessor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            working_set_size=working_set_size,
            selector=selector,
            local_dense=True,
        )
    if name == "dense-graph":
        return DenseGraphProcessor(hidden_dim=hidden_dim, num_heads=int(kwargs.get("num_heads", 4)))
    if name == "graph-transformer":
        return GraphTransformerProcessor(
            hidden_dim=hidden_dim,
            num_heads=int(kwargs.get("num_heads", 4)),
            layers=int(kwargs.get("layers", 2)),
        )
    if name == "serialized-token":
        return SerializedTokenProcessor(
            hidden_dim=hidden_dim,
            num_heads=int(kwargs.get("num_heads", 4)),
            layers=int(kwargs.get("layers", 2)),
        )
    raise ValueError(f"unknown model: {name}")
