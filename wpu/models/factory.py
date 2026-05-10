from __future__ import annotations

from torch import nn

from wpu.engines.scheduler import ExecutionPath
from wpu.models.baselines import DenseGraphProcessor, GraphTransformerProcessor, SerializedTokenProcessor
from wpu.models.world_state_processor import WorldStateProcessor


MODEL_NAMES = [
    "wpu-routed",
    "wpu-sparse",
    "wpu-hybrid",
    "wpu-dense",
    "dense-graph",
    "graph-transformer",
    "serialized-token",
]


def create_model(name: str, hidden_dim: int = 64) -> nn.Module:
    if name == "wpu-routed":
        return WorldStateProcessor(hidden_dim=hidden_dim)
    if name == "wpu-sparse":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.SPARSE)
    if name == "wpu-hybrid":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.HYBRID)
    if name == "wpu-dense":
        return WorldStateProcessor(hidden_dim=hidden_dim, forced_path=ExecutionPath.DENSE)
    if name == "dense-graph":
        return DenseGraphProcessor(hidden_dim=hidden_dim)
    if name == "graph-transformer":
        return GraphTransformerProcessor(hidden_dim=hidden_dim)
    if name == "serialized-token":
        return SerializedTokenProcessor(hidden_dim=hidden_dim)
    raise ValueError(f"unknown model: {name}")
