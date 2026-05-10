from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from wpu.engines.scheduler import ExecutionPath
from wpu.models.batch import EVENT_FEATURE_DIM, OBJECT_FEATURE_DIM, RELATION_FEATURE_DIM, StateGraphBatch
from wpu.models.world_state_processor import StatePrediction


class DenseGraphProcessor(nn.Module):
    """Dense object-set baseline with global attention over every object."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.object_encoder = nn.Linear(OBJECT_FEATURE_DIM, hidden_dim)
        self.event_encoder = nn.Linear(EVENT_FEATURE_DIM, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.transition = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.object_delta_head = nn.Linear(hidden_dim, OBJECT_FEATURE_DIM)
        self.relation_head = nn.Linear(hidden_dim * 2 + RELATION_FEATURE_DIM, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 8)

    def forward(
        self,
        batch: StateGraphBatch,
        horizon: int = 1,
        num_branches: int = 3,
        route_branches: int | None = None,
    ) -> StatePrediction:
        del horizon
        del route_branches
        hidden = self.object_encoder(batch.object_features)
        event_hidden = self.event_encoder(batch.event_features).unsqueeze(1)
        hidden = hidden + event_hidden
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=~batch.object_mask,
            need_weights=False,
        )
        hidden = (hidden + self.transition(attended)).masked_fill(~batch.object_mask.unsqueeze(-1), 0.0)
        branch_logits = self.branch_head(_masked_mean(hidden, batch.object_mask))[:, :num_branches]
        return StatePrediction(
            object_delta=self.object_delta_head(hidden),
            relation_logits=self._relation_logits(hidden, batch),
            uncertainty=torch.sigmoid(self.uncertainty_head(hidden)),
            branch_logits=branch_logits,
            branch_probabilities=F.softmax(branch_logits, dim=-1),
            selected_paths=[ExecutionPath.DENSE for _ in range(batch.object_features.size(0))],
        )

    def _relation_logits(self, hidden: torch.Tensor, batch: StateGraphBatch) -> torch.Tensor:
        logits = torch.zeros((hidden.size(0), batch.relation_indices.size(1), 1), device=hidden.device)
        for batch_index in range(hidden.size(0)):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                logits[batch_index, edge_index] = self.relation_head(
                    torch.cat([hidden[batch_index, src], hidden[batch_index, dst], batch.relation_features[batch_index, edge_index]])
                )
        return logits


class SerializedTokenProcessor(nn.Module):
    """Tokenized-state baseline that flattens objects, relations, and event into one sequence."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.object_encoder = nn.Linear(OBJECT_FEATURE_DIM, hidden_dim)
        self.relation_encoder = nn.Linear(RELATION_FEATURE_DIM, hidden_dim)
        self.event_encoder = nn.Linear(EVENT_FEATURE_DIM, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.object_delta_head = nn.Linear(hidden_dim, OBJECT_FEATURE_DIM)
        self.relation_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 8)

    def forward(
        self,
        batch: StateGraphBatch,
        horizon: int = 1,
        num_branches: int = 3,
        route_branches: int | None = None,
    ) -> StatePrediction:
        del horizon
        del route_branches
        object_tokens = self.object_encoder(batch.object_features)
        relation_tokens = self.relation_encoder(batch.relation_features)
        event_token = self.event_encoder(batch.event_features).unsqueeze(1)
        sequence = torch.cat([object_tokens, relation_tokens, event_token], dim=1)
        mask = torch.cat(
            [
                batch.object_mask,
                batch.relation_mask,
                torch.ones((batch.object_mask.size(0), 1), dtype=torch.bool, device=batch.object_mask.device),
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=~mask)
        object_encoded = encoded[:, : batch.object_features.size(1)].masked_fill(~batch.object_mask.unsqueeze(-1), 0.0)
        relation_start = batch.object_features.size(1)
        relation_end = relation_start + batch.relation_features.size(1)
        relation_encoded = encoded[:, relation_start:relation_end]
        branch_logits = self.branch_head(encoded[:, -1])[:, :num_branches]
        return StatePrediction(
            object_delta=self.object_delta_head(object_encoded),
            relation_logits=self.relation_head(relation_encoded),
            uncertainty=torch.sigmoid(self.uncertainty_head(object_encoded)),
            branch_logits=branch_logits,
            branch_probabilities=F.softmax(branch_logits, dim=-1),
            selected_paths=[ExecutionPath.DENSE for _ in range(batch.object_features.size(0))],
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), 0.0)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    return masked.sum(dim=1) / denom


class GraphTransformerProcessor(nn.Module):
    """Relation-aware dense graph Transformer baseline.

    This baseline keeps explicit object and relation tensors, but does not use
    WPU-style sparse routing or branch overlays. It is intentionally strong
    enough to test whether the WPU advantage is only "a GNN in disguise".
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.object_encoder = nn.Linear(OBJECT_FEATURE_DIM, hidden_dim)
        self.relation_encoder = nn.Linear(RELATION_FEATURE_DIM, hidden_dim)
        self.event_encoder = nn.Linear(EVENT_FEATURE_DIM, hidden_dim)
        self.relation_message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.object_delta_head = nn.Linear(hidden_dim, OBJECT_FEATURE_DIM)
        self.relation_head = nn.Linear(hidden_dim * 2 + RELATION_FEATURE_DIM, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 8)

    def forward(
        self,
        batch: StateGraphBatch,
        horizon: int = 1,
        num_branches: int = 3,
        route_branches: int | None = None,
    ) -> StatePrediction:
        del horizon
        del route_branches
        hidden = self.object_encoder(batch.object_features)
        event_hidden = self.event_encoder(batch.event_features).unsqueeze(1)
        relation_hidden = self.relation_encoder(batch.relation_features)
        hidden = hidden + event_hidden
        hidden = hidden + self._relation_messages(hidden, relation_hidden, batch)
        encoded = self.encoder(hidden, src_key_padding_mask=~batch.object_mask)
        encoded = encoded.masked_fill(~batch.object_mask.unsqueeze(-1), 0.0)
        branch_logits = self.branch_head(_masked_mean(encoded, batch.object_mask))[:, :num_branches]
        return StatePrediction(
            object_delta=self.object_delta_head(encoded),
            relation_logits=self._relation_logits(encoded, batch),
            uncertainty=torch.sigmoid(self.uncertainty_head(encoded)),
            branch_logits=branch_logits,
            branch_probabilities=F.softmax(branch_logits, dim=-1),
            selected_paths=[ExecutionPath.DENSE for _ in range(batch.object_features.size(0))],
        )

    def _relation_messages(
        self,
        hidden: torch.Tensor,
        relation_hidden: torch.Tensor,
        batch: StateGraphBatch,
    ) -> torch.Tensor:
        messages = torch.zeros_like(hidden)
        for batch_index in range(hidden.size(0)):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                relation = relation_hidden[batch_index, edge_index]
                src_message = self.relation_message(torch.cat([hidden[batch_index, src], relation, hidden[batch_index, dst]]))
                dst_message = self.relation_message(torch.cat([hidden[batch_index, dst], relation, hidden[batch_index, src]]))
                messages[batch_index, dst] = messages[batch_index, dst] + src_message
                messages[batch_index, src] = messages[batch_index, src] + dst_message
        return messages

    def _relation_logits(self, hidden: torch.Tensor, batch: StateGraphBatch) -> torch.Tensor:
        logits = torch.zeros((hidden.size(0), batch.relation_indices.size(1), 1), device=hidden.device)
        for batch_index in range(hidden.size(0)):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                logits[batch_index, edge_index] = self.relation_head(
                    torch.cat([hidden[batch_index, src], hidden[batch_index, dst], batch.relation_features[batch_index, edge_index]])
                )
        return logits
