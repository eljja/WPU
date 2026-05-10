from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from wpu.engines.scheduler import ExecutionPath, Scheduler
from wpu.models.batch import EVENT_FEATURE_DIM, OBJECT_FEATURE_DIM, RELATION_FEATURE_DIM, StateGraphBatch


@dataclass(slots=True)
class StatePrediction:
    object_delta: torch.Tensor
    relation_logits: torch.Tensor
    uncertainty: torch.Tensor
    branch_logits: torch.Tensor
    branch_probabilities: torch.Tensor
    selected_paths: list[ExecutionPath]


class WorldStateProcessor(nn.Module):
    def __init__(
        self,
        object_feature_dim: int = OBJECT_FEATURE_DIM,
        relation_feature_dim: int = RELATION_FEATURE_DIM,
        event_feature_dim: int = EVENT_FEATURE_DIM,
        hidden_dim: int = 64,
        num_heads: int = 4,
        forced_path: ExecutionPath | None = None,
    ) -> None:
        super().__init__()
        self.forced_path = forced_path
        self.scheduler = Scheduler()
        self.object_encoder = nn.Linear(object_feature_dim, hidden_dim)
        self.relation_encoder = nn.Linear(relation_feature_dim, hidden_dim)
        self.event_encoder = nn.Linear(event_feature_dim, hidden_dim)
        self.frontier_scorer = nn.Linear(hidden_dim * 2, 1)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_transition = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.global_transition = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.path_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.object_delta_head = nn.Linear(hidden_dim, object_feature_dim)
        self.relation_head = nn.Linear(hidden_dim * 3, 1)
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
        hidden = self.object_encoder(batch.object_features)
        event_hidden = self.event_encoder(batch.event_features).unsqueeze(1)
        selected_paths = self._selected_paths(batch, route_branches or num_branches)
        unique_paths = set(selected_paths)
        if unique_paths == {ExecutionPath.SPARSE}:
            routed_hidden = self._sparse_path(hidden, event_hidden, batch)
        elif unique_paths == {ExecutionPath.DENSE}:
            routed_hidden = self._dense_path(hidden, batch.object_mask)
        elif unique_paths == {ExecutionPath.HYBRID}:
            sparse_hidden = self._sparse_path(hidden, event_hidden, batch)
            dense_hidden = self._dense_path(hidden, batch.object_mask)
            event_expanded = event_hidden.expand(-1, hidden.size(1), -1)
            routed_hidden = self._hybrid_path(sparse_hidden, dense_hidden, hidden, event_expanded)
        else:
            sparse_hidden = self._sparse_path(hidden, event_hidden, batch)
            dense_hidden = self._dense_path(hidden, batch.object_mask)
            event_expanded = event_hidden.expand(-1, hidden.size(1), -1)
            hybrid_hidden = self._hybrid_path(sparse_hidden, dense_hidden, hidden, event_expanded)
            routed_hidden = self._route(selected_paths, sparse_hidden, hybrid_hidden, dense_hidden)
        routed_hidden = routed_hidden.masked_fill(~batch.object_mask.unsqueeze(-1), 0.0)

        object_delta = self.object_delta_head(routed_hidden)
        relation_logits = self._relation_logits(routed_hidden, batch)
        uncertainty = torch.sigmoid(self.uncertainty_head(routed_hidden))
        pooled = _masked_mean(routed_hidden, batch.object_mask)
        branch_logits = self.branch_head(pooled)[:, :num_branches]
        branch_probabilities = F.softmax(branch_logits, dim=-1)
        return StatePrediction(
            object_delta=object_delta,
            relation_logits=relation_logits,
            uncertainty=uncertainty,
            branch_logits=branch_logits,
            branch_probabilities=branch_probabilities,
            selected_paths=selected_paths,
        )

    def _sparse_path(self, hidden: torch.Tensor, event_hidden: torch.Tensor, batch: StateGraphBatch) -> torch.Tensor:
        batch_size, _, hidden_dim = hidden.shape
        messages = torch.zeros_like(hidden)
        relation_hidden = self.relation_encoder(batch.relation_features)

        for batch_index in range(batch_size):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                src_hidden = hidden[batch_index, src]
                rel_hidden = relation_hidden[batch_index, edge_index]
                evt_hidden = event_hidden[batch_index, 0]
                message = self.message_mlp(torch.cat([src_hidden, rel_hidden, evt_hidden], dim=-1))
                messages[batch_index, dst] = messages[batch_index, dst] + message
                reverse_message = self.message_mlp(torch.cat([hidden[batch_index, dst], rel_hidden, evt_hidden], dim=-1))
                messages[batch_index, src] = messages[batch_index, src] + reverse_message

        frontier_scores = torch.sigmoid(
            self.frontier_scorer(torch.cat([hidden, event_hidden.expand_as(hidden)], dim=-1))
        )
        target_boost = torch.zeros_like(frontier_scores)
        target_boost.scatter_(1, batch.target_indices.view(-1, 1, 1), 1.0)
        frontier_scores = torch.maximum(frontier_scores, target_boost)
        return hidden + self.local_transition(messages * frontier_scores)

    def _dense_path(self, hidden: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~object_mask
        attended, _ = self.global_attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return hidden + self.global_transition(attended)

    def _hybrid_path(
        self,
        sparse_hidden: torch.Tensor,
        dense_hidden: torch.Tensor,
        base_hidden: torch.Tensor,
        event_expanded: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.path_gate(torch.cat([base_hidden, event_expanded], dim=-1)))
        return sparse_hidden + gate * (dense_hidden - base_hidden)

    def _relation_logits(self, hidden: torch.Tensor, batch: StateGraphBatch) -> torch.Tensor:
        relation_hidden = self.relation_encoder(batch.relation_features)
        logits = torch.zeros((hidden.size(0), batch.relation_indices.size(1), 1), device=hidden.device)
        for batch_index in range(hidden.size(0)):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                logits[batch_index, edge_index] = self.relation_head(
                    torch.cat([hidden[batch_index, src], hidden[batch_index, dst], relation_hidden[batch_index, edge_index]])
                )
        return logits

    def _selected_paths(self, batch: StateGraphBatch, num_branches: int) -> list[ExecutionPath]:
        if self.forced_path is not None:
            return [self.forced_path for _ in range(batch.object_features.size(0))]
        if batch.scheduler_metrics is None:
            return [ExecutionPath.SPARSE for _ in range(batch.object_features.size(0))]
        decisions = []
        for metrics in batch.scheduler_metrics:
            adjusted_metrics = type(metrics)(
                delta_n=metrics.delta_n,
                fanout=metrics.fanout,
                depth=metrics.depth,
                branches=num_branches,
                total_n=metrics.total_n,
                uncertainty_growth=metrics.uncertainty_growth,
                time_budget_ms=metrics.time_budget_ms,
            )
            decisions.append(self.scheduler.choose_path(adjusted_metrics).path)
        return decisions

    def _route(
        self,
        selected_paths: list[ExecutionPath],
        sparse_hidden: torch.Tensor,
        hybrid_hidden: torch.Tensor,
        dense_hidden: torch.Tensor,
    ) -> torch.Tensor:
        routed = torch.empty_like(sparse_hidden)
        for index, path in enumerate(selected_paths):
            if path == ExecutionPath.SPARSE:
                routed[index] = sparse_hidden[index]
            elif path == ExecutionPath.HYBRID:
                routed[index] = hybrid_hidden[index]
            else:
                routed[index] = dense_hidden[index]
        return routed


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), 0.0)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    return masked.sum(dim=1) / denom
