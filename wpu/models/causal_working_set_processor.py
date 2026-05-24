from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from wpu.core.causal_index import CausalIndex, CausalIndexQuery
from wpu.engines.scheduler import ExecutionPath
from wpu.models.batch import EVENT_FEATURE_DIM, OBJECT_FEATURE_DIM, RELATION_FEATURE_DIM, StateGraphBatch
from wpu.models.world_state_processor import StatePrediction


@dataclass(frozen=True, slots=True)
class WorkingSetStats:
    mean_selected: float
    max_selected: int
    mean_causal_recall: float
    sparse_ratio: float = 1.0
    local_dense_ratio: float = 0.0
    dense_compute_ratio: float = 0.0
    mean_selector_confidence: float = 0.0


class CausalWorkingSetProcessor(nn.Module):
    """WPU v2 candidate that predicts over an event-conditioned causal subset.

    The model keeps the public `StatePrediction` API, but its branch decision and
    non-zero deltas are computed from a bounded working set `K` rather than from
    a global mean over all `N` objects.
    """

    def __init__(
        self,
        object_feature_dim: int = OBJECT_FEATURE_DIM,
        relation_feature_dim: int = RELATION_FEATURE_DIM,
        event_feature_dim: int = EVENT_FEATURE_DIM,
        hidden_dim: int = 256,
        num_heads: int = 8,
        layers: int = 2,
        working_set_size: int = 16,
        selector: str = "learned",
        local_dense: bool = True,
        adaptive_hybrid: bool = False,
        adaptive_route: str = "hard",
        adaptive_confidence_threshold: float = 0.45,
        adaptive_k_threshold: int | None = None,
    ) -> None:
        super().__init__()
        if selector not in {"learned", "target", "frontier", "indexed", "oracle"}:
            raise ValueError(f"unknown selector: {selector}")
        if adaptive_route not in {"hard", "learned", "interaction", "geometry"}:
            raise ValueError(f"unknown adaptive route: {adaptive_route}")
        self.selector = selector
        self.working_set_size = working_set_size
        self.local_dense = local_dense
        self.adaptive_hybrid = adaptive_hybrid
        self.adaptive_route = adaptive_route
        self.adaptive_confidence_threshold = adaptive_confidence_threshold
        self.adaptive_k_threshold = adaptive_k_threshold or max(4, int(working_set_size * 0.75))
        self.object_encoder = nn.Linear(object_feature_dim, hidden_dim)
        self.relation_encoder = nn.Linear(relation_feature_dim, hidden_dim)
        self.event_encoder = nn.Linear(event_feature_dim, hidden_dim)
        self.relevance_scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.working_set_encoder = (
            nn.TransformerEncoder(encoder_layer, num_layers=layers) if local_dense or adaptive_hybrid else nn.Identity()
        )
        self.object_delta_head = nn.Linear(hidden_dim, object_feature_dim)
        self.relation_head = nn.Linear(hidden_dim * 2 + relation_feature_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.delta_branch_encoder = nn.Sequential(
            nn.LayerNorm(object_feature_dim),
            nn.Linear(object_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.route_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, max(hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )
        self.geometry_context_encoder = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        self.branch_head = nn.Linear(hidden_dim * 2, 8)
        self.last_working_set_stats: WorkingSetStats | None = None
        self._last_relevance_logits: torch.Tensor | None = None
        self._last_batch: StateGraphBatch | None = None

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
        event_hidden = self.event_encoder(batch.event_features)
        relevance_logits = self._relevance_logits(hidden, event_hidden, batch.object_mask)
        self._last_relevance_logits = relevance_logits
        self._last_batch = batch
        selected_indices, selected_mask = self._select_indices(batch, relevance_logits)

        selected_counts = selected_mask.sum(dim=1)
        selector_confidence = self._selection_confidence(relevance_logits, selected_indices, selected_mask)

        sparse_gathered = _batched_gather(hidden, selected_indices)
        sparse_gathered = sparse_gathered + event_hidden.unsqueeze(1)
        selected_object_features = _batched_gather(batch.object_features, selected_indices)
        interaction_density = _interaction_density(selected_object_features, selected_mask)

        if self.adaptive_hybrid and self.adaptive_route == "geometry":
            dense_gathered = sparse_gathered
            dense_compute_weight = torch.zeros_like(selector_confidence)
        elif self.local_dense or self.adaptive_hybrid:
            dense_gathered = self.working_set_encoder(sparse_gathered, src_key_padding_mask=~selected_mask)
            dense_compute_weight = torch.ones_like(selector_confidence)
        else:
            dense_gathered = self.working_set_encoder(sparse_gathered)
            dense_compute_weight = torch.zeros_like(selector_confidence)

        if self.adaptive_hybrid and self.adaptive_route == "learned":
            dense_weight = self._learned_dense_weight(
                sparse_gathered,
                selected_mask,
                selected_counts,
                selector_confidence,
            )
            gathered = sparse_gathered * (1.0 - dense_weight.view(-1, 1, 1)) + dense_gathered * dense_weight.view(-1, 1, 1)
        elif self.adaptive_hybrid and self.adaptive_route == "interaction":
            dense_weight = self._interaction_dense_weight(interaction_density)
            gathered = sparse_gathered * (1.0 - dense_weight.view(-1, 1, 1)) + dense_gathered * dense_weight.view(-1, 1, 1)
        elif self.adaptive_hybrid and self.adaptive_route == "geometry":
            dense_weight = torch.zeros_like(selector_confidence)
            geometry_context = self.geometry_context_encoder(interaction_density.unsqueeze(-1))
            gathered = sparse_gathered + geometry_context.unsqueeze(1)
        elif self.adaptive_hybrid:
            dense_mask = self._adaptive_dense_mask(selected_counts, selector_confidence)
            dense_weight = dense_mask.to(sparse_gathered.dtype)
            gathered = torch.where(dense_mask.view(-1, 1, 1), dense_gathered, sparse_gathered)
        elif self.local_dense:
            dense_mask = torch.ones_like(selected_counts, dtype=torch.bool)
            dense_weight = dense_mask.to(sparse_gathered.dtype)
            gathered = dense_gathered
        else:
            dense_mask = torch.zeros_like(selected_counts, dtype=torch.bool)
            dense_weight = dense_mask.to(sparse_gathered.dtype)
            gathered = sparse_gathered
        dense_mask = dense_weight >= 0.5
        gathered = gathered.masked_fill(~selected_mask.unsqueeze(-1), 0.0)

        object_delta = torch.zeros_like(batch.object_features)
        selected_delta = self.object_delta_head(gathered).masked_fill(~selected_mask.unsqueeze(-1), 0.0)
        object_delta.scatter_add_(1, selected_indices.unsqueeze(-1).expand(-1, -1, selected_delta.size(-1)), selected_delta)

        uncertainty = torch.zeros((*batch.object_features.shape[:2], 1), device=batch.object_features.device)
        selected_uncertainty = torch.sigmoid(self.uncertainty_head(gathered)).masked_fill(~selected_mask.unsqueeze(-1), 0.0)
        uncertainty.scatter_add_(1, selected_indices.unsqueeze(-1), selected_uncertainty)

        pooled = self._pool_working_set(gathered, selected_mask, relevance_logits, selected_indices)
        delta_summary = _masked_mean(selected_delta, selected_mask)
        delta_branch_hidden = self.delta_branch_encoder(delta_summary)
        branch_logits = self.branch_head(torch.cat([pooled, delta_branch_hidden], dim=-1))[:, :num_branches]
        self.last_working_set_stats = WorkingSetStats(
            mean_selected=float(selected_counts.float().mean().detach().cpu().item()),
            max_selected=int(selected_counts.max().detach().cpu().item()),
            mean_causal_recall=self._causal_recall(batch, selected_indices, selected_mask),
            sparse_ratio=float((1.0 - dense_weight).mean().detach().cpu().item()),
            local_dense_ratio=float(dense_weight.mean().detach().cpu().item()),
            dense_compute_ratio=float(dense_compute_weight.mean().detach().cpu().item()),
            mean_selector_confidence=float(selector_confidence.mean().detach().cpu().item()),
        )
        return StatePrediction(
            object_delta=object_delta,
            relation_logits=self._relation_logits(hidden, gathered, selected_indices, batch),
            uncertainty=uncertainty,
            branch_logits=branch_logits,
            branch_probabilities=F.softmax(branch_logits, dim=-1),
            selected_paths=[
                ExecutionPath.HYBRID if use_dense else ExecutionPath.SPARSE
                for use_dense in dense_mask.detach().cpu().tolist()
            ],
        )

    def selector_loss(self) -> torch.Tensor:
        """Binary relevance supervision for causal object selection.

        The loss is optional and only available when `batch.object_ids` exists.
        It is designed for synthetic CWS experiments where the causal core is
        known, letting us distinguish selector failure from WPU-core failure.
        """

        if self._last_relevance_logits is None or self._last_batch is None:
            raise RuntimeError("selector_loss() requires a forward pass first")
        batch = self._last_batch
        logits = self._last_relevance_logits
        if batch.object_ids is None:
            return logits.sum() * 0.0
        targets = torch.zeros_like(logits)
        for batch_index, object_ids in enumerate(batch.object_ids):
            for object_index, object_id in enumerate(object_ids):
                if _is_causal_object_id(object_id):
                    targets[batch_index, object_index] = 1.0
        valid = batch.object_mask
        positives = targets[valid].sum().clamp_min(1.0)
        negatives = (1.0 - targets[valid]).sum().clamp_min(1.0)
        pos_weight = negatives / positives
        per_object = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        return per_object.masked_select(valid).mean()

    def _relevance_logits(
        self,
        hidden: torch.Tensor,
        event_hidden: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        event_expanded = event_hidden.unsqueeze(1).expand_as(hidden)
        logits = self.relevance_scorer(torch.cat([hidden, event_expanded], dim=-1)).squeeze(-1)
        return logits.masked_fill(~object_mask, float("-inf"))

    def _select_indices(
        self,
        batch: StateGraphBatch,
        relevance_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selector == "learned":
            indices = torch.topk(relevance_logits, k=min(self.working_set_size, relevance_logits.size(1)), dim=1).indices
            mask = torch.isfinite(torch.gather(relevance_logits, 1, indices))
            return _pad_indices(indices, mask, self.working_set_size)

        rows: list[list[int]] = []
        for batch_index in range(batch.object_features.size(0)):
            target = int(batch.target_indices[batch_index].item())
            if self.selector == "target":
                chosen = [target]
            elif self.selector == "frontier":
                chosen = self._frontier_indices(batch, batch_index, target)
            elif self.selector == "indexed":
                chosen = self._indexed_indices(batch, batch_index, target)
            else:
                chosen = self._oracle_indices(batch, batch_index, target)
            rows.append(chosen[: self.working_set_size])
        return _rows_to_index_tensor(rows, self.working_set_size, batch.object_features.device)

    def _indexed_indices(self, batch: StateGraphBatch, batch_index: int, target: int) -> list[int]:
        index = CausalIndex(batch.relation_indices, batch.relation_mask, batch.object_mask)
        return index.query(
            batch_index,
            CausalIndexQuery(target_index=target, max_nodes=self.working_set_size, max_depth=1),
        )

    def _frontier_indices(self, batch: StateGraphBatch, batch_index: int, target: int) -> list[int]:
        chosen = {target}
        valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
        for edge_index in valid_edges.tolist():
            src = int(batch.relation_indices[batch_index, edge_index, 0].item())
            dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
            if src in chosen or dst in chosen:
                chosen.add(src)
                chosen.add(dst)
        return sorted(chosen)

    def _oracle_indices(self, batch: StateGraphBatch, batch_index: int, target: int) -> list[int]:
        if batch.object_ids is None:
            return self._frontier_indices(batch, batch_index, target)
        chosen = [
            index
            for index, object_id in enumerate(batch.object_ids[batch_index])
            if _is_causal_object_id(object_id)
        ]
        if target not in chosen:
            chosen.insert(0, target)
        return chosen

    def _causal_recall(
        self,
        batch: StateGraphBatch,
        selected_indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> float:
        if batch.object_ids is None:
            return 0.0
        recalls: list[float] = []
        selected_cpu = selected_indices.detach().cpu()
        mask_cpu = selected_mask.detach().cpu()
        for batch_index, object_ids in enumerate(batch.object_ids):
            causal_indices = {index for index, object_id in enumerate(object_ids) if _is_causal_object_id(object_id)}
            if not causal_indices:
                recalls.append(0.0)
                continue
            selected = {
                int(index)
                for index, valid in zip(selected_cpu[batch_index].tolist(), mask_cpu[batch_index].tolist(), strict=True)
                if valid
            }
            recalls.append(len(causal_indices & selected) / len(causal_indices))
        return float(sum(recalls) / max(len(recalls), 1))

    def _selection_confidence(
        self,
        relevance_logits: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        gathered_logits = torch.gather(relevance_logits, 1, selected_indices)
        gathered_probs = torch.sigmoid(gathered_logits).masked_fill(~selected_mask, 0.0)
        counts = selected_mask.sum(dim=1).clamp_min(1).to(gathered_probs.dtype)
        return gathered_probs.sum(dim=1) / counts

    def _adaptive_dense_mask(
        self,
        selected_counts: torch.Tensor,
        selector_confidence: torch.Tensor,
    ) -> torch.Tensor:
        large_working_set = selected_counts >= self.adaptive_k_threshold
        uncertain_selection = selector_confidence < self.adaptive_confidence_threshold
        return large_working_set | uncertain_selection

    def _learned_dense_weight(
        self,
        sparse_gathered: torch.Tensor,
        selected_mask: torch.Tensor,
        selected_counts: torch.Tensor,
        selector_confidence: torch.Tensor,
    ) -> torch.Tensor:
        sparse_summary = _masked_mean(sparse_gathered, selected_mask)
        k_pressure = selected_counts.to(sparse_summary.dtype).unsqueeze(-1) / max(float(self.working_set_size), 1.0)
        confidence = selector_confidence.unsqueeze(-1).to(sparse_summary.dtype)
        gate_input = torch.cat([sparse_summary, k_pressure, confidence], dim=-1)
        return torch.sigmoid(self.route_gate(gate_input)).squeeze(-1)

    def _interaction_dense_weight(
        self,
        interaction_density: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid((interaction_density - 0.35) * 10.0)

    def _pool_working_set(
        self,
        gathered: torch.Tensor,
        selected_mask: torch.Tensor,
        relevance_logits: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        gathered_scores = torch.gather(relevance_logits, 1, selected_indices).masked_fill(~selected_mask, float("-inf"))
        weights = torch.softmax(gathered_scores, dim=1).unsqueeze(-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (gathered * weights).sum(dim=1)

    def _relation_logits(
        self,
        hidden: torch.Tensor,
        gathered: torch.Tensor,
        selected_indices: torch.Tensor,
        batch: StateGraphBatch,
    ) -> torch.Tensor:
        del gathered
        relation_logits = torch.zeros((hidden.size(0), batch.relation_indices.size(1), 1), device=hidden.device)
        selected_sets = [set(row.tolist()) for row in selected_indices.detach().cpu()]
        for batch_index in range(hidden.size(0)):
            valid_edges = batch.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
            for edge_index in valid_edges.tolist():
                src = int(batch.relation_indices[batch_index, edge_index, 0].item())
                dst = int(batch.relation_indices[batch_index, edge_index, 1].item())
                if src not in selected_sets[batch_index] and dst not in selected_sets[batch_index]:
                    continue
                relation_logits[batch_index, edge_index] = self.relation_head(
                    torch.cat([hidden[batch_index, src], hidden[batch_index, dst], batch.relation_features[batch_index, edge_index]])
                )
        return relation_logits


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return torch.gather(values, 1, expanded)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    total = (values * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return total / count


def _interaction_density(selected_object_features: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    positions = selected_object_features[..., 1:3]
    pair_delta = positions.unsqueeze(2) - positions.unsqueeze(1)
    pair_distance = pair_delta.square().sum(dim=-1).sqrt()
    pair_mask = selected_mask.unsqueeze(2) & selected_mask.unsqueeze(1)
    diagonal = torch.eye(selected_mask.size(1), dtype=torch.bool, device=selected_mask.device).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal
    close_affinity = torch.exp(-pair_distance / 0.08).masked_fill(~pair_mask, 0.0)
    pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1).to(close_affinity.dtype)
    return close_affinity.sum(dim=(1, 2)) / pair_count


def _pad_indices(
    indices: torch.Tensor,
    mask: torch.Tensor,
    target_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.size(1) == target_size:
        return indices, mask
    pad_width = target_size - indices.size(1)
    pad_values = indices[:, :1].expand(-1, pad_width)
    pad_mask = torch.zeros((indices.size(0), pad_width), dtype=torch.bool, device=indices.device)
    return torch.cat([indices, pad_values], dim=1), torch.cat([mask, pad_mask], dim=1)


def _rows_to_index_tensor(
    rows: list[list[int]],
    target_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.zeros((len(rows), target_size), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), target_size), dtype=torch.bool, device=device)
    for row_index, row in enumerate(rows):
        if not row:
            row = [0]
        width = min(len(row), target_size)
        indices[row_index, :width] = torch.tensor(row[:width], dtype=torch.long, device=device)
        if width < target_size:
            indices[row_index, width:] = row[0]
        mask[row_index, :width] = True
    return indices, mask


def _is_causal_object_id(object_id: str) -> bool:
    causal_prefixes = ("cup_", "table_", "hand_", "edge_", "catcher_", "obstacle_")
    return object_id.startswith(causal_prefixes)
