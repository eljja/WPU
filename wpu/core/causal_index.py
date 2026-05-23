from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch


@dataclass(frozen=True, slots=True)
class CausalIndexQuery:
    target_index: int
    max_nodes: int
    max_depth: int = 1


class CausalIndex:
    """Lightweight state index for event-local object retrieval.

    This is the first v2 retrieval primitive. It uses identity and relation
    adjacency already present in `StateGraphBatch` instead of learned global
    relevance scores. The implementation is intentionally simple: target object
    plus bounded relation-frontier expansion.
    """

    def __init__(self, relation_indices: torch.Tensor, relation_mask: torch.Tensor, object_mask: torch.Tensor) -> None:
        self.relation_indices = relation_indices.detach().cpu()
        self.relation_mask = relation_mask.detach().cpu()
        self.object_mask = object_mask.detach().cpu()

    def query(self, batch_index: int, query: CausalIndexQuery) -> list[int]:
        valid_objects = int(self.object_mask[batch_index].sum().item())
        if valid_objects <= 0:
            return [0]
        target = min(max(int(query.target_index), 0), valid_objects - 1)
        adjacency = self._adjacency(batch_index, valid_objects)
        visited = {target}
        ordered = [target]
        frontier: deque[tuple[int, int]] = deque([(target, 0)])
        while frontier and len(ordered) < query.max_nodes:
            node, depth = frontier.popleft()
            if depth >= query.max_depth:
                continue
            for neighbor in sorted(adjacency[node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                ordered.append(neighbor)
                if len(ordered) >= query.max_nodes:
                    break
                frontier.append((neighbor, depth + 1))
        return ordered

    def _adjacency(self, batch_index: int, valid_objects: int) -> list[set[int]]:
        adjacency = [set() for _ in range(valid_objects)]
        valid_edges = self.relation_mask[batch_index].nonzero(as_tuple=False).flatten()
        for edge_index in valid_edges.tolist():
            src = int(self.relation_indices[batch_index, edge_index, 0].item())
            dst = int(self.relation_indices[batch_index, edge_index, 1].item())
            if src >= valid_objects or dst >= valid_objects:
                continue
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        return adjacency


__all__ = ["CausalIndex", "CausalIndexQuery"]
