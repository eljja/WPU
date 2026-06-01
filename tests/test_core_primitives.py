import pytest
import torch

from wpu.core.causal_index import CausalIndex, CausalIndexQuery
from wpu.core.state import Relation
from wpu.core.uncertainty import multiply_confidence


def test_causal_index_returns_bounded_relation_frontier() -> None:
    relation_indices = torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 4]]])
    relation_mask = torch.tensor([[True, True, True, True]])
    object_mask = torch.tensor([[True, True, True, True, True]])
    index = CausalIndex(relation_indices, relation_mask, object_mask)

    result = index.query(0, CausalIndexQuery(target_index=0, max_nodes=4, max_depth=2))

    assert result == [0, 1, 4, 2]


def test_causal_index_clamps_invalid_target_and_handles_empty_state() -> None:
    relation_indices = torch.zeros((2, 1, 2), dtype=torch.long)
    relation_mask = torch.zeros((2, 1), dtype=torch.bool)
    object_mask = torch.tensor([[True, True], [False, False]])
    index = CausalIndex(relation_indices, relation_mask, object_mask)

    assert index.query(0, CausalIndexQuery(target_index=99, max_nodes=2)) == [1]
    assert index.query(1, CausalIndexQuery(target_index=99, max_nodes=2)) == [0]


def test_relation_helpers_and_confidence_clamp() -> None:
    relation = Relation("cup_001", "table_001", "on_top_of")

    assert relation.touches("cup_001")
    assert relation.other("cup_001") == "table_001"
    assert relation.other("hand_001") is None
    assert multiply_confidence(0.8, 0.5) == pytest.approx(0.4)
    assert multiply_confidence(2.0, 0.8) == 1.0
    assert multiply_confidence(-1.0, 0.8) == 0.0
