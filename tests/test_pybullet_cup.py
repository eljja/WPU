from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from wpu.data.pybullet_cup import PyBulletCupDataset, collate_indexed_pybullet_cup_samples
from wpu.models.factory import create_model


pytest.importorskip("pybullet")

ROOT = Path(__file__).resolve().parents[1]


def test_pybullet_cup_dataset_produces_state_batch() -> None:
    dataset = PyBulletCupDataset(size=2, seed=5, background_objects=2, steps=24)
    batch, target_delta, labels, causal_k = collate_indexed_pybullet_cup_samples([dataset[0], dataset[1]], max_nodes=8)

    assert batch.object_features.shape[0] == 2
    assert target_delta.shape[:2] == batch.object_features.shape[:2]
    assert labels.shape == (2,)
    assert causal_k.min().item() >= 4
    assert batch.object_ids is not None
    assert "cup_001" in batch.object_ids[0]


def test_pybullet_cup_batch_backward_smoke() -> None:
    dataset = PyBulletCupDataset(size=4, seed=7, background_objects=2, steps=24)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda samples: collate_indexed_pybullet_cup_samples(samples, max_nodes=8),
    )
    batch, target_delta, labels, _ = next(iter(loader))
    model = create_model("wpu-cws-indexed-sparse", hidden_dim=16, num_heads=2, working_set_size=8)
    prediction = model(batch, num_branches=3)
    loss = torch.nn.functional.cross_entropy(prediction.branch_logits, labels)
    loss = loss + 0.1 * torch.nn.functional.mse_loss(prediction.object_delta, target_delta)
    loss.backward()

    assert torch.isfinite(loss)


def test_pybullet_cup_benchmark_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_cup_benchmark.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
