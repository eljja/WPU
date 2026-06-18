from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from wpu.core.objectification import evaluate_objectification
from wpu.data.pybullet_cup import (
    ObjectificationCorruptionConfig,
    PyBulletCupDataset,
    collate_indexed_pybullet_cup_samples,
    corrupt_pybullet_cup_sample,
)
from wpu.models.factory import create_model
from scripts.pybullet_shift_generalization import _target_object_delta_loss


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
    assert dataset[0].source_index == 0


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


def test_target_object_delta_loss_ignores_background_dilution() -> None:
    dataset = PyBulletCupDataset(size=2, seed=8, background_objects=64, steps=24)
    batch, target_delta, _, _ = collate_indexed_pybullet_cup_samples([dataset[0], dataset[1]], max_nodes=12)
    prediction = torch.zeros_like(target_delta)
    baseline_loss = _target_object_delta_loss(prediction, target_delta, batch)

    batch_indices = torch.arange(target_delta.size(0))
    prediction[batch_indices, batch.target_indices] = target_delta[batch_indices, batch.target_indices]
    matched_loss = _target_object_delta_loss(prediction, target_delta, batch)

    assert baseline_loss > 0
    assert matched_loss == 0


def test_pybullet_cup_objectification_corruption_preserves_batch_alignment() -> None:
    dataset = PyBulletCupDataset(size=2, seed=9, background_objects=4, steps=24, balanced_labels=True)
    clean = dataset[0]
    corrupted = corrupt_pybullet_cup_sample(
        clean,
        config=ObjectificationCorruptionConfig(
            relation_drop_rate=1.0,
            non_target_object_drop_rate=0.5,
            position_noise_std=0.01,
            confidence_scale=0.5,
            identity_swap_rate=1.0,
        ),
        seed=123,
    )
    batch, target_delta, labels, causal_k = collate_indexed_pybullet_cup_samples([corrupted], max_nodes=8)
    report = evaluate_objectification(corrupted.state)

    assert batch.object_features.shape[:2] == target_delta.shape[:2]
    assert labels.item() in {0, 1, 2}
    assert causal_k.item() >= 1
    assert report.contract_score < evaluate_objectification(clean.state).contract_score


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


def test_pybullet_objectification_stress_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_objectification_stress.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_closed_loop_rollout_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_closed_loop_rollout.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_closed_loop_rollout_guarded_projection_runs(tmp_path: Path) -> None:
    output = tmp_path / "guarded_rollout.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pybullet_closed_loop_rollout.py",
            "--models",
            "wpu-cws-indexed-sparse",
            "--horizons",
            "2",
            "--background-objects",
            "2",
            "--seeds",
            "3",
            "--steps",
            "1",
            "--sim-steps",
            "24",
            "--train-sim-steps",
            "4",
            "--samples",
            "4",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--num-heads",
            "2",
            "--working-set-size",
            "8",
            "--delta-clip",
            "0.25",
            "--branch-loss-weight",
            "0.5",
            "--delta-loss-weight",
            "0.2",
            "--multihorizon-train-steps",
            "4",
            "8",
            "--multihorizon-loss-weight",
            "0.1",
            "--grad-clip-norm",
            "1.0",
            "--integrity-projection",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "integrity_projection" in text
    assert "raw_delta_norm_mean" in text
    assert "train_sim_steps" in text
    assert "branch_loss_weight" in text
    assert "delta_loss_weight" in text
    assert "multihorizon_train_steps" in text
    assert "multihorizon_loss_weight" in text
    assert "grad_clip_norm" in text


def test_pybullet_local_law_revision_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_local_law_revision.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_system_profile_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_system_profile.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_objectification_quality_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_objectification_quality.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_shift_generalization_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pybullet_shift_generalization.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_pybullet_shift_route_regret_options_skip_sparse_baseline(tmp_path: Path) -> None:
    output = tmp_path / "route_regret_sparse_skip.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pybullet_shift_generalization.py",
            "--models",
            "wpu-cws-indexed-sparse",
            "wpu-cws-indexed-physics-regret-hybrid",
            "--train-mechanisms",
            "nominal",
            "--eval-mechanisms",
            "nominal",
            "--background-objects",
            "4",
            "--seeds",
            "11",
            "--steps",
            "1",
            "--sim-steps",
            "20",
            "--samples",
            "2",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--layers",
            "1",
            "--num-heads",
            "4",
            "--working-set-size",
            "8",
            "--route-regret-loss-weight",
            "1.0",
            "--select-route-regret-threshold",
            "--route-regret-selection-samples",
            "2",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
