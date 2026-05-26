from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM,
    _candidate_features,
    _candidate_ids,
    _evaluate_modes,
    _make_samples,
    _selected_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-K generalization of learned state retrievers.")
    parser.add_argument("--n-value", type=int, default=2048)
    parser.add_argument("--train-k-groups", nargs="+", default=["16", "8,16,32"])
    parser.add_argument("--test-k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[19, 23])
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--test-samples", type=int, default=160)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--balance-train-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_learned_retriever_cross_k_probe.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for group in args.train_k_groups:
        train_k_values = _parse_k_group(group)
        train_label = ",".join(str(value) for value in train_k_values)
        print(f"cross-k retriever train_K={train_label}", flush=True)
        train_sample_groups = []
        for k_value in train_k_values:
            train_sample_groups.append(_make_samples(args.n_value, k_value, args.train_seeds, args.train_samples))
        if args.balance_train_k:
            model = _train_group_balanced_model(train_sample_groups, args.budget, args.steps, args.hidden_dim, args.lr)
        else:
            train_samples = [sample for group in train_sample_groups for sample in group]
            model = _train_group_balanced_model([train_samples], args.budget, args.steps, args.hidden_dim, args.lr)
        for test_k in args.test_k_values:
            test_samples = _make_samples(args.n_value, test_k, args.test_seeds, args.test_samples)
            for row in _evaluate_modes(test_samples, args.n_value, test_k, args.budget, model):
                row["train_causal_k"] = train_label
                row["balance_train_k"] = int(args.balance_train_k)
                rows.append(row)
        _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _parse_k_group(group: str) -> list[int]:
    values = [int(part.strip()) for part in group.split(",") if part.strip()]
    if not values:
        raise ValueError("train K group cannot be empty")
    return values


def _train_group_balanced_model(
    sample_groups,
    budget: int,
    steps: int,
    hidden_dim: int,
    lr: float,
) -> nn.Module:
    tensors = [_group_tensors(samples, budget) for samples in sample_groups]
    model = nn.Sequential(
        nn.LayerNorm(FEATURE_DIM),
        nn.Linear(FEATURE_DIM, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weights = []
    for _, label_tensor in tensors:
        positive = label_tensor.sum().clamp_min(1.0)
        negative = (1.0 - label_tensor).sum().clamp_min(1.0)
        pos_weights.append(negative / positive)
    for _ in range(steps):
        losses = []
        for (feature_tensor, label_tensor), pos_weight in zip(tensors, pos_weights, strict=True):
            logits = model(feature_tensor).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(logits, label_tensor, pos_weight=pos_weight))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def _group_tensors(samples, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for sample in samples:
        teacher_ids = _selected_ids(sample, "interaction", budget)
        target = sample.event.target
        for object_id in _candidate_ids(sample.state, sample.event):
            if object_id == target:
                continue
            features.append(_candidate_features(sample.state, sample.event, object_id))
            labels.append(float(object_id in teacher_ids))
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
