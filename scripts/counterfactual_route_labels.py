from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.working_set_physics import (
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
    collate_working_set_samples,
)
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure counterfactual dense-needed route labels.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/counterfactual_route_labels.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"counterfactual seed={seed} N={n_value} K={k_value}", flush=True)
                rows.append(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    sparse_model = _train_model("wpu-cws-indexed-sparse", background_objects, causal_obstacles, seed, args, device)
    dense_model = _train_model("wpu-cws-indexed-local-dense", background_objects, causal_obstacles, seed, args, device)
    return _evaluate_pair(sparse_model, dense_model, background_objects, causal_obstacles, seed, args, device)


def _train_model(
    model_name: str,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    dataset = WorkingSetPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    class_weights = _class_weights(dataset).to(device) if args.class_weights else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step, (batch, target_delta, labels, _) in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        prediction = model(batch, num_branches=3, route_branches=3)
        loss = F.mse_loss(prediction.object_delta, target_delta)
        loss = loss + F.cross_entropy(prediction.branch_logits, labels, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= args.steps:
            break
    return model


def _evaluate_pair(
    sparse_model: torch.nn.Module,
    dense_model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset = WorkingSetPhysicsDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    sparse_model.eval()
    dense_model.eval()
    total = 0
    sparse_correct = 0
    dense_correct = 0
    dense_fixes_sparse = 0
    dense_breaks_sparse = 0
    both_correct = 0
    both_wrong = 0
    dense_lower_loss = 0
    dense_needed_labels = 0
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = sparse_model(batch, num_branches=3, route_branches=3)
            dense_prediction = dense_model(batch, num_branches=3, route_branches=3)
            sparse_pred = sparse_prediction.branch_probabilities.argmax(dim=-1)
            dense_pred = dense_prediction.branch_probabilities.argmax(dim=-1)
            sparse_is_correct = sparse_pred == labels
            dense_is_correct = dense_pred == labels
            sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
            dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
            dense_is_better = dense_loss + 1e-6 < sparse_loss
            dense_needed = (~sparse_is_correct & dense_is_correct) | (dense_is_better & ~sparse_is_correct)

            total += int(labels.numel())
            sparse_correct += int(sparse_is_correct.sum().item())
            dense_correct += int(dense_is_correct.sum().item())
            dense_fixes_sparse += int((~sparse_is_correct & dense_is_correct).sum().item())
            dense_breaks_sparse += int((sparse_is_correct & ~dense_is_correct).sum().item())
            both_correct += int((sparse_is_correct & dense_is_correct).sum().item())
            both_wrong += int((~sparse_is_correct & ~dense_is_correct).sum().item())
            dense_lower_loss += int(dense_is_better.sum().item())
            dense_needed_labels += int(dense_needed.sum().item())

    return {
        "status": "ok",
        "seed": seed,
        "total_objects_n": background_objects + 4 + causal_obstacles,
        "causal_k": 4 + causal_obstacles,
        "samples": total,
        "interaction_mode": args.interaction_mode,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "working_set_size": args.working_set_size,
        "sparse_accuracy": round(sparse_correct / max(total, 1), 6),
        "dense_accuracy": round(dense_correct / max(total, 1), 6),
        "dense_needed_rate": round(dense_needed_labels / max(total, 1), 6),
        "dense_fix_rate": round(dense_fixes_sparse / max(total, 1), 6),
        "dense_break_rate": round(dense_breaks_sparse / max(total, 1), 6),
        "both_correct_rate": round(both_correct / max(total, 1), 6),
        "both_wrong_rate": round(both_wrong / max(total, 1), 6),
        "dense_lower_loss_rate": round(dense_lower_loss / max(total, 1), 6),
    }


def _collate_fn(args: argparse.Namespace):
    if not args.pre_tensor_indexed:
        return collate_working_set_samples

    def collate(samples):
        return collate_indexed_working_set_samples(
            samples,
            max_nodes=args.working_set_size,
            max_depth=args.index_depth,
        )

    return collate


def _class_weights(dataset: WorkingSetPhysicsDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _move_batch(batch, device: torch.device):
    batch.object_features = batch.object_features.to(device)
    batch.relation_indices = batch.relation_indices.to(device)
    batch.relation_features = batch.relation_features.to(device)
    batch.event_features = batch.event_features.to(device)
    batch.object_mask = batch.object_mask.to(device)
    batch.relation_mask = batch.relation_mask.to(device)
    batch.target_indices = batch.target_indices.to(device)
    batch.time_features = batch.time_features.to(device)
    return batch


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
