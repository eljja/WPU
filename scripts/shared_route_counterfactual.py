from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_indexed_working_set_samples
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure same-model sparse-vs-local-dense counterfactual route labels.")
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
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/shared_route_counterfactual.csv"))
    parser.add_argument("--examples-out", type=Path, default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.examples_out is not None:
        args.examples_out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"shared-route seed={seed} N={n_value} K={k_value}", flush=True)
                rows.append(_run_condition(background_objects, causal_obstacles, seed, args, example_rows))
                _write_csv(args.out, rows)
                if args.examples_out is not None:
                    _write_csv(args.examples_out, example_rows)
    _write_csv(args.out, rows)
    if args.examples_out is not None:
        _write_csv(args.examples_out, example_rows)
    print(f"wrote={args.out}", flush=True)
    if args.examples_out is not None:
        print(f"examples={args.examples_out}", flush=True)


def _run_condition(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    example_rows: list[dict[str, object]],
) -> dict[str, object]:
    device = torch.device(args.device)
    model = _train_dual_path_model(background_objects, causal_obstacles, seed, args, device)
    return _evaluate_dual_paths(model, background_objects, causal_obstacles, seed, args, device, example_rows)


def _train_dual_path_model(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = create_model(
        "wpu-cws-indexed-local-dense",
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
        sparse_prediction = model(batch, num_branches=3, force_route="sparse")
        dense_prediction = model(batch, num_branches=3, force_route="local_dense")
        loss = _prediction_loss(sparse_prediction, target_delta, labels, class_weights)
        loss = loss + _prediction_loss(dense_prediction, target_delta, labels, class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= args.steps:
            break
    return model


def _evaluate_dual_paths(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    example_rows: list[dict[str, object]],
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
    model.eval()
    total = 0
    sparse_correct = 0
    dense_correct = 0
    dense_fixes_sparse = 0
    dense_breaks_sparse = 0
    dense_lower_loss = 0
    dense_needed_labels = 0
    branch_disagreement = 0
    with torch.no_grad():
        sample_offset = 0
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            sparse_pred = sparse_prediction.branch_probabilities.argmax(dim=-1)
            dense_pred = dense_prediction.branch_probabilities.argmax(dim=-1)
            sparse_is_correct = sparse_pred == labels
            dense_is_correct = dense_pred == labels
            sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
            dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
            dense_regret = dense_loss - sparse_loss
            dense_is_better = dense_loss + 1e-6 < sparse_loss
            dense_needed = (~sparse_is_correct & dense_is_correct) | (dense_is_better & ~sparse_is_correct)
            route_features = _route_features(batch)
            sparse_features = _prediction_features(sparse_prediction)

            total += int(labels.numel())
            sparse_correct += int(sparse_is_correct.sum().item())
            dense_correct += int(dense_is_correct.sum().item())
            dense_fixes_sparse += int((~sparse_is_correct & dense_is_correct).sum().item())
            dense_breaks_sparse += int((sparse_is_correct & ~dense_is_correct).sum().item())
            dense_lower_loss += int(dense_is_better.sum().item())
            dense_needed_labels += int(dense_needed.sum().item())
            branch_disagreement += int((sparse_pred != dense_pred).sum().item())
            for row_index in range(int(labels.numel())):
                example_rows.append(
                    {
                        "seed": seed,
                        "sample_index": sample_offset + row_index,
                        "total_objects_n": background_objects + 4 + causal_obstacles,
                        "causal_k": 4 + causal_obstacles,
                        "interaction_mode": args.interaction_mode,
                        "interaction_density": round(float(route_features["interaction_density"][row_index].detach().cpu().item()), 8),
                        "min_pair_distance": round(float(route_features["min_pair_distance"][row_index].detach().cpu().item()), 8),
                        "mean_pair_distance": round(float(route_features["mean_pair_distance"][row_index].detach().cpu().item()), 8),
                        "target_x": round(float(route_features["target_x"][row_index].detach().cpu().item()), 8),
                        "target_y": round(float(route_features["target_y"][row_index].detach().cpu().item()), 8),
                        "event_norm": round(float(route_features["event_norm"][row_index].detach().cpu().item()), 8),
                        "sparse_entropy": round(float(sparse_features["entropy"][row_index].detach().cpu().item()), 8),
                        "sparse_margin": round(float(sparse_features["margin"][row_index].detach().cpu().item()), 8),
                        "sparse_confidence": round(float(sparse_features["confidence"][row_index].detach().cpu().item()), 8),
                        "sparse_delta_norm": round(float(sparse_features["delta_norm"][row_index].detach().cpu().item()), 8),
                        "sparse_uncertainty_mean": round(float(sparse_features["uncertainty_mean"][row_index].detach().cpu().item()), 8),
                        "sparse_correct": int(sparse_is_correct[row_index].detach().cpu().item()),
                        "dense_correct": int(dense_is_correct[row_index].detach().cpu().item()),
                        "dense_needed": int(dense_needed[row_index].detach().cpu().item()),
                        "dense_fixes_sparse": int((~sparse_is_correct[row_index] & dense_is_correct[row_index]).detach().cpu().item()),
                        "dense_breaks_sparse": int((sparse_is_correct[row_index] & ~dense_is_correct[row_index]).detach().cpu().item()),
                        "dense_beneficial": int(dense_is_better[row_index].detach().cpu().item()),
                        "sparse_loss": round(float(sparse_loss[row_index].detach().cpu().item()), 8),
                        "dense_loss": round(float(dense_loss[row_index].detach().cpu().item()), 8),
                        "dense_regret": round(float(dense_regret[row_index].detach().cpu().item()), 8),
                    }
                )
            sample_offset += int(labels.numel())
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
        "dense_lower_loss_rate": round(dense_lower_loss / max(total, 1), 6),
        "branch_disagreement_rate": round(branch_disagreement / max(total, 1), 6),
    }


def _prediction_loss(prediction, target_delta: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None) -> torch.Tensor:
    return F.mse_loss(prediction.object_delta, target_delta) + F.cross_entropy(
        prediction.branch_logits,
        labels,
        weight=class_weights,
    )


def _prediction_features(prediction) -> dict[str, torch.Tensor]:
    probabilities = prediction.branch_probabilities
    sorted_probabilities = torch.sort(probabilities, dim=-1, descending=True).values
    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    margin = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
    delta_norm = prediction.object_delta.square().sum(dim=-1).sqrt().mean(dim=-1)
    uncertainty_mean = prediction.uncertainty.mean(dim=(1, 2))
    return {
        "entropy": entropy,
        "margin": margin,
        "confidence": sorted_probabilities[:, 0],
        "delta_norm": delta_norm,
        "uncertainty_mean": uncertainty_mean,
    }


def _route_features(batch) -> dict[str, torch.Tensor]:
    positions = batch.object_features[..., 1:3]
    pair_delta = positions.unsqueeze(2) - positions.unsqueeze(1)
    pair_distance = pair_delta.square().sum(dim=-1).sqrt()
    pair_mask = batch.object_mask.unsqueeze(2) & batch.object_mask.unsqueeze(1)
    diagonal = torch.eye(batch.object_mask.size(1), dtype=torch.bool, device=batch.object_mask.device).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal
    close_affinity = torch.exp(-pair_distance / 0.08).masked_fill(~pair_mask, 0.0)
    pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1).to(close_affinity.dtype)
    masked_distance = pair_distance.masked_fill(~pair_mask, 0.0)
    min_distance = pair_distance.masked_fill(~pair_mask, float("inf")).amin(dim=(1, 2))
    min_distance = torch.where(torch.isfinite(min_distance), min_distance, torch.zeros_like(min_distance))
    mean_distance = masked_distance.sum(dim=(1, 2)) / pair_count
    target_indices = batch.target_indices.clamp_min(0)
    target_positions = torch.gather(positions, 1, target_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    return {
        "interaction_density": close_affinity.sum(dim=(1, 2)) / pair_count,
        "min_pair_distance": min_distance,
        "mean_pair_distance": mean_distance,
        "target_x": target_positions[:, 0],
        "target_y": target_positions[:, 1],
        "event_norm": batch.event_features.norm(dim=-1),
    }


def _collate_fn(args: argparse.Namespace):
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
