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
    parser = argparse.ArgumentParser(description="Train WPU regret routing in separated propagation and route-head stages.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=32)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--regret-steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=3e-3)
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--calibrate-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=[-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5],
    )
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_regret_hybrid.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"staged-regret seed={seed} N={n_value} K={k_value}", flush=True)
                rows.append(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(background_objects: int, causal_obstacles: int, seed: int, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = create_model(
        "wpu-cws-indexed-regret-hybrid",
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    train_dataset = WorkingSetPhysicsDataset(
        size=max((args.propagation_steps + args.regret_steps) * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    _train_propagation(model, train_dataset, class_weights, args, device)
    _train_regret_head(model, train_dataset, args, device)
    route_threshold = (
        _choose_regret_threshold(model, background_objects, causal_obstacles, seed, args, device)
        if args.calibrate_thresholds
        else 0.0
    )
    return _evaluate(model, background_objects, causal_obstacles, seed, args, device, route_threshold)


def _train_propagation(
    model: torch.nn.Module,
    dataset: WorkingSetPhysicsDataset,
    class_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
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
        if step >= args.propagation_steps:
            break


def _train_regret_head(
    model: torch.nn.Module,
    dataset: WorkingSetPhysicsDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.route_regret_head.parameters():
        parameter.requires_grad_(True)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    optimizer = torch.optim.AdamW(model.route_regret_head.parameters(), lr=args.regret_lr)
    model.train()
    for step, (batch, _, labels, _) in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        labels = labels.to(device)
        target_regret = _counterfactual_regret_target(model, batch, labels, args.compute_cost)
        model(batch, num_branches=3)
        loss = model.route_regret_loss(target_regret)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= args.regret_steps:
            break
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _evaluate(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    route_threshold: float,
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
    routed_correct = 0
    sparse_loss_total = 0.0
    dense_loss_total = 0.0
    routed_loss_total = 0.0
    oracle_loss_total = 0.0
    dense_compute_values: list[float] = []
    regret_values: list[float] = []
    negative_values: list[float] = []
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
            dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none") + args.compute_cost
            route_dense = model.route_regret_prediction() < route_threshold
            routed_loss = torch.where(route_dense, dense_loss, sparse_loss)
            oracle_loss = torch.minimum(sparse_loss, dense_loss)

            sparse_pred = sparse_prediction.branch_probabilities.argmax(dim=-1)
            dense_pred = dense_prediction.branch_probabilities.argmax(dim=-1)
            routed_logits = torch.where(route_dense.view(-1, 1), dense_prediction.branch_logits, sparse_prediction.branch_logits)
            routed_pred = routed_logits.argmax(dim=-1)
            total += int(labels.numel())
            sparse_correct += int((sparse_pred == labels).sum().item())
            dense_correct += int((dense_pred == labels).sum().item())
            routed_correct += int((routed_pred == labels).sum().item())
            sparse_loss_total += float(sparse_loss.sum().item())
            dense_loss_total += float(dense_loss.sum().item())
            routed_loss_total += float(routed_loss.sum().item())
            oracle_loss_total += float(oracle_loss.sum().item())
            dense_compute_values.append(float(route_dense.float().mean().detach().cpu().item()))
            regret_prediction = model.route_regret_prediction()
            if regret_prediction.numel() > 0:
                regret_values.append(float(regret_prediction.mean().detach().cpu().item()))
                negative_values.append(float((regret_prediction < 0.0).float().mean().detach().cpu().item()))
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
        "propagation_steps": args.propagation_steps,
        "regret_steps": args.regret_steps,
        "compute_cost": args.compute_cost,
        "calibrated_threshold": round(route_threshold, 6),
        "sparse_accuracy": round(sparse_correct / max(total, 1), 6),
        "dense_accuracy": round(dense_correct / max(total, 1), 6),
        "routed_accuracy": round(routed_correct / max(total, 1), 6),
        "sparse_loss": round(sparse_loss_total / max(total, 1), 6),
        "dense_loss": round(dense_loss_total / max(total, 1), 6),
        "routed_loss": round(routed_loss_total / max(total, 1), 6),
        "oracle_loss": round(oracle_loss_total / max(total, 1), 6),
        "routed_delta_vs_sparse": round((routed_loss_total - sparse_loss_total) / max(total, 1), 6),
        "routed_excess_over_oracle": round((routed_loss_total - oracle_loss_total) / max(total, 1), 6),
        "dense_compute_ratio": round(sum(dense_compute_values) / max(len(dense_compute_values), 1), 6),
        "route_regret_mean": round(sum(regret_values) / max(len(regret_values), 1), 6),
        "route_regret_negative_ratio": round(sum(negative_values) / max(len(negative_values), 1), 6),
    }


def _choose_regret_threshold(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    dataset = WorkingSetPhysicsDataset(
        size=args.validation_samples,
        seed=seed + 5_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    predictions: list[torch.Tensor] = []
    sparse_losses: list[torch.Tensor] = []
    dense_losses: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            predictions.append(model.route_regret_prediction().detach().cpu())
            sparse_losses.append(F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none").detach().cpu())
            dense_losses.append(
                (
                    F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none") + args.compute_cost
                ).detach().cpu()
            )
    regret_prediction = torch.cat(predictions)
    sparse_loss = torch.cat(sparse_losses)
    dense_loss = torch.cat(dense_losses)
    quantiles = torch.quantile(regret_prediction, torch.linspace(0.1, 0.9, 9)).tolist()
    candidates = sorted(set(float(value) for value in [*args.threshold_grid, *quantiles]))
    best_threshold = 0.0
    best_loss = float("inf")
    for threshold in candidates:
        route_dense = regret_prediction < threshold
        routed_loss = torch.where(route_dense, dense_loss, sparse_loss).mean().item()
        if routed_loss < best_loss:
            best_loss = routed_loss
            best_threshold = threshold
    return best_threshold


def _counterfactual_regret_target(model: torch.nn.Module, batch, labels: torch.Tensor, compute_cost: float) -> torch.Tensor:
    with torch.no_grad():
        sparse_prediction = model(batch, num_branches=3, force_route="sparse")
        dense_prediction = model(batch, num_branches=3, force_route="local_dense")
        sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
        dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
    return (dense_loss - sparse_loss + compute_cost).detach()


def _prediction_loss(prediction, target_delta: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None) -> torch.Tensor:
    return F.mse_loss(prediction.object_delta, target_delta) + F.cross_entropy(
        prediction.branch_logits,
        labels,
        weight=class_weights,
    )


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
