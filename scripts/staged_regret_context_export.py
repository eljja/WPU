from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.staged_regret_hybrid import (  # noqa: E402
    _class_weights,
    _collate_fn,
    _move_batch,
    _train_propagation,
    _train_regret_head,
)
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.causal_working_set_processor import _batched_gather, _interaction_density  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export staged WPU route-regret sample context features.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
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
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_regret_context.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"context-export seed={seed} N={n_value} K={k_value}", flush=True)
                rows.extend(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(background_objects: int, causal_obstacles: int, seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
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
    return _export_eval_rows(model, background_objects, causal_obstacles, seed, args, device)


def _export_eval_rows(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = WorkingSetPhysicsDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    rows: list[dict[str, object]] = []
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            context = _route_context(model, batch, sparse_prediction)
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
            dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
            sparse_correct = sparse_prediction.branch_logits.argmax(dim=-1) == labels
            dense_correct = dense_prediction.branch_logits.argmax(dim=-1) == labels
            predicted_regret = model.route_regret_prediction()
            for index in range(labels.numel()):
                rows.append(
                    {
                        "seed": seed,
                        "sample_index": sample_offset + index,
                        "total_objects_n": background_objects + 4 + causal_obstacles,
                        "causal_k": 4 + causal_obstacles,
                        "interaction_mode": args.interaction_mode,
                        "hidden_dim": args.hidden_dim,
                        "working_set_size": args.working_set_size,
                        "compute_cost": args.compute_cost,
                        "label": int(labels[index].detach().cpu().item()),
                        "sparse_loss": _round(sparse_loss[index]),
                        "dense_loss": _round(dense_loss[index]),
                        "dense_regret": _round(dense_loss[index] - sparse_loss[index]),
                        "sparse_correct": int(sparse_correct[index].detach().cpu().item()),
                        "dense_correct": int(dense_correct[index].detach().cpu().item()),
                        "sparse_dense_disagreement": int(
                            (
                                sparse_prediction.branch_logits[index].argmax()
                                != dense_prediction.branch_logits[index].argmax()
                            )
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "predicted_regret": _round(predicted_regret[index]),
                        "regret_abs": _round(predicted_regret[index].abs()),
                        **{key: _round(value[index]) for key, value in context.items()},
                    }
                )
            sample_offset += int(labels.numel())
    return rows


def _route_context(model: torch.nn.Module, batch, sparse_prediction) -> dict[str, torch.Tensor]:
    hidden = model.object_encoder(batch.object_features)
    event_hidden = model.event_encoder(batch.event_features)
    relevance_logits = model._relevance_logits(hidden, event_hidden, batch.object_mask)
    selected_indices, selected_mask = model._select_indices(batch, relevance_logits)
    selected_features = _batched_gather(batch.object_features, selected_indices)
    interaction_density = _interaction_density(selected_features, selected_mask)
    selector_confidence = model._selection_confidence(relevance_logits, selected_indices, selected_mask)
    pair_stats = _pair_distance_stats(selected_features, selected_mask)
    sparse_prob = F.softmax(sparse_prediction.branch_logits, dim=-1)
    sparse_entropy = -(sparse_prob.clamp_min(1e-8).log() * sparse_prob).sum(dim=-1)
    top2 = torch.topk(sparse_prob, k=min(2, sparse_prob.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        sparse_margin = top2[:, 0]
    else:
        sparse_margin = top2[:, 0] - top2[:, 1]
    sparse_confidence = top2[:, 0]
    object_mask = batch.object_mask.to(sparse_prediction.object_delta.dtype)
    object_count = object_mask.sum(dim=1).clamp_min(1.0)
    sparse_delta_norm = sparse_prediction.object_delta.norm(dim=-1).mul(object_mask).sum(dim=1) / object_count
    sparse_uncertainty_mean = sparse_prediction.uncertainty.squeeze(-1).mul(object_mask).sum(dim=1) / object_count
    target_features = batch.object_features[
        torch.arange(batch.object_features.size(0), device=batch.object_features.device),
        batch.target_indices,
    ]
    return {
        "interaction_density": interaction_density,
        "min_pair_distance": pair_stats["min_pair_distance"],
        "mean_pair_distance": pair_stats["mean_pair_distance"],
        "target_x": target_features[:, 1],
        "target_y": target_features[:, 2],
        "event_norm": batch.event_features.norm(dim=-1),
        "selector_confidence": selector_confidence,
        "selected_fraction": selected_mask.sum(dim=1).to(torch.float32) / max(float(model.working_set_size), 1.0),
        "sparse_entropy": sparse_entropy,
        "sparse_margin": sparse_margin,
        "sparse_confidence": sparse_confidence,
        "sparse_delta_norm": sparse_delta_norm,
        "sparse_uncertainty_mean": sparse_uncertainty_mean,
    }


def _pair_distance_stats(selected_features: torch.Tensor, selected_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    min_values: list[torch.Tensor] = []
    mean_values: list[torch.Tensor] = []
    positions = selected_features[..., 1:3]
    for index in range(positions.size(0)):
        valid_positions = positions[index, selected_mask[index]]
        if valid_positions.size(0) < 2:
            zero = positions.new_tensor(0.0)
            min_values.append(zero)
            mean_values.append(zero)
            continue
        distances = torch.pdist(valid_positions)
        min_values.append(distances.min())
        mean_values.append(distances.mean())
    return {
        "min_pair_distance": torch.stack(min_values),
        "mean_pair_distance": torch.stack(mean_values),
    }


def _round(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return round(float(value), 6)


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
