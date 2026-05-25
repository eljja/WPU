from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.staged_regret_hybrid import (
    _choose_regret_threshold,
    _class_weights,
    _collate_fn,
    _move_batch,
    _regret_fit_metrics,
    _train_propagation,
    _train_regret_head,
)
from wpu.data.working_set_physics import WorkingSetPhysicsDataset
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed sparse-favoring margins for staged WPU regret routing.")
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
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=3e-3)
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
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
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_regret_margin_sweep.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"margin-sweep seed={seed} N={n_value} K={k_value}", flush=True)
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
    calibrated_threshold = _choose_regret_threshold(model, background_objects, causal_obstacles, seed, args, device)
    cached = _collect_eval_tensors(model, background_objects, causal_obstacles, seed, args, device)
    rows = [
        _policy_row(
            cached,
            background_objects,
            causal_obstacles,
            seed,
            args,
            policy="calibrated",
            threshold=calibrated_threshold,
            margin=None,
        )
    ]
    for margin in args.margins:
        rows.append(
            _policy_row(
                cached,
                background_objects,
                causal_obstacles,
                seed,
                args,
                policy=f"margin_{margin:g}",
                threshold=-float(margin),
                margin=float(margin),
            )
        )
    return rows


def _collect_eval_tensors(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = WorkingSetPhysicsDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    sparse_losses: list[torch.Tensor] = []
    dense_losses: list[torch.Tensor] = []
    sparse_predictions: list[torch.Tensor] = []
    dense_predictions: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    regret_predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            sparse_losses.append(F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none").detach().cpu())
            dense_losses.append(
                (F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none") + args.compute_cost).detach().cpu()
            )
            sparse_predictions.append(sparse_prediction.branch_logits.argmax(dim=-1).detach().cpu())
            dense_predictions.append(dense_prediction.branch_logits.argmax(dim=-1).detach().cpu())
            labels_list.append(labels.detach().cpu())
            regret_predictions.append(model.route_regret_prediction().detach().cpu())
    return {
        "sparse_loss": torch.cat(sparse_losses).float(),
        "dense_loss": torch.cat(dense_losses).float(),
        "sparse_pred": torch.cat(sparse_predictions),
        "dense_pred": torch.cat(dense_predictions),
        "labels": torch.cat(labels_list),
        "regret": torch.cat(regret_predictions).float(),
    }


def _policy_row(
    cached: dict[str, torch.Tensor],
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    policy: str,
    threshold: float,
    margin: float | None,
) -> dict[str, object]:
    sparse_loss = cached["sparse_loss"]
    dense_loss = cached["dense_loss"]
    sparse_pred = cached["sparse_pred"]
    dense_pred = cached["dense_pred"]
    labels = cached["labels"]
    regret = cached["regret"]
    route_dense = regret < threshold
    oracle_dense = dense_loss < sparse_loss
    routed_loss = torch.where(route_dense, dense_loss, sparse_loss)
    oracle_loss = torch.minimum(sparse_loss, dense_loss)
    routed_pred = torch.where(route_dense, dense_pred, sparse_pred)
    regret_corr, regret_mse = _regret_fit_metrics([regret], [dense_loss - sparse_loss])
    total = max(int(labels.numel()), 1)
    return {
        "status": "ok",
        "policy": policy,
        "margin": "" if margin is None else round(margin, 6),
        "route_threshold": round(threshold, 6),
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
        "sparse_accuracy": round(float((sparse_pred == labels).float().mean().item()), 6),
        "dense_accuracy": round(float((dense_pred == labels).float().mean().item()), 6),
        "routed_accuracy": round(float((routed_pred == labels).float().mean().item()), 6),
        "sparse_loss": round(float(sparse_loss.mean().item()), 6),
        "dense_loss": round(float(dense_loss.mean().item()), 6),
        "routed_loss": round(float(routed_loss.mean().item()), 6),
        "oracle_loss": round(float(oracle_loss.mean().item()), 6),
        "routed_delta_vs_sparse": round(float((routed_loss - sparse_loss).mean().item()), 6),
        "routed_excess_over_oracle": round(float((routed_loss - oracle_loss).mean().item()), 6),
        "dense_compute_ratio": round(float(route_dense.float().mean().item()), 6),
        "oracle_dense_compute_ratio": round(float(oracle_dense.float().mean().item()), 6),
        "route_regret_mean": round(float(regret.mean().item()), 6),
        "route_regret_eval_corr": round(regret_corr, 6),
        "route_regret_eval_mse": round(regret_mse, 6),
    }


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
