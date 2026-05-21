from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_working_set_samples
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon CWS drift and consistency smoke evaluation.")
    parser.add_argument("--models", nargs="+", default=["wpu-cws-oracle", "wpu-cws-learned", "serialized-token"])
    parser.add_argument("--background-objects", type=int, default=252)
    parser.add_argument("--causal-obstacles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cws_long_horizon/long_horizon.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        rows.append(_evaluate_model(model_name, args))
    _write_csv(args.output, rows)
    print(f"wrote={args.output}")


def _evaluate_model(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    model.eval()

    total_predictions = 0
    correct = 0
    branch_flips = 0
    drift_total = 0.0
    recall_total = 0.0
    recall_count = 0
    previous_predictions: torch.Tensor | None = None

    for step in range(args.horizon):
        dataset = WorkingSetPhysicsDataset(
            size=args.samples,
            seed=args.seed + step * 10_000,
            background_objects=args.background_objects,
            causal_obstacles=args.causal_obstacles,
        )
        batch, target_delta, labels, causal_k = collate_working_set_samples([dataset[index] for index in range(args.samples)])
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction = model(batch, num_branches=3, route_branches=3)
        predicted = prediction.branch_probabilities.argmax(dim=-1)
        correct += int((predicted == labels).sum().item())
        total_predictions += int(labels.numel())
        drift_total += float(F.mse_loss(prediction.object_delta, target_delta).item())
        if previous_predictions is not None:
            branch_flips += int((predicted != previous_predictions).sum().item())
        previous_predictions = predicted.detach()
        stats = getattr(model, "last_working_set_stats", None)
        if stats is not None:
            recall_total += float(stats.mean_causal_recall)
            recall_count += 1
        else:
            recall_total += 1.0
            recall_count += 1

    flip_denominator = max((args.horizon - 1) * args.samples, 1)
    return {
        "model": model_name,
        "params": _count_parameters(model),
        "total_objects_n": args.background_objects + args.causal_obstacles + 4,
        "causal_k": args.causal_obstacles + 4,
        "horizon": args.horizon,
        "samples": args.samples,
        "branch_accuracy": round(correct / max(total_predictions, 1), 6),
        "branch_flip_rate": round(branch_flips / flip_denominator, 6),
        "delta_mse_mean": round(drift_total / max(args.horizon, 1), 6),
        "causal_recall_mean": round(recall_total / max(recall_count, 1), 6),
    }


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


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
