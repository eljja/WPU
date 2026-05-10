from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.engines.scheduler import ExecutionPath
from wpu.models.factory import MODEL_NAMES, create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["wpu-routed", "wpu-sparse", "wpu-dense", "dense-graph", "serialized-token"])
    parser.add_argument("--background-sizes", type=int, nargs="+", default=[0, 8, 20, 80, 200])
    parser.add_argument("--branches", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/regime_sweep.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        model = create_model(model_name)
        checkpoint = args.checkpoint_dir / f"{model_name}.pt"
        checkpoint_loaded = checkpoint.exists()
        if checkpoint_loaded:
            payload = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        for background_objects in args.background_sizes:
            for branches in args.branches:
                rows.append(_evaluate(model_name, model, background_objects, branches, checkpoint_loaded, args))

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.output}")
    for row in rows:
        print(row)


def _evaluate(
    model_name: str,
    model: torch.nn.Module,
    background_objects: int,
    branches: int,
    checkpoint_loaded: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = ObjectPhysicsDataset(size=args.samples, seed=args.seed, background_objects=background_objects)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    total = 0
    correct = 0
    mse_total = 0.0
    nll_total = 0.0
    elapsed = 0.0
    path_counts = {path: 0 for path in ExecutionPath}
    with torch.no_grad():
        for batch, target_delta, labels in loader:
            start = time.perf_counter()
            prediction = model(batch, num_branches=3, route_branches=branches)
            elapsed += time.perf_counter() - start
            usable_labels = labels
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * labels.numel()
            nll_total += float(F.cross_entropy(prediction.branch_logits, usable_labels).item()) * labels.numel()
            correct += int((prediction.branch_probabilities.argmax(dim=-1) == usable_labels).sum().item())
            total += int(labels.numel())
            for path in prediction.selected_paths:
                path_counts[path] += 1

    total_objects = background_objects + 4
    rho = (1 * (1.0**3) * branches) / total_objects
    sparse_work = branches
    dense_work = total_objects * total_objects
    routed_work = _routed_work(model_name, path_counts, total, sparse_work, total_objects, dense_work)
    return {
        "model": model_name,
        "checkpoint_loaded": checkpoint_loaded,
        "background_objects": background_objects,
        "total_objects": total_objects,
        "branches": branches,
        "rho": round(rho, 6),
        "sparse_work_proxy": sparse_work,
        "dense_work_proxy": dense_work,
        "routed_work_proxy": round(routed_work, 6),
        "mse": round(mse_total / max(total, 1), 6),
        "branch_nll": round(nll_total / max(total, 1), 6),
        "branch_accuracy": round(correct / max(total, 1), 6),
        "ms_per_sample": round((elapsed * 1000.0) / max(total, 1), 6),
        "sparse_ratio": round(path_counts[ExecutionPath.SPARSE] / max(total, 1), 6),
        "hybrid_ratio": round(path_counts[ExecutionPath.HYBRID] / max(total, 1), 6),
        "dense_ratio": round(path_counts[ExecutionPath.DENSE] / max(total, 1), 6),
    }


def _routed_work(
    model_name: str,
    path_counts: dict[ExecutionPath, int],
    total: int,
    sparse_work: int,
    total_objects: int,
    dense_work: int,
) -> float:
    if model_name == "serialized-token":
        return float((total_objects + 4) ** 2)
    if model_name in {"dense-graph", "wpu-dense"}:
        return float(dense_work)
    sparse_ratio = path_counts[ExecutionPath.SPARSE] / max(total, 1)
    hybrid_ratio = path_counts[ExecutionPath.HYBRID] / max(total, 1)
    dense_ratio = path_counts[ExecutionPath.DENSE] / max(total, 1)
    hybrid_work = sparse_work + total_objects
    return sparse_ratio * sparse_work + hybrid_ratio * hybrid_work + dense_ratio * dense_work


if __name__ == "__main__":
    main()
