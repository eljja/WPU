from __future__ import annotations

import argparse
from collections import Counter
import csv
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.pybullet_cup import (
    PyBulletCupDataset,
    collate_indexed_pybullet_cup_samples,
    collate_pybullet_cup_samples,
)
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.factory import create_model


DEFAULT_MODELS = [
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "graph-transformer",
    "serialized-token",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator-grounded PyBullet cup benchmark for WPU.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--background-objects", type=int, nargs="+", default=[0, 8, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sim-steps", type=int, default=240)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-params", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runtime-repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_cup_benchmark.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for background_objects in args.background_objects:
            for seed in args.seeds:
                print(f"run model={model_name} seed={seed} background={background_objects}", flush=True)
                rows.append(_run_condition(model_name, background_objects, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(
    model_name: str,
    background_objects: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    hidden_dim = _matched_hidden_dim(model_name, args)
    model = create_model(
        model_name,
        hidden_dim=hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset = PyBulletCupDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    model.train()
    last_loss = 0.0
    for step, (batch, target_delta, labels, _) in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        prediction = model(batch, num_branches=3, route_branches=3)
        loss = F.cross_entropy(prediction.branch_logits, labels, weight=class_weights)
        loss = loss + 0.1 * F.mse_loss(prediction.object_delta, target_delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if step >= args.steps:
            break

    eval_metrics = _evaluate(model, background_objects, seed, args, device)
    runtime_metrics = _profile_runtime(model, background_objects, seed, args, device)
    return {
        "model": model_name,
        "seed": seed,
        "params": _count_parameters(model),
        "hidden_dim": hidden_dim,
        "target_params": args.target_params,
        "param_match_error": abs(_count_parameters(model) - args.target_params) if args.target_params > 0 else 0,
        "layers": args.layers,
        "total_objects_n": background_objects + 5,
        "background_objects": background_objects,
        "balanced_labels": args.balanced_labels,
        "pre_tensor_indexed": _uses_pre_tensor_index(args, model_name),
        "train_loss": round(last_loss, 6),
        **eval_metrics,
        **runtime_metrics,
    }


def _evaluate(
    model: torch.nn.Module,
    background_objects: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset = PyBulletCupDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, _model_name(model)))
    model.eval()
    total = 0
    correct = 0
    mse_total = 0.0
    label_counts: Counter[int] = Counter()
    selected_k_values: list[float] = []
    causal_recall_values: list[float] = []
    dense_compute_values: list[float] = []
    with torch.no_grad():
        for batch, target_delta, labels, causal_k in loader:
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, route_branches=3)
            predicted = prediction.branch_probabilities.argmax(dim=-1)
            batch_total = int(labels.numel())
            total += batch_total
            correct += int((predicted == labels).sum().item())
            label_counts.update(int(label) for label in labels.detach().cpu().tolist())
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * batch_total
            selected_k, causal_recall, dense_compute = _working_set_stats(model, causal_k)
            selected_k_values.append(selected_k)
            causal_recall_values.append(causal_recall)
            dense_compute_values.append(dense_compute)
    model.train()
    return {
        "branch_accuracy": round(correct / max(total, 1), 6),
        "majority_accuracy": round(max(label_counts.values(), default=0) / max(total, 1), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "selected_k_mean": round(sum(selected_k_values) / max(len(selected_k_values), 1), 6),
        "causal_recall_mean": round(sum(causal_recall_values) / max(len(causal_recall_values), 1), 6),
        "dense_compute_ratio": round(sum(dense_compute_values) / max(len(dense_compute_values), 1), 6),
    }


def _profile_runtime(
    model: torch.nn.Module,
    background_objects: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset = PyBulletCupDataset(
        size=args.batch_size,
        seed=seed + 20_000,
        background_objects=background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    batch, _, _, _ = _collate_fn(args, _model_name(model))([dataset[index] for index in range(args.batch_size)])
    batch = _move_batch(batch, device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(2):
            model(batch, num_branches=3, route_branches=3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runtime_repeats):
            model(batch, num_branches=3, route_branches=3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "ms_per_sample_forward": round((elapsed * 1000.0) / (args.runtime_repeats * args.batch_size), 6),
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 6) if device.type == "cuda" else 0.0,
    }


def _class_weights(dataset: PyBulletCupDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _collate_fn(args: argparse.Namespace, model_name: str):
    if not _uses_pre_tensor_index(args, model_name):
        return collate_pybullet_cup_samples

    def collate(samples):
        return collate_indexed_pybullet_cup_samples(
            samples,
            max_nodes=args.working_set_size,
            max_depth=args.index_depth,
        )

    return collate


def _uses_pre_tensor_index(args: argparse.Namespace, model_name: str) -> bool:
    return bool(args.pre_tensor_indexed and model_name.startswith("wpu-cws-indexed"))


def _model_name(model: torch.nn.Module) -> str:
    if isinstance(model, CausalWorkingSetProcessor):
        return "wpu-cws-indexed"
    return model.__class__.__name__


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


def _working_set_stats(model: torch.nn.Module, causal_k: torch.Tensor) -> tuple[float, float, float]:
    if isinstance(model, CausalWorkingSetProcessor) and model.last_working_set_stats is not None:
        stats = model.last_working_set_stats
        return stats.mean_selected, stats.mean_causal_recall, stats.dense_compute_ratio
    return float(causal_k.float().mean().item()), 1.0, 1.0


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _matched_hidden_dim(model_name: str, args: argparse.Namespace) -> int:
    if args.target_params <= 0:
        return args.hidden_dim
    candidates = [
        hidden_dim
        for hidden_dim in range(max(args.num_heads, 8), 513, max(args.num_heads, 8))
        if hidden_dim % args.num_heads == 0
    ]
    best_hidden = args.hidden_dim
    best_error: int | None = None
    for hidden_dim in candidates:
        try:
            model = create_model(
                model_name,
                hidden_dim=hidden_dim,
                layers=args.layers,
                num_heads=args.num_heads,
                working_set_size=args.working_set_size,
            )
        except ValueError:
            continue
        error = abs(_count_parameters(model) - args.target_params)
        if best_error is None or error < best_error:
            best_hidden = hidden_dim
            best_error = error
    return best_hidden


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
