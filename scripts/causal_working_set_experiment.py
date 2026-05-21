from __future__ import annotations

import argparse
from collections import Counter
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_working_set_samples
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.factory import create_model


DEFAULT_MODELS = [
    "wpu-cws-frontier",
    "wpu-cws-oracle",
    "wpu-cws-learned",
    "serialized-token",
    "graph-transformer",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare causal-working-set WPU against token/graph baselines.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-values", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--distractor-values", type=int, nargs="+", default=[0, 8, 16, 32])
    parser.add_argument("--fixed-k", type=int, default=8)
    parser.add_argument("--mode", choices=["n-sweep", "k-sweep", "distractor-sweep"], default="n-sweep")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--working-set-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runtime-repeats", type=int, default=20)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/causal_working_set_v1"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or [args.seed]
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for value in _sweep_values(args):
            background_objects, causal_obstacles, adversarial_distractors = _condition(value, args)
            for seed in seeds:
                try:
                    total_n = _total_objects(background_objects, causal_obstacles, adversarial_distractors)
                    print(f"run model={model_name} seed={seed} N={total_n} K={4 + causal_obstacles} distractors={adversarial_distractors}")
                    rows.append(_run_condition(model_name, background_objects, causal_obstacles, adversarial_distractors, seed, args))
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    rows.append(_failed_row(model_name, background_objects, causal_obstacles, adversarial_distractors, seed, args, f"cuda_oom: {error}"))
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        rows.append(_failed_row(model_name, background_objects, causal_obstacles, adversarial_distractors, seed, args, f"oom: {error}"))
                    else:
                        raise
    _write_csv(args.out_dir / f"{args.mode}.csv", rows)
    print(f"wrote={args.out_dir / f'{args.mode}.csv'}")


def _sweep_values(args: argparse.Namespace) -> list[int]:
    if args.mode == "n-sweep":
        return args.n_values
    if args.mode == "k-sweep":
        return args.k_values
    return args.distractor_values


def _condition(value: int, args: argparse.Namespace) -> tuple[int, int, int]:
    if args.mode == "n-sweep":
        causal_obstacles = max(0, args.fixed_k - 4)
        background_objects = max(0, value - 4 - causal_obstacles)
        return background_objects, causal_obstacles, 0
    if args.mode == "k-sweep":
        causal_obstacles = max(0, value - 4)
        background_objects = max(0, max(args.n_values) - 4 - causal_obstacles)
        return background_objects, causal_obstacles, 0
    causal_obstacles = max(0, args.fixed_k - 4)
    total_n = max(args.n_values)
    adversarial_distractors = value
    background_objects = max(0, total_n - 4 - causal_obstacles - adversarial_distractors)
    return background_objects, causal_obstacles, adversarial_distractors


def _run_condition(
    model_name: str,
    background_objects: int,
    causal_obstacles: int,
    adversarial_distractors: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset = WorkingSetPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        adversarial_distractors=adversarial_distractors,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_working_set_samples)
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    model.train()
    last_loss = 0.0
    for step, (batch, target_delta, labels, _) in enumerate(loader, start=1):
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        prediction = model(batch, num_branches=3, route_branches=3)
        loss = F.mse_loss(prediction.object_delta, target_delta)
        loss = loss + F.cross_entropy(prediction.branch_logits, labels, weight=class_weights)
        if args.selector_loss_weight > 0.0 and hasattr(model, "selector_loss"):
            loss = loss + args.selector_loss_weight * model.selector_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if step >= args.steps:
            break

    eval_metrics = _evaluate(model, background_objects, causal_obstacles, adversarial_distractors, seed, args, device)
    runtime_metrics = _profile_runtime(model, background_objects, causal_obstacles, adversarial_distractors, seed, args, device)
    checkpoint_path = ""
    if args.save_checkpoints:
        checkpoint_path = _save_checkpoint(model, model_name, background_objects, causal_obstacles, adversarial_distractors, seed, args)
    return {
        "status": "ok",
        "model": model_name,
        "seed": seed,
        "params": _count_parameters(model),
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "total_objects_n": _total_objects(background_objects, causal_obstacles, adversarial_distractors),
        "causal_k": 4 + causal_obstacles,
        "adversarial_distractors": adversarial_distractors,
        "background_objects": background_objects,
        "train_loss": round(last_loss, 6),
        "checkpoint": checkpoint_path,
        **eval_metrics,
        **runtime_metrics,
    }


def _evaluate(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    adversarial_distractors: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset = WorkingSetPhysicsDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        adversarial_distractors=adversarial_distractors,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_working_set_samples)
    model.eval()
    total = 0
    correct = 0
    mse_total = 0.0
    label_counts: Counter[int] = Counter()
    selected_k_values: list[float] = []
    causal_recall_values: list[float] = []
    with torch.no_grad():
        for batch, target_delta, labels, causal_k in loader:
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, route_branches=3)
            predicted = prediction.branch_probabilities.argmax(dim=-1)
            batch_total = int(labels.numel())
            total += batch_total
            label_counts.update(int(label) for label in labels.detach().cpu().tolist())
            correct += int((predicted == labels).sum().item())
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * batch_total
            selected_k, causal_recall = _working_set_stats(model, causal_k)
            selected_k_values.append(selected_k)
            causal_recall_values.append(causal_recall)
    model.train()
    return {
        "branch_accuracy": round(correct / max(total, 1), 6),
        "majority_accuracy": round(max(label_counts.values(), default=0) / max(total, 1), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "selected_k_mean": round(sum(selected_k_values) / max(len(selected_k_values), 1), 6),
        "causal_recall_mean": round(sum(causal_recall_values) / max(len(causal_recall_values), 1), 6),
    }


def _profile_runtime(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    adversarial_distractors: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset = WorkingSetPhysicsDataset(
        size=args.batch_size,
        seed=seed + 20_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        adversarial_distractors=adversarial_distractors,
    )
    batch, _, _, _ = collate_working_set_samples([dataset[index] for index in range(args.batch_size)])
    batch = _move_batch(batch, device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):
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


def _working_set_stats(model: torch.nn.Module, causal_k: torch.Tensor) -> tuple[float, float]:
    if isinstance(model, CausalWorkingSetProcessor) and model.last_working_set_stats is not None:
        return model.last_working_set_stats.mean_selected, model.last_working_set_stats.mean_causal_recall
    return float(causal_k.float().mean().item()), 1.0


def _class_weights(dataset: WorkingSetPhysicsDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _total_objects(background_objects: int, causal_obstacles: int, adversarial_distractors: int) -> int:
    return background_objects + 4 + causal_obstacles + adversarial_distractors


def _failed_row(
    model_name: str,
    background_objects: int,
    causal_obstacles: int,
    adversarial_distractors: int,
    seed: int,
    args: argparse.Namespace,
    error: str,
) -> dict[str, object]:
    return {
        "status": "failed",
        "model": model_name,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "total_objects_n": _total_objects(background_objects, causal_obstacles, adversarial_distractors),
        "causal_k": 4 + causal_obstacles,
        "adversarial_distractors": adversarial_distractors,
        "background_objects": background_objects,
        "error": error[:500],
    }


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _save_checkpoint(
    model: torch.nn.Module,
    model_name: str,
    background_objects: int,
    causal_obstacles: int,
    adversarial_distractors: int,
    seed: int,
    args: argparse.Namespace,
) -> str:
    safe_model = model_name.replace("/", "_")
    total_n = _total_objects(background_objects, causal_obstacles, adversarial_distractors)
    path = args.out_dir / "checkpoints" / f"{safe_model}_N{total_n}_K{4 + causal_obstacles}_D{adversarial_distractors}_seed{seed}.pt"
    torch.save(
        {
            "model": model_name,
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "num_heads": args.num_heads,
            "working_set_size": args.working_set_size,
            "total_objects_n": total_n,
            "causal_k": 4 + causal_obstacles,
            "adversarial_distractors": adversarial_distractors,
            "seed": seed,
        },
        path,
    )
    return path.as_posix()


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
