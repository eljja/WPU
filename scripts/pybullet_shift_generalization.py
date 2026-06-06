from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from wpu.data.pybullet_cup import PyBulletCupDataset, PyBulletCupSample, collate_indexed_pybullet_cup_samples, collate_pybullet_cup_samples
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.factory import create_model


DEFAULT_MODELS = [
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "graph-transformer",
    "serialized-token",
]

MECHANISMS = {
    "nominal": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.45,
    },
    "high_force": {
        "force_range": (1.1, 2.4),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.45,
    },
    "edge_shift": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.84, 1.02),
        "catch_probability": 0.45,
    },
    "catch_heavy": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.85,
    },
    "no_catch": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.05,
    },
    "edge_high_force": {
        "force_range": (1.1, 2.4),
        "cup_x_range": (0.84, 1.02),
        "catch_probability": 0.45,
    },
    "edge_catch_heavy": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.84, 1.02),
        "catch_probability": 0.85,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PyBullet cross-mechanism generalization and calibration.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--train-mechanisms", nargs="+", default=["nominal"])
    parser.add_argument("--eval-mechanisms", nargs="+", default=list(MECHANISMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrate-temperature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--temperature-steps", type=int, default=80)
    parser.add_argument("--temperature-lr", type=float, default=5e-2)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_shift_generalization.csv"))
    args = parser.parse_args()

    unknown = [mechanism for mechanism in args.eval_mechanisms if mechanism not in MECHANISMS]
    unknown.extend(mechanism for mechanism in args.train_mechanisms if mechanism not in MECHANISMS)
    if unknown:
        raise ValueError(f"unknown mechanisms: {unknown}")

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for seed in args.seeds:
            train_label = "+".join(args.train_mechanisms)
            print(f"train {train_label} model={model_name} seed={seed}", flush=True)
            model = _train_model(model_name, seed, args)
            temperature = _fit_temperature(model, model_name, seed, args) if args.calibrate_temperature else 1.0
            for mechanism in args.eval_mechanisms:
                print(f"eval model={model_name} seed={seed} mechanism={mechanism}", flush=True)
                rows.append(_evaluate(model, model_name, seed, mechanism, temperature, args))
                _write_csv(args.out, rows)
    rows.extend(_summary(rows))
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _train_model(model_name: str, seed: int, args: argparse.Namespace) -> torch.nn.Module:
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
    train_dataset = _training_dataset(seed, args)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    model.train()
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
        if step >= args.steps:
            break
    return model


def _training_dataset(seed: int, args: argparse.Namespace):
    total_samples = max(args.steps * args.batch_size, args.batch_size)
    per_mechanism = max(args.batch_size, total_samples // max(len(args.train_mechanisms), 1))
    datasets = [
        _dataset(
            mechanism=mechanism,
            samples=per_mechanism,
            seed=seed + 101 * index,
            args=args,
            balanced_labels=args.balanced_labels,
        )
        for index, mechanism in enumerate(args.train_mechanisms)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _fit_temperature(model: torch.nn.Module, model_name: str, seed: int, args: argparse.Namespace) -> float:
    device = torch.device(args.device)
    dataset = _dataset(
        mechanism=args.train_mechanisms[0],
        samples=args.calibration_samples,
        seed=seed + 30_000,
        args=args,
        balanced_labels=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([log_temperature], lr=args.temperature_lr)
    model.eval()
    for _ in range(args.temperature_steps):
        total_loss = torch.zeros((), device=device)
        total_count = 0
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(batch, num_branches=3, route_branches=3).branch_logits.detach()
            temperature = log_temperature.exp().clamp(0.25, 8.0)
            total_loss = total_loss + F.cross_entropy(logits / temperature, labels, reduction="sum")
            total_count += int(labels.numel())
        loss = total_loss / max(total_count, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return round(float(log_temperature.detach().exp().clamp(0.25, 8.0).cpu().item()), 6)


def _evaluate(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    mechanism: str,
    temperature: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    dataset = _dataset(
        mechanism=mechanism,
        samples=args.samples,
        seed=seed + 10_000,
        args=args,
        balanced_labels=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    model.eval()
    total = 0
    correct = 0
    mse_total = 0.0
    nll_total = 0.0
    brier_total = 0.0
    confidences: list[float] = []
    correctness: list[float] = []
    selected_k_values: list[float] = []
    causal_recall_values: list[float] = []
    dense_compute_values: list[float] = []
    label_counts: Counter[int] = Counter()
    with torch.no_grad():
        for batch, target_delta, labels, causal_k in loader:
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, route_branches=3)
            probabilities = F.softmax(prediction.branch_logits / max(float(temperature), 1e-6), dim=-1)
            predicted = probabilities.argmax(dim=-1)
            batch_total = int(labels.numel())
            total += batch_total
            correct_mask = predicted == labels
            correct += int(correct_mask.sum().item())
            label_counts.update(int(label) for label in labels.detach().cpu().tolist())
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * batch_total
            nll_total += float(F.nll_loss(probabilities.clamp_min(1e-8).log(), labels, reduction="sum").item())
            brier_total += float(_brier_score(probabilities, labels).item()) * batch_total
            confidence = probabilities.max(dim=-1).values.detach().cpu()
            confidences.extend(float(value) for value in confidence.tolist())
            correctness.extend(float(value) for value in correct_mask.detach().cpu().tolist())
            selected_k, causal_recall, dense_compute = _working_set_stats(model, causal_k)
            selected_k_values.append(selected_k)
            causal_recall_values.append(causal_recall)
            dense_compute_values.append(dense_compute)
    model.train()
    return {
        "row_type": "seed",
        "model": model_name,
        "seed": seed,
        "train_mechanism": "+".join(args.train_mechanisms),
        "eval_mechanism": mechanism,
        "temperature": temperature,
        "background_objects": args.background_objects,
        "total_objects_n": args.background_objects + 5,
        "samples": args.samples,
        "branch_accuracy": round(correct / max(total, 1), 6),
        "majority_accuracy": round(max(label_counts.values(), default=0) / max(total, 1), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "nll": round(nll_total / max(total, 1), 6),
        "brier": round(brier_total / max(total, 1), 6),
        "ece": round(_ece(confidences, correctness), 6),
        "selected_k_mean": round(sum(selected_k_values) / max(len(selected_k_values), 1), 6),
        "causal_recall_mean": round(sum(causal_recall_values) / max(len(causal_recall_values), 1), 6),
        "dense_compute_ratio": round(sum(dense_compute_values) / max(len(dense_compute_values), 1), 6),
        "seed_count": 1,
    }


def _dataset(*, mechanism: str, samples: int, seed: int, args: argparse.Namespace, balanced_labels: bool) -> PyBulletCupDataset:
    config = MECHANISMS[mechanism]
    return PyBulletCupDataset(
        size=samples,
        seed=seed,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=balanced_labels,
        force_range=config["force_range"],
        cup_x_range=config["cup_x_range"],
        catch_probability=float(config["catch_probability"]),
    )


def _brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = F.one_hot(labels, num_classes=probabilities.size(-1)).float()
    return ((probabilities - target) ** 2).sum(dim=-1).mean()


def _ece(confidences: list[float], correctness: list[float], bins: int = 10) -> float:
    total = len(confidences)
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if lower < confidence <= upper or (bin_index == 0 and confidence == 0.0)
        ]
        if not indices:
            continue
        bin_confidence = sum(confidences[index] for index in indices) / len(indices)
        bin_accuracy = sum(correctness[index] for index in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_confidence - bin_accuracy)
    return ece


def _class_weights(dataset: PyBulletCupDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _collate_fn(args: argparse.Namespace, model_name: str):
    if not bool(args.pre_tensor_indexed and model_name.startswith("wpu-cws-indexed")):
        return collate_pybullet_cup_samples

    def collate(samples: list[PyBulletCupSample]):
        return collate_indexed_pybullet_cup_samples(samples, max_nodes=args.working_set_size, max_depth=args.index_depth)

    return collate


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


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["row_type"] != "seed":
            continue
        grouped.setdefault((str(row["model"]), str(row["train_mechanism"]), str(row["eval_mechanism"])), []).append(row)
    output: list[dict[str, object]] = []
    numeric_fields = [
        "branch_accuracy",
        "majority_accuracy",
        "mse",
        "nll",
        "brier",
        "ece",
        "selected_k_mean",
        "causal_recall_mean",
        "dense_compute_ratio",
        "temperature",
    ]
    for (model, train_mechanism, mechanism), group in sorted(grouped.items()):
        row = {
            "row_type": "summary",
            "model": model,
            "seed": "all",
            "train_mechanism": train_mechanism,
            "eval_mechanism": mechanism,
            "background_objects": group[0]["background_objects"],
            "total_objects_n": group[0]["total_objects_n"],
            "samples": group[0]["samples"],
            "seed_count": len(group),
        }
        for field in numeric_fields:
            row[field] = round(sum(float(item[field]) for item in group) / len(group), 6)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
