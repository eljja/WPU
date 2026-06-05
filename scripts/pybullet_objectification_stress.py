from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.core.objectification import evaluate_objectification
from wpu.data.pybullet_cup import (
    ObjectificationCorruptionConfig,
    PyBulletCupDataset,
    PyBulletCupSample,
    collate_indexed_pybullet_cup_samples,
    collate_pybullet_cup_samples,
    corrupt_pybullet_cup_sample,
)
from wpu.models.causal_working_set_processor import CausalWorkingSetProcessor
from wpu.models.factory import create_model


DEFAULT_MODELS = [
    "wpu-cws-indexed-sparse",
    "wpu-cws-indexed-local-dense",
    "graph-transformer",
]


CORRUPTION_PRESETS: dict[str, ObjectificationCorruptionConfig] = {
    "clean": ObjectificationCorruptionConfig(),
    "drop_relations_light": ObjectificationCorruptionConfig(relation_drop_rate=0.25),
    "drop_relations_heavy": ObjectificationCorruptionConfig(relation_drop_rate=0.60),
    "drop_objects_light": ObjectificationCorruptionConfig(non_target_object_drop_rate=0.20),
    "position_noise": ObjectificationCorruptionConfig(position_noise_std=0.08),
    "low_confidence": ObjectificationCorruptionConfig(confidence_scale=0.55),
    "identity_swap": ObjectificationCorruptionConfig(identity_swap_rate=1.0),
    "combined": ObjectificationCorruptionConfig(
        relation_drop_rate=0.35,
        non_target_object_drop_rate=0.15,
        position_noise_std=0.05,
        confidence_scale=0.70,
        identity_swap_rate=0.50,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WPU robustness to objectification corruption on PyBullet state.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTION_PRESETS))
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_objectification_stress.csv"))
    args = parser.parse_args()

    unknown = [name for name in args.corruptions if name not in CORRUPTION_PRESETS]
    if unknown:
        raise ValueError(f"unknown corruption presets: {unknown}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"train model={model_name} seed={seed}", flush=True)
            model = _train_clean_model(model_name, seed, args)
            for corruption_name in args.corruptions:
                print(f"eval model={model_name} seed={seed} corruption={corruption_name}", flush=True)
                rows.append(_evaluate_corruption(model, model_name, seed, corruption_name, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _train_clean_model(model_name: str, seed: int, args: argparse.Namespace) -> torch.nn.Module:
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
    dataset = PyBulletCupDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    class_weights = _class_weights(dataset).to(device) if args.class_weights else None
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


def _evaluate_corruption(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    corruption_name: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    base_dataset = PyBulletCupDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=args.background_objects,
        steps=args.sim_steps,
        balanced_labels=args.balanced_labels,
    )
    config = CORRUPTION_PRESETS[corruption_name]
    samples = [
        corrupt_pybullet_cup_sample(
            base_dataset[index],
            config=config,
            seed=seed * 100_000 + index,
        )
        for index in range(args.samples)
    ]
    loader = DataLoader(samples, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    model.eval()
    total = 0
    correct = 0
    mse_total = 0.0
    label_counts: Counter[int] = Counter()
    objectification_scores: list[float] = []
    identity_coverage_values: list[float] = []
    relation_validity_values: list[float] = []
    object_confidence_values: list[float] = []
    relation_confidence_values: list[float] = []
    relation_counts: list[int] = []
    object_counts: list[int] = []
    selected_k_values: list[float] = []
    model_causal_recall_values: list[float] = []
    frontier_recall_values: list[float] = []
    with torch.no_grad():
        for sample in samples:
            report = evaluate_objectification(sample.state)
            objectification_scores.append(report.contract_score)
            identity_coverage_values.append(report.identity_coverage)
            relation_validity_values.append(report.relation_validity)
            object_confidence_values.append(report.object_confidence)
            relation_confidence_values.append(report.relation_confidence)
            relation_counts.append(report.relation_count)
            object_counts.append(report.object_count)
            frontier_recall_values.append(_expected_frontier_recall(sample))

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
            selected_k, causal_recall = _working_set_stats(model, causal_k)
            selected_k_values.append(selected_k)
            model_causal_recall_values.append(causal_recall)
    model.train()
    return {
        "model": model_name,
        "seed": seed,
        "corruption": corruption_name,
        "background_objects": args.background_objects,
        "total_objects_n": args.background_objects + 5,
        "branch_accuracy": round(correct / max(total, 1), 6),
        "majority_accuracy": round(max(label_counts.values(), default=0) / max(total, 1), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "objectification_score": round(_mean(objectification_scores), 6),
        "identity_coverage": round(_mean(identity_coverage_values), 6),
        "relation_validity": round(_mean(relation_validity_values), 6),
        "object_confidence": round(_mean(object_confidence_values), 6),
        "relation_confidence": round(_mean(relation_confidence_values), 6),
        "object_count_mean": round(_mean(object_counts), 6),
        "relation_count_mean": round(_mean(relation_counts), 6),
        "selected_k_mean": round(_mean(selected_k_values), 6),
        "model_causal_recall_mean": round(_mean(model_causal_recall_values), 6),
        "frontier_causal_recall_mean": round(_mean(frontier_recall_values), 6),
    }


def _class_weights(dataset: PyBulletCupDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _collate_fn(args: argparse.Namespace, model_name: str):
    if not _uses_pre_tensor_index(args, model_name):
        return collate_pybullet_cup_samples

    def collate(samples: list[PyBulletCupSample]):
        return collate_indexed_pybullet_cup_samples(
            samples,
            max_nodes=args.working_set_size,
            max_depth=args.index_depth,
        )

    return collate


def _uses_pre_tensor_index(args: argparse.Namespace, model_name: str) -> bool:
    return bool(args.pre_tensor_indexed and model_name.startswith("wpu-cws-indexed"))


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
        stats = model.last_working_set_stats
        return stats.mean_selected, stats.mean_causal_recall
    return float(causal_k.float().mean().item()), 1.0


def _expected_frontier_recall(sample: PyBulletCupSample) -> float:
    expected = {"cup_001", "table_001", "hand_001", "edge_001"}
    if sample.simulator_metadata.get("catch_action"):
        expected.add("catcher_001")
    selected = {sample.event.target}
    for relation in sample.state.relations_for(sample.event.target):
        other = relation.other(sample.event.target)
        if other is not None:
            selected.add(other)
    return len(expected & selected) / max(len(expected), 1)


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


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
