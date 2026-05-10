from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.engines.scheduler import ExecutionPath
from wpu.models.factory import MODEL_NAMES, create_model


DEFAULT_MODELS = ["wpu-routed", "wpu-sparse", "wpu-hybrid", "graph-transformer", "serialized-token"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled WPU stress tests for relation noise and affected fraction.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=DEFAULT_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--background-objects", type=int, default=80)
    parser.add_argument("--relation-noise", type=int, nargs="+", default=[0, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--affected-counts", type=int, nargs="+", default=[0, 4, 16, 32, 64])
    parser.add_argument("--background-delta-scale", type=float, default=0.01)
    parser.add_argument("--skip-relation-noise", action="store_true")
    parser.add_argument("--skip-affected-count", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/controlled_stress_v1"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    relation_rows = [] if args.skip_relation_noise else _run_relation_noise(args)
    affected_rows = [] if args.skip_affected_count else _run_affected_count(args)
    _write_csv(args.out_dir / "relation_noise.csv", relation_rows)
    _write_csv(args.out_dir / "affected_count.csv", affected_rows)
    _write_summary(args.out_dir / "summary.md", relation_rows, affected_rows)


def _run_relation_noise(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"relation_noise train clean model={model_name} seed={seed}")
            model = _train_model(
                model_name=model_name,
                seed=seed,
                args=args,
                relation_noise=0,
                affected_background_objects=0,
                background_delta_scale=args.background_delta_scale,
            )
            for relation_noise in args.relation_noise:
                metrics = _evaluate(
                    model=model,
                    model_name=model_name,
                    seed=seed + 10_000,
                    args=args,
                    relation_noise=relation_noise,
                    affected_background_objects=0,
                    background_delta_scale=args.background_delta_scale,
                )
                rows.append({"experiment": "relation_noise", "seed": seed, **metrics})
    return rows


def _run_affected_count(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for affected_count in args.affected_counts:
        for model_name in args.models:
            for seed in args.seeds:
                print(f"affected_count train model={model_name} affected={affected_count} seed={seed}")
                model = _train_model(
                    model_name=model_name,
                    seed=seed + affected_count * 101,
                    args=args,
                    relation_noise=0,
                    affected_background_objects=affected_count,
                    background_delta_scale=args.background_delta_scale,
                )
                metrics = _evaluate(
                    model=model,
                    model_name=model_name,
                    seed=seed + 20_000 + affected_count * 101,
                    args=args,
                    relation_noise=0,
                    affected_background_objects=affected_count,
                    background_delta_scale=args.background_delta_scale,
                )
                rows.append({"experiment": "affected_count", "seed": seed, **metrics})
    return rows


def _train_model(
    *,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    relation_noise: int,
    affected_background_objects: int,
    background_delta_scale: float,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    dataset = ObjectPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=args.background_objects,
        relation_noise=relation_noise,
        affected_background_objects=affected_background_objects,
        background_delta_scale=background_delta_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model = create_model(model_name, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = _class_weights(dataset)
    model.train()
    for step, (batch, target_delta, branch_label) in enumerate(loader, start=1):
        prediction = model(batch, num_branches=3, route_branches=3)
        delta_loss = F.mse_loss(prediction.object_delta, target_delta)
        branch_loss = F.cross_entropy(prediction.branch_logits, branch_label, weight=class_weights)
        loss = delta_loss + branch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= args.steps:
            break
    model.eval()
    return model


def _class_weights(dataset: ObjectPhysicsDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _evaluate(
    *,
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    relation_noise: int,
    affected_background_objects: int,
    background_delta_scale: float,
) -> dict[str, object]:
    dataset = ObjectPhysicsDataset(
        size=args.samples,
        seed=seed,
        background_objects=args.background_objects,
        relation_noise=relation_noise,
        affected_background_objects=affected_background_objects,
        background_delta_scale=background_delta_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    total = 0
    correct = 0
    mse_total = 0.0
    cup_mse_total = 0.0
    bg_mse_total = 0.0
    bg_count = 0
    path_counts = {path: 0 for path in ExecutionPath}
    with torch.no_grad():
        for batch, target_delta, labels in loader:
            prediction = model(batch, num_branches=3, route_branches=3)
            batch_total = int(labels.numel())
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * batch_total
            cup_mse_total += float(F.mse_loss(prediction.object_delta[:, :4], target_delta[:, :4]).item()) * batch_total
            if affected_background_objects > 0:
                affected_slice = slice(4, 4 + min(affected_background_objects, args.background_objects))
                bg_mse_total += float(F.mse_loss(prediction.object_delta[:, affected_slice], target_delta[:, affected_slice]).item()) * batch_total
                bg_count += batch_total
            predicted = prediction.branch_probabilities.argmax(dim=-1)
            correct += int((predicted == labels).sum().item())
            total += batch_total
            for path in prediction.selected_paths:
                path_counts[path] += 1
    return {
        "model": model_name,
        "background_objects": args.background_objects,
        "total_objects": args.background_objects + 4,
        "relation_noise": relation_noise,
        "affected_background_objects": affected_background_objects,
        "background_delta_scale": background_delta_scale,
        "affected_fraction": round((affected_background_objects + 1) / (args.background_objects + 4), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "cup_mse": round(cup_mse_total / max(total, 1), 6),
        "affected_background_mse": round(bg_mse_total / max(bg_count, 1), 6),
        "branch_accuracy": round(correct / max(total, 1), 6),
        "sparse_ratio": round(path_counts[ExecutionPath.SPARSE] / max(total, 1), 6),
        "hybrid_ratio": round(path_counts[ExecutionPath.HYBRID] / max(total, 1), 6),
        "dense_ratio": round(path_counts[ExecutionPath.DENSE] / max(total, 1), 6),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={path}")


def _write_summary(path: Path, relation_rows: list[dict[str, object]], affected_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Controlled Stress Suite v1",
        "",
        "## Relation Noise",
        "",
        _table(_aggregate(relation_rows, ["model", "relation_noise"], ["branch_accuracy", "mse", "cup_mse"])),
        "",
        "## Affected Background Count",
        "",
        _table(_aggregate(affected_rows, ["model", "affected_background_objects"], ["branch_accuracy", "mse", "cup_mse", "affected_background_mse"])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={path}")


def _aggregate(rows: list[dict[str, object]], group_keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out = {name: value for name, value in zip(group_keys, key, strict=True)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = round(mean(values), 6)
            out[f"{metric}_ci95"] = round(_ci95(values), 6)
        output.append(out)
    return output


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
