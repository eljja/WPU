from __future__ import annotations

import argparse
import csv
import math
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.engines.scheduler import ExecutionPath
from wpu.models.factory import MODEL_NAMES, create_model

try:
    import psutil
except ImportError:  # pragma: no cover - optional profiling dependency
    psutil = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust WPU/token/graph experiments with raw CSV outputs.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--eval-at", type=int, nargs="+", default=[0, 25, 50, 100, 150])
    parser.add_argument("--train-background", type=int, default=80)
    parser.add_argument("--eval-backgrounds", type=int, nargs="+", default=[0, 20, 80, 200])
    parser.add_argument("--branches", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--runtime-repeats", type=int, default=30)
    parser.add_argument("--runtime-warmup", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/robust"))
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    learning_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    eval_at = sorted(set(step for step in args.eval_at if 0 <= step <= args.steps))
    if args.steps not in eval_at:
        eval_at.append(args.steps)

    for model_name in args.models:
        for seed in args.seeds:
            print(f"train model={model_name} seed={seed}")
            model = _train_with_checkpoints(model_name, seed, args, eval_at, learning_rows)

            for background_objects in args.eval_backgrounds:
                metrics = _evaluate(model, model_name, background_objects, branches=3, seed=seed + 10_000, args=args)
                final_rows.append({"experiment": "final_baseline", "seed": seed, **metrics})

            for background_objects in args.eval_backgrounds:
                for branches in args.branches:
                    metrics = _evaluate(model, model_name, background_objects, branches=branches, seed=seed + 20_000, args=args)
                    regime_rows.append({"experiment": "regime", "seed": seed, **metrics})

            for background_objects in args.eval_backgrounds:
                for branches in args.branches:
                    runtime_rows.append(
                        {
                            "experiment": "runtime_memory",
                            "seed": seed,
                            **_profile_runtime(model, model_name, background_objects, branches, seed + 30_000, args),
                        }
                    )

    _write_csv(args.out_dir / "learning_curves.csv", learning_rows)
    _write_csv(args.out_dir / "final_baselines.csv", final_rows)
    _write_csv(args.out_dir / "regime_sweep.csv", regime_rows)
    _write_csv(args.out_dir / "runtime_memory.csv", runtime_rows)
    _write_summary(args.out_dir / "summary.md", learning_rows, final_rows, regime_rows, runtime_rows)
    print(f"wrote_dir={args.out_dir}")


def _train_with_checkpoints(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    eval_at: list[int],
    learning_rows: list[dict[str, object]],
) -> torch.nn.Module:
    torch.manual_seed(seed)
    dataset = ObjectPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=args.train_background,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model = create_model(model_name, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = _class_weights(dataset)
    eval_points = set(eval_at)

    if 0 in eval_points:
        metrics = _evaluate(model, model_name, args.train_background, branches=3, seed=seed + 1_000, args=args)
        learning_rows.append({"experiment": "learning_curve", "seed": seed, "step": 0, **metrics})

    model.train()
    for step, (batch, target_delta, branch_label) in enumerate(loader, start=1):
        prediction = model(batch, num_branches=3, route_branches=3)
        delta_loss = F.mse_loss(prediction.object_delta, target_delta)
        branch_loss = F.cross_entropy(prediction.branch_logits, branch_label, weight=class_weights)
        loss = delta_loss + branch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in eval_points:
            metrics = _evaluate(model, model_name, args.train_background, branches=3, seed=seed + 1_000, args=args)
            learning_rows.append(
                {
                    "experiment": "learning_curve",
                    "seed": seed,
                    "step": step,
                    "train_loss": round(float(loss.item()), 6),
                    **metrics,
                }
            )
        if step >= args.steps:
            break
    return model


def _class_weights(dataset: ObjectPhysicsDataset) -> torch.Tensor:
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    weights = torch.tensor([len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)], dtype=torch.float32)
    return weights / weights.mean()


def _evaluate(
    model: torch.nn.Module,
    model_name: str,
    background_objects: int,
    branches: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = ObjectPhysicsDataset(size=args.samples, seed=seed, background_objects=background_objects)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model.eval()
    total = 0
    correct = 0
    mse_total = 0.0
    nll_total = 0.0
    confidences: list[float] = []
    correctness: list[float] = []
    path_counts = {path: 0 for path in ExecutionPath}
    with torch.no_grad():
        for batch, target_delta, labels in loader:
            prediction = model(batch, num_branches=3, route_branches=branches)
            probs = prediction.branch_probabilities
            predicted = probs.argmax(dim=-1)
            batch_total = int(labels.numel())
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * batch_total
            nll_total += float(F.cross_entropy(prediction.branch_logits, labels).item()) * batch_total
            correct_mask = predicted == labels
            correct += int(correct_mask.sum().item())
            total += batch_total
            confidences.extend(float(value) for value in probs.max(dim=-1).values.tolist())
            correctness.extend(float(value) for value in correct_mask.float().tolist())
            for path in prediction.selected_paths:
                path_counts[path] += 1
    model.train()
    total_objects = background_objects + 4
    sparse_work = branches
    dense_work = total_objects * total_objects
    return {
        "model": model_name,
        "background_objects": background_objects,
        "total_objects": total_objects,
        "branches": branches,
        "rho": round(branches / total_objects, 6),
        "params": _count_parameters(model),
        "mse": round(mse_total / max(total, 1), 6),
        "branch_nll": round(nll_total / max(total, 1), 6),
        "branch_accuracy": round(correct / max(total, 1), 6),
        "branch_ece": round(_ece(confidences, correctness), 6),
        "sparse_work_proxy": sparse_work,
        "dense_work_proxy": dense_work,
        "routed_work_proxy": round(_routed_work(model_name, path_counts, total, sparse_work, total_objects, dense_work), 6),
        "sparse_ratio": round(path_counts[ExecutionPath.SPARSE] / max(total, 1), 6),
        "hybrid_ratio": round(path_counts[ExecutionPath.HYBRID] / max(total, 1), 6),
        "dense_ratio": round(path_counts[ExecutionPath.DENSE] / max(total, 1), 6),
    }


def _profile_runtime(
    model: torch.nn.Module,
    model_name: str,
    background_objects: int,
    branches: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = ObjectPhysicsDataset(size=args.batch_size, seed=seed, background_objects=background_objects)
    batch, _, _ = collate_physics_samples([dataset[index] for index in range(args.batch_size)])
    model.eval()
    with torch.no_grad():
        for _ in range(args.runtime_warmup):
            model(batch, num_branches=3, route_branches=branches)

    process = psutil.Process() if psutil is not None else None
    rss_before = process.memory_info().rss if process is not None else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runtime_repeats):
            model(batch, num_branches=3, route_branches=branches)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss if process is not None else 0
    cuda_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    total_objects = background_objects + 4
    return {
        "model": model_name,
        "background_objects": background_objects,
        "total_objects": total_objects,
        "branches": branches,
        "rho": round(branches / total_objects, 6),
        "params": _count_parameters(model),
        "ms_per_sample_forward": round((elapsed * 1000.0) / (args.runtime_repeats * args.batch_size), 6),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 6),
        "rss_delta_mb": round((rss_after - rss_before) / (1024 * 1024), 6) if process is not None else "",
        "cuda_peak_mb": round(cuda_peak / (1024 * 1024), 6),
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
    if model_name in {"dense-graph", "graph-transformer", "wpu-dense"}:
        return float(dense_work)
    sparse_ratio = path_counts[ExecutionPath.SPARSE] / max(total, 1)
    hybrid_ratio = path_counts[ExecutionPath.HYBRID] / max(total, 1)
    dense_ratio = path_counts[ExecutionPath.DENSE] / max(total, 1)
    hybrid_work = sparse_work + total_objects
    return sparse_ratio * sparse_work + hybrid_ratio * hybrid_work + dense_ratio * dense_work


def _ece(confidences: list[float], correctness: list[float], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        selected = [i for i, confidence in enumerate(confidences) if lo <= confidence < hi or (index == bins - 1 and confidence == 1.0)]
        if not selected:
            continue
        avg_confidence = mean(confidences[i] for i in selected)
        avg_accuracy = mean(correctness[i] for i in selected)
        error += (len(selected) / total) * abs(avg_confidence - avg_accuracy)
    return error


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
    print(f"wrote={path}")


def _write_summary(
    path: Path,
    learning_rows: list[dict[str, object]],
    final_rows: list[dict[str, object]],
    regime_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Robust WPU Experiment Summary",
        "",
        "This file is generated from raw CSVs. Values are mean with approximate 95% confidence intervals over seeds.",
        "",
        "## Final Baselines",
        "",
        _markdown_table(_aggregate(final_rows, ["model", "total_objects"], ["branch_accuracy", "branch_ece", "mse", "routed_work_proxy"])),
        "",
        "## Regime Sweep",
        "",
        _markdown_table(_aggregate(regime_rows, ["model", "total_objects", "branches"], ["branch_accuracy", "routed_work_proxy", "sparse_ratio", "hybrid_ratio", "dense_ratio"])),
        "",
        "## Runtime And Memory",
        "",
        _markdown_table(_aggregate(runtime_rows, ["model", "total_objects", "branches"], ["ms_per_sample_forward", "tracemalloc_peak_mb", "rss_delta_mb"])),
        "",
        "## Learning Curve Endpoints",
        "",
        _markdown_table(_aggregate(learning_rows, ["model", "step"], ["branch_accuracy", "branch_ece", "mse"])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={path}")


def _aggregate(rows: list[dict[str, object]], group_keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out = {name: value for name, value in zip(group_keys, key, strict=True)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in group if _to_float(row.get(metric)) is not None]
            if not values:
                continue
            avg = mean(values)
            ci = _ci95(values)
            out[f"{metric}_mean"] = round(avg, 6)
            out[f"{metric}_ci95"] = round(ci, 6)
        output.append(out)
    return output


def _to_float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ci95(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
