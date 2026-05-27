from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    _selected_ids,
    _selected_pair_density,
    _train_model as _train_retriever,
)
from scripts.staged_regret_hybrid import _class_weights, _move_batch, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_selected_working_set_samples  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


MODES = ("indexed", "proximity", "interaction", "learned")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether downstream branch loss favors a different state retriever than the current teacher."
    )
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_regret_oracle.csv"))
    parser.add_argument("--sample-out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_regret_oracle_samples.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"retriever-regret-oracle seed={seed} N={n_value} K={k_value}", flush=True)
                condition_summary, condition_samples = _run_condition(background_objects, causal_obstacles, seed, args)
                summary_rows.extend(condition_summary)
                sample_rows.extend(condition_samples)
                _write_csv(args.out, summary_rows)
                _write_csv(args.sample_out, sample_rows)
    _write_csv(args.out, summary_rows)
    _write_csv(args.sample_out, sample_rows)
    print(f"wrote={args.out}", flush=True)
    print(f"wrote_samples={args.sample_out}", flush=True)


def _run_condition(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    train_dataset = WorkingSetPhysicsDataset(
        size=max(args.propagation_steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    retriever = _train_retriever(
        [train_dataset[index] for index in range(len(train_dataset))],
        args.budget,
        args.retriever_steps,
        args.retriever_hidden_dim,
        args.retriever_lr,
    )
    model = create_model(
        args.model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.budget,
    ).to(device)
    train_args = argparse.Namespace(**vars(args), working_set_size=args.budget, selection_mode="interaction")
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    _train_propagation(model, train_dataset, class_weights, train_args, device)
    validation_rows = _collect_mode_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 5_000,
        args.validation_samples,
        args,
        retriever,
        device,
        split="validation",
    )
    test_rows = _collect_mode_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 10_000,
        args.samples,
        args,
        retriever,
        device,
        split="test",
    )
    static_mode = _best_static_mode(validation_rows)
    summary = _summarize_rows(test_rows, static_mode)
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
    for row in summary:
        row.update(
            {
                "seed": seed,
                "total_objects_n": total_n,
                "causal_k": causal_k,
                "budget": args.budget,
                "interaction_mode": args.interaction_mode,
                "propagation_steps": args.propagation_steps,
                "retriever_steps": args.retriever_steps,
                "static_mode_from_validation": static_mode,
            }
        )
    for row in test_rows:
        row.update(
            {
                "seed": seed,
                "total_objects_n": total_n,
                "causal_k": causal_k,
                "budget": args.budget,
                "static_mode_from_validation": static_mode,
            }
        )
    return summary, test_rows


def _collect_mode_rows(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    dataset_seed: int,
    sample_count: int,
    args: argparse.Namespace,
    retriever: torch.nn.Module,
    device: torch.device,
    *,
    split: str,
) -> list[dict[str, object]]:
    dataset = WorkingSetPhysicsDataset(
        size=sample_count,
        seed=dataset_seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    labels = [sample.branch_label for sample in samples]
    selected_by_mode = {
        mode: [_selected_ids(sample, mode, args.budget, retriever if mode == "learned" else None) for sample in samples]
        for mode in MODES
    }
    losses_by_mode: dict[str, list[float]] = {}
    correct_by_mode: dict[str, list[int]] = {}
    for mode in MODES:
        losses, correct = _evaluate_selected(model, samples, selected_by_mode[mode], args.batch_size, device)
        losses_by_mode[mode] = losses
        correct_by_mode[mode] = correct

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        mode_losses = {mode: losses_by_mode[mode][index] for mode in MODES}
        best_mode = min(MODES, key=lambda mode: (mode_losses[mode], mode))
        row: dict[str, object] = {
            "split": split,
            "sample_index": index,
            "label": labels[index],
            "best_mode": best_mode,
            "oracle_loss": round(mode_losses[best_mode], 6),
            "oracle_correct": correct_by_mode[best_mode][index],
            "teacher_is_oracle": int(best_mode == "interaction"),
            "learned_is_oracle": int(best_mode == "learned"),
        }
        for mode in MODES:
            selected_ids = selected_by_mode[mode][index]
            row[f"{mode}_loss"] = round(mode_losses[mode], 6)
            row[f"{mode}_correct"] = correct_by_mode[mode][index]
            row[f"{mode}_selected_hand"] = int("hand_001" in selected_ids)
            row[f"{mode}_selected_obstacles"] = sum(object_id.startswith("obstacle_") for object_id in selected_ids)
            row[f"{mode}_pair_density"] = round(
                _selected_pair_density(
                    sample.state,
                    [object_id for object_id in selected_ids if object_id.startswith("obstacle_")],
                ),
                6,
            )
        rows.append(row)
    return rows


def _evaluate_selected(
    model: torch.nn.Module,
    samples,
    selected_ids_by_sample: list[list[str]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    losses: list[float] = []
    correct: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_selected = selected_ids_by_sample[start : start + batch_size]
            batch, _, labels, _ = collate_selected_working_set_samples(batch_samples, batch_selected)
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, force_route="sparse")
            batch_losses = F.cross_entropy(prediction.branch_logits, labels, reduction="none")
            batch_correct = prediction.branch_logits.argmax(dim=-1) == labels
            losses.extend(round(float(value), 8) for value in batch_losses.detach().cpu())
            correct.extend(int(value) for value in batch_correct.detach().cpu())
    return losses, correct


def _best_static_mode(rows: list[dict[str, object]]) -> str:
    losses = {mode: mean(float(row[f"{mode}_loss"]) for row in rows) for mode in MODES}
    return min(MODES, key=lambda mode: (losses[mode], mode))


def _summarize_rows(rows: list[dict[str, object]], static_mode: str) -> list[dict[str, object]]:
    summary = []
    oracle_loss = mean(float(row["oracle_loss"]) for row in rows)
    oracle_accuracy = mean(float(row["oracle_correct"]) for row in rows)
    static_loss = mean(float(row[f"{static_mode}_loss"]) for row in rows)
    static_accuracy = mean(float(row[f"{static_mode}_correct"]) for row in rows)
    best_counts = Counter(str(row["best_mode"]) for row in rows)
    for mode in MODES:
        mode_loss = mean(float(row[f"{mode}_loss"]) for row in rows)
        mode_accuracy = mean(float(row[f"{mode}_correct"]) for row in rows)
        summary.append(
            {
                "policy": f"static_{mode}",
                "mode": mode,
                "samples": len(rows),
                "loss": round(mode_loss, 6),
                "accuracy": round(mode_accuracy, 6),
                "delta_vs_static_validation_choice": round(mode_loss - static_loss, 6),
                "excess_over_oracle": round(mode_loss - oracle_loss, 6),
                "oracle_mode_rate": round(best_counts.get(mode, 0) / max(len(rows), 1), 6),
                "static_validation_loss": round(static_loss, 6),
                "static_validation_accuracy": round(static_accuracy, 6),
                "oracle_loss": round(oracle_loss, 6),
                "oracle_accuracy": round(oracle_accuracy, 6),
            }
        )
    summary.append(
        {
            "policy": "oracle_over_retrievers",
            "mode": "oracle",
            "samples": len(rows),
            "loss": round(oracle_loss, 6),
            "accuracy": round(oracle_accuracy, 6),
            "delta_vs_static_validation_choice": round(oracle_loss - static_loss, 6),
            "excess_over_oracle": 0.0,
            "oracle_mode_rate": 1.0,
            "static_validation_loss": round(static_loss, 6),
            "static_validation_accuracy": round(static_accuracy, 6),
            "oracle_loss": round(oracle_loss, 6),
            "oracle_accuracy": round(oracle_accuracy, 6),
        }
    )
    return summary


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
