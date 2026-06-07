from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.models.factory import create_model

from scripts.pybullet_shift_generalization import (
    DEFAULT_MODELS,
    MECHANISMS,
    _class_weights,
    _collate_fn,
    _dataset,
    _default_calibrator,
    _evaluate,
    _move_batch,
    _summary,
    _train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate few-shot mechanism adaptation on PyBullet shift mechanisms."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--train-mechanisms", nargs="+", default=["nominal"])
    parser.add_argument("--eval-mechanisms", nargs="+", default=["nominal", "high_force", "edge_shift", "catch_heavy"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--adaptation-steps", type=int, default=8)
    parser.add_argument("--adaptation-samples", type=int, default=36)
    parser.add_argument("--adaptation-lr", type=float, default=5e-4)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_fewshot_mechanism_adaptation.csv"))
    args = parser.parse_args()

    unknown = [mechanism for mechanism in args.eval_mechanisms if mechanism not in MECHANISMS]
    unknown.extend(mechanism for mechanism in args.train_mechanisms if mechanism not in MECHANISMS)
    if unknown:
        raise ValueError(f"unknown mechanisms: {unknown}")

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"train {'+'.join(args.train_mechanisms)} model={model_name} seed={seed}", flush=True)
            base_model = _train_model(model_name, seed, args)
            for mechanism in args.eval_mechanisms:
                print(f"adapt/eval model={model_name} seed={seed} mechanism={mechanism}", flush=True)
                adapted_model = _adapt_model(_clone_model(base_model, model_name, args), model_name, seed, mechanism, args)
                row = _evaluate(adapted_model, model_name, seed, mechanism, _default_calibrator(), args)
                row["adaptation_policy"] = "fewshot_mechanism"
                row["adaptation_steps"] = args.adaptation_steps
                row["adaptation_samples"] = args.adaptation_samples
                row["adaptation_lr"] = args.adaptation_lr
                rows.append(row)
                _write_csv(args.out, rows)
    rows.extend(_summary(rows))
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _clone_model(base_model: torch.nn.Module, model_name: str, args: argparse.Namespace) -> torch.nn.Module:
    clone = create_model(
        model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(torch.device(args.device))
    clone.load_state_dict(base_model.state_dict())
    return clone


def _adapt_model(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    mechanism: str,
    args: argparse.Namespace,
) -> torch.nn.Module:
    torch.manual_seed(seed + 60_000)
    device = torch.device(args.device)
    dataset = _dataset(
        mechanism=mechanism,
        samples=args.adaptation_samples,
        seed=seed + 60_000,
        args=args,
        balanced_labels=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, model_name))
    class_weights = _class_weights(dataset).to(device) if args.class_weights else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.adaptation_lr)
    model.train()
    iterator = iter(loader)
    for _ in range(args.adaptation_steps):
        try:
            batch, target_delta, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, target_delta, labels, _ = next(iterator)
        batch = _move_batch(batch, device)
        target_delta = target_delta.to(device)
        labels = labels.to(device)
        prediction = model(batch, num_branches=3, route_branches=3)
        loss = F.cross_entropy(prediction.branch_logits, labels, weight=class_weights)
        loss = loss + 0.1 * F.mse_loss(prediction.object_delta, target_delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


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
