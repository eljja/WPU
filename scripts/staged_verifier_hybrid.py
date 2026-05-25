from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.staged_regret_context_export import _route_context, _round  # noqa: E402
from scripts.staged_regret_hybrid import (  # noqa: E402
    _class_weights,
    _collate_fn,
    _move_batch,
    _train_propagation,
    _train_regret_head,
)
from scripts.structured_verifier_probe import (  # noqa: E402
    PHYSICAL_FEATURES,
    _choose_gate_rule,
    _choose_regret_threshold,
    _format_rule,
    _matches,
    _mean_policy_loss,
)
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train staged WPU regret routing with validation-calibrated verifier gates.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=32)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--regret-steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=3e-3)
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--min-gate-gain", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_verifier_hybrid.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"staged-verifier seed={seed} N={n_value} K={k_value}", flush=True)
                rows.extend(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = create_model(
        args.model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.working_set_size,
    ).to(device)
    train_dataset = WorkingSetPhysicsDataset(
        size=max((args.propagation_steps + args.regret_steps) * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    _train_propagation(model, train_dataset, class_weights, args, device)
    _train_regret_head(model, train_dataset, args, device)
    validation_rows = _collect_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 5_000,
        args.validation_samples,
        args,
        device,
    )
    test_rows = _collect_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 10_000,
        args.samples,
        args,
        device,
    )
    route_threshold = _choose_regret_threshold(validation_rows, args.compute_cost)
    structured_rule, structured_gain = _choose_safe_gate_rule(
        validation_rows,
        route_threshold,
        args.compute_cost,
        args.quantiles,
        args.min_gate_gain,
    )
    physical_rule, physical_gain = _choose_safe_gate_rule(
        validation_rows,
        route_threshold,
        args.compute_cost,
        args.quantiles,
        args.min_gate_gain,
        features=PHYSICAL_FEATURES,
    )
    output = [
        _policy_row("calibrated_regret_route", test_rows, args.compute_cost, route_threshold, None, 0.0),
        _policy_row("structured_verifier_gate", test_rows, args.compute_cost, route_threshold, structured_rule, structured_gain),
        _policy_row("physical_verifier_gate", test_rows, args.compute_cost, route_threshold, physical_rule, physical_gain),
    ]
    for row in output:
        row.update(
            {
                "status": "ok",
                "model": args.model_name,
                "seed": seed,
                "total_objects_n": background_objects + 4 + causal_obstacles,
                "causal_k": 4 + causal_obstacles,
                "interaction_mode": args.interaction_mode,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "working_set_size": args.working_set_size,
                "propagation_steps": args.propagation_steps,
                "regret_steps": args.regret_steps,
                "validation_samples": len(validation_rows),
                "min_gate_gain": args.min_gate_gain,
            }
        )
    return output


def _choose_safe_gate_rule(
    rows: list[dict[str, str]],
    route_threshold: float,
    compute_cost: float,
    quantiles: list[float],
    min_gain: float,
    *,
    features: list[str] | None = None,
) -> tuple[tuple[tuple[str, str, float], ...] | None, float]:
    base_loss = _mean_policy_loss(rows, compute_cost, route_threshold)
    rule = (
        _choose_gate_rule(rows, route_threshold, compute_cost, quantiles)
        if features is None
        else _choose_gate_rule(rows, route_threshold, compute_cost, quantiles, features=features)
    )
    if rule is None:
        return None, 0.0
    rule_loss = _mean_policy_loss(rows, compute_cost, route_threshold, gate_rule=rule)
    gain = base_loss - rule_loss
    if gain < min_gain:
        return None, round(gain, 6)
    return rule, round(gain, 6)


def _collect_rows(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    dataset_seed: int,
    samples: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, str]]:
    dataset = WorkingSetPhysicsDataset(
        size=samples,
        seed=dataset_seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args))
    rows: list[dict[str, str]] = []
    model.eval()
    with torch.no_grad():
        for batch, _, labels, _ in loader:
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            sparse_prediction = model(batch, num_branches=3, force_route="sparse")
            context = _route_context(model, batch, sparse_prediction)
            dense_prediction = model(batch, num_branches=3, force_route="local_dense")
            sparse_loss = F.cross_entropy(sparse_prediction.branch_logits, labels, reduction="none")
            dense_loss = F.cross_entropy(dense_prediction.branch_logits, labels, reduction="none")
            sparse_correct = sparse_prediction.branch_logits.argmax(dim=-1) == labels
            dense_correct = dense_prediction.branch_logits.argmax(dim=-1) == labels
            predicted_regret = model.route_regret_prediction()
            for index in range(labels.numel()):
                row = {
                    "predicted_regret": str(_round(predicted_regret[index])),
                    "regret_abs": str(_round(predicted_regret[index].abs())),
                    "sparse_loss": str(_round(sparse_loss[index])),
                    "dense_loss": str(_round(dense_loss[index])),
                    "sparse_correct": str(int(sparse_correct[index].detach().cpu().item())),
                    "dense_correct": str(int(dense_correct[index].detach().cpu().item())),
                    "sparse_dense_disagreement": str(
                        int(
                            (
                                sparse_prediction.branch_logits[index].argmax()
                                != dense_prediction.branch_logits[index].argmax()
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                    ),
                    "causal_k": str(4 + causal_obstacles),
                    "total_objects_n": str(background_objects + 4 + causal_obstacles),
                }
                row.update({key: str(_round(value[index])) for key, value in context.items()})
                rows.append(row)
    return rows


def _policy_row(
    policy: str,
    rows: list[dict[str, str]],
    compute_cost: float,
    route_threshold: float,
    rule: tuple[tuple[str, str, float], ...] | None,
    validation_gate_gain: float,
) -> dict[str, object]:
    losses = []
    sparse_losses = []
    dense_losses = []
    oracle_losses = []
    dense_routes = []
    trigger_values = []
    correct_values = []
    for row in rows:
        sparse_loss = float(row["sparse_loss"])
        dense_loss = float(row["dense_loss"]) + compute_cost
        use_dense = float(row["predicted_regret"]) < route_threshold
        if rule is not None and not _matches(row, rule):
            use_dense = False
        sparse_correct = bool(int(row["sparse_correct"]))
        dense_correct = bool(int(row["dense_correct"]))
        losses.append(dense_loss if use_dense else sparse_loss)
        sparse_losses.append(sparse_loss)
        dense_losses.append(dense_loss)
        oracle_losses.append(min(sparse_loss, dense_loss))
        dense_routes.append(use_dense)
        trigger_values.append(rule is not None and _matches(row, rule))
        correct_values.append(dense_correct if use_dense else sparse_correct)
    return {
        "policy": policy,
        "rule": _format_rule(rule),
        "compute_cost": compute_cost,
        "route_threshold": round(route_threshold, 6),
        "samples": len(rows),
        "validation_gate_gain": validation_gate_gain,
        "dense_compute_ratio": round(_mean_bool(dense_routes), 6),
        "verifier_trigger_rate": round(_mean_bool(trigger_values), 6),
        "policy_accuracy": round(_mean_bool(correct_values), 6),
        "sparse_loss": round(sum(sparse_losses) / max(len(sparse_losses), 1), 6),
        "dense_loss": round(sum(dense_losses) / max(len(dense_losses), 1), 6),
        "policy_loss": round(sum(losses) / max(len(losses), 1), 6),
        "oracle_loss": round(sum(oracle_losses) / max(len(oracle_losses), 1), 6),
        "policy_delta_vs_sparse": round(
            sum(loss - sparse for loss, sparse in zip(losses, sparse_losses, strict=True)) / max(len(losses), 1),
            6,
        ),
        "policy_excess_over_oracle": round(
            sum(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)) / max(len(losses), 1),
            6,
        ),
    }


def _mean_bool(values: list[bool]) -> float:
    return sum(float(value) for value in values) / max(len(values), 1)


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
