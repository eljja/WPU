from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.staged_regret_context_export import _route_context, _round  # noqa: E402
from scripts.staged_regret_hybrid import (  # noqa: E402
    _class_weights,
    _move_batch,
    _train_propagation,
    _train_regret_head,
)
from scripts.structured_verifier_probe import (  # noqa: E402
    FEATURES,
    PHYSICAL_FEATURES,
    _candidate_conditions,
    _choose_regret_threshold,
    _format_rule,
    _matches,
)
from wpu.data.working_set_physics import (  # noqa: E402
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
)
from wpu.models.factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deployed verifier-triggered K expansion.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--initial-working-set-size", type=int, default=8)
    parser.add_argument("--expanded-working-set-size", type=int, default=32)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--regret-steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=3e-3)
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--expansion-cost", type=float, default=0.02)
    parser.add_argument("--max-expansion-rate", type=float, default=0.5)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts/staged_k_expansion_hybrid.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"staged-k-expansion seed={seed} N={n_value} K={k_value}", flush=True)
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
        working_set_size=args.expanded_working_set_size,
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
    train_args = argparse.Namespace(**vars(args), working_set_size=args.expanded_working_set_size)
    _train_propagation(model, train_dataset, class_weights, train_args, device)
    _train_regret_head(model, train_dataset, train_args, device)
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
    route_threshold = _choose_regret_threshold(_as_regret_rows(validation_rows), args.compute_cost)
    structured_rule = _choose_expansion_rule(
        validation_rows,
        route_threshold,
        args.compute_cost,
        args.expansion_cost,
        args.max_expansion_rate,
        args.quantiles,
        FEATURES,
    )
    physical_rule = _choose_expansion_rule(
        validation_rows,
        route_threshold,
        args.compute_cost,
        args.expansion_cost,
        args.max_expansion_rate,
        args.quantiles,
        PHYSICAL_FEATURES,
    )
    output = [
        _policy_row("initial_calibrated_regret", test_rows, args.compute_cost, args.expansion_cost, route_threshold, None),
        _policy_row(
            "always_expand_sparse",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            ("__always__",),
            expansion_path="sparse",
        ),
        _policy_row(
            "always_expand_dense",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            ("__always__",),
            expansion_path="dense",
        ),
        _policy_row(
            "structured_expansion_gate",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            structured_rule,
            expansion_path="sparse",
        ),
        _policy_row(
            "physical_expansion_gate",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            physical_rule,
            expansion_path="sparse",
        ),
        _policy_row(
            "structured_dense_expansion_gate",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            structured_rule,
            expansion_path="dense",
        ),
        _policy_row(
            "physical_dense_expansion_gate",
            test_rows,
            args.compute_cost,
            args.expansion_cost,
            route_threshold,
            physical_rule,
            expansion_path="dense",
        ),
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
                "initial_working_set_size": args.initial_working_set_size,
                "expanded_working_set_size": args.expanded_working_set_size,
                "propagation_steps": args.propagation_steps,
                "regret_steps": args.regret_steps,
                "validation_samples": len(validation_rows),
                "max_expansion_rate": args.max_expansion_rate,
            }
        )
    return output


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
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_dual_collate(args))
    rows: list[dict[str, str]] = []
    model.eval()
    with torch.no_grad():
        for initial_batch, expanded_batch, labels in loader:
            initial_batch = _move_batch(initial_batch, device)
            expanded_batch = _move_batch(expanded_batch, device)
            labels = labels.to(device)
            initial_sparse = model(initial_batch, num_branches=3, force_route="sparse")
            context = _route_context(model, initial_batch, initial_sparse)
            initial_dense = model(initial_batch, num_branches=3, force_route="local_dense")
            initial_regret = model.route_regret_prediction()
            expanded_sparse = model(expanded_batch, num_branches=3, force_route="sparse")
            expanded_dense = model(expanded_batch, num_branches=3, force_route="local_dense")
            initial_sparse_loss = F.cross_entropy(initial_sparse.branch_logits, labels, reduction="none")
            initial_dense_loss = F.cross_entropy(initial_dense.branch_logits, labels, reduction="none")
            expanded_sparse_loss = F.cross_entropy(expanded_sparse.branch_logits, labels, reduction="none")
            expanded_dense_loss = F.cross_entropy(expanded_dense.branch_logits, labels, reduction="none")
            initial_sparse_correct = initial_sparse.branch_logits.argmax(dim=-1) == labels
            initial_dense_correct = initial_dense.branch_logits.argmax(dim=-1) == labels
            expanded_sparse_correct = expanded_sparse.branch_logits.argmax(dim=-1) == labels
            expanded_dense_correct = expanded_dense.branch_logits.argmax(dim=-1) == labels
            for index in range(labels.numel()):
                row = {
                    "predicted_regret": str(_round(initial_regret[index])),
                    "regret_abs": str(_round(initial_regret[index].abs())),
                    "sparse_loss": str(_round(initial_sparse_loss[index])),
                    "dense_loss": str(_round(initial_dense_loss[index])),
                    "expanded_sparse_loss": str(_round(expanded_sparse_loss[index])),
                    "expanded_dense_loss": str(_round(expanded_dense_loss[index])),
                    "sparse_correct": str(int(initial_sparse_correct[index].detach().cpu().item())),
                    "dense_correct": str(int(initial_dense_correct[index].detach().cpu().item())),
                    "expanded_sparse_correct": str(int(expanded_sparse_correct[index].detach().cpu().item())),
                    "expanded_dense_correct": str(int(expanded_dense_correct[index].detach().cpu().item())),
                    "sparse_dense_disagreement": str(
                        int(
                            (
                                initial_sparse.branch_logits[index].argmax()
                                != initial_dense.branch_logits[index].argmax()
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


def _choose_expansion_rule(
    rows: list[dict[str, str]],
    route_threshold: float,
    compute_cost: float,
    expansion_cost: float,
    max_expansion_rate: float,
    quantiles: list[float],
    features: list[str],
) -> tuple[tuple[str, str, float], ...] | None:
    base_loss = _mean_expansion_policy_loss(rows, route_threshold, compute_cost, expansion_cost, None)
    best_loss = base_loss
    best_rule: tuple[tuple[str, str, float], ...] | None = None
    conditions = _candidate_conditions(rows, quantiles, features)
    for condition in conditions:
        rule = (condition,)
        if _trigger_rate(rows, rule) > max_expansion_rate:
            continue
        loss = _mean_expansion_policy_loss(rows, route_threshold, compute_cost, expansion_cost, rule)
        if loss < best_loss:
            best_loss = loss
            best_rule = rule
    single_best = list(best_rule or ())
    for first in single_best:
        for second in conditions:
            if second == first:
                continue
            rule = (first, second)
            if _trigger_rate(rows, rule) > max_expansion_rate:
                continue
            loss = _mean_expansion_policy_loss(rows, route_threshold, compute_cost, expansion_cost, rule)
            if loss < best_loss:
                best_loss = loss
                best_rule = rule
    return best_rule


def _policy_row(
    policy: str,
    rows: list[dict[str, str]],
    compute_cost: float,
    expansion_cost: float,
    route_threshold: float,
    expansion_rule: tuple[tuple[str, str, float], ...] | tuple[str] | None,
    *,
    expansion_path: str = "sparse",
) -> dict[str, object]:
    losses = []
    sparse_losses = []
    initial_dense_losses = []
    expanded_losses = []
    expanded_sparse_losses = []
    expanded_dense_losses = []
    oracle_losses = []
    initial_dense_routes = []
    expansion_routes = []
    correct_values = []
    for row in rows:
        sparse_loss = float(row["sparse_loss"])
        initial_dense_loss = float(row["dense_loss"]) + compute_cost
        if expansion_path == "sparse":
            expanded_loss = float(row["expanded_sparse_loss"]) + expansion_cost
            expanded_correct = bool(int(row["expanded_sparse_correct"]))
        elif expansion_path == "dense":
            expanded_loss = float(row["expanded_dense_loss"]) + compute_cost + expansion_cost
            expanded_correct = bool(int(row["expanded_dense_correct"]))
        else:
            raise ValueError(f"unknown expansion path: {expansion_path}")
        use_initial_dense = float(row["predicted_regret"]) < route_threshold
        use_expansion = _expansion_matches(row, expansion_rule)
        loss = expanded_loss if use_expansion else initial_dense_loss if use_initial_dense else sparse_loss
        sparse_correct = bool(int(row["sparse_correct"]))
        initial_dense_correct = bool(int(row["dense_correct"]))
        correct = expanded_correct if use_expansion else initial_dense_correct if use_initial_dense else sparse_correct
        losses.append(loss)
        sparse_losses.append(sparse_loss)
        initial_dense_losses.append(initial_dense_loss)
        expanded_losses.append(expanded_loss)
        expanded_sparse_losses.append(float(row["expanded_sparse_loss"]) + expansion_cost)
        expanded_dense_losses.append(float(row["expanded_dense_loss"]) + compute_cost + expansion_cost)
        oracle_losses.append(min(sparse_loss, initial_dense_loss, expanded_loss))
        initial_dense_routes.append(use_initial_dense and not use_expansion)
        expansion_routes.append(use_expansion)
        correct_values.append(correct)
    return {
        "policy": policy,
        "rule": _format_expansion_rule(expansion_rule),
        "expansion_path": expansion_path,
        "compute_cost": compute_cost,
        "expansion_cost": expansion_cost,
        "route_threshold": round(route_threshold, 6),
        "samples": len(rows),
        "initial_dense_ratio": round(_mean_bool(initial_dense_routes), 6),
        "expansion_ratio": round(_mean_bool(expansion_routes), 6),
        "total_compute_ratio": round(_mean_bool(initial_dense_routes) + _mean_bool(expansion_routes), 6),
        "policy_accuracy": round(_mean_bool(correct_values), 6),
        "sparse_loss": round(mean(sparse_losses), 6),
        "initial_dense_loss": round(mean(initial_dense_losses), 6),
        "expanded_sparse_loss": round(mean(expanded_sparse_losses), 6),
        "expanded_dense_loss": round(mean(expanded_dense_losses), 6),
        "expanded_path_loss": round(mean(expanded_losses), 6),
        "oracle_loss": round(mean(oracle_losses), 6),
        "policy_loss": round(mean(losses), 6),
        "policy_delta_vs_sparse": round(mean(loss - sparse for loss, sparse in zip(losses, sparse_losses, strict=True)), 6),
        "policy_excess_over_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
    }


def _mean_expansion_policy_loss(
    rows: list[dict[str, str]],
    route_threshold: float,
    compute_cost: float,
    expansion_cost: float,
    rule: tuple[tuple[str, str, float], ...] | None,
) -> float:
    losses = []
    for row in rows:
        sparse_loss = float(row["sparse_loss"])
        initial_dense_loss = float(row["dense_loss"]) + compute_cost
        expanded_loss = float(row["expanded_sparse_loss"]) + expansion_cost
        use_initial_dense = float(row["predicted_regret"]) < route_threshold
        losses.append(expanded_loss if rule is not None and _matches(row, rule) else initial_dense_loss if use_initial_dense else sparse_loss)
    return mean(losses)


def _dual_collate(args: argparse.Namespace):
    def collate(samples):
        initial_batch, _, labels, _ = collate_indexed_working_set_samples(
            samples,
            max_nodes=args.initial_working_set_size,
            max_depth=args.index_depth,
        )
        expanded_batch, _, expanded_labels, _ = collate_indexed_working_set_samples(
            samples,
            max_nodes=args.expanded_working_set_size,
            max_depth=args.index_depth,
        )
        if not torch.equal(labels, expanded_labels):
            raise RuntimeError("initial and expanded labels diverged")
        return initial_batch, expanded_batch, labels

    return collate


def _as_regret_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **row,
            "dense_regret": str(float(row["dense_loss"]) - float(row["sparse_loss"])),
        }
        for row in rows
    ]


def _expansion_matches(row: dict[str, str], rule: tuple[tuple[str, str, float], ...] | tuple[str] | None) -> bool:
    if rule is None:
        return False
    if rule == ("__always__",):
        return True
    return _matches(row, rule)  # type: ignore[arg-type]


def _format_expansion_rule(rule: tuple[tuple[str, str, float], ...] | tuple[str] | None) -> str:
    if rule == ("__always__",):
        return "always"
    return _format_rule(rule)  # type: ignore[arg-type]


def _trigger_rate(rows: list[dict[str, str]], rule: tuple[tuple[str, str, float], ...]) -> float:
    return mean(float(_matches(row, rule)) for row in rows)


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
