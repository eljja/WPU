from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F


STATE_FEATURES = [
    "causal_k",
    "interaction_density",
    "min_pair_distance",
    "mean_pair_distance",
    "target_x",
    "target_y",
    "event_norm",
]

SPARSE_DIAGNOSTIC_FEATURES = STATE_FEATURES + [
    "sparse_entropy",
    "sparse_margin",
    "sparse_confidence",
    "sparse_delta_norm",
    "sparse_uncertainty_mean",
]

STATE_SELECTOR_FEATURES = STATE_FEATURES + [
    "selector_confidence",
    "selected_fraction",
]

STATE_REGRET_SCALAR_FEATURES = STATE_FEATURES + [
    "predicted_regret",
    "regret_abs",
]

ROUTE_CONTEXT_FEATURES = SPARSE_DIAGNOSTIC_FEATURES + [
    "selector_confidence",
    "selected_fraction",
    "predicted_regret",
    "regret_abs",
    "sparse_dense_disagreement",
]

SUMMARY_METRICS = [
    "target_mean",
    "prediction_mean",
    "regret_mse",
    "regret_mae",
    "regret_r2",
    "regret_pearson",
    "dense_rate",
    "policy_loss",
    "sparse_loss",
    "dense_loss",
    "oracle_loss",
    "policy_delta_vs_sparse",
    "policy_excess_over_oracle",
    "policy_accuracy",
    "sparse_accuracy",
    "dense_accuracy",
    "oracle_accuracy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe continuous dense-vs-sparse route regret.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/route_regret_probe.csv"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=248366)
    parser.add_argument("--compute-costs", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = _read_rows(args.input)
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, object]] = []
    for test_seed in seeds:
        train_rows = [row for row in rows if int(row["seed"]) != test_seed]
        test_rows = [row for row in rows if int(row["seed"]) == test_seed]
        if not train_rows or not test_rows:
            continue
        output.extend(_run_split(train_rows, test_rows, test_seed, args, STATE_FEATURES, "mlp_state_regret"))
        if rows and all(feature in rows[0] for feature in SPARSE_DIAGNOSTIC_FEATURES):
            output.extend(
                _run_split(
                    train_rows,
                    test_rows,
                    test_seed,
                    args,
                    SPARSE_DIAGNOSTIC_FEATURES,
                    "mlp_sparse_diagnostics_regret",
                )
            )
        if rows and all(feature in rows[0] for feature in STATE_SELECTOR_FEATURES):
            output.extend(
                _run_split(
                    train_rows,
                    test_rows,
                    test_seed,
                    args,
                    STATE_SELECTOR_FEATURES,
                    "mlp_state_selector_regret",
                )
            )
        if rows and all(feature in rows[0] for feature in STATE_REGRET_SCALAR_FEATURES):
            output.extend(
                _run_split(
                    train_rows,
                    test_rows,
                    test_seed,
                    args,
                    STATE_REGRET_SCALAR_FEATURES,
                    "mlp_state_regret_scalar",
                )
            )
        if rows and all(feature in rows[0] for feature in ROUTE_CONTEXT_FEATURES):
            output.extend(
                _run_split(
                    train_rows,
                    test_rows,
                    test_seed,
                    args,
                    ROUTE_CONTEXT_FEATURES,
                    "mlp_route_context_regret",
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, output)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.summary_out, _summarize(output))
    print(f"wrote={args.out}", flush=True)
    if args.summary_out is not None:
        print(f"summary={args.summary_out}", flush=True)


def _run_split(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    test_seed: int,
    args: argparse.Namespace,
    features: list[str],
    model_name: str,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    train_x, train_y = _tensorize_regret(train_rows, features, device)
    test_x, test_y = _tensorize_regret(test_rows, features, device)
    mean_x = train_x.mean(dim=0, keepdim=True)
    std_x = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean_x) / std_x
    test_x = (test_x - mean_x) / std_x

    model = nn.Sequential(
        nn.Linear(train_x.size(1), args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        prediction = model(train_x).squeeze(-1)
        loss = F.mse_loss(prediction, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(test_x).squeeze(-1)
    test_payload = _tensorize_policy_payload(test_rows, device)
    rows = []
    for compute_cost in args.compute_costs:
        rows.append(_metrics(test_seed, model_name, len(features), compute_cost, test_y, predictions, test_payload))
    return rows


def _metrics(
    test_seed: int,
    model_name: str,
    feature_count: int,
    compute_cost: float,
    regret: torch.Tensor,
    predicted_regret: torch.Tensor,
    payload: dict[str, torch.Tensor],
) -> dict[str, object]:
    route_dense = predicted_regret + compute_cost < 0.0
    sparse_loss = payload["sparse_loss"]
    dense_loss = payload["dense_loss"] + compute_cost
    policy_loss = torch.where(route_dense, dense_loss, sparse_loss)
    oracle_dense = dense_loss < sparse_loss
    oracle_loss = torch.minimum(sparse_loss, dense_loss)

    sparse_correct = payload["sparse_correct"].bool()
    dense_correct = payload["dense_correct"].bool()
    policy_correct = torch.where(route_dense, dense_correct, sparse_correct)
    oracle_correct = sparse_correct | dense_correct

    return {
        "model": model_name,
        "test_seed": test_seed,
        "feature_count": feature_count,
        "compute_cost": compute_cost,
        "samples": int(regret.numel()),
        "target_mean": _round(regret.mean()),
        "prediction_mean": _round(predicted_regret.mean()),
        "regret_mse": _round((predicted_regret - regret).square().mean()),
        "regret_mae": _round((predicted_regret - regret).abs().mean()),
        "regret_r2": _round(_r2(regret, predicted_regret)),
        "regret_pearson": _round(_pearson(regret, predicted_regret)),
        "dense_rate": _round(route_dense.float().mean()),
        "oracle_dense_rate": _round(oracle_dense.float().mean()),
        "policy_loss": _round(policy_loss.mean()),
        "sparse_loss": _round(sparse_loss.mean()),
        "dense_loss": _round(dense_loss.mean()),
        "oracle_loss": _round(oracle_loss.mean()),
        "policy_delta_vs_sparse": _round((policy_loss - sparse_loss).mean()),
        "policy_excess_over_oracle": _round((policy_loss - oracle_loss).mean()),
        "policy_accuracy": _round(policy_correct.float().mean()),
        "sparse_accuracy": _round(sparse_correct.float().mean()),
        "dense_accuracy": _round(dense_correct.float().mean()),
        "oracle_accuracy": _round(oracle_correct.float().mean()),
    }


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    denominator = (target - target.mean()).square().sum().clamp_min(1e-8)
    numerator = (target - prediction).square().sum()
    return 1.0 - numerator / denominator


def _pearson(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (target_centered.square().sum() * prediction_centered.square().sum()).sqrt().clamp_min(1e-8)
    return (target_centered * prediction_centered).sum() / denominator


def _tensorize_regret(rows: list[dict[str, str]], feature_names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = []
    regrets = []
    for row in rows:
        values = []
        for feature in feature_names:
            if feature == "causal_k":
                values.append(float(row[feature]) / 32.0)
            else:
                values.append(float(row[feature]))
        feature_rows.append(values)
        regrets.append(float(row["dense_regret"]))
    return torch.tensor(feature_rows, dtype=torch.float32, device=device), torch.tensor(regrets, dtype=torch.float32, device=device)


def _tensorize_policy_payload(rows: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "sparse_loss": torch.tensor([float(row["sparse_loss"]) for row in rows], dtype=torch.float32, device=device),
        "dense_loss": torch.tensor([float(row["dense_loss"]) for row in rows], dtype=torch.float32, device=device),
        "sparse_correct": torch.tensor([float(row["sparse_correct"]) for row in rows], dtype=torch.float32, device=device),
        "dense_correct": torch.tensor([float(row["dense_correct"]) for row in rows], dtype=torch.float32, device=device),
    }


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), float(row["compute_cost"]))].append(row)
    summaries = []
    for (model_name, compute_cost), group_rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "model": model_name,
            "compute_cost": compute_cost,
            "seeds": len({row["test_seed"] for row in group_rows}),
            "samples": sum(int(row["samples"]) for row in group_rows),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = round(mean(float(row[metric]) for row in group_rows), 6)
        summaries.append(summary)
    return summaries


def _round(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return round(float(value), 6)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
