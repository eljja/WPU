from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.route_regret_probe import STATE_FEATURES, SUMMARY_METRICS, _read_rows, _round, _tensorize_regret, _write_csv  # noqa: E402


GATE_FEATURES = [
    "selector_confidence",
    "sparse_entropy",
    "sparse_margin",
    "sparse_confidence",
    "sparse_uncertainty_mean",
    "sparse_delta_norm",
    "regret_abs",
    "interaction_density",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diagnostics as safety gates over state-based WPU regret routing.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/diagnostic_safety_gate_probe.csv"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--compute-costs", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = _read_rows(args.input)
    missing = [feature for feature in [*STATE_FEATURES, *GATE_FEATURES] if feature not in rows[0]]
    if missing:
        raise ValueError(f"missing required features: {missing}")
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, object]] = []
    for test_seed in seeds:
        train_rows = [row for row in rows if int(row["seed"]) != test_seed]
        test_rows = [row for row in rows if int(row["seed"]) == test_seed]
        output.extend(_run_split(train_rows, test_rows, test_seed, args))
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
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    train_x, train_y = _normalized_state_tensors(train_rows, device)
    test_x, test_y = _normalized_state_tensors(test_rows, device, reference_rows=train_rows)
    model = _fit_regressor(train_x, train_y, args.hidden_dim, args.steps, args.lr)
    with torch.no_grad():
        train_prediction = model(train_x).squeeze(-1)
        test_prediction = model(test_x).squeeze(-1)
    train_payload = _payload(train_rows, device)
    test_payload = _payload(test_rows, device)
    output: list[dict[str, object]] = []
    for compute_cost in args.compute_costs:
        output.append(
            _policy_row(
                test_seed,
                "state_route_no_gate",
                compute_cost,
                test_y,
                test_prediction,
                test_payload,
                gate=None,
                selected_by="none",
            )
        )
        train_gate = _choose_gate(train_rows, train_prediction, train_payload, compute_cost, device)
        output.append(
            _policy_row(
                test_seed,
                "loso_diagnostic_gate",
                compute_cost,
                test_y,
                test_prediction,
                test_payload,
                gate=_gate_mask(test_rows, train_gate, device),
                selected_by=_gate_description(train_gate),
            )
        )
        oracle_gate = _choose_gate(test_rows, test_prediction, test_payload, compute_cost, device)
        output.append(
            _policy_row(
                test_seed,
                "test_oracle_diagnostic_gate",
                compute_cost,
                test_y,
                test_prediction,
                test_payload,
                gate=_gate_mask(test_rows, oracle_gate, device),
                selected_by=_gate_description(oracle_gate),
            )
        )
    return output


def _choose_gate(
    rows: list[dict[str, str]],
    predicted_regret: torch.Tensor,
    payload: dict[str, torch.Tensor],
    compute_cost: float,
    device: torch.device,
) -> tuple[str, str, float] | None:
    base_route = predicted_regret + compute_cost < 0.0
    sparse_loss = payload["sparse_loss"]
    dense_loss = payload["dense_loss"] + compute_cost
    best_loss = torch.where(base_route, dense_loss, sparse_loss).mean().item()
    best_dense_rate = base_route.float().mean().item()
    best_gate: tuple[str, str, float] | None = None
    for feature in GATE_FEATURES:
        values = torch.tensor([float(row[feature]) for row in rows], dtype=torch.float32, device=device)
        thresholds = torch.quantile(values, torch.linspace(0.1, 0.9, 9, device=device)).detach().cpu().tolist()
        for threshold in thresholds:
            for direction in ("le", "ge"):
                gate = values <= threshold if direction == "le" else values >= threshold
                route_dense = base_route & gate
                policy_loss = torch.where(route_dense, dense_loss, sparse_loss).mean().item()
                dense_rate = route_dense.float().mean().item()
                if (policy_loss, dense_rate) < (best_loss, best_dense_rate):
                    best_loss = policy_loss
                    best_dense_rate = dense_rate
                    best_gate = (feature, direction, float(threshold))
    return best_gate


def _policy_row(
    test_seed: int,
    policy: str,
    compute_cost: float,
    regret: torch.Tensor,
    predicted_regret: torch.Tensor,
    payload: dict[str, torch.Tensor],
    gate: torch.Tensor | None,
    selected_by: str,
) -> dict[str, object]:
    base_route = predicted_regret + compute_cost < 0.0
    route_dense = base_route if gate is None else base_route & gate
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
        "model": policy,
        "selected_by": selected_by,
        "test_seed": test_seed,
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


def _gate_mask(rows: list[dict[str, str]], gate: tuple[str, str, float] | None, device: torch.device) -> torch.Tensor | None:
    if gate is None:
        return None
    feature, direction, threshold = gate
    values = torch.tensor([float(row[feature]) for row in rows], dtype=torch.float32, device=device)
    return values <= threshold if direction == "le" else values >= threshold


def _gate_description(gate: tuple[str, str, float] | None) -> str:
    if gate is None:
        return "no_gate"
    feature, direction, threshold = gate
    return f"{feature}:{direction}:{threshold:.6f}"


def _normalized_state_tensors(
    rows: list[dict[str, str]],
    device: torch.device,
    reference_rows: list[dict[str, str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _tensorize_regret(rows, STATE_FEATURES, device)
    reference_x = x if reference_rows is None else _tensorize_regret(reference_rows, STATE_FEATURES, device)[0]
    mean_x = reference_x.mean(dim=0, keepdim=True)
    std_x = reference_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean_x) / std_x, y


def _payload(rows: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "sparse_loss": torch.tensor([float(row["sparse_loss"]) for row in rows], dtype=torch.float32, device=device),
        "dense_loss": torch.tensor([float(row["dense_loss"]) for row in rows], dtype=torch.float32, device=device),
        "sparse_correct": torch.tensor([float(row["sparse_correct"]) for row in rows], dtype=torch.float32, device=device),
        "dense_correct": torch.tensor([float(row["dense_correct"]) for row in rows], dtype=torch.float32, device=device),
    }


def _fit_regressor(x: torch.Tensor, y: torch.Tensor, hidden_dim: int, steps: int, lr: float) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(x.size(1), hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    ).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        prediction = model(x).squeeze(-1)
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


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
            "selected_by": ";".join(sorted({str(row["selected_by"]) for row in group_rows})),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = round(mean(float(row[metric]) for row in group_rows), 6)
        summaries.append(summary)
    return summaries


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    denominator = (target - target.mean()).square().sum().clamp_min(1e-8)
    numerator = (target - prediction).square().sum()
    return 1.0 - numerator / denominator


def _pearson(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (target_centered.square().sum() * prediction_centered.square().sum()).sqrt().clamp_min(1e-8)
    return (target_centered * prediction_centered).sum() / denominator


if __name__ == "__main__":
    main()
