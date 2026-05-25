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

from scripts.route_regret_probe import (  # noqa: E402
    ROUTE_CONTEXT_FEATURES,
    STATE_FEATURES,
    SUMMARY_METRICS,
    _metrics,
    _read_rows,
    _tensorize_policy_payload,
    _tensorize_regret,
    _write_csv,
)


DIAGNOSTIC_FEATURES = [feature for feature in ROUTE_CONTEXT_FEATURES if feature not in STATE_FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe state-grounded regret prediction with clipped diagnostic residuals.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/clipped_diagnostic_probe.csv"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=5318008)
    parser.add_argument("--residual-clips", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--compute-costs", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = _read_rows(args.input)
    missing = [feature for feature in [*STATE_FEATURES, *DIAGNOSTIC_FEATURES] if feature not in rows[0]]
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
    train_base_x, train_y = _normalized_tensors(train_rows, STATE_FEATURES, device)
    test_base_x, test_y = _normalized_tensors(test_rows, STATE_FEATURES, device, reference_rows=train_rows)
    base_model = _fit_regressor(train_base_x, train_y, args.hidden_dim, args.steps, args.lr)
    with torch.no_grad():
        train_base_prediction = base_model(train_base_x).squeeze(-1)
        test_base_prediction = base_model(test_base_x).squeeze(-1)

    train_diag_x, _ = _normalized_tensors(train_rows, DIAGNOSTIC_FEATURES, device)
    test_diag_x, _ = _normalized_tensors(test_rows, DIAGNOSTIC_FEATURES, device, reference_rows=train_rows)
    residual_model = _fit_regressor(
        train_diag_x,
        train_y - train_base_prediction.detach(),
        args.hidden_dim,
        args.steps,
        args.lr,
    )
    with torch.no_grad():
        residual_prediction = residual_model(test_diag_x).squeeze(-1)
    payload = _tensorize_policy_payload(test_rows, device)
    output: list[dict[str, object]] = []
    for residual_clip in args.residual_clips:
        clipped_residual = residual_prediction.clamp(min=-residual_clip, max=residual_clip)
        prediction = test_base_prediction + clipped_residual
        model_name = f"state_base_plus_clipped_diag_{residual_clip:g}"
        for compute_cost in args.compute_costs:
            row = _metrics(test_seed, model_name, len(STATE_FEATURES) + len(DIAGNOSTIC_FEATURES), compute_cost, test_y, prediction, payload)
            row["residual_clip"] = residual_clip
            output.append(row)
    return output


def _normalized_tensors(
    rows: list[dict[str, str]],
    features: list[str],
    device: torch.device,
    reference_rows: list[dict[str, str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _tensorize_regret(rows, features, device)
    reference_x = x if reference_rows is None else _tensorize_regret(reference_rows, features, device)[0]
    mean_x = reference_x.mean(dim=0, keepdim=True)
    std_x = reference_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean_x) / std_x, y


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
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), float(row["compute_cost"]), float(row["residual_clip"]))].append(row)
    summaries = []
    for (model_name, compute_cost, residual_clip), group_rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "model": model_name,
            "compute_cost": compute_cost,
            "residual_clip": residual_clip,
            "seeds": len({row["test_seed"] for row in group_rows}),
            "samples": sum(int(row["samples"]) for row in group_rows),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = round(mean(float(row[metric]) for row in group_rows), 6)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    main()
