from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F


FEATURES = [
    "causal_k",
    "interaction_density",
    "min_pair_distance",
    "mean_pair_distance",
    "target_x",
    "target_y",
    "event_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether dense-needed labels are identifiable from state features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/route_label_probe.csv"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, object]] = []
    for test_seed in seeds:
        train_rows = [row for row in rows if int(row["seed"]) != test_seed]
        test_rows = [row for row in rows if int(row["seed"]) == test_seed]
        output.extend(_run_split(train_rows, test_rows, test_seed, args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, output)
    print(f"wrote={args.out}", flush=True)


def _run_split(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    test_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    train_x, train_y = _tensorize(train_rows, device)
    test_x, test_y = _tensorize(test_rows, device)
    mean_x = train_x.mean(dim=0, keepdim=True)
    std_x = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean_x) / std_x
    test_x = (test_x - mean_x) / std_x

    model = nn.Sequential(
        nn.Linear(train_x.size(1), args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, 1),
    ).to(device)
    positive = train_y.sum().clamp_min(1.0)
    negative = (1.0 - train_y).sum().clamp_min(1.0)
    pos_weight = negative / positive
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        logits = model(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(test_x).squeeze(-1))
    outputs = []
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        predictions = probabilities >= threshold
        outputs.append(_metrics(test_seed, test_y, predictions, probabilities, threshold, "mlp"))
    heuristic_scores = _heuristic_scores(test_rows, device)
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
        predictions = heuristic_scores >= threshold
        outputs.append(_metrics(test_seed, test_y, predictions, heuristic_scores, threshold, "interaction_density"))
    return outputs


def _metrics(
    test_seed: int,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    model_name: str,
) -> dict[str, object]:
    labels_bool = labels.bool()
    true_positive = int((predictions & labels_bool).sum().item())
    false_positive = int((predictions & ~labels_bool).sum().item())
    true_negative = int((~predictions & ~labels_bool).sum().item())
    false_negative = int((~predictions & labels_bool).sum().item())
    total = max(int(labels.numel()), 1)
    positive_total = max(true_positive + false_negative, 1)
    negative_total = max(true_negative + false_positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / positive_total
    specificity = true_negative / negative_total
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "model": model_name,
        "test_seed": test_seed,
        "threshold": threshold,
        "samples": total,
        "label_rate": round(float(labels.mean().detach().cpu().item()), 6),
        "predicted_dense_rate": round(float(predictions.float().mean().detach().cpu().item()), 6),
        "accuracy": round((true_positive + true_negative) / total, 6),
        "balanced_accuracy": round((recall + specificity) / 2.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "score_mean": round(float(scores.mean().detach().cpu().item()), 6),
    }


def _tensorize(rows: list[dict[str, str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for row in rows:
        causal_k = float(row["causal_k"]) / 32.0
        features.append(
            [
                causal_k,
                float(row["interaction_density"]),
                float(row["min_pair_distance"]),
                float(row["mean_pair_distance"]),
                float(row["target_x"]),
                float(row["target_y"]),
                float(row["event_norm"]),
            ]
        )
        labels.append(float(row["dense_needed"]))
    return torch.tensor(features, dtype=torch.float32, device=device), torch.tensor(labels, dtype=torch.float32, device=device)


def _heuristic_scores(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(row["interaction_density"]) for row in rows], dtype=torch.float32, device=device)


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
