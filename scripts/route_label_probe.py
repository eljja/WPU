from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether dense-needed labels are identifiable from state features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/route_label_probe.csv"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=248366)
    parser.add_argument("--calibration-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = _read_rows(args.input)
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, object]] = []
    for test_seed in seeds:
        train_rows = [row for row in rows if int(row["seed"]) != test_seed]
        test_rows = [row for row in rows if int(row["seed"]) == test_seed]
        output.extend(_run_split(train_rows, test_rows, test_seed, args, STATE_FEATURES, "mlp_state"))
        if rows and all(feature in rows[0] for feature in SPARSE_DIAGNOSTIC_FEATURES):
            output.extend(_run_split(train_rows, test_rows, test_seed, args, SPARSE_DIAGNOSTIC_FEATURES, "mlp_sparse_diagnostics"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, output)
    print(f"wrote={args.out}", flush=True)


def _run_split(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    test_seed: int,
    args: argparse.Namespace,
    features: list[str],
    model_name: str,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    train_x, train_y = _tensorize(train_rows, features, device)
    test_x, test_y = _tensorize(test_rows, features, device)
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
        train_probabilities = torch.sigmoid(model(train_x).squeeze(-1))
        probabilities = torch.sigmoid(model(test_x).squeeze(-1))
    outputs = []
    for threshold in args.calibration_thresholds:
        predictions = probabilities >= threshold
        outputs.append(_metrics(test_seed, test_y, predictions, probabilities, threshold, model_name, len(features)))
    outputs.extend(_calibrated_metrics(test_seed, train_y, train_probabilities, test_y, probabilities, args, model_name, len(features)))
    if model_name == "mlp_state":
        heuristic_scores = _heuristic_scores(test_rows, device)
        train_heuristic_scores = _heuristic_scores(train_rows, device)
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
            predictions = heuristic_scores >= threshold
            outputs.append(_metrics(test_seed, test_y, predictions, heuristic_scores, threshold, "interaction_density", 1))
        outputs.extend(
            _calibrated_metrics(
                test_seed,
                train_y,
                train_heuristic_scores,
                test_y,
                heuristic_scores,
                args,
                "interaction_density",
                1,
            )
        )
    return outputs


def _calibrated_metrics(
    test_seed: int,
    train_labels: torch.Tensor,
    train_scores: torch.Tensor,
    test_labels: torch.Tensor,
    test_scores: torch.Tensor,
    args: argparse.Namespace,
    model_name: str,
    feature_count: int,
) -> list[dict[str, object]]:
    rows = []
    for criterion in ["balanced_accuracy", "f1"]:
        threshold = _select_threshold(train_labels, train_scores, args.calibration_thresholds, criterion)
        predictions = test_scores >= threshold
        rows.append(_metrics(test_seed, test_labels, predictions, test_scores, threshold, f"{model_name}_cal_{criterion}", feature_count))
    return rows


def _select_threshold(labels: torch.Tensor, scores: torch.Tensor, thresholds: list[float], criterion: str) -> float:
    best_threshold = thresholds[0]
    best_value = -1.0
    for threshold in thresholds:
        metrics = _confusion_metrics(labels, scores >= threshold)
        value = metrics[criterion]
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return best_threshold


def _metrics(
    test_seed: int,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    model_name: str,
    feature_count: int,
) -> dict[str, object]:
    confusion = _confusion_metrics(labels, predictions)
    rank = _rank_metrics(labels, scores)
    return {
        "model": model_name,
        "test_seed": test_seed,
        "threshold": threshold,
        "samples": max(int(labels.numel()), 1),
        "feature_count": feature_count,
        "label_rate": round(float(labels.mean().detach().cpu().item()), 6),
        "predicted_dense_rate": round(float(predictions.float().mean().detach().cpu().item()), 6),
        "accuracy": round(confusion["accuracy"], 6),
        "balanced_accuracy": round(confusion["balanced_accuracy"], 6),
        "precision": round(confusion["precision"], 6),
        "recall": round(confusion["recall"], 6),
        "f1": round(confusion["f1"], 6),
        "score_mean": round(float(scores.mean().detach().cpu().item()), 6),
        "roc_auc": round(rank["roc_auc"], 6),
        "average_precision": round(rank["average_precision"], 6),
        "brier_score": round(rank["brier_score"], 6),
        "ece": round(rank["ece"], 6),
    }


def _confusion_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
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
        "accuracy": (true_positive + true_negative) / total,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _rank_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels_cpu = labels.detach().float().cpu()
    scores_cpu = scores.detach().float().cpu()
    positive_count = int(labels_cpu.sum().item())
    negative_count = int(labels_cpu.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        roc_auc = 0.5
        average_precision = float(labels_cpu.mean().item())
    else:
        order = torch.argsort(scores_cpu)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, scores_cpu.numel() + 1, dtype=torch.float32)
        positive_rank_sum = ranks[labels_cpu.bool()].sum().item()
        roc_auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)

        descending = torch.argsort(scores_cpu, descending=True)
        sorted_labels = labels_cpu[descending]
        cumulative_positive = torch.cumsum(sorted_labels, dim=0)
        precision_at_k = cumulative_positive / torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
        average_precision = float((precision_at_k * sorted_labels).sum().item() / max(positive_count, 1))

    brier_score = float((scores_cpu.clamp(0.0, 1.0) - labels_cpu).square().mean().item())
    ece = _expected_calibration_error(labels_cpu, scores_cpu)
    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier_score": brier_score,
        "ece": ece,
    }


def _expected_calibration_error(labels: torch.Tensor, scores: torch.Tensor, bins: int = 10) -> float:
    scores = scores.clamp(0.0, 1.0)
    total = max(labels.numel(), 1)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if not bool(mask.any()):
            continue
        confidence = float(scores[mask].mean().item())
        accuracy = float(labels[mask].mean().item())
        error += float(mask.float().mean().item()) * abs(confidence - accuracy)
    return error


def _tensorize(rows: list[dict[str, str]], feature_names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = []
    labels = []
    for row in rows:
        feature_values = []
        for feature in feature_names:
            if feature == "causal_k":
                feature_values.append(float(row[feature]) / 32.0)
            else:
                feature_values.append(float(row[feature]))
        feature_rows.append(feature_values)
        labels.append(float(row["dense_needed"]))
    return torch.tensor(feature_rows, dtype=torch.float32, device=device), torch.tensor(labels, dtype=torch.float32, device=device)


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
