from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


TRAIN_MECHANISMS = ("contact_transfer", "support_transfer")
EVAL_MECHANISMS = ("hidden_field",)
POLICIES = ("no_relation", "geometry_only", "type_prior", "history_scorer", "oracle_relation")
SUMMARY_COLUMNS = (
    "relation_precision",
    "relation_recall",
    "mean_selected_k",
    "downstream_accuracy",
    "downstream_loss",
)


@dataclass(slots=True)
class Candidate:
    object_id: str
    object_type: str
    distance: float
    source_history: list[float]
    target_history: list[float]
    is_causal: bool


@dataclass(slots=True)
class Sample:
    mechanism: str
    candidates: list[Candidate]
    target_label: int


@dataclass(slots=True)
class LinearHistoryScorer:
    weights: torch.Tensor
    bias: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether relation candidates learned from object histories "
            "transfer to a held-out mechanism family."
        )
    )
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--seeds", type=int, nargs="*", help="Optional explicit seed list for a multi-seed run.")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/object_history_hidden_mechanism_probe.csv"),
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    rows: list[dict[str, str]] = []
    for seed in seeds:
        scorer = train_history_scorer(
            samples=args.train_samples,
            seed=seed,
            candidates=args.candidates,
            history_steps=args.history_steps,
            train_steps=args.train_steps,
        )
        seed_rows = run_probe(
            scorer=scorer,
            samples=args.eval_samples,
            seed=seed + 100_000,
            candidates=args.candidates,
            history_steps=args.history_steps,
            threshold=args.threshold,
        )
        for row in seed_rows:
            rows.append({"row_type": "seed", "seed": str(seed), **row, "seed_count": "1"})
    if len(seeds) > 1:
        rows.extend(summarize_seed_rows(rows, seeds=seeds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if len(seeds) > 1 and row["row_type"] != "summary":
            continue
        print(
            "hidden_mechanism_probe "
            f"row_type={row['row_type']} "
            f"policy={row['policy']} "
            f"mechanism={row['mechanism']} "
            f"relation_recall={row['relation_recall']} "
            f"relation_precision={row['relation_precision']} "
            f"downstream_accuracy={row['downstream_accuracy']} "
            f"downstream_loss={row['downstream_loss']}"
        )


def train_history_scorer(
    *,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    train_steps: int,
) -> LinearHistoryScorer:
    rng = random.Random(seed)
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for _ in range(samples):
        mechanism = rng.choice(TRAIN_MECHANISMS)
        sample = generate_sample(rng, mechanism=mechanism, candidates=candidates, history_steps=history_steps)
        for candidate in sample.candidates:
            features.append(candidate_features(candidate))
            labels.append(1.0 if candidate.is_causal else 0.0)

    x = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    weights = torch.zeros((x.size(1), 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    positive_count = max(float(y.sum().item()), 1.0)
    negative_count = max(float(y.numel() - y.sum().item()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count]))
    optimizer = torch.optim.Adam([weights, bias], lr=0.06)

    for _ in range(train_steps):
        optimizer.zero_grad()
        loss = loss_fn(x @ weights + bias, y)
        loss.backward()
        optimizer.step()

    return LinearHistoryScorer(weights=weights.detach().squeeze(1), bias=bias.detach())


def run_probe(
    *,
    scorer: LinearHistoryScorer,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    threshold: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mechanism in EVAL_MECHANISMS:
        for policy in POLICIES:
            rows.append(
                evaluate_policy(
                    scorer=scorer,
                    policy=policy,
                    mechanism=mechanism,
                    samples=samples,
                    seed=seed,
                    candidates=candidates,
                    history_steps=history_steps,
                    threshold=threshold,
                )
            )
    return rows


def summarize_seed_rows(rows: list[dict[str, str]], *, seeds: list[int]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["mechanism"], row["policy"])].append(row)

    summaries: list[dict[str, str]] = []
    for (mechanism, policy), group in sorted(groups.items()):
        first = group[0]
        summary = {
            "row_type": "summary",
            "seed": "all",
            "mechanism": mechanism,
            "policy": policy,
            "samples": str(sum(int(row["samples"]) for row in group)),
            "candidate_count": first["candidate_count"],
        }
        for column in SUMMARY_COLUMNS:
            values = [float(row[column]) for row in group]
            summary[column] = f"{sum(values) / len(values):.6f}"
        summary["seed_count"] = str(len(seeds))
        summaries.append(summary)
    return summaries


def evaluate_policy(
    *,
    scorer: LinearHistoryScorer,
    policy: str,
    mechanism: str,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    threshold: float,
) -> dict[str, str]:
    rng = random.Random(seed)
    totals = {
        "selected": 0,
        "true_positive": 0,
        "expected": 0,
        "correct": 0,
        "loss": 0.0,
        "selected_k": 0,
    }
    for _ in range(samples):
        sample = generate_sample(rng, mechanism=mechanism, candidates=candidates, history_steps=history_steps)
        selected = select_candidates(sample, scorer=scorer, policy=policy, threshold=threshold)
        true_positive = sum(1 for candidate in selected if candidate.is_causal)
        expected = sum(1 for candidate in sample.candidates if candidate.is_causal)
        predicted_label, loss = downstream_prediction(sample, selected)

        totals["selected"] += len(selected)
        totals["true_positive"] += true_positive
        totals["expected"] += expected
        totals["correct"] += int(predicted_label == sample.target_label)
        totals["loss"] += loss
        totals["selected_k"] += len(selected)

    precision = totals["true_positive"] / max(totals["selected"], 1)
    recall = totals["true_positive"] / max(totals["expected"], 1)
    return {
        "mechanism": mechanism,
        "policy": policy,
        "samples": str(samples),
        "candidate_count": str(candidates),
        "relation_precision": f"{precision:.6f}",
        "relation_recall": f"{recall:.6f}",
        "mean_selected_k": f"{totals['selected_k'] / samples:.6f}",
        "downstream_accuracy": f"{totals['correct'] / samples:.6f}",
        "downstream_loss": f"{totals['loss'] / samples:.6f}",
    }


def generate_sample(
    rng: random.Random,
    *,
    mechanism: str,
    candidates: int,
    history_steps: int,
) -> Sample:
    causal_index = rng.randrange(candidates)
    target_label = rng.randrange(2)
    event_sign = 1.0 if target_label == 1 else -1.0
    result: list[Candidate] = []
    for index in range(candidates):
        is_causal = index == causal_index
        distance = rng.uniform(0.04, 0.32)
        if mechanism == "hidden_field":
            object_type = "latent_body" if is_causal else "ambient_body"
        elif mechanism == "support_transfer":
            object_type = "support" if is_causal else "prop"
        else:
            object_type = "contact" if is_causal else "prop"
        source_history = _source_history(rng, event_sign, history_steps)
        target_history = _target_history(
            rng,
            source_history=source_history,
            is_causal=is_causal,
            event_sign=event_sign,
            mechanism=mechanism,
        )
        result.append(
            Candidate(
                object_id=f"obj_{index:03d}",
                object_type=object_type,
                distance=distance,
                source_history=source_history,
                target_history=target_history,
                is_causal=is_causal,
            )
        )
    return Sample(mechanism=mechanism, candidates=result, target_label=target_label)


def select_candidates(
    sample: Sample,
    *,
    scorer: LinearHistoryScorer,
    policy: str,
    threshold: float,
) -> list[Candidate]:
    if policy == "no_relation":
        return []
    if policy == "oracle_relation":
        return [candidate for candidate in sample.candidates if candidate.is_causal]
    if policy == "geometry_only":
        return [candidate for candidate in sample.candidates if candidate.distance <= 0.16]
    if policy == "type_prior":
        return [
            candidate
            for candidate in sample.candidates
            if candidate.object_type in {"contact", "support"}
        ]
    if policy == "history_scorer":
        selected = [
            candidate
            for candidate in sample.candidates
            if score_candidate(candidate, scorer) >= threshold
        ]
        if selected:
            return selected
        return [max(sample.candidates, key=lambda candidate: score_candidate(candidate, scorer))]
    raise ValueError(f"unknown policy: {policy}")


def downstream_prediction(sample: Sample, selected: list[Candidate]) -> tuple[int, float]:
    if not selected:
        logits = [0.0, 0.0]
    else:
        evidence = sum(candidate.target_history[-1] for candidate in selected) / math.sqrt(len(selected))
        logits = [-evidence, evidence]
    prediction = 1 if logits[1] >= logits[0] else 0
    return prediction, _cross_entropy(logits, sample.target_label)


def candidate_features(candidate: Candidate) -> torch.Tensor:
    return torch.tensor(
        [
            candidate.distance,
            1.0 / max(candidate.distance, 1e-3),
            _lagged_correlation(candidate),
            _same_step_correlation(candidate),
            _target_energy(candidate),
            1.0 if candidate.object_type in {"contact", "support"} else 0.0,
            1.0 if candidate.object_type in {"latent_body", "ambient_body"} else 0.0,
        ],
        dtype=torch.float32,
    )


def score_candidate(candidate: Candidate, scorer: LinearHistoryScorer) -> float:
    features = candidate_features(candidate)
    return float(torch.sigmoid(features @ scorer.weights + scorer.bias).item())


def _source_history(rng: random.Random, event_sign: float, history_steps: int) -> list[float]:
    values = [rng.gauss(0.0, 0.08) for _ in range(history_steps)]
    values[history_steps // 2] += event_sign
    return values


def _target_history(
    rng: random.Random,
    *,
    source_history: list[float],
    is_causal: bool,
    event_sign: float,
    mechanism: str,
) -> list[float]:
    values = [rng.gauss(0.0, 0.08) for _ in source_history]
    if is_causal:
        gain = {"contact_transfer": 0.9, "support_transfer": 0.75, "hidden_field": 0.82}[mechanism]
        for index in range(1, len(source_history)):
            values[index] += gain * source_history[index - 1]
        values[-1] += 0.6 * event_sign
    return values


def _lagged_influence(candidate: Candidate) -> float:
    return sum(
        candidate.source_history[index - 1] * candidate.target_history[index]
        for index in range(1, len(candidate.source_history))
    ) / max(len(candidate.source_history) - 1, 1)


def _lagged_correlation(candidate: Candidate) -> float:
    return _correlation(candidate.source_history[:-1], candidate.target_history[1:])


def _same_step_correlation(candidate: Candidate) -> float:
    return _correlation(candidate.source_history, candidate.target_history)


def _target_energy(candidate: Candidate) -> float:
    return sum(value * value for value in candidate.target_history) / len(candidate.target_history)


def _correlation(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 1e-9:
        return 0.0
    return numerator / denominator


def _cross_entropy(logits: list[float], target: int) -> float:
    max_logit = max(logits)
    exp_sum = sum(math.exp(logit - max_logit) for logit in logits)
    log_prob = logits[target] - max_logit - math.log(exp_sum)
    return -log_prob


if __name__ == "__main__":
    main()
