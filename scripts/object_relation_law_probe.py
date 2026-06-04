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


TRAIN_MECHANISMS = ("contact_inverse", "support_inverse")
EVAL_MECHANISMS = ("hidden_inverse",)
POLICIES = (
    "no_relation",
    "geometry_law",
    "type_prior_law",
    "history_relation_law",
    "oracle_relation_law",
)
SUMMARY_COLUMNS = (
    "relation_precision",
    "relation_recall",
    "mean_selected_k",
    "delta_mse",
    "sign_accuracy",
)


@dataclass(slots=True)
class Candidate:
    object_id: str
    object_type: str
    distance: float
    current_impulse: float
    source_history: list[float]
    target_history: list[float]
    current_delta: float
    is_causal: bool


@dataclass(slots=True)
class Sample:
    mechanism: str
    candidates: list[Candidate]
    target_delta: float


@dataclass(slots=True)
class LinearScorer:
    weights: torch.Tensor
    bias: torch.Tensor


@dataclass(slots=True)
class LocalLaw:
    weights: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether objectified relation histories support a transferable "
            "local inverse-distance law on held-out object names."
        )
    )
    parser.add_argument("--train-samples", type=int, default=768)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--seeds", type=int, nargs="*", help="Optional explicit seed list for a multi-seed run.")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=14)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--geometry-threshold", type=float, default=0.85)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/object_relation_law_probe.csv"),
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    rows: list[dict[str, str]] = []
    for seed in seeds:
        scorer, law = train_models(
            samples=args.train_samples,
            seed=seed,
            candidates=args.candidates,
            history_steps=args.history_steps,
            train_steps=args.train_steps,
        )
        seed_rows = run_probe(
            scorer=scorer,
            law=law,
            samples=args.eval_samples,
            seed=seed + 200_000,
            candidates=args.candidates,
            history_steps=args.history_steps,
            threshold=args.threshold,
            geometry_threshold=args.geometry_threshold,
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
            "object_relation_law_probe "
            f"row_type={row['row_type']} "
            f"policy={row['policy']} "
            f"mechanism={row['mechanism']} "
            f"relation_recall={row['relation_recall']} "
            f"relation_precision={row['relation_precision']} "
            f"delta_mse={row['delta_mse']} "
            f"sign_accuracy={row['sign_accuracy']}"
        )


def train_models(
    *,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    train_steps: int,
) -> tuple[LinearScorer, LocalLaw]:
    rng = random.Random(seed)
    scorer_features: list[torch.Tensor] = []
    scorer_labels: list[float] = []
    law_features: list[torch.Tensor] = []
    law_targets: list[float] = []

    for _ in range(samples):
        mechanism = rng.choice(TRAIN_MECHANISMS)
        sample = generate_sample(rng, mechanism=mechanism, candidates=candidates, history_steps=history_steps)
        for candidate in sample.candidates:
            scorer_features.append(relation_features(candidate))
            scorer_labels.append(1.0 if candidate.is_causal else 0.0)
            if candidate.is_causal:
                law_features.append(law_features_for(candidate))
                law_targets.append(candidate.current_delta)

    scorer = train_relation_scorer(scorer_features, scorer_labels, train_steps=train_steps)
    law = fit_local_law(law_features, law_targets)
    return scorer, law


def train_relation_scorer(features: list[torch.Tensor], labels: list[float], *, train_steps: int) -> LinearScorer:
    x = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    weights = torch.zeros((x.size(1), 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    positive_count = max(float(y.sum().item()), 1.0)
    negative_count = max(float(y.numel() - y.sum().item()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count]))
    optimizer = torch.optim.Adam([weights, bias], lr=0.05)

    for _ in range(train_steps):
        optimizer.zero_grad()
        loss = loss_fn(x @ weights + bias, y)
        loss.backward()
        optimizer.step()

    return LinearScorer(weights=weights.detach().squeeze(1), bias=bias.detach())


def fit_local_law(features: list[torch.Tensor], targets: list[float]) -> LocalLaw:
    x = torch.stack(features)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    ridge = 1e-3 * torch.eye(x.size(1), dtype=torch.float32)
    weights = torch.linalg.solve(x.T @ x + ridge, x.T @ y).squeeze(1)
    return LocalLaw(weights=weights.detach())


def run_probe(
    *,
    scorer: LinearScorer,
    law: LocalLaw,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    threshold: float,
    geometry_threshold: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mechanism in EVAL_MECHANISMS:
        for policy in POLICIES:
            rows.append(
                evaluate_policy(
                    scorer=scorer,
                    law=law,
                    policy=policy,
                    mechanism=mechanism,
                    samples=samples,
                    seed=seed,
                    candidates=candidates,
                    history_steps=history_steps,
                    threshold=threshold,
                    geometry_threshold=geometry_threshold,
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
    scorer: LinearScorer,
    law: LocalLaw,
    policy: str,
    mechanism: str,
    samples: int,
    seed: int,
    candidates: int,
    history_steps: int,
    threshold: float,
    geometry_threshold: float,
) -> dict[str, str]:
    rng = random.Random(seed)
    totals = {
        "selected": 0,
        "true_positive": 0,
        "expected": 0,
        "mse": 0.0,
        "sign_correct": 0,
        "selected_k": 0,
    }
    for _ in range(samples):
        sample = generate_sample(rng, mechanism=mechanism, candidates=candidates, history_steps=history_steps)
        selected = select_candidates(
            sample,
            scorer=scorer,
            policy=policy,
            threshold=threshold,
            geometry_threshold=geometry_threshold,
        )
        prediction = sum(predict_delta(candidate, law) for candidate in selected)
        error = prediction - sample.target_delta
        true_positive = sum(1 for candidate in selected if candidate.is_causal)
        expected = sum(1 for candidate in sample.candidates if candidate.is_causal)

        totals["selected"] += len(selected)
        totals["true_positive"] += true_positive
        totals["expected"] += expected
        totals["mse"] += error * error
        totals["sign_correct"] += int(_sign(prediction) == _sign(sample.target_delta))
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
        "delta_mse": f"{totals['mse'] / samples:.6f}",
        "sign_accuracy": f"{totals['sign_correct'] / samples:.6f}",
    }


def generate_sample(
    rng: random.Random,
    *,
    mechanism: str,
    candidates: int,
    history_steps: int,
) -> Sample:
    causal_index = rng.randrange(candidates)
    current_impulse = _signed_impulse(rng)
    result: list[Candidate] = []
    target_delta = 0.0

    for index in range(candidates):
        is_causal = index == causal_index
        distance = rng.uniform(0.35, 1.85)
        object_type = _object_type(mechanism, is_causal)
        source_history = _source_history(rng, history_steps)
        target_history = _target_history(
            rng,
            source_history=source_history,
            distance=distance,
            is_causal=is_causal,
            mechanism=mechanism,
        )
        current_delta = rng.gauss(0.0, 0.015)
        if is_causal:
            current_delta += _gain(mechanism) * current_impulse / _distance_denominator(distance)
            target_delta = current_delta
        result.append(
            Candidate(
                object_id=f"obj_{index:03d}",
                object_type=object_type,
                distance=distance,
                current_impulse=current_impulse,
                source_history=source_history,
                target_history=target_history,
                current_delta=current_delta,
                is_causal=is_causal,
            )
        )
    return Sample(mechanism=mechanism, candidates=result, target_delta=target_delta)


def select_candidates(
    sample: Sample,
    *,
    scorer: LinearScorer,
    policy: str,
    threshold: float,
    geometry_threshold: float,
) -> list[Candidate]:
    if policy == "no_relation":
        return []
    if policy == "oracle_relation_law":
        return [candidate for candidate in sample.candidates if candidate.is_causal]
    if policy == "geometry_law":
        return [candidate for candidate in sample.candidates if candidate.distance <= geometry_threshold]
    if policy == "type_prior_law":
        return [candidate for candidate in sample.candidates if candidate.object_type in {"contact", "support"}]
    if policy == "history_relation_law":
        selected = [
            candidate
            for candidate in sample.candidates
            if score_candidate(candidate, scorer) >= threshold
        ]
        if selected:
            return selected
        return [max(sample.candidates, key=lambda candidate: score_candidate(candidate, scorer))]
    raise ValueError(f"unknown policy: {policy}")


def relation_features(candidate: Candidate) -> torch.Tensor:
    return torch.tensor(
        [
            candidate.distance,
            1.0 / _distance_denominator(candidate.distance),
            _lagged_correlation(candidate),
            _same_step_correlation(candidate),
            _target_energy(candidate),
            candidate.current_impulse,
            1.0 if candidate.object_type in {"contact", "support"} else 0.0,
            1.0 if candidate.object_type in {"latent_body", "ambient_body"} else 0.0,
        ],
        dtype=torch.float32,
    )


def law_features_for(candidate: Candidate) -> torch.Tensor:
    impulse = candidate.current_impulse
    distance = candidate.distance
    return torch.tensor(
        [
            impulse / _distance_denominator(distance),
            impulse / (distance + 0.1),
            impulse,
            impulse * distance,
            distance,
            1.0,
        ],
        dtype=torch.float32,
    )


def score_candidate(candidate: Candidate, scorer: LinearScorer) -> float:
    features = relation_features(candidate)
    return float(torch.sigmoid(features @ scorer.weights + scorer.bias).item())


def predict_delta(candidate: Candidate, law: LocalLaw) -> float:
    return float(law_features_for(candidate) @ law.weights)


def _object_type(mechanism: str, is_causal: bool) -> str:
    if mechanism == "hidden_inverse":
        return "latent_body" if is_causal else "ambient_body"
    if mechanism == "support_inverse":
        return "support" if is_causal else "prop"
    return "contact" if is_causal else "prop"


def _gain(mechanism: str) -> float:
    return {"contact_inverse": 0.55, "support_inverse": 0.65, "hidden_inverse": 0.60}[mechanism]


def _distance_denominator(distance: float) -> float:
    return distance * distance + 0.20


def _signed_impulse(rng: random.Random) -> float:
    magnitude = rng.uniform(0.45, 1.25)
    return magnitude if rng.random() < 0.5 else -magnitude


def _source_history(rng: random.Random, history_steps: int) -> list[float]:
    values = [rng.gauss(0.0, 0.06) for _ in range(history_steps)]
    spike_index = rng.randrange(1, history_steps - 2)
    values[spike_index] += _signed_impulse(rng)
    return values


def _target_history(
    rng: random.Random,
    *,
    source_history: list[float],
    distance: float,
    is_causal: bool,
    mechanism: str,
) -> list[float]:
    values = [rng.gauss(0.0, 0.04) for _ in source_history]
    if is_causal:
        scale = _gain(mechanism) / _distance_denominator(distance)
        for index in range(1, len(source_history)):
            values[index] += scale * source_history[index - 1]
    return values


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


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


if __name__ == "__main__":
    main()
