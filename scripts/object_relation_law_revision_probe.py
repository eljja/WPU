from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import object_relation_law_probe as base  # noqa: E402


REVISION_MECHANISMS = ("hidden_inverse_gain_shift", "hidden_power_shift")
POLICIES = (
    "base_history_law",
    "gain_calibrated_history_law",
    "form_revised_history_law",
    "form_revised_oracle_law",
)
FORM_CANDIDATES = ("inverse_square", "inverse_cube", "inverse_linear", "mixed")
SUMMARY_COLUMNS = (
    "relation_precision",
    "relation_recall",
    "mean_selected_k",
    "delta_mse",
    "sign_accuracy",
    "calibration_mse",
)


@dataclass(slots=True)
class RevisionLaw:
    form: str
    weights: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether OOD residuals over objectified relations can revise "
            "a local law with a small calibration set."
        )
    )
    parser.add_argument("--train-samples", type=int, default=768)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--seeds", type=int, nargs="*", help="Optional explicit seed list for a multi-seed run.")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=14)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--mechanisms",
        nargs="*",
        default=list(REVISION_MECHANISMS),
        choices=base.SUPPORTED_MECHANISMS,
        help="Held-out mechanisms to revise and evaluate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/object_relation_law_revision_probe.csv"),
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    rows: list[dict[str, str]] = []
    for seed in seeds:
        scorer, trained_law = base.train_models(
            samples=args.train_samples,
            seed=seed,
            candidates=args.candidates,
            history_steps=args.history_steps,
            train_steps=args.train_steps,
        )
        for mechanism in args.mechanisms:
            calibration = generate_samples(
                seed=seed + 300_000,
                mechanism=mechanism,
                samples=args.calibration_samples,
                candidates=args.candidates,
                history_steps=args.history_steps,
            )
            test_samples = generate_samples(
                seed=seed + 400_000,
                mechanism=mechanism,
                samples=args.eval_samples,
                candidates=args.candidates,
                history_steps=args.history_steps,
            )
            revisions = build_revisions(
                scorer=scorer,
                base_law=trained_law,
                calibration=calibration,
                threshold=args.threshold,
            )
            for policy in POLICIES:
                row = evaluate_revision(
                    scorer=scorer,
                    base_law=trained_law,
                    revisions=revisions,
                    policy=policy,
                    mechanism=mechanism,
                    samples=test_samples,
                    threshold=args.threshold,
                    candidate_count=args.candidates,
                )
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
            "object_relation_law_revision_probe "
            f"row_type={row['row_type']} "
            f"mechanism={row['mechanism']} "
            f"policy={row['policy']} "
            f"selected_form={row['selected_form']} "
            f"delta_mse={row['delta_mse']} "
            f"calibration_mse={row['calibration_mse']}"
        )


def generate_samples(
    *,
    seed: int,
    mechanism: str,
    samples: int,
    candidates: int,
    history_steps: int,
) -> list[base.Sample]:
    rng = random.Random(seed)
    return [
        base.generate_sample(
            rng,
            mechanism=mechanism,
            candidates=candidates,
            history_steps=history_steps,
        )
        for _ in range(samples)
    ]


def build_revisions(
    *,
    scorer: base.LinearScorer,
    base_law: base.LocalLaw,
    calibration: list[base.Sample],
    threshold: float,
) -> dict[str, tuple[RevisionLaw, float]]:
    history_pairs = collect_pairs(calibration, scorer=scorer, policy="history", threshold=threshold)
    oracle_pairs = collect_pairs(calibration, scorer=scorer, policy="oracle", threshold=threshold)

    gain = fit_gain_calibration(history_pairs, base_law)
    gain_law = RevisionLaw(form="gain_scaled_base", weights=torch.tensor([gain], dtype=torch.float32))
    history_form_law = select_best_form(history_pairs)
    oracle_form_law = select_best_form(oracle_pairs)
    return {
        "gain_calibrated_history_law": (gain_law, calibration_mse(history_pairs, gain_law, base_law=base_law)),
        "form_revised_history_law": (history_form_law, calibration_mse(history_pairs, history_form_law)),
        "form_revised_oracle_law": (oracle_form_law, calibration_mse(oracle_pairs, oracle_form_law)),
    }


def collect_pairs(
    samples: list[base.Sample],
    *,
    scorer: base.LinearScorer,
    policy: str,
    threshold: float,
) -> list[tuple[base.Candidate, float]]:
    pairs: list[tuple[base.Candidate, float]] = []
    for sample in samples:
        if policy == "oracle":
            selected = [candidate for candidate in sample.candidates if candidate.is_causal]
        else:
            selected = [
                candidate
                for candidate in sample.candidates
                if base.score_candidate(candidate, scorer) >= threshold
            ]
            if not selected:
                selected = [max(sample.candidates, key=lambda candidate: base.score_candidate(candidate, scorer))]
        for candidate in selected:
            pairs.append((candidate, candidate.current_delta))
    return pairs


def fit_gain_calibration(pairs: list[tuple[base.Candidate, float]], trained_law: base.LocalLaw) -> float:
    numerator = 0.0
    denominator = 0.0
    for candidate, target in pairs:
        prediction = base.predict_delta(candidate, trained_law)
        numerator += prediction * target
        denominator += prediction * prediction
    if denominator <= 1e-9:
        return 1.0
    return numerator / denominator


def select_best_form(pairs: list[tuple[base.Candidate, float]]) -> RevisionLaw:
    best_law: RevisionLaw | None = None
    best_mse = float("inf")
    for form in FORM_CANDIDATES:
        law = fit_form_law(pairs, form=form)
        mse = calibration_mse(pairs, law)
        if mse < best_mse:
            best_law = law
            best_mse = mse
    if best_law is None:
        return RevisionLaw(form="inverse_square", weights=torch.zeros(2, dtype=torch.float32))
    return best_law


def fit_form_law(pairs: list[tuple[base.Candidate, float]], *, form: str) -> RevisionLaw:
    if not pairs:
        return RevisionLaw(form=form, weights=torch.zeros(form_features(None, form=form).numel(), dtype=torch.float32))
    x = torch.stack([form_features(candidate, form=form) for candidate, _ in pairs])
    y = torch.tensor([target for _, target in pairs], dtype=torch.float32).unsqueeze(1)
    ridge = 1e-3 * torch.eye(x.size(1), dtype=torch.float32)
    weights = torch.linalg.solve(x.T @ x + ridge, x.T @ y).squeeze(1)
    return RevisionLaw(form=form, weights=weights.detach())


def form_features(candidate: base.Candidate | None, *, form: str) -> torch.Tensor:
    if candidate is None:
        widths = {"inverse_square": 2, "inverse_cube": 2, "inverse_linear": 2, "mixed": 4}
        return torch.zeros(widths[form], dtype=torch.float32)
    impulse = candidate.current_impulse
    distance = candidate.distance
    if form == "inverse_square":
        values = [impulse / (distance * distance + 0.20), 1.0]
    elif form == "inverse_cube":
        values = [impulse / (distance**3 + 0.20), 1.0]
    elif form == "inverse_linear":
        values = [impulse / (distance + 0.10), 1.0]
    elif form == "mixed":
        values = [
            impulse / (distance * distance + 0.20),
            impulse / (distance**3 + 0.20),
            impulse,
            1.0,
        ]
    else:
        raise ValueError(f"unknown form: {form}")
    return torch.tensor(values, dtype=torch.float32)


def calibration_mse(
    pairs: list[tuple[base.Candidate, float]],
    law: RevisionLaw,
    *,
    base_law: base.LocalLaw | None = None,
) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for candidate, target in pairs:
        prediction = predict_revision(candidate, law, base_law=base_law)
        total += (prediction - target) ** 2
    return total / len(pairs)


def evaluate_revision(
    *,
    scorer: base.LinearScorer,
    base_law: base.LocalLaw,
    revisions: dict[str, tuple[RevisionLaw, float]],
    policy: str,
    mechanism: str,
    samples: list[base.Sample],
    threshold: float,
    candidate_count: int,
) -> dict[str, str]:
    totals = {
        "selected": 0,
        "true_positive": 0,
        "expected": 0,
        "mse": 0.0,
        "sign_correct": 0,
        "selected_k": 0,
    }
    selected_form = "trained_base"
    calibration_error = 0.0
    revision_law: RevisionLaw | None = None
    if policy in revisions:
        revision_law, calibration_error = revisions[policy]
        selected_form = revision_law.form

    for sample in samples:
        if policy == "form_revised_oracle_law":
            selected = [candidate for candidate in sample.candidates if candidate.is_causal]
        else:
            selected = [
                candidate
                for candidate in sample.candidates
                if base.score_candidate(candidate, scorer) >= threshold
            ]
            if not selected:
                selected = [max(sample.candidates, key=lambda candidate: base.score_candidate(candidate, scorer))]

        prediction = 0.0
        for candidate in selected:
            if policy == "base_history_law":
                prediction += base.predict_delta(candidate, base_law)
            elif policy == "gain_calibrated_history_law":
                assert revision_law is not None
                prediction += predict_revision(candidate, revision_law, base_law=base_law)
            else:
                assert revision_law is not None
                prediction += predict_revision(candidate, revision_law)

        error = prediction - sample.target_delta
        totals["selected"] += len(selected)
        totals["true_positive"] += sum(1 for candidate in selected if candidate.is_causal)
        totals["expected"] += sum(1 for candidate in sample.candidates if candidate.is_causal)
        totals["mse"] += error * error
        totals["sign_correct"] += int(base._sign(prediction) == base._sign(sample.target_delta))
        totals["selected_k"] += len(selected)

    precision = totals["true_positive"] / max(totals["selected"], 1)
    recall = totals["true_positive"] / max(totals["expected"], 1)
    return {
        "mechanism": mechanism,
        "policy": policy,
        "selected_form": selected_form,
        "samples": str(len(samples)),
        "candidate_count": str(candidate_count),
        "relation_precision": f"{precision:.6f}",
        "relation_recall": f"{recall:.6f}",
        "mean_selected_k": f"{totals['selected_k'] / max(len(samples), 1):.6f}",
        "delta_mse": f"{totals['mse'] / max(len(samples), 1):.6f}",
        "sign_accuracy": f"{totals['sign_correct'] / max(len(samples), 1):.6f}",
        "calibration_mse": f"{calibration_error:.6f}",
    }


def predict_revision(
    candidate: base.Candidate,
    law: RevisionLaw,
    *,
    base_law: base.LocalLaw | None = None,
) -> float:
    if law.form == "gain_scaled_base":
        if base_law is None:
            raise ValueError("gain-scaled law requires base_law")
        return float(law.weights[0].item() * base.predict_delta(candidate, base_law))
    return float(form_features(candidate, form=law.form) @ law.weights)


def summarize_seed_rows(rows: list[dict[str, str]], *, seeds: list[int]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["mechanism"], row["policy"])].append(row)

    summaries: list[dict[str, str]] = []
    for (mechanism, policy), group in sorted(groups.items()):
        first = group[0]
        form_counts = Counter(row["selected_form"] for row in group)
        summary = {
            "row_type": "summary",
            "seed": "all",
            "mechanism": mechanism,
            "policy": policy,
            "selected_form": form_counts.most_common(1)[0][0],
            "samples": str(sum(int(row["samples"]) for row in group)),
            "candidate_count": first["candidate_count"],
        }
        for column in SUMMARY_COLUMNS:
            values = [float(row[column]) for row in group]
            summary[column] = f"{sum(values) / len(values):.6f}"
        summary["seed_count"] = str(len(seeds))
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    main()
