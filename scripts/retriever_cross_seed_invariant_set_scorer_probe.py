from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class InvariantSetScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, candidate_count, feature_dim = features.shape
        scores = self.scorer(features.view(batch_size * candidate_count, feature_dim))
        return scores.view(batch_size, candidate_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate invariant candidate-set scoring under cross-seed transfer.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--regret-retriever-steps", type=int, default=600)
    parser.add_argument("--composition-steps", type=int, default=600)
    parser.add_argument("--scorer-steps", type=int, default=600)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--scorer-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--scorer-lr", type=float, default=3e-3)
    parser.add_argument("--safe-margin", type=float, default=0.0)
    parser.add_argument("--mechanism-safe-margin", type=float, default=0.0)
    parser.add_argument("--mechanism-min-win-rate", type=float, default=0.75)
    parser.add_argument("--mechanism-max-seed-harm", type=float, default=0.0)
    parser.add_argument("--mechanism-risk-penalty", type=float, default=0.5)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--normalized-utility-weight", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            rows.extend(_run_group(background_objects, causal_obstacles, args))
            _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_group(background_objects: int, causal_obstacles: int, args: argparse.Namespace) -> list[dict[str, object]]:
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
    base_candidate_names = [*BASE_MODES, *[f"generated_{index}" for index in range(args.generated_candidates)]]
    candidate_names = [*base_candidate_names, *COMPOSITION_MODES]
    contexts = {}
    for seed in args.seeds:
        print(f"invariant-set collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, base_candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, base_candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, base_candidate_names, args)
        train_examples = []
        for seed, context in contexts.items():
            if seed == heldout_seed:
                continue
            seed_examples = _collect_examples(context, object_retriever, composition_prior, candidate_names, args, split="validation")
            for example in seed_examples:
                example["source_seed"] = seed
            train_examples.extend(seed_examples)
        test_examples = _collect_examples(heldout, object_retriever, composition_prior, candidate_names, args, split="test")
        for variant in ("role_geometry_family", "role_geometry_only"):
            scorer = _train_invariant_scorer(train_examples, candidate_names, args, variant)
            condition_rows = _summarize(test_examples, train_examples, scorer, candidate_names, variant, args)
            for row in condition_rows:
                row.update(
                    {
                        "seed": heldout_seed,
                        "heldout_seed": heldout_seed,
                        "train_seed_count": len(args.seeds) - 1,
                        "total_objects_n": total_n,
                        "causal_k": causal_k,
                        "budget": args.budget,
                        "generated_candidates": args.generated_candidates,
                        "feature_variant": variant,
                        "interaction_mode": args.interaction_mode,
                        "propagation_steps": args.propagation_steps,
                        "retriever_steps": args.retriever_steps,
                        "regret_retriever_steps": args.regret_retriever_steps,
                        "composition_steps": args.composition_steps,
                        "scorer_steps": args.scorer_steps,
                        "validation_samples_per_seed": args.validation_samples,
                        "test_samples": args.samples,
                    }
                )
            rows.extend(condition_rows)
    return rows


def _invariant_features(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    variant: str,
) -> torch.Tensor:
    candidate_count = len(candidate_names)
    features = []
    for example in examples:
        context = example["context_features"][:, candidate_count:].clone()  # type: ignore[index]
        if variant == "role_geometry_only":
            context[:, -2:] = 0.0
        elif variant != "role_geometry_family":
            raise ValueError(f"unknown feature variant: {variant}")
        features.append(context)
    return torch.stack(features)


def _loss_tensor(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(example[f"{name}_loss"]) for name in candidate_names] for example in examples],
        dtype=torch.float32,
    )


def _train_invariant_scorer(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
    variant: str,
) -> InvariantSetScorer:
    features = _invariant_features(examples, candidate_names, variant)
    losses = _loss_tensor(examples, candidate_names)
    centered_loss = losses - losses.mean(dim=1, keepdim=True)
    scale = losses.std(dim=1, keepdim=True).clamp_min(1e-3)
    utilities = -(centered_loss / scale)
    model = InvariantSetScorer(features.size(-1), args.scorer_hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.scorer_lr)
    targets = utilities.argmax(dim=1)
    soft_targets = F.softmax(utilities / args.utility_temperature, dim=1)
    model.train()
    for _ in range(args.scorer_steps):
        scores = model(features)
        log_probs = F.log_softmax(scores, dim=1)
        loss = (
            F.cross_entropy(scores, targets)
            + 0.5 * -(soft_targets * log_probs).sum(dim=1).mean()
            + args.normalized_utility_weight * F.mse_loss(scores, utilities)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def _predict_modes(
    examples: list[dict[str, object]],
    scorer: InvariantSetScorer,
    candidate_names: list[str],
    variant: str,
) -> list[str]:
    features = _invariant_features(examples, candidate_names, variant)
    with torch.no_grad():
        scores = scorer(features)
    return [candidate_names[int(index)] for index in scores.argmax(dim=1).tolist()]


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    scorer: InvariantSetScorer,
    candidate_names: list[str],
    variant: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    scorer_modes = _predict_modes(test_examples, scorer, candidate_names, variant)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    train_scorer_modes = _predict_modes(train_examples, scorer, candidate_names, variant)
    train_scorer_loss = _mean_policy_loss(train_examples, train_scorer_modes)
    train_static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    mechanism_modes, selected_mechanism = _choose_train_mechanism_modes(
        train_examples,
        test_examples,
        train_scorer_modes,
        scorer_modes,
    )
    stable_modes, stable_metrics = _choose_seed_stable_mechanism_modes(
        train_examples,
        test_examples,
        train_scorer_modes,
        scorer_modes,
        args,
    )
    risk_modes, risk_metrics = _choose_risk_adjusted_mechanism_modes(
        train_examples,
        test_examples,
        train_scorer_modes,
        scorer_modes,
        args,
    )
    safe_uses_scorer = train_scorer_loss + args.safe_margin < train_static_loss
    safe_modes = scorer_modes if safe_uses_scorer else static_modes
    rows = [
        _policy_row(test_examples, "static_learned_interaction", static_modes, candidate_names),
        _policy_row(test_examples, "invariant_set_scorer", scorer_modes, candidate_names),
        _policy_row(test_examples, "train_safe_invariant_set_scorer", safe_modes, candidate_names),
        _policy_row(test_examples, "train_selected_mechanism", mechanism_modes, candidate_names),
        _policy_row(test_examples, "seed_stable_selected_mechanism", stable_modes, candidate_names),
        _policy_row(test_examples, "risk_adjusted_selected_mechanism", risk_modes, candidate_names),
        _policy_row(test_examples, "generated_plus_composition_oracle", oracle_modes, candidate_names),
    ]
    for row in rows:
        row["train_invariant_scorer_loss"] = round(train_scorer_loss, 6)
        row["train_static_learned_loss"] = round(train_static_loss, 6)
        row["train_delta_vs_static"] = round(train_scorer_loss - train_static_loss, 6)
        row["selected_non_learned_rate"] = round(mean(float(mode != "learned") for mode in scorer_modes), 6)
        row["safe_margin"] = args.safe_margin
        row["safe_uses_invariant_scorer"] = int(safe_uses_scorer)
        row["train_selected_mechanism"] = selected_mechanism
        row["seed_stable_selected_mechanism"] = stable_metrics["selected_mechanism"]
        row["seed_stable_mean_delta"] = stable_metrics["mean_delta"]
        row["seed_stable_win_rate"] = stable_metrics["win_rate"]
        row["seed_stable_max_delta"] = stable_metrics["max_delta"]
        row["mechanism_safe_margin"] = args.mechanism_safe_margin
        row["mechanism_min_win_rate"] = args.mechanism_min_win_rate
        row["mechanism_max_seed_harm"] = args.mechanism_max_seed_harm
        row["risk_adjusted_selected_mechanism"] = risk_metrics["selected_mechanism"]
        row["risk_adjusted_mean_delta"] = risk_metrics["mean_delta"]
        row["risk_adjusted_win_rate"] = risk_metrics["win_rate"]
        row["risk_adjusted_max_delta"] = risk_metrics["max_delta"]
        row["risk_adjusted_score"] = risk_metrics["score"]
        row["mechanism_risk_penalty"] = args.mechanism_risk_penalty
    return rows


def _choose_train_mechanism_modes(
    train_examples: list[dict[str, object]],
    test_examples: list[dict[str, object]],
    train_scorer_modes: list[str],
    test_scorer_modes: list[str],
) -> tuple[list[str], str]:
    train_candidates = {
        "learned": ["learned"] * len(train_examples),
        "composition_argmax": ["composition_argmax"] * len(train_examples),
        "composition_expected": ["composition_expected"] * len(train_examples),
        "composition_count_only": ["composition_count_only"] * len(train_examples),
        "invariant": train_scorer_modes,
    }
    test_candidates = {
        "learned": ["learned"] * len(test_examples),
        "composition_argmax": ["composition_argmax"] * len(test_examples),
        "composition_expected": ["composition_expected"] * len(test_examples),
        "composition_count_only": ["composition_count_only"] * len(test_examples),
        "invariant": test_scorer_modes,
    }
    best_name = min(train_candidates, key=lambda name: (_mean_policy_loss(train_examples, train_candidates[name]), name))
    return test_candidates[best_name], best_name


def _choose_seed_stable_mechanism_modes(
    train_examples: list[dict[str, object]],
    test_examples: list[dict[str, object]],
    train_scorer_modes: list[str],
    test_scorer_modes: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, object]]:
    train_candidates = {
        "learned": ["learned"] * len(train_examples),
        "composition_argmax": ["composition_argmax"] * len(train_examples),
        "composition_expected": ["composition_expected"] * len(train_examples),
        "composition_count_only": ["composition_count_only"] * len(train_examples),
        "invariant": train_scorer_modes,
    }
    test_candidates = {
        "learned": ["learned"] * len(test_examples),
        "composition_argmax": ["composition_argmax"] * len(test_examples),
        "composition_expected": ["composition_expected"] * len(test_examples),
        "composition_count_only": ["composition_count_only"] * len(test_examples),
        "invariant": test_scorer_modes,
    }
    seeds = sorted({int(example["source_seed"]) for example in train_examples})
    static_seed_losses = _seed_losses(train_examples, train_candidates["learned"], seeds)
    scored: list[tuple[float, str, float, float, float]] = []
    for name, modes in train_candidates.items():
        losses = _seed_losses(train_examples, modes, seeds)
        deltas = [losses[seed] - static_seed_losses[seed] for seed in seeds]
        mean_delta = mean(deltas)
        win_rate = mean(float(delta + args.mechanism_safe_margin < 0.0) for delta in deltas)
        max_delta = max(deltas)
        if (
            name == "learned"
            or (
                mean_delta + args.mechanism_safe_margin < 0.0
                and win_rate >= args.mechanism_min_win_rate
                and max_delta <= args.mechanism_max_seed_harm
            )
        ):
            scored.append((mean_delta, name, win_rate, max_delta, mean(losses.values())))
    best_mean_delta, best_name, best_win_rate, best_max_delta, _ = min(scored, key=lambda item: (item[0], item[1]))
    return test_candidates[best_name], {
        "selected_mechanism": best_name,
        "mean_delta": round(best_mean_delta, 6),
        "win_rate": round(best_win_rate, 6),
        "max_delta": round(best_max_delta, 6),
    }


def _choose_risk_adjusted_mechanism_modes(
    train_examples: list[dict[str, object]],
    test_examples: list[dict[str, object]],
    train_scorer_modes: list[str],
    test_scorer_modes: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, object]]:
    train_candidates = {
        "learned": ["learned"] * len(train_examples),
        "composition_argmax": ["composition_argmax"] * len(train_examples),
        "composition_expected": ["composition_expected"] * len(train_examples),
        "composition_count_only": ["composition_count_only"] * len(train_examples),
        "invariant": train_scorer_modes,
    }
    test_candidates = {
        "learned": ["learned"] * len(test_examples),
        "composition_argmax": ["composition_argmax"] * len(test_examples),
        "composition_expected": ["composition_expected"] * len(test_examples),
        "composition_count_only": ["composition_count_only"] * len(test_examples),
        "invariant": test_scorer_modes,
    }
    seeds = sorted({int(example["source_seed"]) for example in train_examples})
    static_seed_losses = _seed_losses(train_examples, train_candidates["learned"], seeds)
    scored: list[tuple[float, str, float, float, float]] = []
    for name, modes in train_candidates.items():
        losses = _seed_losses(train_examples, modes, seeds)
        deltas = [losses[seed] - static_seed_losses[seed] for seed in seeds]
        mean_delta = mean(deltas)
        win_rate = mean(float(delta + args.mechanism_safe_margin < 0.0) for delta in deltas)
        max_delta = max(deltas)
        score = mean_delta + args.mechanism_risk_penalty * max(0.0, max_delta)
        scored.append((score, name, mean_delta, win_rate, max_delta))
    best_score, best_name, best_mean_delta, best_win_rate, best_max_delta = min(scored, key=lambda item: (item[0], item[1]))
    if best_score + args.mechanism_safe_margin >= 0.0:
        best_name = "learned"
        best_score = 0.0
        best_mean_delta = 0.0
        best_win_rate = 0.0
        best_max_delta = 0.0
    return test_candidates[best_name], {
        "selected_mechanism": best_name,
        "mean_delta": round(best_mean_delta, 6),
        "win_rate": round(best_win_rate, 6),
        "max_delta": round(best_max_delta, 6),
        "score": round(best_score, 6),
    }


def _seed_losses(
    examples: list[dict[str, object]],
    selected_modes: list[str],
    seeds: list[int],
) -> dict[int, float]:
    losses = {}
    for seed in seeds:
        seed_examples = []
        seed_modes = []
        for example, mode in zip(examples, selected_modes, strict=True):
            if int(example["source_seed"]) == seed:
                seed_examples.append(example)
                seed_modes.append(mode)
        losses[seed] = _mean_policy_loss(seed_examples, seed_modes)
    return losses


if __name__ == "__main__":
    main()
