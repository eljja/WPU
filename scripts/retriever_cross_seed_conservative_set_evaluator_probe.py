from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _example_tensors,
    _mean_policy_loss,
    _policy_row,
    _train_set_evaluator,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conservative cross-seed candidate-set evaluator gating.")
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
    parser.add_argument("--set-evaluator-steps", type=int, default=600)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--set-evaluator-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--set-evaluator-lr", type=float, default=3e-3)
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_conservative_set_evaluator.csv"))
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
        print(f"conservative-set collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
            source_examples = _collect_examples(context, object_retriever, composition_prior, candidate_names, args, split="validation")
            for example in source_examples:
                example["source_seed"] = seed
            train_examples.extend(source_examples)
        test_examples = _collect_examples(heldout, object_retriever, composition_prior, candidate_names, args, split="test")
        evaluator = _train_set_evaluator(train_examples, candidate_names, args)
        threshold = _choose_margin_threshold(train_examples, evaluator, candidate_names)
        robust_threshold = _choose_robust_margin_threshold(train_examples, evaluator, candidate_names)
        condition_rows = _summarize(test_examples, train_examples, evaluator, candidate_names, threshold, robust_threshold)
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
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "regret_retriever_steps": args.regret_retriever_steps,
                    "composition_steps": args.composition_steps,
                    "set_evaluator_steps": args.set_evaluator_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "gate_margin_threshold": round(threshold, 6),
                    "robust_gate_margin_threshold": round(robust_threshold, 6),
                }
            )
        rows.extend(condition_rows)
    return rows


def _score_predictions(examples: list[dict[str, object]], evaluator, candidate_names: list[str]) -> tuple[list[str], list[float]]:
    objects, masks, context, _ = _example_tensors(examples, candidate_names)
    with torch.no_grad():
        scores = evaluator(objects, masks, context)
    top2 = scores.topk(k=2, dim=1)
    modes = [candidate_names[int(index)] for index in top2.indices[:, 0].tolist()]
    margins = (top2.values[:, 0] - top2.values[:, 1]).tolist()
    return modes, [float(value) for value in margins]


def _choose_margin_threshold(examples: list[dict[str, object]], evaluator, candidate_names: list[str]) -> float:
    evaluator_modes, margins = _score_predictions(examples, evaluator, candidate_names)
    candidate_thresholds = sorted(set([0.0, *margins, float("inf")]))
    best_threshold = float("inf")
    best_loss = _mean_policy_loss(examples, ["learned"] * len(examples))
    for threshold in candidate_thresholds:
        selected = [
            mode if margin >= threshold else "learned"
            for mode, margin in zip(evaluator_modes, margins, strict=True)
        ]
        loss = _mean_policy_loss(examples, selected)
        if loss < best_loss - 1e-6 or (abs(loss - best_loss) <= 1e-6 and threshold > best_threshold):
            best_loss = loss
            best_threshold = threshold
    return best_threshold


def _choose_robust_margin_threshold(examples: list[dict[str, object]], evaluator, candidate_names: list[str]) -> float:
    evaluator_modes, margins = _score_predictions(examples, evaluator, candidate_names)
    candidate_thresholds = sorted(set([0.0, *margins, float("inf")]))
    source_seeds = sorted({int(example.get("source_seed", -1)) for example in examples})
    static_loss_by_seed = {
        seed: _mean_policy_loss(
            [example for example in examples if int(example.get("source_seed", -1)) == seed],
            ["learned"] * sum(int(example.get("source_seed", -1)) == seed for example in examples),
        )
        for seed in source_seeds
    }
    best_threshold = float("inf")
    best_mean_loss = _mean_policy_loss(examples, ["learned"] * len(examples))
    for threshold in candidate_thresholds:
        selected = [
            mode if margin >= threshold else "learned"
            for mode, margin in zip(evaluator_modes, margins, strict=True)
        ]
        loss_by_seed = {}
        for seed in source_seeds:
            seed_rows = []
            seed_modes = []
            for example, mode in zip(examples, selected, strict=True):
                if int(example.get("source_seed", -1)) == seed:
                    seed_rows.append(example)
                    seed_modes.append(mode)
            loss_by_seed[seed] = _mean_policy_loss(seed_rows, seed_modes)
        if any(loss_by_seed[seed] > static_loss_by_seed[seed] + 1e-6 for seed in source_seeds):
            continue
        mean_loss = _mean_policy_loss(examples, selected)
        if mean_loss < best_mean_loss - 1e-6 or (abs(mean_loss - best_mean_loss) <= 1e-6 and threshold > best_threshold):
            best_mean_loss = mean_loss
            best_threshold = threshold
    return best_threshold


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    evaluator,
    candidate_names: list[str],
    threshold: float,
    robust_threshold: float,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    evaluator_modes, test_margins = _score_predictions(test_examples, evaluator, candidate_names)
    conservative_modes = [
        mode if margin >= threshold else "learned"
        for mode, margin in zip(evaluator_modes, test_margins, strict=True)
    ]
    robust_modes = [
        mode if margin >= robust_threshold else "learned"
        for mode, margin in zip(evaluator_modes, test_margins, strict=True)
    ]
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    train_evaluator_modes, train_margins = _score_predictions(train_examples, evaluator, candidate_names)
    train_conservative_modes = [
        mode if margin >= threshold else "learned"
        for mode, margin in zip(train_evaluator_modes, train_margins, strict=True)
    ]
    train_robust_modes = [
        mode if margin >= robust_threshold else "learned"
        for mode, margin in zip(train_evaluator_modes, train_margins, strict=True)
    ]
    rows = [
        _policy_row(test_examples, "static_learned_interaction", static_modes, candidate_names),
        _policy_row(test_examples, "set_evaluator", evaluator_modes, candidate_names),
        _policy_row(test_examples, "conservative_margin_gate", conservative_modes, candidate_names),
        _policy_row(test_examples, "robust_per_seed_margin_gate", robust_modes, candidate_names),
        _policy_row(test_examples, "generated_plus_composition_oracle", oracle_modes, candidate_names),
    ]
    train_static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    train_evaluator_loss = _mean_policy_loss(train_examples, train_evaluator_modes)
    train_conservative_loss = _mean_policy_loss(train_examples, train_conservative_modes)
    train_robust_loss = _mean_policy_loss(train_examples, train_robust_modes)
    for row in rows:
        row["train_static_learned_loss"] = round(train_static_loss, 6)
        row["train_evaluator_loss"] = round(train_evaluator_loss, 6)
        row["train_conservative_loss"] = round(train_conservative_loss, 6)
        row["train_robust_loss"] = round(train_robust_loss, 6)
        row["test_evaluator_use_rate"] = round(mean(float(mode != "learned") for mode in conservative_modes), 6)
        row["test_robust_use_rate"] = round(mean(float(mode != "learned") for mode in robust_modes), 6)
        row["test_mean_margin"] = round(mean(test_margins), 6)
    return rows


if __name__ == "__main__":
    main()
