from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import _train_cross_seed_regret_retriever  # noqa: E402
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _collect_seed_context,
    _example_tensors,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, GeneratedSetReranker, _write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an end-to-end candidate selector trained on downstream "
            "propagation loss and no-harm mass for WPU v2 priority-1."
        )
    )
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
    parser.add_argument("--selector-steps", type=int, default=900)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--selector-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--selector-lr", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--imitation-weight", type=float, default=0.15)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--no-harm-weight", type=float, default=1.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--group-dro-weight", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--prob-margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--min-accept-probs", type=float, nargs="+", default=[0.0, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_end_to_end_candidate_selector.csv"))
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
        print(f"end-to-end-selector collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
            examples = _collect_examples(context, object_retriever, composition_prior, candidate_names, args, split="validation")
            for example in examples:
                example["source_seed"] = seed
            train_examples.extend(examples)
        test_examples = _collect_examples(heldout, object_retriever, composition_prior, candidate_names, args, split="test")
        selector = _train_selector(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, selector, candidate_names, args)
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
                    "selector_steps": args.selector_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "no_harm_weight": args.no_harm_weight,
                    "utility_weight": args.utility_weight,
                    "imitation_weight": args.imitation_weight,
                    "temperature": args.temperature,
                }
            )
        rows.extend(condition_rows)
    return rows


def _train_selector(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> GeneratedSetReranker:
    objects, masks, context, losses = _example_tensors(examples, candidate_names)
    learned_index = candidate_names.index("learned")
    baseline = losses[:, learned_index : learned_index + 1]
    excess = losses - baseline
    centered = excess - excess.mean(dim=1, keepdim=True)
    scale = excess.std(dim=1, keepdim=True).clamp_min(1e-3)
    normalized_excess = centered / scale
    best_indices = losses.argmin(dim=1)
    source_seeds = torch.tensor([int(example["source_seed"]) for example in examples], dtype=torch.long)
    unique_seeds = torch.unique(source_seeds)
    selector = GeneratedSetReranker(objects.size(-1), context.size(-1), args.selector_hidden_dim)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.selector_lr)
    selector.train()
    for _ in range(args.selector_steps):
        scores = selector(objects, masks, context)
        probabilities = F.softmax(scores / max(args.temperature, 1e-4), dim=1)
        group_losses = []
        for seed in unique_seeds:
            mask = source_seeds == seed
            seed_probs = probabilities[mask]
            seed_excess = excess[mask]
            seed_normalized = normalized_excess[mask]
            seed_scores = scores[mask]
            expected_loss = (seed_probs * seed_normalized).sum(dim=1).mean()
            harmful_mass = (seed_probs * F.relu(seed_excess + args.no_harm_margin)).sum(dim=1).mean()
            imitation = F.cross_entropy(seed_scores, best_indices[mask])
            entropy = -(seed_probs * seed_probs.clamp_min(1e-8).log()).sum(dim=1).mean()
            group_losses.append(
                args.utility_weight * expected_loss
                + args.no_harm_weight * harmful_mass
                + args.imitation_weight * imitation
                - args.entropy_weight * entropy
            )
        group_loss = torch.stack(group_losses)
        loss = group_loss.mean() + args.group_dro_weight * group_loss.max()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return selector.eval()


def _predict_modes(
    examples: list[dict[str, object]],
    selector: GeneratedSetReranker,
    candidate_names: list[str],
    *,
    prob_margin: float,
    min_accept_prob: float,
    temperature: float,
) -> tuple[list[str], dict[str, float]]:
    objects, masks, context, losses = _example_tensors(examples, candidate_names)
    learned_index = candidate_names.index("learned")
    with torch.no_grad():
        scores = selector(objects, masks, context)
        probabilities = F.softmax(scores / max(temperature, 1e-4), dim=1)
    selected_indices = scores.argmax(dim=1)
    selected_modes: list[str] = []
    accepted: list[float] = []
    harmful: list[float] = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        selected_probability = float(probabilities[row_index, candidate_index].item())
        learned_probability = float(probabilities[row_index, learned_index].item())
        use_candidate = (
            candidate_index != learned_index
            and selected_probability >= min_accept_prob
            and selected_probability - learned_probability >= prob_margin
        )
        selected_modes.append(candidate_names[candidate_index] if use_candidate else "learned")
        accepted.append(float(use_candidate))
        harmful.append(float(use_candidate and losses[row_index, candidate_index] > losses[row_index, learned_index]))
    return selected_modes, {
        "accept_rate": round(mean(accepted), 6),
        "harmful_accept_rate": round(mean(harmful), 6),
        "selected_probability_mean": round(float(probabilities.max(dim=1).values.mean().item()), 6),
        "learned_probability_mean": round(float(probabilities[:, learned_index].mean().item()), 6),
        "deployment_prob_margin": round(float(prob_margin), 6),
        "deployment_min_accept_prob": round(float(min_accept_prob), 6),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    selector: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    default_modes, default_metrics = _predict_modes(
        test_examples,
        selector,
        candidate_names,
        prob_margin=0.0,
        min_accept_prob=0.0,
        temperature=args.temperature,
    )
    train_default_modes, train_default_metrics = _predict_modes(
        train_examples,
        selector,
        candidate_names,
        prob_margin=0.0,
        min_accept_prob=0.0,
        temperature=args.temperature,
    )
    configs = _deployment_configs(args)
    selected_config, selected_train_metrics = _select_train_deployment(
        train_examples,
        selector,
        candidate_names,
        args,
        configs,
    )
    selected_modes, selected_metrics = _predict_modes(
        test_examples,
        selector,
        candidate_names,
        prob_margin=float(selected_config["prob_margin"]),
        min_accept_prob=float(selected_config["min_accept_prob"]),
        temperature=args.temperature,
    )
    selected_metrics.update(
        {
            "selection_train_loss": selected_train_metrics["loss"],
            "selection_train_gap_closure": selected_train_metrics["gap_closure"],
            "selection_train_accept_rate": selected_train_metrics["accept_rate"],
            "selection_train_harmful_accept_rate": selected_train_metrics["harmful_accept_rate"],
            "selection_train_policy": selected_config["name"],
        }
    )
    policies = [
        ("static_learned_interaction", static_modes, {}),
        ("end_to_end_candidate_selector", default_modes, default_metrics),
        ("train_selected_end_to_end_candidate_selector", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in configs:
        if config["name"] == "end_to_end_candidate_selector":
            continue
        modes, metrics = _predict_modes(
            test_examples,
            selector,
            candidate_names,
            prob_margin=float(config["prob_margin"]),
            min_accept_prob=float(config["min_accept_prob"]),
            temperature=args.temperature,
        )
        policies.append((str(config["name"]), modes, metrics))

    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_end_to_end_selector_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_end_to_end_selector_accept_rate"] = train_default_metrics["accept_rate"]
        rows.append(row)
    return rows


def _deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = [
        {
            "name": "end_to_end_candidate_selector",
            "prob_margin": 0.0,
            "min_accept_prob": 0.0,
        }
    ]
    for prob_margin in _unique_floats(args.prob_margins):
        for min_accept_prob in _unique_floats(args.min_accept_probs):
            name = f"end_to_end_selector_pg{_float_token(prob_margin)}_pmin{_float_token(min_accept_prob)}"
            configs.append({"name": name, "prob_margin": prob_margin, "min_accept_prob": min_accept_prob})
    return configs


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    selector: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
    configs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, float]]:
    scored = []
    for config in configs:
        modes, metrics = _predict_modes(
            train_examples,
            selector,
            candidate_names,
            prob_margin=float(config["prob_margin"]),
            min_accept_prob=float(config["min_accept_prob"]),
            temperature=args.temperature,
        )
        policy_loss = _mean_policy_loss(train_examples, modes)
        static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
        oracle_loss = _mean_policy_loss(train_examples, [str(example["best_mode"]) for example in train_examples])
        oracle_gap = max(static_loss - oracle_loss, 1e-8)
        gap_closure = (static_loss - policy_loss) / oracle_gap
        metrics.update({"loss": round(policy_loss, 6), "gap_closure": round(gap_closure, 6)})
        scored.append((config, metrics))
    feasible = [
        item for item in scored if item[1]["harmful_accept_rate"] <= args.selection_harmful_limit and item[1]["gap_closure"] > 0.0
    ]
    if feasible:
        return max(feasible, key=lambda item: (item[1]["gap_closure"], -item[1]["harmful_accept_rate"]))  # type: ignore[return-value]
    return min(scored, key=lambda item: (item[1]["harmful_accept_rate"], item[1]["loss"]))  # type: ignore[return-value]


def _unique_floats(values: list[float]) -> list[float]:
    return sorted({round(float(value), 8) for value in values})


def _float_token(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
