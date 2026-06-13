from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import _candidate_features, _candidate_ids, _selected_pair_density  # noqa: E402
from scripts.retriever_cross_seed_composition_regret_probe import (  # noqa: E402
    _composition_selected_ids,
    _train_composition_prior,
)
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _evaluate_policy,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _context_features,
    _mean_policy_loss,
    _policy_row,
    _selected_geometry_features,
    _train_set_evaluator,
)
from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    _selected_object_tensor,
    _write_csv,
)
from scripts.retriever_regret_distillation_probe import ObjectRegretRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a cross-seed learned candidate generator trained from "
            "downstream-regret object membership for WPU v2 priority-1."
        )
    )
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--learned-generated-candidates", type=int, default=4)
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
    parser.add_argument("--regret-positive-temperature", type=float, default=0.05)
    parser.add_argument("--regret-gain-weight", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_joint_candidate_generator.csv"))
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
    learned_candidate_names = [
        f"learned_generated_{index}" for index in range(args.learned_generated_candidates)
    ]
    candidate_names = [*base_candidate_names, *learned_candidate_names, *COMPOSITION_MODES]
    contexts = {}
    for seed in args.seeds:
        print(f"joint-generator collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, base_candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, base_candidate_names, args)
        generator = _train_weighted_generator(train_contexts, base_candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, base_candidate_names, args)
        train_examples = []
        for context in train_contexts:
            train_examples.extend(
                _collect_examples(
                    context,
                    object_retriever,
                    generator,
                    composition_prior,
                    candidate_names,
                    learned_candidate_names,
                    args,
                    split="validation",
                )
            )
        test_examples = _collect_examples(
            heldout,
            object_retriever,
            generator,
            composition_prior,
            candidate_names,
            learned_candidate_names,
            args,
            split="test",
        )
        evaluator = _train_set_evaluator(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, evaluator, candidate_names, learned_candidate_names)
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
                    "learned_generated_candidates": args.learned_generated_candidates,
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "regret_retriever_steps": args.regret_retriever_steps,
                    "composition_steps": args.composition_steps,
                    "set_evaluator_steps": args.set_evaluator_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                }
            )
        rows.extend(condition_rows)
    return rows


def _train_weighted_generator(
    contexts: list[dict[str, object]],
    base_candidate_names: list[str],
    args: argparse.Namespace,
) -> ObjectRegretRetriever:
    features: list[torch.Tensor] = []
    labels: list[float] = []
    weights: list[float] = []
    for context in contexts:
        samples = context["validation_samples"]
        candidates = context["validation_candidates"]
        candidate_losses = context["validation_losses"]
        for sample_index, sample in enumerate(samples):  # type: ignore[arg-type]
            learned_loss = float(candidate_losses["learned"][0][sample_index])  # type: ignore[index]
            losses = {name: float(candidate_losses[name][0][sample_index]) for name in base_candidate_names}  # type: ignore[index]
            best_name = min(base_candidate_names, key=lambda name: (losses[name], name))
            best_loss = losses[best_name]
            best_ids = set(candidates[best_name][sample_index])  # type: ignore[index]
            soft_gain = max(0.0, learned_loss - best_loss)
            target = sample.event.target
            for object_id in _candidate_ids(sample.state, sample.event):
                if object_id == target:
                    continue
                features.append(_candidate_features(sample.state, sample.event, object_id))
                labels.append(float(object_id in best_ids))
                weights.append(1.0 + args.regret_gain_weight * soft_gain)
    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    generator = ObjectRegretRetriever(args.retriever_hidden_dim)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.retriever_lr)
    positive = (label_tensor * weight_tensor).sum().clamp_min(1.0)
    negative = ((1.0 - label_tensor) * weight_tensor).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(0.25, 8.0)
    generator.train()
    for _ in range(args.regret_retriever_steps):
        logits = generator(feature_tensor)
        bce = F.binary_cross_entropy_with_logits(logits, label_tensor, pos_weight=pos_weight, reduction="none")
        loss = (bce * weight_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return generator.eval()


def _collect_examples(
    context: dict[str, object],
    object_retriever: torch.nn.Module,
    generator: ObjectRegretRetriever,
    composition_prior: torch.nn.Module,
    candidate_names: list[str],
    learned_candidate_names: list[str],
    args: argparse.Namespace,
    *,
    split: str,
) -> list[dict[str, object]]:
    samples = context[f"{split}_samples"]
    candidates = {name: list(selected) for name, selected in context[f"{split}_candidates"].items()}  # type: ignore[union-attr]
    model = context["model"]
    device = context["device"]
    losses = {name: tuple(values) for name, values in context[f"{split}_losses"].items()}  # type: ignore[union-attr]
    split_offset = 5_000 if split == "validation" else 10_000

    for learned_index, name in enumerate(learned_candidate_names):
        selected = [
            _learned_generated_ids(
                sample,
                args.budget,
                generator,
                variant=learned_index,
                seed=split_offset + sample_index,
            )
            for sample_index, sample in enumerate(samples)  # type: ignore[arg-type]
        ]
        candidates[name] = selected
        losses[name] = _evaluate_policy(model, samples, selected, args, device)  # type: ignore[arg-type]

    composition_selected = {
        "composition_argmax": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="argmax")
            for sample in samples  # type: ignore[arg-type]
        ],
        "composition_expected": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="expected")
            for sample in samples  # type: ignore[arg-type]
        ],
        "composition_count_only": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="count_only")
            for sample in samples  # type: ignore[arg-type]
        ],
    }
    for name, selected in composition_selected.items():
        candidates[name] = selected
        losses[name] = _evaluate_policy(model, samples, selected, args, device)  # type: ignore[arg-type]

    examples = []
    for sample_index, sample in enumerate(samples):  # type: ignore[arg-type]
        row: dict[str, object] = {}
        candidate_losses = {name: losses[name][0][sample_index] for name in candidate_names}
        best_name = min(candidate_names, key=lambda name: (candidate_losses[name], name))
        row["best_mode"] = best_name
        row["oracle_loss"] = round(candidate_losses[best_name], 6)
        row["oracle_correct"] = losses[best_name][1][sample_index]
        object_features = []
        object_masks = []
        context_features = []
        for candidate_index, name in enumerate(candidate_names):
            selected_ids = candidates[name][sample_index]
            row[f"{name}_loss"] = round(candidate_losses[name], 6)
            row[f"{name}_correct"] = losses[name][1][sample_index]
            row[f"{name}_selected_hand"] = int("hand_001" in selected_ids)
            row[f"{name}_selected_obstacles"] = sum(object_id.startswith("obstacle_") for object_id in selected_ids)
            row[f"{name}_pair_density"] = round(
                _selected_pair_density(
                    sample.state,
                    [object_id for object_id in selected_ids if object_id.startswith("obstacle_")],
                ),
                6,
            )
            for feature_name, value in _selected_geometry_features(sample, selected_ids).items():
                row[f"{name}_{feature_name}"] = round(value, 6)
            object_tensor, mask_tensor = _selected_object_tensor(sample, selected_ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            context_features.append(_context_features(row, name, candidate_index, len(candidate_names), args))
        row["object_features"] = torch.stack(object_features)
        row["object_masks"] = torch.stack(object_masks)
        row["context_features"] = torch.tensor(context_features, dtype=torch.float32)
        examples.append(row)
    return examples


def _learned_generated_ids(
    sample,
    budget: int,
    generator: ObjectRegretRetriever,
    *,
    variant: int,
    seed: int,
) -> list[str]:
    target = sample.event.target
    object_ids = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not object_ids:
        return [target]
    features = torch.stack([_candidate_features(sample.state, sample.event, object_id) for object_id in object_ids])
    with torch.no_grad():
        scores = generator(features).float()
    rng = random.Random(seed * 8191 + variant * 131071)
    selected = [target]
    hand_ids = [object_id for object_id in object_ids if sample.state.objects[object_id].type == "robot_hand"]
    if hand_ids and variant in {0, 2}:
        selected.append(hand_ids[0])
    adjusted: list[tuple[float, str]] = []
    for index, object_id in enumerate(object_ids):
        feature = features[index]
        score = float(scores[index].item())
        if variant == 1:
            score += 0.25 * float(feature[7]) - 0.05 * float(feature[6])
        elif variant == 2:
            score += 0.15 * float(feature[1]) + 0.15 * float(feature[0])
        elif variant == 3:
            score += rng.random() * 0.35
        adjusted.append((score, object_id))
    for _, object_id in sorted(adjusted, reverse=True):
        if object_id not in selected:
            selected.append(object_id)
        if len(selected) >= budget:
            break
    return selected[:budget]


def _predict_modes(examples: list[dict[str, object]], evaluator: torch.nn.Module, candidate_names: list[str]) -> list[str]:
    from scripts.retriever_cross_seed_set_evaluator_probe import _example_tensors

    objects, masks, context, _ = _example_tensors(examples, candidate_names)
    with torch.no_grad():
        scores = evaluator(objects, masks, context)
    return [candidate_names[int(index)] for index in scores.argmax(dim=1).tolist()]


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    evaluator: torch.nn.Module,
    candidate_names: list[str],
    learned_candidate_names: list[str],
) -> list[dict[str, object]]:
    static_learned = ["learned"] * len(test_examples)
    evaluator_modes = _predict_modes(test_examples, evaluator, candidate_names)
    train_evaluator_modes = _predict_modes(train_examples, evaluator, candidate_names)
    train_evaluator_loss = _mean_policy_loss(train_examples, train_evaluator_modes)
    train_static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    all_oracle_modes = [str(example["best_mode"]) for example in test_examples]
    learned_oracle_modes = [
        min(learned_candidate_names, key=lambda name: (float(example[f"{name}_loss"]), name))
        for example in test_examples
    ]
    rows = [
        _policy_row(test_examples, "static_learned_interaction", static_learned, candidate_names),
        _policy_row(test_examples, "joint_candidate_generator_evaluator", evaluator_modes, candidate_names),
        _policy_row(test_examples, "learned_generated_oracle", learned_oracle_modes, candidate_names),
        _policy_row(test_examples, "generated_plus_composition_oracle", all_oracle_modes, candidate_names),
    ]
    for row in rows:
        row["train_evaluator_loss"] = round(train_evaluator_loss, 6)
        row["train_static_learned_loss"] = round(train_static_loss, 6)
        row["selected_learned_generated_rate"] = round(
            sum(
                float(str(mode).startswith("learned_generated_"))
                for mode in (
                    evaluator_modes
                    if row["policy"] == "joint_candidate_generator_evaluator"
                    else learned_oracle_modes
                    if row["policy"] == "learned_generated_oracle"
                    else all_oracle_modes
                    if row["policy"] == "generated_plus_composition_oracle"
                    else static_learned
                )
            )
            / max(len(test_examples), 1),
            6,
        )
    return rows


if __name__ == "__main__":
    main()
