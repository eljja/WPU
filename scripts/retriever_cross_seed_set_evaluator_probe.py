from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import FEATURE_DIM as OBJECT_DIM, _selected_pair_density  # noqa: E402
from scripts.retriever_generated_candidate_probe import BASE_MODES, GeneratedSetReranker, _selected_object_tensor, _write_csv  # noqa: E402
from scripts.retriever_cross_seed_composition_regret_probe import (  # noqa: E402
    _composition_selected_ids,
    _train_composition_prior,
)
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _evaluate_policy,
    _train_cross_seed_regret_retriever,
)


COMPOSITION_MODES = ("composition_argmax", "composition_expected", "composition_count_only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-seed candidate-set evaluation after composition-aware retrieval.")
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_set_evaluator.csv"))
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
        print(f"set-evaluator collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, base_candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, base_candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, base_candidate_names, args)
        train_examples = [
            example
            for context in train_contexts
            for example in _collect_examples(context, object_retriever, composition_prior, candidate_names, args, split="validation")
        ]
        test_examples = _collect_examples(heldout, object_retriever, composition_prior, candidate_names, args, split="test")
        evaluator = _train_set_evaluator(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, evaluator, candidate_names)
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
                }
            )
        rows.extend(condition_rows)
    return rows


def _collect_examples(
    context: dict[str, object],
    object_retriever: nn.Module,
    composition_prior: nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
    *,
    split: str,
) -> list[dict[str, object]]:
    samples = context[f"{split}_samples"]
    candidates = {name: list(selected) for name, selected in context[f"{split}_candidates"].items()}  # type: ignore[union-attr]
    if split == "validation":
        model = context["model"]
        device = context["device"]
        losses = {name: tuple(values) for name, values in context["validation_losses"].items()}  # type: ignore[union-attr]
    else:
        model = context["model"]
        device = context["device"]
        losses = {name: tuple(values) for name, values in context["test_losses"].items()}  # type: ignore[union-attr]

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
            geometry = _selected_geometry_features(sample, selected_ids)
            for feature_name, value in geometry.items():
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


def _context_features(
    row: dict[str, object],
    name: str,
    candidate_index: int,
    candidate_count: int,
    args: argparse.Namespace,
) -> list[float]:
    one_hot = [0.0 for _ in range(candidate_count)]
    one_hot[candidate_index] = 1.0
    selected_hand = float(row[f"{name}_selected_hand"])
    selected_obstacles = float(row[f"{name}_selected_obstacles"])
    pair_density = float(row[f"{name}_pair_density"])
    obstacle_ratio = selected_obstacles / max(float(args.budget), 1.0)
    return [
        *one_hot,
        selected_hand,
        obstacle_ratio,
        pair_density,
        selected_hand * pair_density,
        obstacle_ratio * pair_density,
        float(row[f"{name}_event_force"]),
        float(row[f"{name}_obstacle_distance_min"]),
        float(row[f"{name}_obstacle_distance_mean"]),
        float(row[f"{name}_obstacle_distance_max"]),
        float(row[f"{name}_obstacle_distance_span"]),
        float(row[f"{name}_obstacle_abs_y_mean"]),
        float(row[f"{name}_obstacle_abs_y_max"]),
        float(row[f"{name}_obstacle_axis_ratio"]),
        float(row[f"{name}_hand_distance"]),
        float(row[f"{name}_edge_distance"]),
        float(name.startswith("generated_")),
        float(name.startswith("composition_")),
    ]


def _selected_geometry_features(sample, selected_ids: list[str]) -> dict[str, float]:
    target_xy = _object_xy(sample.state, sample.event.target)
    obstacle_offsets: list[tuple[float, float]] = []
    for object_id in selected_ids:
        if object_id.startswith("obstacle_") and object_id in sample.state.objects:
            object_xy = _object_xy(sample.state, object_id)
            obstacle_offsets.append((object_xy[0] - target_xy[0], object_xy[1] - target_xy[1]))
    distances = [(x * x + y * y) ** 0.5 for x, y in obstacle_offsets] or [1.0]
    abs_y = [abs(y) for _, y in obstacle_offsets] or [1.0]
    axis_ratio = mean(float(abs(x) < 0.05) for x, _ in obstacle_offsets) if obstacle_offsets else 0.0
    hand_distance = (
        _distance_xy(target_xy, _object_xy(sample.state, "hand_001"))
        if "hand_001" in selected_ids and "hand_001" in sample.state.objects
        else 1.0
    )
    edge_distance = (
        _distance_xy(target_xy, _object_xy(sample.state, "edge_001"))
        if "edge_001" in selected_ids and "edge_001" in sample.state.objects
        else 1.0
    )
    return {
        "event_force": float(sample.event.delta.get("force", 0.0)),
        "obstacle_distance_min": min(distances),
        "obstacle_distance_mean": mean(distances),
        "obstacle_distance_max": max(distances),
        "obstacle_distance_span": max(distances) - min(distances),
        "obstacle_abs_y_mean": mean(abs_y),
        "obstacle_abs_y_max": max(abs_y),
        "obstacle_axis_ratio": axis_ratio,
        "hand_distance": hand_distance,
        "edge_distance": edge_distance,
    }


def _object_xy(state, object_id: str) -> tuple[float, float]:
    position = state.objects[object_id].attributes.get("position", [0.0, 0.0, 0.0])
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return 0.0, 0.0
    return float(position[0]), float(position[1])


def _distance_xy(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _train_set_evaluator(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> GeneratedSetReranker:
    objects, masks, context, losses = _example_tensors(examples, candidate_names)
    centered_loss = losses - losses.mean(dim=1, keepdim=True)
    scale = losses.std(dim=1, keepdim=True).clamp_min(1e-3)
    utilities = -(centered_loss / scale)
    evaluator = GeneratedSetReranker(OBJECT_DIM, context.size(-1), args.set_evaluator_hidden_dim)
    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=args.set_evaluator_lr)
    targets = utilities.argmax(dim=1)
    soft_targets = F.softmax(utilities / args.utility_temperature, dim=1)
    evaluator.train()
    for _ in range(args.set_evaluator_steps):
        scores = evaluator(objects, masks, context)
        log_probs = F.log_softmax(scores, dim=1)
        loss = (
            F.cross_entropy(scores, targets)
            + 0.5 * -(soft_targets * log_probs).sum(dim=1).mean()
            + args.normalized_utility_weight * F.mse_loss(scores, utilities)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluator.eval()


def _example_tensors(
    examples: list[dict[str, object]],
    candidate_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objects = torch.stack([example["object_features"] for example in examples])  # type: ignore[list-item]
    masks = torch.stack([example["object_masks"] for example in examples])  # type: ignore[list-item]
    context = torch.stack([example["context_features"] for example in examples])  # type: ignore[list-item]
    losses = torch.tensor(
        [[float(example[f"{name}_loss"]) for name in candidate_names] for example in examples],
        dtype=torch.float32,
    )
    return objects, masks, context, losses


def _predict_modes(examples: list[dict[str, object]], evaluator: GeneratedSetReranker, candidate_names: list[str]) -> list[str]:
    objects, masks, context, _ = _example_tensors(examples, candidate_names)
    with torch.no_grad():
        scores = evaluator(objects, masks, context)
    return [candidate_names[int(index)] for index in scores.argmax(dim=1).tolist()]


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    evaluator: GeneratedSetReranker,
    candidate_names: list[str],
) -> list[dict[str, object]]:
    static_learned = ["learned"] * len(test_examples)
    evaluator_modes = _predict_modes(test_examples, evaluator, candidate_names)
    train_evaluator_modes = _predict_modes(train_examples, evaluator, candidate_names)
    train_evaluator_loss = _mean_policy_loss(train_examples, train_evaluator_modes)
    train_static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    rows = [
        _policy_row(test_examples, "static_learned_interaction", static_learned, candidate_names),
        _policy_row(test_examples, "set_evaluator", evaluator_modes, candidate_names),
        _policy_row(test_examples, "generated_plus_composition_oracle", oracle_modes, candidate_names),
    ]
    for row in rows:
        row["train_evaluator_loss"] = round(train_evaluator_loss, 6)
        row["train_static_learned_loss"] = round(train_static_loss, 6)
    return rows


def _mean_policy_loss(rows: list[dict[str, object]], selected_modes: list[str]) -> float:
    return mean(float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True))


def _policy_row(
    rows: list[dict[str, object]],
    policy: str,
    selected_modes: list[str],
    candidate_names: list[str],
) -> dict[str, object]:
    losses = [float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True)]
    correct = [float(row[f"{mode}_correct"]) for row, mode in zip(rows, selected_modes, strict=True)]
    oracle_losses = [float(row["oracle_loss"]) for row in rows]
    output = {
        "policy": policy,
        "loss": round(mean(losses), 6),
        "accuracy": round(mean(correct), 6),
        "candidate_oracle_loss": round(mean(oracle_losses), 6),
        "excess_over_candidate_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "oracle_match_rate": round(mean(float(mode == str(row["best_mode"])) for row, mode in zip(rows, selected_modes, strict=True)), 6),
        "selected_generated_rate": round(mean(float(mode.startswith("generated_")) for mode in selected_modes), 6),
        "selected_composition_rate": round(mean(float(mode.startswith("composition_")) for mode in selected_modes), 6),
    }
    for name in candidate_names:
        output[f"selected_{name}_rate"] = round(sum(mode == name for mode in selected_modes) / max(len(selected_modes), 1), 6)
    return output


if __name__ == "__main__":
    main()
