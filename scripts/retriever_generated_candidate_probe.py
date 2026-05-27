from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM as OBJECT_CANDIDATE_FEATURE_DIM,
    _candidate_features,
    _candidate_ids,
    _selected_ids,
    _selected_pair_density,
    _train_model as _train_retriever,
)
from scripts.retriever_regret_oracle_probe import _evaluate_selected  # noqa: E402
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


BASE_MODES = ("indexed", "proximity", "interaction", "learned")


class GeneratedSetReranker(nn.Module):
    def __init__(self, object_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(object_dim),
            nn.Linear(object_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + context_dim),
            nn.Linear(hidden_dim * 2 + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, objects: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, candidate_count, budget, object_dim = objects.shape
        encoded = self.object_encoder(objects.view(batch_size * candidate_count * budget, object_dim))
        encoded = encoded.view(batch_size, candidate_count, budget, -1)
        float_mask = mask.unsqueeze(-1).float()
        pooled_mean = (encoded * float_mask).sum(dim=2) / float_mask.sum(dim=2).clamp_min(1.0)
        pooled_max = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=2)
        return self.scorer(torch.cat([pooled_mean, pooled_max, context], dim=-1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stochastic candidate generation for WPU object-set retrieval.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--reranker-steps", type=int, default=600)
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-lr", type=float, default=3e-3)
    parser.add_argument("--safe-margin", type=float, default=0.005)
    parser.add_argument("--utility-temperature", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_generated_candidates.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"generated-candidate-reranker seed={seed} N={n_value} K={k_value}", flush=True)
                rows.extend(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    train_dataset = WorkingSetPhysicsDataset(
        size=max(args.propagation_steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    retriever = _train_retriever(
        [train_dataset[index] for index in range(len(train_dataset))],
        args.budget,
        args.retriever_steps,
        args.retriever_hidden_dim,
        args.retriever_lr,
    )
    model = create_model(
        args.model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.budget,
    ).to(device)
    train_args = argparse.Namespace(**vars(args), working_set_size=args.budget, selection_mode="interaction")
    class_weights = _class_weights(train_dataset).to(device) if args.class_weights else None
    _train_propagation(model, train_dataset, class_weights, train_args, device)
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
    candidate_names = [*BASE_MODES, *[f"generated_{index}" for index in range(args.generated_candidates)]]
    validation_examples = _collect_examples(
        model,
        background_objects,
        causal_obstacles,
        seed + 5_000,
        args.validation_samples,
        args,
        retriever,
        device,
        total_n,
        causal_k,
        candidate_names,
    )
    test_examples = _collect_examples(
        model,
        background_objects,
        causal_obstacles,
        seed + 10_000,
        args.samples,
        args,
        retriever,
        device,
        total_n,
        causal_k,
        candidate_names,
    )
    reranker = _train_reranker(validation_examples, args, len(candidate_names))
    rows = _summarize(test_examples, validation_examples, reranker, candidate_names, args)
    for row in rows:
        row.update(
            {
                "seed": seed,
                "total_objects_n": total_n,
                "causal_k": causal_k,
                "budget": args.budget,
                "generated_candidates": args.generated_candidates,
                "interaction_mode": args.interaction_mode,
                "propagation_steps": args.propagation_steps,
                "retriever_steps": args.retriever_steps,
                "reranker_steps": args.reranker_steps,
                "validation_samples": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return rows


def _collect_examples(
    model: torch.nn.Module,
    background_objects: int,
    causal_obstacles: int,
    dataset_seed: int,
    sample_count: int,
    args: argparse.Namespace,
    retriever: torch.nn.Module,
    device: torch.device,
    total_n: int,
    causal_k: int,
    candidate_names: list[str],
) -> list[dict[str, object]]:
    dataset = WorkingSetPhysicsDataset(
        size=sample_count,
        seed=dataset_seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    selected_by_candidate: dict[str, list[list[str]]] = {name: [] for name in candidate_names}
    for sample_index, sample in enumerate(samples):
        for mode in BASE_MODES:
            selected_by_candidate[mode].append(
                _selected_ids(sample, mode, args.budget, retriever if mode == "learned" else None)
            )
        generated = _generated_candidates(sample, args.budget, args.generated_candidates, seed=dataset_seed + sample_index)
        for index, selected_ids in enumerate(generated):
            selected_by_candidate[f"generated_{index}"].append(selected_ids)

    losses_by_candidate: dict[str, list[float]] = {}
    correct_by_candidate: dict[str, list[int]] = {}
    for name in candidate_names:
        losses, correct = _evaluate_selected(model, samples, selected_by_candidate[name], args.batch_size, device)
        losses_by_candidate[name] = losses
        correct_by_candidate[name] = correct

    examples: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        row: dict[str, object] = {}
        candidate_losses = {name: losses_by_candidate[name][sample_index] for name in candidate_names}
        base_losses = {name: candidate_losses[name] for name in BASE_MODES}
        best_candidate = min(candidate_names, key=lambda name: (candidate_losses[name], name))
        best_base = min(BASE_MODES, key=lambda name: (base_losses[name], name))
        row["best_mode"] = best_candidate
        row["best_base_mode"] = best_base
        row["oracle_loss"] = round(candidate_losses[best_candidate], 6)
        row["oracle_correct"] = correct_by_candidate[best_candidate][sample_index]
        row["base_oracle_loss"] = round(base_losses[best_base], 6)
        row["base_oracle_correct"] = correct_by_candidate[best_base][sample_index]
        object_features = []
        object_masks = []
        context_features = []
        for candidate_index, name in enumerate(candidate_names):
            selected_ids = selected_by_candidate[name][sample_index]
            row[f"{name}_loss"] = round(candidate_losses[name], 6)
            row[f"{name}_correct"] = correct_by_candidate[name][sample_index]
            row[f"{name}_selected_hand"] = int("hand_001" in selected_ids)
            row[f"{name}_selected_obstacles"] = sum(object_id.startswith("obstacle_") for object_id in selected_ids)
            row[f"{name}_pair_density"] = round(
                _selected_pair_density(
                    sample.state,
                    [object_id for object_id in selected_ids if object_id.startswith("obstacle_")],
                ),
                6,
            )
            object_tensor, mask_tensor = _selected_object_tensor(sample, selected_ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            context_features.append(
                _context_features(row, name, candidate_index, len(candidate_names), total_n, causal_k, args.budget)
            )
        row["object_features"] = torch.stack(object_features)
        row["object_masks"] = torch.stack(object_masks)
        row["context_features"] = torch.tensor(context_features, dtype=torch.float32)
        examples.append(row)
    return examples


def _generated_candidates(sample, budget: int, count: int, seed: int) -> list[list[str]]:
    target = sample.event.target
    candidate_ids = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    hand_ids = [object_id for object_id in candidate_ids if sample.state.objects[object_id].type == "robot_hand"]
    anchors = [object_id for object_id in candidate_ids if sample.state.objects[object_id].type in {"table_edge", "table"}]
    obstacles = [object_id for object_id in candidate_ids if sample.state.objects[object_id].type == "obstacle"]
    ranked_obstacles = sorted(
        obstacles,
        key=lambda object_id: (
            -float(_candidate_features(sample.state, sample.event, object_id)[7]),
            float(_candidate_features(sample.state, sample.event, object_id)[6]),
            object_id,
        ),
    )
    generated: list[list[str]] = []
    for index in range(count):
        rng = random.Random(seed * 7919 + index * 104729)
        selected = [target]
        if hand_ids and index % 3 != 1:
            selected.append(hand_ids[0])
        if index % 2 == 0:
            obstacle_pool = ranked_obstacles[: max(4, min(len(ranked_obstacles), budget * 3))]
            rng.shuffle(obstacle_pool)
            selected.extend(obstacle_pool[: max(0, budget - len(selected))])
        else:
            scored = []
            for object_id in candidate_ids:
                features = _candidate_features(sample.state, sample.event, object_id)
                score = (
                    1.5 * float(features[1])
                    + 1.0 * float(features[0])
                    + 2.0 * float(features[7])
                    - 0.2 * float(features[6])
                    + rng.random() * 0.5
                )
                scored.append((score, object_id))
            for _, object_id in sorted(scored, reverse=True):
                if object_id not in selected:
                    selected.append(object_id)
                if len(selected) >= budget:
                    break
        for object_id in [*anchors, *candidate_ids]:
            if len(selected) >= budget:
                break
            if object_id not in selected:
                selected.append(object_id)
        generated.append(selected[:budget])
    return generated


def _selected_object_tensor(sample, selected_ids: list[str], budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.zeros((budget, OBJECT_CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((budget,), dtype=torch.bool)
    for index, object_id in enumerate(selected_ids[:budget]):
        features[index] = _candidate_features(sample.state, sample.event, object_id)
        mask[index] = True
    return features, mask


def _context_features(
    row: dict[str, object],
    name: str,
    candidate_index: int,
    candidate_count: int,
    total_n: int,
    causal_k: int,
    budget: int,
) -> list[float]:
    one_hot = [0.0 for _ in range(candidate_count)]
    one_hot[candidate_index] = 1.0
    selected_hand = float(row[f"{name}_selected_hand"])
    selected_obstacles = float(row[f"{name}_selected_obstacles"])
    pair_density = float(row[f"{name}_pair_density"])
    obstacle_ratio = selected_obstacles / max(float(budget), 1.0)
    is_generated = float(name.startswith("generated_"))
    return [
        *one_hot,
        selected_hand,
        obstacle_ratio,
        pair_density,
        selected_hand * pair_density,
        obstacle_ratio * pair_density,
        causal_k / 64.0,
        budget / 64.0,
        min(total_n / 4096.0, 4.0),
        is_generated,
    ]


def _train_reranker(
    examples: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_count: int,
) -> GeneratedSetReranker:
    objects, masks, context, loss_tensor = _example_tensors(examples, candidate_count)
    utilities = -loss_tensor
    reranker = GeneratedSetReranker(OBJECT_CANDIDATE_FEATURE_DIM, candidate_count + 9, args.reranker_hidden_dim)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.reranker_lr)
    targets = utilities.argmax(dim=1)
    soft_targets = F.softmax(utilities / args.utility_temperature, dim=1)
    reranker.train()
    for _ in range(args.reranker_steps):
        scores = reranker(objects, masks, context)
        log_probs = F.log_softmax(scores, dim=1)
        ce_loss = F.cross_entropy(scores, targets)
        soft_ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        utility_loss = F.mse_loss(scores, utilities)
        loss = ce_loss + 0.5 * soft_ce_loss + 0.1 * utility_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reranker.eval()


def _example_tensors(
    examples: list[dict[str, object]],
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objects = torch.stack([example["object_features"] for example in examples])  # type: ignore[list-item]
    masks = torch.stack([example["object_masks"] for example in examples])  # type: ignore[list-item]
    context = torch.stack([example["context_features"] for example in examples])  # type: ignore[list-item]
    losses = torch.tensor(
        [[float(example[f"{_candidate_names(candidate_count)[index]}_loss"]) for index in range(candidate_count)] for example in examples],
        dtype=torch.float32,
    )
    return objects, masks, context, losses


def _candidate_names(candidate_count: int) -> list[str]:
    generated_count = candidate_count - len(BASE_MODES)
    return [*BASE_MODES, *[f"generated_{index}" for index in range(generated_count)]]


def _summarize(
    test_examples: list[dict[str, object]],
    validation_examples: list[dict[str, object]],
    reranker: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(validation_examples, BASE_MODES)
    generated_static_mode = _best_static_mode(validation_examples, candidate_names)
    summary = [
        _policy_row(test_examples, policy=f"static_{mode}", selected_modes=[mode] * len(test_examples), candidate_names=candidate_names)
        for mode in BASE_MODES
    ]
    summary.append(
        _policy_row(
            test_examples,
            policy="static_base_validation_choice",
            selected_modes=[static_mode] * len(test_examples),
            candidate_names=candidate_names,
        )
    )
    summary.append(
        _policy_row(
            test_examples,
            policy="static_generated_validation_choice",
            selected_modes=[generated_static_mode] * len(test_examples),
            candidate_names=candidate_names,
        )
    )
    reranker_modes = _predict_modes(test_examples, reranker, len(candidate_names))
    validation_reranker_modes = _predict_modes(validation_examples, reranker, len(candidate_names))
    validation_reranker_loss = _mean_policy_loss(validation_examples, validation_reranker_modes)
    validation_static_loss = _mean_policy_loss(validation_examples, [static_mode] * len(validation_examples))
    margin_safe = validation_reranker_loss + args.safe_margin < validation_static_loss
    safe_modes = reranker_modes if margin_safe else [static_mode] * len(test_examples)
    summary.append(
        _policy_row(
            test_examples,
            policy="generated_set_reranker",
            selected_modes=reranker_modes,
            candidate_names=candidate_names,
        )
    )
    safe_row = _policy_row(
        test_examples,
        policy="generated_margin_safe_reranker",
        selected_modes=safe_modes,
        candidate_names=candidate_names,
    )
    safe_row["safe_uses_reranker"] = int(margin_safe)
    summary.append(safe_row)
    summary.append(_oracle_row(test_examples, "base_oracle", [str(example["best_base_mode"]) for example in test_examples], candidate_names))
    summary.append(_oracle_row(test_examples, "generated_oracle", [str(example["best_mode"]) for example in test_examples], candidate_names))
    for row in summary:
        row["static_mode_from_validation"] = static_mode
        row["static_generated_mode_from_validation"] = generated_static_mode
        row["validation_reranker_loss"] = round(validation_reranker_loss, 6)
        row["validation_static_loss"] = round(validation_static_loss, 6)
        row["safe_margin"] = args.safe_margin
        row.setdefault("safe_uses_reranker", "")
    return summary


def _predict_modes(examples: list[dict[str, object]], reranker: GeneratedSetReranker, candidate_count: int) -> list[str]:
    objects, masks, context, _ = _example_tensors(examples, candidate_count)
    with torch.no_grad():
        scores = reranker(objects, masks, context)
    names = _candidate_names(candidate_count)
    return [names[int(index)] for index in scores.argmax(dim=1).tolist()]


def _best_static_mode(rows: list[dict[str, object]], candidate_names) -> str:
    losses = {name: mean(float(row[f"{name}_loss"]) for row in rows) for name in candidate_names}
    return min(candidate_names, key=lambda name: (losses[name], name))


def _mean_policy_loss(rows: list[dict[str, object]], selected_modes: list[str]) -> float:
    return mean(float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True))


def _policy_row(
    rows: list[dict[str, object]],
    *,
    policy: str,
    selected_modes: list[str],
    candidate_names: list[str],
) -> dict[str, object]:
    losses = [float(row[f"{selected}_loss"]) for row, selected in zip(rows, selected_modes, strict=True)]
    correct = [float(row[f"{selected}_correct"]) for row, selected in zip(rows, selected_modes, strict=True)]
    oracle_losses = [float(row["oracle_loss"]) for row in rows]
    oracle_correct = [float(row["oracle_correct"]) for row in rows]
    base_oracle_losses = [float(row["base_oracle_loss"]) for row in rows]
    selected_counts = Counter(selected_modes)
    output = {
        "policy": policy,
        "loss": round(mean(losses), 6),
        "accuracy": round(mean(correct), 6),
        "generated_oracle_loss": round(mean(oracle_losses), 6),
        "generated_oracle_accuracy": round(mean(oracle_correct), 6),
        "base_oracle_loss": round(mean(base_oracle_losses), 6),
        "excess_over_generated_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "excess_over_base_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, base_oracle_losses, strict=True)), 6),
        "generated_oracle_match_rate": round(
            mean(float(selected == str(row["best_mode"])) for row, selected in zip(rows, selected_modes, strict=True)),
            6,
        ),
        "base_oracle_match_rate": round(
            mean(float(selected == str(row["best_base_mode"])) for row, selected in zip(rows, selected_modes, strict=True)),
            6,
        ),
        "selected_generated_rate": round(
            mean(float(selected.startswith("generated_")) for selected in selected_modes),
            6,
        ),
    }
    for name in candidate_names:
        output[f"selected_{name}_rate"] = round(selected_counts.get(name, 0) / max(len(selected_modes), 1), 6)
    return output


def _oracle_row(
    rows: list[dict[str, object]],
    policy: str,
    selected_modes: list[str],
    candidate_names: list[str],
) -> dict[str, object]:
    row = _policy_row(rows, policy=policy, selected_modes=selected_modes, candidate_names=candidate_names)
    if policy == "generated_oracle":
        row["excess_over_generated_oracle"] = 0.0
        row["generated_oracle_match_rate"] = 1.0
    if policy == "base_oracle":
        row["excess_over_base_oracle"] = 0.0
        row["base_oracle_match_rate"] = 1.0
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
