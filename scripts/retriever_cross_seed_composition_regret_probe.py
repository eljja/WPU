from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import _candidate_features, _candidate_ids  # noqa: E402
from scripts.retriever_generated_candidate_probe import BASE_MODES  # noqa: E402
from scripts.retriever_regret_distillation_probe import ObjectRegretRetriever, _policy_row, _write_csv  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _evaluate_policy,
    _train_cross_seed_regret_retriever,
)


COMPOSITION_FEATURE_DIM = 9


class CompositionPrior(nn.Module):
    def __init__(self, hidden_dim: int, budget: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(COMPOSITION_FEATURE_DIM),
            nn.Linear(COMPOSITION_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.count_head = nn.Linear(hidden_dim, budget + 1)
        self.hand_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.count_head(hidden), self.hand_head(hidden).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate state-conditioned composition priors for cross-seed regret retrieval.")
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
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_composition_regret.csv"))
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
    candidate_names = [*BASE_MODES, *[f"generated_{index}" for index in range(args.generated_candidates)]]
    contexts = {}
    for seed in args.seeds:
        print(f"composition-regret collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, candidate_names, args)
        rows.extend(
            _summarize_heldout(
                heldout_seed,
                heldout,
                object_retriever,
                composition_prior,
                candidate_names,
                args,
                total_n,
                causal_k,
            )
        )
    return rows


def _train_composition_prior(
    contexts: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> CompositionPrior:
    features: list[torch.Tensor] = []
    obstacle_counts: list[int] = []
    hand_labels: list[float] = []
    for context in contexts:
        samples = context["validation_samples"]
        candidates = context["validation_candidates"]
        candidate_losses = context["validation_losses"]
        for sample_index, sample in enumerate(samples):  # type: ignore[arg-type]
            best_name = min(candidate_names, key=lambda name: (candidate_losses[name][0][sample_index], name))  # type: ignore[index]
            best_ids = set(candidates[best_name][sample_index])  # type: ignore[index]
            features.append(_composition_features(sample, args.budget))
            obstacle_counts.append(min(sum(object_id.startswith("obstacle_") for object_id in best_ids), args.budget))
            hand_labels.append(float("hand_001" in best_ids))
    feature_tensor = torch.stack(features)
    count_tensor = torch.tensor(obstacle_counts, dtype=torch.long)
    hand_tensor = torch.tensor(hand_labels, dtype=torch.float32)
    model = CompositionPrior(args.retriever_hidden_dim, args.budget)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.composition_lr)
    model.train()
    for _ in range(args.composition_steps):
        count_logits, hand_logits = model(feature_tensor)
        loss = F.cross_entropy(count_logits, count_tensor) + 0.5 * F.binary_cross_entropy_with_logits(hand_logits, hand_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def _composition_features(sample, budget: int) -> torch.Tensor:
    target = sample.event.target
    candidates = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not candidates:
        return torch.zeros((COMPOSITION_FEATURE_DIM,), dtype=torch.float32)
    candidate_features = torch.stack([_candidate_features(sample.state, sample.event, object_id) for object_id in candidates])
    obstacle_mask = candidate_features[:, 1] > 0.5
    obstacle_distances = candidate_features[obstacle_mask, 6] if obstacle_mask.any() else torch.tensor([1.0])
    obstacle_densities = candidate_features[obstacle_mask, 7] if obstacle_mask.any() else torch.tensor([0.0])
    force = float(sample.event.delta.get("force", 0.0))
    return torch.tensor(
        [
            force,
            len(candidates) / 64.0,
            float(obstacle_mask.float().sum()) / 64.0,
            float((candidate_features[:, 0] > 0.5).any()),
            float(obstacle_distances.mean()),
            float(obstacle_distances.min()),
            float(obstacle_densities.mean()),
            float(obstacle_densities.max()),
            budget / 64.0,
        ],
        dtype=torch.float32,
    )


def _summarize_heldout(
    heldout_seed: int,
    context: dict[str, object],
    object_retriever: ObjectRegretRetriever,
    composition_prior: CompositionPrior,
    candidate_names: list[str],
    args: argparse.Namespace,
    total_n: int,
    causal_k: int,
) -> list[dict[str, object]]:
    model = context["model"]
    device = context["device"]
    test_samples = context["test_samples"]
    test_candidates = context["test_candidates"]
    test_losses = context["test_losses"]
    validation_losses = context["validation_losses"]
    validation_static = min(BASE_MODES, key=lambda name: (mean(validation_losses[name][0]), name))  # type: ignore[index]
    selected_by_policy = {
        "static_validation_base_choice": test_candidates[validation_static],  # type: ignore[index]
        "static_learned_interaction": test_candidates["learned"],  # type: ignore[index]
        "composition_regret_argmax": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="argmax")
            for sample in test_samples  # type: ignore[arg-type]
        ],
        "composition_regret_expected": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="expected")
            for sample in test_samples  # type: ignore[arg-type]
        ],
        "composition_regret_count_only": [
            _composition_selected_ids(sample, args.budget, object_retriever, composition_prior, mode="count_only")
            for sample in test_samples  # type: ignore[arg-type]
        ],
    }
    rows = []
    for policy, selected in selected_by_policy.items():
        losses, correct = _evaluate_policy(model, test_samples, selected, args, device)  # type: ignore[arg-type]
        rows.append(_policy_row(policy, losses, correct, test_losses, candidate_names, selected))  # type: ignore[arg-type]
    oracle_modes = [
        min(candidate_names, key=lambda name: (test_losses[name][0][index], name))  # type: ignore[index]
        for index in range(len(test_samples))  # type: ignore[arg-type]
    ]
    oracle_selected = [test_candidates[name][index] for index, name in enumerate(oracle_modes)]  # type: ignore[index]
    oracle_losses = [test_losses[name][0][index] for index, name in enumerate(oracle_modes)]  # type: ignore[index]
    oracle_correct = [test_losses[name][1][index] for index, name in enumerate(oracle_modes)]  # type: ignore[index]
    rows.append(_policy_row("generated_oracle", oracle_losses, oracle_correct, test_losses, candidate_names, oracle_selected))  # type: ignore[arg-type]
    for row in rows:
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
                "interaction_retriever_steps": args.retriever_steps,
                "regret_retriever_steps": args.regret_retriever_steps,
                "composition_steps": args.composition_steps,
                "validation_static_base_mode": validation_static,
                "validation_samples_per_seed": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return rows


def _composition_selected_ids(
    sample,
    budget: int,
    object_retriever: ObjectRegretRetriever,
    composition_prior: CompositionPrior,
    *,
    mode: str,
) -> list[str]:
    target = sample.event.target
    candidates = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not candidates:
        return [target]
    object_features = torch.stack([_candidate_features(sample.state, sample.event, object_id) for object_id in candidates])
    with torch.no_grad():
        object_scores = object_retriever(object_features)
        count_logits, hand_logit = composition_prior(_composition_features(sample, budget).unsqueeze(0))
        count_probs = F.softmax(count_logits.squeeze(0), dim=0)
    if mode == "expected":
        predicted_obstacles = int(round(float((count_probs * torch.arange(count_probs.numel())).sum())))
    else:
        predicted_obstacles = int(count_probs.argmax())
    predicted_obstacles = max(0, min(predicted_obstacles, budget - 1))
    include_hand = bool(torch.sigmoid(hand_logit).item() >= 0.5) if mode != "count_only" else False
    scored = sorted(zip(object_scores.tolist(), candidates, strict=True), reverse=True)
    selected = [target]
    if include_hand and "hand_001" in candidates and len(selected) < budget:
        selected.append("hand_001")
    obstacle_ranked = [object_id for _, object_id in scored if object_id.startswith("obstacle_")]
    for object_id in obstacle_ranked[:predicted_obstacles]:
        if object_id not in selected:
            selected.append(object_id)
        if len(selected) >= budget:
            return selected
    for _, object_id in scored:
        if object_id not in selected:
            selected.append(object_id)
        if len(selected) >= budget:
            break
    return selected


def _evaluate_policy(model, samples, selected, args: argparse.Namespace, device: torch.device):
    from scripts.retriever_regret_oracle_probe import _evaluate_selected

    return _evaluate_selected(model, samples, selected, args.batch_size, device)


if __name__ == "__main__":
    main()
