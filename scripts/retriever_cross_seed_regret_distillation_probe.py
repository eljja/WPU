from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import _candidate_features, _candidate_ids, _selected_ids, _train_model as _train_interaction_retriever  # noqa: E402
from scripts.retriever_generated_candidate_probe import BASE_MODES  # noqa: E402
from scripts.retriever_regret_distillation_probe import (  # noqa: E402
    ObjectRegretRetriever,
    _candidate_losses,
    _candidate_sets,
    _make_samples,
    _policy_row,
    _regret_selected_ids,
    _write_csv,
)
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-seed regret-distilled WPU state retrieval.")
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
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_regret_distillation.csv"))
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
        print(f"cross-seed-regret collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        cross_seed_retriever = _train_cross_seed_regret_retriever(train_contexts, candidate_names, args)
        rows.extend(_summarize_heldout(heldout_seed, heldout, cross_seed_retriever, candidate_names, args, total_n, causal_k))
    return rows


def _collect_seed_context(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    candidate_names: list[str],
) -> dict[str, object]:
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
    interaction_retriever = _train_interaction_retriever(
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
    validation_samples = _make_samples(background_objects, causal_obstacles, seed + 5_000, args)
    test_samples = _make_samples(background_objects, causal_obstacles, seed + 10_000, args, sample_count=args.samples)
    validation_candidates = _candidate_sets(validation_samples, args, interaction_retriever, seed + 5_000, candidate_names)
    test_candidates = _candidate_sets(test_samples, args, interaction_retriever, seed + 10_000, candidate_names)
    validation_losses = _candidate_losses(model, validation_samples, validation_candidates, candidate_names, args, device)
    test_losses = _candidate_losses(model, test_samples, test_candidates, candidate_names, args, device)
    return {
        "model": model,
        "device": device,
        "interaction_retriever": interaction_retriever,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "validation_candidates": validation_candidates,
        "test_candidates": test_candidates,
        "validation_losses": validation_losses,
        "test_losses": test_losses,
    }


def _train_cross_seed_regret_retriever(
    contexts: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> ObjectRegretRetriever:
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for context in contexts:
        samples = context["validation_samples"]
        candidates = context["validation_candidates"]
        candidate_losses = context["validation_losses"]
        for sample_index, sample in enumerate(samples):  # type: ignore[arg-type]
            best_name = min(candidate_names, key=lambda name: (candidate_losses[name][0][sample_index], name))  # type: ignore[index]
            best_ids = set(candidates[best_name][sample_index])  # type: ignore[index]
            target = sample.event.target
            for object_id in _candidate_ids(sample.state, sample.event):
                if object_id == target:
                    continue
                features.append(_candidate_features(sample.state, sample.event, object_id))
                labels.append(float(object_id in best_ids))
    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    retriever = ObjectRegretRetriever(args.retriever_hidden_dim)
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=args.retriever_lr)
    positive = label_tensor.sum().clamp_min(1.0)
    negative = (1.0 - label_tensor).sum().clamp_min(1.0)
    pos_weight = negative / positive
    retriever.train()
    for _ in range(args.regret_retriever_steps):
        logits = retriever(feature_tensor)
        loss = F.binary_cross_entropy_with_logits(logits, label_tensor, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return retriever.eval()


def _summarize_heldout(
    heldout_seed: int,
    context: dict[str, object],
    cross_seed_retriever: ObjectRegretRetriever,
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
        "static_interaction": test_candidates["interaction"],  # type: ignore[index]
        "static_learned_interaction": test_candidates["learned"],  # type: ignore[index]
        "cross_seed_regret_distilled_retriever": [
            _regret_selected_ids(sample, args.budget, cross_seed_retriever) for sample in test_samples  # type: ignore[arg-type]
        ],
        "cross_seed_regret_min1_obstacle": [
            _regret_selected_ids_with_min_obstacles(sample, args.budget, cross_seed_retriever, min_obstacles=1)
            for sample in test_samples  # type: ignore[arg-type]
        ],
        "cross_seed_regret_min2_obstacles": [
            _regret_selected_ids_with_min_obstacles(sample, args.budget, cross_seed_retriever, min_obstacles=2)
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
                "validation_static_base_mode": validation_static,
                "validation_samples_per_seed": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return rows


def _evaluate_policy(model, samples, selected, args: argparse.Namespace, device: torch.device):
    from scripts.retriever_regret_oracle_probe import _evaluate_selected  # local import avoids circular tooling surprises

    return _evaluate_selected(model, samples, selected, args.batch_size, device)


def _regret_selected_ids_with_min_obstacles(
    sample,
    budget: int,
    retriever: ObjectRegretRetriever,
    *,
    min_obstacles: int,
) -> list[str]:
    target = sample.event.target
    candidates = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not candidates:
        return [target]
    features = torch.stack([_candidate_features(sample.state, sample.event, object_id) for object_id in candidates])
    with torch.no_grad():
        scores = retriever(features)
    scored = sorted(zip(scores.tolist(), candidates, strict=True), reverse=True)
    selected = [target]
    obstacle_ranked = [object_id for _, object_id in scored if object_id.startswith("obstacle_")]
    for object_id in obstacle_ranked[:min_obstacles]:
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


if __name__ == "__main__":
    main()
