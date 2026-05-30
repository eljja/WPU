from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM,
    _candidate_features,
    _candidate_ids,
    _selected_ids,
    _train_model as _train_interaction_retriever,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _generated_candidates  # noqa: E402
from scripts.retriever_regret_oracle_probe import _evaluate_selected  # noqa: E402
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


class ObjectRegretRetriever(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill downstream-regret oracle candidate sets into a state retriever.")
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_regret_distillation.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"regret-distillation seed={seed} N={n_value} K={k_value}", flush=True)
                rows.extend(_run_condition(background_objects, causal_obstacles, seed, args))
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_condition(background_objects: int, causal_obstacles: int, seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
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
    candidate_names = [*BASE_MODES, *[f"generated_{index}" for index in range(args.generated_candidates)]]
    validation_candidates = _candidate_sets(validation_samples, args, interaction_retriever, seed + 5_000, candidate_names)
    validation_losses = _candidate_losses(model, validation_samples, validation_candidates, candidate_names, args, device)
    regret_retriever = _train_regret_retriever(
        validation_samples,
        validation_candidates,
        validation_losses,
        candidate_names,
        args,
    )
    test_candidates = _candidate_sets(test_samples, args, interaction_retriever, seed + 10_000, candidate_names)
    test_losses = _candidate_losses(model, test_samples, test_candidates, candidate_names, args, device)
    rows = _summarize(
        model,
        validation_samples,
        test_samples,
        validation_candidates,
        validation_losses,
        test_candidates,
        test_losses,
        interaction_retriever,
        regret_retriever,
        candidate_names,
        args,
        device,
    )
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
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
                "interaction_retriever_steps": args.retriever_steps,
                "regret_retriever_steps": args.regret_retriever_steps,
                "validation_samples": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return rows


def _make_samples(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    *,
    sample_count: int | None = None,
):
    dataset = WorkingSetPhysicsDataset(
        size=sample_count if sample_count is not None else args.validation_samples,
        seed=seed,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    return [dataset[index] for index in range(len(dataset))]


def _candidate_sets(
    samples,
    args: argparse.Namespace,
    interaction_retriever: nn.Module,
    dataset_seed: int,
    candidate_names: list[str],
) -> dict[str, list[list[str]]]:
    selected_by_candidate: dict[str, list[list[str]]] = {name: [] for name in candidate_names}
    for sample_index, sample in enumerate(samples):
        for mode in BASE_MODES:
            selected_by_candidate[mode].append(
                _selected_ids(sample, mode, args.budget, interaction_retriever if mode == "learned" else None)
            )
        generated = _generated_candidates(sample, args.budget, args.generated_candidates, seed=dataset_seed + sample_index)
        for index, selected_ids in enumerate(generated):
            selected_by_candidate[f"generated_{index}"].append(selected_ids)
    return selected_by_candidate


def _candidate_losses(
    model: torch.nn.Module,
    samples,
    candidates: dict[str, list[list[str]]],
    candidate_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, tuple[list[float], list[int]]]:
    return {
        name: _evaluate_selected(model, samples, candidates[name], args.batch_size, device)
        for name in candidate_names
    }


def _train_regret_retriever(
    samples,
    candidates: dict[str, list[list[str]]],
    candidate_losses: dict[str, tuple[list[float], list[int]]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> ObjectRegretRetriever:
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for sample_index, sample in enumerate(samples):
        best_name = min(candidate_names, key=lambda name: (candidate_losses[name][0][sample_index], name))
        best_ids = set(candidates[best_name][sample_index])
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


def _regret_selected_ids(sample, budget: int, retriever: ObjectRegretRetriever) -> list[str]:
    target = sample.event.target
    candidates = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not candidates:
        return [target]
    features = torch.stack([_candidate_features(sample.state, sample.event, object_id) for object_id in candidates])
    with torch.no_grad():
        scores = retriever(features)
    ranked = [object_id for _, object_id in sorted(zip(scores.tolist(), candidates, strict=True), reverse=True)]
    selected = [target]
    for object_id in ranked:
        if object_id not in selected:
            selected.append(object_id)
        if len(selected) >= budget:
            break
    return selected


def _summarize(
    model: torch.nn.Module,
    validation_samples,
    test_samples,
    validation_candidates: dict[str, list[list[str]]],
    validation_losses: dict[str, tuple[list[float], list[int]]],
    test_candidates: dict[str, list[list[str]]],
    test_losses: dict[str, tuple[list[float], list[int]]],
    interaction_retriever: nn.Module,
    regret_retriever: ObjectRegretRetriever,
    candidate_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    del validation_samples, validation_candidates
    validation_static = min(BASE_MODES, key=lambda name: (mean(validation_losses[name][0]), name))
    deployable = {
        "static_validation_base_choice": test_candidates[validation_static],
        "static_interaction": test_candidates["interaction"],
        "static_learned_interaction": test_candidates["learned"],
        "regret_distilled_retriever": [_regret_selected_ids(sample, args.budget, regret_retriever) for sample in test_samples],
        "interaction_retrained_reference": [
            _selected_ids(sample, "learned", args.budget, interaction_retriever) for sample in test_samples
        ],
    }
    rows = []
    for policy, selected in deployable.items():
        losses, correct = _evaluate_selected(model, test_samples, selected, args.batch_size, device)
        rows.append(_policy_row(policy, losses, correct, test_losses, candidate_names, selected))
    oracle_modes = [
        min(candidate_names, key=lambda name: (test_losses[name][0][index], name))
        for index in range(len(test_samples))
    ]
    oracle_selected = [test_candidates[name][index] for index, name in enumerate(oracle_modes)]
    oracle_losses = [test_losses[name][0][index] for index, name in enumerate(oracle_modes)]
    oracle_correct = [test_losses[name][1][index] for index, name in enumerate(oracle_modes)]
    rows.append(_policy_row("generated_oracle", oracle_losses, oracle_correct, test_losses, candidate_names, oracle_selected))
    for row in rows:
        row["validation_static_base_mode"] = validation_static
    return rows


def _policy_row(
    policy: str,
    losses: list[float],
    correct: list[int],
    test_losses: dict[str, tuple[list[float], list[int]]],
    candidate_names: list[str],
    selected_ids: list[list[str]],
) -> dict[str, object]:
    oracle_losses = [
        min(test_losses[name][0][index] for name in candidate_names)
        for index in range(len(losses))
    ]
    return {
        "policy": policy,
        "loss": round(mean(losses), 6),
        "accuracy": round(mean(float(value) for value in correct), 6),
        "excess_over_generated_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "selected_obstacles": round(mean(sum(object_id.startswith("obstacle_") for object_id in ids) for ids in selected_ids), 6),
        "selected_hand_rate": round(mean(float("hand_001" in ids) for ids in selected_ids), 6),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
