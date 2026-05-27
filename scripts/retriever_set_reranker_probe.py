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
    FEATURE_DIM as OBJECT_CANDIDATE_FEATURE_DIM,
    _candidate_features,
    _selected_ids,
    _selected_pair_density,
    _train_model as _train_retriever,
)
from scripts.retriever_regret_oracle_probe import MODES, _evaluate_selected  # noqa: E402
from scripts.retriever_reranker_probe import (  # noqa: E402
    _best_static_mode,
    _mean_policy_loss,
    _oracle_row,
    _policy_row,
)
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


CONTEXT_DIM = len(MODES) + 8


class SetReranker(nn.Module):
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
        batch_size, mode_count, budget, object_dim = objects.shape
        encoded = self.object_encoder(objects.view(batch_size * mode_count * budget, object_dim))
        encoded = encoded.view(batch_size, mode_count, budget, -1)
        float_mask = mask.unsqueeze(-1).float()
        masked_sum = (encoded * float_mask).sum(dim=2)
        counts = float_mask.sum(dim=2).clamp_min(1.0)
        pooled_mean = masked_sum / counts
        pooled_max = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=2)
        features = torch.cat([pooled_mean, pooled_max, context], dim=-1)
        return self.scorer(features).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an object-set reranker over explicit WPU retrieval candidates.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--budget", type=int, default=4)
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_set_reranker.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"retriever-set-reranker seed={seed} N={n_value} K={k_value}", flush=True)
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
    )
    reranker = _train_set_reranker(validation_examples, args)
    summary = _summarize(test_examples, validation_examples, reranker, args)
    for row in summary:
        row.update(
            {
                "seed": seed,
                "total_objects_n": total_n,
                "causal_k": causal_k,
                "budget": args.budget,
                "interaction_mode": args.interaction_mode,
                "propagation_steps": args.propagation_steps,
                "retriever_steps": args.retriever_steps,
                "reranker_steps": args.reranker_steps,
                "validation_samples": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return summary


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
    selected_by_mode = {
        mode: [_selected_ids(sample, mode, args.budget, retriever if mode == "learned" else None) for sample in samples]
        for mode in MODES
    }
    losses_by_mode: dict[str, list[float]] = {}
    correct_by_mode: dict[str, list[int]] = {}
    for mode in MODES:
        losses, correct = _evaluate_selected(model, samples, selected_by_mode[mode], args.batch_size, device)
        losses_by_mode[mode] = losses
        correct_by_mode[mode] = correct

    examples: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        row: dict[str, object] = {}
        mode_losses = {mode: losses_by_mode[mode][sample_index] for mode in MODES}
        best_mode = min(MODES, key=lambda mode: (mode_losses[mode], mode))
        row["best_mode"] = best_mode
        row["oracle_loss"] = round(mode_losses[best_mode], 6)
        row["oracle_correct"] = correct_by_mode[best_mode][sample_index]
        object_features = []
        object_masks = []
        context_features = []
        for mode_index, mode in enumerate(MODES):
            selected_ids = selected_by_mode[mode][sample_index]
            row[f"{mode}_loss"] = round(mode_losses[mode], 6)
            row[f"{mode}_correct"] = correct_by_mode[mode][sample_index]
            row[f"{mode}_selected_hand"] = int("hand_001" in selected_ids)
            row[f"{mode}_selected_obstacles"] = sum(object_id.startswith("obstacle_") for object_id in selected_ids)
            row[f"{mode}_pair_density"] = round(
                _selected_pair_density(
                    sample.state,
                    [object_id for object_id in selected_ids if object_id.startswith("obstacle_")],
                ),
                6,
            )
            object_tensor, mask_tensor = _selected_object_tensor(sample, selected_ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            context_features.append(_context_features(row, mode, mode_index, total_n, causal_k, args.budget))
        row["object_features"] = torch.stack(object_features)
        row["object_masks"] = torch.stack(object_masks)
        row["context_features"] = torch.tensor(context_features, dtype=torch.float32)
        examples.append(row)
    return examples


def _selected_object_tensor(sample, selected_ids: list[str], budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.zeros((budget, OBJECT_CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((budget,), dtype=torch.bool)
    for index, object_id in enumerate(selected_ids[:budget]):
        features[index] = _candidate_features(sample.state, sample.event, object_id)
        mask[index] = True
    return features, mask


def _context_features(
    row: dict[str, object],
    mode: str,
    mode_index: int,
    total_n: int,
    causal_k: int,
    budget: int,
) -> list[float]:
    one_hot = [0.0 for _ in MODES]
    one_hot[mode_index] = 1.0
    selected_hand = float(row[f"{mode}_selected_hand"])
    selected_obstacles = float(row[f"{mode}_selected_obstacles"])
    pair_density = float(row[f"{mode}_pair_density"])
    obstacle_ratio = selected_obstacles / max(float(budget), 1.0)
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
    ]


def _train_set_reranker(examples: list[dict[str, object]], args: argparse.Namespace) -> SetReranker:
    objects, masks, context, loss_tensor = _example_tensors(examples)
    utilities = -loss_tensor
    reranker = SetReranker(OBJECT_CANDIDATE_FEATURE_DIM, CONTEXT_DIM, args.reranker_hidden_dim)
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


def _example_tensors(examples: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objects = torch.stack([example["object_features"] for example in examples])  # type: ignore[list-item]
    masks = torch.stack([example["object_masks"] for example in examples])  # type: ignore[list-item]
    context = torch.stack([example["context_features"] for example in examples])  # type: ignore[list-item]
    losses = torch.tensor(
        [[float(example[f"{mode}_loss"]) for mode in MODES] for example in examples],
        dtype=torch.float32,
    )
    return objects, masks, context, losses


def _summarize(
    test_examples: list[dict[str, object]],
    validation_examples: list[dict[str, object]],
    reranker: SetReranker,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(validation_examples)
    summary = [_policy_row(test_examples, mode=f"static_{mode}", selected_modes=[mode] * len(test_examples)) for mode in MODES]
    summary.append(_policy_row(test_examples, mode="static_validation_choice", selected_modes=[static_mode] * len(test_examples)))
    reranker_modes = _predict_modes(test_examples, reranker)
    validation_reranker_modes = _predict_modes(validation_examples, reranker)
    validation_reranker_loss = _mean_policy_loss(validation_examples, validation_reranker_modes)
    validation_static_loss = _mean_policy_loss(validation_examples, [static_mode] * len(validation_examples))
    safe_uses_reranker = validation_reranker_loss < validation_static_loss
    margin_safe_uses_reranker = validation_reranker_loss + args.safe_margin < validation_static_loss
    summary.append(_policy_row(test_examples, mode="set_deployed_reranker", selected_modes=reranker_modes))
    safe_modes = reranker_modes if safe_uses_reranker else [static_mode] * len(test_examples)
    safe_row = _policy_row(test_examples, mode="set_validation_safe_reranker", selected_modes=safe_modes)
    safe_row["safe_uses_reranker"] = int(safe_uses_reranker)
    summary.append(safe_row)
    margin_modes = reranker_modes if margin_safe_uses_reranker else [static_mode] * len(test_examples)
    margin_row = _policy_row(test_examples, mode="set_margin_safe_reranker", selected_modes=margin_modes)
    margin_row["safe_uses_reranker"] = int(margin_safe_uses_reranker)
    summary.append(margin_row)
    summary.append(_oracle_row(test_examples))
    for row in summary:
        row["static_mode_from_validation"] = static_mode
        row["validation_reranker_loss"] = round(validation_reranker_loss, 6)
        row["validation_static_loss"] = round(validation_static_loss, 6)
        row["safe_margin"] = args.safe_margin
        row.setdefault("safe_uses_reranker", "")
    return summary


def _predict_modes(examples: list[dict[str, object]], reranker: SetReranker) -> list[str]:
    objects, masks, context, _ = _example_tensors(examples)
    with torch.no_grad():
        scores = reranker(objects, masks, context)
    return [MODES[int(index)] for index in scores.argmax(dim=1).tolist()]


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
