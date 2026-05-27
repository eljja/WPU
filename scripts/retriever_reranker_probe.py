from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import _train_model as _train_retriever  # noqa: E402
from scripts.retriever_regret_oracle_probe import MODES, _collect_mode_rows  # noqa: E402
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


FEATURE_DIM = len(MODES) + 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a deployed state-native reranker over explicit working-set retrieval candidates."
    )
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_reranker.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"retriever-reranker seed={seed} N={n_value} K={k_value}", flush=True)
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
    validation_rows = _collect_mode_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 5_000,
        args.validation_samples,
        args,
        retriever,
        device,
        split="validation",
    )
    test_rows = _collect_mode_rows(
        model,
        background_objects,
        causal_obstacles,
        seed + 10_000,
        args.samples,
        args,
        retriever,
        device,
        split="test",
    )
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
    for row in [*validation_rows, *test_rows]:
        row["total_objects_n"] = total_n
        row["causal_k"] = causal_k
    reranker = _train_reranker(validation_rows, args)
    summary = _summarize_policies(test_rows, validation_rows, reranker, total_n, causal_k, args)
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


def _train_reranker(rows: list[dict[str, object]], args: argparse.Namespace) -> nn.Module:
    feature_tensor, loss_tensor = _candidate_tensors(rows, args)
    utilities = -loss_tensor
    model = nn.Sequential(
        nn.LayerNorm(FEATURE_DIM),
        nn.Linear(FEATURE_DIM, args.reranker_hidden_dim),
        nn.GELU(),
        nn.Linear(args.reranker_hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.reranker_lr)
    targets = utilities.argmax(dim=1)
    soft_targets = F.softmax(utilities / args.utility_temperature, dim=1)
    for _ in range(args.reranker_steps):
        scores = model(feature_tensor.view(-1, FEATURE_DIM)).view(feature_tensor.size(0), len(MODES))
        log_probs = F.log_softmax(scores, dim=1)
        ce_loss = F.cross_entropy(scores, targets)
        soft_ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        utility_loss = F.mse_loss(scores, utilities)
        loss = ce_loss + 0.5 * soft_ce_loss + 0.1 * utility_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def _candidate_tensors(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    losses = []
    total_n = float(rows[0].get("total_objects_n", 0) or args.n_values[0])
    causal_k = float(rows[0].get("causal_k", 0) or args.k_values[0])
    for row in rows:
        row_features = []
        row_losses = []
        for mode_index, mode in enumerate(MODES):
            row_features.append(_candidate_features(row, mode, mode_index, total_n, causal_k, args.budget))
            row_losses.append(float(row[f"{mode}_loss"]))
        features.append(torch.tensor(row_features, dtype=torch.float32))
        losses.append(torch.tensor(row_losses, dtype=torch.float32))
    return torch.stack(features), torch.stack(losses)


def _candidate_features(
    row: dict[str, object],
    mode: str,
    mode_index: int,
    total_n: float,
    causal_k: float,
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


def _summarize_policies(
    test_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    reranker: nn.Module,
    total_n: int,
    causal_k: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(validation_rows)
    summary = [_policy_row(test_rows, mode=f"static_{mode}", selected_modes=[mode] * len(test_rows)) for mode in MODES]
    summary.append(_policy_row(test_rows, mode="static_validation_choice", selected_modes=[static_mode] * len(test_rows)))
    reranker_modes = _predict_modes(test_rows, reranker, total_n, causal_k, args)
    validation_reranker_modes = _predict_modes(validation_rows, reranker, total_n, causal_k, args)
    validation_reranker_loss = _mean_policy_loss(validation_rows, validation_reranker_modes)
    validation_static_loss = _mean_policy_loss(validation_rows, [static_mode] * len(validation_rows))
    safe_uses_reranker = validation_reranker_loss < validation_static_loss
    safe_modes = reranker_modes if safe_uses_reranker else [static_mode] * len(test_rows)
    margin_safe_uses_reranker = validation_reranker_loss + args.safe_margin < validation_static_loss
    margin_safe_modes = reranker_modes if margin_safe_uses_reranker else [static_mode] * len(test_rows)
    summary.append(_policy_row(test_rows, mode="deployed_reranker", selected_modes=reranker_modes))
    safe_row = _policy_row(test_rows, mode="validation_safe_reranker", selected_modes=safe_modes)
    safe_row["safe_uses_reranker"] = int(safe_uses_reranker)
    safe_row["validation_reranker_loss"] = round(validation_reranker_loss, 6)
    safe_row["validation_static_loss"] = round(validation_static_loss, 6)
    summary.append(safe_row)
    margin_safe_row = _policy_row(test_rows, mode="margin_safe_reranker", selected_modes=margin_safe_modes)
    margin_safe_row["safe_uses_reranker"] = int(margin_safe_uses_reranker)
    margin_safe_row["safe_margin"] = args.safe_margin
    margin_safe_row["validation_reranker_loss"] = round(validation_reranker_loss, 6)
    margin_safe_row["validation_static_loss"] = round(validation_static_loss, 6)
    summary.append(margin_safe_row)
    summary.append(_oracle_row(test_rows))
    for row in summary:
        row["static_mode_from_validation"] = static_mode
        row.setdefault("safe_margin", args.safe_margin)
        row.setdefault("safe_uses_reranker", "")
        row.setdefault("validation_reranker_loss", round(validation_reranker_loss, 6))
        row.setdefault("validation_static_loss", round(validation_static_loss, 6))
    return summary


def _predict_modes(
    rows: list[dict[str, object]],
    reranker: nn.Module,
    total_n: int,
    causal_k: int,
    args: argparse.Namespace,
) -> list[str]:
    selected = []
    with torch.no_grad():
        for row in rows:
            features = torch.tensor(
                [
                    _candidate_features(row, mode, mode_index, float(total_n), float(causal_k), args.budget)
                    for mode_index, mode in enumerate(MODES)
                ],
                dtype=torch.float32,
            )
            scores = reranker(features).squeeze(-1)
            selected.append(MODES[int(scores.argmax().item())])
    return selected


def _best_static_mode(rows: list[dict[str, object]]) -> str:
    losses = {mode: mean(float(row[f"{mode}_loss"]) for row in rows) for mode in MODES}
    return min(MODES, key=lambda mode: (losses[mode], mode))


def _mean_policy_loss(rows: list[dict[str, object]], selected_modes: list[str]) -> float:
    return mean(float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True))


def _policy_row(rows: list[dict[str, object]], *, mode: str, selected_modes: list[str]) -> dict[str, object]:
    losses = [float(row[f"{selected}_loss"]) for row, selected in zip(rows, selected_modes, strict=True)]
    correct = [float(row[f"{selected}_correct"]) for row, selected in zip(rows, selected_modes, strict=True)]
    oracle_losses = [float(row["oracle_loss"]) for row in rows]
    oracle_correct = [float(row["oracle_correct"]) for row in rows]
    selected_counts = Counter(selected_modes)
    return {
        "policy": mode,
        "loss": round(mean(losses), 6),
        "accuracy": round(mean(correct), 6),
        "oracle_loss": round(mean(oracle_losses), 6),
        "oracle_accuracy": round(mean(oracle_correct), 6),
        "excess_over_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "oracle_match_rate": round(
            mean(float(selected == str(row["best_mode"])) for row, selected in zip(rows, selected_modes, strict=True)),
            6,
        ),
        **{f"selected_{candidate}_rate": round(selected_counts.get(candidate, 0) / max(len(selected_modes), 1), 6) for candidate in MODES},
    }


def _oracle_row(rows: list[dict[str, object]]) -> dict[str, object]:
    selected_modes = [str(row["best_mode"]) for row in rows]
    row = _policy_row(rows, mode="oracle_over_retrievers", selected_modes=selected_modes)
    row["excess_over_oracle"] = 0.0
    row["oracle_match_rate"] = 1.0
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
