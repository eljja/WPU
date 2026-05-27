from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM as OBJECT_CANDIDATE_FEATURE_DIM,
    _train_model as _train_retriever,
)
from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    GeneratedSetReranker,
    _best_static_mode,
    _candidate_names,
    _collect_examples,
    _example_tensors,
    _mean_policy_loss,
    _oracle_row,
    _policy_row,
    _predict_modes,
    _write_csv,
)
from scripts.staged_regret_hybrid import _class_weights, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalized cross-seed WPU retrieval reranker.")
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
    parser.add_argument("--reranker-steps", type=int, default=600)
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-lr", type=float, default=3e-3)
    parser.add_argument("--safe-margin", type=float, default=0.005)
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
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_normalized_reranker.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    candidate_names = _candidate_names(len(BASE_MODES) + args.generated_candidates)
    validation_by_seed: dict[int, list[dict[str, object]]] = {}
    test_by_seed: dict[int, list[dict[str, object]]] = {}
    for seed in args.seeds:
        print(f"normalized-cross-seed collect seed={seed} N={total_n} K={causal_k}", flush=True)
        validation_examples, test_examples = _collect_seed_examples(
            background_objects,
            causal_obstacles,
            seed,
            args,
            total_n,
            causal_k,
            candidate_names,
        )
        validation_by_seed[seed] = validation_examples
        test_by_seed[seed] = test_examples

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_examples = [
            example
            for seed, examples in validation_by_seed.items()
            if seed != heldout_seed
            for example in examples
        ]
        test_examples = test_by_seed[heldout_seed]
        reranker = _train_normalized_reranker(train_examples, args, len(candidate_names))
        condition_rows = _summarize_normalized(test_examples, train_examples, reranker, candidate_names, args)
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
                    "normalized_utility_weight": args.normalized_utility_weight,
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "reranker_steps": args.reranker_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                }
            )
        rows.extend(condition_rows)
    return rows


def _collect_seed_examples(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    total_n: int,
    causal_k: int,
    candidate_names: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    return validation_examples, test_examples


def _train_normalized_reranker(
    examples: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_count: int,
) -> GeneratedSetReranker:
    objects, masks, context, loss_tensor = _example_tensors(examples, candidate_count)
    centered_loss = loss_tensor - loss_tensor.mean(dim=1, keepdim=True)
    scale = loss_tensor.std(dim=1, keepdim=True).clamp_min(1e-3)
    normalized_utilities = -(centered_loss / scale)
    reranker = GeneratedSetReranker(OBJECT_CANDIDATE_FEATURE_DIM, candidate_count + 9, args.reranker_hidden_dim)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.reranker_lr)
    targets = normalized_utilities.argmax(dim=1)
    soft_targets = F.softmax(normalized_utilities / args.utility_temperature, dim=1)
    reranker.train()
    for _ in range(args.reranker_steps):
        scores = reranker(objects, masks, context)
        log_probs = F.log_softmax(scores, dim=1)
        ce_loss = F.cross_entropy(scores, targets)
        soft_ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        utility_loss = F.mse_loss(scores, normalized_utilities)
        loss = ce_loss + 0.5 * soft_ce_loss + args.normalized_utility_weight * utility_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reranker.eval()


def _summarize_normalized(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    reranker: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(train_examples, BASE_MODES)
    generated_static_mode = _best_static_mode(train_examples, candidate_names)
    reranker_modes = _predict_modes(test_examples, reranker, len(candidate_names))
    train_reranker_modes = _predict_modes(train_examples, reranker, len(candidate_names))
    train_reranker_loss = _mean_policy_loss(train_examples, train_reranker_modes)
    train_static_loss = _mean_policy_loss(train_examples, [static_mode] * len(train_examples))
    margin_safe = train_reranker_loss + args.safe_margin < train_static_loss
    safe_modes = reranker_modes if margin_safe else [static_mode] * len(test_examples)
    rows = [
        _policy_row(
            test_examples,
            policy="normalized_cross_seed_static_base_choice",
            selected_modes=[static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="normalized_cross_seed_static_generated_choice",
            selected_modes=[generated_static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="normalized_cross_seed_generated_reranker",
            selected_modes=reranker_modes,
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="normalized_cross_seed_margin_safe_reranker",
            selected_modes=safe_modes,
            candidate_names=candidate_names,
        ),
        _oracle_row(test_examples, "base_oracle", [str(example["best_base_mode"]) for example in test_examples], candidate_names),
        _oracle_row(test_examples, "generated_oracle", [str(example["best_mode"]) for example in test_examples], candidate_names),
    ]
    rows[3]["safe_uses_reranker"] = int(margin_safe)
    for row in rows:
        row["static_mode_from_train_seeds"] = static_mode
        row["static_generated_mode_from_train_seeds"] = generated_static_mode
        row["train_reranker_loss"] = round(train_reranker_loss, 6)
        row["train_static_loss"] = round(train_static_loss, 6)
        row["safe_margin"] = args.safe_margin
        row.setdefault("safe_uses_reranker", "")
    return rows


if __name__ == "__main__":
    main()
