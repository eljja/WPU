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
    parser = argparse.ArgumentParser(description="Evaluate pairwise-ranking loss for generated WPU state candidates.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--reranker-steps", type=int, default=600)
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-lr", type=float, default=3e-3)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--soft-ce-weight", type=float, default=0.25)
    parser.add_argument("--utility-weight", type=float, default=0.05)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_pairwise_reranker.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            for seed in args.seeds:
                print(f"pairwise-reranker seed={seed} N={n_value} K={k_value}", flush=True)
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
    candidate_names = _candidate_names(len(BASE_MODES) + args.generated_candidates)
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
    reranker = _train_pairwise_reranker(validation_examples, args, len(candidate_names))
    rows = _summarize_pairwise(test_examples, validation_examples, reranker, candidate_names, args)
    for row in rows:
        row.update(
            {
                "seed": seed,
                "total_objects_n": total_n,
                "causal_k": causal_k,
                "budget": args.budget,
                "generated_candidates": args.generated_candidates,
                "pairwise_weight": args.pairwise_weight,
                "soft_ce_weight": args.soft_ce_weight,
                "utility_weight": args.utility_weight,
                "interaction_mode": args.interaction_mode,
                "propagation_steps": args.propagation_steps,
                "retriever_steps": args.retriever_steps,
                "reranker_steps": args.reranker_steps,
                "validation_samples": args.validation_samples,
                "test_samples": args.samples,
            }
        )
    return rows


def _train_pairwise_reranker(
    examples: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_count: int,
) -> GeneratedSetReranker:
    objects, masks, context, loss_tensor = _example_tensors(examples, candidate_count)
    utilities = -loss_tensor
    reranker = GeneratedSetReranker(OBJECT_CANDIDATE_FEATURE_DIM, candidate_count + 9, args.reranker_hidden_dim)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.reranker_lr)
    soft_targets = F.softmax(utilities / args.utility_temperature, dim=1)
    for _ in range(args.reranker_steps):
        scores = reranker(objects, masks, context)
        score_delta = scores.unsqueeze(2) - scores.unsqueeze(1)
        loss_delta = loss_tensor.unsqueeze(2) - loss_tensor.unsqueeze(1)
        better_mask = loss_delta < 0.0
        pair_weights = loss_delta.abs().clamp(max=0.25)
        pairwise_loss = (F.softplus(-score_delta) * pair_weights)[better_mask].mean()
        log_probs = F.log_softmax(scores, dim=1)
        soft_ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        utility_loss = F.mse_loss(scores, utilities)
        loss = args.pairwise_weight * pairwise_loss
        loss = loss + args.soft_ce_weight * soft_ce_loss
        loss = loss + args.utility_weight * utility_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reranker.eval()


def _summarize_pairwise(
    test_examples: list[dict[str, object]],
    validation_examples: list[dict[str, object]],
    reranker: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(validation_examples, BASE_MODES)
    generated_static_mode = _best_static_mode(validation_examples, candidate_names)
    reranker_modes = _predict_modes(test_examples, reranker, len(candidate_names))
    validation_reranker_modes = _predict_modes(validation_examples, reranker, len(candidate_names))
    validation_reranker_loss = _mean_policy_loss(validation_examples, validation_reranker_modes)
    validation_static_loss = _mean_policy_loss(validation_examples, [static_mode] * len(validation_examples))
    margin_safe = validation_reranker_loss + args.safe_margin < validation_static_loss
    safe_modes = reranker_modes if margin_safe else [static_mode] * len(test_examples)
    rows = [
        _policy_row(
            test_examples,
            policy="static_base_validation_choice",
            selected_modes=[static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="static_generated_validation_choice",
            selected_modes=[generated_static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="pairwise_generated_reranker",
            selected_modes=reranker_modes,
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="pairwise_margin_safe_reranker",
            selected_modes=safe_modes,
            candidate_names=candidate_names,
        ),
        _oracle_row(test_examples, "base_oracle", [str(example["best_base_mode"]) for example in test_examples], candidate_names),
        _oracle_row(test_examples, "generated_oracle", [str(example["best_mode"]) for example in test_examples], candidate_names),
    ]
    rows[3]["safe_uses_reranker"] = int(margin_safe)
    for row in rows:
        row["static_mode_from_validation"] = static_mode
        row["static_generated_mode_from_validation"] = generated_static_mode
        row["validation_reranker_loss"] = round(validation_reranker_loss, 6)
        row["validation_static_loss"] = round(validation_static_loss, 6)
        row["safe_margin"] = args.safe_margin
        row.setdefault("safe_uses_reranker", "")
    return rows


if __name__ == "__main__":
    main()
