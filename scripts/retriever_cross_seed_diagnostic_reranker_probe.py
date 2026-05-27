from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM as OBJECT_CANDIDATE_FEATURE_DIM,
    _selected_ids,
    _selected_pair_density,
    _train_model as _train_retriever,
)
from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    GeneratedSetReranker,
    _best_static_mode,
    _candidate_names,
    _context_features,
    _example_tensors,
    _generated_candidates,
    _mean_policy_loss,
    _oracle_row,
    _policy_row,
    _predict_modes,
    _selected_object_tensor,
    _write_csv,
)
from scripts.staged_regret_hybrid import _class_weights, _move_batch, _train_propagation  # noqa: E402
from wpu.data.working_set_physics import WorkingSetPhysicsDataset, collate_selected_working_set_samples  # noqa: E402
from wpu.models.factory import create_model  # noqa: E402


DIAGNOSTIC_DIM = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-diagnostic cross-seed WPU retrieval reranking.")
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
    parser.add_argument("--cv-safe-margin", type=float, default=0.0)
    parser.add_argument("--cv-min-win-rate", type=float, default=0.5)
    parser.add_argument(
        "--context-variants",
        nargs="+",
        choices=["full", "no_identity", "no_selector_type", "set_only", "diagnostics_only"],
        default=["full"],
    )
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_diagnostic_reranker.csv"))
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
        print(f"diagnostic-cross-seed collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
        train_by_seed = {seed: examples for seed, examples in validation_by_seed.items() if seed != heldout_seed}
        test_examples = test_by_seed[heldout_seed]
        for variant in args.context_variants:
            variant_train_examples = _with_context_variant(train_examples, len(candidate_names), variant)
            variant_test_examples = _with_context_variant(test_examples, len(candidate_names), variant)
            variant_train_by_seed = {
                seed: _with_context_variant(examples, len(candidate_names), variant)
                for seed, examples in train_by_seed.items()
            }
            reranker = _train_diagnostic_reranker(variant_train_examples, args, len(candidate_names))
            cv_gate = _cross_seed_cv_gate(variant_train_by_seed, args, candidate_names)
            condition_rows = _summarize_diagnostic(
                variant_test_examples,
                variant_train_examples,
                reranker,
                candidate_names,
                args,
                cv_gate,
            )
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
                        "context_variant": variant,
                        "diagnostic_dim": DIAGNOSTIC_DIM,
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
    validation_examples = _collect_examples_with_diagnostics(
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
    test_examples = _collect_examples_with_diagnostics(
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
    for example in validation_examples:
        example["source_seed"] = seed
    for example in test_examples:
        example["source_seed"] = seed
    return validation_examples, test_examples


def _collect_examples_with_diagnostics(
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
    diagnostics_by_candidate: dict[str, list[list[float]]] = {}
    for name in candidate_names:
        losses, correct, diagnostics = _evaluate_selected_with_diagnostics(
            model,
            samples,
            selected_by_candidate[name],
            args.batch_size,
            device,
        )
        losses_by_candidate[name] = losses
        correct_by_candidate[name] = correct
        diagnostics_by_candidate[name] = diagnostics

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
            entropy, max_probability, logit_margin = diagnostics_by_candidate[name][sample_index]
            row[f"{name}_branch_entropy"] = round(entropy, 6)
            row[f"{name}_branch_max_probability"] = round(max_probability, 6)
            row[f"{name}_branch_margin"] = round(logit_margin, 6)
            object_tensor, mask_tensor = _selected_object_tensor(sample, selected_ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            base_context = _context_features(row, name, candidate_index, len(candidate_names), total_n, causal_k, args.budget)
            context_features.append([*base_context, entropy, max_probability, logit_margin])
        row["object_features"] = torch.stack(object_features)
        row["object_masks"] = torch.stack(object_masks)
        row["context_features"] = torch.tensor(context_features, dtype=torch.float32)
        examples.append(row)
    return examples


def _evaluate_selected_with_diagnostics(
    model: torch.nn.Module,
    samples,
    selected_ids_by_sample: list[list[str]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[int], list[list[float]]]:
    losses: list[float] = []
    correct: list[int] = []
    diagnostics: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_selected = selected_ids_by_sample[start : start + batch_size]
            batch, _, labels, _ = collate_selected_working_set_samples(batch_samples, batch_selected)
            batch = _move_batch(batch, device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, force_route="sparse")
            logits = prediction.branch_logits
            probabilities = torch.softmax(logits, dim=-1)
            batch_losses = F.cross_entropy(logits, labels, reduction="none")
            batch_correct = logits.argmax(dim=-1) == labels
            entropy = -(probabilities * (probabilities.clamp_min(1e-8)).log()).sum(dim=-1) / math.log(logits.size(-1))
            top2 = logits.topk(2, dim=-1).values
            margin = torch.tanh(top2[:, 0] - top2[:, 1])
            max_probability = probabilities.max(dim=-1).values
            losses.extend(round(float(value), 8) for value in batch_losses.detach().cpu())
            correct.extend(int(value) for value in batch_correct.detach().cpu())
            diagnostics.extend(
                [float(entropy_value), float(max_value), float(margin_value)]
                for entropy_value, max_value, margin_value in zip(
                    entropy.detach().cpu(),
                    max_probability.detach().cpu(),
                    margin.detach().cpu(),
                    strict=True,
                )
            )
    return losses, correct, diagnostics


def _train_diagnostic_reranker(
    examples: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_count: int,
) -> GeneratedSetReranker:
    objects, masks, context, loss_tensor = _example_tensors(examples, candidate_count)
    centered_loss = loss_tensor - loss_tensor.mean(dim=1, keepdim=True)
    scale = loss_tensor.std(dim=1, keepdim=True).clamp_min(1e-3)
    normalized_utilities = -(centered_loss / scale)
    reranker = GeneratedSetReranker(OBJECT_CANDIDATE_FEATURE_DIM, context.size(-1), args.reranker_hidden_dim)
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


def _with_context_variant(
    examples: list[dict[str, object]],
    candidate_count: int,
    variant: str,
) -> list[dict[str, object]]:
    if variant == "full":
        return examples
    transformed: list[dict[str, object]] = []
    generated_flag_index = candidate_count + 8
    diagnostic_start = candidate_count + 9
    for example in examples:
        row = dict(example)
        context = example["context_features"].clone()  # type: ignore[union-attr]
        if variant in {"no_identity", "no_selector_type", "diagnostics_only"}:
            context[:, :candidate_count] = 0.0
        if variant in {"no_selector_type", "diagnostics_only"}:
            context[:, generated_flag_index] = 0.0
        if variant == "diagnostics_only":
            context[:, candidate_count:diagnostic_start] = 0.0
        if variant == "set_only":
            context[:, diagnostic_start:] = 0.0
        row["context_features"] = context
        transformed.append(row)
    return transformed


def _summarize_diagnostic(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    reranker: nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
    cv_gate: dict[str, object],
) -> list[dict[str, object]]:
    static_mode = _best_static_mode(train_examples, BASE_MODES)
    generated_static_mode = _best_static_mode(train_examples, candidate_names)
    reranker_modes = _predict_modes(test_examples, reranker, len(candidate_names))
    train_reranker_modes = _predict_modes(train_examples, reranker, len(candidate_names))
    train_reranker_loss = _mean_policy_loss(train_examples, train_reranker_modes)
    train_static_loss = _mean_policy_loss(train_examples, [static_mode] * len(train_examples))
    margin_safe = train_reranker_loss + args.safe_margin < train_static_loss
    safe_modes = reranker_modes if margin_safe else [static_mode] * len(test_examples)
    cv_safe_modes = reranker_modes if bool(cv_gate["uses_reranker"]) else [static_mode] * len(test_examples)
    rows = [
        _policy_row(
            test_examples,
            policy="diagnostic_cross_seed_static_base_choice",
            selected_modes=[static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="diagnostic_cross_seed_static_generated_choice",
            selected_modes=[generated_static_mode] * len(test_examples),
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="diagnostic_cross_seed_generated_reranker",
            selected_modes=reranker_modes,
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="diagnostic_cross_seed_margin_safe_reranker",
            selected_modes=safe_modes,
            candidate_names=candidate_names,
        ),
        _policy_row(
            test_examples,
            policy="diagnostic_cross_seed_cv_safe_reranker",
            selected_modes=cv_safe_modes,
            candidate_names=candidate_names,
        ),
        _oracle_row(test_examples, "base_oracle", [str(example["best_base_mode"]) for example in test_examples], candidate_names),
        _oracle_row(test_examples, "generated_oracle", [str(example["best_mode"]) for example in test_examples], candidate_names),
    ]
    rows[3]["safe_uses_reranker"] = int(margin_safe)
    rows[4]["safe_uses_reranker"] = int(bool(cv_gate["uses_reranker"]))
    for row in rows:
        row["static_mode_from_train_seeds"] = static_mode
        row["static_generated_mode_from_train_seeds"] = generated_static_mode
        row["train_reranker_loss"] = round(train_reranker_loss, 6)
        row["train_static_loss"] = round(train_static_loss, 6)
        row["safe_margin"] = args.safe_margin
        row["cv_safe_margin"] = args.cv_safe_margin
        row["cv_min_win_rate"] = args.cv_min_win_rate
        row["cv_mean_delta"] = cv_gate["mean_delta"]
        row["cv_win_rate"] = cv_gate["win_rate"]
        row.setdefault("safe_uses_reranker", "")
    return rows


def _cross_seed_cv_gate(
    train_by_seed: dict[int, list[dict[str, object]]],
    args: argparse.Namespace,
    candidate_names: list[str],
) -> dict[str, object]:
    deltas: list[float] = []
    for gate_seed, gate_examples in train_by_seed.items():
        scorer_examples = [
            example
            for seed, examples in train_by_seed.items()
            if seed != gate_seed
            for example in examples
        ]
        if not scorer_examples:
            continue
        static_mode = _best_static_mode(scorer_examples, BASE_MODES)
        cv_reranker = _train_diagnostic_reranker(scorer_examples, args, len(candidate_names))
        reranker_modes = _predict_modes(gate_examples, cv_reranker, len(candidate_names))
        reranker_loss = _mean_policy_loss(gate_examples, reranker_modes)
        static_loss = _mean_policy_loss(gate_examples, [static_mode] * len(gate_examples))
        deltas.append(reranker_loss - static_loss)
    if not deltas:
        return {"uses_reranker": 0, "mean_delta": "", "win_rate": ""}
    mean_delta = sum(deltas) / len(deltas)
    win_rate = sum(delta + args.cv_safe_margin < 0.0 for delta in deltas) / len(deltas)
    uses_reranker = mean_delta + args.cv_safe_margin < 0.0 and win_rate >= args.cv_min_win_rate
    return {
        "uses_reranker": int(uses_reranker),
        "mean_delta": round(mean_delta, 6),
        "win_rate": round(win_rate, 6),
    }


if __name__ == "__main__":
    main()
