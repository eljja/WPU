from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_candidate_regret_gate_probe import (  # noqa: E402
    CandidateRegretGate,
    _deployment_configs,
    _predict_modes,
    _select_train_deployment,
)
from scripts.retriever_cross_seed_composition_regret_probe import (  # noqa: E402
    _composition_selected_ids,
    _train_composition_prior,
)
from scripts.retriever_cross_seed_joint_candidate_generator_probe import (  # noqa: E402
    _learned_generated_ids,
    _train_weighted_generator,
)
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _evaluate_policy,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _context_features,
    _mean_policy_loss,
    _policy_row,
    _selected_geometry_features,
)
from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    _selected_object_tensor,
    _write_csv,
)
from scripts.staged_regret_hybrid import _move_batch  # noqa: E402
from wpu.data.working_set_physics import collate_selected_working_set_samples  # noqa: E402


class NormalizedCandidateRegretGate(torch.nn.Module):
    def __init__(self, gate: CandidateRegretGate, features: torch.Tensor) -> None:
        super().__init__()
        self.gate = gate
        self.register_buffer("mean", features.mean(dim=(0, 1), keepdim=True))
        self.register_buffer("std", features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-4))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate((features - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a verified P1 candidate controller. The selector sees "
            "candidate set descriptors plus label-free sparse/local-dense "
            "propagation consistency signatures."
        )
    )
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--learned-generated-candidates", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--regret-retriever-steps", type=int, default=600)
    parser.add_argument("--composition-steps", type=int, default=600)
    parser.add_argument("--gate-steps", type=int, default=700)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--regret-gain-weight", type=float, default=1.5)
    parser.add_argument("--risk-penalty", type=float, default=0.5)
    parser.add_argument("--reject-margin", type=float, default=0.0)
    parser.add_argument("--sweep-reject-margins", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--sweep-risk-penalties", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--bce-weight", type=float, default=0.35)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--harmful-accept-weight", type=float, default=0.5)
    parser.add_argument("--safe-ranking-weight", type=float, default=0.1)
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--feature-dropout", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_verified_candidate_controller.csv"))
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
    learned_candidate_names = [f"learned_generated_{index}" for index in range(args.learned_generated_candidates)]
    candidate_names = [*base_candidate_names, *learned_candidate_names, *COMPOSITION_MODES]
    contexts = {}
    for seed in args.seeds:
        print(f"verified-controller collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, base_candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, base_candidate_names, args)
        generator = _train_weighted_generator(train_contexts, base_candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, base_candidate_names, args)
        train_examples = []
        for seed, context in contexts.items():
            if seed == heldout_seed:
                continue
            examples = _collect_verified_examples(
                context,
                object_retriever,
                generator,
                composition_prior,
                candidate_names,
                learned_candidate_names,
                args,
                split="validation",
            )
            for example in examples:
                example["source_seed"] = seed
            train_examples.extend(examples)
        test_examples = _collect_verified_examples(
            heldout,
            object_retriever,
            generator,
            composition_prior,
            candidate_names,
            learned_candidate_names,
            args,
            split="test",
        )
        gate = _train_gate(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, gate, candidate_names, args)
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
                    "learned_generated_candidates": args.learned_generated_candidates,
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "regret_retriever_steps": args.regret_retriever_steps,
                    "composition_steps": args.composition_steps,
                    "gate_steps": args.gate_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "verification_feature_count": 13,
                }
            )
        rows.extend(condition_rows)
    return rows


def _collect_verified_examples(
    context: dict[str, object],
    object_retriever: torch.nn.Module,
    generator: torch.nn.Module,
    composition_prior: torch.nn.Module,
    candidate_names: list[str],
    learned_candidate_names: list[str],
    args: argparse.Namespace,
    *,
    split: str,
) -> list[dict[str, object]]:
    samples = context[f"{split}_samples"]
    candidates = {name: list(selected) for name, selected in context[f"{split}_candidates"].items()}  # type: ignore[union-attr]
    model = context["model"]
    device = context["device"]
    losses = {name: tuple(values) for name, values in context[f"{split}_losses"].items()}  # type: ignore[union-attr]
    split_offset = 5_000 if split == "validation" else 10_000

    for learned_index, name in enumerate(learned_candidate_names):
        selected = [
            _learned_generated_ids(
                sample,
                args.budget,
                generator,
                variant=learned_index,
                seed=split_offset + sample_index,
            )
            for sample_index, sample in enumerate(samples)  # type: ignore[arg-type]
        ]
        candidates[name] = selected
        losses[name] = _evaluate_policy(model, samples, selected, args, device)  # type: ignore[arg-type]

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

    verification_features = _verification_features_by_candidate(
        model,
        samples,  # type: ignore[arg-type]
        candidates,
        candidate_names,
        args,
        device,
    )

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
            row[f"{name}_pair_density"] = round(_selected_pair_density_safe(sample.state, selected_ids), 6)
            for feature_name, value in _selected_geometry_features(sample, selected_ids).items():
                row[f"{name}_{feature_name}"] = round(value, 6)
            object_tensor, mask_tensor = _selected_object_tensor(sample, selected_ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            context_features.append(
                [
                    *_context_features(row, name, candidate_index, len(candidate_names), args),
                    *verification_features[name][sample_index],
                ]
            )
        row["object_features"] = torch.stack(object_features)
        row["object_masks"] = torch.stack(object_masks)
        row["context_features"] = torch.tensor(context_features, dtype=torch.float32)
        examples.append(row)
    return examples


def _selected_pair_density_safe(state, selected_ids: list[str]) -> float:
    from scripts.learned_retriever_probe import _selected_pair_density

    return _selected_pair_density(
        state,
        [object_id for object_id in selected_ids if object_id.startswith("obstacle_")],
    )


def _verification_features_by_candidate(
    model: torch.nn.Module,
    samples,
    candidates: dict[str, list[list[str]]],
    candidate_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[list[float]]]:
    return {
        name: _candidate_verification_features(model, samples, candidates[name], args.batch_size, device)
        for name in candidate_names
    }


def _candidate_verification_features(
    model: torch.nn.Module,
    samples,
    selected_ids_by_sample: list[list[str]],
    batch_size: int,
    device: torch.device,
) -> list[list[float]]:
    features: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_selected = selected_ids_by_sample[start : start + batch_size]
            batch, _, _, _ = collate_selected_working_set_samples(batch_samples, batch_selected)
            batch = _move_batch(batch, device)
            sparse = model(batch, num_branches=3, force_route="sparse")
            dense = model(batch, num_branches=3, force_route="local_dense")
            sparse_probs = sparse.branch_probabilities.float()
            dense_probs = dense.branch_probabilities.float()
            sparse_conf, sparse_margin, sparse_entropy = _probability_signature(sparse_probs)
            dense_conf, dense_margin, dense_entropy = _probability_signature(dense_probs)
            prob_l1 = (sparse_probs - dense_probs).abs().mean(dim=1)
            top_agree = (sparse_probs.argmax(dim=1) == dense_probs.argmax(dim=1)).float()
            sparse_delta = sparse.object_delta.float()
            dense_delta = dense.object_delta.float()
            sparse_norm = _scaled_norm(sparse_delta)
            dense_norm = _scaled_norm(dense_delta)
            delta_gap = _scaled_norm(sparse_delta - dense_delta)
            sparse_finite = torch.isfinite(sparse_delta).float().flatten(1).mean(dim=1)
            dense_finite = torch.isfinite(dense_delta).float().flatten(1).mean(dim=1)
            for index in range(sparse_probs.size(0)):
                features.append(
                    [
                        float(sparse_conf[index].item()),
                        float(sparse_entropy[index].item()),
                        float(sparse_margin[index].item()),
                        float(dense_conf[index].item()),
                        float(dense_entropy[index].item()),
                        float(dense_margin[index].item()),
                        float(prob_l1[index].item()),
                        float(top_agree[index].item()),
                        float(sparse_norm[index].item()),
                        float(dense_norm[index].item()),
                        float(delta_gap[index].item()),
                        float(sparse_finite[index].item()),
                        float(dense_finite[index].item()),
                    ]
                )
    return features


def _probability_signature(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_probs = probs.sort(dim=1, descending=True).values
    confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
    return confidence, margin, entropy


def _scaled_norm(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten(1)
    return flat.norm(dim=1) / max(float(flat.size(1)) ** 0.5, 1.0)


def _candidate_features(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    candidate_count = len(candidate_names)
    return torch.stack([example["context_features"][:, candidate_count:].clone() for example in examples])  # type: ignore[index]


def _regret_targets(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    rows = []
    for example in examples:
        baseline = float(example["learned_loss"])
        rows.append([float(example[f"{name}_loss"]) - baseline for name in candidate_names])
    return torch.tensor(rows, dtype=torch.float32)


def _perturb_features(features: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    output = features
    if args.feature_noise_std > 0.0:
        output = output + torch.randn_like(output) * args.feature_noise_std
    if args.feature_dropout > 0.0:
        keep = (torch.rand_like(output) >= args.feature_dropout).float()
        output = output * keep / max(1.0 - args.feature_dropout, 1e-6)
    return output


def _train_gate(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> NormalizedCandidateRegretGate:
    raw_features = _candidate_features(examples, candidate_names)
    normalizer_probe = NormalizedCandidateRegretGate(CandidateRegretGate(raw_features.size(-1), args.gate_hidden_dim), raw_features)
    features = (raw_features - normalizer_probe.mean) / normalizer_probe.std
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    gate = normalizer_probe.gate
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        train_features = _perturb_features(features, args)
        pred_mean, pred_log_var = gate(train_features)
        pred_var = pred_log_var.exp().clamp_min(1e-4)
        regression = F.smooth_l1_loss(pred_mean, regrets)
        nll = 0.5 * (pred_log_var + (regrets - pred_mean).pow(2) / pred_var).mean()
        no_harm = F.binary_cross_entropy_with_logits(-pred_mean, safe)
        loss = regression + args.variance_weight * nll + args.bce_weight * no_harm
        if args.harmful_accept_weight > 0.0:
            unsafe = (regrets > args.no_harm_margin).float()
            accept_probability = torch.sigmoid(-pred_mean)
            loss = loss + args.harmful_accept_weight * (accept_probability * unsafe).mean()
        if args.safe_ranking_weight > 0.0:
            best_indices = regrets.argmin(dim=1)
            loss = loss + args.safe_ranking_weight * F.cross_entropy(-pred_mean, best_indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    normalizer_probe.eval()
    return normalizer_probe


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gate: torch.nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    regret_modes, regret_metrics = _predict_modes(test_examples, gate, candidate_names, args, risk_adjusted=False)
    risk_modes, risk_metrics = _predict_modes(test_examples, gate, candidate_names, args, risk_adjusted=True)
    train_regret_modes, train_regret_metrics = _predict_modes(train_examples, gate, candidate_names, args, risk_adjusted=False)
    train_risk_modes, train_risk_metrics = _predict_modes(train_examples, gate, candidate_names, args, risk_adjusted=True)
    deployment_configs = _deployment_configs(args)
    selected_config, selected_train_metrics = _select_train_deployment(
        train_examples,
        gate,
        candidate_names,
        args,
        deployment_configs,
    )
    selected_modes, selected_metrics = _predict_modes(
        test_examples,
        gate,
        candidate_names,
        args,
        risk_adjusted=bool(selected_config["risk_adjusted"]),
        reject_margin=float(selected_config["reject_margin"]),
        risk_penalty=float(selected_config["risk_penalty"]),
    )
    selected_metrics.update(
        {
            "selection_train_loss": selected_train_metrics["loss"],
            "selection_train_gap_closure": selected_train_metrics["gap_closure"],
            "selection_train_accept_rate": selected_train_metrics["accept_rate"],
            "selection_train_harmful_accept_rate": selected_train_metrics["harmful_accept_rate"],
            "selection_train_policy": selected_config["name"],
        }
    )
    policies = [
        ("static_learned_interaction", static_modes, {}),
        ("verified_candidate_regret_gate", regret_modes, regret_metrics),
        ("verified_uncertainty_regret_gate", risk_modes, risk_metrics),
        ("train_selected_verified_candidate_controller", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in deployment_configs:
        name = f"verified_{config['name']}"
        if name in {"verified_candidate_regret_gate", "verified_uncertainty_regret_gate"}:
            continue
        modes, metrics = _predict_modes(
            test_examples,
            gate,
            candidate_names,
            args,
            risk_adjusted=bool(config["risk_adjusted"]),
            reject_margin=float(config["reject_margin"]),
            risk_penalty=float(config["risk_penalty"]),
        )
        policies.append((name, modes, metrics))

    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_verified_candidate_regret_loss"] = round(_mean_policy_loss(train_examples, train_regret_modes), 6)
        row["train_verified_uncertainty_regret_loss"] = round(_mean_policy_loss(train_examples, train_risk_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_verified_candidate_accept_rate"] = train_regret_metrics["accept_rate"]
        row["train_verified_uncertainty_accept_rate"] = train_risk_metrics["accept_rate"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
