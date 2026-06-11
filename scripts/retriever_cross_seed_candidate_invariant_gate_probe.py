from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class InvariantCandidateGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.safe_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, feature_dim = features.shape
        hidden = self.trunk(features.view(batch_size * candidate_count, feature_dim))
        regret = self.regret_head(hidden).view(batch_size, candidate_count)
        safe_logit = self.safe_head(hidden).view(batch_size, candidate_count)
        return regret, safe_logit


class FeatureNormalizer:
    def __init__(self, features: torch.Tensor) -> None:
        self.mean = features.mean(dim=(0, 1), keepdim=True)
        self.std = features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-4)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a seed-invariant candidate selector for WPU v2 P1."
    )
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
    parser.add_argument("--gate-steps", type=int, default=900)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--safe-bce-weight", type=float, default=1.0)
    parser.add_argument("--regret-weight", type=float, default=0.35)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--harmful-accept-weight", type=float, default=0.75)
    parser.add_argument("--group-dro-weight", type=float, default=0.75)
    parser.add_argument("--group-variance-weight", type=float, default=0.10)
    parser.add_argument("--calibration-variance-weight", type=float, default=0.05)
    parser.add_argument("--safe-prob-thresholds", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--utility-margins", type=float, nargs="+", default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02])
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_candidate_invariant_gate.csv"))
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
    candidate_names = [*base_candidate_names, *COMPOSITION_MODES]
    contexts = {}
    for seed in args.seeds:
        print(f"invariant-candidate collect seed={seed} N={total_n} K={causal_k}", flush=True)
        contexts[seed] = _collect_seed_context(background_objects, causal_obstacles, seed, args, base_candidate_names)

    rows: list[dict[str, object]] = []
    for heldout_seed in args.seeds:
        train_contexts = [context for seed, context in contexts.items() if seed != heldout_seed]
        heldout = contexts[heldout_seed]
        object_retriever = _train_cross_seed_regret_retriever(train_contexts, base_candidate_names, args)
        composition_prior = _train_composition_prior(train_contexts, base_candidate_names, args)
        train_examples = []
        for seed, context in contexts.items():
            if seed == heldout_seed:
                continue
            seed_examples = _collect_examples(context, object_retriever, composition_prior, candidate_names, args, split="validation")
            for example in seed_examples:
                example["source_seed"] = seed
            train_examples.extend(seed_examples)
        test_examples = _collect_examples(heldout, object_retriever, composition_prior, candidate_names, args, split="test")
        gate, normalizer = _train_gate(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, gate, normalizer, candidate_names, args)
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
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "regret_retriever_steps": args.regret_retriever_steps,
                    "composition_steps": args.composition_steps,
                    "gate_steps": args.gate_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "group_dro_weight": args.group_dro_weight,
                    "group_variance_weight": args.group_variance_weight,
                    "calibration_variance_weight": args.calibration_variance_weight,
                    "no_harm_margin": args.no_harm_margin,
                }
            )
        rows.extend(condition_rows)
    return rows


def _candidate_features(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    candidate_count = len(candidate_names)
    return torch.stack([example["context_features"][:, candidate_count:].clone() for example in examples])  # type: ignore[index]


def _regret_targets(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    rows = []
    for example in examples:
        baseline = float(example["learned_loss"])
        rows.append([float(example[f"{name}_loss"]) - baseline for name in candidate_names])
    return torch.tensor(rows, dtype=torch.float32)


def _train_gate(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> tuple[InvariantCandidateGate, FeatureNormalizer]:
    raw_features = _candidate_features(examples, candidate_names)
    normalizer = FeatureNormalizer(raw_features)
    features = normalizer.transform(raw_features)
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    source_seeds = torch.tensor([int(example["source_seed"]) for example in examples], dtype=torch.long)
    unique_seeds = torch.unique(source_seeds)
    positive = safe.sum().clamp_min(1.0)
    negative = (1.0 - safe).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(0.25, 8.0)
    gate = InvariantCandidateGate(features.size(-1), args.gate_hidden_dim)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        train_features = _perturb_features(features, args)
        pred_regret, safe_logit = gate(train_features)
        group_losses = []
        safe_prob_means = []
        regret_means = []
        for seed in unique_seeds:
            mask = source_seeds == seed
            group_losses.append(_selector_loss(pred_regret[mask], safe_logit[mask], regrets[mask], safe[mask], pos_weight, args))
            safe_prob_means.append(torch.sigmoid(safe_logit[mask]).mean())
            regret_means.append(pred_regret[mask].mean())
        group_loss_tensor = torch.stack(group_losses)
        calibration_var = torch.stack(safe_prob_means).var(unbiased=False) + torch.stack(regret_means).var(unbiased=False)
        loss = (
            group_loss_tensor.mean()
            + args.group_dro_weight * group_loss_tensor.max()
            + args.group_variance_weight * group_loss_tensor.var(unbiased=False)
            + args.calibration_variance_weight * calibration_var
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gate.eval(), normalizer


def _selector_loss(
    pred_regret: torch.Tensor,
    safe_logit: torch.Tensor,
    regrets: torch.Tensor,
    safe: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    regret_loss = F.smooth_l1_loss(pred_regret, regrets)
    safe_loss = F.binary_cross_entropy_with_logits(safe_logit, safe, pos_weight=pos_weight)
    best_indices = regrets.argmin(dim=1)
    rank_loss = F.cross_entropy(-pred_regret, best_indices)
    unsafe = (regrets > args.no_harm_margin).float()
    accept_probability = torch.sigmoid(safe_logit) * torch.sigmoid(-pred_regret)
    harmful_loss = (accept_probability * unsafe).mean()
    return (
        args.regret_weight * regret_loss
        + args.safe_bce_weight * safe_loss
        + args.ranking_weight * rank_loss
        + args.harmful_accept_weight * harmful_loss
    )


def _perturb_features(features: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    output = features
    if args.feature_noise_std > 0.0:
        output = output + torch.randn_like(output) * args.feature_noise_std
    if args.feature_dropout > 0.0:
        keep = (torch.rand_like(output) >= args.feature_dropout).float()
        output = output * keep / max(1.0 - args.feature_dropout, 1e-6)
    return output


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gate: InvariantCandidateGate,
    normalizer: FeatureNormalizer,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    train_configs = _deployment_configs(args)
    selected_config, train_metrics = _select_train_deployment(
        train_examples,
        gate,
        normalizer,
        candidate_names,
        args,
        train_configs,
    )
    policies: list[tuple[str, list[str], dict[str, float]]] = [
        ("static_learned_interaction", static_modes, {}),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in train_configs:
        modes, metrics = _predict_modes(test_examples, gate, normalizer, candidate_names, args, config)
        policies.append((str(config["name"]), modes, metrics))
    selected_modes, selected_metrics = _predict_modes(test_examples, gate, normalizer, candidate_names, args, selected_config)
    selected_metrics.update(
        {
            "selection_train_loss": train_metrics["loss"],
            "selection_train_gap_closure": train_metrics["gap_closure"],
            "selection_train_accept_rate": train_metrics["accept_rate"],
            "selection_train_harmful_accept_rate": train_metrics["harmful_accept_rate"],
            "selection_train_policy": str(selected_config["name"]),
        }
    )
    policies.append(("train_selected_invariant_gate", selected_modes, selected_metrics))

    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        rows.append(row)
    return rows


def _deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = []
    for threshold in _unique_floats(args.safe_prob_thresholds):
        for margin in _unique_floats(args.utility_margins):
            configs.append(
                {
                    "name": f"invariant_gate_p{_float_token(threshold)}_m{_float_token(margin)}",
                    "safe_prob_threshold": threshold,
                    "utility_margin": margin,
                }
            )
    return configs


def _predict_modes(
    examples: list[dict[str, object]],
    gate: InvariantCandidateGate,
    normalizer: FeatureNormalizer,
    candidate_names: list[str],
    args: argparse.Namespace,
    config: dict[str, object],
) -> tuple[list[str], dict[str, float]]:
    raw_features = _candidate_features(examples, candidate_names)
    features = normalizer.transform(raw_features)
    regrets = _regret_targets(examples, candidate_names)
    with torch.no_grad():
        pred_regret, safe_logit = gate(features)
    safe_prob = torch.sigmoid(safe_logit)
    threshold = float(config["safe_prob_threshold"])
    margin = float(config["utility_margin"])
    masked_score = pred_regret.masked_fill(safe_prob < threshold, 1e6)
    selected_indices = masked_score.argmin(dim=1)
    selected_modes = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        use_candidate = (
            float(safe_prob[row_index, candidate_index].item()) >= threshold
            and float(pred_regret[row_index, candidate_index].item()) < -margin
        )
        selected_modes.append(candidate_names[candidate_index] if use_candidate else "learned")
        accepted.append(float(use_candidate))
        harmful.append(float(use_candidate and float(regrets[row_index, candidate_index].item()) > 0.0))
    pred = pred_regret.flatten()
    target = regrets.flatten()
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    corr = float(
        (pred_centered * target_centered).mean()
        / (pred_centered.std(unbiased=False).clamp_min(1e-6) * target_centered.std(unbiased=False).clamp_min(1e-6))
    )
    return selected_modes, {
        "accept_rate": round(mean(accepted), 6),
        "harmful_accept_rate": round(mean(harmful), 6),
        "regret_corr": round(corr, 6),
        "predicted_regret_mean": round(float(pred_regret.mean().item()), 6),
        "predicted_safe_prob_mean": round(float(safe_prob.mean().item()), 6),
        "deployment_safe_prob_threshold": round(threshold, 6),
        "deployment_utility_margin": round(margin, 6),
    }


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    gate: InvariantCandidateGate,
    normalizer: FeatureNormalizer,
    candidate_names: list[str],
    args: argparse.Namespace,
    configs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, float]]:
    static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    oracle_modes = [str(example["best_mode"]) for example in train_examples]
    oracle_loss = _mean_policy_loss(train_examples, oracle_modes)
    oracle_gap = max(static_loss - oracle_loss, 1e-8)
    evaluated = []
    for config in configs:
        modes, metrics = _predict_modes(train_examples, gate, normalizer, candidate_names, args, config)
        loss = _mean_policy_loss(train_examples, modes)
        metrics = dict(metrics)
        metrics["loss"] = round(loss, 6)
        metrics["gap_closure"] = round((static_loss - loss) / oracle_gap, 6)
        evaluated.append((config, metrics))
    safe = [
        item
        for item in evaluated
        if item[1]["harmful_accept_rate"] <= args.selection_harmful_limit
        and item[1]["gap_closure"] > 0.0
    ]
    candidates = safe or evaluated
    return max(candidates, key=lambda item: (item[1]["gap_closure"], -item[1]["harmful_accept_rate"]))


def _unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        rounded = round(float(value), 6)
        if rounded not in out:
            out.append(rounded)
    return out


def _float_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "neg").replace(".", "p")


if __name__ == "__main__":
    main()
