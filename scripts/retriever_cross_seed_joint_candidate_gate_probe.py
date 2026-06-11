from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_candidate_regret_gate_probe import (  # noqa: E402
    _float_token,
    _regret_targets,
)
from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import _train_cross_seed_regret_retriever  # noqa: E402
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _collect_seed_context,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class JointCandidateGate(nn.Module):
    """Score candidate working sets from explicit object-set state and compact context."""

    def __init__(self, object_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(object_dim),
            nn.Linear(object_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(context_dim + hidden_dim * 2 + 1),
            nn.Linear(context_dim + hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        self.safe_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        object_features: torch.Tensor,
        object_masks: torch.Tensor,
        context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, budget, object_dim = object_features.shape
        flat_objects = object_features.view(batch_size * candidate_count * budget, object_dim)
        encoded = self.object_encoder(flat_objects).view(batch_size, candidate_count, budget, -1)
        mask = object_masks.unsqueeze(-1).float()
        count = mask.sum(dim=2).clamp_min(1.0)
        mean_pool = (encoded * mask).sum(dim=2) / count
        masked_encoded = encoded.masked_fill(mask <= 0.0, -1e4)
        max_pool = masked_encoded.max(dim=2).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        count_feature = count.squeeze(-1) / max(float(budget), 1.0)
        features = torch.cat([context_features, mean_pool, max_pool, count_feature.unsqueeze(-1)], dim=-1)
        hidden = self.trunk(features.view(batch_size * candidate_count, -1))
        regret = self.regret_head(hidden).view(batch_size, candidate_count)
        log_var = self.log_var_head(hidden).view(batch_size, candidate_count).clamp(-6.0, 3.0)
        safe_logit = self.safe_head(hidden).view(batch_size, candidate_count)
        return regret, log_var, safe_logit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a joint object-set candidate gate for WPU v2 priority-1."
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
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--safe-bce-weight", type=float, default=0.75)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--harmful-accept-weight", type=float, default=0.5)
    parser.add_argument("--group-dro-weight", type=float, default=0.25)
    parser.add_argument("--safe-prob-thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--utility-margins", type=float, nargs="+", default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02])
    parser.add_argument("--risk-penalties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.005)
    parser.add_argument("--context-dropout", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_candidate_joint_gate.csv"))
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
        print(f"joint-candidate collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
                    "interaction_mode": args.interaction_mode,
                    "propagation_steps": args.propagation_steps,
                    "retriever_steps": args.retriever_steps,
                    "regret_retriever_steps": args.regret_retriever_steps,
                    "composition_steps": args.composition_steps,
                    "gate_steps": args.gate_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "no_harm_margin": args.no_harm_margin,
                }
            )
        rows.extend(condition_rows)
    return rows


def _candidate_tensors(
    examples: list[dict[str, object]],
    candidate_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_count = len(candidate_names)
    object_features = torch.stack([example["object_features"].clone() for example in examples])  # type: ignore[union-attr]
    object_masks = torch.stack([example["object_masks"].clone() for example in examples])  # type: ignore[union-attr]
    context_features = torch.stack(
        [example["context_features"][:, candidate_count:].clone() for example in examples]  # type: ignore[index]
    )
    return object_features.float(), object_masks.float(), context_features.float()


def _train_gate(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> JointCandidateGate:
    object_features, object_masks, context_features = _candidate_tensors(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    source_seeds = torch.tensor([int(example["source_seed"]) for example in examples], dtype=torch.long)
    unique_seeds = torch.unique(source_seeds)
    positive = safe.sum().clamp_min(1.0)
    negative = (1.0 - safe).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(0.25, 8.0)
    gate = JointCandidateGate(
        object_dim=object_features.size(-1),
        context_dim=context_features.size(-1),
        hidden_dim=args.gate_hidden_dim,
    )
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        train_objects, train_context = _perturb_inputs(object_features, context_features, args)
        pred_regret, pred_log_var, safe_logit = gate(train_objects, object_masks, train_context)
        group_losses = []
        for seed in unique_seeds:
            mask = source_seeds == seed
            group_losses.append(
                _selector_loss(
                    pred_regret[mask],
                    pred_log_var[mask],
                    safe_logit[mask],
                    regrets[mask],
                    safe[mask],
                    pos_weight,
                    args,
                )
            )
        group_loss = torch.stack(group_losses)
        loss = group_loss.mean() + args.group_dro_weight * group_loss.max()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gate.eval()


def _perturb_inputs(
    object_features: torch.Tensor,
    context_features: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    objects = object_features
    context = context_features
    if args.feature_noise_std > 0.0:
        objects = objects + torch.randn_like(objects) * args.feature_noise_std
        context = context + torch.randn_like(context) * args.feature_noise_std
    if args.context_dropout > 0.0:
        keep = (torch.rand_like(context) >= args.context_dropout).float()
        context = context * keep / max(1.0 - args.context_dropout, 1e-6)
    return objects, context


def _selector_loss(
    pred_regret: torch.Tensor,
    pred_log_var: torch.Tensor,
    safe_logit: torch.Tensor,
    regrets: torch.Tensor,
    safe: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pred_var = pred_log_var.exp().clamp_min(1e-4)
    regression = F.smooth_l1_loss(pred_regret, regrets)
    nll = 0.5 * (pred_log_var + (regrets - pred_regret).pow(2) / pred_var).mean()
    safe_loss = F.binary_cross_entropy_with_logits(safe_logit, safe, pos_weight=pos_weight)
    best_indices = regrets.argmin(dim=1)
    rank_loss = F.cross_entropy(-pred_regret, best_indices)
    unsafe = (regrets > args.no_harm_margin).float()
    accept_probability = torch.sigmoid(safe_logit) * torch.sigmoid(-pred_regret)
    harmful_loss = (accept_probability * unsafe).mean()
    return (
        args.regret_weight * regression
        + args.variance_weight * nll
        + args.safe_bce_weight * safe_loss
        + args.ranking_weight * rank_loss
        + args.harmful_accept_weight * harmful_loss
    )


def _predict_modes(
    examples: list[dict[str, object]],
    gate: JointCandidateGate,
    candidate_names: list[str],
    *,
    safe_prob_threshold: float,
    utility_margin: float,
    risk_penalty: float,
) -> tuple[list[str], dict[str, float]]:
    object_features, object_masks, context_features = _candidate_tensors(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    with torch.no_grad():
        pred_regret, pred_log_var, safe_logit = gate(object_features, object_masks, context_features)
    sigma = (0.5 * pred_log_var).exp()
    safe_prob = torch.sigmoid(safe_logit)
    score = pred_regret + risk_penalty * sigma
    selected_indices = score.argmin(dim=1)
    selected_modes: list[str] = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        use_candidate = (
            float(safe_prob[row_index, candidate_index].item()) >= safe_prob_threshold
            and float(score[row_index, candidate_index].item()) < -utility_margin
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
        "predicted_sigma_mean": round(float(sigma.mean().item()), 6),
        "predicted_safe_probability_mean": round(float(safe_prob.mean().item()), 6),
        "deployment_safe_prob_threshold": round(float(safe_prob_threshold), 6),
        "deployment_utility_margin": round(float(utility_margin), 6),
        "deployment_risk_penalty": round(float(risk_penalty), 6),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gate: JointCandidateGate,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    default_modes, default_metrics = _predict_modes(
        test_examples,
        gate,
        candidate_names,
        safe_prob_threshold=0.5,
        utility_margin=0.0,
        risk_penalty=0.0,
    )
    risk_modes, risk_metrics = _predict_modes(
        test_examples,
        gate,
        candidate_names,
        safe_prob_threshold=0.5,
        utility_margin=0.0,
        risk_penalty=0.5,
    )
    train_default_modes, train_default_metrics = _predict_modes(
        train_examples,
        gate,
        candidate_names,
        safe_prob_threshold=0.5,
        utility_margin=0.0,
        risk_penalty=0.0,
    )
    configs = _deployment_configs(args)
    selected_config, selected_train_metrics = _select_train_deployment(train_examples, gate, candidate_names, args, configs)
    selected_modes, selected_metrics = _predict_modes(
        test_examples,
        gate,
        candidate_names,
        safe_prob_threshold=float(selected_config["safe_prob_threshold"]),
        utility_margin=float(selected_config["utility_margin"]),
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
        ("joint_candidate_gate", default_modes, default_metrics),
        ("joint_uncertainty_gate", risk_modes, risk_metrics),
        ("train_selected_joint_candidate_gate", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in configs:
        if config["name"] in {"joint_candidate_gate", "joint_uncertainty_gate"}:
            continue
        modes, metrics = _predict_modes(
            test_examples,
            gate,
            candidate_names,
            safe_prob_threshold=float(config["safe_prob_threshold"]),
            utility_margin=float(config["utility_margin"]),
            risk_penalty=float(config["risk_penalty"]),
        )
        policies.append((str(config["name"]), modes, metrics))

    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_joint_candidate_gate_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_joint_candidate_accept_rate"] = train_default_metrics["accept_rate"]
        rows.append(row)
    return rows


def _deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [
        {
            "name": "joint_candidate_gate",
            "safe_prob_threshold": 0.5,
            "utility_margin": 0.0,
            "risk_penalty": 0.0,
        },
        {
            "name": "joint_uncertainty_gate",
            "safe_prob_threshold": 0.5,
            "utility_margin": 0.0,
            "risk_penalty": 0.5,
        },
    ]
    for threshold in _unique_floats(args.safe_prob_thresholds):
        for margin in _unique_floats(args.utility_margins):
            for risk_penalty in _unique_floats(args.risk_penalties):
                name = (
                    f"joint_gate_p{_float_token(threshold)}"
                    f"_m{_float_token(margin)}"
                    f"_r{_float_token(risk_penalty)}"
                )
                configs.append(
                    {
                        "name": name,
                        "safe_prob_threshold": threshold,
                        "utility_margin": margin,
                        "risk_penalty": risk_penalty,
                    }
                )
    return configs


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    gate: JointCandidateGate,
    candidate_names: list[str],
    args: argparse.Namespace,
    configs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, float]]:
    static_loss = _mean_policy_loss(train_examples, ["learned"] * len(train_examples))
    oracle_modes = [str(example["best_mode"]) for example in train_examples]
    oracle_loss = _mean_policy_loss(train_examples, oracle_modes)
    oracle_gap = max(static_loss - oracle_loss, 1e-8)
    evaluated: list[tuple[dict[str, object], dict[str, float]]] = []
    for config in configs:
        modes, metrics = _predict_modes(
            train_examples,
            gate,
            candidate_names,
            safe_prob_threshold=float(config["safe_prob_threshold"]),
            utility_margin=float(config["utility_margin"]),
            risk_penalty=float(config["risk_penalty"]),
        )
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


if __name__ == "__main__":
    main()
