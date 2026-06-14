from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_candidate_regret_gate_probe import (  # noqa: E402
    _deployment_configs,
)
from scripts.retriever_cross_seed_composition_regret_probe import (  # noqa: E402
    _train_composition_prior,
)
from scripts.retriever_cross_seed_joint_candidate_generator_probe import (  # noqa: E402
    _train_weighted_generator,
)
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _collect_seed_context,
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    OBJECT_DIM,
    _example_tensors,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_cross_seed_verified_candidate_controller_probe import (  # noqa: E402
    _collect_verified_examples,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class JointUtilityVerifier(nn.Module):
    def __init__(
        self,
        object_dim: int,
        context_dim: int,
        hidden_dim: int,
        object_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("object_mean", object_features.mean(dim=(0, 1, 2), keepdim=True))
        self.register_buffer(
            "object_std",
            object_features.std(dim=(0, 1, 2), keepdim=True, unbiased=False).clamp_min(1e-4),
        )
        self.register_buffer("context_mean", context_features.mean(dim=(0, 1), keepdim=True))
        self.register_buffer(
            "context_std",
            context_features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-4),
        )
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(object_dim),
            nn.Linear(object_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.regret_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        self.safe_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        objects: torch.Tensor,
        masks: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = (objects - self.object_mean) / self.object_std
        context = (context - self.context_mean) / self.context_std
        batch_size, candidate_count, budget, object_dim = objects.shape
        flat_objects = objects.reshape(batch_size * candidate_count * budget, object_dim)
        encoded_objects = self.object_encoder(flat_objects).reshape(batch_size, candidate_count, budget, -1)
        mask = masks.float().unsqueeze(-1)
        pooled_objects = (encoded_objects * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        encoded_context = self.context_encoder(context)
        hidden = self.trunk(torch.cat([pooled_objects, encoded_context], dim=-1))
        return (
            self.regret_head(hidden).squeeze(-1),
            self.log_var_head(hidden).squeeze(-1).clamp(-6.0, 3.0),
            self.safe_head(hidden).squeeze(-1),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a P1 joint utility verifier. The verifier jointly encodes "
            "candidate object sets, compact context, and sparse/local-dense "
            "verification signatures, then predicts regret, uncertainty, and "
            "no-harm safety for deployment."
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
    parser.add_argument("--verifier-steps", type=int, default=700)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--verifier-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--verifier-lr", type=float, default=3e-3)
    parser.add_argument("--regret-gain-weight", type=float, default=1.5)
    parser.add_argument("--utility-temperature", type=float, default=0.5)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--soft-utility-weight", type=float, default=0.35)
    parser.add_argument("--safe-bce-weight", type=float, default=0.45)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--harmful-accept-weight", type=float, default=0.3)
    parser.add_argument("--risk-penalty", type=float, default=0.5)
    parser.add_argument("--reject-margin", type=float, default=0.0)
    parser.add_argument("--sweep-reject-margins", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--sweep-risk-penalties", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--safe-thresholds", type=float, nargs="+", default=[0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_joint_utility_verifier.csv"))
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
        print(f"joint-utility-verifier collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
        verifier = _train_verifier(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, verifier, candidate_names, args)
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
                    "verifier_steps": args.verifier_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                }
            )
        rows.extend(condition_rows)
    return rows


def _train_verifier(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> JointUtilityVerifier:
    objects, masks, context, losses = _example_tensors(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    centered = losses - losses.mean(dim=1, keepdim=True)
    scale = losses.std(dim=1, keepdim=True).clamp_min(1e-3)
    utilities = -(centered / scale)
    soft_targets = F.softmax(utilities / max(args.utility_temperature, 1e-4), dim=1)
    best_indices = losses.argmin(dim=1)
    verifier = JointUtilityVerifier(OBJECT_DIM, context.size(-1), args.verifier_hidden_dim, objects, context)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.verifier_lr)
    verifier.train()
    for _ in range(args.verifier_steps):
        pred_regret, pred_log_var, safe_logit = verifier(objects, masks, context)
        pred_var = pred_log_var.exp().clamp_min(1e-4)
        regression = F.smooth_l1_loss(pred_regret, regrets)
        nll = 0.5 * (pred_log_var + (regrets - pred_regret).pow(2) / pred_var).mean()
        ranking = F.cross_entropy(-pred_regret, best_indices)
        soft_utility = -(soft_targets * F.log_softmax(-pred_regret, dim=1)).sum(dim=1).mean()
        safe_bce = F.binary_cross_entropy_with_logits(safe_logit, safe)
        unsafe = (regrets > args.no_harm_margin).float()
        harmful_accept = (torch.sigmoid(safe_logit) * unsafe).mean()
        loss = (
            args.regression_weight * regression
            + args.variance_weight * nll
            + args.ranking_weight * ranking
            + args.soft_utility_weight * soft_utility
            + args.safe_bce_weight * safe_bce
            + args.harmful_accept_weight * harmful_accept
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return verifier.eval()


def _regret_targets(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    rows = []
    for example in examples:
        baseline = float(example["learned_loss"])
        rows.append([float(example[f"{name}_loss"]) - baseline for name in candidate_names])
    return torch.tensor(rows, dtype=torch.float32)


def _predict_modes(
    examples: list[dict[str, object]],
    verifier: JointUtilityVerifier,
    candidate_names: list[str],
    args: argparse.Namespace,
    *,
    risk_adjusted: bool,
    reject_margin: float,
    risk_penalty: float,
    safe_threshold: float,
) -> tuple[list[str], dict[str, float]]:
    objects, masks, context, _ = _example_tensors(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    with torch.no_grad():
        pred_regret, pred_log_var, safe_logit = verifier(objects, masks, context)
    sigma = (0.5 * pred_log_var).exp()
    safe_probability = torch.sigmoid(safe_logit)
    score = pred_regret + risk_penalty * sigma if risk_adjusted else pred_regret
    selected_indices = score.argmin(dim=1)
    selected_modes: list[str] = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        use_candidate = (
            float(score[row_index, candidate_index].item()) < -reject_margin
            and float(safe_probability[row_index, candidate_index].item()) >= safe_threshold
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
        "predicted_safe_probability_mean": round(float(safe_probability.mean().item()), 6),
        "deployment_reject_margin": round(float(reject_margin), 6),
        "deployment_risk_penalty": round(float(risk_penalty), 6),
        "deployment_risk_adjusted": float(risk_adjusted),
        "deployment_safe_threshold": round(float(safe_threshold), 6),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    verifier: JointUtilityVerifier,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    default_safe_threshold = 0.5
    utility_modes, utility_metrics = _predict_modes(
        test_examples,
        verifier,
        candidate_names,
        args,
        risk_adjusted=False,
        reject_margin=float(args.reject_margin),
        risk_penalty=float(args.risk_penalty),
        safe_threshold=default_safe_threshold,
    )
    safe_modes, safe_metrics = _predict_modes(
        test_examples,
        verifier,
        candidate_names,
        args,
        risk_adjusted=True,
        reject_margin=float(args.reject_margin),
        risk_penalty=float(args.risk_penalty),
        safe_threshold=default_safe_threshold,
    )
    train_utility_modes, train_utility_metrics = _predict_modes(
        train_examples,
        verifier,
        candidate_names,
        args,
        risk_adjusted=False,
        reject_margin=float(args.reject_margin),
        risk_penalty=float(args.risk_penalty),
        safe_threshold=default_safe_threshold,
    )
    train_safe_modes, train_safe_metrics = _predict_modes(
        train_examples,
        verifier,
        candidate_names,
        args,
        risk_adjusted=True,
        reject_margin=float(args.reject_margin),
        risk_penalty=float(args.risk_penalty),
        safe_threshold=default_safe_threshold,
    )
    deployment_configs = _joint_deployment_configs(args)
    selected_config, selected_train_metrics = _select_train_deployment(
        train_examples,
        verifier,
        candidate_names,
        args,
        deployment_configs,
    )
    selected_modes, selected_metrics = _predict_modes(
        test_examples,
        verifier,
        candidate_names,
        args,
        risk_adjusted=bool(selected_config["risk_adjusted"]),
        reject_margin=float(selected_config["reject_margin"]),
        risk_penalty=float(selected_config["risk_penalty"]),
        safe_threshold=float(selected_config["safe_threshold"]),
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
        ("joint_utility_verifier", utility_modes, utility_metrics),
        ("joint_utility_safe_verifier", safe_modes, safe_metrics),
        ("train_selected_joint_utility_verifier", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in deployment_configs:
        name = f"joint_utility_{config['name']}"
        if name in {"joint_utility_verifier", "joint_utility_safe_verifier"}:
            continue
        modes, metrics = _predict_modes(
            test_examples,
            verifier,
            candidate_names,
            args,
            risk_adjusted=bool(config["risk_adjusted"]),
            reject_margin=float(config["reject_margin"]),
            risk_penalty=float(config["risk_penalty"]),
            safe_threshold=float(config["safe_threshold"]),
        )
        policies.append((name, modes, metrics))

    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_joint_utility_verifier_loss"] = round(_mean_policy_loss(train_examples, train_utility_modes), 6)
        row["train_joint_utility_safe_verifier_loss"] = round(_mean_policy_loss(train_examples, train_safe_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_joint_utility_accept_rate"] = train_utility_metrics["accept_rate"]
        row["train_joint_utility_safe_accept_rate"] = train_safe_metrics["accept_rate"]
        rows.append(row)
    return rows


def _joint_deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = []
    for base in _deployment_configs(args):
        for threshold in _unique_floats(args.safe_thresholds):
            config = dict(base)
            config["safe_threshold"] = threshold
            config["name"] = f"{base['name']}_s{_float_token(threshold)}"
            configs.append(config)
    return configs


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    verifier: JointUtilityVerifier,
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
            verifier,
            candidate_names,
            args,
            risk_adjusted=bool(config["risk_adjusted"]),
            reject_margin=float(config["reject_margin"]),
            risk_penalty=float(config["risk_penalty"]),
            safe_threshold=float(config["safe_threshold"]),
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


def _float_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "neg").replace(".", "p")


if __name__ == "__main__":
    main()
