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
    _deployment_configs as _regret_deployment_configs,
    _float_token,
    _perturb_features,
    _regret_targets,
)
from scripts.retriever_cross_seed_composition_regret_probe import _train_composition_prior  # noqa: E402
from scripts.retriever_cross_seed_regret_distillation_probe import (  # noqa: E402
    _train_cross_seed_regret_retriever,
)
from scripts.retriever_cross_seed_set_evaluator_probe import (  # noqa: E402
    COMPOSITION_MODES,
    _collect_examples,
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_cross_seed_set_evaluator_probe import _collect_seed_context  # noqa: E402
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class CandidateSafetyGate(nn.Module):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate safety/utility gate for WPU v2 priority-1."
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
    parser.add_argument("--gate-steps", type=int, default=800)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--safe-bce-weight", type=float, default=1.0)
    parser.add_argument("--regret-weight", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--harmful-accept-weight", type=float, default=0.5)
    parser.add_argument("--safe-prob-thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--utility-margins", type=float, nargs="+", default=[0.0, 0.001, 0.0025, 0.005, 0.01])
    parser.add_argument("--risk-penalties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_candidate_safety_gate.csv"))
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
        print(f"candidate-safety-gate collect seed={seed} N={total_n} K={causal_k}", flush=True)
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


def _candidate_features(examples: list[dict[str, object]], candidate_names: list[str]) -> torch.Tensor:
    candidate_count = len(candidate_names)
    features = []
    for example in examples:
        context = example["context_features"][:, candidate_count:].clone()  # type: ignore[index]
        features.append(context)
    return torch.stack(features)


def _train_gate(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> CandidateSafetyGate:
    features = _candidate_features(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    positive = safe.sum().clamp_min(1.0)
    negative = (1.0 - safe).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(0.25, 8.0)
    gate = CandidateSafetyGate(features.size(-1), args.gate_hidden_dim)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        train_features = _perturb_features(features, args)
        pred_regret, safe_logit = gate(train_features)
        regret_loss = F.smooth_l1_loss(pred_regret, regrets)
        safe_loss = F.binary_cross_entropy_with_logits(safe_logit, safe, pos_weight=pos_weight)
        best_indices = regrets.argmin(dim=1)
        rank_loss = F.cross_entropy(-pred_regret, best_indices)
        unsafe = (regrets > args.no_harm_margin).float()
        accept_probability = torch.sigmoid(safe_logit) * torch.sigmoid(-pred_regret)
        harmful_loss = (accept_probability * unsafe).mean()
        loss = (
            args.regret_weight * regret_loss
            + args.safe_bce_weight * safe_loss
            + args.ranking_weight * rank_loss
            + args.harmful_accept_weight * harmful_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gate.eval()


def _predict_modes(
    examples: list[dict[str, object]],
    gate: CandidateSafetyGate,
    candidate_names: list[str],
    *,
    safe_prob_threshold: float,
    utility_margin: float,
    risk_penalty: float,
) -> tuple[list[str], dict[str, float]]:
    features = _candidate_features(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    with torch.no_grad():
        pred_regret, safe_logit = gate(features)
    safe_prob = torch.sigmoid(safe_logit)
    predicted_gain = (-pred_regret).clamp_min(0.0)
    utility = safe_prob * predicted_gain - risk_penalty * (1.0 - safe_prob)
    selected_indices = utility.argmax(dim=1)
    selected_modes: list[str] = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        score = float(utility[row_index, candidate_index].item())
        probability = float(safe_prob[row_index, candidate_index].item())
        use_candidate = score > utility_margin and probability >= safe_prob_threshold
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
    labels = (regrets < 0.0).float()
    brier = float((safe_prob.flatten() - labels.flatten()).pow(2).mean().item())
    return selected_modes, {
        "accept_rate": round(mean(accepted), 6),
        "harmful_accept_rate": round(mean(harmful), 6),
        "regret_corr": round(corr, 6),
        "predicted_regret_mean": round(float(pred_regret.mean().item()), 6),
        "predicted_sigma_mean": round(float(safe_prob.std(unbiased=False).item()), 6),
        "safe_probability_mean": round(float(safe_prob.mean().item()), 6),
        "safe_probability_brier": round(brier, 6),
        "deployment_safe_prob_threshold": round(float(safe_prob_threshold), 6),
        "deployment_utility_margin": round(float(utility_margin), 6),
        "deployment_risk_penalty": round(float(risk_penalty), 6),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gate: CandidateSafetyGate,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
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
        ("train_selected_safety_utility_gate", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in configs:
        modes, metrics = _predict_modes(
            test_examples,
            gate,
            candidate_names,
            safe_prob_threshold=float(config["safe_prob_threshold"]),
            utility_margin=float(config["utility_margin"]),
            risk_penalty=float(config["risk_penalty"]),
        )
        policies.append((str(config["name"]), modes, metrics))

    train_default_modes, train_default_metrics = _predict_modes(
        train_examples,
        gate,
        candidate_names,
        safe_prob_threshold=0.5,
        utility_margin=0.0,
        risk_penalty=0.0,
    )
    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_candidate_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_uncertainty_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_candidate_regret_accept_rate"] = train_default_metrics["accept_rate"]
        row["train_uncertainty_regret_accept_rate"] = train_default_metrics["accept_rate"]
        rows.append(row)
    return rows


def _deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for safe_prob_threshold in _unique_floats(args.safe_prob_thresholds):
        for utility_margin in _unique_floats(args.utility_margins):
            for risk_penalty in _unique_floats(args.risk_penalties):
                configs.append(
                    {
                        "name": (
                            f"safety_utility_gate_p{_float_token(safe_prob_threshold)}"
                            f"_m{_float_token(utility_margin)}"
                            f"_r{_float_token(risk_penalty)}"
                        ),
                        "safe_prob_threshold": safe_prob_threshold,
                        "utility_margin": utility_margin,
                        "risk_penalty": risk_penalty,
                    }
                )
    return configs


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    gate: CandidateSafetyGate,
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
