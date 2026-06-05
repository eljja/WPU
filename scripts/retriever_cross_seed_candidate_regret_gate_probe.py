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


class CandidateRegretGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, feature_dim = features.shape
        hidden = self.trunk(features.view(batch_size * candidate_count, feature_dim))
        mean_regret = self.mean_head(hidden).view(batch_size, candidate_count)
        log_var = self.log_var_head(hidden).view(batch_size, candidate_count).clamp(-6.0, 3.0)
        return mean_regret, log_var


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-seed candidate regret prediction with uncertainty and no-harm rejection."
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
    parser.add_argument("--gate-steps", type=int, default=600)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--risk-penalty", type=float, default=0.5)
    parser.add_argument("--reject-margin", type=float, default=0.0)
    parser.add_argument("--sweep-reject-margins", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--sweep-risk-penalties", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_candidate_regret_gate.csv"))
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
        print(f"candidate-regret-gate collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
                    "risk_penalty": args.risk_penalty,
                    "reject_margin": args.reject_margin,
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
) -> CandidateRegretGate:
    features = _candidate_features(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    safe = (regrets < -args.no_harm_margin).float()
    gate = CandidateRegretGate(features.size(-1), args.gate_hidden_dim)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        pred_mean, pred_log_var = gate(features)
        pred_var = pred_log_var.exp().clamp_min(1e-4)
        regression = F.smooth_l1_loss(pred_mean, regrets)
        nll = 0.5 * (pred_log_var + (regrets - pred_mean).pow(2) / pred_var).mean()
        no_harm = F.binary_cross_entropy_with_logits(-pred_mean, safe)
        loss = regression + args.variance_weight * nll + args.bce_weight * no_harm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gate.eval()


def _predict_modes(
    examples: list[dict[str, object]],
    gate: CandidateRegretGate,
    candidate_names: list[str],
    args: argparse.Namespace,
    *,
    risk_adjusted: bool,
    reject_margin: float | None = None,
    risk_penalty: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    features = _candidate_features(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    with torch.no_grad():
        pred_mean, pred_log_var = gate(features)
    sigma = (0.5 * pred_log_var).exp()
    deployed_risk_penalty = args.risk_penalty if risk_penalty is None else risk_penalty
    deployed_reject_margin = args.reject_margin if reject_margin is None else reject_margin
    score = pred_mean + deployed_risk_penalty * sigma if risk_adjusted else pred_mean
    selected_indices = score.argmin(dim=1)
    selected_modes: list[str] = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        use_candidate = float(score[row_index, candidate_index].item()) < -deployed_reject_margin
        selected_modes.append(candidate_names[candidate_index] if use_candidate else "learned")
        accepted.append(float(use_candidate))
        harmful.append(float(use_candidate and float(regrets[row_index, candidate_index].item()) > 0.0))
    pred = pred_mean.flatten()
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
        "predicted_regret_mean": round(float(pred_mean.mean().item()), 6),
        "predicted_sigma_mean": round(float(sigma.mean().item()), 6),
        "deployment_reject_margin": round(float(deployed_reject_margin), 6),
        "deployment_risk_penalty": round(float(deployed_risk_penalty), 6),
        "deployment_risk_adjusted": float(risk_adjusted),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gate: CandidateRegretGate,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    regret_modes, regret_metrics = _predict_modes(test_examples, gate, candidate_names, args, risk_adjusted=False)
    risk_modes, risk_metrics = _predict_modes(test_examples, gate, candidate_names, args, risk_adjusted=True)
    train_regret_modes, train_regret_metrics = _predict_modes(train_examples, gate, candidate_names, args, risk_adjusted=False)
    train_risk_modes, train_risk_metrics = _predict_modes(train_examples, gate, candidate_names, args, risk_adjusted=True)

    policies = [
        ("static_learned_interaction", static_modes, {}),
        ("candidate_regret_gate", regret_modes, regret_metrics),
        ("uncertainty_regret_gate", risk_modes, risk_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for margin in _unique_floats([args.reject_margin, *args.sweep_reject_margins]):
        if margin == args.reject_margin:
            continue
        modes, metrics = _predict_modes(
            test_examples,
            gate,
            candidate_names,
            args,
            risk_adjusted=False,
            reject_margin=margin,
        )
        policies.append((f"candidate_regret_gate_m{_float_token(margin)}", modes, metrics))
    for risk_penalty in _unique_floats([args.risk_penalty, *args.sweep_risk_penalties]):
        for margin in _unique_floats([args.reject_margin, *args.sweep_reject_margins]):
            if risk_penalty == args.risk_penalty and margin == args.reject_margin:
                continue
            modes, metrics = _predict_modes(
                test_examples,
                gate,
                candidate_names,
                args,
                risk_adjusted=True,
                reject_margin=margin,
                risk_penalty=risk_penalty,
            )
            policies.append((f"uncertainty_regret_gate_r{_float_token(risk_penalty)}_m{_float_token(margin)}", modes, metrics))
    rows = []
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_candidate_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_regret_modes), 6)
        row["train_uncertainty_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_risk_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_candidate_regret_accept_rate"] = train_regret_metrics["accept_rate"]
        row["train_uncertainty_regret_accept_rate"] = train_risk_metrics["accept_rate"]
        rows.append(row)
    return rows


def _unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        rounded = round(float(value), 6)
        if rounded not in out:
            out.append(rounded)
    return out


def _float_token(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "neg").replace(".", "p")
    return text


if __name__ == "__main__":
    main()
