from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_cross_seed_candidate_regret_gate_probe import (  # noqa: E402
    CandidateRegretGate,
    _candidate_features,
    _float_token,
    _perturb_features,
    _regret_targets,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a cross-fit ensemble candidate-regret gate. Deployment "
            "thresholds are selected using out-of-source-seed predictions, not "
            "in-sample train predictions."
        )
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
    parser.add_argument("--reject-margins", type=float, nargs="+", default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--risk-penalties", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--disagreement-penalties", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--vote-thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0])
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--no-harm-margin", type=float, default=0.0)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_candidate_regret_crossfit.csv"))
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
        print(f"candidate-regret-crossfit collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
        gates_by_seed = _train_crossfit_gates(train_examples, candidate_names, args)
        condition_rows = _summarize(test_examples, train_examples, gates_by_seed, candidate_names, args)
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


def _train_crossfit_gates(
    train_examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> dict[int, CandidateRegretGate]:
    seeds = sorted({int(example["source_seed"]) for example in train_examples})
    gates: dict[int, CandidateRegretGate] = {}
    for source_seed in seeds:
        subset = [example for example in train_examples if int(example["source_seed"]) != source_seed]
        gates[source_seed] = _train_gate(subset or train_examples, candidate_names, args)
    return gates


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
        train_features = _perturb_features(features, args)
        pred_mean, pred_log_var = gate(train_features)
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
    gates: list[CandidateRegretGate],
    candidate_names: list[str],
    *,
    reject_margin: float,
    risk_penalty: float,
    disagreement_penalty: float,
    vote_threshold: float,
) -> tuple[list[str], dict[str, float]]:
    features = _candidate_features(examples, candidate_names)
    regrets = _regret_targets(examples, candidate_names)
    means = []
    sigmas = []
    with torch.no_grad():
        for gate in gates:
            pred_mean, pred_log_var = gate(features)
            means.append(pred_mean)
            sigmas.append((0.5 * pred_log_var).exp())
    mean_stack = torch.stack(means)
    sigma_stack = torch.stack(sigmas)
    pred_mean = mean_stack.mean(dim=0)
    aleatoric = sigma_stack.mean(dim=0)
    epistemic = mean_stack.std(dim=0, unbiased=False)
    score = pred_mean + risk_penalty * aleatoric + disagreement_penalty * epistemic
    per_gate_score = mean_stack + risk_penalty * sigma_stack
    vote_fraction = (per_gate_score < -reject_margin).float().mean(dim=0)
    selected_indices = score.argmin(dim=1)
    selected_modes: list[str] = []
    accepted = []
    harmful = []
    for row_index, candidate_index in enumerate(selected_indices.tolist()):
        use_candidate = (
            float(score[row_index, candidate_index].item()) < -reject_margin
            and float(vote_fraction[row_index, candidate_index].item()) >= vote_threshold
        )
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
        "predicted_sigma_mean": round(float(aleatoric.mean().item()), 6),
        "predicted_epistemic_mean": round(float(epistemic.mean().item()), 6),
        "deployment_reject_margin": round(float(reject_margin), 6),
        "deployment_risk_penalty": round(float(risk_penalty), 6),
        "deployment_disagreement_penalty": round(float(disagreement_penalty), 6),
        "deployment_vote_threshold": round(float(vote_threshold), 6),
    }


def _summarize(
    test_examples: list[dict[str, object]],
    train_examples: list[dict[str, object]],
    gates_by_seed: dict[int, CandidateRegretGate],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    static_modes = ["learned"] * len(test_examples)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    configs = _deployment_configs(args)
    selected_config, selected_train_metrics = _select_train_deployment(train_examples, gates_by_seed, candidate_names, args, configs)
    selected_modes, selected_metrics = _predict_modes(
        test_examples,
        list(gates_by_seed.values()),
        candidate_names,
        reject_margin=float(selected_config["reject_margin"]),
        risk_penalty=float(selected_config["risk_penalty"]),
        disagreement_penalty=float(selected_config["disagreement_penalty"]),
        vote_threshold=float(selected_config["vote_threshold"]),
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
        ("crossfit_selected_candidate_regret_gate", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in configs:
        modes, metrics = _predict_modes(
            test_examples,
            list(gates_by_seed.values()),
            candidate_names,
            reject_margin=float(config["reject_margin"]),
            risk_penalty=float(config["risk_penalty"]),
            disagreement_penalty=float(config["disagreement_penalty"]),
            vote_threshold=float(config["vote_threshold"]),
        )
        policies.append((str(config["name"]), modes, metrics))

    rows = []
    train_default_modes = _predict_train_crossfit_modes(train_examples, gates_by_seed, candidate_names, configs[0])
    for policy, modes, metrics in policies:
        row = _policy_row(test_examples, policy, modes, candidate_names)
        row.update(metrics)
        row["train_candidate_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_uncertainty_regret_gate_loss"] = round(_mean_policy_loss(train_examples, train_default_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_candidate_regret_accept_rate"] = selected_train_metrics["accept_rate"]
        row["train_uncertainty_regret_accept_rate"] = selected_train_metrics["accept_rate"]
        rows.append(row)
    return rows


def _predict_train_crossfit_modes(
    train_examples: list[dict[str, object]],
    gates_by_seed: dict[int, CandidateRegretGate],
    candidate_names: list[str],
    config: dict[str, object],
) -> list[str]:
    out: list[str] = []
    for source_seed in sorted(gates_by_seed):
        subset = [example for example in train_examples if int(example["source_seed"]) == source_seed]
        if not subset:
            continue
        modes, _ = _predict_modes(
            subset,
            [gates_by_seed[source_seed]],
            candidate_names,
            reject_margin=float(config["reject_margin"]),
            risk_penalty=float(config["risk_penalty"]),
            disagreement_penalty=float(config["disagreement_penalty"]),
            vote_threshold=float(config["vote_threshold"]),
        )
        out.extend(modes)
    return out


def _select_train_deployment(
    train_examples: list[dict[str, object]],
    gates_by_seed: dict[int, CandidateRegretGate],
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
        modes = _predict_train_crossfit_modes(train_examples, gates_by_seed, candidate_names, config)
        loss = _mean_policy_loss(train_examples, modes)
        regrets = _regret_targets(train_examples, candidate_names)
        harmful = [
            float(mode != "learned" and float(regrets[row_index, candidate_names.index(mode)].item()) > 0.0)
            for row_index, mode in enumerate(modes)
        ]
        accepted = [float(mode != "learned") for mode in modes]
        metrics = {
            "loss": round(loss, 6),
            "gap_closure": round((static_loss - loss) / oracle_gap, 6),
            "accept_rate": round(mean(accepted), 6),
            "harmful_accept_rate": round(mean(harmful), 6),
        }
        evaluated.append((config, metrics))
    safe = [
        item
        for item in evaluated
        if item[1]["harmful_accept_rate"] <= args.selection_harmful_limit
        and item[1]["gap_closure"] > 0.0
    ]
    candidates = safe or evaluated
    return max(candidates, key=lambda item: (item[1]["gap_closure"], -item[1]["harmful_accept_rate"]))


def _deployment_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for margin in _unique_floats(args.reject_margins):
        for risk_penalty in _unique_floats(args.risk_penalties):
            for disagreement_penalty in _unique_floats(args.disagreement_penalties):
                for vote_threshold in _unique_floats(args.vote_thresholds):
                    configs.append(
                        {
                            "name": (
                                f"crossfit_regret_gate_m{_float_token(margin)}"
                                f"_r{_float_token(risk_penalty)}"
                                f"_d{_float_token(disagreement_penalty)}"
                                f"_v{_float_token(vote_threshold)}"
                            ),
                            "reject_margin": margin,
                            "risk_penalty": risk_penalty,
                            "disagreement_penalty": disagreement_penalty,
                            "vote_threshold": vote_threshold,
                        }
                    )
    return configs


def _unique_floats(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        rounded = round(float(value), 6)
        if rounded not in out:
            out.append(rounded)
    return out


if __name__ == "__main__":
    main()
