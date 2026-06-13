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
    _train_gate,
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
    _mean_policy_loss,
    _policy_row,
)
from scripts.retriever_cross_seed_verified_candidate_controller_probe import (  # noqa: E402
    _collect_verified_examples,
    _candidate_features,
)
from scripts.retriever_generated_candidate_probe import BASE_MODES, _write_csv  # noqa: E402


class NormalizedPropagationAdapter(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, features: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", features.mean(dim=(0, 1), keepdim=True))
        self.register_buffer("std", features.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-4))
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim),
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.branch_head = torch.nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.mean) / self.std
        batch_size, candidate_count, feature_dim = normalized.shape
        hidden = self.trunk(normalized.reshape(batch_size * candidate_count, feature_dim))
        return self.branch_head(hidden).reshape(batch_size, candidate_count, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a P1 joint propagation-adapter probe. The probe trains a "
            "candidate-aware branch-logit adapter from propagation verification "
            "features, then deploys regret/no-harm selection on adapted losses."
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
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--gate-steps", type=int, default=700)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--composition-lr", type=float, default=3e-3)
    parser.add_argument("--adapter-lr", type=float, default=3e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--regret-gain-weight", type=float, default=1.5)
    parser.add_argument("--adapter-temperature", type=float, default=0.4)
    parser.add_argument("--adapter-utility-weight", type=float, default=1.0)
    parser.add_argument("--adapter-uniform-weight", type=float, default=0.15)
    parser.add_argument("--adapter-entropy-weight", type=float, default=0.005)
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_joint_propagation_adapter.csv"))
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
        print(f"joint-propagation-adapter collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
        adapter = _train_adapter(train_examples, candidate_names, args)
        _apply_adapter_losses(train_examples, adapter, candidate_names)
        _apply_adapter_losses(test_examples, adapter, candidate_names)
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
                    "adapter_steps": args.adapter_steps,
                    "gate_steps": args.gate_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                }
            )
        rows.extend(condition_rows)
    return rows


def _train_adapter(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> NormalizedPropagationAdapter:
    features = _candidate_features(examples, candidate_names)
    raw_losses = torch.tensor(
        [[float(example[f"{name}_loss"]) for name in candidate_names] for example in examples],
        dtype=torch.float32,
    )
    labels = torch.tensor([int(example["branch_label"]) for example in examples], dtype=torch.long)
    centered = raw_losses - raw_losses.mean(dim=1, keepdim=True)
    scale = raw_losses.std(dim=1, keepdim=True).clamp_min(1e-3)
    utility = -(centered / scale)
    utility_weights = F.softmax(utility / max(args.adapter_temperature, 1e-4), dim=1) * len(candidate_names)
    adapter = NormalizedPropagationAdapter(features.size(-1), args.adapter_hidden_dim, features)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr)
    adapter.train()
    for _ in range(args.adapter_steps):
        logits = adapter(features)
        flat_logits = logits.reshape(-1, 3)
        flat_labels = labels.repeat_interleave(len(candidate_names))
        ce = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape_as(raw_losses)
        weighted_ce = (ce * utility_weights).mean()
        uniform_ce = ce.mean()
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = (
            args.adapter_utility_weight * weighted_ce
            + args.adapter_uniform_weight * uniform_ce
            - args.adapter_entropy_weight * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return adapter.eval()


def _apply_adapter_losses(
    examples: list[dict[str, object]],
    adapter: NormalizedPropagationAdapter,
    candidate_names: list[str],
) -> None:
    features = _candidate_features(examples, candidate_names)
    labels = torch.tensor([int(example["branch_label"]) for example in examples], dtype=torch.long)
    with torch.no_grad():
        logits = adapter(features)
        flat_logits = logits.reshape(-1, 3)
        flat_labels = labels.repeat_interleave(len(candidate_names))
        losses = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(len(examples), len(candidate_names))
        predictions = logits.argmax(dim=-1)
    for row_index, example in enumerate(examples):
        for candidate_index, name in enumerate(candidate_names):
            example[f"{name}_base_loss"] = example[f"{name}_loss"]
            example[f"{name}_base_correct"] = example[f"{name}_correct"]
            example[f"{name}_loss"] = round(float(losses[row_index, candidate_index].item()), 6)
            example[f"{name}_correct"] = int(predictions[row_index, candidate_index].item() == labels[row_index].item())
        best_name = min(candidate_names, key=lambda candidate: (float(example[f"{candidate}_loss"]), candidate))
        example["best_mode"] = best_name
        example["oracle_loss"] = round(float(example[f"{best_name}_loss"]), 6)
        example["oracle_correct"] = int(example[f"{best_name}_correct"])


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
        ("joint_adapter_candidate_regret_gate", regret_modes, regret_metrics),
        ("joint_adapter_uncertainty_regret_gate", risk_modes, risk_metrics),
        ("train_selected_joint_propagation_adapter", selected_modes, selected_metrics),
        ("generated_plus_composition_oracle", oracle_modes, {}),
    ]
    for config in deployment_configs:
        name = f"joint_adapter_{config['name']}"
        if name in {"joint_adapter_candidate_regret_gate", "joint_adapter_uncertainty_regret_gate"}:
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
        row["train_joint_adapter_candidate_regret_loss"] = round(_mean_policy_loss(train_examples, train_regret_modes), 6)
        row["train_joint_adapter_uncertainty_regret_loss"] = round(_mean_policy_loss(train_examples, train_risk_modes), 6)
        row["train_static_learned_loss"] = round(_mean_policy_loss(train_examples, ["learned"] * len(train_examples)), 6)
        row["train_joint_adapter_candidate_accept_rate"] = train_regret_metrics["accept_rate"]
        row["train_joint_adapter_uncertainty_accept_rate"] = train_risk_metrics["accept_rate"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
