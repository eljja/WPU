from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    _candidate_names,
    _write_csv,
)
from scripts.retriever_cross_seed_normalized_reranker_probe import (  # noqa: E402
    _collect_seed_examples,
    _summarize_normalized,
    _train_normalized_reranker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate candidate identity features in cross-seed WPU reranking.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--ablation-modes", nargs="+", default=["no_identity_keep_type", "no_identity_no_type"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--reranker-steps", type=int, default=600)
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-lr", type=float, default=3e-3)
    parser.add_argument("--safe-margin", type=float, default=0.005)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_cross_seed_context_ablation.csv"))
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
        print(f"context-ablation collect seed={seed} N={total_n} K={causal_k}", flush=True)
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
        base_train_examples = [
            example
            for seed, examples in validation_by_seed.items()
            if seed != heldout_seed
            for example in examples
        ]
        base_test_examples = test_by_seed[heldout_seed]
        for ablation_mode in args.ablation_modes:
            train_examples = _ablate_examples(base_train_examples, len(candidate_names), ablation_mode)
            test_examples = _ablate_examples(base_test_examples, len(candidate_names), ablation_mode)
            reranker = _train_normalized_reranker(train_examples, args, len(candidate_names))
            condition_rows = _summarize_normalized(test_examples, train_examples, reranker, candidate_names, args)
            for row in condition_rows:
                row.update(
                    {
                        "context_ablation": ablation_mode,
                        "seed": heldout_seed,
                        "heldout_seed": heldout_seed,
                        "train_seed_count": len(args.seeds) - 1,
                        "total_objects_n": total_n,
                        "causal_k": causal_k,
                        "budget": args.budget,
                        "generated_candidates": args.generated_candidates,
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


def _ablate_examples(
    examples: list[dict[str, object]],
    candidate_count: int,
    ablation_mode: str,
) -> list[dict[str, object]]:
    if ablation_mode not in {"full", "no_identity_keep_type", "no_identity_no_type"}:
        raise ValueError(f"unknown ablation mode: {ablation_mode}")
    ablated = []
    for example in examples:
        row = dict(example)
        context = example["context_features"].clone()  # type: ignore[union-attr]
        if ablation_mode in {"no_identity_keep_type", "no_identity_no_type"}:
            context[:, :candidate_count] = 0.0
        if ablation_mode == "no_identity_no_type":
            context[:, -1] = 0.0
        row["context_features"] = context
        ablated.append(row)
    return ablated


if __name__ == "__main__":
    main()
