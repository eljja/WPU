from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_cross_k_probe import _train_group_balanced_model  # noqa: E402
from scripts.learned_retriever_probe import _make_samples  # noqa: E402
from scripts.staged_k_expansion_hybrid import _run_condition, _write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a mixed-K global learned retriever inside staged WPU.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--retriever-train-k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--initial-working-set-size", type=int, default=4)
    parser.add_argument("--expanded-working-set-size", type=int, default=32)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--regret-steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regret-lr", type=float, default=3e-3)
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--expansion-cost", type=float, default=0.02)
    parser.add_argument("--max-expansion-rate", type=float, default=0.5)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--retriever-train-samples", type=int, default=160)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_staged_global_retriever_initial4_5seed.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for seed in args.seeds:
            initial_retriever, expanded_retriever = _train_global_retrievers(n_value, seed, args)
            for k_value in args.k_values:
                causal_obstacles = max(0, k_value - 4)
                background_objects = max(0, n_value - 4 - causal_obstacles)
                run_args = argparse.Namespace(**vars(args))
                run_args.selection_mode = "learned_interaction_global"
                run_args.initial_selection_retriever = initial_retriever
                run_args.expanded_selection_retriever = expanded_retriever
                print(f"global-retriever seed={seed} N={n_value} K={k_value}", flush=True)
                condition_rows = _run_condition(background_objects, causal_obstacles, seed, run_args)
                for row in condition_rows:
                    row["retriever_train_k_values"] = ",".join(str(value) for value in args.retriever_train_k_values)
                    row["retriever_train_samples_per_k"] = args.retriever_train_samples
                rows.extend(condition_rows)
                _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _train_global_retrievers(n_value: int, seed: int, args: argparse.Namespace):
    sample_groups = [
        _make_samples(n_value, k_value, [seed], args.retriever_train_samples)
        for k_value in args.retriever_train_k_values
    ]
    initial_retriever = _train_group_balanced_model(
        sample_groups,
        args.initial_working_set_size,
        args.retriever_steps,
        args.retriever_hidden_dim,
        args.retriever_lr,
    )
    expanded_retriever = _train_group_balanced_model(
        sample_groups,
        args.expanded_working_set_size,
        args.retriever_steps,
        args.retriever_hidden_dim,
        args.retriever_lr,
    )
    return initial_retriever, expanded_retriever


if __name__ == "__main__":
    main()
