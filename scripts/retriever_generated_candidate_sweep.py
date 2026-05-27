from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retriever_generated_candidate_probe import _run_condition, _write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep generated WPU retrieval candidate count.")
    parser.add_argument("--model-name", default="wpu-cws-indexed-regret-hybrid")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidate-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--propagation-steps", type=int, default=40)
    parser.add_argument("--reranker-steps", type=int, default=600)
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-lr", type=float, default=3e-3)
    parser.add_argument("--safe-margin", type=float, default=0.005)
    parser.add_argument("--utility-temperature", type=float, default=0.05)
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
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_retriever_generated_candidate_sweep.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for generated_count in args.generated_candidate_values:
        run_args = argparse.Namespace(**vars(args), generated_candidates=generated_count)
        for n_value in args.n_values:
            for k_value in args.k_values:
                causal_obstacles = max(0, k_value - 4)
                background_objects = max(0, n_value - 4 - causal_obstacles)
                for seed in args.seeds:
                    print(
                        f"candidate-sweep generated={generated_count} seed={seed} N={n_value} K={k_value}",
                        flush=True,
                    )
                    rows.extend(_run_condition(background_objects, causal_obstacles, seed, run_args))
                    _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


if __name__ == "__main__":
    main()
