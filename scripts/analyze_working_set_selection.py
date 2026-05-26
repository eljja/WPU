from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parents[1]))

from wpu.data.working_set_physics import (  # noqa: E402
    WorkingSetPhysicsDataset,
    collate_indexed_working_set_samples,
    collate_interaction_working_set_samples,
    collate_proximity_working_set_samples,
)


COLLATE_BY_MODE = {
    "indexed": collate_indexed_working_set_samples,
    "proximity": collate_proximity_working_set_samples,
    "interaction": collate_interaction_working_set_samples,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze selected working-set composition before tensorization.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--selection-modes", nargs="+", default=["indexed", "proximity", "interaction"])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_selection_composition.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            dataset = WorkingSetPhysicsDataset(
                size=args.samples,
                seed=args.seed + n_value * 17 + k_value,
                background_objects=background_objects,
                causal_obstacles=causal_obstacles,
                balanced_labels=args.balanced_labels,
                interaction_mode=args.interaction_mode,
            )
            samples = [dataset[index] for index in range(len(dataset))]
            for budget in args.budgets:
                for mode in args.selection_modes:
                    rows.append(_analyze_condition(samples, n_value, k_value, budget, mode, args.index_depth))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _analyze_condition(
    samples,
    n_value: int,
    k_value: int,
    budget: int,
    mode: str,
    index_depth: int,
) -> dict[str, object]:
    collate_fn = COLLATE_BY_MODE[mode]
    causal_ids = {"cup_001", "table_001", "hand_001", "edge_001"}
    causal_ids.update(f"obstacle_{index:03d}" for index in range(max(0, k_value - 4)))
    selected_counts: list[int] = []
    causal_recalls: list[float] = []
    obstacle_recalls: list[float] = []
    obstacle_counts: list[int] = []
    hand_hits: list[float] = []
    table_hits: list[float] = []
    edge_hits: list[float] = []
    pair_density_values: list[float] = []
    for sample in samples:
        batch, _, _, _ = collate_fn([sample], max_nodes=budget, max_depth=index_depth)
        object_ids = batch.object_ids[0] if batch.object_ids is not None else []
        selected = set(object_ids)
        obstacle_ids = [object_id for object_id in object_ids if object_id.startswith("obstacle_")]
        causal_obstacles = [object_id for object_id in causal_ids if object_id.startswith("obstacle_")]
        selected_counts.append(len(object_ids))
        causal_recalls.append(len(selected & causal_ids) / max(len(causal_ids), 1))
        obstacle_recalls.append(len(selected & set(causal_obstacles)) / max(len(causal_obstacles), 1))
        obstacle_counts.append(len(obstacle_ids))
        hand_hits.append(float("hand_001" in selected))
        table_hits.append(float("table_001" in selected))
        edge_hits.append(float("edge_001" in selected))
        pair_density_values.append(_selected_pair_density(sample, obstacle_ids))
    return {
        "total_objects_n": n_value,
        "causal_k": k_value,
        "budget": budget,
        "selection_mode": mode,
        "samples": len(samples),
        "mean_selected": round(mean(selected_counts), 6),
        "causal_recall": round(mean(causal_recalls), 6),
        "obstacle_recall": round(mean(obstacle_recalls), 6),
        "selected_obstacles": round(mean(obstacle_counts), 6),
        "hand_hit_rate": round(mean(hand_hits), 6),
        "table_hit_rate": round(mean(table_hits), 6),
        "edge_hit_rate": round(mean(edge_hits), 6),
        "selected_obstacle_pair_density": round(mean(pair_density_values), 6),
    }


def _selected_pair_density(sample, obstacle_ids: list[str]) -> float:
    if len(obstacle_ids) < 2:
        return 0.0
    positions = {
        object_id: sample.state.objects[object_id].attributes.get("position", [0.0, 0.0, 0.0])
        for object_id in obstacle_ids
        if object_id in sample.state.objects
    }
    close_pairs = 0
    possible_pairs = 0
    for left_index, left_id in enumerate(obstacle_ids):
        left = positions.get(left_id)
        if left is None:
            continue
        for right_id in obstacle_ids[left_index + 1 :]:
            right = positions.get(right_id)
            if right is None:
                continue
            possible_pairs += 1
            distance = ((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2) ** 0.5
            close_pairs += int(distance < 0.075)
    return close_pairs / max(possible_pairs, 1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
