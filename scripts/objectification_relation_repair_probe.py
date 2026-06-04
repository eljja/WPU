from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wpu  # noqa: E402
from wpu.core.state import Event, WorldObject, WorldState  # noqa: E402
from wpu.engines.sparse_engine import SparsePropagationEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether objectification relation repair restores sparse frontier recall.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--near-distance", type=float, default=0.25)
    parser.add_argument("--contact-distance", type=float, default=0.08)
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/objectification_relation_repair_probe.csv"),
    )
    args = parser.parse_args()

    rows = run_probe(
        samples=args.samples,
        seed=args.seed,
        near_distance=args.near_distance,
        contact_distance=args.contact_distance,
        background_objects=args.background_objects,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = rows[-1]
    print(
        "relation_repair_probe "
        f"samples={summary['samples']} "
        f"before_recall={summary['mean_before_frontier_recall']} "
        f"after_recall={summary['mean_after_frontier_recall']} "
        f"repair_precision={summary['repair_precision']} "
        f"repair_recall={summary['repair_recall']}"
    )


def run_probe(
    *,
    samples: int,
    seed: int,
    near_distance: float,
    contact_distance: float,
    background_objects: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sparse = SparsePropagationEngine(max_depth=2)
    totals = {
        "before_recall": 0.0,
        "after_recall": 0.0,
        "added": 0,
        "candidate_pairs": 0,
        "true_positive_edges": 0,
        "expected_edges": 0,
    }

    for index in range(samples):
        state = _state_with_missing_relations(rng, background_objects=background_objects)
        event = Event("hand_touched_cup", "cup_001", {"force": 0.4}, confidence=0.9)
        expected_objects = {"cup_001", "hand_001", "edge_001", "table_001"}
        expected_edges = _expected_near_edges(
            state,
            near_distance=near_distance,
            contact_distance=contact_distance,
        )

        before = sparse.sparse_propagate(state, event)
        repaired, repair_report = wpu.repair_objectification_relations(
            state,
            near_distance=near_distance,
            contact_distance=contact_distance,
        )
        after = sparse.sparse_propagate(repaired, event)
        repaired_edges = {
            _edge_key(relation.src, relation.dst, relation.type)
            for relation in repaired.relations
            if relation.type in {"near", "touching"}
        }

        totals["before_recall"] += len(before.affected_objects & expected_objects) / len(expected_objects)
        totals["after_recall"] += len(after.affected_objects & expected_objects) / len(expected_objects)
        totals["added"] += repair_report.added_relation_count
        totals["candidate_pairs"] += repair_report.candidate_pair_count
        totals["true_positive_edges"] += len(repaired_edges & expected_edges)
        totals["expected_edges"] += len(expected_edges)

    repair_precision = totals["true_positive_edges"] / max(totals["added"], 1)
    repair_recall = totals["true_positive_edges"] / max(totals["expected_edges"], 1)
    return [
        {
            "samples": str(samples),
            "seed": str(seed),
            "near_distance": f"{near_distance:.6f}",
            "contact_distance": f"{contact_distance:.6f}",
            "background_objects": str(background_objects),
            "mean_before_frontier_recall": f"{totals['before_recall'] / samples:.6f}",
            "mean_after_frontier_recall": f"{totals['after_recall'] / samples:.6f}",
            "mean_added_relations": f"{totals['added'] / samples:.6f}",
            "mean_candidate_pairs": f"{totals['candidate_pairs'] / samples:.6f}",
            "repair_precision": f"{repair_precision:.6f}",
            "repair_recall": f"{repair_recall:.6f}",
        }
    ]


def _state_with_missing_relations(rng: random.Random, *, background_objects: int) -> WorldState:
    hand_x = rng.uniform(0.08, 0.18)
    edge_x = rng.uniform(0.16, 0.22)
    state = WorldState(metadata={"scenario": "objectification_relation_repair_probe"})
    state.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 0.82]}, confidence=0.95))
    state.add_object(WorldObject("hand_001", "robot_hand", {"position": [hand_x, 0.0, 0.82]}, confidence=0.92))
    state.add_object(WorldObject("edge_001", "table_edge", {"position": [edge_x, 0.0, 0.82]}, confidence=0.94))
    state.add_object(WorldObject("table_001", "table", {"position": [0.02, 0.0, 0.75]}, confidence=0.98))
    for index in range(background_objects):
        state.add_object(
            WorldObject(
                f"context_{index:04d}",
                "background_object",
                {"position": [10.0 + index, 10.0, 0.0]},
                confidence=0.75,
            )
        )
    return state


def _expected_near_edges(state: WorldState, *, near_distance: float, contact_distance: float = 0.08) -> set[tuple[str, str, str]]:
    core_ids = ["cup_001", "hand_001", "edge_001", "table_001"]
    expected: set[tuple[str, str, str]] = set()
    for left_index, left_id in enumerate(core_ids):
        left = _position(state, left_id)
        for right_id in core_ids[left_index + 1 :]:
            right = _position(state, right_id)
            distance = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2) ** 0.5
            if distance <= near_distance:
                expected.add(_edge_key(left_id, right_id, "near"))
            if distance <= contact_distance:
                expected.add(_edge_key(left_id, right_id, "touching"))
    return expected


def _position(state: WorldState, object_id: str) -> tuple[float, float, float]:
    value = state.objects[object_id].attributes["position"]
    return float(value[0]), float(value[1]), float(value[2])


def _edge_key(src: str, dst: str, relation_type: str) -> tuple[str, str, str]:
    if dst < src:
        return dst, src, relation_type
    return src, dst, relation_type


if __name__ == "__main__":
    main()
