from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import sys
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wpu
from wpu.data.pybullet_cup import (
    ObjectificationCorruptionConfig,
    PyBulletCupDataset,
    PyBulletCupSample,
    corrupt_pybullet_cup_sample,
)
from wpu.data.working_set_physics import _indexed_object_ids


CORRUPTION_PRESETS: dict[str, ObjectificationCorruptionConfig] = {
    "clean": ObjectificationCorruptionConfig(),
    "drop_relations_heavy": ObjectificationCorruptionConfig(relation_drop_rate=0.60),
    "drop_objects_light": ObjectificationCorruptionConfig(non_target_object_drop_rate=0.20),
    "position_noise": ObjectificationCorruptionConfig(position_noise_std=0.08),
    "low_confidence": ObjectificationCorruptionConfig(confidence_scale=0.55),
    "identity_swap": ObjectificationCorruptionConfig(identity_swap_rate=1.0),
    "combined": ObjectificationCorruptionConfig(
        relation_drop_rate=0.35,
        non_target_object_drop_rate=0.15,
        position_noise_std=0.05,
        confidence_scale=0.70,
        identity_swap_rate=0.50,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure PyBullet objectification quality against clean simulator state."
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--background-objects", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTION_PRESETS))
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--position-tolerance", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_objectification_quality.csv"))
    args = parser.parse_args()

    unknown = [name for name in args.corruptions if name not in CORRUPTION_PRESETS]
    if unknown:
        raise ValueError(f"unknown corruption presets: {unknown}")

    rows: list[dict[str, str]] = []
    for seed in args.seeds:
        for background in args.background_objects:
            dataset = PyBulletCupDataset(
                size=args.samples,
                seed=seed,
                background_objects=background,
                steps=args.sim_steps,
                balanced_labels=False,
            )
            for sample_index in range(args.samples):
                clean = dataset[sample_index]
                for corruption in args.corruptions:
                    corrupted = corrupt_pybullet_cup_sample(
                        clean,
                        config=CORRUPTION_PRESETS[corruption],
                        seed=seed * 100_000 + background * 1_000 + sample_index,
                    )
                    rows.append(
                        _measure(
                            seed=seed,
                            sample_index=sample_index,
                            background_objects=background,
                            corruption=corruption,
                            clean=clean,
                            corrupted=corrupted,
                            max_nodes=args.max_nodes,
                            max_depth=args.max_depth,
                            position_tolerance=args.position_tolerance,
                        )
                    )

    output_rows = rows + _summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    for row in output_rows:
        if row["row_type"] != "summary":
            continue
        print(
            "pybullet_objectification_quality "
            f"N={row['total_objects']} "
            f"corruption={row['corruption']} "
            f"contract={row['objectification_score']} "
            f"rel_recall={row['relation_recall']} "
            f"frontier_recall={row['frontier_recall']}"
        )


def _measure(
    *,
    seed: int,
    sample_index: int,
    background_objects: int,
    corruption: str,
    clean: PyBulletCupSample,
    corrupted: PyBulletCupSample,
    max_nodes: int,
    max_depth: int,
    position_tolerance: float,
) -> dict[str, str]:
    clean_ids = set(clean.state.objects)
    corrupted_ids = set(corrupted.state.objects)
    shared_ids = clean_ids & corrupted_ids
    clean_relations = _relation_set(clean.state)
    corrupted_relations = _relation_set(corrupted.state)
    clean_frontier = _frontier(clean)
    selected_ids = set(_indexed_object_ids(corrupted.state, corrupted.event, max_nodes=max_nodes, max_depth=max_depth))
    report = wpu.evaluate_objectification(
        corrupted.state,
        expected_working_set=clean_frontier,
        event_target=corrupted.event.target,
        reference_state=clean.state,
        position_tolerance=position_tolerance,
    )
    semantic_identity_consistency = (
        report.semantic_identity_consistency
        if report.semantic_identity_consistency is not None
        else _semantic_identity_consistency(clean, corrupted, position_tolerance)
    )
    frontier_completeness = report.frontier_completeness if report.frontier_completeness is not None else 1.0

    return {
        "row_type": "sample",
        "seed": str(seed),
        "sample_index": str(sample_index),
        "background_objects": str(background_objects),
        "corruption": corruption,
        "total_objects": str(len(clean_ids)),
        "corrupted_objects": str(len(corrupted_ids)),
        "identity_recall": f"{_safe_ratio(len(shared_ids), len(clean_ids), 1.0):.6f}",
        "semantic_identity_consistency": f"{semantic_identity_consistency:.6f}",
        "relation_precision": f"{_safe_ratio(len(clean_relations & corrupted_relations), len(corrupted_relations), 1.0):.6f}",
        "relation_recall": f"{_safe_ratio(len(clean_relations & corrupted_relations), len(clean_relations), 1.0):.6f}",
        "frontier_recall": f"{frontier_completeness:.6f}",
        "selected_k": str(len(selected_ids)),
        "objectification_score": f"{report.contract_score:.6f}",
        "identity_coverage_report": f"{report.identity_coverage:.6f}",
        "relation_validity_report": f"{report.relation_validity:.6f}",
        "object_confidence_report": f"{report.object_confidence:.6f}",
        "relation_confidence_report": f"{report.relation_confidence:.6f}",
        "delta_locality_report": f"{(report.delta_locality if report.delta_locality is not None else 1.0):.6f}",
        "frontier_completeness_report": f"{frontier_completeness:.6f}",
        "semantic_identity_consistency_report": f"{semantic_identity_consistency:.6f}",
        "sample_count": "1",
    }


def _relation_set(state: wpu.WorldState) -> set[tuple[str, str, str]]:
    return {(relation.src, relation.dst, relation.type) for relation in state.relations}


def _frontier(sample: PyBulletCupSample) -> set[str]:
    target = sample.event.target
    frontier = {target}
    for relation in sample.state.relations:
        if relation.src == target:
            frontier.add(relation.dst)
        if relation.dst == target:
            frontier.add(relation.src)
    return frontier


def _semantic_identity_consistency(
    clean: PyBulletCupSample,
    corrupted: PyBulletCupSample,
    position_tolerance: float,
) -> float:
    scores: list[float] = []
    for object_id, clean_object in clean.state.objects.items():
        corrupted_object = corrupted.state.objects.get(object_id)
        if corrupted_object is None:
            scores.append(0.0)
            continue
        type_match = clean_object.type == corrupted_object.type
        clean_position = clean_object.attributes.get("position")
        corrupted_position = corrupted_object.attributes.get("position")
        position_match = _position_distance(clean_position, corrupted_position) <= position_tolerance
        scores.append(1.0 if type_match and position_match else 0.0)
    return statistics.fmean(scores) if scores else 1.0


def _position_distance(left: object, right: object) -> float:
    if not isinstance(left, list) or not isinstance(right, list):
        return math.inf
    if len(left) < 3 or len(right) < 3:
        return math.inf
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _safe_ratio(numerator: int, denominator: int, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return float(numerator) / float(denominator)


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["background_objects"], row["corruption"])].append(row)

    numeric_fields = [
        "total_objects",
        "corrupted_objects",
        "identity_recall",
        "semantic_identity_consistency",
        "relation_precision",
        "relation_recall",
        "frontier_recall",
        "selected_k",
        "objectification_score",
        "identity_coverage_report",
        "relation_validity_report",
        "object_confidence_report",
        "relation_confidence_report",
        "delta_locality_report",
        "frontier_completeness_report",
        "semantic_identity_consistency_report",
    ]
    summary: list[dict[str, str]] = []
    for (background, corruption), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        row = {
            "row_type": "summary",
            "seed": "all",
            "sample_index": "all",
            "background_objects": background,
            "corruption": corruption,
            "sample_count": str(len(group)),
        }
        for field in numeric_fields:
            value = statistics.fmean(float(item[field]) for item in group)
            if field in {"total_objects", "corrupted_objects", "selected_k"}:
                row[field] = f"{value:.3f}"
            else:
                row[field] = f"{value:.6f}"
        summary.append(row)
    return summary


if __name__ == "__main__":
    main()
