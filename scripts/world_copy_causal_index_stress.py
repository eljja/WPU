from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery
from wpu.core.state import Event, Relation, WorldObject, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test WPU v3 causal indexing under relation noise.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--false-positive-rates", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    parser.add_argument("--relation-confidence-thresholds", type=float, nargs="+", default=[0.0, 0.3])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_causal_index_stress.csv"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/world_copy_causal_index_stress_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/world_copy_causal_index_stress_results.ko.md"),
    )
    args = parser.parse_args()

    rows = []
    rng = random.Random(args.seed)
    for total_n in args.world_sizes:
        for k_ref in args.k_values:
            if k_ref >= total_n:
                continue
            for missing_rate in args.missing_rates:
                for false_positive_rate in args.false_positive_rates:
                    state, hierarchy, event, expected = _build_world(
                        total_n=total_n,
                        k_ref=k_ref,
                        missing_rate=missing_rate,
                        false_positive_rate=false_positive_rate,
                        rng=rng,
                    )
                    for min_relation_confidence in args.relation_confidence_thresholds:
                        started = time.perf_counter()
                        causal_slice = WorldCausalIndex(state, hierarchy).query(
                            WorldCausalQuery(
                                event=event,
                                max_objects=max(64, k_ref * 4),
                                relation_depth=1,
                                spatial_radius=0.25,
                                include_uncertain=False,
                                include_recent=False,
                                min_relation_confidence=min_relation_confidence,
                            )
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        selected = set(causal_slice.object_ids)
                        true_positive = len(selected & expected)
                        false_positive = len(selected - expected)
                        recall = true_positive / max(len(expected), 1)
                        precision = true_positive / max(len(selected), 1)
                        selected_k = causal_slice.causal_working_set_size
                        full_scan_units = len(state.objects) + len(state.relations)
                        touched_units = int(causal_slice.retrieval_metrics["objects_examined"]) + int(
                            causal_slice.retrieval_metrics["relations_examined"]
                        )
                        rows.append(
                            {
                                "total_n": len(state.objects),
                                "k_ref": len(expected),
                                "missing_rate": missing_rate,
                                "false_positive_rate": false_positive_rate,
                                "min_relation_confidence": min_relation_confidence,
                                "selected_k": selected_k,
                                "recall": round(recall, 6),
                                "precision": round(precision, 6),
                                "false_non_causal_selected": false_positive,
                                "affected_fraction": round(causal_slice.affected_fraction, 8),
                                "objects_examined": causal_slice.retrieval_metrics["objects_examined"],
                                "relations_examined": causal_slice.retrieval_metrics["relations_examined"],
                                "relations_rejected_low_confidence": causal_slice.retrieval_metrics[
                                    "relations_rejected_low_confidence"
                                ],
                                "candidate_scope_size": causal_slice.retrieval_metrics["candidate_scope_size"],
                                "touch_units": touched_units,
                                "full_scan_units": full_scan_units,
                                "touch_ratio": round(touched_units / max(full_scan_units, 1), 8),
                                "latency_ms": round(elapsed_ms, 6),
                            }
                        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.out_md.write_text(_report(rows, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_report(rows, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _build_world(
    *,
    total_n: int,
    k_ref: int,
    missing_rate: float,
    false_positive_rate: float,
    rng: random.Random,
) -> tuple[WorldState, HierarchicalWorldState, Event, set[str]]:
    state = WorldState(time=1.0, metadata={"scenario": "world_copy_causal_index_stress"})
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active", parent_id="world")
    hierarchy.add_region("background", parent_id="world")

    expected: set[str] = set()
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        expected.add(object_id)
        state.add_object(
            WorldObject(
                object_id,
                "causal_object",
                {"position": [0.05 * float(index), 0.0, 0.0], "causal_role": "active"},
                confidence=0.95,
                last_updated=0.5 if index == 0 else 0.0,
            )
        )
        hierarchy.assign_object(object_id, "active")
        if index > 0 and rng.random() >= missing_rate:
            state.add_relation(Relation("target", object_id, "causes", confidence=0.95))

    background_count = total_n - k_ref
    background_ids = []
    for index in range(background_count):
        object_id = f"bg_{index}"
        background_ids.append(object_id)
        state.add_object(
            WorldObject(
                object_id,
                "background",
                {"position": [1000.0 + float(index), 1000.0, 0.0], "causal_role": "none"},
                confidence=0.8,
            )
        )
        hierarchy.assign_object(object_id, "background")

    false_edges = int(round(false_positive_rate * max(k_ref - 1, 1)))
    for object_id in rng.sample(background_ids, k=min(false_edges, len(background_ids))):
        state.add_relation(Relation("target", object_id, "spurious", confidence=0.2))

    event = Event("local_event", "target", {"force": 1.0}, confidence=0.95, time=1.0)
    return state, hierarchy, event, expected


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    grouped = _summarize(rows)
    if korean:
        intro = [
            "# World-Copy Causal Index Stress",
            "",
            "이 benchmark는 WPU v3 causal index가 large `N`과 relation noise 아래에서 event-local causal slice를 얼마나 안정적으로 검색하는지 측정한다.",
            "이는 학습된 world-model accuracy가 아니라 causal retrieval substrate 검증이다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
            "| min relation confidence | missing rate | false-positive rate | mean recall | mean precision | max selected K | max touch ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        intro = [
            "# World-Copy Causal Index Stress",
            "",
            "This benchmark measures whether the WPU v3 causal index retrieves event-local causal slices under large `N` and relation noise.",
            "It is causal-retrieval substrate evidence, not trained world-model accuracy.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
            "| min relation confidence | missing rate | false-positive rate | mean recall | mean precision | max selected K | max touch ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    table = []
    for key, values in grouped.items():
        min_relation_confidence, missing_rate, false_positive_rate = key
        table.append(
            f"| {min_relation_confidence} | {missing_rate} | {false_positive_rate} | {values['mean_recall']:.6f} | "
            f"{values['mean_precision']:.6f} | {values['max_selected_k']} | {values['max_touch_ratio']:.8f} |"
        )
    if korean:
        notes = [
            "",
            "## Interpretation",
            "",
            "- Region-scoped retrieval은 `N`이 커져도 touched units를 full-state scan보다 훨씬 낮게 유지한다.",
            "- 누락된 true relation은 active region이 causal scope로 작동하기 때문에 일부 복구된다. 단, 이는 objectification과 region assignment가 맞다는 가정에 의존한다.",
            "- Relation confidence gate는 low-confidence false-positive relation을 억제해 precision을 회복한다.",
            "- 다음 failure boundary는 true causal relation의 confidence도 낮거나 calibration이 틀린 경우다.",
        ]
    else:
        notes = [
            "",
            "## Interpretation",
            "",
            "- Region-scoped retrieval keeps touched units far below full-state scan as `N` grows.",
            "- Missing true relations are partly recovered because the active region is a causal scope, but this assumes correct objectification/region assignment.",
            "- Relation confidence gating suppresses low-confidence false-positive relations and recovers precision.",
            "- The next failure boundary is low-confidence or miscalibrated true causal relations.",
        ]
    return "\n".join([*intro, *table, *notes, ""])


def _summarize(rows: list[dict[str, object]]) -> dict[tuple[float, float, float], dict[str, float]]:
    grouped: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                float(row["min_relation_confidence"]),
                float(row["missing_rate"]),
                float(row["false_positive_rate"]),
            ),
            [],
        ).append(row)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {
            "mean_recall": sum(float(row["recall"]) for row in items) / len(items),
            "mean_precision": sum(float(row["precision"]) for row in items) / len(items),
            "max_selected_k": max(int(row["selected_k"]) for row in items),
            "max_touch_ratio": max(float(row["touch_ratio"]) for row in items),
        }
    return summary


if __name__ == "__main__":
    main()
