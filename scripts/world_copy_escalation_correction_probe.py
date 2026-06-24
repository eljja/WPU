from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery
from wpu.core.state import Event, Relation, WorldObject, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe WPU v3 escalation-to-local-correction behavior.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--true-relation-confidences", type=float, nargs="+", default=[0.95, 0.2])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--false-positive-rates", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--min-relation-confidence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/world_copy_escalation_correction_probe.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/world_copy_escalation_correction_probe_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/world_copy_escalation_correction_probe_results.ko.md"),
    )
    args = parser.parse_args()

    rows = []
    rng = random.Random(args.seed)
    for total_n in args.world_sizes:
        for k_ref in args.k_values:
            if k_ref >= total_n:
                continue
            for true_relation_confidence in args.true_relation_confidences:
                for missing_rate in args.missing_rates:
                    for false_positive_rate in args.false_positive_rates:
                        state, hierarchy, event, expected = _build_world(
                            total_n=total_n,
                            k_ref=k_ref,
                            true_relation_confidence=true_relation_confidence,
                            missing_rate=missing_rate,
                            false_positive_rate=false_positive_rate,
                            rng=rng,
                        )
                        causal_slice = WorldCausalIndex(state, hierarchy).query(
                            WorldCausalQuery(
                                event=event,
                                max_objects=max(64, k_ref * 4),
                                relation_depth=1,
                                spatial_radius=0.25,
                                include_uncertain=False,
                                include_recent=False,
                                min_relation_confidence=args.min_relation_confidence,
                            )
                        )
                        full_scan_units = len(state.objects) + len(state.relations)
                        touch_units = int(causal_slice.retrieval_metrics["objects_examined"]) + int(
                            causal_slice.retrieval_metrics["relations_examined"]
                        )
                        for mode, predicted in _prediction_modes(causal_slice).items():
                            metrics = _set_metrics(predicted, expected)
                            rows.append(
                                {
                                    "mode": mode,
                                    "total_n": len(state.objects),
                                    "k_ref": len(expected),
                                    "true_relation_confidence": true_relation_confidence,
                                    "missing_rate": missing_rate,
                                    "false_positive_rate": false_positive_rate,
                                    "min_relation_confidence": args.min_relation_confidence,
                                    "selected_k": causal_slice.causal_working_set_size,
                                    "updated_k": len(predicted),
                                    "escalation_required": causal_slice.retrieval_metrics["escalation_required"],
                                    "recall": round(metrics["recall"], 6),
                                    "precision": round(metrics["precision"], 6),
                                    "f1": round(metrics["f1"], 6),
                                    "false_updates": metrics["false_updates"],
                                    "missed_updates": metrics["missed_updates"],
                                    "touch_ratio": round(touch_units / max(full_scan_units, 1), 8),
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
    true_relation_confidence: float,
    missing_rate: float,
    false_positive_rate: float,
    rng: random.Random,
) -> tuple[WorldState, HierarchicalWorldState, Event, set[str]]:
    state = WorldState(time=1.0, metadata={"scenario": "world_copy_escalation_correction_probe"})
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
            )
        )
        hierarchy.assign_object(object_id, "active")
        if index > 0 and rng.random() >= missing_rate:
            state.add_relation(Relation("target", object_id, "causes", confidence=true_relation_confidence))

    background_ids = []
    for index in range(total_n - k_ref):
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

    return state, hierarchy, Event("local_event", "target", {"force": 1.0}, time=1.0), expected


def _prediction_modes(causal_slice) -> dict[str, set[str]]:
    confident_relation = {
        object_id
        for object_id, reasons in causal_slice.reason_by_object.items()
        if "event_target" in reasons or "relation_frontier" in reasons
    }
    hybrid = set(causal_slice.object_ids) if causal_slice.retrieval_metrics["escalation_required"] else confident_relation
    return {
        "sparse_confident_relations": confident_relation,
        "hybrid_escalation_region": hybrid,
    }


def _set_metrics(predicted: set[str], expected: set[str]) -> dict[str, float | int]:
    true_positive = len(predicted & expected)
    false_updates = len(predicted - expected)
    missed_updates = len(expected - predicted)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(expected), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_updates": false_updates,
        "missed_updates": missed_updates,
    }


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    summary = _summarize(rows)
    if korean:
        intro = [
            "# World-Copy Escalation Correction Probe",
            "",
            "이 probe는 `escalation_required=1`일 때 sparse-only propagation과 local hybrid correction 후보 사용을 비교한다.",
            "이는 학습된 dynamics benchmark가 아니라 escalation 이후 causal update set을 회복하는지 보는 v3 substrate diagnostic이다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
            "| mode | true relation confidence | mean recall | mean precision | mean F1 | mean escalation | max selected K | max touch ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        intro = [
            "# World-Copy Escalation Correction Probe",
            "",
            "This probe compares sparse-only propagation with local hybrid correction candidates when `escalation_required=1`.",
            "It is a v3 substrate diagnostic for recovering the causal update set after escalation, not a learned dynamics benchmark.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
            "| mode | true relation confidence | mean recall | mean precision | mean F1 | mean escalation | max selected K | max touch ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    table = [
        f"| {mode} | {true_conf} | {values['mean_recall']:.6f} | {values['mean_precision']:.6f} | "
        f"{values['mean_f1']:.6f} | {values['mean_escalation']:.6f} | {values['max_selected_k']} | "
        f"{values['max_touch_ratio']:.8f} |"
        for (mode, true_conf), values in summary.items()
    ]
    if korean:
        notes = [
            "",
            "## Interpretation",
            "",
            "- True relation confidence가 높아도 missing relation이 있으면 sparse confident relation propagation은 일부 causal update를 놓친다.",
            "- True relation confidence가 낮으면 confidence gate가 relation frontier를 버리므로 sparse-only recall이 크게 떨어진다.",
            "- Escalation이 켜진 경우 local region correction 후보를 사용하면 controlled setup에서 causal update recall이 회복된다.",
            "- 다음 단계는 이 correction 후보를 실제 learned propagation head가 update quality로 전환하는지 검증하는 것이다.",
        ]
    else:
        notes = [
            "",
            "## Interpretation",
            "",
            "- Even when true relation confidence is high, missing relations make sparse confident-relation propagation miss some causal updates.",
            "- When true relation confidence is low, the confidence gate removes relation-frontier evidence and sparse-only recall drops sharply.",
            "- When escalation is active, using local region correction candidates recovers causal update recall in this controlled setup.",
            "- The next step is to test whether a learned propagation head turns these correction candidates into better update quality.",
        ]
    return "\n".join([*intro, *table, *notes, ""])


def _summarize(rows: list[dict[str, object]]) -> dict[tuple[str, float], dict[str, float]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), float(row["true_relation_confidence"])), []).append(row)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {
            "mean_recall": sum(float(row["recall"]) for row in items) / len(items),
            "mean_precision": sum(float(row["precision"]) for row in items) / len(items),
            "mean_f1": sum(float(row["f1"]) for row in items) / len(items),
            "mean_escalation": sum(float(row["escalation_required"]) for row in items) / len(items),
            "max_selected_k": max(int(row["selected_k"]) for row in items),
            "max_touch_ratio": max(float(row["touch_ratio"]) for row in items),
        }
    return summary


if __name__ == "__main__":
    main()
