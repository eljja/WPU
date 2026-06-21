from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery
from wpu.core.state import Event, Relation, WorldObject, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe WPU v3 large-world causal indexing.")
    parser.add_argument("--background-sizes", type=int, nargs="+", default=[100, 1000, 5000, 10000])
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_index_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_index_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_index_probe_results.ko.md"))
    args = parser.parse_args()

    rows = []
    for background_size in args.background_sizes:
        state, hierarchy, event = _build_world(background_size)
        causal_slice = WorldCausalIndex(state, hierarchy).query(
            WorldCausalQuery(event=event, max_objects=args.max_objects, relation_depth=1, spatial_radius=0.5)
        )
        rows.append(
            {
                "total_objects": len(state.objects),
                "background_objects": background_size,
                "selected_k": causal_slice.causal_working_set_size,
                "affected_fraction": round(causal_slice.affected_fraction, 8),
                "selected_objects": " ".join(causal_slice.object_ids),
                "non_causal_selected": sum(1 for object_id in causal_slice.object_ids if object_id.startswith("bg_")),
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


def _build_world(background_objects: int) -> tuple[WorldState, HierarchicalWorldState, Event]:
    state = WorldState(time=1.0, metadata={"scenario": "world_copy_index_probe"})
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.8]}, confidence=0.95, last_updated=0.8))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.75]}, confidence=0.98))
    state.add_object(WorldObject("hand", "robot_hand", {"position": [0.2, 0.0, 0.9]}, confidence=0.9))
    state.add_object(WorldObject("edge", "table_edge", {"position": [0.4, 0.0, 0.75]}, confidence=0.98))
    state.add_relation(Relation("cup", "table", "on"))
    state.add_relation(Relation("hand", "cup", "near"))
    state.add_relation(Relation("cup", "edge", "near"))
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("kitchen", parent_id="apartment")
    for object_id in ["cup", "table", "hand", "edge"]:
        hierarchy.assign_object(object_id, "kitchen")
    for index in range(background_objects):
        object_id = f"bg_{index}"
        region_id = f"zone_{index // 100}"
        state.add_object(
            WorldObject(
                object_id,
                "background",
                {"position": [1000.0 + float(index), 0.0, 0.0]},
                confidence=0.8,
            )
        )
        hierarchy.add_region(region_id, parent_id="world")
        hierarchy.assign_object(object_id, region_id)
    event = Event("hand_touched_cup", "cup", {"force": 0.5}, confidence=0.95, time=1.0)
    return state, hierarchy, event


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    if korean:
        intro = [
            "# World-Copy Index Probe",
            "",
            "이 probe는 실세계 copy형 WPU에서 전체 `N`이 커져도 event-local causal working set `K`를 작게 유지할 수 있는지 확인한다.",
            "현재 probe는 학습 성능 실험이 아니라 v3 state/index substrate 검증이다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
        ]
    else:
        intro = [
            "# World-Copy Index Probe",
            "",
            "This probe checks whether a world-copy style WPU can keep the event-local causal working set `K` small as total `N` grows.",
            "It is an index/substrate validation, not a trained accuracy benchmark.",
            f"Source CSV: `{source_csv.as_posix()}`.",
        ]
    table = [
        "",
        "| total N | selected K | affected fraction | non-causal selected | selected objects |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['total_objects']} | {row['selected_k']} | {row['affected_fraction']} | "
            f"{row['non_causal_selected']} | `{row['selected_objects']}` |"
        )
    return "\n".join([*intro, *table, ""])


if __name__ == "__main__":
    main()
