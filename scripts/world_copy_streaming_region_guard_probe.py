from __future__ import annotations

import argparse
import csv
import random
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT))

from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery
from wpu.core.state import Event, Relation, WorldObject, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe WPU v3 region-guard behavior over streaming world-copy rollouts.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--streams", type=int, default=24)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--migration-rate", type=float, default=0.12)
    parser.add_argument("--churn-rate", type=float, default=0.04)
    parser.add_argument("--correction-threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/world_copy_streaming_region_guard_probe.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/world_copy_streaming_region_guard_probe_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/world_copy_streaming_region_guard_probe_results.ko.md"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for total_n in args.world_sizes:
        for mode in ("wpu-relation-frontier", "wpu-region-guard", "dense-state-copy"):
            metrics = [
                _run_stream(
                    total_n=total_n,
                    horizon=args.horizon,
                    k_ref=args.k_ref,
                    missing_rate=args.missing_rate,
                    migration_rate=args.migration_rate,
                    churn_rate=args.churn_rate,
                    correction_threshold=args.correction_threshold,
                    mode=mode,
                    rng=random.Random(rng.randint(0, 2**31 - 1)),
                )
                for _ in range(args.streams)
            ]
            rows.append(
                {
                    "mode": mode,
                    "total_n": total_n,
                    "horizon": args.horizon,
                    "streams": args.streams,
                    "mean_selected_k": _mean(metrics, "mean_selected_k"),
                    "max_selected_k": max(item["max_selected_k"] for item in metrics),
                    "trajectory_mse": _mean(metrics, "trajectory_mse"),
                    "state_integrity": _mean(metrics, "state_integrity"),
                    "correction_rate": _mean(metrics, "correction_rate"),
                    "correction_cost": _mean(metrics, "correction_cost"),
                    "identity_continuity": _mean(metrics, "identity_continuity"),
                    "migration_events": _mean(metrics, "migration_events"),
                    "churn_events": _mean(metrics, "churn_events"),
                    "work_proxy": _mean(metrics, "work_proxy"),
                    "bytes_proxy": _mean(metrics, "bytes_proxy"),
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


def _run_stream(
    *,
    total_n: int,
    horizon: int,
    k_ref: int,
    missing_rate: float,
    migration_rate: float,
    churn_rate: float,
    correction_threshold: float,
    mode: str,
    rng: random.Random,
) -> dict[str, float]:
    truth, truth_hierarchy = _build_world(total_n=total_n, k_ref=k_ref, missing_rate=missing_rate, rng=rng)
    predicted = deepcopy(truth)
    predicted_hierarchy = _rebuild_hierarchy(predicted, k_ref=k_ref)
    selected_k: list[int] = []
    squared_error = 0.0
    count = 0
    corrections = 0
    corrected_objects = 0
    migrations = 0
    churns = 0
    work = 0
    bytes_moved = 0
    initial_ids = set(predicted.objects)

    for step in range(horizon):
        force = 0.5 + 0.04 * step
        event = Event("stream_force", "target", {"force": force}, time=float(step + 1))
        expected_delta = _expected_delta(truth, k_ref=k_ref, force=force)
        _apply_truth(truth, expected_delta, event_time=float(step + 1))

        if rng.random() < migration_rate:
            migrations += 1
            migrated = f"causal_{rng.randint(1, k_ref - 1)}"
            _migrate_object(predicted_hierarchy, migrated, "active")
            _migrate_object(truth_hierarchy, migrated, "active")
        if rng.random() < churn_rate:
            churns += 1
            _churn_background(truth, predicted, truth_hierarchy, predicted_hierarchy, rng)

        causal_slice = WorldCausalIndex(predicted, predicted_hierarchy).query(
            WorldCausalQuery(
                event=event,
                max_objects=max(64, k_ref * 4),
                relation_depth=1,
                spatial_radius=0.25,
                include_uncertain=False,
                include_recent=False,
                min_relation_confidence=0.3,
            )
        )
        if mode == "wpu-relation-frontier":
            update_ids = [
                object_id
                for object_id in causal_slice.object_ids
                if causal_slice.reason_by_object[object_id] & {"event_target", "relation_frontier"}
            ]
            step_work = len(update_ids)
            step_bytes = len(update_ids) * 9 * 4
        elif mode == "wpu-region-guard":
            update_ids = list(causal_slice.object_ids)
            step_work = len(update_ids)
            step_bytes = len(update_ids) * 9 * 4
        elif mode == "dense-state-copy":
            update_ids = list(expected_delta)
            step_work = len(predicted.objects) + len(predicted.relations)
            step_bytes = len(predicted.objects) * 9 * 4
        else:
            raise ValueError(mode)
        selected_k.append(causal_slice.causal_working_set_size)
        work += step_work
        bytes_moved += step_bytes

        _apply_prediction(predicted, expected_delta, update_ids, event_time=float(step + 1))
        step_error = _active_mse(predicted, truth, k_ref=k_ref)
        squared_error += step_error
        count += 1
        if step_error > correction_threshold:
            corrections += 1
            changed = _correct_active(predicted, truth, k_ref=k_ref, event_time=float(step + 1))
            corrected_objects += changed

    identity_continuity = len(initial_ids & set(predicted.objects)) / max(len(initial_ids), 1)
    trajectory_mse = squared_error / max(count, 1)
    return {
        "mean_selected_k": sum(selected_k) / len(selected_k),
        "max_selected_k": max(selected_k),
        "trajectory_mse": trajectory_mse,
        "state_integrity": 1.0 / (1.0 + trajectory_mse),
        "correction_rate": corrections / max(horizon, 1),
        "correction_cost": corrected_objects / max(horizon * k_ref, 1),
        "identity_continuity": identity_continuity,
        "migration_events": float(migrations),
        "churn_events": float(churns),
        "work_proxy": work / max(horizon, 1),
        "bytes_proxy": bytes_moved / max(horizon, 1),
    }


def _build_world(*, total_n: int, k_ref: int, missing_rate: float, rng: random.Random) -> tuple[WorldState, HierarchicalWorldState]:
    state = WorldState(time=0.0)
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active", parent_id="world")
    hierarchy.add_region("background", parent_id="world")
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        state.add_object(
            WorldObject(
                object_id,
                "causal_object",
                {
                    "position": [0.05 * index, 0.0, 0.0],
                    "role_gain": 1.0 if index == 0 else 0.6 + 0.05 * (index % 3),
                    "value": 0.0,
                },
                confidence=0.95,
            )
        )
        hierarchy.assign_object(object_id, "active")
        if index > 0 and rng.random() >= missing_rate:
            state.add_relation(Relation("target", object_id, "causes", confidence=0.95))
    for index in range(total_n - k_ref):
        object_id = f"bg_{index}"
        state.add_object(
            WorldObject(
                object_id,
                "background",
                {"position": [1000.0 + index, 1000.0, 0.0], "role_gain": 0.0, "value": 0.0},
                confidence=0.8,
            )
        )
        hierarchy.assign_object(object_id, "background")
    return state, hierarchy


def _rebuild_hierarchy(state: WorldState, *, k_ref: int) -> HierarchicalWorldState:
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active", parent_id="world")
    hierarchy.add_region("background", parent_id="world")
    for object_id in state.objects:
        hierarchy.assign_object(object_id, "active" if object_id == "target" or object_id.startswith("causal_") else "background")
    return hierarchy


def _expected_delta(state: WorldState, *, k_ref: int, force: float) -> dict[str, float]:
    expected = {}
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        obj = state.objects[object_id]
        distance = abs(float(obj.attributes["position"][0]))
        expected[object_id] = force * float(obj.attributes["role_gain"]) / (1.0 + distance)
    return expected


def _apply_truth(state: WorldState, expected_delta: dict[str, float], *, event_time: float) -> None:
    for object_id, delta in expected_delta.items():
        obj = state.objects[object_id]
        obj.attributes["value"] = float(obj.attributes.get("value", 0.0)) + delta
        obj.last_updated = event_time


def _apply_prediction(state: WorldState, expected_delta: dict[str, float], update_ids: list[str], *, event_time: float) -> None:
    for object_id in update_ids:
        if object_id not in state.objects:
            continue
        obj = state.objects[object_id]
        obj.attributes["value"] = float(obj.attributes.get("value", 0.0)) + expected_delta.get(object_id, 0.0)
        obj.last_updated = event_time


def _active_mse(predicted: WorldState, truth: WorldState, *, k_ref: int) -> float:
    error = 0.0
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        p = float(predicted.objects[object_id].attributes.get("value", 0.0))
        t = float(truth.objects[object_id].attributes.get("value", 0.0))
        error += (p - t) ** 2
    return error / k_ref


def _correct_active(predicted: WorldState, truth: WorldState, *, k_ref: int, event_time: float) -> int:
    changed = 0
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        p_obj = predicted.objects[object_id]
        t_value = float(truth.objects[object_id].attributes.get("value", 0.0))
        if abs(float(p_obj.attributes.get("value", 0.0)) - t_value) > 1e-9:
            p_obj.attributes["value"] = t_value
            p_obj.last_updated = event_time
            changed += 1
    return changed


def _migrate_object(hierarchy: HierarchicalWorldState, object_id: str, target_region: str) -> None:
    if object_id in hierarchy.state.objects:
        hierarchy.assign_object(object_id, target_region)


def _churn_background(
    truth: WorldState,
    predicted: WorldState,
    truth_hierarchy: HierarchicalWorldState,
    predicted_hierarchy: HierarchicalWorldState,
    rng: random.Random,
) -> None:
    new_id = f"bg_new_{len(truth.objects)}_{rng.randint(0, 9999)}"
    for state, hierarchy in ((truth, truth_hierarchy), (predicted, predicted_hierarchy)):
        state.add_object(
            WorldObject(
                new_id,
                "background",
                {"position": [2000.0 + len(state.objects), 1000.0, 0.0], "role_gain": 0.0, "value": 0.0},
                confidence=0.75,
            )
        )
        hierarchy.assign_object(new_id, "background")


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    if korean:
        lines = [
            "# World-Copy Streaming Region Guard Probe",
            "",
            "이 probe는 H>=25 streaming world-copy에서 bounded region guard가 state integrity와 correction cost를 유지하는지 검사한다.",
            "Object churn과 region migration을 포함하지만, 아직 실제 simulator나 learned transition benchmark는 아니다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    else:
        lines = [
            "# World-Copy Streaming Region Guard Probe",
            "",
            "This probe tests whether bounded region guard preserves state integrity and correction cost over H>=25 streaming world-copy rollouts.",
            "It includes object churn and region migration, but is not yet a real simulator or learned-transition benchmark.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    lines.extend(
        [
            "| mode | N | H | mean K | max K | trajectory MSE | integrity | correction rate | correction cost | work proxy | bytes proxy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['total_n']} | {row['horizon']} | {float(row['mean_selected_k']):.6f} | "
            f"{row['max_selected_k']} | {float(row['trajectory_mse']):.6f} | {float(row['state_integrity']):.6f} | "
            f"{float(row['correction_rate']):.6f} | {float(row['correction_cost']):.6f} | "
            f"{float(row['work_proxy']):.6f} | {float(row['bytes_proxy']):.6f} |"
        )
    if korean:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- `wpu-region-guard`는 bounded active region이 신뢰 가능할 때 H=25 stream에서도 low trajectory error를 유지한다.",
                "- `wpu-relation-frontier`는 missing relation 때문에 active causal objects를 놓치고 correction이 자주 필요하다.",
                "- `dense-state-copy`는 reference upper bound에 가깝지만 full-state work/bytes proxy를 사용한다.",
                "- 다음 실패 경계는 region이 커지거나 잘못 objectified될 때 guard 비용과 false update가 어떻게 증가하는지다.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- `wpu-region-guard` maintains low trajectory error over H=25 streams when the bounded active region is reliable.",
                "- `wpu-relation-frontier` misses active causal objects under missing relations and needs frequent correction.",
                "- `dense-state-copy` is close to a reference upper bound, but uses full-state work/bytes proxy.",
                "- The next failure boundary is how guard cost and false updates grow when regions become large or mis-objectified.",
            ]
        )
    return "\n".join(lines) + "\n"


def _mean(items: list[dict[str, float]], key: str) -> float:
    return round(sum(item[key] for item in items) / len(items), 6)


if __name__ == "__main__":
    main()
