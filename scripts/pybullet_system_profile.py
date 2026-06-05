from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import wpu
from wpu.core.state import Branch, DeltaState
from wpu.data.pybullet_cup import PyBulletCupDataset
from wpu.data.working_set_physics import _indexed_object_ids, _project_state
from wpu.models.batch import StateGraphBatch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile PyBullet WPU state/tensor/branch memory proxies without training."
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--background-objects", type=int, nargs="+", default=[0, 32, 128, 512])
    parser.add_argument("--branch-counts", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--forward-repeats", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_system_profile.csv"))
    args = parser.parse_args()

    forward_models = _make_forward_models(args) if args.forward_repeats > 0 else {}
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
                sample = dataset[sample_index]
                for branch_count in args.branch_counts:
                    rows.append(
                        _profile_sample(
                            seed=seed,
                            sample_index=sample_index,
                            background_objects=background,
                            branch_count=branch_count,
                            state=sample.state,
                            event=sample.event,
                            max_nodes=args.max_nodes,
                            max_depth=args.max_depth,
                            forward_models=forward_models,
                            forward_repeats=args.forward_repeats,
                        )
                    )

    summary_rows = _summarize(rows)
    output_rows = rows + summary_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    for row in summary_rows:
        print(
            "pybullet_system_profile "
            f"N={row['total_objects']} "
            f"B={row['branch_count']} "
            f"K={row['selected_objects']} "
            f"tensor_reduction={row['tensor_byte_reduction']} "
            f"branch_memory_reduction={row['branch_memory_reduction']}"
        )


def _profile_sample(
    *,
    seed: int,
    sample_index: int,
    background_objects: int,
    branch_count: int,
    state: wpu.WorldState,
    event: wpu.Event,
    max_nodes: int,
    max_depth: int,
    forward_models: dict[str, torch.nn.Module],
    forward_repeats: int,
) -> dict[str, str]:
    selected_ids = _indexed_object_ids(state, event, max_nodes=max_nodes, max_depth=max_depth)
    selected_state = _project_state(state, selected_ids)
    full_start = time.perf_counter()
    full_batch = StateGraphBatch.from_world_states([state], [event])
    full_tensorize_ms = (time.perf_counter() - full_start) * 1000.0
    selected_start = time.perf_counter()
    selected_batch = StateGraphBatch.from_world_states([selected_state], [event])
    selected_tensorize_ms = (time.perf_counter() - selected_start) * 1000.0

    full_tensor_bytes = _batch_tensor_bytes(full_batch)
    selected_tensor_bytes = _batch_tensor_bytes(selected_batch)
    full_state_memory = wpu.estimate_memory(state).total
    selected_state_memory = wpu.estimate_memory(selected_state).total
    branches = _make_branch_overlays(selected_ids, branch_count)
    branch_overlay_bytes = wpu.estimate_memory(state, branches=branches).branch_memory
    base_plus_overlay_bytes = full_state_memory + branch_overlay_bytes
    full_copy_branch_bytes = full_state_memory * max(branch_count, 1)
    report = wpu.evaluate_objectification(state, expected_working_set=set(selected_ids))

    total_objects = len(state.objects)
    selected_objects = len(selected_ids)
    total_relations = len(state.relations)
    selected_relations = len(selected_state.relations)
    dense_object_work_proxy = total_objects * total_objects * branch_count
    sparse_object_work_proxy = max(selected_objects, 1) * max(selected_relations, 1) * branch_count
    forward_profile = _forward_profile(
        full_batch=full_batch,
        selected_batch=selected_batch,
        models=forward_models,
        repeats=forward_repeats,
    )

    row = {
        "row_type": "sample",
        "seed": str(seed),
        "sample_index": str(sample_index),
        "background_objects": str(background_objects),
        "branch_count": str(branch_count),
        "total_objects": str(total_objects),
        "total_relations": str(total_relations),
        "selected_objects": str(selected_objects),
        "selected_relations": str(selected_relations),
        "objectification_score": f"{report.contract_score:.6f}",
        "full_tensor_bytes": str(full_tensor_bytes),
        "selected_tensor_bytes": str(selected_tensor_bytes),
        "tensor_byte_reduction": f"{_safe_reduction(full_tensor_bytes, selected_tensor_bytes):.6f}",
        "full_tensorize_ms": f"{full_tensorize_ms:.6f}",
        "selected_tensorize_ms": f"{selected_tensorize_ms:.6f}",
        "tensorize_latency_reduction": f"{_safe_reduction(full_tensorize_ms, selected_tensorize_ms):.6f}",
        "full_state_memory_bytes": str(full_state_memory),
        "selected_state_memory_bytes": str(selected_state_memory),
        "selected_state_memory_reduction": f"{_safe_reduction(full_state_memory, selected_state_memory):.6f}",
        "branch_overlay_bytes": str(branch_overlay_bytes),
        "base_plus_overlay_bytes": str(base_plus_overlay_bytes),
        "full_copy_branch_bytes": str(full_copy_branch_bytes),
        "branch_memory_reduction": f"{_safe_reduction(full_copy_branch_bytes, base_plus_overlay_bytes):.6f}",
        "dense_object_work_proxy": str(dense_object_work_proxy),
        "sparse_object_work_proxy": str(sparse_object_work_proxy),
        "work_proxy_reduction": f"{_safe_reduction(dense_object_work_proxy, sparse_object_work_proxy):.6f}",
        "full_graph_forward_ms": f"{forward_profile['full_graph_forward_ms']:.6f}",
        "selected_sparse_forward_ms": f"{forward_profile['selected_sparse_forward_ms']:.6f}",
        "selected_local_dense_forward_ms": f"{forward_profile['selected_local_dense_forward_ms']:.6f}",
        "sparse_forward_latency_reduction": f"{_safe_reduction(forward_profile['full_graph_forward_ms'], forward_profile['selected_sparse_forward_ms']):.6f}",
        "local_dense_forward_latency_reduction": f"{_safe_reduction(forward_profile['full_graph_forward_ms'], forward_profile['selected_local_dense_forward_ms']):.6f}",
        "sample_count": "1",
    }
    return row


def _make_forward_models(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    models = {
        "graph-transformer": wpu.create_model(
            "graph-transformer",
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            num_heads=args.num_heads,
        ).eval(),
        "wpu-cws-indexed-sparse": wpu.create_model(
            "wpu-cws-indexed-sparse",
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            num_heads=args.num_heads,
            working_set_size=args.max_nodes,
        ).eval(),
        "wpu-cws-indexed-local-dense": wpu.create_model(
            "wpu-cws-indexed-local-dense",
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            num_heads=args.num_heads,
            working_set_size=args.max_nodes,
        ).eval(),
    }
    return models


def _forward_profile(
    *,
    full_batch: StateGraphBatch,
    selected_batch: StateGraphBatch,
    models: dict[str, torch.nn.Module],
    repeats: int,
) -> dict[str, float]:
    if repeats <= 0 or not models:
        return {
            "full_graph_forward_ms": 0.0,
            "selected_sparse_forward_ms": 0.0,
            "selected_local_dense_forward_ms": 0.0,
        }
    return {
        "full_graph_forward_ms": _measure_forward_ms(models["graph-transformer"], full_batch, repeats),
        "selected_sparse_forward_ms": _measure_forward_ms(models["wpu-cws-indexed-sparse"], selected_batch, repeats),
        "selected_local_dense_forward_ms": _measure_forward_ms(models["wpu-cws-indexed-local-dense"], selected_batch, repeats),
    }


def _measure_forward_ms(model: torch.nn.Module, batch: StateGraphBatch, repeats: int) -> float:
    with torch.no_grad():
        model(batch, num_branches=3, route_branches=3)
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch, num_branches=3, route_branches=3)
        return (time.perf_counter() - start) * 1000.0 / max(repeats, 1)


def _batch_tensor_bytes(batch: StateGraphBatch) -> int:
    tensors = (
        batch.object_features,
        batch.relation_indices,
        batch.relation_features,
        batch.event_features,
        batch.object_mask,
        batch.relation_mask,
        batch.target_indices,
        batch.time_features,
    )
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def _make_branch_overlays(selected_ids: list[str], branch_count: int) -> list[Branch]:
    branches: list[Branch] = []
    probability = 1.0 / max(branch_count, 1)
    for branch_index in range(branch_count):
        delta = DeltaState(time=float(branch_index + 1), metadata={"profile": "branch_overlay"})
        for object_id in selected_ids:
            delta.record_object(
                object_id,
                {
                    "position": [0.0, 0.0, 0.0],
                    "confidence": max(0.0, 1.0 - 0.01 * branch_index),
                },
            )
        branches.append(
            Branch(
                id=f"profile_branch_{branch_index}",
                parent_id=None,
                probability=probability,
                delta_state=delta,
                time=float(branch_index + 1),
                label=f"branch_{branch_index}",
            )
        )
    return branches


def _safe_reduction(full_value: int | float, reduced_value: int | float) -> float:
    full = float(full_value)
    if full <= 0:
        return 0.0
    return max(0.0, 1.0 - float(reduced_value) / full)


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["background_objects"], row["branch_count"])].append(row)

    summary: list[dict[str, str]] = []
    numeric_fields = [
        "total_objects",
        "total_relations",
        "selected_objects",
        "selected_relations",
        "objectification_score",
        "full_tensor_bytes",
        "selected_tensor_bytes",
        "tensor_byte_reduction",
        "full_tensorize_ms",
        "selected_tensorize_ms",
        "tensorize_latency_reduction",
        "full_state_memory_bytes",
        "selected_state_memory_bytes",
        "selected_state_memory_reduction",
        "branch_overlay_bytes",
        "base_plus_overlay_bytes",
        "full_copy_branch_bytes",
        "branch_memory_reduction",
        "dense_object_work_proxy",
        "sparse_object_work_proxy",
        "work_proxy_reduction",
        "full_graph_forward_ms",
        "selected_sparse_forward_ms",
        "selected_local_dense_forward_ms",
        "sparse_forward_latency_reduction",
        "local_dense_forward_latency_reduction",
    ]
    for (background_objects, branch_count), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        row = {
            "row_type": "summary",
            "seed": "all",
            "sample_index": "all",
            "background_objects": background_objects,
            "branch_count": branch_count,
            "sample_count": str(len(group)),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in group]
            mean_value = statistics.fmean(values)
            if field.endswith("bytes") or field.endswith("objects") or field.endswith("relations") or field.endswith("proxy"):
                row[field] = f"{mean_value:.3f}"
            else:
                row[field] = f"{mean_value:.6f}"
        summary.append(row)
    return summary


if __name__ == "__main__":
    main()
