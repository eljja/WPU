from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery, WorldCausalSlice
from wpu.core.state import Event, Relation, WorldObject, WorldState


@dataclass(slots=True)
class Sample:
    state: WorldState
    hierarchy: HierarchicalWorldState
    event: Event
    expected_delta: dict[str, float]


class LocalDeltaHead(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe learned local correction after WPU v3 escalation.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--true-relation-confidences", type=float, nargs="+", default=[0.95, 0.2])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--false-positive-rates", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--min-relation-confidence", type=float, default=0.3)
    parser.add_argument("--train-samples", type=int, default=384)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/world_copy_learned_correction_probe.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/world_copy_learned_correction_probe_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/world_copy_learned_correction_probe_results.ko.md"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_samples = [
        _random_sample(
            rng,
            world_sizes=args.world_sizes,
            k_values=args.k_values,
            true_relation_confidences=args.true_relation_confidences,
            missing_rates=args.missing_rates,
            false_positive_rates=args.false_positive_rates,
        )
        for _ in range(args.train_samples)
    ]
    train_features, train_targets = _tensorize_samples(
        train_samples,
        mode="hybrid_escalation_region",
        min_relation_confidence=args.min_relation_confidence,
    )
    model = LocalDeltaHead(input_dim=train_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(args.train_steps):
        prediction = model(train_features)
        loss = torch.nn.functional.mse_loss(prediction, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rows = []
    for total_n in args.world_sizes:
        for k_ref in args.k_values:
            if k_ref >= total_n:
                continue
            for true_relation_confidence in args.true_relation_confidences:
                for missing_rate in args.missing_rates:
                    for false_positive_rate in args.false_positive_rates:
                        eval_samples = [
                            _build_world(
                                total_n=total_n,
                                k_ref=k_ref,
                                true_relation_confidence=true_relation_confidence,
                                missing_rate=missing_rate,
                                false_positive_rate=false_positive_rate,
                                rng=rng,
                            )
                            for _ in range(args.eval_samples)
                        ]
                        for mode in ("sparse_confident_relations", "hybrid_escalation_region"):
                            metrics = _evaluate(
                                model,
                                eval_samples,
                                mode=mode,
                                min_relation_confidence=args.min_relation_confidence,
                            )
                            rows.append(
                                {
                                    "mode": mode,
                                    "total_n": total_n,
                                    "k_ref": k_ref,
                                    "true_relation_confidence": true_relation_confidence,
                                    "missing_rate": missing_rate,
                                    "false_positive_rate": false_positive_rate,
                                    **metrics,
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


def _random_sample(
    rng: random.Random,
    *,
    world_sizes: list[int],
    k_values: list[int],
    true_relation_confidences: list[float],
    missing_rates: list[float],
    false_positive_rates: list[float],
) -> Sample:
    total_n = rng.choice(world_sizes)
    k_ref = rng.choice([value for value in k_values if value < total_n])
    return _build_world(
        total_n=total_n,
        k_ref=k_ref,
        true_relation_confidence=rng.choice(true_relation_confidences),
        missing_rate=rng.choice(missing_rates),
        false_positive_rate=rng.choice(false_positive_rates),
        rng=rng,
    )


def _build_world(
    *,
    total_n: int,
    k_ref: int,
    true_relation_confidence: float,
    missing_rate: float,
    false_positive_rate: float,
    rng: random.Random,
) -> Sample:
    state = WorldState(time=1.0, metadata={"scenario": "world_copy_learned_correction_probe"})
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active", parent_id="world")
    hierarchy.add_region("background", parent_id="world")
    force = rng.uniform(0.5, 1.5)
    expected_delta: dict[str, float] = {}
    for index in range(k_ref):
        object_id = "target" if index == 0 else f"causal_{index}"
        distance = 0.05 * float(index)
        role_gain = 1.0 if index == 0 else 0.6 + 0.05 * float(index % 3)
        delta = force * role_gain / (1.0 + distance)
        expected_delta[object_id] = delta
        state.add_object(
            WorldObject(
                object_id,
                "causal_object",
                {
                    "position": [distance, 0.0, 0.0],
                    "causal_role": "active",
                    "role_gain": role_gain,
                },
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
                {
                    "position": [1000.0 + float(index), 1000.0, 0.0],
                    "causal_role": "none",
                    "role_gain": 0.0,
                },
                confidence=0.8,
            )
        )
        hierarchy.assign_object(object_id, "background")

    false_edges = int(round(false_positive_rate * max(k_ref - 1, 1)))
    for object_id in rng.sample(background_ids, k=min(false_edges, len(background_ids))):
        state.add_relation(Relation("target", object_id, "spurious", confidence=0.2))

    return Sample(
        state=state,
        hierarchy=hierarchy,
        event=Event("local_event", "target", {"force": force}, time=1.0),
        expected_delta=expected_delta,
    )


def _tensorize_samples(
    samples: list[Sample],
    *,
    mode: str,
    min_relation_confidence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
        for object_id in _selected_objects(causal_slice, mode):
            features.append(_features(sample, causal_slice, object_id))
            targets.append(sample.expected_delta.get(object_id, 0.0))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def _evaluate(
    model: LocalDeltaHead,
    samples: list[Sample],
    *,
    mode: str,
    min_relation_confidence: float,
) -> dict[str, float]:
    squared_error = 0.0
    absolute_error = 0.0
    zero_squared_error = 0.0
    count = 0
    selected_k = []
    updated_k = []
    touch_ratios = []
    escalations = []
    missed = []
    with torch.no_grad():
        for sample in samples:
            causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
            selected = _selected_objects(causal_slice, mode)
            predicted = {object_id: 0.0 for object_id in sample.expected_delta}
            if selected:
                features = torch.tensor([_features(sample, causal_slice, object_id) for object_id in selected], dtype=torch.float32)
                values = model(features).tolist()
                predicted.update(dict(zip(selected, values)))
            for object_id, target in sample.expected_delta.items():
                error = predicted[object_id] - target
                squared_error += error * error
                absolute_error += abs(error)
                zero_squared_error += target * target
                count += 1
            selected_k.append(causal_slice.causal_working_set_size)
            updated_k.append(len(selected))
            full_scan_units = len(sample.state.objects) + len(sample.state.relations)
            touch_units = int(causal_slice.retrieval_metrics["objects_examined"]) + int(
                causal_slice.retrieval_metrics["relations_examined"]
            )
            touch_ratios.append(touch_units / max(full_scan_units, 1))
            escalations.append(float(causal_slice.retrieval_metrics["escalation_required"]))
            missed.append(len(set(sample.expected_delta) - set(selected)))
    mse = squared_error / max(count, 1)
    zero_mse = zero_squared_error / max(count, 1)
    return {
        "delta_mse": round(mse, 6),
        "delta_mae": round(absolute_error / max(count, 1), 6),
        "zero_delta_mse": round(zero_mse, 6),
        "relative_mse": round(mse / max(zero_mse, 1e-12), 6),
        "mean_selected_k": round(sum(selected_k) / len(selected_k), 6),
        "mean_updated_k": round(sum(updated_k) / len(updated_k), 6),
        "max_selected_k": max(selected_k),
        "mean_touch_ratio": round(sum(touch_ratios) / len(touch_ratios), 8),
        "max_touch_ratio": round(max(touch_ratios), 8),
        "mean_escalation": round(sum(escalations) / len(escalations), 6),
        "mean_missed_causal": round(sum(missed) / len(missed), 6),
    }


def _query(sample: Sample, *, min_relation_confidence: float) -> WorldCausalSlice:
    return WorldCausalIndex(sample.state, sample.hierarchy).query(
        WorldCausalQuery(
            event=sample.event,
            max_objects=64,
            relation_depth=1,
            spatial_radius=0.25,
            include_uncertain=False,
            include_recent=False,
            min_relation_confidence=min_relation_confidence,
        )
    )


def _selected_objects(causal_slice: WorldCausalSlice, mode: str) -> list[str]:
    confident_relation = [
        object_id
        for object_id in causal_slice.object_ids
        if causal_slice.reason_by_object[object_id] & {"event_target", "relation_frontier"}
    ]
    if mode == "sparse_confident_relations":
        return confident_relation
    if mode == "hybrid_escalation_region":
        if causal_slice.retrieval_metrics["escalation_required"]:
            return list(causal_slice.object_ids)
        return confident_relation
    raise ValueError(f"unknown mode: {mode}")


def _features(sample: Sample, causal_slice: WorldCausalSlice, object_id: str) -> list[float]:
    obj = sample.state.objects[object_id]
    target = sample.state.objects[sample.event.target]
    position = obj.attributes["position"]
    target_position = target.attributes["position"]
    distance = abs(float(position[0]) - float(target_position[0]))
    reasons = causal_slice.reason_by_object.get(object_id, set())
    relation_confidence = _relation_confidence(sample.state, sample.event.target, object_id)
    return [
        1.0 if object_id == sample.event.target else 0.0,
        1.0 if "relation_frontier" in reasons else 0.0,
        1.0 if "same_region" in reasons else 0.0,
        1.0 if "spatial_neighbor" in reasons else 0.0,
        float(sample.event.delta["force"]),
        distance,
        float(obj.attributes.get("role_gain", 0.0)),
        float(obj.confidence),
        relation_confidence,
    ]


def _relation_confidence(state: WorldState, source: str, target: str) -> float:
    if source == target:
        return 1.0
    best = 0.0
    for relation in state.relations_for(source):
        if relation.other(source) == target:
            best = max(best, relation.confidence)
    return best


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    summary = _summarize(rows)
    if korean:
        intro = [
            "# World-Copy Learned Correction Probe",
            "",
            "이 probe는 escalation 이후 local correction 후보가 실제 learned delta update 품질을 개선하는지 측정한다.",
            "모델은 작은 relation/state-conditioned MLP이며, token/graph baseline 우월성을 주장하기 위한 실험은 아니다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    else:
        intro = [
            "# World-Copy Learned Correction Probe",
            "",
            "This probe tests whether local correction candidates after escalation improve learned delta-update quality.",
            "The model is a small relation/state-conditioned MLP; this is not a token/graph superiority benchmark.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    table_header = [
        "| mode | true relation confidence | mean delta MSE | relative MSE vs zero | mean missed causal | mean updated K | max selected K | max touch ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = [
        f"| {mode} | {true_conf} | {values['mean_delta_mse']:.6f} | {values['mean_relative_mse']:.6f} | "
        f"{values['mean_missed_causal']:.6f} | {values['mean_updated_k']:.6f} | {values['max_selected_k']} | "
        f"{values['max_touch_ratio']:.8f} |"
        for (mode, true_conf), values in summary.items()
    ]
    if korean:
        notes = [
            "",
            "## Interpretation",
            "",
            "- Low-confidence relation regime에서 sparse confident-relation update는 많은 causal object를 갱신하지 못해 learned delta MSE가 크게 남는다.",
            "- Escalation 이후 local region 후보를 허용하면 selected `K`를 bounded 상태로 유지하면서 missing causal delta를 학습 가능한 입력으로 되돌린다.",
            "- 이 결과는 correction 후보가 실제 update 품질로 이어질 수 있음을 보이는 substrate-level positive다.",
            "- 한계: controlled synthetic local law이며, dense/token/graph baseline 및 long-horizon world-copy integrity는 아직 검증하지 않는다.",
        ]
    else:
        notes = [
            "",
            "## Interpretation",
            "",
            "- In the low-confidence relation regime, sparse confident-relation updates miss many causal objects and leave high learned delta MSE.",
            "- Allowing local region candidates after escalation returns missing causal deltas to the learned update head while keeping selected `K` bounded.",
            "- This is a substrate-level positive showing that correction candidates can translate into better learned update quality.",
            "- Limitation: the law is controlled and synthetic; dense/token/graph baselines and long-horizon world-copy integrity are not tested here.",
        ]
    return "\n".join([*intro, *table_header, *table, *notes, ""])


def _summarize(rows: list[dict[str, object]]) -> dict[tuple[str, float], dict[str, float]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), float(row["true_relation_confidence"])), []).append(row)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {
            "mean_delta_mse": sum(float(row["delta_mse"]) for row in items) / len(items),
            "mean_relative_mse": sum(float(row["relative_mse"]) for row in items) / len(items),
            "mean_missed_causal": sum(float(row["mean_missed_causal"]) for row in items) / len(items),
            "mean_updated_k": sum(float(row["mean_updated_k"]) for row in items) / len(items),
            "max_selected_k": max(int(row["max_selected_k"]) for row in items),
            "max_touch_ratio": max(float(row["max_touch_ratio"]) for row in items),
        }
    return summary


if __name__ == "__main__":
    main()
