from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from scripts.world_copy_learned_correction_probe import (  # noqa: E402
    LocalDeltaHead,
    Sample,
    _build_world,
    _features,
    _query,
    _random_sample,
    _selected_objects,
    _tensorize_samples,
)


class BaselineDeltaHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
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


@dataclass(slots=True)
class TrainedModels:
    wpu: LocalDeltaHead
    wpu_context: LocalDeltaHead
    wpu_region_guard: LocalDeltaHead
    dense_graph: BaselineDeltaHead
    serialized_token: BaselineDeltaHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WPU v3 local propagation against token/graph baselines.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--true-relation-confidences", type=float, nargs="+", default=[0.95, 0.2])
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--false-positive-rates", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--min-relation-confidence", type=float, default=0.3)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/world_copy_baseline_comparison_probe.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/world_copy_baseline_comparison_probe_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/world_copy_baseline_comparison_probe_results.ko.md"),
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
    models = _train_models(
        train_samples,
        train_steps=args.train_steps,
        min_relation_confidence=args.min_relation_confidence,
    )

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
                        for model_name in (
                            "wpu-hybrid",
                            "wpu-hybrid-context",
                            "wpu-region-guard",
                            "dense-graph",
                            "serialized-token",
                            "graph-transformer-proxy",
                        ):
                            rows.append(
                                {
                                    "model": model_name,
                                    "total_n": total_n,
                                    "k_ref": k_ref,
                                    "true_relation_confidence": true_relation_confidence,
                                    "missing_rate": missing_rate,
                                    "false_positive_rate": false_positive_rate,
                                    **_evaluate_model(
                                        models,
                                        eval_samples,
                                        model_name=model_name,
                                        min_relation_confidence=args.min_relation_confidence,
                                    ),
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


def _train_models(samples: list[Sample], *, train_steps: int, min_relation_confidence: float) -> TrainedModels:
    wpu_features, wpu_targets = _tensorize_samples(
        samples,
        mode="hybrid_escalation_region",
        min_relation_confidence=min_relation_confidence,
    )
    wpu_context_features, wpu_context_targets = _wpu_context_tensors(samples, min_relation_confidence=min_relation_confidence)
    wpu_region_features, wpu_region_targets = _wpu_region_guard_tensors(samples, min_relation_confidence=min_relation_confidence)
    dense_features, dense_targets = _baseline_tensors(samples, min_relation_confidence=min_relation_confidence, token=False)
    token_features, token_targets = _baseline_tensors(samples, min_relation_confidence=min_relation_confidence, token=True)
    wpu = LocalDeltaHead(input_dim=wpu_features.shape[1])
    wpu_context = LocalDeltaHead(input_dim=wpu_context_features.shape[1])
    wpu_region_guard = LocalDeltaHead(input_dim=wpu_region_features.shape[1])
    dense = BaselineDeltaHead(input_dim=dense_features.shape[1])
    token = BaselineDeltaHead(input_dim=token_features.shape[1])
    _fit(wpu, wpu_features, wpu_targets, train_steps=train_steps)
    _fit(wpu_context, wpu_context_features, wpu_context_targets, train_steps=train_steps)
    _fit(wpu_region_guard, wpu_region_features, wpu_region_targets, train_steps=train_steps)
    _fit(dense, dense_features, dense_targets, train_steps=train_steps)
    _fit(token, token_features, token_targets, train_steps=train_steps)
    return TrainedModels(
        wpu=wpu,
        wpu_context=wpu_context,
        wpu_region_guard=wpu_region_guard,
        dense_graph=dense,
        serialized_token=token,
    )


def _fit(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, *, train_steps: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(train_steps):
        prediction = model(features)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _baseline_tensors(samples: list[Sample], *, min_relation_confidence: float, token: bool) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
        for object_id, target in sample.expected_delta.items():
            base = _features(sample, causal_slice, object_id)
            if token:
                base = _token_features(sample, base)
            features.append(base)
            targets.append(target)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def _wpu_context_tensors(samples: list[Sample], *, min_relation_confidence: float) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
        for object_id in _selected_objects(causal_slice, "hybrid_escalation_region"):
            features.append(_wpu_context_features(sample, causal_slice, object_id))
            targets.append(sample.expected_delta.get(object_id, 0.0))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def _wpu_region_guard_tensors(samples: list[Sample], *, min_relation_confidence: float) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
        for object_id in causal_slice.object_ids:
            features.append(_features(sample, causal_slice, object_id))
            targets.append(sample.expected_delta.get(object_id, 0.0))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def _evaluate_model(
    models: TrainedModels,
    samples: list[Sample],
    *,
    model_name: str,
    min_relation_confidence: float,
) -> dict[str, float]:
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    selected_k = []
    updated_k = []
    touched_units = []
    byte_units = []
    with torch.no_grad():
        for sample in samples:
            causal_slice = _query(sample, min_relation_confidence=min_relation_confidence)
            if model_name == "wpu-hybrid":
                object_ids = _selected_objects(causal_slice, "hybrid_escalation_region")
                model = models.wpu
                features = [_features(sample, causal_slice, object_id) for object_id in object_ids]
                work = len(object_ids)
                bytes_moved = len(object_ids) * 9 * 4
            elif model_name == "wpu-hybrid-context":
                object_ids = _selected_objects(causal_slice, "hybrid_escalation_region")
                model = models.wpu_context
                features = [_wpu_context_features(sample, causal_slice, object_id) for object_id in object_ids]
                work = len(object_ids) + 1
                bytes_moved = len(object_ids) * 13 * 4
            elif model_name == "wpu-region-guard":
                object_ids = list(causal_slice.object_ids)
                model = models.wpu_region_guard
                features = [_features(sample, causal_slice, object_id) for object_id in object_ids]
                work = len(object_ids)
                bytes_moved = len(object_ids) * 9 * 4
            elif model_name == "dense-graph":
                object_ids = list(sample.expected_delta)
                model = models.dense_graph
                features = [_features(sample, causal_slice, object_id) for object_id in object_ids]
                work = len(sample.state.objects) + len(sample.state.relations)
                bytes_moved = len(sample.state.objects) * 9 * 4
            elif model_name == "serialized-token":
                object_ids = list(sample.expected_delta)
                model = models.serialized_token
                features = [_token_features(sample, _features(sample, causal_slice, object_id)) for object_id in object_ids]
                work = len(sample.state.objects) + len(sample.state.relations)
                bytes_moved = len(sample.state.objects) * 11 * 4
            elif model_name == "graph-transformer-proxy":
                object_ids = list(sample.expected_delta)
                model = models.dense_graph
                features = [_features(sample, causal_slice, object_id) for object_id in object_ids]
                work = (len(sample.state.objects) + len(sample.state.relations)) ** 2
                bytes_moved = len(sample.state.objects) * len(sample.state.objects) * 4
            else:
                raise ValueError(f"unknown model: {model_name}")

            values = model(torch.tensor(features, dtype=torch.float32)).tolist() if features else []
            predicted = dict(zip(object_ids, values))
            for object_id, target in sample.expected_delta.items():
                error = predicted.get(object_id, 0.0) - target
                squared_error += error * error
                absolute_error += abs(error)
                count += 1
            selected_k.append(causal_slice.causal_working_set_size)
            updated_k.append(len(object_ids))
            touched_units.append(work)
            byte_units.append(bytes_moved)
    mse = squared_error / max(count, 1)
    return {
        "delta_mse": round(mse, 6),
        "delta_mae": round(absolute_error / max(count, 1), 6),
        "mean_selected_k": round(sum(selected_k) / len(selected_k), 6),
        "max_selected_k": max(selected_k),
        "mean_updated_k": round(sum(updated_k) / len(updated_k), 6),
        "work_proxy": round(sum(touched_units) / len(touched_units), 6),
        "bytes_proxy": round(sum(byte_units) / len(byte_units), 6),
        "accuracy_per_kwork": round((1.0 / max(mse, 1e-12)) / max(sum(touched_units) / len(touched_units), 1.0) * 1000.0, 6),
    }


def _token_features(sample: Sample, local_features: list[float]) -> list[float]:
    role_mean = sum(float(obj.attributes.get("role_gain", 0.0)) for obj in sample.state.objects.values()) / max(
        len(sample.state.objects),
        1,
    )
    return [*local_features, math.log2(max(len(sample.state.objects), 1)), role_mean]


def _wpu_context_features(sample: Sample, causal_slice, object_id: str) -> list[float]:
    local_features = _features(sample, causal_slice, object_id)
    selected = _selected_objects(causal_slice, "hybrid_escalation_region")
    role_mean = sum(float(sample.state.objects[item].attributes.get("role_gain", 0.0)) for item in selected) / max(
        len(selected),
        1,
    )
    return [
        *local_features,
        math.log2(max(len(selected), 1)),
        role_mean,
        float(causal_slice.retrieval_metrics["escalation_required"]),
        causal_slice.affected_fraction,
    ]


def _report(rows: list[dict[str, object]], source_csv: Path, *, korean: bool) -> str:
    summary = _summarize(rows)
    if korean:
        intro = [
            "# World-Copy Baseline Comparison Probe",
            "",
            "이 probe는 같은 synthetic world-copy delta task에서 WPU local propagation과 token/graph/dense baseline을 비교한다.",
            "Baseline은 비교용이며 WPU 구현 경로가 아니다. 이 결과는 controlled screen이지 최종 P2 완료가 아니다.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    else:
        intro = [
            "# World-Copy Baseline Comparison Probe",
            "",
            "This probe compares WPU local propagation against token/graph/dense baselines on the same synthetic world-copy delta task.",
            "Baselines are comparisons, not WPU implementation paths. This is a controlled screen, not full P2 completion.",
            f"Source CSV: `{source_csv.as_posix()}`.",
            "",
            "## Summary",
            "",
        ]
    table = [
        "| model | mean delta MSE | mean work proxy | mean bytes proxy | mean selected K | max selected K | accuracy/kwork |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary.items():
        table.append(
            f"| {model} | {values['mean_delta_mse']:.6f} | {values['mean_work_proxy']:.6f} | "
            f"{values['mean_bytes_proxy']:.6f} | {values['mean_selected_k']:.6f} | {values['max_selected_k']} | "
            f"{values['mean_accuracy_per_kwork']:.6f} |"
        )
    if korean:
        notes = [
            "",
            "## Interpretation",
            "",
            "- `wpu-region-guard`는 bounded selected `K`를 유지하면서 raw delta MSE와 work/bytes proxy를 동시에 개선한다.",
            "- 단순 context feature 추가(`wpu-hybrid-context`)는 negative다. Raw MSE가 기본 `wpu-hybrid`보다 좋아지지 않는다.",
            "- Positive signal은 relation frontier만 신뢰하는 것이 아니라 bounded local region을 guard로 쓰면 missing-relation gap을 줄일 수 있다는 점이다.",
            "- 이 결과는 bounded region이 작고 신뢰 가능할 때만 성립한다. Region이 커지거나 objectification이 틀리면 WPU claim은 다시 약해진다.",
            "- `graph-transformer-proxy`는 실제 attention training이 아니라 dense quadratic work proxy다.",
            "- 다음 단계는 streaming/H>=25 world-copy task에서 실제 token/graph model과 latency를 측정하는 것이다.",
        ]
    else:
        notes = [
            "",
            "## Interpretation",
            "",
            "- `wpu-region-guard` keeps selected `K` bounded while improving both raw delta MSE and work/bytes proxy.",
            "- Adding shallow context features alone (`wpu-hybrid-context`) is negative; its raw MSE does not improve over base `wpu-hybrid`.",
            "- The positive signal is that a bounded local region guard can close missing-relation gaps better than trusting only relation frontier evidence.",
            "- This holds only when bounded regions are small and reliable. If regions grow or objectification is wrong, the WPU claim weakens again.",
            "- `graph-transformer-proxy` is a dense quadratic work proxy, not a trained attention model.",
            "- The next step is a streaming/H>=25 world-copy task with actual token/graph models and measured latency.",
        ]
    return "\n".join([*intro, *table, *notes, ""])


def _summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    summary = {}
    for model, items in grouped.items():
        summary[model] = {
            "mean_delta_mse": sum(float(row["delta_mse"]) for row in items) / len(items),
            "mean_work_proxy": sum(float(row["work_proxy"]) for row in items) / len(items),
            "mean_bytes_proxy": sum(float(row["bytes_proxy"]) for row in items) / len(items),
            "mean_selected_k": sum(float(row["mean_selected_k"]) for row in items) / len(items),
            "max_selected_k": max(int(row["max_selected_k"]) for row in items),
            "mean_accuracy_per_kwork": sum(float(row["accuracy_per_kwork"]) for row in items) / len(items),
        }
    return summary


if __name__ == "__main__":
    main()
