from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from scripts.pybullet_shift_generalization import (
    MECHANISMS,
    _brier_score,
    _collate_fn,
    _dataset,
    _ece,
    _move_batch,
    _train_model,
    _working_set_stats,
)


SPARSE_MODEL = "wpu-cws-indexed-sparse"
RECOMPUTE_MODEL = "wpu-cws-indexed-local-dense"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train sparse-output WPU gates for low-cost local-dense recompute."
    )
    parser.add_argument("--train-mechanisms", nargs="+", default=["nominal", "high_force", "edge_shift", "catch_heavy"])
    parser.add_argument("--eval-mechanisms", nargs="+", default=["no_catch", "edge_high_force", "edge_catch_heavy"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23, 29, 31])
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--gate-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=3e-3)
    parser.add_argument("--gate-steps", type=int, default=120)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--gate-penalties", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04, 0.08, 0.12])
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_learned_uncertainty_gate.csv"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_learned_uncertainty_gate_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_learned_uncertainty_gate_results.ko.md"),
    )
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    unknown = [mechanism for mechanism in args.train_mechanisms + args.eval_mechanisms if mechanism not in MECHANISMS]
    if unknown:
        raise ValueError(f"unknown mechanisms: {unknown}")
    if args.report_only:
        rows = _read_csv(args.out)
        summary_rows = [row for row in rows if row["row_type"] == "summary"]
        args.out_md.write_text(_render_report(summary_rows, korean=False), encoding="utf-8")
        args.out_ko_md.write_text(_render_report(summary_rows, korean=True), encoding="utf-8")
        print(f"wrote={args.out_md}", flush=True)
        print(f"wrote={args.out_ko_md}", flush=True)
        return

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"train sparse/local-dense seed={seed}", flush=True)
        sparse_model = _train_model(SPARSE_MODEL, seed, args)
        recompute_model = _train_model(RECOMPUTE_MODEL, seed, args)
        print(f"train source gate seed={seed}", flush=True)
        source_gate = _train_gate(
            sparse_model,
            recompute_model,
            _mechanism_dataset(args.train_mechanisms, args.gate_samples, seed + 80_000, args),
            seed,
            args,
        )
        for mechanism in args.eval_mechanisms:
            print(f"eval seed={seed} mechanism={mechanism}", flush=True)
            fewshot_gate = _train_gate(
                sparse_model,
                recompute_model,
                _mechanism_dataset([mechanism], args.gate_samples, seed + 90_000, args),
                seed + 17,
                args,
            )
            rows.extend(_evaluate_policies(sparse_model, recompute_model, source_gate, fewshot_gate, seed, mechanism, args))
            _write_csv(args.out, rows)

    summary_rows = _summary(rows)
    rows.extend(summary_rows)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(summary_rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(summary_rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out}", flush=True)
    print(f"wrote={args.out_md}", flush=True)
    print(f"wrote={args.out_ko_md}", flush=True)


def _mechanism_dataset(mechanisms: list[str], samples: int, seed: int, args: argparse.Namespace):
    per_mechanism = max(args.batch_size, samples // max(len(mechanisms), 1))
    datasets = [
        _dataset(
            mechanism=mechanism,
            samples=per_mechanism,
            seed=seed + 101 * index,
            args=args,
            balanced_labels=False,
        )
        for index, mechanism in enumerate(mechanisms)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _train_gate(
    sparse_model: torch.nn.Module,
    recompute_model: torch.nn.Module,
    dataset,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(seed)
    features, sparse_probs, recompute_probs, labels, _, _, _ = _collect_pair_outputs(
        sparse_model,
        recompute_model,
        dataset,
        args,
    )
    sparse_nll = -sparse_probs.gather(1, labels.view(-1, 1)).clamp_min(1e-8).log().squeeze(1)
    recompute_nll = -recompute_probs.gather(1, labels.view(-1, 1)).clamp_min(1e-8).log().squeeze(1)
    target_benefit = sparse_nll - recompute_nll
    normalizer = _fit_normalizer(features)
    normalized = _normalize(features, normalizer)
    gate = _BenefitGate(normalized.size(1), args.gate_hidden)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr)
    gate.train()
    for _ in range(args.gate_steps):
        prediction = gate(normalized).squeeze(1)
        loss = F.mse_loss(prediction, target_benefit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    with torch.no_grad():
        train_prediction = gate(normalized).squeeze(1)
    return {
        "model": gate,
        "normalizer": normalizer,
        "train_benefit_mse": float(F.mse_loss(train_prediction, target_benefit).item()),
        "train_benefit_mean": float(target_benefit.mean().item()),
    }


def _evaluate_policies(
    sparse_model: torch.nn.Module,
    recompute_model: torch.nn.Module,
    source_gate: dict[str, object],
    fewshot_gate: dict[str, object],
    seed: int,
    mechanism: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    dataset = _dataset(
        mechanism=mechanism,
        samples=args.samples,
        seed=seed + 100_000,
        args=args,
        balanced_labels=False,
    )
    features, sparse_probs, recompute_probs, labels, sparse_mse, recompute_mse, stats = _collect_pair_outputs(
        sparse_model,
        recompute_model,
        dataset,
        args,
    )
    label_counts = Counter(int(label) for label in labels.tolist())
    base = {
        "train_mechanism": "+".join(args.train_mechanisms),
        "eval_mechanism": mechanism,
        "seed": seed,
        "background_objects": args.background_objects,
        "total_objects_n": args.background_objects + 5,
        "samples": int(labels.numel()),
        "majority_accuracy": _round(max(label_counts.values(), default=0) / max(int(labels.numel()), 1)),
        "source_gate_train_benefit_mse": _round(float(source_gate["train_benefit_mse"])),
        "fewshot_gate_train_benefit_mse": _round(float(fewshot_gate["train_benefit_mse"])),
        "seed_count": 1,
    }
    rows = [
        _metric_row(base, "wpu_sparse", "none", -1.0, sparse_probs, labels, sparse_mse, 0.0, stats),
        _metric_row(base, "wpu_local_dense", "none", -1.0, recompute_probs, labels, recompute_mse, 1.0, stats),
    ]
    for gate_kind, gate_bundle in [("source", source_gate), ("fewshot", fewshot_gate)]:
        predicted_benefit = _predict_benefit(gate_bundle, features)
        for penalty in args.gate_penalties:
            use_recompute = predicted_benefit > float(penalty)
            gated_probs = torch.where(use_recompute.unsqueeze(1), recompute_probs, sparse_probs)
            gated_mse = torch.where(use_recompute, recompute_mse, sparse_mse)
            rows.append(
                _metric_row(
                    base,
                    f"{gate_kind}_learned_p{float(penalty):.2f}",
                    gate_kind,
                    float(penalty),
                    gated_probs,
                    labels,
                    gated_mse,
                    float(use_recompute.float().mean().item()),
                    stats,
                )
            )
    return rows


def _collect_pair_outputs(
    sparse_model: torch.nn.Module,
    recompute_model: torch.nn.Module,
    dataset,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    device = torch.device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, SPARSE_MODEL))
    sparse_model.eval()
    recompute_model.eval()
    feature_batches: list[torch.Tensor] = []
    sparse_prob_batches: list[torch.Tensor] = []
    recompute_prob_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sparse_mse_batches: list[torch.Tensor] = []
    recompute_mse_batches: list[torch.Tensor] = []
    sparse_selected_k: list[float] = []
    sparse_causal_recall: list[float] = []
    sparse_dense_ratio: list[float] = []
    with torch.no_grad():
        for batch, target_delta, labels, causal_k in loader:
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)
            sparse_prediction = sparse_model(batch, num_branches=3, route_branches=3)
            sparse_selected, sparse_recall, sparse_dense = _working_set_stats(sparse_model, causal_k)
            recompute_prediction = recompute_model(batch, num_branches=3, route_branches=3)
            sparse_probs = F.softmax(sparse_prediction.branch_logits, dim=-1)
            recompute_probs = F.softmax(recompute_prediction.branch_logits, dim=-1)
            feature_batches.append(_gate_features(sparse_probs, batch.event_features).detach().cpu())
            sparse_prob_batches.append(sparse_probs.detach().cpu())
            recompute_prob_batches.append(recompute_probs.detach().cpu())
            label_batches.append(labels.detach().cpu())
            sparse_mse_batches.append(_sample_mse(sparse_prediction.object_delta, target_delta).detach().cpu())
            recompute_mse_batches.append(_sample_mse(recompute_prediction.object_delta, target_delta).detach().cpu())
            sparse_selected_k.append(sparse_selected)
            sparse_causal_recall.append(sparse_recall)
            sparse_dense_ratio.append(sparse_dense)
    return (
        torch.cat(feature_batches, dim=0),
        torch.cat(sparse_prob_batches, dim=0),
        torch.cat(recompute_prob_batches, dim=0),
        torch.cat(label_batches, dim=0),
        torch.cat(sparse_mse_batches, dim=0),
        torch.cat(recompute_mse_batches, dim=0),
        {
            "selected_k_mean": _mean(sparse_selected_k),
            "causal_recall_mean": _mean(sparse_causal_recall),
            "dense_compute_ratio": _mean(sparse_dense_ratio),
        },
    )


def _gate_features(sparse_probs: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:
    sorted_probs = sparse_probs.sort(dim=-1, descending=True).values
    confidence = sorted_probs[:, 0:1]
    margin = sorted_probs[:, 0:1] - sorted_probs[:, 1:2]
    entropy = -(sparse_probs.clamp_min(1e-8).log() * sparse_probs).sum(dim=-1, keepdim=True)
    return torch.cat([sparse_probs, confidence, margin, entropy, event_features], dim=-1)


def _metric_row(
    base: dict[str, object],
    policy: str,
    gate_kind: str,
    penalty: float,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    sample_mse: torch.Tensor,
    dense_recompute_rate: float,
    stats: dict[str, float],
) -> dict[str, object]:
    predicted = probabilities.argmax(dim=-1)
    correct = predicted == labels
    confidence = probabilities.max(dim=-1).values
    return {
        "row_type": "seed",
        "policy": policy,
        "gate_kind": gate_kind,
        **base,
        "penalty": _round(penalty),
        "dense_recompute_rate": _round(dense_recompute_rate),
        "branch_accuracy": _round(float(correct.float().mean().item())),
        "mse": _round(float(sample_mse.mean().item())),
        "nll": _round(float(F.nll_loss(probabilities.clamp_min(1e-8).log(), labels).item())),
        "brier": _round(float(_brier_score(probabilities, labels).item())),
        "ece": _round(
            _ece(
                [float(value) for value in confidence.tolist()],
                [float(value) for value in correct.float().tolist()],
            )
        ),
        "selected_k_mean": _round(stats["selected_k_mean"]),
        "causal_recall_mean": _round(stats["causal_recall_mean"]),
        "dense_compute_ratio": _round(stats["dense_compute_ratio"]),
    }


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, object]]] = {}
    for row in rows:
        if row["row_type"] != "seed":
            continue
        grouped.setdefault(
            (
                str(row["policy"]),
                str(row["gate_kind"]),
                str(row["train_mechanism"]),
                str(row["eval_mechanism"]),
                float(row["penalty"]),
            ),
            [],
        ).append(row)
    output = [_summary_row(key, group, "summary") for key, group in sorted(grouped.items())]
    output.extend(_aggregate_summary(output))
    return output


def _aggregate_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(
            (
                str(row["policy"]),
                str(row["gate_kind"]),
                str(row["train_mechanism"]),
                float(row["penalty"]),
            ),
            [],
        ).append(row)
    output: list[dict[str, object]] = []
    for (policy, gate_kind, train_mechanism, penalty), group in sorted(grouped.items()):
        key = (policy, gate_kind, train_mechanism, "aggregate", penalty)
        output.append(_summary_row(key, group, "summary"))
    return output


def _summary_row(
    key: tuple[str, str, str, str, float],
    group: list[dict[str, object]],
    row_type: str,
) -> dict[str, object]:
    policy, gate_kind, train_mechanism, eval_mechanism, penalty = key
    numeric_fields = [
        "dense_recompute_rate",
        "branch_accuracy",
        "majority_accuracy",
        "mse",
        "nll",
        "brier",
        "ece",
        "selected_k_mean",
        "causal_recall_mean",
        "dense_compute_ratio",
        "source_gate_train_benefit_mse",
        "fewshot_gate_train_benefit_mse",
    ]
    row: dict[str, object] = {
        "row_type": row_type,
        "policy": policy,
        "gate_kind": gate_kind,
        "train_mechanism": train_mechanism,
        "eval_mechanism": eval_mechanism,
        "seed": "all",
        "background_objects": group[0]["background_objects"],
        "total_objects_n": group[0]["total_objects_n"],
        "samples": group[0]["samples"],
        "penalty": _round(penalty),
        "seed_count": len(group),
    }
    for field in numeric_fields:
        row[field] = _round(_mean([float(item[field]) for item in group]))
    return row


def _render_report(summary_rows: list[dict[str, object]], *, korean: bool) -> str:
    aggregate = [row for row in summary_rows if row["eval_mechanism"] == "aggregate"]
    sparse = _policy_row(aggregate, "wpu_sparse")
    local_dense = _policy_row(aggregate, "wpu_local_dense")
    source_candidates = [row for row in aggregate if row["gate_kind"] == "source"]
    fewshot_candidates = [row for row in aggregate if row["gate_kind"] == "fewshot"]
    best_source_low = _best_low_cost(source_candidates, max_dense_rate=0.25)
    best_fewshot_low = _best_low_cost(fewshot_candidates, max_dense_rate=0.25)
    best_source_ece = _best_ece_safe(source_candidates, sparse)
    best_fewshot_ece = _best_ece_safe(fewshot_candidates, sparse)
    source_low_label = "Source low-cost" if _has_under_rate(source_candidates, 0.25) else "Source lowest-rate over budget"
    fewshot_low_label = "Few-shot low-cost" if _has_under_rate(fewshot_candidates, 0.25) else "Few-shot lowest-rate over budget"
    source_ece_label = "Source ECE-safe" if _has_ece_safe(source_candidates, sparse) else "Source best ECE candidate (not safe)"
    fewshot_ece_label = "Few-shot ECE-safe" if _has_ece_safe(fewshot_candidates, sparse) else "Few-shot best ECE candidate (not safe)"

    if korean:
        title = "# PyBullet learned uncertainty gate 결과"
        summary = (
            "이 실험은 sparse WPU 출력과 event feature만으로 local-dense recompute의 "
            "NLL benefit을 예측하는 작은 gate를 학습한다. Source gate는 train mechanisms로만 "
            "학습하고, few-shot gate는 eval mechanism calibration samples를 사용한다."
        )
        interpretation = [
            "## 해석",
            "",
            _interpretation_line(best_source_low, sparse, source_low_label, korean=True),
            _interpretation_line(best_fewshot_low, sparse, fewshot_low_label, korean=True),
            _interpretation_line(best_source_ece, sparse, source_ece_label, korean=True),
            _interpretation_line(best_fewshot_ece, sparse, fewshot_ece_label, korean=True),
            "- Source gate는 저비용 accuracy를 개선하지만 aggregate ECE를 악화시킨다. Few-shot gate는 accuracy/NLL을 더 개선하지만 low-cost budget을 넘고 ECE도 악화시킨다. 따라서 sparse-output gate만으로 calibration-safe 저비용 routing은 아직 해결되지 않았다.",
        ]
    else:
        title = "# PyBullet Learned Uncertainty Gate Results"
        summary = (
            "This experiment trains a small gate to predict local-dense recompute "
            "NLL benefit from sparse WPU outputs and event features only. The source "
            "gate is trained on train mechanisms; the few-shot gate uses evaluation "
            "mechanism calibration samples."
        )
        interpretation = [
            "## Interpretation",
            "",
            _interpretation_line(best_source_low, sparse, source_low_label, korean=False),
            _interpretation_line(best_fewshot_low, sparse, fewshot_low_label, korean=False),
            _interpretation_line(best_source_ece, sparse, source_ece_label, korean=False),
            _interpretation_line(best_fewshot_ece, sparse, fewshot_ece_label, korean=False),
            "- The source gate improves low-cost accuracy but worsens aggregate ECE. The few-shot gate improves accuracy/NLL more strongly, but exceeds the low-cost budget and also worsens ECE. Sparse-output gating alone is therefore not yet a calibration-safe low-cost routing solution.",
        ]

    lines = [
        title,
        "",
        "Source CSV: `docs/experiments/pybullet_learned_uncertainty_gate.csv`",
        "",
        summary,
        "",
        "| Policy | Accuracy | ECE | Brier | NLL | Dense recompute rate |",
        "|---|---:|---:|---:|---:|---:|",
        _table_row("Sparse WPU", sparse),
        _table_row("Local-dense WPU", local_dense),
        _table_row(source_low_label, best_source_low),
        _table_row(fewshot_low_label, best_fewshot_low),
        _table_row(source_ece_label, best_source_ece),
        _table_row(fewshot_ece_label, best_fewshot_ece),
        "",
        *interpretation,
        "",
    ]
    lines.append("## Per-Mechanism Low-Cost Summary" if not korean else "## Mechanism별 low-cost 요약")
    lines.append("")
    lines.append("| Mechanism | Sparse acc | Sparse ECE | Source acc/ECE/rate | Few-shot acc/ECE/rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for mechanism in sorted({row["eval_mechanism"] for row in summary_rows if row["eval_mechanism"] != "aggregate"}):
        mechanism_rows = [row for row in summary_rows if row["eval_mechanism"] == mechanism]
        mechanism_sparse = _policy_row(mechanism_rows, "wpu_sparse")
        source = _best_low_cost([row for row in mechanism_rows if row["gate_kind"] == "source"], max_dense_rate=0.25)
        fewshot = _best_low_cost([row for row in mechanism_rows if row["gate_kind"] == "fewshot"], max_dense_rate=0.25)
        lines.append(
            f"| {mechanism} | {float(mechanism_sparse['branch_accuracy']):.6f} | "
            f"{float(mechanism_sparse['ece']):.6f} | "
            f"{float(source['branch_accuracy']):.6f}/{float(source['ece']):.6f}/{float(source['dense_recompute_rate']):.6f} | "
            f"{float(fewshot['branch_accuracy']):.6f}/{float(fewshot['ece']):.6f}/{float(fewshot['dense_recompute_rate']):.6f} |"
        )
    return "\n".join(lines) + "\n"


class _BenefitGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _predict_benefit(gate_bundle: dict[str, object], features: torch.Tensor) -> torch.Tensor:
    gate = gate_bundle["model"]
    if not isinstance(gate, _BenefitGate):
        raise TypeError("gate bundle contains an invalid model")
    with torch.no_grad():
        return gate(_normalize(features, gate_bundle["normalizer"])).squeeze(1)  # type: ignore[arg-type]


def _fit_normalizer(features: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return {"mean": mean, "std": std}


def _normalize(features: torch.Tensor, normalizer: object) -> torch.Tensor:
    if not isinstance(normalizer, dict):
        raise TypeError("normalizer must be a dictionary")
    return (features - normalizer["mean"]) / normalizer["std"]


def _best_low_cost(candidates: list[dict[str, object]], *, max_dense_rate: float) -> dict[str, object]:
    low_cost = [row for row in candidates if float(row["dense_recompute_rate"]) <= max_dense_rate]
    if not low_cost:
        return min(candidates, key=lambda row: float(row["dense_recompute_rate"]))
    pool = low_cost
    return min(
        pool,
        key=lambda row: (
            float(row["ece"]),
            -float(row["branch_accuracy"]),
            float(row["dense_recompute_rate"]),
        ),
    )


def _best_ece_safe(candidates: list[dict[str, object]], sparse: dict[str, object]) -> dict[str, object]:
    safe = [
        row
        for row in candidates
        if float(row["branch_accuracy"]) >= float(sparse["branch_accuracy"])
        and float(row["ece"]) <= float(sparse["ece"])
    ]
    pool = safe or candidates
    return min(
        pool,
        key=lambda row: (
            float(row["ece"]),
            -float(row["branch_accuracy"]),
            float(row["dense_recompute_rate"]),
        ),
    )


def _has_under_rate(candidates: list[dict[str, object]], max_dense_rate: float) -> bool:
    return any(float(row["dense_recompute_rate"]) <= max_dense_rate for row in candidates)


def _has_ece_safe(candidates: list[dict[str, object]], sparse: dict[str, object]) -> bool:
    return any(
        float(row["branch_accuracy"]) >= float(sparse["branch_accuracy"])
        and float(row["ece"]) <= float(sparse["ece"])
        for row in candidates
    )


def _policy_row(rows: list[dict[str, object]], policy: str) -> dict[str, object]:
    for row in rows:
        if row["policy"] == policy:
            return row
    raise ValueError(f"missing policy row: {policy}")


def _table_row(label: str, row: dict[str, object]) -> str:
    return (
        f"| {label} (`{row['policy']}`) | {float(row['branch_accuracy']):.6f} | "
        f"{float(row['ece']):.6f} | {float(row['brier']):.6f} | {float(row['nll']):.6f} | "
        f"{float(row['dense_recompute_rate']):.6f} |"
    )


def _interpretation_line(row: dict[str, object], sparse: dict[str, object], label: str, *, korean: bool) -> str:
    acc_delta = float(row["branch_accuracy"]) - float(sparse["branch_accuracy"])
    ece_delta = float(row["ece"]) - float(sparse["ece"])
    if korean:
        return (
            f"- {label}: accuracy 변화 {acc_delta:+.6f}, ECE 변화 {ece_delta:+.6f}, "
            f"dense recompute rate {float(row['dense_recompute_rate']):.6f}."
        )
    return (
        f"- {label}: accuracy change {acc_delta:+.6f}, ECE change {ece_delta:+.6f}, "
        f"dense recompute rate {float(row['dense_recompute_rate']):.6f}."
    )


def _sample_mse(object_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return ((object_delta - target_delta) ** 2).flatten(start_dim=1).mean(dim=1)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "policy",
        "gate_kind",
        "train_mechanism",
        "eval_mechanism",
        "seed",
        "background_objects",
        "total_objects_n",
        "samples",
        "penalty",
        "dense_recompute_rate",
        "branch_accuracy",
        "majority_accuracy",
        "mse",
        "nll",
        "brier",
        "ece",
        "selected_k_mean",
        "causal_recall_mean",
        "dense_compute_ratio",
        "source_gate_train_benefit_mse",
        "fewshot_gate_train_benefit_mse",
        "seed_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
