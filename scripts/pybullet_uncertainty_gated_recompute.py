from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        description=(
            "Evaluate WPU-only uncertainty-gated local-dense recompute on "
            "PyBullet mechanism shifts."
        )
    )
    parser.add_argument("--train-mechanisms", nargs="+", default=["nominal", "high_force", "edge_shift", "catch_heavy"])
    parser.add_argument("--eval-mechanisms", nargs="+", default=["no_catch", "edge_high_force", "edge_catch_heavy"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23, 29, 31])
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--working-set-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pre-tensor-indexed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.34, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_uncertainty_gated_recompute.csv"))
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/pybullet_uncertainty_gated_recompute_results.md"),
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_uncertainty_gated_recompute_results.ko.md"),
    )
    parser.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    unknown = [mechanism for mechanism in args.train_mechanisms + args.eval_mechanisms if mechanism not in MECHANISMS]
    if unknown:
        raise ValueError(f"unknown mechanisms: {unknown}")
    if args.report_only:
        rows = _read_csv(args.out)
        report_rows = [row for row in rows if row["row_type"] == "summary"]
        args.out_md.write_text(_render_report(report_rows, korean=False), encoding="utf-8")
        args.out_ko_md.write_text(_render_report(report_rows, korean=True), encoding="utf-8")
        print(f"wrote={args.out_md}", flush=True)
        print(f"wrote={args.out_ko_md}", flush=True)
        return

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"train sparse/local-dense seed={seed}", flush=True)
        sparse_model = _train_model(SPARSE_MODEL, seed, args)
        recompute_model = _train_model(RECOMPUTE_MODEL, seed, args)
        for mechanism in args.eval_mechanisms:
            print(f"eval seed={seed} mechanism={mechanism}", flush=True)
            evaluation = _evaluate_pair(sparse_model, recompute_model, seed, mechanism, args)
            rows.extend(evaluation)
            _write_csv(args.out, rows)

    summary_rows = _summary(rows)
    rows.extend(summary_rows)
    _write_csv(args.out, rows)
    report_rows = [row for row in summary_rows if row["row_type"] == "summary"]
    args.out_md.write_text(_render_report(report_rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(report_rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out}", flush=True)
    print(f"wrote={args.out_md}", flush=True)
    print(f"wrote={args.out_ko_md}", flush=True)


def _evaluate_pair(
    sparse_model: torch.nn.Module,
    recompute_model: torch.nn.Module,
    seed: int,
    mechanism: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    dataset = _dataset(
        mechanism=mechanism,
        samples=args.samples,
        seed=seed + 70_000,
        args=args,
        balanced_labels=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_fn(args, SPARSE_MODEL))
    sparse_model.eval()
    recompute_model.eval()
    sparse_batches: list[torch.Tensor] = []
    recompute_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sparse_mse_batches: list[torch.Tensor] = []
    recompute_mse_batches: list[torch.Tensor] = []
    sparse_selected_k: list[float] = []
    sparse_causal_recall: list[float] = []
    sparse_dense_ratio: list[float] = []
    recompute_selected_k: list[float] = []
    recompute_causal_recall: list[float] = []
    recompute_dense_ratio: list[float] = []
    label_counts: Counter[int] = Counter()

    with torch.no_grad():
        for batch, target_delta, labels, causal_k in loader:
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)

            sparse_prediction = sparse_model(batch, num_branches=3, route_branches=3)
            sparse_selected, sparse_recall, sparse_dense = _working_set_stats(sparse_model, causal_k)
            recompute_prediction = recompute_model(batch, num_branches=3, route_branches=3)
            recompute_selected, recompute_recall, recompute_dense = _working_set_stats(recompute_model, causal_k)

            sparse_batches.append(F.softmax(sparse_prediction.branch_logits, dim=-1).detach().cpu())
            recompute_batches.append(F.softmax(recompute_prediction.branch_logits, dim=-1).detach().cpu())
            label_batches.append(labels.detach().cpu())
            sparse_mse_batches.append(_sample_mse(sparse_prediction.object_delta, target_delta).detach().cpu())
            recompute_mse_batches.append(_sample_mse(recompute_prediction.object_delta, target_delta).detach().cpu())
            sparse_selected_k.append(sparse_selected)
            sparse_causal_recall.append(sparse_recall)
            sparse_dense_ratio.append(sparse_dense)
            recompute_selected_k.append(recompute_selected)
            recompute_causal_recall.append(recompute_recall)
            recompute_dense_ratio.append(recompute_dense)
            label_counts.update(int(label) for label in labels.detach().cpu().tolist())

    sparse_probs = torch.cat(sparse_batches, dim=0)
    recompute_probs = torch.cat(recompute_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    sparse_mse = torch.cat(sparse_mse_batches, dim=0)
    recompute_mse = torch.cat(recompute_mse_batches, dim=0)

    base = {
        "train_mechanism": "+".join(args.train_mechanisms),
        "eval_mechanism": mechanism,
        "seed": seed,
        "background_objects": args.background_objects,
        "total_objects_n": args.background_objects + 5,
        "samples": int(labels.numel()),
        "majority_accuracy": _round(max(label_counts.values(), default=0) / max(int(labels.numel()), 1)),
        "seed_count": 1,
    }
    rows = [
        _metric_row(
            base,
            "wpu_sparse",
            sparse_probs,
            labels,
            sparse_mse,
            threshold=-1.0,
            dense_recompute_rate=0.0,
            selected_k=_mean(sparse_selected_k),
            causal_recall=_mean(sparse_causal_recall),
            dense_compute_ratio=_mean(sparse_dense_ratio),
        ),
        _metric_row(
            base,
            "wpu_local_dense",
            recompute_probs,
            labels,
            recompute_mse,
            threshold=-1.0,
            dense_recompute_rate=1.0,
            selected_k=_mean(recompute_selected_k),
            causal_recall=_mean(recompute_causal_recall),
            dense_compute_ratio=_mean(recompute_dense_ratio),
        ),
    ]

    confidence = sparse_probs.max(dim=-1).values
    for threshold in args.thresholds:
        use_recompute = confidence < float(threshold)
        gated_probs = torch.where(use_recompute.unsqueeze(-1), recompute_probs, sparse_probs)
        gated_mse = torch.where(use_recompute, recompute_mse, sparse_mse)
        rows.append(
            _metric_row(
                base,
                f"wpu_gated_t{float(threshold):.2f}",
                gated_probs,
                labels,
                gated_mse,
                threshold=float(threshold),
                dense_recompute_rate=float(use_recompute.float().mean().item()),
                selected_k=_mean(sparse_selected_k),
                causal_recall=_mean(sparse_causal_recall),
                dense_compute_ratio=_mean(sparse_dense_ratio),
            )
        )

    sparse_model.train()
    recompute_model.train()
    return rows


def _metric_row(
    base: dict[str, object],
    policy: str,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    sample_mse: torch.Tensor,
    *,
    threshold: float,
    dense_recompute_rate: float,
    selected_k: float,
    causal_recall: float,
    dense_compute_ratio: float,
) -> dict[str, object]:
    predicted = probabilities.argmax(dim=-1)
    correct = predicted == labels
    confidence = probabilities.max(dim=-1).values
    nll = F.nll_loss(probabilities.clamp_min(1e-8).log(), labels)
    brier = _brier_score(probabilities, labels)
    return {
        "row_type": "seed",
        "policy": policy,
        **base,
        "threshold": _round(threshold),
        "dense_recompute_rate": _round(dense_recompute_rate),
        "branch_accuracy": _round(float(correct.float().mean().item())),
        "mse": _round(float(sample_mse.mean().item())),
        "nll": _round(float(nll.item())),
        "brier": _round(float(brier.item())),
        "ece": _round(
            _ece(
                [float(value) for value in confidence.tolist()],
                [float(value) for value in correct.float().tolist()],
            )
        ),
        "selected_k_mean": _round(selected_k),
        "causal_recall_mean": _round(causal_recall),
        "dense_compute_ratio": _round(dense_compute_ratio),
    }


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, object]]] = {}
    for row in rows:
        if row["row_type"] != "seed":
            continue
        grouped.setdefault(
            (
                str(row["policy"]),
                str(row["train_mechanism"]),
                str(row["eval_mechanism"]),
                float(row["threshold"]),
            ),
            [],
        ).append(row)

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
    ]
    output: list[dict[str, object]] = []
    for (policy, train_mechanism, eval_mechanism, threshold), group in sorted(grouped.items()):
        row: dict[str, object] = {
            "row_type": "summary",
            "policy": policy,
            "train_mechanism": train_mechanism,
            "eval_mechanism": eval_mechanism,
            "seed": "all",
            "background_objects": group[0]["background_objects"],
            "total_objects_n": group[0]["total_objects_n"],
            "samples": group[0]["samples"],
            "threshold": _round(threshold),
            "seed_count": len(group),
        }
        for field in numeric_fields:
            row[field] = _round(_mean([float(item[field]) for item in group]))
        output.append(row)
    output.extend(_aggregate_summary(output))
    return output


def _aggregate_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(
            (
                str(row["policy"]),
                str(row["train_mechanism"]),
                float(row["threshold"]),
            ),
            [],
        ).append(row)

    output: list[dict[str, object]] = []
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
    ]
    for (policy, train_mechanism, threshold), group in sorted(grouped.items()):
        row: dict[str, object] = {
            "row_type": "summary",
            "policy": policy,
            "train_mechanism": train_mechanism,
            "eval_mechanism": "aggregate",
            "seed": "all",
            "background_objects": group[0]["background_objects"],
            "total_objects_n": group[0]["total_objects_n"],
            "samples": group[0]["samples"],
            "threshold": _round(threshold),
            "seed_count": group[0]["seed_count"],
        }
        for field in numeric_fields:
            row[field] = _round(_mean([float(item[field]) for item in group]))
        output.append(row)
    return output


def _render_report(summary_rows: list[dict[str, object]], *, korean: bool) -> str:
    aggregate = [row for row in summary_rows if row["eval_mechanism"] == "aggregate"]
    sparse = _policy_row(aggregate, "wpu_sparse")
    local_dense = _policy_row(aggregate, "wpu_local_dense")
    gated_candidates = [row for row in aggregate if str(row["policy"]).startswith("wpu_gated")]
    best_ece_safe = _best_calibration_safe(gated_candidates, sparse)
    low_cost_gate = _best_low_cost(gated_candidates, max_dense_rate=0.25)
    best_nll = min(gated_candidates, key=lambda row: (float(row["nll"]), float(row["dense_recompute_rate"])))

    if korean:
        title = "# PyBullet uncertainty-gated recompute 결과"
        summary = (
            "이 실험은 sparse WPU가 낮은 branch confidence를 보일 때 같은 WPU family의 "
            "local-dense recompute로만 넘기는 정책을 평가한다. Token 또는 graph baseline으로 "
            "fallback하지 않기 때문에 WPU 내부 uncertainty routing 검증이다."
        )
        table = "| 정책 | Accuracy | ECE | Brier | NLL | Dense recompute rate |"
        rows = [
            _table_row_ko("Sparse WPU", sparse),
            _table_row_ko("Local-dense WPU", local_dense),
            _table_row_ko("Best ECE-safe gate", best_ece_safe),
            _table_row_ko("Best low-cost gate", low_cost_gate),
            _table_row_ko("Best NLL gate", best_nll),
        ]
        interpretation = [
            "## 해석",
            "",
            (
                f"- ECE-safe gate는 sparse 대비 accuracy를 "
                f"{_delta(best_ece_safe, sparse, 'branch_accuracy'):+.6f}, ECE를 "
                f"{_delta(best_ece_safe, sparse, 'ece'):+.6f}, dense recompute rate를 "
                f"{float(best_ece_safe['dense_recompute_rate']):.6f}로 만든다."
            ),
            (
                f"- NLL-selected gate는 sparse 대비 NLL을 "
                f"{_delta(best_nll, sparse, 'nll'):+.6f}, ECE를 "
                f"{_delta(best_nll, sparse, 'ece'):+.6f} 변화시킨다."
            ),
            (
                f"- Low-cost gate는 dense recompute rate "
                f"{float(low_cost_gate['dense_recompute_rate']):.6f}에서 accuracy를 "
                f"{_delta(low_cost_gate, sparse, 'branch_accuracy'):+.6f}, ECE를 "
                f"{_delta(low_cost_gate, sparse, 'ece'):+.6f} 변화시킨다. 따라서 현재 threshold gate는 "
                "calibration을 개선할 수 있지만, 저비용 sparse routing 해법으로는 아직 부족하다."
            ),
            "- 이 결과는 WPU의 calibration 개선 방향이 token fallback이 아니라 state-native uncertainty routing일 수 있음을 검증한다. 다만 threshold는 아직 hand policy이고, 유의미한 개선은 거의 full recompute에 가까우므로 학습 가능한 gate와 held-out threshold selection이 다음 단계다.",
        ]
    else:
        title = "# PyBullet Uncertainty-Gated Recompute Results"
        summary = (
            "This experiment routes low-confidence sparse WPU predictions to a "
            "local-dense recompute path within the WPU family. It is not a token "
            "or graph fallback; it tests state-native uncertainty routing."
        )
        table = "| Policy | Accuracy | ECE | Brier | NLL | Dense recompute rate |"
        rows = [
            _table_row("Sparse WPU", sparse),
            _table_row("Local-dense WPU", local_dense),
            _table_row("Best ECE-safe gate", best_ece_safe),
            _table_row("Best low-cost gate", low_cost_gate),
            _table_row("Best NLL gate", best_nll),
        ]
        interpretation = [
            "## Interpretation",
            "",
            (
                f"- The ECE-safe gate changes accuracy by "
                f"{_delta(best_ece_safe, sparse, 'branch_accuracy'):+.6f}, ECE by "
                f"{_delta(best_ece_safe, sparse, 'ece'):+.6f}, with dense recompute rate "
                f"{float(best_ece_safe['dense_recompute_rate']):.6f}."
            ),
            (
                f"- The NLL-selected gate changes NLL by "
                f"{_delta(best_nll, sparse, 'nll'):+.6f} and ECE by "
                f"{_delta(best_nll, sparse, 'ece'):+.6f} versus sparse WPU."
            ),
            (
                f"- The low-cost gate has dense recompute rate "
                f"{float(low_cost_gate['dense_recompute_rate']):.6f}, changing accuracy by "
                f"{_delta(low_cost_gate, sparse, 'branch_accuracy'):+.6f} and ECE by "
                f"{_delta(low_cost_gate, sparse, 'ece'):+.6f}. The current threshold gate can improve "
                "calibration, but it is not yet a low-cost sparse-routing solution."
            ),
            "- The result tests whether WPU calibration can be improved by state-native uncertainty routing rather than by returning to token processing. The remaining gap is that the threshold is still a hand policy and the useful aggregate improvement is close to full recompute; the next step is a learned gate with held-out threshold selection.",
        ]

    source_line = (
        "Source CSV: `docs/experiments/pybullet_uncertainty_gated_recompute.csv`"
        if not korean
        else "Source CSV: `docs/experiments/pybullet_uncertainty_gated_recompute.csv`"
    )
    lines = [
        title,
        "",
        source_line,
        "",
        summary,
        "",
        table,
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        *interpretation,
    ]
    lines.append("")
    lines.append("## Per-Mechanism Summary" if not korean else "## Mechanism별 요약")
    lines.append("")
    lines.append("| Mechanism | Sparse acc | Sparse ECE | Best gate | Gate acc | Gate ECE | Dense rate |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    mechanisms = sorted({str(row["eval_mechanism"]) for row in summary_rows if row["eval_mechanism"] != "aggregate"})
    for mechanism in mechanisms:
        mechanism_rows = [row for row in summary_rows if row["eval_mechanism"] == mechanism]
        mechanism_sparse = _policy_row(mechanism_rows, "wpu_sparse")
        mechanism_candidates = [row for row in mechanism_rows if str(row["policy"]).startswith("wpu_gated")]
        mechanism_gate = _best_calibration_safe(mechanism_candidates, mechanism_sparse)
        safe_label = "" if _has_calibration_safe(mechanism_candidates, mechanism_sparse) else " (not ECE-safe)"
        lines.append(
            f"| {mechanism} | {float(mechanism_sparse['branch_accuracy']):.6f} | "
            f"{float(mechanism_sparse['ece']):.6f} | `{mechanism_gate['policy']}`{safe_label} | "
            f"{float(mechanism_gate['branch_accuracy']):.6f} | {float(mechanism_gate['ece']):.6f} | "
            f"{float(mechanism_gate['dense_recompute_rate']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _best_calibration_safe(candidates: list[dict[str, object]], sparse: dict[str, object]) -> dict[str, object]:
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


def _has_calibration_safe(candidates: list[dict[str, object]], sparse: dict[str, object]) -> bool:
    return any(
        float(row["branch_accuracy"]) >= float(sparse["branch_accuracy"])
        and float(row["ece"]) <= float(sparse["ece"])
        for row in candidates
    )


def _best_low_cost(candidates: list[dict[str, object]], *, max_dense_rate: float) -> dict[str, object]:
    low_cost = [row for row in candidates if float(row["dense_recompute_rate"]) <= max_dense_rate]
    pool = low_cost or candidates
    return min(
        pool,
        key=lambda row: (
            float(row["ece"]),
            -float(row["branch_accuracy"]),
            float(row["dense_recompute_rate"]),
        ),
    )


def _sample_mse(object_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return ((object_delta - target_delta) ** 2).flatten(start_dim=1).mean(dim=1)


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


def _table_row_ko(label: str, row: dict[str, object]) -> str:
    return _table_row(label, row)


def _delta(row: dict[str, object], baseline: dict[str, object], field: str) -> float:
    return float(row[field]) - float(baseline[field])


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
        "train_mechanism",
        "eval_mechanism",
        "seed",
        "background_objects",
        "total_objects_n",
        "samples",
        "threshold",
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
