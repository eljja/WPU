from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("docs/experiments")


PREDICTOR_FIELDS = [
    "objectification_score",
    "frontier_causal_recall_mean",
    "selected_k_mean",
    "identity_coverage",
    "relation_validity",
    "object_confidence",
    "relation_confidence",
    "quality_identity_recall",
    "quality_semantic_identity_consistency",
    "quality_relation_precision",
    "quality_relation_recall",
    "quality_frontier_recall",
    "quality_frontier_completeness_report",
    "quality_semantic_identity_consistency_report",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join PyBullet objectification quality metrics to downstream propagation degradation."
    )
    parser.add_argument("--stress", type=Path, default=ROOT / "pybullet_objectification_stress.csv")
    parser.add_argument("--quality", type=Path, default=ROOT / "pybullet_objectification_quality.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "pybullet_objectification_loss_coupling.csv")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "pybullet_objectification_loss_coupling_results.md",
    )
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=ROOT / "pybullet_objectification_loss_coupling_results.ko.md",
    )
    args = parser.parse_args()

    stress_rows = _read_rows(args.stress)
    quality_rows = _quality_summary_by_key(args.quality)
    joined = _joined_rows(stress_rows, quality_rows)
    rows = _corruption_summary(joined) + _predictor_correlations(joined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _quality_summary_by_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_rows(path)
    return {
        (row["background_objects"], row["corruption"]): row
        for row in rows
        if row["row_type"] == "summary"
    }


def _joined_rows(
    stress_rows: list[dict[str, str]],
    quality_rows: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, float | str]]:
    clean_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in stress_rows:
        if row["corruption"] == "clean":
            clean_by_key[(row["model"], row["seed"], row["background_objects"])] = row

    joined: list[dict[str, float | str]] = []
    for row in stress_rows:
        clean = clean_by_key.get((row["model"], row["seed"], row["background_objects"]))
        if clean is None:
            continue
        quality = quality_rows.get((row["background_objects"], row["corruption"]), {})
        clean_quality = quality_rows.get((row["background_objects"], "clean"), {})
        record: dict[str, float | str] = {
            "model": row["model"],
            "seed": row["seed"],
            "background_objects": row["background_objects"],
            "corruption": row["corruption"],
            "branch_accuracy": _num(row["branch_accuracy"]),
            "clean_branch_accuracy": _num(clean["branch_accuracy"]),
            "accuracy_drop": _num(clean["branch_accuracy"]) - _num(row["branch_accuracy"]),
            "mse": _num(row["mse"]),
            "clean_mse": _num(clean["mse"]),
            "mse_increase": _num(row["mse"]) - _num(clean["mse"]),
            "objectification_score": _num(row["objectification_score"]),
            "objectification_score_drop": _num(clean["objectification_score"]) - _num(row["objectification_score"]),
            "frontier_causal_recall_mean": _num(row["frontier_causal_recall_mean"]),
            "frontier_causal_recall_drop": _num(clean["frontier_causal_recall_mean"])
            - _num(row["frontier_causal_recall_mean"]),
            "selected_k_mean": _num(row["selected_k_mean"]),
            "selected_k_drop": _num(clean["selected_k_mean"]) - _num(row["selected_k_mean"]),
            "identity_coverage": _num(row["identity_coverage"]),
            "relation_validity": _num(row["relation_validity"]),
            "object_confidence": _num(row["object_confidence"]),
            "relation_confidence": _num(row["relation_confidence"]),
        }
        for source_field, target_field in [
            ("identity_recall", "quality_identity_recall"),
            ("semantic_identity_consistency", "quality_semantic_identity_consistency"),
            ("relation_precision", "quality_relation_precision"),
            ("relation_recall", "quality_relation_recall"),
            ("frontier_recall", "quality_frontier_recall"),
            ("frontier_completeness_report", "quality_frontier_completeness_report"),
            ("semantic_identity_consistency_report", "quality_semantic_identity_consistency_report"),
        ]:
            record[target_field] = _num(quality.get(source_field, "nan"))
            record[f"{target_field}_drop"] = _num(clean_quality.get(source_field, "nan")) - _num(
                quality.get(source_field, "nan")
            )
        joined.append(record)
    return joined


def _corruption_summary(joined: list[dict[str, float | str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in joined:
        groups[(str(row["model"]), str(row["corruption"]))].append(row)

    rows: list[dict[str, object]] = []
    for (model, corruption), group in sorted(groups.items()):
        rows.append(
            {
                "row_type": "corruption_summary",
                "model": model,
                "corruption": corruption,
                "predictor": "",
                "sample_count": len(group),
                "accuracy_drop": _round(_mean(row["accuracy_drop"] for row in group)),
                "mse_increase": _round(_mean(row["mse_increase"] for row in group)),
                "objectification_score_drop": _round(_mean(row["objectification_score_drop"] for row in group)),
                "frontier_causal_recall_drop": _round(_mean(row["frontier_causal_recall_drop"] for row in group)),
                "selected_k_drop": _round(_mean(row["selected_k_drop"] for row in group)),
                "pearson_accuracy_drop": "",
                "pearson_mse_increase": "",
                "abs_pearson_accuracy_drop": "",
                "abs_pearson_mse_increase": "",
                "interpretation": _corruption_interpretation(corruption),
            }
        )
    return rows


def _predictor_correlations(joined: list[dict[str, float | str]]) -> list[dict[str, object]]:
    corrupted = [row for row in joined if row["corruption"] != "clean"]
    rows: list[dict[str, object]] = []
    for predictor in PREDICTOR_FIELDS:
        values = [1.0 - float(row[predictor]) for row in corrupted if _finite(row.get(predictor))]
        acc = [float(row["accuracy_drop"]) for row in corrupted if _finite(row.get(predictor))]
        mse = [float(row["mse_increase"]) for row in corrupted if _finite(row.get(predictor))]
        acc_corr = _round(_pearson(values, acc))
        mse_corr = _round(_pearson(values, mse))
        rows.append(
            {
                "row_type": "predictor_correlation",
                "model": "all",
                "corruption": "non_clean",
                "predictor": predictor,
                "sample_count": len(values),
                "accuracy_drop": "",
                "mse_increase": "",
                "objectification_score_drop": "",
                "frontier_causal_recall_drop": "",
                "selected_k_drop": "",
                "pearson_accuracy_drop": acc_corr,
                "pearson_mse_increase": mse_corr,
                "abs_pearson_accuracy_drop": _round(abs(acc_corr)) if _finite(acc_corr) else "",
                "abs_pearson_mse_increase": _round(abs(mse_corr)) if _finite(mse_corr) else "",
                "interpretation": "positive means lower component quality is associated with worse downstream metric",
            }
        )
    return sorted(rows, key=lambda row: float(row["abs_pearson_mse_increase"] or 0.0), reverse=True)


def _render_report(rows: list[dict[str, object]], *, korean: bool) -> str:
    summaries = [row for row in rows if row["row_type"] == "corruption_summary"]
    correlations = [row for row in rows if row["row_type"] == "predictor_correlation"]
    non_clean_summaries = [row for row in summaries if row["corruption"] != "clean"]
    worst_acc = max(non_clean_summaries, key=lambda row: float(row["accuracy_drop"]))
    worst_mse = max(non_clean_summaries, key=lambda row: float(row["mse_increase"]))
    best_mse_predictor = max(correlations, key=lambda row: float(row["abs_pearson_mse_increase"] or 0.0))
    best_acc_predictor = max(correlations, key=lambda row: float(row["abs_pearson_accuracy_drop"] or 0.0))

    if korean:
        title = "# PyBullet Objectification-Loss Coupling Audit"
        intro = (
            "이 파생 감사는 PyBullet objectification quality metric과 downstream propagation "
            "degradation을 연결한다. 목표는 objectification score가 존재한다는 사실을 넘어서, "
            "어떤 component가 branch accuracy와 MSE 변화를 설명하는지 확인하는 것이다."
        )
        summary = [
            f"가장 큰 평균 accuracy drop은 `{worst_acc['model']}` / `{worst_acc['corruption']}`에서 `{float(worst_acc['accuracy_drop']):.6f}`이다.",
            f"가장 큰 평균 MSE increase는 `{worst_mse['model']}` / `{worst_mse['corruption']}`에서 `{float(worst_mse['mse_increase']):.6f}`이다.",
            f"MSE degradation과 가장 강하게 연결된 component deficit은 `{best_mse_predictor['predictor']}`이며 |r|=`{float(best_mse_predictor['abs_pearson_mse_increase']):.6f}`이다.",
            f"Accuracy degradation과 가장 강하게 연결된 component deficit은 `{best_acc_predictor['predictor']}`이며 |r|=`{float(best_acc_predictor['abs_pearson_accuracy_drop']):.6f}`이다.",
        ]
        interpretation = (
            "현재 stress task는 branch accuracy 변화가 작아서 accuracy coupling은 약하다. "
            "반면 MSE와 selected-K/frontier 관련 component는 objectification failure가 propagation "
            "손실로 이어지는 경로를 더 잘 보여준다. 따라서 P7은 부분적으로 개선됐지만, 더 강한 "
            "closed-loop/horizon corruption 실험이 필요하다."
        )
    else:
        title = "# PyBullet Objectification-Loss Coupling Audit"
        intro = (
            "This derived audit links PyBullet objectification quality metrics to downstream propagation "
            "degradation. The goal is to move beyond reporting an objectification score and identify which "
            "components explain branch-accuracy and MSE changes."
        )
        summary = [
            f"Largest mean accuracy drop: `{worst_acc['model']}` / `{worst_acc['corruption']}` at `{float(worst_acc['accuracy_drop']):.6f}`.",
            f"Largest mean MSE increase: `{worst_mse['model']}` / `{worst_mse['corruption']}` at `{float(worst_mse['mse_increase']):.6f}`.",
            f"Strongest component deficit for MSE degradation: `{best_mse_predictor['predictor']}` with |r|=`{float(best_mse_predictor['abs_pearson_mse_increase']):.6f}`.",
            f"Strongest component deficit for accuracy degradation: `{best_acc_predictor['predictor']}` with |r|=`{float(best_acc_predictor['abs_pearson_accuracy_drop']):.6f}`.",
        ]
        interpretation = (
            "The current stress task has small branch-accuracy movement, so accuracy coupling is weak. "
            "MSE and selected-K/frontier components give a clearer path from objectification failure to "
            "propagation degradation. P7 is therefore improved but not solved; a stronger closed-loop or "
            "multi-horizon corruption experiment is still needed."
        )

    table = [
        "| row_type | model | corruption | predictor | n | acc_drop | mse_increase | r_acc | r_mse |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['row_type']} | {row['model']} | {row['corruption']} | {row['predictor']} | "
            f"{row['sample_count']} | {_fmt(row['accuracy_drop'])} | {_fmt(row['mse_increase'])} | "
            f"{_fmt(row['pearson_accuracy_drop'])} | {_fmt(row['pearson_mse_increase'])} |"
        )

    return (
        f"{title}\n\n"
        f"{intro}\n\n"
        "Source CSV: `docs/experiments/pybullet_objectification_loss_coupling.csv`\n\n"
        "## Summary\n\n"
        + "\n".join(f"- {item}" for item in summary)
        + "\n\n## Interpretation\n\n"
        + interpretation
        + "\n\n## Rows\n\n"
        + "\n".join(table)
        + "\n"
    )


def _corruption_interpretation(corruption: str) -> str:
    if corruption == "clean":
        return "reference condition"
    if "relation" in corruption:
        return "relation corruption should primarily reduce frontier recall and selected K"
    if corruption == "low_confidence":
        return "confidence corruption should change contract score without changing topology"
    if corruption == "position_noise":
        return "position corruption should affect geometry-sensitive predictions"
    if corruption == "identity_swap":
        return "semantic identity corruption should require role/history checks"
    if corruption == "combined":
        return "combined corruption mixes topology, confidence, geometry, and identity failures"
    return "corruption condition"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "row_type",
        "model",
        "corruption",
        "predictor",
        "sample_count",
        "accuracy_drop",
        "mse_increase",
        "objectification_score_drop",
        "frontier_causal_recall_drop",
        "selected_k_drop",
        "pearson_accuracy_drop",
        "pearson_mse_increase",
        "abs_pearson_accuracy_drop",
        "abs_pearson_mse_increase",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _num(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values) -> float:
    clean = [float(value) for value in values if _finite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    if len(pairs) < 2:
        return float("nan")
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator > 0.0 else float("nan")


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _fmt(value: object) -> str:
    if value == "" or not _finite(value):
        return ""
    return f"{float(value):.6f}"


def _round(value: float) -> float | str:
    return round(value, 6) if _finite(value) else ""


if __name__ == "__main__":
    main()
