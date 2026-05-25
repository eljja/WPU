from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


FEATURES = [
    "causal_k",
    "interaction_density",
    "min_pair_distance",
    "mean_pair_distance",
    "target_x",
    "event_norm",
    "selector_confidence",
    "selected_fraction",
    "sparse_entropy",
    "sparse_margin",
    "sparse_confidence",
    "sparse_delta_norm",
    "sparse_uncertainty_mean",
    "regret_abs",
    "sparse_dense_disagreement",
]

PHYSICAL_FEATURES = [
    "causal_k",
    "interaction_density",
    "min_pair_distance",
    "mean_pair_distance",
    "target_x",
    "event_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe structured verifier policies for WPU route decisions.")
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/wpu_v2_staged_regret_context_samples.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_structured_verifier_probe.csv"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("docs/experiments/wpu_v2_structured_verifier_probe_summary.csv"),
    )
    parser.add_argument("--compute-cost", type=float, default=0.05)
    parser.add_argument("--expansion-costs", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--max-expansion-rates", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    rows = _read_rows(args.input)
    _validate_rows(rows)
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, object]] = []
    for test_seed in seeds:
        train_rows = [row for row in rows if int(row["seed"]) != test_seed]
        test_rows = [row for row in rows if int(row["seed"]) == test_seed]
        output.extend(_run_split(train_rows, test_rows, test_seed, args))
    _write_csv(args.out, output)
    _write_csv(args.summary_out, _summarize(output))
    print(f"wrote={args.out}")
    print(f"summary={args.summary_out}")


def _run_split(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    test_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    threshold = _choose_regret_threshold(train_rows, args.compute_cost)
    gate_rule = _choose_gate_rule(train_rows, threshold, args.compute_cost, args.quantiles)
    physical_gate_rule = _choose_gate_rule(
        train_rows,
        threshold,
        args.compute_cost,
        args.quantiles,
        features=PHYSICAL_FEATURES,
    )
    output = [
        _policy_row(test_seed, "calibrated_regret_route", test_rows, args.compute_cost, threshold, None, 0.0),
        _policy_row(test_seed, "structured_verifier_gate", test_rows, args.compute_cost, threshold, gate_rule, 0.0),
        _policy_row(
            test_seed,
            "physical_verifier_gate",
            test_rows,
            args.compute_cost,
            threshold,
            physical_gate_rule,
            0.0,
        ),
    ]
    for expansion_cost in args.expansion_costs:
        expansion_rule = _choose_expansion_rule(train_rows, threshold, args.compute_cost, expansion_cost, args.quantiles)
        physical_expansion_rule = _choose_expansion_rule(
            train_rows,
            threshold,
            args.compute_cost,
            expansion_cost,
            args.quantiles,
            features=PHYSICAL_FEATURES,
        )
        output.append(
            _policy_row(
                test_seed,
                "structured_expansion_upper_bound",
                test_rows,
                args.compute_cost,
                threshold,
                expansion_rule,
                expansion_cost,
                expansion=True,
            )
        )
        output.append(
            _policy_row(
                test_seed,
                "physical_expansion_upper_bound",
                test_rows,
                args.compute_cost,
                threshold,
                physical_expansion_rule,
                expansion_cost,
                expansion=True,
            )
        )
        for max_rate in args.max_expansion_rates:
            budgeted_expansion_rule = _choose_expansion_rule(
                train_rows,
                threshold,
                args.compute_cost,
                expansion_cost,
                args.quantiles,
                max_trigger_rate=max_rate,
            )
            physical_budgeted_expansion_rule = _choose_expansion_rule(
                train_rows,
                threshold,
                args.compute_cost,
                expansion_cost,
                args.quantiles,
                features=PHYSICAL_FEATURES,
                max_trigger_rate=max_rate,
            )
            output.append(
                _policy_row(
                    test_seed,
                    f"structured_expansion_upper_bound_budget_{max_rate:g}",
                    test_rows,
                    args.compute_cost,
                    threshold,
                    budgeted_expansion_rule,
                    expansion_cost,
                    expansion=True,
                )
            )
            output.append(
                _policy_row(
                    test_seed,
                    f"physical_expansion_upper_bound_budget_{max_rate:g}",
                    test_rows,
                    args.compute_cost,
                    threshold,
                    physical_budgeted_expansion_rule,
                    expansion_cost,
                    expansion=True,
                )
            )
    return output


def _choose_regret_threshold(rows: list[dict[str, str]], compute_cost: float) -> float:
    predictions = [float(row["predicted_regret"]) for row in rows]
    candidates = sorted({-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, *predictions})
    best_threshold = 0.0
    best_loss = float("inf")
    for threshold in candidates:
        loss = _mean_policy_loss(rows, compute_cost, threshold)
        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold
    return best_threshold


def _choose_gate_rule(
    rows: list[dict[str, str]],
    threshold: float,
    compute_cost: float,
    quantiles: list[float],
    *,
    features: list[str] = FEATURES,
) -> tuple[tuple[str, str, float], ...] | None:
    base_loss = _mean_policy_loss(rows, compute_cost, threshold)
    best_loss = base_loss
    best_rule: tuple[tuple[str, str, float], ...] | None = None
    conditions = _candidate_conditions(rows, quantiles, features)
    for condition in conditions:
        loss = _mean_policy_loss(rows, compute_cost, threshold, gate_rule=(condition,))
        if loss < best_loss:
            best_loss = loss
            best_rule = (condition,)
    single_best = list(best_rule or ())
    search_pairs = [(first, second) for first in single_best for second in conditions if second != first]
    for first, second in search_pairs:
        loss = _mean_policy_loss(rows, compute_cost, threshold, gate_rule=(first, second))
        if loss < best_loss:
            best_loss = loss
            best_rule = (first, second)
    return best_rule


def _choose_expansion_rule(
    rows: list[dict[str, str]],
    threshold: float,
    compute_cost: float,
    expansion_cost: float,
    quantiles: list[float],
    *,
    features: list[str] = FEATURES,
    max_trigger_rate: float | None = None,
) -> tuple[tuple[str, str, float], ...] | None:
    base_loss = _mean_policy_loss(rows, compute_cost, threshold)
    best_loss = base_loss
    best_rule: tuple[tuple[str, str, float], ...] | None = None
    conditions = _candidate_conditions(rows, quantiles, features)
    for condition in conditions:
        if max_trigger_rate is not None and _trigger_rate(rows, (condition,)) > max_trigger_rate:
            continue
        loss = _mean_policy_loss(
            rows,
            compute_cost,
            threshold,
            expansion_rule=(condition,),
            expansion_cost=expansion_cost,
        )
        if loss < best_loss:
            best_loss = loss
            best_rule = (condition,)
    single_best = list(best_rule or ())
    search_pairs = [(first, second) for first in single_best for second in conditions if second != first]
    for first, second in search_pairs:
        if max_trigger_rate is not None and _trigger_rate(rows, (first, second)) > max_trigger_rate:
            continue
        loss = _mean_policy_loss(
            rows,
            compute_cost,
            threshold,
            expansion_rule=(first, second),
            expansion_cost=expansion_cost,
        )
        if loss < best_loss:
            best_loss = loss
            best_rule = (first, second)
    return best_rule


def _candidate_conditions(
    rows: list[dict[str, str]],
    quantiles: list[float],
    features: list[str],
) -> list[tuple[str, str, float]]:
    conditions: list[tuple[str, str, float]] = []
    for feature in features:
        values = sorted(float(row[feature]) for row in rows)
        if not values:
            continue
        for quantile in quantiles:
            threshold = values[min(int(quantile * (len(values) - 1)), len(values) - 1)]
            conditions.append((feature, "le", threshold))
            conditions.append((feature, "ge", threshold))
    return conditions


def _mean_policy_loss(
    rows: list[dict[str, str]],
    compute_cost: float,
    threshold: float,
    gate_rule: tuple[tuple[str, str, float], ...] | None = None,
    expansion_rule: tuple[tuple[str, str, float], ...] | None = None,
    expansion_cost: float = 0.0,
) -> float:
    losses = []
    for row in rows:
        sparse_loss = float(row["sparse_loss"])
        dense_loss = float(row["dense_loss"]) + compute_cost
        base_dense = float(row["predicted_regret"]) < threshold
        if gate_rule is not None and not _matches(row, gate_rule):
            base_dense = False
        loss = dense_loss if base_dense else sparse_loss
        if expansion_rule is not None and _matches(row, expansion_rule):
            loss = min(sparse_loss, dense_loss) + expansion_cost
        losses.append(loss)
    return mean(losses)


def _policy_row(
    test_seed: int,
    policy: str,
    rows: list[dict[str, str]],
    compute_cost: float,
    threshold: float,
    rule: tuple[tuple[str, str, float], ...] | None,
    expansion_cost: float,
    *,
    expansion: bool = False,
) -> dict[str, object]:
    losses = []
    sparse_losses = []
    dense_losses = []
    oracle_losses = []
    route_dense = []
    trigger = []
    correct = []
    oracle_correct = []
    for row in rows:
        sparse_loss = float(row["sparse_loss"])
        dense_loss = float(row["dense_loss"]) + compute_cost
        sparse_correct = bool(int(row["sparse_correct"]))
        dense_correct = bool(int(row["dense_correct"]))
        use_dense = float(row["predicted_regret"]) < threshold
        if not expansion and rule is not None and not _matches(row, rule):
            use_dense = False
        use_trigger = rule is not None and _matches(row, rule)
        loss = dense_loss if use_dense else sparse_loss
        is_correct = dense_correct if use_dense else sparse_correct
        if expansion and use_trigger:
            loss = min(sparse_loss, dense_loss) + expansion_cost
            is_correct = sparse_correct or dense_correct
        losses.append(loss)
        sparse_losses.append(sparse_loss)
        dense_losses.append(dense_loss)
        oracle_losses.append(min(sparse_loss, dense_loss))
        route_dense.append(use_dense)
        trigger.append(use_trigger)
        correct.append(is_correct)
        oracle_correct.append(sparse_correct or dense_correct)
    return {
        "policy": policy,
        "test_seed": test_seed,
        "compute_cost": compute_cost,
        "expansion_cost": expansion_cost,
        "route_threshold": round(threshold, 6),
        "rule": _format_rule(rule),
        "samples": len(rows),
        "dense_rate": round(mean(float(value) for value in route_dense), 6),
        "trigger_rate": round(mean(float(value) for value in trigger), 6),
        "policy_loss": round(mean(losses), 6),
        "sparse_loss": round(mean(sparse_losses), 6),
        "dense_loss": round(mean(dense_losses), 6),
        "oracle_loss": round(mean(oracle_losses), 6),
        "policy_delta_vs_sparse": round(mean(loss - sparse for loss, sparse in zip(losses, sparse_losses, strict=True)), 6),
        "policy_excess_over_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "policy_accuracy": round(mean(float(value) for value in correct), 6),
        "oracle_accuracy": round(mean(float(value) for value in oracle_correct), 6),
    }


def _matches(row: dict[str, str], rule: tuple[tuple[str, str, float], ...]) -> bool:
    for feature, direction, threshold in rule:
        value = float(row[feature])
        if direction == "le" and value > threshold:
            return False
        if direction == "ge" and value < threshold:
            return False
    return True


def _trigger_rate(rows: list[dict[str, str]], rule: tuple[tuple[str, str, float], ...]) -> float:
    return mean(float(_matches(row, rule)) for row in rows)


def _format_rule(rule: tuple[tuple[str, str, float], ...] | None) -> str:
    if rule is None:
        return "none"
    return " AND ".join(f"{feature}:{direction}:{threshold:.6f}" for feature, direction, threshold in rule)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), float(row["expansion_cost"]))].append(row)
    output: list[dict[str, object]] = []
    metrics = [
        "dense_rate",
        "trigger_rate",
        "policy_loss",
        "policy_delta_vs_sparse",
        "policy_excess_over_oracle",
        "policy_accuracy",
    ]
    for (policy, expansion_cost), group_rows in sorted(grouped.items()):
        record: dict[str, object] = {"policy": policy, "expansion_cost": expansion_cost, "n": len(group_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            record[f"{metric}_mean"] = round(mean(values), 6)
            record[f"{metric}_std"] = round(pstdev(values), 6)
        output.append(record)
    return output


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("input contains no rows")
    required = {"seed", "predicted_regret", "sparse_loss", "dense_loss", "sparse_correct", "dense_correct", *FEATURES}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
