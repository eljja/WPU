from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import wpu
from wpu.data.pybullet_cup import PyBulletCupDataset, PyBulletCupSample


MECHANISMS = {
    "nominal": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.45,
    },
    "high_force": {
        "force_range": (1.1, 2.4),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.45,
    },
    "edge_shift": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.84, 1.02),
        "catch_probability": 0.45,
    },
    "catch_heavy": {
        "force_range": (0.2, 1.8),
        "cup_x_range": (0.62, 0.96),
        "catch_probability": 0.85,
    },
}

POLICIES = (
    "base_law",
    "gain_calibrated_law",
    "form_revised_law",
    "oracle_form_law",
)


@dataclass(slots=True)
class LocalLaw:
    form: str
    weights: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and revise simple local laws over PyBullet-derived WorldState samples."
    )
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument("--background-objects", type=int, default=16)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--mechanisms", nargs="+", default=["nominal", "high_force", "edge_shift", "catch_heavy"])
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/pybullet_local_law_revision.csv"))
    args = parser.parse_args()

    unknown = [mechanism for mechanism in args.mechanisms if mechanism not in MECHANISMS]
    if unknown:
        raise ValueError(f"unknown mechanisms: {unknown}")

    rows: list[dict[str, str]] = []
    for seed in args.seeds:
        train_samples = _samples(
            mechanism="nominal",
            samples=args.train_samples,
            seed=seed,
            background_objects=args.background_objects,
            sim_steps=args.sim_steps,
        )
        base_law = _fit_law(train_samples, form="base")
        for mechanism in args.mechanisms:
            calibration = _samples(
                mechanism=mechanism,
                samples=args.calibration_samples,
                seed=seed + 10_000,
                background_objects=args.background_objects,
                sim_steps=args.sim_steps,
            )
            eval_samples = _samples(
                mechanism=mechanism,
                samples=args.eval_samples,
                seed=seed + 20_000,
                background_objects=args.background_objects,
                sim_steps=args.sim_steps,
            )
            policy_rows = _evaluate_mechanism(
                mechanism=mechanism,
                base_law=base_law,
                calibration=calibration,
                eval_samples=eval_samples,
            )
            for row in policy_rows:
                rows.append({"row_type": "seed", "seed": str(seed), **row, "seed_count": "1"})
    if len(args.seeds) > 1:
        rows.extend(_summarize(rows, seeds=args.seeds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if len(args.seeds) > 1 and row["row_type"] != "summary":
            continue
        print(
            "pybullet_local_law_revision "
            f"row_type={row['row_type']} "
            f"mechanism={row['mechanism']} "
            f"policy={row['policy']} "
            f"selected_form={row['selected_form']} "
            f"decision={row['revision_decision']} "
            f"delta_mse={row['delta_mse']} "
            f"relative_improvement={row['relative_improvement']}"
        )


def _samples(
    *,
    mechanism: str,
    samples: int,
    seed: int,
    background_objects: int,
    sim_steps: int,
) -> list[PyBulletCupSample]:
    config = MECHANISMS[mechanism]
    dataset = PyBulletCupDataset(
        size=samples,
        seed=seed,
        background_objects=background_objects,
        steps=sim_steps,
        balanced_labels=False,
        force_range=config["force_range"],
        cup_x_range=config["cup_x_range"],
        catch_probability=float(config["catch_probability"]),
    )
    return [dataset[index] for index in range(samples)]


def _evaluate_mechanism(
    *,
    mechanism: str,
    base_law: LocalLaw,
    calibration: list[PyBulletCupSample],
    eval_samples: list[PyBulletCupSample],
) -> list[dict[str, str]]:
    gain_law = _fit_gain(base_law, calibration)
    revised_law = _select_best_form(calibration)
    oracle_law = _select_best_form(eval_samples)
    policy_laws = {
        "base_law": base_law,
        "gain_calibrated_law": gain_law,
        "form_revised_law": revised_law,
        "oracle_form_law": oracle_law,
    }
    base_error = _mse(base_law, eval_samples)
    oracle_error = _mse(oracle_law, eval_samples)
    rows: list[dict[str, str]] = []
    for policy in POLICIES:
        law = policy_laws[policy]
        delta_mse = _mse(law, eval_samples)
        sign_accuracy = _sign_accuracy(law, eval_samples)
        calibration_mse = _mse(law, calibration)
        hypothesis = wpu.LocalLawHypothesis(
            name=f"pybullet:{mechanism}:{policy}",
            relation_type="simulated_impulse_response",
            expression=_expression(law.form),
            input_fields=("force", "edge_distance", "hand_distance", "catch_action", "relation_strength"),
            parameters={"weight_norm": float(law.weights.norm().item())},
            evidence={"delta_mse": delta_mse, "calibration_mse": calibration_mse},
            status="base" if policy == "base_law" else "revised",
        )
        report = wpu.evaluate_law_revision(
            base_error=base_error,
            revised_error=delta_mse,
            selected_hypothesis=hypothesis,
            calibration_samples=len(calibration),
            oracle_relation_error=oracle_error,
        )
        rows.append(
            {
                "mechanism": mechanism,
                "policy": policy,
                "selected_form": law.form,
                "samples": str(len(eval_samples)),
                "delta_mse": f"{delta_mse:.6f}",
                "sign_accuracy": f"{sign_accuracy:.6f}",
                "calibration_mse": f"{calibration_mse:.6f}",
                "revision_decision": "baseline" if policy == "base_law" else report.decision,
                "relative_improvement": f"{report.relative_improvement:.6f}",
                "law_residual_gap": f"{(report.law_residual_gap or 0.0):.6f}",
                "revision_gap": f"{max(0.0, delta_mse - oracle_error):.6f}",
            }
        )
    return rows


def _fit_gain(base_law: LocalLaw, samples: list[PyBulletCupSample]) -> LocalLaw:
    numerator = 0.0
    denominator = 0.0
    for sample in samples:
        prediction = _predict(base_law, sample)
        target = _target(sample)
        numerator += prediction * target
        denominator += prediction * prediction
    gain = numerator / denominator if denominator > 1e-9 else 1.0
    return LocalLaw(form="gain_scaled_base", weights=torch.cat([torch.tensor([gain], dtype=torch.float32), base_law.weights]))


def _select_best_form(samples: list[PyBulletCupSample]) -> LocalLaw:
    forms = ("base", "edge_form", "catch_form", "quadratic_form")
    candidates = [_fit_law(samples, form=form) for form in forms]
    return min(candidates, key=lambda law: _mse(law, samples))


def _fit_law(samples: list[PyBulletCupSample], *, form: str) -> LocalLaw:
    x = torch.stack([_features(sample, form=form) for sample in samples])
    y = torch.tensor([_target(sample) for sample in samples], dtype=torch.float32).unsqueeze(1)
    ridge = 1e-3 * torch.eye(x.size(1), dtype=torch.float32)
    weights = torch.linalg.solve(x.T @ x + ridge, x.T @ y).squeeze(1)
    return LocalLaw(form=form, weights=weights.detach())


def _mse(law: LocalLaw, samples: list[PyBulletCupSample]) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        error = _predict(law, sample) - _target(sample)
        total += error * error
    return total / len(samples)


def _sign_accuracy(law: LocalLaw, samples: list[PyBulletCupSample]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        correct += int((_predict(law, sample) >= 0.0) == (_target(sample) >= 0.0))
    return correct / len(samples)


def _predict(law: LocalLaw, sample: PyBulletCupSample) -> float:
    if law.form == "gain_scaled_base":
        gain = float(law.weights[0].item())
        base = LocalLaw(form="base", weights=law.weights[1:])
        return gain * _predict(base, sample)
    return float(_features(sample, form=law.form) @ law.weights)


def _features(sample: PyBulletCupSample, *, form: str) -> torch.Tensor:
    force = float(sample.event.delta.get("force", 0.0))
    catch_action = float(sample.event.delta.get("catch_action", 0.0))
    cup = sample.state.objects["cup_001"]
    edge_distance = float(cup.attributes.get("edge_distance", 0.0))
    hand_distance = float(sample.state.objects.get("hand_001", cup).attributes.get("hand_distance", 0.0))
    edge_strength = _relation_strength(sample, "cup_001", "edge_001")
    hand_strength = _relation_strength(sample, "hand_001", "cup_001")
    if form == "base":
        values = [force, edge_distance, hand_distance, catch_action, edge_strength, hand_strength, 1.0]
    elif form == "edge_form":
        values = [force, edge_distance, force * edge_distance, edge_strength, force * edge_strength, 1.0]
    elif form == "catch_form":
        values = [force, catch_action, force * catch_action, hand_distance, hand_strength, 1.0]
    elif form == "quadratic_form":
        values = [force, force * force, edge_distance, edge_distance * edge_distance, catch_action, hand_distance, 1.0]
    else:
        raise ValueError(f"unknown law form: {form}")
    return torch.tensor(values, dtype=torch.float32)


def _target(sample: PyBulletCupSample) -> float:
    object_ids = list(sample.state.objects)
    cup_index = object_ids.index("cup_001")
    return float(sample.target_object_delta[cup_index, 1].item())


def _relation_strength(sample: PyBulletCupSample, left: str, right: str) -> float:
    for relation in sample.state.relations:
        if {relation.src, relation.dst} == {left, right}:
            return float(relation.strength)
    return 0.0


def _expression(form: str) -> str:
    return {
        "base": "linear(force, edge_distance, hand_distance, catch_action, relation_strengths)",
        "gain_scaled_base": "alpha * base",
        "edge_form": "linear(force, edge_distance, force*edge_distance, edge_strength)",
        "catch_form": "linear(force, catch_action, force*catch_action, hand_distance)",
        "quadratic_form": "linear(force, force^2, edge_distance, edge_distance^2, catch_action)",
    }[form]


def _summarize(rows: list[dict[str, str]], *, seeds: list[int]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["mechanism"], row["policy"])].append(row)
    summaries: list[dict[str, str]] = []
    for (mechanism, policy), group in sorted(groups.items()):
        first = group[0]
        summary = {
            "row_type": "summary",
            "seed": "all",
            "mechanism": mechanism,
            "policy": policy,
            "selected_form": _mode([row["selected_form"] for row in group]),
            "samples": str(sum(int(row["samples"]) for row in group)),
        }
        for column in (
            "delta_mse",
            "sign_accuracy",
            "calibration_mse",
            "relative_improvement",
            "law_residual_gap",
            "revision_gap",
        ):
            values = [float(row[column]) for row in group]
            summary[column] = f"{sum(values) / len(values):.6f}"
        summary["revision_decision"] = _mode([row["revision_decision"] for row in group])
        summary["seed_count"] = str(len(seeds))
        summaries.append(summary)
    return summaries


def _mode(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


if __name__ == "__main__":
    main()
