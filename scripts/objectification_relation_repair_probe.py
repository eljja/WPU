from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import wpu  # noqa: E402
from wpu.core.state import Event, Relation, WorldObject, WorldState  # noqa: E402
from wpu.engines.sparse_engine import SparsePropagationEngine  # noqa: E402

TYPE_VOCAB = ["background_object", "cup", "robot_hand", "table", "table_edge"]
RELATION_VOCAB = ["near", "touching"]
ROLE_KEYS = ["dynamic", "manipulator", "support", "boundary", "context"]
BRANCH_LABELS = ["stable", "caught", "falls"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether objectification relation repair restores sparse frontier recall.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--near-distance", type=float, default=0.25)
    parser.add_argument("--contact-distance", type=float, default=0.08)
    parser.add_argument("--background-objects", type=int, default=32)
    parser.add_argument("--near-distractors", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--learned-steps", type=int, default=80)
    parser.add_argument("--learned-threshold", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/objectification_relation_repair_probe.csv"),
    )
    args = parser.parse_args()

    rows = run_probe(
        samples=args.samples,
        seed=args.seed,
        near_distance=args.near_distance,
        contact_distance=args.contact_distance,
        background_objects=args.background_objects,
        near_distractors=args.near_distractors,
        train_samples=args.train_samples,
        learned_steps=args.learned_steps,
        learned_threshold=args.learned_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for summary in rows:
        print(
            "relation_repair_probe "
            f"scenario={summary['scenario']} "
            f"policy={summary['repair_policy']} "
            f"samples={summary['samples']} "
            f"before_recall={summary['mean_before_frontier_recall']} "
            f"after_recall={summary['mean_after_frontier_recall']} "
            f"repair_precision={summary['repair_precision']} "
            f"repair_recall={summary['repair_recall']} "
            f"downstream_accuracy={summary['downstream_branch_accuracy']} "
            f"downstream_loss={summary['downstream_branch_loss']}"
        )


def run_probe(
    *,
    samples: int,
    seed: int,
    near_distance: float,
    contact_distance: float,
    background_objects: int,
    near_distractors: int,
    train_samples: int,
    learned_steps: int,
    learned_threshold: float,
) -> list[dict[str, str]]:
    learned_scorer = _train_relation_scorer(
        samples=train_samples,
        seed=seed + 10_000,
        near_distance=near_distance,
        contact_distance=contact_distance,
        background_objects=background_objects,
        near_distractors=near_distractors,
        steps=learned_steps,
    )
    rows: list[dict[str, str]] = []
    scenarios = (
        ("in_distribution", background_objects, near_distractors, False, True),
        ("dense_distractors", max(background_objects, 128), max(near_distractors, 24), False, True),
        ("aliased_types_with_roles", background_objects, near_distractors, True, True),
        ("aliased_types_without_roles", background_objects, near_distractors, True, False),
    )
    for scenario, eval_background, eval_distractors, alias_types, include_roles in scenarios:
        for repair_policy, allowed_type_pairs, learned_filter in (
            ("no_repair", None, None),
            ("ungated", None, None),
            ("type_gated", _core_allowed_type_pairs(), None),
            (
                "learned_scorer",
                None,
                lambda state, relation, scorer=learned_scorer: _score_relation_candidate(state, relation, scorer)
                >= learned_threshold,
            ),
        ):
            rows.append(
                _run_policy(
                    samples=samples,
                    seed=seed,
                    near_distance=near_distance,
                    contact_distance=contact_distance,
                    background_objects=eval_background,
                    near_distractors=eval_distractors,
                    repair_policy=repair_policy,
                    allowed_type_pairs=allowed_type_pairs,
                    learned_filter=learned_filter,
                    scenario=scenario,
                    alias_types=alias_types,
                    include_roles=include_roles,
                )
            )
    return rows


def _run_policy(
    *,
    samples: int,
    seed: int,
    near_distance: float,
    contact_distance: float,
    background_objects: int,
    near_distractors: int,
    repair_policy: str,
    allowed_type_pairs: set[tuple[str, str]] | None,
    learned_filter,
    scenario: str,
    alias_types: bool,
    include_roles: bool,
) -> dict[str, str]:
    rng = random.Random(seed)
    sparse = SparsePropagationEngine(max_depth=2)
    totals = {
        "before_recall": 0.0,
        "after_recall": 0.0,
        "added": 0,
        "candidate_pairs": 0,
        "skipped_pairs": 0,
        "true_positive_edges": 0,
        "expected_edges": 0,
        "downstream_loss": 0.0,
        "downstream_correct": 0,
    }

    for index in range(samples):
        state = _state_with_missing_relations(
            rng,
            background_objects=background_objects,
            near_distractors=near_distractors,
            alias_types=alias_types,
            include_roles=include_roles,
        )
        event = Event("hand_touched_cup", "cup_001", {"force": 0.4}, confidence=0.9)
        expected_objects = {"cup_001", "hand_001", "edge_001", "table_001"}
        expected_edges = _expected_near_edges(
            state,
            near_distance=near_distance,
            contact_distance=contact_distance,
        )

        before = sparse.sparse_propagate(state, event)
        if repair_policy == "no_repair":
            repaired = state
            repair_report = wpu.ObjectificationRepairReport(
                added_relation_count=0,
                added_relation_types={},
                candidate_pair_count=0,
            )
        elif learned_filter is None:
            repaired, repair_report = wpu.repair_objectification_relations(
                state,
                near_distance=near_distance,
                contact_distance=contact_distance,
                allowed_type_pairs=allowed_type_pairs,
            )
        else:
            repaired, repair_report = _repair_with_learned_filter(
                state,
                near_distance=near_distance,
                contact_distance=contact_distance,
                learned_filter=learned_filter,
            )
        after = sparse.sparse_propagate(repaired, event)
        repaired_edges = {
            _edge_key(relation.src, relation.dst, relation.type)
            for relation in repaired.relations
            if relation.type in {"near", "touching"}
        }

        totals["before_recall"] += len(before.affected_objects & expected_objects) / len(expected_objects)
        totals["after_recall"] += len(after.affected_objects & expected_objects) / len(expected_objects)
        totals["added"] += repair_report.added_relation_count
        totals["candidate_pairs"] += repair_report.candidate_pair_count
        totals["skipped_pairs"] += repair_report.skipped_pair_count
        totals["true_positive_edges"] += len(repaired_edges & expected_edges)
        totals["expected_edges"] += len(expected_edges)
        target_label = _oracle_branch_label(state)
        predicted_label, loss = _predict_branch_from_frontier(repaired, after.affected_objects, target_label)
        totals["downstream_loss"] += loss
        totals["downstream_correct"] += int(predicted_label == target_label)

    repair_precision = totals["true_positive_edges"] / max(totals["added"], 1)
    repair_recall = totals["true_positive_edges"] / max(totals["expected_edges"], 1)
    return {
        "scenario": scenario,
        "repair_policy": repair_policy,
        "samples": str(samples),
        "seed": str(seed),
        "near_distance": f"{near_distance:.6f}",
        "contact_distance": f"{contact_distance:.6f}",
        "background_objects": str(background_objects),
        "near_distractors": str(near_distractors),
        "mean_before_frontier_recall": f"{totals['before_recall'] / samples:.6f}",
        "mean_after_frontier_recall": f"{totals['after_recall'] / samples:.6f}",
        "mean_added_relations": f"{totals['added'] / samples:.6f}",
        "mean_candidate_pairs": f"{totals['candidate_pairs'] / samples:.6f}",
        "mean_skipped_pairs": f"{totals['skipped_pairs'] / samples:.6f}",
        "repair_precision": f"{repair_precision:.6f}",
        "repair_recall": f"{repair_recall:.6f}",
        "downstream_branch_accuracy": f"{totals['downstream_correct'] / samples:.6f}",
        "downstream_branch_loss": f"{totals['downstream_loss'] / samples:.6f}",
    }


def _state_with_missing_relations(
    rng: random.Random,
    *,
    background_objects: int,
    near_distractors: int,
    alias_types: bool = False,
    include_roles: bool = True,
) -> WorldState:
    hand_x = rng.uniform(0.08, 0.18)
    edge_x = rng.uniform(0.16, 0.22)
    state = WorldState(metadata={"scenario": "objectification_relation_repair_probe"})
    cup_type = "vessel" if alias_types else "cup"
    hand_type = "end_effector" if alias_types else "robot_hand"
    edge_type = "support_boundary" if alias_types else "table_edge"
    table_type = "support_plane" if alias_types else "table"
    state.add_object(
        WorldObject(
            "cup_001",
            cup_type,
            _object_attributes([0.0, 0.0, 0.82], include_roles=include_roles, dynamic=1.0),
            confidence=0.95,
        )
    )
    state.add_object(
        WorldObject(
            "hand_001",
            hand_type,
            _object_attributes([hand_x, 0.0, 0.82], include_roles=include_roles, manipulator=1.0),
            confidence=0.92,
        )
    )
    state.add_object(
        WorldObject(
            "edge_001",
            edge_type,
            _object_attributes([edge_x, 0.0, 0.82], include_roles=include_roles, boundary=1.0),
            confidence=0.94,
        )
    )
    state.add_object(
        WorldObject(
            "table_001",
            table_type,
            _object_attributes([0.02, 0.0, 0.75], include_roles=include_roles, support=1.0),
            confidence=0.98,
        )
    )
    for index in range(near_distractors):
        state.add_object(
            WorldObject(
                f"near_context_{index:04d}",
                "scene_prop" if alias_types else "background_object",
                _object_attributes(
                    [rng.uniform(0.02, 0.22), rng.uniform(-0.04, 0.04), 0.82],
                    include_roles=include_roles,
                    context=1.0,
                ),
                confidence=0.75,
            )
        )
    for index in range(background_objects):
        state.add_object(
            WorldObject(
                f"context_{index:04d}",
                "scene_prop" if alias_types else "background_object",
                _object_attributes([10.0 + index, 10.0, 0.0], include_roles=include_roles, context=1.0),
                confidence=0.75,
            )
        )
    return state


def _object_attributes(
    position: list[float],
    *,
    include_roles: bool,
    dynamic: float = 0.0,
    manipulator: float = 0.0,
    support: float = 0.0,
    boundary: float = 0.0,
    context: float = 0.0,
) -> dict[str, object]:
    attributes: dict[str, object] = {"position": position}
    if include_roles:
        attributes["objectification_roles"] = {
            "dynamic": dynamic,
            "manipulator": manipulator,
            "support": support,
            "boundary": boundary,
            "context": context,
        }
    return attributes


@dataclass(slots=True)
class LinearRelationScorer:
    weights: torch.Tensor
    bias: torch.Tensor


def _train_relation_scorer(
    *,
    samples: int,
    seed: int,
    near_distance: float,
    contact_distance: float,
    background_objects: int,
    near_distractors: int,
    steps: int,
) -> LinearRelationScorer:
    rng = random.Random(seed)
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for _ in range(samples):
        state = _state_with_missing_relations(
            rng,
            background_objects=background_objects,
            near_distractors=near_distractors,
        )
        expected = _expected_near_edges(state, near_distance=near_distance, contact_distance=contact_distance)
        for relation in _candidate_relations(state, near_distance=near_distance, contact_distance=contact_distance):
            features.append(_relation_candidate_features(state, relation))
            labels.append(1.0 if _edge_key(relation.src, relation.dst, relation.type) in expected else 0.0)

    x = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    weights = torch.zeros((x.size(1), 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.08)
    positive_count = max(float(y.sum().item()), 1.0)
    negative_count = max(float(y.numel() - y.sum().item()), 1.0)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = x @ weights + bias
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return LinearRelationScorer(weights=weights.detach().squeeze(1), bias=bias.detach())


def _repair_with_learned_filter(
    state: WorldState,
    *,
    near_distance: float,
    contact_distance: float,
    learned_filter,
) -> tuple[WorldState, wpu.ObjectificationRepairReport]:
    delta = wpu.DeltaState(time=state.time, metadata={"objectification_repair": "learned_scorer"})
    added_types: dict[str, int] = {}
    candidate_pair_count = 0
    for relation in _candidate_relations(state, near_distance=near_distance, contact_distance=contact_distance):
        candidate_pair_count += 1
        if not learned_filter(state, relation):
            continue
        delta.relation_updates.append(relation)
        added_types[relation.type] = added_types.get(relation.type, 0) + 1
    report = wpu.ObjectificationRepairReport(
        added_relation_count=len(delta.relation_updates),
        added_relation_types=added_types,
        candidate_pair_count=candidate_pair_count,
    )
    return state.apply_delta(delta), report


def _candidate_relations(state: WorldState, *, near_distance: float, contact_distance: float) -> list[Relation]:
    relations: list[Relation] = []
    object_ids = list(state.objects)
    for left_index, left_id in enumerate(object_ids):
        left = _position_or_none(state, left_id)
        if left is None:
            continue
        for right_id in object_ids[left_index + 1 :]:
            right = _position_or_none(state, right_id)
            if right is None:
                continue
            distance = _distance(left, right)
            if distance <= near_distance:
                relations.append(
                    Relation(
                        left_id,
                        right_id,
                        "near",
                        strength=max(0.05, 1.0 - distance / near_distance),
                        confidence=0.55,
                    )
                )
            if distance <= contact_distance:
                relations.append(
                    Relation(
                        left_id,
                        right_id,
                        "touching",
                        strength=max(0.05, 1.0 - distance / max(contact_distance, 1e-6)),
                        confidence=0.55,
                    )
                )
    return relations


def _core_allowed_type_pairs() -> set[tuple[str, str]]:
    return {
        ("cup", "robot_hand"),
        ("cup", "table"),
        ("cup", "table_edge"),
        ("robot_hand", "table"),
        ("robot_hand", "table_edge"),
        ("table", "table_edge"),
    }


def _expected_near_edges(state: WorldState, *, near_distance: float, contact_distance: float = 0.08) -> set[tuple[str, str, str]]:
    core_ids = ["cup_001", "hand_001", "edge_001", "table_001"]
    expected: set[tuple[str, str, str]] = set()
    for left_index, left_id in enumerate(core_ids):
        left = _position(state, left_id)
        for right_id in core_ids[left_index + 1 :]:
            right = _position(state, right_id)
            distance = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2) ** 0.5
            if distance <= near_distance:
                expected.add(_edge_key(left_id, right_id, "near"))
            if distance <= contact_distance:
                expected.add(_edge_key(left_id, right_id, "touching"))
    return expected


def _oracle_branch_label(state: WorldState) -> str:
    hand_distance = _distance(_position(state, "cup_001"), _position(state, "hand_001"))
    edge_distance = _distance(_position(state, "cup_001"), _position(state, "edge_001"))
    if hand_distance <= 0.12:
        return "caught"
    if edge_distance <= 0.19:
        return "falls"
    return "stable"


def _predict_branch_from_frontier(state: WorldState, visible_ids: set[str], target_label: str) -> tuple[str, float]:
    logits = _frontier_branch_logits(state, visible_ids)
    prediction = BRANCH_LABELS[max(range(len(logits)), key=lambda index: logits[index])]
    target_index = BRANCH_LABELS.index(target_label)
    return prediction, _cross_entropy(logits, target_index)


def _frontier_branch_logits(state: WorldState, visible_ids: set[str]) -> list[float]:
    cup_position = _position(state, "cup_001")
    logits = [0.6, -0.8, -0.8]
    if "hand_001" in visible_ids:
        hand_distance = _distance(cup_position, _position(state, "hand_001"))
        logits[1] = 2.8 * max(0.0, 1.0 - hand_distance / 0.18)
    if "edge_001" in visible_ids:
        edge_distance = _distance(cup_position, _position(state, "edge_001"))
        logits[2] = 2.8 * max(0.0, 1.0 - edge_distance / 0.22)
    if "table_001" in visible_ids:
        logits[0] += 0.4

    context_count = sum(1 for object_id in visible_ids if _is_context_object(state.objects[object_id]))
    if context_count:
        confusion = min(1.6, 0.08 * context_count)
        logits[0] += 0.5 * confusion
        logits[1] -= confusion
        logits[2] -= confusion
    return logits


def _cross_entropy(logits: list[float], target_index: int) -> float:
    max_logit = max(logits)
    exp_sum = sum(math.exp(logit - max_logit) for logit in logits)
    log_prob = logits[target_index] - max_logit - math.log(exp_sum)
    return -log_prob


def _score_relation_candidate(state: WorldState, relation: Relation, scorer: LinearRelationScorer) -> float:
    feature = _relation_candidate_features(state, relation)
    return float(torch.sigmoid(feature @ scorer.weights + scorer.bias).item())


def _relation_candidate_features(state: WorldState, relation: Relation) -> torch.Tensor:
    left = _position(state, relation.src)
    right = _position(state, relation.dst)
    distance = _distance(left, right)
    left_type = state.objects[relation.src].type
    right_type = state.objects[relation.dst].type
    type_features = _type_features(left_type, right_type)
    relation_features = [1.0 if relation.type == value else 0.0 for value in RELATION_VOCAB]
    return torch.tensor(
        [
            distance,
            relation.strength,
            1.0 if "background_object" in {left_type, right_type} else 0.0,
            *relation_features,
            *type_features,
            *_role_pair_features(state, relation.src, relation.dst),
        ],
        dtype=torch.float32,
    )


def _type_features(left_type: str, right_type: str) -> list[float]:
    counts = {value: 0.0 for value in TYPE_VOCAB}
    if left_type in counts:
        counts[left_type] += 1.0
    if right_type in counts:
        counts[right_type] += 1.0
    return [counts[value] for value in TYPE_VOCAB]


def _role_pair_features(state: WorldState, left_id: str, right_id: str) -> list[float]:
    left_roles = _roles(state.objects[left_id])
    right_roles = _roles(state.objects[right_id])
    summed = [left_roles[key] + right_roles[key] for key in ROLE_KEYS]
    products = [
        left_roles["dynamic"] * right_roles["manipulator"] + left_roles["manipulator"] * right_roles["dynamic"],
        left_roles["dynamic"] * right_roles["support"] + left_roles["support"] * right_roles["dynamic"],
        left_roles["dynamic"] * right_roles["boundary"] + left_roles["boundary"] * right_roles["dynamic"],
        left_roles["context"] + right_roles["context"],
    ]
    return [*summed, *products]


def _roles(obj: WorldObject) -> dict[str, float]:
    value = obj.attributes.get("objectification_roles")
    if not isinstance(value, dict):
        return {key: 0.0 for key in ROLE_KEYS}
    return {key: float(value.get(key, 0.0)) for key in ROLE_KEYS}


def _is_context_object(obj: WorldObject) -> bool:
    return _roles(obj)["context"] > 0.0 or obj.type in {"background_object", "scene_prop"}


def _position(state: WorldState, object_id: str) -> tuple[float, float, float]:
    value = state.objects[object_id].attributes["position"]
    return float(value[0]), float(value[1]), float(value[2])


def _position_or_none(state: WorldState, object_id: str) -> tuple[float, float, float] | None:
    value = state.objects[object_id].attributes.get("position")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return float(value[0]), float(value[1]), float(value[2])


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2) ** 0.5


def _edge_key(src: str, dst: str, relation_type: str) -> tuple[str, str, str]:
    if dst < src:
        return dst, src, relation_type
    return src, dst, relation_type


if __name__ == "__main__":
    main()
