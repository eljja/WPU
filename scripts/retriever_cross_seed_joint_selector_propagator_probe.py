from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.learned_retriever_probe import (  # noqa: E402
    FEATURE_DIM as OBJECT_FEATURE_DIM,
    _candidate_features,
    _candidate_ids,
    _selected_ids,
    _selected_pair_density,
    _train_model as _train_interaction_retriever,
)
from scripts.retriever_generated_candidate_probe import (  # noqa: E402
    BASE_MODES,
    GeneratedSetReranker,
    _generated_candidates,
    _selected_object_tensor,
    _write_csv,
)
from scripts.retriever_regret_oracle_probe import _evaluate_selected  # noqa: E402
from scripts.retriever_cross_seed_set_evaluator_probe import _selected_geometry_features  # noqa: E402
from scripts.staged_regret_hybrid import _class_weights, _move_batch  # noqa: E402
from wpu.data.working_set_physics import (  # noqa: E402
    WorkingSetPhysicsDataset,
    collate_selected_working_set_samples,
)
from wpu.models.factory import create_model  # noqa: E402


CONTEXT_EXTRA_DIM = 19
VERIFICATION_CONTEXT_DIM = 6


class VerificationAwareGeneratedSetReranker(nn.Module):
    """Candidate scorer with an explicit no-harm verification head."""

    def __init__(self, object_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(object_dim),
            nn.Linear(object_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + context_dim),
            nn.Linear(hidden_dim * 2 + context_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(self, objects: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, budget, object_dim = objects.shape
        encoded = self.object_encoder(objects.view(batch_size * candidate_count * budget, object_dim))
        encoded = encoded.view(batch_size, candidate_count, budget, -1)
        float_mask = mask.unsqueeze(-1).float()
        pooled_mean = (encoded * float_mask).sum(dim=2) / float_mask.sum(dim=2).clamp_min(1.0)
        pooled_max = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=2)
        hidden = self.shared(torch.cat([pooled_mean, pooled_max, context], dim=-1))
        return self.score_head(hidden).squeeze(-1), self.harm_head(hidden).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a minimal joint P1 selector-propagator objective. Candidate scores "
            "and WPU branch losses are optimized in the same graph, rather than "
            "training a post-hoc selector over a fixed propagation model."
        )
    )
    parser.add_argument("--model-name", default="wpu-cws-indexed-sparse")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2048])
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17, 19, 23])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--generated-candidates", type=int, default=4)
    parser.add_argument("--structured-candidates", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--selector-hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--joint-steps", type=int, default=120)
    parser.add_argument("--retriever-steps", type=int, default=400)
    parser.add_argument("--retriever-hidden-dim", type=int, default=64)
    parser.add_argument("--retriever-lr", type=float, default=3e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--selector-lr", type=float, default=2e-3)
    parser.add_argument("--selector-temperature", type=float, default=0.75)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--no-harm-weight", type=float, default=0.4)
    parser.add_argument("--pairwise-no-harm-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-no-harm-margin", type=float, default=0.25)
    parser.add_argument("--score-regression-weight", type=float, default=0.0)
    parser.add_argument("--target-delta-weight", type=float, default=0.05)
    parser.add_argument(
        "--verification-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append label-free propagation signatures to candidate selector context.",
    )
    parser.add_argument(
        "--verification-context-detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach propagation signatures before selector scoring to keep verification observational.",
    )
    parser.add_argument(
        "--verification-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train an explicit candidate harmfulness head and penalize unsafe deployment scores.",
    )
    parser.add_argument("--verification-head-weight", type=float, default=0.4)
    parser.add_argument("--verification-score-penalty", type=float, default=1.0)
    parser.add_argument(
        "--use-geometry-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append candidate geometry/force descriptors to selector context.",
    )
    parser.add_argument("--safe-margin", type=float, default=0.005)
    parser.add_argument("--selection-harmful-limit", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=90)
    parser.add_argument("--validation-samples", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interaction-mode", choices=["standard", "pairwise"], default="pairwise")
    parser.add_argument("--index-depth", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/wpu_v2_joint_selector_propagator.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for n_value in args.n_values:
        for k_value in args.k_values:
            causal_obstacles = max(0, k_value - 4)
            background_objects = max(0, n_value - 4 - causal_obstacles)
            rows.extend(_run_group(background_objects, causal_obstacles, args))
            _write_csv(args.out, rows)
    _write_csv(args.out, rows)
    print(f"wrote={args.out}", flush=True)


def _run_group(background_objects: int, causal_obstacles: int, args: argparse.Namespace) -> list[dict[str, object]]:
    total_n = background_objects + 4 + causal_obstacles
    causal_k = 4 + causal_obstacles
    candidate_names = [
        *BASE_MODES,
        *[f"generated_{index}" for index in range(args.generated_candidates)],
        *[f"structured_{index}" for index in range(args.structured_candidates)],
    ]
    contexts = {
        seed: _make_context(background_objects, causal_obstacles, seed, args, candidate_names)
        for seed in args.seeds
    }
    rows: list[dict[str, object]] = []
    for heldout_seed, heldout in contexts.items():
        print(f"joint-selector-propagator heldout={heldout_seed} N={total_n} K={causal_k}", flush=True)
        train_samples = [
            sample
            for seed, context in contexts.items()
            if seed != heldout_seed
            for sample in context["validation_samples"]
        ]
        retriever = _train_interaction_retriever(
            train_samples,
            args.budget,
            args.retriever_steps,
            args.retriever_hidden_dim,
            args.retriever_lr,
        )
        model, selector = _train_joint_model(train_samples, retriever, candidate_names, args)
        validation_examples = _collect_examples(
            model,
            selector,
            heldout["validation_samples"],
            retriever,
            candidate_names,
            args,
            total_n,
            causal_k,
            seed_offset=heldout_seed + 5_000,
        )
        test_examples = _collect_examples(
            model,
            selector,
            heldout["test_samples"],
            retriever,
            candidate_names,
            args,
            total_n,
            causal_k,
            seed_offset=heldout_seed + 10_000,
        )
        condition_rows = _summarize(test_examples, validation_examples, selector, candidate_names, args)
        for row in condition_rows:
            row.update(
                {
                    "seed": heldout_seed,
                    "heldout_seed": heldout_seed,
                    "train_seed_count": len(args.seeds) - 1,
                    "total_objects_n": total_n,
                    "causal_k": causal_k,
                    "budget": args.budget,
                    "generated_candidates": args.generated_candidates,
                    "structured_candidates": args.structured_candidates,
                    "interaction_mode": args.interaction_mode,
                    "joint_steps": args.joint_steps,
                    "retriever_steps": args.retriever_steps,
                    "validation_samples_per_seed": args.validation_samples,
                    "test_samples": args.samples,
                    "use_geometry_context": int(bool(args.use_geometry_context)),
                    "pairwise_no_harm_weight": args.pairwise_no_harm_weight,
                    "pairwise_no_harm_margin": args.pairwise_no_harm_margin,
                    "score_regression_weight": args.score_regression_weight,
                    "verification_context": int(bool(args.verification_context)),
                    "verification_context_detach": int(bool(args.verification_context_detach)),
                    "verification_head": int(bool(args.verification_head)),
                    "verification_head_weight": args.verification_head_weight,
                    "verification_score_penalty": args.verification_score_penalty,
                }
            )
        rows.extend(condition_rows)
    return rows


def _make_context(
    background_objects: int,
    causal_obstacles: int,
    seed: int,
    args: argparse.Namespace,
    candidate_names: list[str],
) -> dict[str, object]:
    del candidate_names
    validation_dataset = WorkingSetPhysicsDataset(
        size=args.validation_samples,
        seed=seed + 5_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    test_dataset = WorkingSetPhysicsDataset(
        size=args.samples,
        seed=seed + 10_000,
        background_objects=background_objects,
        causal_obstacles=causal_obstacles,
        balanced_labels=args.balanced_labels,
        interaction_mode=args.interaction_mode,
    )
    return {
        "validation_samples": [validation_dataset[index] for index in range(len(validation_dataset))],
        "test_samples": [test_dataset[index] for index in range(len(test_dataset))],
    }


def _train_joint_model(
    samples,
    retriever: torch.nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, GeneratedSetReranker]:
    device = torch.device(args.device)
    torch.manual_seed(2027 + len(samples) + len(candidate_names))
    model = create_model(
        args.model_name,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        working_set_size=args.budget,
    ).to(device)
    selector_cls = VerificationAwareGeneratedSetReranker if args.verification_head else GeneratedSetReranker
    selector = selector_cls(
        OBJECT_FEATURE_DIM,
        len(candidate_names) + _context_extra_dim(args),
        args.selector_hidden_dim,
    ).to(device)
    class_weights = _class_weights_from_samples(samples).to(device) if args.class_weights else None
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": selector.parameters(), "lr": args.selector_lr},
        ]
    )
    model.train()
    selector.train()
    for step in range(args.joint_steps):
        batch_samples = [samples[(step * args.batch_size + offset) % len(samples)] for offset in range(args.batch_size)]
        selected = _candidate_sets_for_samples(batch_samples, retriever, candidate_names, args, seed_offset=step * 97)
        objects, masks, context = _candidate_selector_tensors(batch_samples, selected, candidate_names, args, 0, 0)
        objects = objects.to(device)
        masks = masks.to(device)
        context = context.to(device)
        candidate_losses = []
        candidate_delta_losses = []
        candidate_signatures = []
        learned_index = candidate_names.index("learned")
        for name in candidate_names:
            batch, target_delta, labels, _ = collate_selected_working_set_samples(batch_samples, selected[name])
            batch = _move_batch(batch, device)
            target_delta = target_delta.to(device)
            labels = labels.to(device)
            prediction = model(batch, num_branches=3, force_route="sparse")
            ce = F.cross_entropy(prediction.branch_logits, labels, weight=class_weights, reduction="none")
            delta_loss = F.mse_loss(prediction.object_delta, target_delta, reduction="none").flatten(1).mean(dim=1)
            candidate_losses.append(ce)
            candidate_delta_losses.append(delta_loss)
            candidate_signatures.append(_prediction_signature(prediction))
        losses = torch.stack(candidate_losses, dim=1)
        delta_losses = torch.stack(candidate_delta_losses, dim=1)
        if args.verification_context:
            signatures = torch.stack(candidate_signatures, dim=1)
            if args.verification_context_detach:
                signatures = signatures.detach()
            context = torch.cat([context, signatures], dim=-1)
        raw_scores, harm_logits = _selector_outputs(selector, objects, masks, context)
        detached_losses = losses.detach()
        learned_losses = detached_losses[:, learned_index].unsqueeze(1)
        scores = _deployment_scores(raw_scores, harm_logits, args)
        weights = F.softmax(scores / max(float(args.selector_temperature), 1e-4), dim=1)
        best_indices = detached_losses.argmin(dim=1)
        harmful_mass = (weights * (detached_losses - learned_losses).clamp_min(0.0)).sum(dim=1).mean()
        verification_loss = _verification_head_loss(harm_logits, detached_losses, learned_index, args.safe_margin)
        pairwise_no_harm = _pairwise_no_harm_loss(
            scores,
            detached_losses,
            learned_index,
            safe_margin=args.safe_margin,
            score_margin=args.pairwise_no_harm_margin,
        )
        score_regression = _score_regression_loss(scores, detached_losses)
        loss = (
            (weights * losses).sum(dim=1).mean()
            + args.target_delta_weight * (weights * delta_losses).sum(dim=1).mean()
            + args.ranking_weight * F.cross_entropy(scores, best_indices)
            + args.no_harm_weight * harmful_mass
            + args.pairwise_no_harm_weight * pairwise_no_harm
            + args.score_regression_weight * score_regression
            + args.verification_head_weight * verification_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval(), selector.eval()


def _selector_outputs(
    selector: torch.nn.Module,
    objects: torch.Tensor,
    masks: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = selector(objects, masks, context)
    if isinstance(output, tuple):
        return output
    return output, None


def _deployment_scores(
    raw_scores: torch.Tensor,
    harm_logits: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    if harm_logits is None:
        return raw_scores
    unsafe_probability = torch.sigmoid(harm_logits)
    return raw_scores - float(args.verification_score_penalty) * unsafe_probability


def _verification_head_loss(
    harm_logits: torch.Tensor | None,
    losses: torch.Tensor,
    learned_index: int,
    safe_margin: float,
) -> torch.Tensor:
    if harm_logits is None:
        return losses.new_tensor(0.0)
    learned_losses = losses[:, learned_index].unsqueeze(1)
    harmful = (losses > learned_losses + safe_margin).float()
    positive = harmful.sum().clamp_min(1.0)
    negative = (1.0 - harmful).sum().clamp_min(1.0)
    positive_weight = (negative / positive).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(harm_logits, harmful, pos_weight=positive_weight.detach())


def _pairwise_no_harm_loss(
    scores: torch.Tensor,
    losses: torch.Tensor,
    learned_index: int,
    *,
    safe_margin: float,
    score_margin: float,
) -> torch.Tensor:
    """Push harmful candidate scores below the learned baseline score.

    Expected-loss weighting can still assign probability to harmful candidates
    when their absolute loss is close to the best candidate. This auxiliary
    margin makes the no-harm constraint local to each candidate: if a candidate
    has higher loss than the learned baseline, its score should sit below the
    learned score by at least ``score_margin``.
    """

    learned_losses = losses[:, learned_index].unsqueeze(1)
    learned_scores = scores[:, learned_index].unsqueeze(1)
    harmful = (losses > learned_losses + safe_margin).float()
    candidate_penalty = F.softplus(scores - learned_scores + score_margin) * harmful
    normalizer = harmful.sum(dim=1).clamp_min(1.0)
    return (candidate_penalty.sum(dim=1) / normalizer).mean()


def _score_regression_loss(scores: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Align candidate scores with relative propagation utility.

    Ranking loss only identifies the best candidate, and expected-loss training
    can underuse near-best structured candidates. This normalized regression
    teaches score differences to approximate per-sample candidate loss
    differences without depending on absolute cross-entropy scale.
    """

    centered_scores = scores - scores.mean(dim=1, keepdim=True)
    utility = -(losses - losses.mean(dim=1, keepdim=True))
    score_scale = centered_scores.detach().std(dim=1, keepdim=True).clamp_min(1.0)
    utility_scale = utility.std(dim=1, keepdim=True).clamp_min(1e-3)
    return F.mse_loss(centered_scores / score_scale, utility / utility_scale)


def _prediction_signature(prediction) -> torch.Tensor:
    """Return label-free propagation features for candidate verification.

    These signatures intentionally exclude the ground-truth branch label and
    target delta. They expose how the current propagator behaves on each
    candidate set so the selector can learn when a candidate is overconfident,
    unstable, or weakly separated before deployment.
    """

    probs = F.softmax(prediction.branch_logits, dim=-1)
    top2 = probs.topk(k=2, dim=-1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / torch.log(
        torch.tensor(float(probs.shape[-1]), device=probs.device)
    )
    delta = prediction.object_delta.flatten(1)
    delta_norm = delta.norm(dim=1).clamp_max(10.0) / 10.0
    delta_abs_mean = delta.abs().mean(dim=1).clamp_max(5.0) / 5.0
    logit_scale = prediction.branch_logits.norm(dim=1).clamp_max(10.0) / 10.0
    return torch.stack([confidence, margin, entropy, delta_norm, delta_abs_mean, logit_scale], dim=1)


def _class_weights_from_samples(samples) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float32)
    for sample in samples:
        counts[int(sample.branch_label)] += 1.0
    total = counts.sum().clamp_min(1.0)
    return total / counts.clamp_min(1.0)


def _candidate_sets_for_samples(
    samples,
    retriever: torch.nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
    *,
    seed_offset: int,
) -> dict[str, list[list[str]]]:
    selected: dict[str, list[list[str]]] = {name: [] for name in candidate_names}
    for sample_index, sample in enumerate(samples):
        for mode in BASE_MODES:
            selected[mode].append(_selected_ids(sample, mode, args.budget, retriever if mode == "learned" else None))
        generated = _generated_candidates(sample, args.budget, args.generated_candidates, seed=seed_offset + sample_index)
        for index, ids in enumerate(generated):
            selected[f"generated_{index}"].append(ids)
        structured = _structured_candidates(sample, args.budget, args.structured_candidates)
        for index, ids in enumerate(structured):
            selected[f"structured_{index}"].append(ids)
    return selected


def _structured_candidates(sample, budget: int, count: int) -> list[list[str]]:
    if count <= 0:
        return []
    target = sample.event.target
    candidate_ids = [object_id for object_id in _candidate_ids(sample.state, sample.event) if object_id != target]
    if not candidate_ids:
        return [[target] for _ in range(count)]
    hand_ids = [object_id for object_id in candidate_ids if sample.state.objects[object_id].type == "robot_hand"]
    anchor_ids = [
        object_id
        for object_id in candidate_ids
        if sample.state.objects[object_id].type in {"table", "table_edge"}
    ]
    obstacle_ids = [object_id for object_id in candidate_ids if sample.state.objects[object_id].type == "obstacle"]
    feature_cache = {
        object_id: _candidate_features(sample.state, sample.event, object_id)
        for object_id in candidate_ids
    }
    rankers = [
        lambda object_id: (
            -float(feature_cache[object_id][7]),
            float(feature_cache[object_id][6]),
            object_id,
        ),
        lambda object_id: (
            -float(feature_cache[object_id][8]),
            -float(feature_cache[object_id][7]),
            float(feature_cache[object_id][6]),
            object_id,
        ),
        lambda object_id: (
            float(feature_cache[object_id][6]),
            -float(feature_cache[object_id][7]),
            object_id,
        ),
        lambda object_id: (
            -(1.5 * float(feature_cache[object_id][7]) + float(feature_cache[object_id][8])),
            float(abs(feature_cache[object_id][5])),
            object_id,
        ),
    ]
    generated: list[list[str]] = []
    for index in range(count):
        ranker = rankers[index % len(rankers)]
        selected = [target]
        if hand_ids and index % 3 != 2:
            selected.append(hand_ids[0])
        obstacle_ranked = sorted(obstacle_ids, key=ranker)
        for object_id in obstacle_ranked:
            if object_id not in selected:
                selected.append(object_id)
            if len(selected) >= budget:
                break
        for object_id in [*anchor_ids, *candidate_ids]:
            if len(selected) >= budget:
                break
            if object_id not in selected:
                selected.append(object_id)
        generated.append(selected[:budget])
    return generated


def _collect_examples(
    model: torch.nn.Module,
    selector: GeneratedSetReranker,
    samples,
    retriever: torch.nn.Module,
    candidate_names: list[str],
    args: argparse.Namespace,
    total_n: int,
    causal_k: int,
    *,
    seed_offset: int,
) -> list[dict[str, object]]:
    device = torch.device(args.device)
    selected = _candidate_sets_for_samples(samples, retriever, candidate_names, args, seed_offset=seed_offset)
    losses = {
        name: _evaluate_selected(model, samples, selected[name], args.batch_size, device)
        for name in candidate_names
    }
    verification_context = (
        _candidate_verification_context(model, samples, selected, candidate_names, args, device)
        if args.verification_context
        else None
    )
    objects, masks, context = _candidate_selector_tensors(
        samples,
        selected,
        candidate_names,
        args,
        total_n,
        causal_k,
        verification_context=verification_context,
    )
    with torch.no_grad():
        raw_scores, harm_logits = _selector_outputs(selector, objects.to(device), masks.to(device), context.to(device))
        scores = _deployment_scores(raw_scores, harm_logits, args).detach().cpu()
    examples = []
    for sample_index, sample in enumerate(samples):
        row: dict[str, object] = {
            "branch_label": int(sample.branch_label),
            "object_features": objects[sample_index],
            "object_masks": masks[sample_index],
            "context_features": context[sample_index],
            "selector_scores": scores[sample_index],
        }
        candidate_losses = {name: losses[name][0][sample_index] for name in candidate_names}
        best_name = min(candidate_names, key=lambda name: (candidate_losses[name], name))
        row["best_mode"] = best_name
        row["oracle_loss"] = round(candidate_losses[best_name], 6)
        row["oracle_correct"] = losses[best_name][1][sample_index]
        for name in candidate_names:
            row[f"{name}_loss"] = round(candidate_losses[name], 6)
            row[f"{name}_correct"] = losses[name][1][sample_index]
        examples.append(row)
    return examples


def _candidate_selector_tensors(
    samples,
    selected: dict[str, list[list[str]]],
    candidate_names: list[str],
    args: argparse.Namespace,
    total_n: int,
    causal_k: int,
    verification_context: dict[str, list[list[float]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    object_rows = []
    mask_rows = []
    context_rows = []
    for sample_index, sample in enumerate(samples):
        object_features = []
        object_masks = []
        context_features = []
        for candidate_index, name in enumerate(candidate_names):
            ids = selected[name][sample_index]
            object_tensor, mask_tensor = _selected_object_tensor(sample, ids, args.budget)
            object_features.append(object_tensor)
            object_masks.append(mask_tensor)
            candidate_context = _context_features(
                    sample,
                    ids,
                    name,
                    candidate_index,
                    len(candidate_names),
                    total_n,
                    causal_k,
                    args.budget,
                    use_geometry_context=bool(args.use_geometry_context),
                )
            if verification_context is not None:
                candidate_context.extend(verification_context[name][sample_index])
            context_features.append(candidate_context)
        object_rows.append(torch.stack(object_features))
        mask_rows.append(torch.stack(object_masks))
        context_rows.append(torch.tensor(context_features, dtype=torch.float32))
    return torch.stack(object_rows), torch.stack(mask_rows), torch.stack(context_rows)


def _context_features(
    sample,
    selected_ids: list[str],
    name: str,
    candidate_index: int,
    candidate_count: int,
    total_n: int,
    causal_k: int,
    budget: int,
    *,
    use_geometry_context: bool,
) -> list[float]:
    one_hot = [0.0 for _ in range(candidate_count)]
    one_hot[candidate_index] = 1.0
    obstacle_ids = [object_id for object_id in selected_ids if object_id.startswith("obstacle_")]
    selected_hand = float("hand_001" in selected_ids)
    obstacle_ratio = len(obstacle_ids) / max(float(budget), 1.0)
    pair_density = _selected_pair_density(sample.state, obstacle_ids)
    features = [
        *one_hot,
        selected_hand,
        obstacle_ratio,
        pair_density,
        selected_hand * pair_density,
        obstacle_ratio * pair_density,
        causal_k / 64.0,
        budget / 64.0,
        min(total_n / 4096.0, 4.0),
        float(name.startswith("generated_") or name.startswith("structured_")),
    ]
    if use_geometry_context:
        geometry = _selected_geometry_features(sample, selected_ids)
        features.extend(
            [
                float(geometry["event_force"]),
                float(geometry["obstacle_distance_min"]),
                float(geometry["obstacle_distance_mean"]),
                float(geometry["obstacle_distance_max"]),
                float(geometry["obstacle_distance_span"]),
                float(geometry["obstacle_abs_y_mean"]),
                float(geometry["obstacle_abs_y_max"]),
                float(geometry["obstacle_axis_ratio"]),
                float(geometry["hand_distance"]),
                float(geometry["edge_distance"]),
            ]
        )
    return features


def _context_extra_dim(args: argparse.Namespace) -> int:
    base_dim = CONTEXT_EXTRA_DIM if bool(args.use_geometry_context) else 9
    if bool(getattr(args, "verification_context", False)):
        base_dim += VERIFICATION_CONTEXT_DIM
    return base_dim


def _candidate_verification_context(
    model: torch.nn.Module,
    samples,
    selected: dict[str, list[list[str]]],
    candidate_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[list[float]]]:
    output: dict[str, list[list[float]]] = {}
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for name in candidate_names:
            rows: list[list[float]] = []
            for start in range(0, len(samples), args.batch_size):
                batch_samples = samples[start : start + args.batch_size]
                batch_ids = selected[name][start : start + args.batch_size]
                batch, _, _, _ = collate_selected_working_set_samples(batch_samples, batch_ids)
                prediction = model(_move_batch(batch, device), num_branches=3, force_route="sparse")
                rows.extend(_prediction_signature(prediction).detach().cpu().tolist())
            output[name] = rows
    if model_was_training:
        model.train()
    return output


def _summarize(
    test_examples: list[dict[str, object]],
    validation_examples: list[dict[str, object]],
    selector: GeneratedSetReranker,
    candidate_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    del selector
    static_modes = ["learned"] * len(test_examples)
    selector_modes = _predict_modes(test_examples, candidate_names, confidence_threshold=0.0)
    validation_selector_modes = _predict_modes(validation_examples, candidate_names, confidence_threshold=0.0)
    train_selector_loss = _mean_policy_loss(validation_examples, validation_selector_modes)
    train_static_loss = _mean_policy_loss(validation_examples, ["learned"] * len(validation_examples))
    train_harmful = _harmful_accept_rate(validation_examples, validation_selector_modes)
    selected_threshold, selected_train = _select_confidence_threshold(validation_examples, candidate_names, args)
    train_selected_modes = _predict_modes(test_examples, candidate_names, confidence_threshold=selected_threshold)
    oracle_modes = [str(example["best_mode"]) for example in test_examples]
    rows = [
        _policy_row(test_examples, "static_learned_interaction", static_modes, candidate_names),
        _policy_row(test_examples, "joint_selector_propagator", selector_modes, candidate_names),
        _policy_row(
            test_examples,
            "confidence_selected_joint_selector_propagator",
            train_selected_modes,
            candidate_names,
        ),
        _policy_row(test_examples, "train_selected_joint_selector_propagator", train_selected_modes, candidate_names),
        _policy_row(test_examples, "generated_plus_composition_oracle", oracle_modes, candidate_names),
    ]
    for row in rows:
        row["train_joint_selector_loss"] = round(train_selector_loss, 6)
        row["train_static_learned_loss"] = round(train_static_loss, 6)
        row["train_joint_selector_harmful_accept_rate"] = round(train_harmful, 6)
        row["selection_train_loss"] = selected_train["loss"]
        row["selection_train_gap_closure"] = selected_train["gap_closure"]
        row["selection_train_accept_rate"] = selected_train["accept_rate"]
        row["selection_train_harmful_accept_rate"] = selected_train["harmful_accept_rate"]
        row["selection_train_policy"] = f"confidence_t{_float_token(selected_threshold)}"
        row["selected_confidence_threshold"] = round(selected_threshold, 6)
    return rows


def _select_confidence_threshold(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    static_loss = _mean_policy_loss(examples, ["learned"] * len(examples))
    oracle_modes = [str(example["best_mode"]) for example in examples]
    oracle_loss = _mean_policy_loss(examples, oracle_modes)
    oracle_gap = max(static_loss - oracle_loss, 1e-8)
    evaluated = []
    for threshold in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        modes = _predict_modes(examples, candidate_names, confidence_threshold=threshold)
        loss = _mean_policy_loss(examples, modes)
        metrics = {
            "loss": round(loss, 6),
            "gap_closure": round((static_loss - loss) / oracle_gap, 6),
            "accept_rate": round(mean(float(mode != "learned") for mode in modes), 6),
            "harmful_accept_rate": round(_harmful_accept_rate(examples, modes), 6),
        }
        evaluated.append((threshold, metrics))
    safe = [
        item
        for item in evaluated
        if item[1]["harmful_accept_rate"] <= args.selection_harmful_limit
        and item[1]["gap_closure"] > 0.0
        and item[1]["loss"] + args.safe_margin < static_loss
    ]
    candidates = safe or evaluated
    return max(candidates, key=lambda item: (item[1]["gap_closure"], -item[1]["harmful_accept_rate"]))


def _predict_modes(
    examples: list[dict[str, object]],
    candidate_names: list[str],
    *,
    confidence_threshold: float,
) -> list[str]:
    modes: list[str] = []
    for example in examples:
        scores = torch.as_tensor(example["selector_scores"])
        probs = F.softmax(scores, dim=0)
        selected_index = int(probs.argmax().item())
        selected = candidate_names[selected_index]
        if float(probs[selected_index].item()) < confidence_threshold:
            selected = "learned"
        modes.append(selected)
    return modes


def _float_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "neg").replace(".", "p")


def _mean_policy_loss(rows: list[dict[str, object]], selected_modes: list[str]) -> float:
    return mean(float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True))


def _harmful_accept_rate(rows: list[dict[str, object]], selected_modes: list[str]) -> float:
    harmful = []
    for row, mode in zip(rows, selected_modes, strict=True):
        if mode == "learned":
            harmful.append(0.0)
        else:
            harmful.append(float(float(row[f"{mode}_loss"]) > float(row["learned_loss"])))
    return mean(harmful) if harmful else 0.0


def _policy_row(
    rows: list[dict[str, object]],
    policy: str,
    selected_modes: list[str],
    candidate_names: list[str],
) -> dict[str, object]:
    losses = [float(row[f"{mode}_loss"]) for row, mode in zip(rows, selected_modes, strict=True)]
    correct = [float(row[f"{mode}_correct"]) for row, mode in zip(rows, selected_modes, strict=True)]
    oracle_losses = [float(row["oracle_loss"]) for row in rows]
    output = {
        "policy": policy,
        "loss": round(mean(losses), 6),
        "accuracy": round(mean(correct), 6),
        "candidate_oracle_loss": round(mean(oracle_losses), 6),
        "excess_over_candidate_oracle": round(mean(loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)), 6),
        "oracle_match_rate": round(mean(float(mode == str(row["best_mode"])) for row, mode in zip(rows, selected_modes, strict=True)), 6),
        "selected_generated_rate": round(mean(float(mode.startswith("generated_")) for mode in selected_modes), 6),
        "selected_composition_rate": 0.0,
        "accept_rate": round(mean(float(mode != "learned") for mode in selected_modes), 6),
        "harmful_accept_rate": round(_harmful_accept_rate(rows, selected_modes), 6),
    }
    for name in candidate_names:
        output[f"selected_{name}_rate"] = round(sum(mode == name for mode in selected_modes) / max(len(selected_modes), 1), 6)
    return output


if __name__ == "__main__":
    main()
