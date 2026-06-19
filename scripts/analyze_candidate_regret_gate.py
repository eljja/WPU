from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUT = Path("docs/experiments/wpu_v2_candidate_regret_gate.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/wpu_v2_candidate_regret_gate_summary.csv")
DEFAULT_OUT_MD = Path("docs/experiments/wpu_v2_candidate_regret_gate_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/wpu_v2_candidate_regret_gate_results.ko.md")


NON_DEPLOYED_POLICIES = {"static_learned_interaction", "generated_plus_composition_oracle"}
MAX_MARKDOWN_ROWS = 18


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize candidate-regret gate closure for WPU v2 priority-1 tracking."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summary = _summarize(rows, args.input)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, summary)
    args.out_md.write_text(_render_markdown(summary, args.input, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(summary, args.input, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]], source: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for causal_k in sorted({int(row["causal_k"]) for row in rows}):
        group = [row for row in rows if int(row["causal_k"]) == causal_k]
        policies = sorted({row["policy"] for row in group})
        losses = {
            policy: statistics.fmean(float(row["loss"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        accuracies = {
            policy: statistics.fmean(float(row["accuracy"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        matches = {
            policy: statistics.fmean(float(row["oracle_match_rate"]) for row in group if row["policy"] == policy)
            for policy in policies
        }
        static_loss = losses["static_learned_interaction"]
        oracle_loss = losses["generated_plus_composition_oracle"]
        oracle_gap = static_loss - oracle_loss
        deployed_policies = [policy for policy in policies if policy not in NON_DEPLOYED_POLICIES]
        for policy in deployed_policies:
            policy_rows = [row for row in group if row["policy"] == policy]
            deployed_gain = static_loss - losses[policy]
            gap_closure = deployed_gain / oracle_gap if oracle_gap > 0 else 0.0
            out.append(
                {
                    "source": source.as_posix(),
                    "total_objects_n": int(float(policy_rows[0]["total_objects_n"])),
                    "causal_k": causal_k,
                    "policy": policy,
                    "static_loss": round(static_loss, 6),
                    "candidate_oracle_loss": round(oracle_loss, 6),
                    "candidate_oracle_gain_over_static": round(oracle_gap, 6),
                    "policy_loss": round(losses[policy], 6),
                    "policy_accuracy": round(accuracies[policy], 6),
                    "oracle_match_rate": round(matches[policy], 6),
                    "deployed_gain_over_static": round(deployed_gain, 6),
                    "gap_closure_fraction": round(gap_closure, 6),
                    "remaining_gap": round(losses[policy] - oracle_loss, 6),
                    "mean_accept_rate": _mean_optional(policy_rows, "accept_rate"),
                    "mean_harmful_accept_rate": _mean_optional(policy_rows, "harmful_accept_rate"),
                    "mean_regret_corr": _mean_optional(policy_rows, "regret_corr"),
                    "mean_predicted_sigma": _mean_optional(policy_rows, "predicted_sigma_mean"),
                    "mean_selection_train_gap_closure": _mean_optional(policy_rows, "selection_train_gap_closure"),
                    "mean_selection_train_harmful_accept_rate": _mean_optional(policy_rows, "selection_train_harmful_accept_rate"),
                    "seed_count": len({int(row["seed"]) for row in policy_rows}),
                    "failure_mode": _failure_mode(gap_closure, _mean_optional(policy_rows, "harmful_accept_rate")),
                }
            )
    return out


def _mean_optional(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return round(statistics.fmean(values), 6) if values else 0.0


def _failure_mode(gap_closure: float, harmful_accept_rate: float) -> str:
    if gap_closure < 0.0:
        return "harmful_transfer"
    if harmful_accept_rate > 0.25:
        return "insufficient_no_harm_rejection"
    if gap_closure < 0.5:
        return "partial_but_insufficient_gap_closure"
    return "passes_current_p1_threshold"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], source: Path, *, korean: bool) -> str:
    is_safety_gate = "candidate_safety_gate" in source.name
    is_invariant_gate = "candidate_invariant_gate" in source.name
    is_joint_gate = "candidate_joint_gate" in source.name
    is_end_to_end_selector = "end_to_end_candidate_selector" in source.name
    is_verified_controller = "verified_candidate_controller" in source.name
    is_joint_adapter = "joint_propagation_adapter" in source.name
    is_joint_utility_verifier = "joint_utility_verifier" in source.name
    is_joint_selector_propagator = "joint_selector_propagator" in source.name
    is_joint_selector_geometry = "joint_selector_propagator_geometry" in source.name
    is_joint_selector_budget = "joint_selector_propagator_budget" in source.name
    is_joint_selector_relation = "joint_selector_propagator_relation" in source.name
    is_joint_selector_pairwise_noharm = "joint_selector_propagator_pairwise_noharm" in source.name
    is_joint_selector_structured_candidates = "joint_selector_propagator_structured_candidates" in source.name
    best = max(rows, key=lambda row: float(row["gap_closure_fraction"]))
    safe_rows = [row for row in rows if float(row["mean_harmful_accept_rate"]) <= 0.25]
    safe_best = max(safe_rows, key=lambda row: float(row["gap_closure_fraction"])) if safe_rows else None
    train_selected_rows = [row for row in rows if str(row["policy"]).startswith("train_selected_")]
    train_selected_best = (
        max(train_selected_rows, key=lambda row: float(row["gap_closure_fraction"]))
        if train_selected_rows
        else None
    )
    best_by_k = []
    for causal_k in sorted({int(row["causal_k"]) for row in rows}):
        group = [row for row in rows if int(row["causal_k"]) == causal_k]
        best_by_k.append(max(group, key=lambda row: float(row["gap_closure_fraction"])))
    table_rows = _top_rows(rows, best_by_k)
    if korean:
        title = (
            "# Candidate Invariant Gate 결과"
            if is_invariant_gate
            else "# Joint Utility Verifier 결과"
            if is_joint_utility_verifier
            else "# Structured-Candidate Joint Selector-Propagator 결과"
            if is_joint_selector_structured_candidates
            else "# Pairwise No-Harm Joint Selector-Propagator 결과"
            if is_joint_selector_pairwise_noharm
            else "# Joint Selector-Propagator 결과"
            if is_joint_selector_propagator
            else "# Joint Propagation Adapter 결과"
            if is_joint_adapter
            else "# Verified Candidate Controller 결과"
            if is_verified_controller
            else "# End-to-End Candidate Selector 결과"
            if is_end_to_end_selector
            else "# Joint Object-Set Candidate Gate 결과"
            if is_joint_gate
            else "# Candidate Safety/Utility Gate 결과"
            if is_safety_gate
            else "# Candidate Regret Gate 결과"
        )
        if is_joint_selector_structured_candidates:
            intro = (
                "이 문서는 joint selector-propagator에 deterministic structured candidate를 "
                "추가한 P1 ablation을 요약한다. 목적은 K=16/32 병목이 나쁜 후보를 "
                "거부하는 문제가 아니라 안전하고 좋은 후보 자체가 부족한 문제인지 검사하는 것이다."
            )
        elif is_joint_selector_pairwise_noharm:
            intro = (
                "이 문서는 joint selector-propagator에 pairwise no-harm score margin을 "
                "추가한 P1 ablation을 요약한다. 목적은 K=16/32에서 높은 harmful accept가 "
                "단순 confidence threshold 문제가 아니라 selector score 자체의 baseline-safe "
                "ordering 문제인지 검사하는 것이다."
            )
        elif is_joint_selector_relation:
            intro = (
                "이 문서는 joint selector-propagator의 propagation model을 relation-conditioned "
                "WPU로 바꾼 P1 ablation을 요약한다. 목적은 K=16/32 병목이 transition dynamics "
                "부족인지 검사하는 것이다. Closure가 오르지만 harmful accept가 남으면 relation-aware "
                "dynamics는 방향이지만 no-harm candidate selection이 아직 부족하다고 해석한다."
            )
        elif is_joint_selector_budget:
            intro = (
                "이 문서는 joint selector-propagator의 working-set budget을 확대한 P1 ablation을 "
                "요약한다. 목적은 K=16/32 실패가 budget=4의 과도한 causal-state 절단 때문인지 "
                "검사하는 것이다."
            )
        elif is_joint_selector_geometry:
            intro = (
                "이 문서는 joint selector-propagator에 후보별 geometry/force context를 추가한 "
                "P1 ablation을 요약한다. 목적은 K=16/32 실패가 단순 descriptor 부족인지 "
                "검사하는 것이다. 결과가 약하면 larger-K 병목은 feature concatenation이 아니라 "
                "retrieval, generation, propagation dynamics의 더 깊은 공동학습 문제로 해석한다."
            )
        elif is_joint_selector_propagator:
            intro = (
                "이 문서는 후보 working set selector와 WPU sparse propagation branch loss를 "
                "같은 학습 그래프에서 최적화한 P1 probe를 요약한다. 기존 post-hoc selector, "
                "고정 propagation verifier, branch-logit adapter보다 candidate choice와 "
                "propagation dynamics를 더 직접적으로 결합하지만, hard object retrieval 자체는 "
                "아직 완전한 미분 가능 end-to-end 생성기가 아니다."
            )
        elif is_joint_utility_verifier:
            intro = (
                "이 문서는 후보 object set, compact context, sparse/local-dense verification "
                "signature를 함께 인코딩하고 candidate regret, uncertainty, no-harm safety를 "
                "동시에 예측하는 P1 probe를 요약한다. 이는 post-hoc feature 추가보다 더 "
                "직접적인 joint utility/safety head지만, propagation model 자체는 아직 고정되어 있다."
            )
        elif is_joint_adapter:
            intro = (
                "이 문서는 후보별 sparse/local-dense verification feature로 branch-logit "
                "propagation adapter를 학습한 뒤, adapted propagation loss 위에서 "
                "candidate-regret/no-harm deployment를 평가한 P1 probe를 요약한다. "
                "이는 full retriever-propagator end-to-end training은 아니지만, selector와 "
                "propagation output correction을 같은 supervision 아래 묶는 중간 단계다."
            )
        elif is_verified_controller:
            intro = (
                "이 문서는 후보 working set의 object/context feature에 sparse 및 local-dense "
                "propagation의 label-free verification signature를 추가한 P1 probe를 요약한다. "
                "signature는 branch confidence, entropy, sparse/dense disagreement, delta norm gap처럼 "
                "정답 label 없이 계산 가능한 값만 사용한다."
            )
        elif is_end_to_end_selector:
            intro = (
                "이 문서는 후보 working set selector를 downstream propagation loss와 "
                "baseline보다 나빠지는 no-harm mass에 직접 맞춰 학습한 P1 probe를 요약한다. "
                "목표는 oracle label imitation이 아니라 선택 정책의 실제 expected loss를 줄이는 것이다. "
                "단, 후보 생성기와 propagation model은 고정되어 있으므로 full joint "
                "retriever-propagator training으로 해석하면 안 된다."
            )
        elif is_joint_gate:
            intro = (
                "이 문서는 후보 working set의 명시적 object-set feature와 compact context를 함께 인코딩하는 "
                "P1 probe를 요약한다. 목표는 post-hoc threshold가 아니라 후보 state 자체를 보고 "
                "candidate regret와 no-harm accept를 예측할 수 있는지 평가하는 것이다."
            )
        elif is_invariant_gate:
            intro = (
                "이 문서는 candidate descriptor를 train split에서 표준화하고, "
                "train seed별 worst-group loss와 no-harm objective를 함께 줄이는 P1 probe를 요약한다."
            )
        else:
            intro = (
            "이 문서는 candidate별 safe probability와 utility를 별도로 예측하고, "
            "예측 utility와 안전 확률이 충분할 때만 baseline 대신 선택하는 P1 probe를 요약한다."
            if is_safety_gate
            else "이 문서는 candidate별 `candidate_loss - learned_loss`를 직접 예측하고, "
            "예측 regret이 충분히 낮을 때만 baseline 대신 선택하는 P1 probe를 요약한다."
            )
        conclusion = (
            f"최고 closure는 `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`)다. P1 목표 `0.5`를 기준으로 "
            f"{'joint selector-propagator deployment' if is_joint_selector_propagator else 'joint utility-verifier deployment' if is_joint_utility_verifier else 'joint propagation-adapter deployment' if is_joint_adapter else 'verified-controller deployment' if is_verified_controller else 'end-to-end selector deployment' if is_end_to_end_selector else 'joint object-set deployment' if is_joint_gate else 'invariant-gate deployment' if is_invariant_gate else 'safety/utility deployment' if is_safety_gate else 'candidate-regret deployment'}가 candidate-oracle gap을 충분히 닫는지와 "
            "harmful accept를 억제하는지를 동시에 본다."
            + (
                f" Harmful accept <= `0.25` 조건의 conservative best는 "
                f"`{float(safe_best['gap_closure_fraction']):.6f}` "
                f"(`{safe_best['policy']}`)다."
                if safe_best is not None
                else " Harmful accept <= `0.25` 조건을 만족하는 deployed policy는 없다."
            )
            + (
                f" Train-selected deployed best는 `{float(train_selected_best['gap_closure_fraction']):.6f}` "
                f"(`K={train_selected_best['causal_k']}`)다."
                if train_selected_best is not None
                else ""
            )
        )
        notes_title = "## 해석"
        notes = [
            "CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.",
            "아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.",
            "좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.",
        ]
        if is_joint_gate:
            notes.append(
                "이 실험이 direct candidate-regret gate보다 약하면, 병목은 object-set feature 부재만이 아니라 cross-seed regret target 자체의 안정성 부족으로 해석한다."
            )
        if is_end_to_end_selector:
            notes.append(
                "이 실험도 direct candidate-regret gate보다 약하면, P1 병목은 후처리 threshold가 아니라 후보 생성/전파 모델과 selector의 더 깊은 공동학습 문제로 해석한다."
            )
        if is_verified_controller:
            notes.append(
                "이 실험이 direct candidate-regret gate보다 약하면, label-free sparse/dense verification signature를 후처리 feature로 추가하는 것만으로는 P1을 해결하지 못한다고 해석한다."
            )
        if is_joint_adapter:
            notes.append(
                "이 실험이 direct candidate-regret gate보다 약하면, 단순 branch-logit adapter도 P1 병목을 풀기에 부족하며 retrieval, propagation dynamics, no-harm objective를 더 깊게 공동학습해야 한다."
            )
        if is_joint_utility_verifier:
            notes.append(
                "이 실험이 direct candidate-regret gate보다 약하면, object set, verification signature, utility/safety head를 결합해도 propagation model이 고정된 상태에서는 P1 병목을 해결하지 못한다고 해석한다."
            )
        if is_joint_selector_propagator:
            notes.append(
                "이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다."
            )
        if is_joint_selector_geometry:
            notes.append(
                "Geometry/force descriptor 추가가 K=16/32를 개선하지 못하면, 단순 feature 추가가 아니라 candidate 생성 및 relation-conditioned transition learning이 다음 병목이다."
            )
        if is_joint_selector_budget:
            notes.append(
                "Budget 확장만으로 closure가 크게 오르지 않으면, larger-K 실패는 working-set 크기만의 문제가 아니라 후보 품질과 transition dynamics의 문제다."
            )
        if is_joint_selector_relation:
            notes.append(
                "Relation-conditioned propagation이 closure를 올리지만 harmful accept를 낮추지 못하면, 다음 단계는 relation-aware transition과 no-harm candidate selection의 공동학습이다."
            )
        if is_joint_selector_pairwise_noharm:
            notes.append(
                "Pairwise no-harm margin이 harmful accept를 낮추지만 closure도 크게 낮추면, P1 병목은 안전 제약 부재만이 아니라 안전한 후보 생성과 relation-aware propagation 품질의 결합 문제다."
            )
        if is_joint_selector_structured_candidates:
            notes.append(
                "Structured candidate 추가가 oracle 또는 deployed closure를 올리지 못하면, 손으로 만든 다양성보다 학습된 candidate generation과 propagation-aware verification이 필요하다고 해석한다."
            )
    else:
        title = (
            "# Candidate Invariant Gate Results"
            if is_invariant_gate
            else "# Joint Utility Verifier Results"
            if is_joint_utility_verifier
            else "# Structured-Candidate Joint Selector-Propagator Results"
            if is_joint_selector_structured_candidates
            else "# Pairwise No-Harm Joint Selector-Propagator Results"
            if is_joint_selector_pairwise_noharm
            else "# Joint Selector-Propagator Results"
            if is_joint_selector_propagator
            else "# Joint Propagation Adapter Results"
            if is_joint_adapter
            else "# Verified Candidate Controller Results"
            if is_verified_controller
            else "# End-to-End Candidate Selector Results"
            if is_end_to_end_selector
            else "# Joint Object-Set Candidate Gate Results"
            if is_joint_gate
            else "# Candidate Safety/Utility Gate Results"
            if is_safety_gate
            else "# Candidate Regret Gate Results"
        )
        if is_joint_selector_structured_candidates:
            intro = (
                "This report summarizes a P1 ablation that adds deterministic "
                "structured candidates to the joint selector-propagator probe. "
                "It tests whether K=16/32 fails because the system lacks safe "
                "high-quality candidates, rather than only because it fails to "
                "reject unsafe ones."
            )
        elif is_joint_selector_pairwise_noharm:
            intro = (
                "This report summarizes a P1 ablation that adds a pairwise "
                "no-harm score margin to the joint selector-propagator objective. "
                "It tests whether high harmful accept at K=16/32 is not merely "
                "a confidence-threshold problem, but a baseline-safe ordering "
                "problem in the selector scores themselves."
            )
        elif is_joint_selector_relation:
            intro = (
                "This report summarizes a P1 ablation that swaps the joint "
                "selector-propagator propagation model to a relation-conditioned "
                "WPU. It tests whether the K=16/32 bottleneck is partly caused by "
                "insufficient transition dynamics. If closure improves but harmful "
                "accept remains high, relation-aware dynamics are useful but "
                "no-harm candidate selection is still insufficient."
            )
        elif is_joint_selector_budget:
            intro = (
                "This report summarizes a P1 ablation that increases the joint "
                "selector-propagator working-set budget. It tests whether K=16/32 "
                "fails mainly because budget=4 cuts too much causal state."
            )
        elif is_joint_selector_geometry:
            intro = (
                "This report summarizes a geometry-context ablation of the joint "
                "selector-propagator probe. It appends candidate-level geometry "
                "and force descriptors to test whether the K=16/32 failure is "
                "just a descriptor bottleneck. Weak results imply that larger-K "
                "P1 needs deeper joint retrieval, generation, and propagation "
                "dynamics rather than feature concatenation."
            )
        elif is_joint_selector_propagator:
            intro = (
                "This report summarizes a P1 probe that optimizes candidate "
                "working-set selector scores and WPU sparse propagation branch "
                "losses in the same computation graph. It couples candidate "
                "choice to propagation dynamics more directly than post-hoc "
                "selectors, fixed-propagator verifiers, or shallow branch-logit "
                "adapters, but hard object retrieval is still not a fully "
                "differentiable end-to-end generator."
            )
        elif is_joint_utility_verifier:
            intro = (
                "This report summarizes a P1 probe that jointly encodes candidate "
                "object sets, compact context, and sparse/local-dense verification "
                "signatures, then predicts candidate regret, uncertainty, and "
                "no-harm safety. It is a more direct joint utility/safety head than "
                "post-hoc feature addition, but the propagation model is still fixed."
            )
        elif is_joint_adapter:
            intro = (
                "This report summarizes a P1 probe that trains a candidate-aware "
                "branch-logit propagation adapter from sparse/local-dense "
                "verification features, then evaluates candidate-regret/no-harm "
                "deployment on adapted propagation losses. It is not full "
                "retriever-propagator end-to-end training, but it couples "
                "selection supervision to propagation-output correction."
            )
        elif is_verified_controller:
            intro = (
                "This report summarizes a P1 probe that augments each candidate "
                "working set with label-free sparse/local-dense propagation "
                "verification signatures. The signature includes branch confidence, "
                "entropy, sparse/dense disagreement, and delta-norm gaps, all "
                "computed without ground-truth labels."
            )
        elif is_end_to_end_selector:
            intro = (
                "This report summarizes a P1 probe that trains the candidate "
                "working-set selector directly on downstream propagation loss and "
                "no-harm mass relative to the learned baseline. The objective is "
                "policy-level expected loss, not only oracle-label imitation. "
                "The candidate generator and propagation model are fixed, so this "
                "is not full joint retriever-propagator training."
            )
        elif is_joint_gate:
            intro = (
                "This report summarizes a P1 probe that jointly encodes each candidate "
                "working set as an explicit object set plus compact context. It tests "
                "whether candidate regret and no-harm acceptance become more transferable "
                "when the selector sees the candidate state itself rather than only "
                "aggregate descriptors."
            )
        elif is_invariant_gate:
            intro = (
                "This report summarizes a P1 probe that standardizes candidate descriptors "
                "on the training split and jointly minimizes no-harm utility loss, "
                "worst-source-seed loss, and cross-source variance."
            )
        else:
            intro = (
            "This report summarizes a P1 probe that predicts safe probability "
            "and utility separately, then deploys a candidate only when the "
            "predicted utility and safety probability are sufficiently favorable."
            if is_safety_gate
            else "This report summarizes a P1 probe that directly predicts "
            "`candidate_loss - learned_loss` and deploys a candidate only when "
            "predicted regret is sufficiently favorable."
            )
        conclusion = (
            f"The best closure is `{float(best['gap_closure_fraction']):.6f}` "
            f"(`K={best['causal_k']}`, `{best['policy']}`). P1 evaluates whether "
            f"{'joint selector-propagator deployment' if is_joint_selector_propagator else 'joint utility-verifier deployment' if is_joint_utility_verifier else 'joint propagation-adapter deployment' if is_joint_adapter else 'verified-controller deployment' if is_verified_controller else 'end-to-end selector deployment' if is_end_to_end_selector else 'joint object-set deployment' if is_joint_gate else 'invariant-gate deployment' if is_invariant_gate else 'safety/utility deployment' if is_safety_gate else 'candidate-regret deployment'} closes the candidate-oracle gap while "
            "controlling harmful accepts."
            + (
                f" The conservative best under harmful-accept <= `0.25` is "
                f"`{float(safe_best['gap_closure_fraction']):.6f}` "
                f"(`{safe_best['policy']}`)."
                if safe_best is not None
                else " No deployed policy satisfies harmful-accept <= `0.25`."
            )
            + (
                f" The train-selected deployed best is `{float(train_selected_best['gap_closure_fraction']):.6f}` "
                f"(`K={train_selected_best['causal_k']}`)."
                if train_selected_best is not None
                else ""
            )
        )
        notes_title = "## Interpretation"
        notes = [
            "The CSV keeps all reject-margin/risk-penalty deployment sweep points.",
            "The table below shows the best policy per K and the strongest overall policies.",
            "A useful deployed policy needs both high closure and low harmful accepts.",
        ]
        if is_joint_gate:
            notes.append(
                "If this probe underperforms the direct candidate-regret gate, the bottleneck is not merely missing object-set features; the cross-seed regret target itself remains unstable."
            )
        if is_end_to_end_selector:
            notes.append(
                "If this fixed-candidate/fixed-propagator probe underperforms the direct candidate-regret gate, P1 is not merely a post-hoc thresholding problem; candidate generation, propagation, and selector training need deeper joint supervision."
            )
        if is_verified_controller:
            notes.append(
                "If this probe underperforms the direct candidate-regret gate, post-hoc label-free sparse/dense verification signatures are not sufficient; verification must be trained jointly with retrieval and propagation."
            )
        if is_joint_adapter:
            notes.append(
                "If this probe underperforms the direct candidate-regret gate, a shallow branch-logit adapter is still insufficient; retrieval, propagation dynamics, and no-harm objectives need deeper joint training."
            )
        if is_joint_utility_verifier:
            notes.append(
                "If this probe underperforms the direct candidate-regret gate, combining object sets, verification signatures, and utility/safety heads is still insufficient while the propagation model remains fixed."
            )
        if is_joint_selector_propagator:
            notes.append(
                "If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training."
            )
        if is_joint_selector_geometry:
            notes.append(
                "If geometry/force descriptors do not improve K=16/32, the next bottleneck is candidate generation and relation-conditioned transition learning, not more post-hoc descriptor concatenation."
            )
        if is_joint_selector_budget:
            notes.append(
                "If increasing the budget gives only small gains, larger-K failure is not just working-set size; candidate quality and transition dynamics remain bottlenecks."
            )
        if is_joint_selector_relation:
            notes.append(
                "If relation-conditioned propagation raises closure but not harmful-accept safety, the next step is joint relation-aware transition learning with no-harm candidate selection."
            )
        if is_joint_selector_pairwise_noharm:
            notes.append(
                "If the pairwise no-harm margin lowers harmful accept but also collapses closure, P1 is not solved by safety regularization alone; safe candidate generation and relation-aware propagation quality must improve together."
            )
        if is_joint_selector_structured_candidates:
            notes.append(
                "If structured candidates do not improve oracle or deployed closure, hand-built diversity is not enough; candidate generation needs learned propagation-aware verification."
            )

    lines = [
        title,
        "",
        intro,
        "",
        f"Source CSV: `{source.as_posix()}`",
        "",
        conclusion,
        "",
        "| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['causal_k']} | `{row['policy']}` | {float(row['policy_loss']):.6f} | "
            f"{float(row['policy_accuracy']):.6f} | {float(row['candidate_oracle_gain_over_static']):.6f} | "
            f"{float(row['deployed_gain_over_static']):.6f} | {float(row['gap_closure_fraction']):.6f} | "
            f"{float(row['mean_accept_rate']):.6f} | {float(row['mean_harmful_accept_rate']):.6f} | "
            f"{float(row['mean_regret_corr']):.6f} | `{row['failure_mode']}` |"
        )
    lines.extend(["", notes_title, ""])
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def _top_rows(rows: list[dict[str, object]], best_by_k: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[int, str]] = set()
    for row in [*best_by_k, *sorted(rows, key=lambda item: float(item["gap_closure_fraction"]), reverse=True)]:
        key = (int(row["causal_k"]), str(row["policy"]))
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
        if len(selected) >= MAX_MARKDOWN_ROWS:
            break
    return selected


if __name__ == "__main__":
    main()
