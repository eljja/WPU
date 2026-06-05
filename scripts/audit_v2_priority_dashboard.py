from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


ROOT = Path("docs/experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative WPU v2 priority dashboard from existing experiment CSVs."
    )
    parser.add_argument("--out-csv", type=Path, default=ROOT / "wpu_v2_priority_dashboard.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "wpu_v2_priority_dashboard.md")
    parser.add_argument("--out-ko-md", type=Path, default=ROOT / "wpu_v2_priority_dashboard.ko.md")
    args = parser.parse_args()

    rows = [
        _priority_candidate_oracle_gap(),
        _priority_state_integrity(),
        _priority_simulator_grounding(),
        _priority_shift_generalization(),
        _priority_calibration(),
        _priority_systems_profile(),
        _priority_objectification_quality(),
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render_markdown(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _priority_candidate_oracle_gap() -> dict[str, object]:
    path = ROOT / "wpu_v2_candidate_oracle_gap_v2.csv"
    rows = _read_rows(path)
    target_rows = [
        row
        for row in rows
        if row["feature_variant"] == "role_geometry_family"
        and row["policy"] == "risk_adjusted_selected_mechanism"
    ]
    aggregate_best = max(float(row["gap_closure_fraction"]) for row in target_rows)
    mean = statistics.fmean(float(row["gap_closure_fraction"]) for row in target_rows)
    noharm_path = ROOT / "wpu_v2_candidate_noharm_gate_summary.csv"
    noharm_best = None
    if noharm_path.exists():
        noharm_rows = _read_rows(noharm_path)
        noharm_best = max(float(row["gap_closure_fraction"]) for row in noharm_rows)
    noharm_note = (
        f" Sample-level no-harm/margin gates were also audited; best closure is {noharm_best:.6f}, so margin gating is not the missing fix."
        if noharm_best is not None
        else ""
    )
    regret_path = ROOT / "wpu_v2_candidate_regret_gate_summary.csv"
    regret_best = None
    if regret_path.exists():
        regret_rows = _read_rows(regret_path)
        regret_best = max(float(row["gap_closure_fraction"]) for row in regret_rows)
    regret_note = (
        f" Direct candidate-regret gating improves best closure to {regret_best:.6f}, but it remains below threshold and has harmful accepts."
        if regret_best is not None
        else ""
    )
    best = max(value for value in [aggregate_best, noharm_best, regret_best] if value is not None)
    source = regret_path if regret_best == best else path
    return _row(
        1,
        "Candidate-oracle gap",
        "fail" if best < 0.5 else "partial",
        best,
        0.5,
        "gap_closure_fraction",
        source,
        f"Best deployed closure is {best:.6f}; previous aggregate-policy best is {aggregate_best:.6f} and mean aggregate closure is {mean:.6f}.{noharm_note}{regret_note}",
        "Strengthen candidate-regret training with calibrated uncertainty, harmful-accept penalties, and cross-seed perturbations.",
    )


def _priority_state_integrity() -> dict[str, object]:
    path = ROOT / "pybullet_state_integrity_audit.csv"
    rows = _read_rows(path)
    wpu_h25 = [
        row
        for row in rows
        if row["model"].startswith("wpu-") and int(row["horizon"]) == 25
    ]
    best = max(float(row["state_integrity_score"]) for row in wpu_h25)
    sparse_clipped = next(
        float(row["state_integrity_score"])
        for row in wpu_h25
        if row["run_label"] == "clipped" and row["model"] == "wpu-cws-indexed-sparse"
    )
    sparse_guarded = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "guarded" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    sparse_regularized = next(
        (
            float(row["state_integrity_score"])
            for row in wpu_h25
            if row["run_label"] == "regularized" and row["model"] == "wpu-cws-indexed-sparse"
        ),
        0.0,
    )
    return _row(
        2,
        "Long-horizon state integrity",
        "fail" if best < 0.8 else "partial",
        best,
        0.8,
        "best_wpu_h25_integrity",
        path,
        f"Best WPU H=25 integrity is {best:.6f}; guarded sparse is {sparse_guarded:.6f}, clipped sparse is {sparse_clipped:.6f}, and regularized raw sparse is {sparse_regularized:.6f}.",
        "Simple delta-norm regularization is insufficient; add rollout-consistency loss, unsafe-delta rejection, rollback, correction, and uncertainty escalation.",
    )


def _priority_simulator_grounding() -> dict[str, object]:
    path = ROOT / "pybullet_cup_benchmark.csv"
    rows = _read_rows(path)
    seed_rows = _rows_of_type(rows, "seed")
    summary_rows = _rows_of_type(rows, "summary")
    seed_count = len({row["seed"] for row in seed_rows})
    max_background = max(int(float(row["background_objects"])) for row in summary_rows)
    status = "partial" if seed_count >= 2 and max_background >= 128 else "fail"
    return _row(
        3,
        "Simulator-backed benchmark",
        status,
        float(seed_count),
        5.0,
        "seed_count",
        path,
        f"PyBullet benchmark exists with {seed_count} seeds and background up to N_bg={max_background}, but it is still small.",
        "Increase seeds, mechanisms, training scale, and long-horizon simulator rollouts.",
    )


def _priority_shift_generalization() -> dict[str, object]:
    path = ROOT / "pybullet_shift_generalization.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    shifts = sorted({row["eval_mechanism"] for row in rows if row["eval_mechanism"] != "nominal"})
    wins = 0
    notes: list[str] = []
    for mechanism in shifts:
        group = [row for row in rows if row["eval_mechanism"] == mechanism]
        best_wpu = max(float(row["branch_accuracy"]) for row in group if row["model"].startswith("wpu-"))
        best_baseline = max(float(row["branch_accuracy"]) for row in group if not row["model"].startswith("wpu-"))
        if best_wpu >= best_baseline:
            wins += 1
        notes.append(f"{mechanism}: WPU {best_wpu:.6f} vs baseline {best_baseline:.6f}")
    win_rate = wins / max(1, len(shifts))
    status = "partial" if 0.0 < win_rate < 1.0 else ("pass" if win_rate == 1.0 else "fail")
    return _row(
        4,
        "Mechanism-family shift generalization",
        status,
        win_rate,
        1.0,
        "wpu_shift_win_rate",
        path,
        "; ".join(notes),
        "Add leave-family-out training, harder shifts, and mechanism-aware branch priors.",
    )


def _priority_calibration() -> dict[str, object]:
    path = ROOT / "pybullet_shift_generalization.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    wpu_ece = statistics.fmean(float(row["ece"]) for row in rows if row["model"].startswith("wpu-"))
    baseline_ece = statistics.fmean(float(row["ece"]) for row in rows if not row["model"].startswith("wpu-"))
    ratio = wpu_ece / baseline_ece if baseline_ece > 0 else float("inf")
    status = "partial" if ratio <= 1.1 else "fail"
    return _row(
        5,
        "Calibration and uncertainty",
        status,
        ratio,
        1.0,
        "wpu_ece_over_baseline_ece",
        path,
        f"Mean WPU ECE is {wpu_ece:.6f}; mean baseline ECE is {baseline_ece:.6f}; ratio is {ratio:.6f}.",
        "Add temperature heads, branch calibration loss, multi-step ECE/Brier/NLL, and uncertainty-gated recompute.",
    )


def _priority_systems_profile() -> dict[str, object]:
    path = ROOT / "pybullet_system_profile.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    max_reduction = max(float(row["tensor_byte_reduction"]) for row in rows)
    max_latency_reduction = max(float(row.get("tensorize_latency_reduction", 0.0)) for row in rows)
    max_n = max(float(row["total_objects"]) for row in rows)
    status = "partial" if max_reduction >= 0.95 else "fail"
    return _row(
        6,
        "Systems profile and memory traffic",
        status,
        max_reduction,
        0.95,
        "max_tensor_byte_reduction",
        path,
        f"Tensor-byte reduction reaches {max_reduction:.6f} at mean total objects {max_n:.1f}; CPU tensorization latency reduction reaches {max_latency_reduction:.6f}, but model-forward/GPU/energy data is absent.",
        "Measure model forward latency, CUDA memory, allocator traffic, sparse-kernel behavior, and matched-accuracy speedups.",
    )


def _priority_objectification_quality() -> dict[str, object]:
    path = ROOT / "pybullet_objectification_quality.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    combined = [row for row in rows if row["corruption"] == "combined"]
    clean = [row for row in rows if row["corruption"] == "clean"]
    combined_score = statistics.fmean(float(row["objectification_score"]) for row in combined)
    clean_score = statistics.fmean(float(row["objectification_score"]) for row in clean)
    frontier = statistics.fmean(float(row["frontier_recall"]) for row in combined)
    status = "partial" if combined_score < clean_score and frontier < 1.0 else "fail"
    return _row(
        7,
        "Objectification quality to propagation loss",
        status,
        combined_score,
        clean_score,
        "combined_objectification_score",
        path,
        f"Clean score {clean_score:.6f}, combined-corruption score {combined_score:.6f}, combined frontier recall {frontier:.6f}. Metrics exist, but downstream loss coupling is incomplete.",
        "Train/evaluate propagation under controlled objectification corruption and regress loss against report components.",
    )


def _row(
    priority: int,
    name: str,
    status: str,
    observed: float,
    target: float,
    metric: str,
    source: Path,
    interpretation: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "priority": priority,
        "name": name,
        "status": status,
        "observed": round(observed, 6),
        "target": round(target, 6),
        "metric": metric,
        "source": source.as_posix(),
        "interpretation": interpretation,
        "next_action": next_action,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_of_type(rows: list[dict[str, str]], row_type: str) -> list[dict[str, str]]:
    if not rows or "row_type" not in rows[0]:
        return rows
    typed = [row for row in rows if row.get("row_type") == row_type]
    return typed or rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], *, korean: bool) -> str:
    if korean:
        title = "# WPU v2 우선순위 대시보드"
        intro = (
            "이 문서는 기존 실험 CSV에서 v2 우선순위 1~7의 현재 상태를 "
            "보수적으로 재계산한다. 목적은 WPU 주장이 실험 증거를 초과하지 "
            "않도록 만드는 것이다."
        )
        headers = "| 우선순위 | 항목 | 상태 | 관측값 | 목표 | 지표 |"
        sep = "|---:|---|---|---:|---:|---|"
        interp = "## 해석"
        next_title = "## 다음 조치"
        boundary = (
            "현재 dashboard는 WPU v2가 유망하지만 아직 완결된 우월성 주장이 "
            "아님을 보여준다. 가장 강한 주장은 large-N 자체가 아니라, "
            "objectified state에서 작은 causal working set K를 tensorization 전에 "
            "식별할 수 있을 때 WPU가 계산량과 메모리 측면에서 유리해진다는 "
            "조건부 주장이다."
        )
    else:
        title = "# WPU v2 Priority Dashboard"
        intro = (
            "This dashboard conservatively recomputes the current status of v2 "
            "priorities 1-7 from existing experiment CSVs. Its purpose is to keep "
            "WPU claims aligned with evidence."
        )
        headers = "| Priority | Item | Status | Observed | Target | Metric |"
        sep = "|---:|---|---|---:|---:|---|"
        interp = "## Interpretation"
        next_title = "## Next Actions"
        boundary = (
            "The dashboard shows that WPU v2 is promising but not a completed "
            "superiority claim. The strongest claim remains conditional: WPU can "
            "reduce compute and memory when objectified state exposes a small "
            "causal working set K before tensorization. Large N alone is not enough."
        )

    lines = [title, "", intro, "", headers, sep]
    for row in rows:
        name = _ko_name(int(row["priority"])) if korean else str(row["name"])
        status = _ko_status(str(row["status"])) if korean else str(row["status"])
        lines.append(
            f"| {row['priority']} | {name} | {status} | "
            f"{float(row['observed']):.6f} | {float(row['target']):.6f} | `{row['metric']}` |"
        )
    lines.extend(["", interp, "", boundary, ""])
    for row in rows:
        if korean:
            lines.append(f"- P{row['priority']} {_ko_name(int(row['priority']))}: {_ko_interpretation(int(row['priority']))}")
        else:
            lines.append(f"- P{row['priority']} {row['name']}: {row['interpretation']}")
    lines.extend(["", next_title, ""])
    for row in rows:
        if korean:
            lines.append(f"- P{row['priority']}: {_ko_next_action(int(row['priority']))}")
        else:
            lines.append(f"- P{row['priority']}: {row['next_action']}")
    return "\n".join(lines) + "\n"


def _ko_name(priority: int) -> str:
    return {
        1: "Candidate-oracle gap",
        2: "장기 state integrity",
        3: "Simulator-backed benchmark",
        4: "Mechanism-family shift generalization",
        5: "Calibration과 uncertainty",
        6: "Systems profile과 memory traffic",
        7: "Objectification quality와 propagation loss",
    }[priority]


def _ko_status(status: str) -> str:
    return {
        "pass": "pass",
        "partial": "partial",
        "fail": "fail",
    }[status]


def _ko_interpretation(priority: int) -> str:
    return {
        1: "최고 deployed closure는 candidate-regret gate의 0.308651로 개선됐다. 이전 aggregate-policy best는 0.244220이고 평균 aggregate closure는 0.160601이다. Sample-level no-harm/margin gate는 최고 0.082804에 그쳤으므로 threshold만으로는 부족하다. Candidate-regret target은 효과가 있지만 harmful accept가 높아 P1은 아직 fail이다.",
        2: "최고 WPU H=25 integrity는 0.964322이고 guarded sparse는 0.958508이다. Regularized raw sparse는 0.087153으로 raw sparse 0.084722보다 거의 개선되지 않는다. 따라서 state-store guard가 적용 state를 보호한 것이지 raw delta model 안정성이 해결된 것은 아니다.",
        3: "PyBullet benchmark는 5개 seed와 background N_bg=128까지 확장됐다. 다만 mechanism 다양성, training scale, long-horizon simulator rollout은 아직 부족하다.",
        4: "5-seed shift benchmark에서 WPU는 catch_heavy에서 앞서지만 edge_shift와 high_force에서는 baseline에 밀린다. Shift generalization은 부분적으로만 성립한다.",
        5: "5-seed 평균 WPU ECE는 0.213693, baseline ECE는 0.244135로 ratio가 0.875306까지 개선됐다. 하지만 multi-step/shift calibration이 해결된 것은 아니므로 partial로 유지한다.",
        6: "Tensor-byte reduction은 mean total objects 2052.6에서 0.997454까지 도달하고 CPU tensorization latency reduction도 0.995549까지 도달한다. 다만 model-forward/GPU/energy 증거는 아직 없다.",
        7: "Clean score는 0.957711, combined-corruption score는 0.821712, frontier recall은 0.742361이다. Objectification metric은 있지만 downstream loss 연결은 미완성이다.",
    }[priority]


def _ko_next_action(priority: int) -> str:
    return {
        1: "Candidate-regret 학습에 calibrated uncertainty, harmful-accept penalty, cross-seed perturbation을 더 강하게 넣는다.",
        2: "단순 delta-norm regularization은 부족하다. Guarded state-store projection을 유지하되, rollout-consistency loss, unsafe-delta rejection, rollback/correction을 학습 단계로 끌어올린다.",
        3: "Seed, mechanism, training scale, long-horizon simulator rollout을 늘린다.",
        4: "Leave-family-out training, 더 어려운 shift, mechanism-aware branch prior를 추가한다.",
        5: "Temperature head, branch calibration loss, multi-step ECE/Brier/NLL, uncertainty-gated recompute를 추가한다.",
        6: "Model forward latency, CUDA memory, allocator traffic, sparse-kernel behavior, matched-accuracy speedup을 측정한다.",
        7: "Controlled objectification corruption에서 propagation을 학습/평가하고 report component와 downstream loss의 관계를 회귀 분석한다.",
    }[priority]


if __name__ == "__main__":
    main()
