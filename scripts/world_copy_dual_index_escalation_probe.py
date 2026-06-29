from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe bounded WPU escalation when region and relation indexes both omit causal objects."
    )
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--dual-omission", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--escape-rate", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--contamination", type=int, default=128)
    parser.add_argument("--streams", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_dual_index_escalation_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_dual_index_escalation_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_dual_index_escalation_probe_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    root_rng = random.Random(args.seed)
    for n in args.world_sizes:
        for dual_omission in args.dual_omission:
            for escape_rate in args.escape_rate:
                for mode in ("wpu-selective-region-guard", "wpu-escalating-neighbor-guard", "dense-state-copy"):
                    trials = [
                        run_stream(
                            n=n,
                            k=args.k_ref,
                            contamination=args.contamination,
                            dual_omission=dual_omission,
                            escape_rate=escape_rate,
                            horizon=args.horizon,
                            mode=mode,
                            rng=random.Random(root_rng.randrange(2**31)),
                        )
                        for _ in range(args.streams)
                    ]
                    rows.append({
                        "mode": mode,
                        "total_n": n,
                        "dual_omission": dual_omission,
                        "escape_rate": escape_rate,
                        "horizon": args.horizon,
                        "streams": args.streams,
                        "mean_selected_k": mean(trials, "selected_k"),
                        "max_selected_k": max(t["selected_k"] for t in trials),
                        "causal_recall": mean(trials, "recall"),
                        "causal_precision": mean(trials, "precision"),
                        "trajectory_mse": mean(trials, "mse"),
                        "state_integrity": mean(trials, "integrity"),
                        "false_updates": mean(trials, "false_updates"),
                        "escalation_rate": mean(trials, "escalated"),
                        "correction_cost": mean(trials, "correction_cost"),
                        "work_proxy": mean(trials, "work"),
                        "bytes_proxy": mean(trials, "bytes"),
                    })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_md.write_text(report(rows, args.out_csv, ko=False), encoding="utf-8")
    args.out_ko_md.write_text(report(rows, args.out_csv, ko=True), encoding="utf-8")
    print(f"wrote={args.out_csv}\nwrote={args.out_md}\nwrote={args.out_ko_md}")


def run_stream(
    n: int,
    k: int,
    contamination: int,
    dual_omission: float,
    escape_rate: float,
    horizon: int,
    mode: str,
    rng: random.Random,
) -> dict[str, float]:
    causal = [f"causal_{i}" for i in range(k)]
    background = [f"bg_{i}" for i in range(n - k)]
    causal_set = set(causal)

    active: set[str] = set()
    relation_frontier: set[str] = set()
    neighbor_pool: set[str] = set()
    hidden_unknown: set[str] = set()
    for object_id in causal:
        if rng.random() < dual_omission:
            if rng.random() < escape_rate:
                hidden_unknown.add(object_id)
            else:
                neighbor_pool.add(object_id)
            continue
        active.add(object_id)
        if rng.random() >= 0.25:
            relation_frontier.add(object_id)

    distractor_count = min(contamination, len(background))
    active.update(rng.sample(background, distractor_count))
    remaining_background = [oid for oid in background if oid not in active]
    neighbor_pool.update(rng.sample(remaining_background, min(contamination, len(remaining_background))))

    confidence = {object_id: rng.uniform(0.88, 1.0) for object_id in causal}
    role = {object_id: "affected" for object_id in causal}
    for object_id in background:
        role[object_id] = "affected" if rng.random() < 0.2 else "background"
        confidence[object_id] = rng.uniform(0.35, 0.78)

    base_candidates = active | relation_frontier
    observed_support = sum(
        1
        for object_id in base_candidates
        if role[object_id] == "affected" and confidence[object_id] >= 0.85
    )
    should_escalate = observed_support < k

    if mode == "wpu-selective-region-guard":
        selected = score_and_cap(base_candidates, relation_frontier, role, confidence, 2 * k)
        correction_cost = 0.0
        escalated = 0.0
    elif mode == "wpu-escalating-neighbor-guard":
        candidates = set(base_candidates)
        escalated = 1.0 if should_escalate else 0.0
        correction_cost = 0.0
        cap = 2 * k
        if should_escalate:
            before = len(candidates)
            candidates.update(neighbor_pool)
            correction_cost = float(len(candidates) - before)
            cap = 3 * k
        selected = score_and_cap(candidates, relation_frontier | neighbor_pool, role, confidence, cap)
    elif mode == "dense-state-copy":
        selected = set(causal)
        correction_cost = 0.0
        escalated = 0.0
    else:
        raise ValueError(mode)

    causal_selected = selected & causal_set
    false_selected = selected - causal_set
    recall = len(causal_selected) / k
    precision = len(causal_selected) / max(len(selected), 1)
    missed_error = 0.0
    false_error = 0.0
    for step in range(horizon):
        force = 0.5 + 0.04 * step
        missed_error += sum(
            (force * (0.6 + 0.05 * (i % 3))) ** 2
            for i, object_id in enumerate(causal)
            if object_id not in selected
        ) / k
        false_error += sum((force * confidence[object_id] * 0.25) ** 2 for object_id in false_selected) / k
    mse = (missed_error + false_error) / horizon
    work = n if mode == "dense-state-copy" else len(selected)
    return {
        "selected_k": float(len(selected)),
        "recall": recall,
        "precision": precision,
        "mse": mse,
        "integrity": 1.0 / (1.0 + mse),
        "false_updates": float(len(false_selected)),
        "escalated": escalated,
        "correction_cost": correction_cost,
        "work": float(work),
        "bytes": float(work * 9 * 4),
    }


def score_and_cap(
    candidates: set[str],
    relation_frontier: set[str],
    role: dict[str, str],
    confidence: dict[str, float],
    cap: int,
) -> set[str]:
    scored = sorted(
        candidates,
        key=lambda oid: (role[oid] == "affected", oid in relation_frontier, confidence[oid]),
        reverse=True,
    )
    return set(scored[:cap])


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6)


def report(rows: list[dict[str, object]], source: Path, ko: bool) -> str:
    if ko:
        lines = [
            "# Dual-index Omission Escalation Probe",
            "",
            "이 probe는 causal object가 region membership과 relation evidence에서 동시에 빠질 때, WPU가 full-state scan으로 돌아가지 않고 제한된 인접 correction pool로 복구할 수 있는지 검증한다.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | dual omission | escape | mean K | max K | recall | precision | MSE | integrity | escalation | correction cost | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Dual-index Omission Escalation Probe",
            "",
            "This probe tests whether WPU can recover when causal objects are omitted from both region membership and relation evidence, without falling back to a full-state scan.",
            "",
            f"Source CSV: `{source.as_posix()}`.",
            "",
            "| mode | N | dual omission | escape | mean K | max K | recall | precision | MSE | integrity | escalation | correction cost | work | bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['total_n']} | {float(row['dual_omission']):.2f} | "
            f"{float(row['escape_rate']):.2f} | {float(row['mean_selected_k']):.3f} | "
            f"{row['max_selected_k']} | {float(row['causal_recall']):.3f} | "
            f"{float(row['causal_precision']):.3f} | {float(row['trajectory_mse']):.4f} | "
            f"{float(row['state_integrity']):.3f} | {float(row['escalation_rate']):.3f} | "
            f"{float(row['correction_cost']):.3f} | {float(row['work_proxy']):.1f} | "
            f"{float(row['bytes_proxy']):.1f} |"
        )
    if ko:
        lines += [
            "",
            "## 해석",
            "",
            "- `wpu-escalating-neighbor-guard`는 불확실성 신호가 발생할 때 인접 correction pool만 추가하고 K를 `3*K_ref`로 제한한다.",
            "- causal object가 인접 pool 안에 남아 있으면 selective guard보다 recall, trajectory MSE, state integrity를 개선할 수 있다.",
            "- 그 대가는 correction cost와 더 큰 K다. 이 비용은 N이 아니라 bounded local pool에 의해 제한된다.",
            "- causal object가 region, relation, 인접 observation pool에서 모두 빠지는 escape 조건에서는 dense-state-copy가 여전히 raw accuracy에서 이긴다.",
            "- 따라서 이 결과는 WPU의 보편 우월성이 아니라, dual-index omission을 제한된 외부/인접 관측으로 복구할 수 있는 조건부 v3 경계를 제시한다.",
        ]
    else:
        lines += [
            "",
            "## Interpretation",
            "",
            "- `wpu-escalating-neighbor-guard` adds only a bounded adjacent correction pool when uncertainty is triggered and caps K at `3*K_ref`.",
            "- If omitted causal objects remain in that adjacent pool, escalation improves recall, trajectory MSE, and state integrity over the selective guard.",
            "- The cost is higher correction work and larger K, but the cost is bounded by the local pool rather than total N.",
            "- If a causal object is absent from region, relation, and adjacent observation pools, dense-state-copy still wins raw accuracy.",
            "- This is therefore not a universal WPU superiority result; it fixes a specific v3 boundary where dual-index omissions are recoverable by bounded external/local observation.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
