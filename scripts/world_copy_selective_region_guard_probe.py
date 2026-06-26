from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress selective WPU region guards under region mis-objectification.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--contamination", type=int, nargs="+", default=[0, 8, 32, 128])
    parser.add_argument("--missing-membership", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--streams", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--k-ref", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/world_copy_selective_region_guard_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/world_copy_selective_region_guard_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/world_copy_selective_region_guard_probe_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    root_rng = random.Random(args.seed)
    for n in args.world_sizes:
        for contamination in args.contamination:
            if contamination > n - args.k_ref:
                continue
            for missing in args.missing_membership:
                for mode in ("wpu-region-guard", "wpu-selective-region-guard", "dense-state-copy"):
                    trials = [
                        run_stream(n, args.k_ref, contamination, missing, args.horizon, mode,
                                   random.Random(root_rng.randrange(2**31)))
                        for _ in range(args.streams)
                    ]
                    rows.append({
                        "mode": mode, "total_n": n, "contamination": contamination,
                        "missing_membership": missing, "horizon": args.horizon, "streams": args.streams,
                        "mean_selected_k": mean(trials, "selected_k"),
                        "max_selected_k": max(t["selected_k"] for t in trials),
                        "causal_recall": mean(trials, "recall"), "causal_precision": mean(trials, "precision"),
                        "trajectory_mse": mean(trials, "mse"), "state_integrity": mean(trials, "integrity"),
                        "false_updates": mean(trials, "false_updates"),
                        "work_proxy": mean(trials, "work"), "bytes_proxy": mean(trials, "bytes"),
                    })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.out_md.write_text(report(rows, args.out_csv, False), encoding="utf-8")
    args.out_ko_md.write_text(report(rows, args.out_csv, True), encoding="utf-8")
    print(f"wrote={args.out_csv}\nwrote={args.out_md}\nwrote={args.out_ko_md}")


def run_stream(n: int, k: int, contamination: int, missing: float, horizon: int,
               mode: str, rng: random.Random) -> dict[str, float]:
    causal = [f"causal_{i}" for i in range(k)]
    background = [f"bg_{i}" for i in range(n - k)]
    region = [object_id for object_id in causal if rng.random() >= missing]
    region += rng.sample(background, contamination)
    relation_frontier = {causal[0]}
    relation_frontier.update(object_id for object_id in causal[1:] if rng.random() >= 0.5)

    # Typed object state is imperfect: 20% of contaminants alias the causal role.
    confidence = {object_id: rng.uniform(0.85, 1.0) for object_id in causal}
    role = {object_id: "affected" for object_id in causal}
    for object_id in background:
        role[object_id] = "affected" if rng.random() < 0.2 else "background"
        confidence[object_id] = rng.uniform(0.35, 0.8)

    if mode == "wpu-region-guard":
        selected = set(region) | relation_frontier
    elif mode == "wpu-selective-region-guard":
        candidates = set(region) | relation_frontier
        scored = sorted(candidates, key=lambda oid: (
            oid in relation_frontier, role[oid] == "affected", confidence[oid]
        ), reverse=True)
        selected = set(scored[: 2 * k])
    elif mode == "dense-state-copy":
        selected = set(causal)
    else:
        raise ValueError(mode)

    causal_selected = selected & set(causal)
    false_selected = selected - set(causal)
    recall = len(causal_selected) / k
    precision = len(causal_selected) / max(len(selected), 1)
    # Missed causal state accumulates drift; false updates corrupt unrelated persistent state.
    missed_error = 0.0
    false_error = 0.0
    for step in range(horizon):
        force = 0.5 + 0.04 * step
        missed_error += sum((force * (0.6 + 0.05 * (i % 3))) ** 2
                            for i, oid in enumerate(causal) if oid not in selected) / k
        false_error += sum((force * confidence[oid] * 0.25) ** 2 for oid in false_selected) / k
    mse = (missed_error + false_error) / horizon
    work = n if mode == "dense-state-copy" else len(selected)
    return {
        "selected_k": float(len(selected)), "recall": recall, "precision": precision,
        "mse": mse, "integrity": 1.0 / (1.0 + mse),
        "false_updates": float(len(false_selected)), "work": float(work), "bytes": float(work * 9 * 4),
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6)


def report(rows: list[dict[str, object]], source: Path, ko: bool) -> str:
    title = "# Selective Region Guard Mis-objectification Probe"
    intro = ("이 probe는 region contamination과 causal membership 누락에서 bounded selective guard의 실패 경계를 검증한다."
             if ko else "This probe tests the failure boundary of a bounded selective guard under region contamination and missing causal membership.")
    lines = [title, "", intro, "", f"Source CSV: `{source.as_posix()}`.", "",
             "| mode | N | contamination | missing | mean K | max K | recall | precision | MSE | integrity | false updates | work | bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['total_n']} | {r['contamination']} | {float(r['missing_membership']):.2f} | "
                     f"{float(r['mean_selected_k']):.3f} | {r['max_selected_k']} | {float(r['causal_recall']):.3f} | "
                     f"{float(r['causal_precision']):.3f} | {float(r['trajectory_mse']):.4f} | {float(r['state_integrity']):.3f} | "
                     f"{float(r['false_updates']):.3f} | {float(r['work_proxy']):.1f} | {float(r['bytes_proxy']):.1f} |")
    lines += (["", "## 해석", "", "- 선택적 guard는 K를 `2*K_ref`로 제한하여 contamination이 증가해도 N에 독립적인 비용을 유지한다.",
               "- 오염 후보의 typed role이 causal object와 겹치면 precision과 무결성은 여전히 저하된다.",
               "- causal membership과 relation evidence가 동시에 누락되면 어떤 local guard도 해당 객체를 복구할 수 없다.",
               "- dense baseline은 완전한 recall을 유지하지만 O(N) state touch를 지불한다."] if ko else
              ["", "## Interpretation", "", "- The selective guard caps K at `2*K_ref`, keeping cost independent of N as contamination grows.",
               "- Precision and integrity still degrade when contaminated candidates alias causal typed roles.",
               "- No local guard can recover an object missing from both region membership and relation evidence.",
               "- The dense baseline preserves full recall but pays O(N) state touches."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
