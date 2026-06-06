from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit matched-accuracy speedup regimes from PyBullet benchmark CSVs."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("docs/experiments/pybullet_matched_baseline_benchmark.csv"),
    )
    parser.add_argument(
        "--cuda-profile",
        type=Path,
        default=Path("docs/experiments/pybullet_system_profile_cuda.csv"),
    )
    parser.add_argument("--accuracy-tolerance", type=float, default=0.03)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_matched_speedup_audit.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_matched_speedup_audit_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_matched_speedup_audit_results.ko.md"),
    )
    args = parser.parse_args()

    benchmark_rows = _read_rows(args.benchmark)
    profile_rows = _read_rows(args.cuda_profile)
    summary_rows = _summarize_benchmark(benchmark_rows)
    rows = _matched_rows(summary_rows, args.accuracy_tolerance)
    profile_summary = _profile_summary(profile_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render_markdown(rows, profile_summary, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(rows, profile_summary, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize_benchmark(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], int(float(row["total_objects_n"]))), []).append(row)
    out: list[dict[str, object]] = []
    for (model, total_n), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        out.append(
            {
                "model": model,
                "total_objects_n": total_n,
                "seed_count": len({row["seed"] for row in group}),
                "params": round(_mean(group, "params"), 3),
                "branch_accuracy": round(_mean(group, "branch_accuracy"), 6),
                "ms_per_sample_forward": round(_mean(group, "ms_per_sample_forward"), 6),
                "cuda_peak_mb": round(_mean(group, "cuda_peak_mb"), 6),
                "selected_k_mean": round(_mean(group, "selected_k_mean"), 6),
            }
        )
    return out


def _matched_rows(summary_rows: list[dict[str, object]], tolerance: float) -> list[dict[str, object]]:
    by_n: dict[int, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_n.setdefault(int(row["total_objects_n"]), []).append(row)
    out: list[dict[str, object]] = []
    for total_n, group in sorted(by_n.items()):
        wpu_rows = [row for row in group if str(row["model"]).startswith("wpu-")]
        baseline_rows = [row for row in group if not str(row["model"]).startswith("wpu-")]
        if not wpu_rows or not baseline_rows:
            continue
        best_wpu = max(wpu_rows, key=lambda row: float(row["branch_accuracy"]))
        best_baseline = max(baseline_rows, key=lambda row: float(row["branch_accuracy"]))
        accuracy_gap = float(best_wpu["branch_accuracy"]) - float(best_baseline["branch_accuracy"])
        speedup = float(best_baseline["ms_per_sample_forward"]) / max(float(best_wpu["ms_per_sample_forward"]), 1e-9)
        memory_ratio = float(best_wpu["cuda_peak_mb"]) / max(float(best_baseline["cuda_peak_mb"]), 1e-9)
        matched = abs(accuracy_gap) <= tolerance
        out.append(
            {
                "total_objects_n": total_n,
                "best_wpu": best_wpu["model"],
                "best_baseline": best_baseline["model"],
                "wpu_accuracy": best_wpu["branch_accuracy"],
                "baseline_accuracy": best_baseline["branch_accuracy"],
                "accuracy_gap": round(accuracy_gap, 6),
                "accuracy_tolerance": tolerance,
                "matched_accuracy": matched,
                "wpu_ms_per_sample": best_wpu["ms_per_sample_forward"],
                "baseline_ms_per_sample": best_baseline["ms_per_sample_forward"],
                "matched_speedup": round(speedup, 6),
                "wpu_cuda_peak_mb": best_wpu["cuda_peak_mb"],
                "baseline_cuda_peak_mb": best_baseline["cuda_peak_mb"],
                "wpu_over_baseline_peak_memory": round(memory_ratio, 6),
                "seed_count": min(int(best_wpu["seed_count"]), int(best_baseline["seed_count"])),
            }
        )
    return out


def _profile_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    summary = [row for row in rows if row.get("row_type") == "summary"] or rows
    if not summary:
        return {}
    return {
        "max_total_objects": max(float(row["total_objects"]) for row in summary),
        "max_sparse_forward_reduction": max(float(row.get("sparse_forward_latency_reduction", 0.0)) for row in summary),
        "max_sparse_peak_memory_reduction": max(float(row.get("sparse_peak_memory_reduction", 0.0)) for row in summary),
    }


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    rows: list[dict[str, object]],
    profile: dict[str, float],
    args: argparse.Namespace,
    *,
    korean: bool,
) -> str:
    if korean:
        title = "# PyBullet Matched-Accuracy Speedup Audit"
        intro = (
            "이 audit은 parameter-matched PyBullet benchmark에서 best WPU와 best non-WPU baseline을 "
            "같은 N별로 비교하고, accuracy gap이 tolerance 안에 있을 때만 speedup을 해석한다."
        )
        interp = (
            "이 결과는 matched-accuracy가 성립하는 구간에서만 speedup을 주장해야 함을 보여준다. "
            "CUDA systems profile은 큰 N에서 random-model sparse forward latency가 크게 줄어드는 "
            "상한 근거를 제공하지만, benchmark 표의 speedup은 실제 학습된 small model 조건이다."
        )
    else:
        title = "# PyBullet Matched-Accuracy Speedup Audit"
        intro = (
            "This audit compares the best WPU and best non-WPU baseline at each N in the "
            "parameter-matched PyBullet benchmark. Speedup is interpreted only when the "
            "accuracy gap is within the configured tolerance."
        )
        interp = (
            "The result enforces a stricter claim: WPU speedup only matters in regimes "
            "where accuracy is matched. The CUDA systems profile gives an upper-bound "
            "random-model latency signal at large N, while the benchmark table reports "
            "trained small-model runtime."
        )
    lines = [
        title,
        "",
        intro,
        "",
        f"Benchmark CSV: `{args.benchmark.as_posix()}`",
        f"CUDA systems CSV: `{args.cuda_profile.as_posix()}`",
        f"Accuracy tolerance: `{args.accuracy_tolerance}`",
        "",
        "| N | best WPU | best baseline | WPU acc | baseline acc | gap | matched | speedup | WPU ms | baseline ms | WPU/baseline peak mem |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['total_objects_n']} | `{row['best_wpu']}` | `{row['best_baseline']}` | "
            f"{float(row['wpu_accuracy']):.6f} | {float(row['baseline_accuracy']):.6f} | "
            f"{float(row['accuracy_gap']):.6f} | {row['matched_accuracy']} | "
            f"{float(row['matched_speedup']):.6f} | {float(row['wpu_ms_per_sample']):.6f} | "
            f"{float(row['baseline_ms_per_sample']):.6f} | {float(row['wpu_over_baseline_peak_memory']):.6f} |"
        )
    lines.extend(["", "## Interpretation", "", interp])
    if profile:
        lines.extend(
            [
                "",
                "## CUDA Profile Context",
                "",
                (
                    f"At max profiled N `{profile['max_total_objects']:.1f}`, random-model sparse forward "
                    f"latency reduction is `{profile['max_sparse_forward_reduction']:.6f}` and sparse "
                    f"peak-memory reduction is `{profile['max_sparse_peak_memory_reduction']:.6f}`."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
