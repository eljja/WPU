from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("docs/experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit which P6 systems claims are supported, bounded, or still unmeasured."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "pybullet_system_claim_boundary.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "pybullet_system_claim_boundary_results.md")
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=ROOT / "pybullet_system_claim_boundary_results.ko.md",
    )
    args = parser.parse_args()

    rows = _build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out, rows)
    args.out_md.write_text(_render_report(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_report(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    profile = _summary_rows(ROOT / "pybullet_system_profile.csv")
    cuda_profile = _summary_rows(ROOT / "pybullet_system_profile_cuda.csv")

    best_tensor = max(profile, key=lambda row: float(row["tensor_byte_reduction"]))
    rows.append(
        _row(
            axis="pre_tensor_working_set_tensor_bytes",
            status="supported_proxy",
            evidence_type="CPU tensorization proxy",
            observed=float(best_tensor["tensor_byte_reduction"]),
            target=0.95,
            supporting_n=float(best_tensor["total_objects"]),
            supporting_b=int(float(best_tensor["branch_count"])),
            source="docs/experiments/pybullet_system_profile.csv",
            interpretation=(
                "Indexed WPU tensorization sharply reduces object/relation tensor bytes before dense tensor compute."
            ),
            limitation="This is a tensorization proxy, not hardware memory-traffic telemetry.",
        )
    )

    best_cpu_forward = max(profile, key=lambda row: float(row["sparse_forward_latency_reduction"]))
    rows.append(
        _row(
            axis="random_cpu_sparse_forward_latency",
            status="supported_proxy",
            evidence_type="random-model CPU forward proxy",
            observed=float(best_cpu_forward["sparse_forward_latency_reduction"]),
            target=0.95,
            supporting_n=float(best_cpu_forward["total_objects"]),
            supporting_b=int(float(best_cpu_forward["branch_count"])),
            source="docs/experiments/pybullet_system_profile.csv",
            interpretation="Random-model sparse forward latency falls sharply at large N.",
            limitation="Random-model latency is an upper-bound signal, not trained matched-accuracy speedup.",
        )
    )

    if cuda_profile:
        best_cuda_forward = max(cuda_profile, key=lambda row: float(row["sparse_forward_latency_reduction"]))
        rows.append(
            _row(
                axis="random_cuda_sparse_forward_latency",
                status="supported_proxy",
                evidence_type="random-model CUDA forward proxy",
                observed=float(best_cuda_forward["sparse_forward_latency_reduction"]),
                target=0.95,
                supporting_n=float(best_cuda_forward["total_objects"]),
                supporting_b=int(float(best_cuda_forward["branch_count"])),
                source="docs/experiments/pybullet_system_profile_cuda.csv",
                interpretation="CUDA sparse-forward latency proxy is strongly positive at large N.",
                limitation="This still uses generic PyTorch modules, not a custom sparse frontier kernel.",
            )
        )

        best_cuda_memory = max(cuda_profile, key=lambda row: float(row["sparse_peak_memory_reduction"]))
        rows.append(
            _row(
                axis="random_cuda_peak_memory",
                status="weak_proxy",
                evidence_type="CUDA peak-memory proxy",
                observed=float(best_cuda_memory["sparse_peak_memory_reduction"]),
                target=0.95,
                supporting_n=float(best_cuda_memory["total_objects"]),
                supporting_b=int(float(best_cuda_memory["branch_count"])),
                source="docs/experiments/pybullet_system_profile_cuda.csv",
                interpretation="CUDA peak-memory reduction is much weaker than CUDA latency reduction.",
                limitation="P6 cannot claim broad GPU-memory dominance from the current PyTorch profile.",
            )
        )

    branch_candidates = [row for row in profile if int(float(row["branch_count"])) > 1]
    best_branch = max(branch_candidates, key=lambda row: float(row["branch_memory_reduction"]))
    rows.append(
        _row(
            axis="branch_overlay_memory",
            status="supported_proxy",
            evidence_type="state-store memory proxy",
            observed=float(best_branch["branch_memory_reduction"]),
            target=0.8,
            supporting_n=float(best_branch["total_objects"]),
            supporting_b=int(float(best_branch["branch_count"])),
            source="docs/experiments/pybullet_system_profile.csv",
            interpretation="BaseState plus branch overlays avoid full state copies when branch count is greater than one.",
            limitation="This is byte accounting from state objects, not allocator-level resident-memory telemetry.",
        )
    )

    matched_path = ROOT / "pybullet_matched_speedup_audit.csv"
    if matched_path.exists():
        matched_rows = _read_rows(matched_path)
        positive = [row for row in matched_rows if row["matched_accuracy"] == "True" and float(row["matched_speedup"]) > 1.0]
        best = max(matched_rows, key=lambda row: float(row["matched_speedup"]))
        rows.append(
            _row(
                axis="trained_matched_or_better_speedup",
                status="partial_matched",
                evidence_type="trained benchmark audit",
                observed=len(positive) / max(1, len(matched_rows)),
                target=1.0,
                supporting_n=float(best["total_objects_n"]),
                supporting_b="",
                source="docs/experiments/pybullet_matched_speedup_audit.csv",
                interpretation=(
                    f"Matched-or-better speedup is positive at {len(positive)}/{len(matched_rows)} audited N values; "
                    f"best speedup is {float(best['matched_speedup']):.6f}."
                ),
                limitation="Current trained speedup evidence has only two N values and is not universal latency dominance.",
            )
        )

    pareto_path = ROOT / "pybullet_pareto_frontier.csv"
    if pareto_path.exists():
        pareto_rows = _read_rows(pareto_path)
        n_values = sorted({row["total_objects_n"] for row in pareto_rows})
        wpu_frontier = [
            row
            for row in pareto_rows
            if row["is_wpu"] == "True" and row["pareto_frontier"] == "True"
        ]
        rows.append(
            _row(
                axis="accuracy_latency_pareto_frontier",
                status="partial_pareto",
                evidence_type="trained Pareto audit",
                observed=len({row["total_objects_n"] for row in wpu_frontier}) / max(1, len(n_values)),
                target=1.0,
                supporting_n=";".join(sorted({row["total_objects_n"] for row in wpu_frontier}, key=float)),
                supporting_b="",
                source="docs/experiments/pybullet_pareto_frontier.csv",
                interpretation=(
                    "WPU is on the accuracy-latency frontier only at the audited large-N point, not at small N."
                ),
                limitation="This separates matched-speedup evidence from full Pareto dominance.",
            )
        )

    energy_path = ROOT / "pybullet_system_energy_proxy.csv"
    if energy_path.exists():
        energy_rows = _read_rows(energy_path)
        best_proxy = max(energy_rows, key=lambda row: float(row["proxy_reduction"]))
        rows.append(
            _row(
                axis="screening_energy_proxy",
                status="screening_only",
                evidence_type="derived energy proxy",
                observed=float(best_proxy["proxy_reduction"]),
                target=0.95,
                supporting_n=float(best_proxy["total_objects"]),
                supporting_b=int(float(best_proxy["branch_count"])),
                source="docs/experiments/pybullet_system_energy_proxy.csv",
                interpretation="Energy proxy reduction is large in the same large-N sparse regime.",
                limitation="This is not wall-plug power, GPU power telemetry, or hardware energy measurement.",
            )
        )

    rows.append(
        _row(
            axis="real_power_or_sparse_kernel",
            status="not_measured",
            evidence_type="missing hardware measurement",
            observed=0.0,
            target=1.0,
            supporting_n="",
            supporting_b="",
            source="",
            interpretation="No committed experiment currently measures real power, hardware counters, or custom sparse kernels.",
            limitation="Hardware/chip/IP claims remain unsupported until this row changes.",
        )
    )
    return rows


def _row(
    *,
    axis: str,
    status: str,
    evidence_type: str,
    observed: float,
    target: float,
    supporting_n: object,
    supporting_b: object,
    source: str,
    interpretation: str,
    limitation: str,
) -> dict[str, object]:
    return {
        "axis": axis,
        "status": status,
        "evidence_type": evidence_type,
        "observed": round(observed, 6),
        "target": round(target, 6),
        "supporting_n": supporting_n,
        "supporting_b": supporting_b,
        "source": source,
        "interpretation": interpretation,
        "limitation": limitation,
    }


def _render_report(rows: list[dict[str, object]], *, korean: bool) -> str:
    positive = [row for row in rows if str(row["status"]).startswith("supported")]
    weak = [row for row in rows if row["status"] in {"weak_proxy", "screening_only"}]
    partial = [row for row in rows if str(row["status"]).startswith("partial")]
    missing = [row for row in rows if row["status"] == "not_measured"]
    best_branch = next(row for row in rows if row["axis"] == "branch_overlay_memory")
    peak_memory = next((row for row in rows if row["axis"] == "random_cuda_peak_memory"), None)

    if korean:
        title = "# PyBullet Systems Claim-Boundary Audit"
        intro = (
            "이 파생 감사는 P6 systems evidence를 claim별로 분리한다. 목표는 WPU의 "
            "systems 장점을 숨기는 것이 아니라, tensorization/latency proxy와 실제 hardware "
            "claim의 경계를 명확히 하는 것이다."
        )
        summary = [
            f"Supported proxy 축은 `{len(positive)}`개이고 partial trained 축은 `{len(partial)}`개다.",
            f"Branch-overlay memory proxy의 최대 reduction은 `{float(best_branch['observed']):.6f}`이다.",
        ]
        if peak_memory is not None:
            summary.append(
                f"CUDA peak-memory proxy는 최대 `{float(peak_memory['observed']):.6f}`라서 latency proxy보다 훨씬 약하다."
            )
        summary.append(f"Screening/weak proxy 축은 `{len(weak)}`개이고, real power/sparse-kernel 축은 아직 `{len(missing)}`개 미측정이다.")
        interpretation = (
            "현재 P6의 강한 주장은 pre-tensor working-set selection, branch-overlay memory accounting, "
            "large-N random-forward latency proxy, 그리고 제한된 trained matched-speedup이다. 반면 "
            "GPU peak memory, 실제 memory traffic, allocator telemetry, real power, custom sparse kernel "
            "evidence는 아직 부족하다. 따라서 WPU는 hardware result가 아니라 hardware로 가야 할 "
            "systems hypothesis로 써야 한다."
        )
    else:
        title = "# PyBullet Systems Claim-Boundary Audit"
        intro = (
            "This derived audit separates P6 systems evidence by claim type. The goal is not to hide "
            "WPU systems advantages, but to make the boundary between tensorization/latency proxies "
            "and hardware claims explicit."
        )
        summary = [
            f"Supported proxy axes: `{len(positive)}`; partial trained axes: `{len(partial)}`.",
            f"Maximum branch-overlay memory proxy reduction: `{float(best_branch['observed']):.6f}`.",
        ]
        if peak_memory is not None:
            summary.append(
                f"Maximum CUDA peak-memory proxy reduction is only `{float(peak_memory['observed']):.6f}`, much weaker than the latency proxy."
            )
        summary.append(f"Screening/weak proxy axes: `{len(weak)}`; real power/sparse-kernel axes still unmeasured: `{len(missing)}`.")
        interpretation = (
            "The strongest current P6 evidence supports pre-tensor working-set selection, branch-overlay "
            "memory accounting, large-N random-forward latency proxies, and limited trained matched-speedup. "
            "It does not yet support broad GPU-memory dominance, real memory-traffic reduction, power savings, "
            "or hardware/chip/IP claims."
        )

    table = [
        "| axis | status | observed | target | N | B | evidence | limitation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['axis']} | {row['status']} | {float(row['observed']):.6f} | "
            f"{float(row['target']):.6f} | {row['supporting_n']} | {row['supporting_b']} | "
            f"{row['evidence_type']} | {row['limitation']} |"
        )

    return (
        f"{title}\n\n"
        f"{intro}\n\n"
        "Source CSV: `docs/experiments/pybullet_system_claim_boundary.csv`\n\n"
        "## Summary\n\n"
        + "\n".join(f"- {item}" for item in summary)
        + "\n\n## Interpretation\n\n"
        + interpretation
        + "\n\n## Boundary Rows\n\n"
        + "\n".join(table)
        + "\n"
    )


def _summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [row for row in _read_rows(path) if row["row_type"] == "summary"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "axis",
        "status",
        "evidence_type",
        "observed",
        "target",
        "supporting_n",
        "supporting_b",
        "source",
        "interpretation",
        "limitation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
