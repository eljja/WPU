from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute screening-only energy proxies from WPU system-profile CSVs."
    )
    parser.add_argument("--cpu-input", type=Path, default=Path("docs/experiments/pybullet_system_profile.csv"))
    parser.add_argument("--cuda-input", type=Path, default=Path("docs/experiments/pybullet_system_profile_cuda.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_system_energy_proxy.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_system_energy_proxy_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_system_energy_proxy_results.ko.md"),
    )
    args = parser.parse_args()

    rows = []
    rows.extend(_cpu_proxy(args.cpu_input))
    if args.cuda_input.exists():
        rows.extend(_cuda_proxy(args.cuda_input))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args.cpu_input, args.cuda_input, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args.cpu_input, args.cuda_input, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("row_type") == "summary"]


def _cpu_proxy(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in _summary_rows(path):
        full = float(row["full_tensorize_ms"]) * float(row["full_tensor_bytes"])
        selected = float(row["selected_tensorize_ms"]) * float(row["selected_tensor_bytes"])
        rows.append(
            {
                "profile": "cpu_tensorization",
                "background_objects": int(float(row["background_objects"])),
                "total_objects": round(float(row["total_objects"]), 6),
                "branch_count": int(float(row["branch_count"])),
                "full_proxy": round(full, 6),
                "selected_proxy": round(selected, 6),
                "proxy_reduction": round(_reduction(full, selected), 6),
                "latency_reduction": round(float(row["tensorize_latency_reduction"]), 6),
                "memory_reduction": round(float(row["tensor_byte_reduction"]), 6),
            }
        )
    return rows


def _cuda_proxy(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in _summary_rows(path):
        full = float(row["full_graph_forward_ms"]) * float(row["full_graph_peak_memory_bytes"])
        selected = float(row["selected_sparse_forward_ms"]) * float(row["selected_sparse_peak_memory_bytes"])
        rows.append(
            {
                "profile": "cuda_forward_screening",
                "background_objects": int(float(row["background_objects"])),
                "total_objects": round(float(row["total_objects"]), 6),
                "branch_count": int(float(row["branch_count"])),
                "full_proxy": round(full, 6),
                "selected_proxy": round(selected, 6),
                "proxy_reduction": round(_reduction(full, selected), 6),
                "latency_reduction": round(float(row["sparse_forward_latency_reduction"]), 6),
                "memory_reduction": round(float(row["sparse_peak_memory_reduction"]), 6),
            }
        )
    return rows


def _reduction(full: float, selected: float) -> float:
    if full <= 0.0:
        return 0.0
    return max(0.0, 1.0 - selected / full)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], cpu_input: Path, cuda_input: Path, output_csv: Path, *, korean: bool) -> str:
    best = max(rows, key=lambda row: float(row["proxy_reduction"]))
    cuda_rows = [row for row in rows if row["profile"] == "cuda_forward_screening"]
    best_cuda = max(cuda_rows, key=lambda row: float(row["proxy_reduction"])) if cuda_rows else None
    if korean:
        title = "# PyBullet System Energy Proxy"
        intro = (
            "이 문서는 실제 전력 측정이 아니라 systems profile에서 계산한 screening-only "
            "energy proxy다. 논문에서는 hardware evidence가 아니라 다음 측정 위치를 정하는 "
            "보조 지표로만 사용해야 한다."
        )
        interpretation = [
            f"최대 proxy reduction은 `{float(best['proxy_reduction']):.6f}`이며 profile `{best['profile']}`, N `{float(best['total_objects']):.1f}`, B `{best['branch_count']}`에서 발생한다.",
        ]
        if best_cuda is not None:
            interpretation.append(
                f"CUDA forward screening proxy의 최대 reduction은 `{float(best_cuda['proxy_reduction']):.6f}`이다."
            )
        interpretation.append("이 값은 전력계 측정, GPU power telemetry, sparse kernel counter를 대체하지 못한다.")
    else:
        title = "# PyBullet System Energy Proxy"
        intro = (
            "This report computes screening-only energy proxies from systems profiles. "
            "It is not a power measurement and must only be used to choose where to run "
            "real energy and sparse-kernel profiling."
        )
        interpretation = [
            f"The maximum proxy reduction is `{float(best['proxy_reduction']):.6f}` for profile `{best['profile']}` at N `{float(best['total_objects']):.1f}`, B `{best['branch_count']}`.",
        ]
        if best_cuda is not None:
            interpretation.append(
                f"The maximum CUDA forward screening proxy reduction is `{float(best_cuda['proxy_reduction']):.6f}`."
            )
        interpretation.append("This does not replace wall-plug power, GPU telemetry, or sparse-kernel counters.")
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
        "",
        f"- `{cpu_input.as_posix()}`",
        f"- `{cuda_input.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{output_csv.as_posix()}`",
        "",
        "| profile | N | B | proxy reduction | latency reduction | memory reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {float(row['total_objects']):.1f} | {row['branch_count']} | "
            f"{float(row['proxy_reduction']):.6f} | {float(row['latency_reduction']):.6f} | "
            f"{float(row['memory_reduction']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
