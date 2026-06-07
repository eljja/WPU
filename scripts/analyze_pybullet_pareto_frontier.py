from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


DEFAULT_INPUT = Path("docs/experiments/pybullet_matched_baseline_benchmark.csv")
DEFAULT_OUT_CSV = Path("docs/experiments/pybullet_pareto_frontier.csv")
DEFAULT_OUT_MD = Path("docs/experiments/pybullet_pareto_frontier_results.md")
DEFAULT_OUT_KO_MD = Path("docs/experiments/pybullet_pareto_frontier_results.ko.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit accuracy-latency Pareto frontiers for PyBullet models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-ko-md", type=Path, default=DEFAULT_OUT_KO_MD)
    args = parser.parse_args()

    rows = _summarize(_read_rows(args.input))
    frontier_rows = _frontier_rows(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, frontier_rows)
    args.out_md.write_text(_render_markdown(frontier_rows, args.input, args.out_csv, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(frontier_rows, args.input, args.out_csv, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((int(float(row["total_objects_n"])), row["model"]), []).append(row)
    out: list[dict[str, object]] = []
    for (total_n, model), group in sorted(grouped.items()):
        out.append(
            {
                "total_objects_n": total_n,
                "model": model,
                "is_wpu": model.startswith("wpu-"),
                "branch_accuracy": round(_mean(group, "branch_accuracy"), 6),
                "ms_per_sample_forward": round(_mean(group, "ms_per_sample_forward"), 6),
                "cuda_peak_mb": round(_mean(group, "cuda_peak_mb"), 6),
                "seed_count": len({row["seed"] for row in group}),
            }
        )
    return out


def _frontier_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_n = sorted({int(row["total_objects_n"]) for row in rows})
    for total_n in by_n:
        group = [row for row in rows if int(row["total_objects_n"]) == total_n]
        for row in group:
            dominators = [
                other
                for other in group
                if other is not row
                and float(other["branch_accuracy"]) >= float(row["branch_accuracy"])
                and float(other["ms_per_sample_forward"]) <= float(row["ms_per_sample_forward"])
                and (
                    float(other["branch_accuracy"]) > float(row["branch_accuracy"])
                    or float(other["ms_per_sample_forward"]) < float(row["ms_per_sample_forward"])
                )
            ]
            frontier = not dominators
            best_dominator = None
            if dominators:
                best_dominator = max(
                    dominators,
                    key=lambda item: (
                        float(item["branch_accuracy"]) - float(row["branch_accuracy"]),
                        float(row["ms_per_sample_forward"]) - float(item["ms_per_sample_forward"]),
                    ),
                )
            out.append(
                {
                    **row,
                    "pareto_frontier": frontier,
                    "dominated_by": "" if best_dominator is None else best_dominator["model"],
                    "dominator_accuracy": "" if best_dominator is None else best_dominator["branch_accuracy"],
                    "dominator_ms_per_sample": "" if best_dominator is None else best_dominator["ms_per_sample_forward"],
                }
            )
    return out


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, object]], source: Path, derived: Path, *, korean: bool) -> str:
    wpu_rows = [row for row in rows if row["is_wpu"]]
    wpu_frontier = [row for row in wpu_rows if row["pareto_frontier"]]
    frontier_by_n = sorted({int(row["total_objects_n"]) for row in wpu_frontier})
    if korean:
        title = "# PyBullet Pareto Frontier Audit"
        intro = "이 보고서는 parameter-matched PyBullet benchmark에서 accuracy-latency Pareto frontier를 계산한다."
        interpretation = [
            f"WPU가 Pareto frontier에 있는 N은 `{frontier_by_n}`이다.",
            "이 audit은 best-accuracy baseline 대비 speedup과 다른 질문을 다룬다. WPU가 어떤 baseline보다 더 정확하고 빠른 지점이 있어도, 더 낮은 accuracy에서 훨씬 빠른 token baseline이 있으면 전체 Pareto dominance는 아니다.",
            "따라서 P6 주장은 large-N matched-or-better evidence와 Pareto-frontier evidence를 분리해야 한다.",
        ]
    else:
        title = "# PyBullet Pareto Frontier Audit"
        intro = "This report computes accuracy-latency Pareto frontiers for the parameter-matched PyBullet benchmark."
        interpretation = [
            f"WPU lies on the Pareto frontier at N values `{frontier_by_n}`.",
            "This audit asks a different question from speedup against the best-accuracy baseline. WPU can be more accurate and faster than one baseline while still not Pareto-dominating a faster, lower-accuracy token baseline.",
            "P6 claims must therefore separate large-N matched-or-better evidence from full Pareto-frontier evidence.",
        ]
    table_rows = [
        row for row in rows
        if row["is_wpu"] or row["pareto_frontier"] or row["dominated_by"]
    ]
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSV:",
        "",
        f"- `{source.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{derived.as_posix()}`",
        "",
        "| N | model | WPU | accuracy | ms/sample | Pareto | dominated by |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['total_objects_n']} | `{row['model']}` | {row['is_wpu']} | "
            f"{float(row['branch_accuracy']):.6f} | {float(row['ms_per_sample_forward']):.6f} | "
            f"{row['pareto_frontier']} | `{row['dominated_by']}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
