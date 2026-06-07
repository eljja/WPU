from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.analyze_matched_accuracy_speedup import _matched_rows, _read_rows, _summarize_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep accuracy tolerances for PyBullet matched-or-better speedup claims.")
    parser.add_argument("--input", type=Path, default=Path("docs/experiments/pybullet_matched_baseline_benchmark.csv"))
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.075, 0.10, 0.125])
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/pybullet_matched_speedup_tolerance.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/pybullet_matched_speedup_tolerance_results.md"))
    parser.add_argument(
        "--out-ko-md",
        type=Path,
        default=Path("docs/experiments/pybullet_matched_speedup_tolerance_results.ko.md"),
    )
    args = parser.parse_args()

    summary_rows = _summarize_benchmark(_read_rows(args.input))
    rows = []
    for tolerance in args.tolerances:
        for row in _matched_rows(summary_rows, tolerance):
            row = dict(row)
            row["accuracy_tolerance"] = round(float(tolerance), 6)
            rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render(rows, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render(rows, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict[str, object]], args: argparse.Namespace, *, korean: bool) -> str:
    strict_rows = [row for row in rows if float(row["accuracy_tolerance"]) == 0.03]
    first_large_n_match = None
    for row in rows:
        if int(row["total_objects_n"]) >= 100 and row["matched_accuracy"] is True:
            if first_large_n_match is None or float(row["accuracy_tolerance"]) < float(first_large_n_match["accuracy_tolerance"]):
                first_large_n_match = row
    if korean:
        title = "# PyBullet Matched-or-Better Speedup Tolerance Sweep"
        intro = "이 문서는 matched-or-better speedup 주장이 accuracy tolerance에 얼마나 민감한지 보여준다."
        interpretation = [
            "`matched`는 WPU accuracy가 baseline보다 tolerance 이상 낮지 않다는 뜻이다. WPU가 더 정확하면 tolerance 0에서도 통과한다.",
        ]
        if first_large_n_match is not None:
            if float(first_large_n_match["accuracy_gap"]) >= 0.0:
                interpretation.append(
                    f"N={int(first_large_n_match['total_objects_n'])} large-N 조건은 WPU accuracy가 더 높기 때문에 tolerance `0.000`에서도 matched-or-better로 통과한다."
                )
            else:
                interpretation.append(
                    f"N={int(first_large_n_match['total_objects_n'])} large-N 조건은 tolerance `{float(first_large_n_match['accuracy_tolerance']):.3f}` 이상에서야 matched-or-better로 간주된다."
                )
        interpretation.append("따라서 현재 P6는 best-accuracy non-WPU baseline 대비 large-N matched-or-better speedup은 보이지만, 모든 baseline에 대한 Pareto 우월성이나 에너지 증명은 아니다.")
    else:
        title = "# PyBullet Matched-or-Better Speedup Tolerance Sweep"
        intro = "This report shows how sensitive matched-or-better speedup claims are to the chosen accuracy tolerance."
        interpretation = [
            "`Matched` means WPU accuracy is not below the baseline by more than the tolerance. If WPU is more accurate, it passes even at tolerance 0.",
        ]
        if first_large_n_match is not None:
            if float(first_large_n_match["accuracy_gap"]) >= 0.0:
                interpretation.append(
                    f"The N={int(first_large_n_match['total_objects_n'])} large-N point passes even at tolerance `0.000` because WPU accuracy is higher."
                )
            else:
                interpretation.append(
                    f"The N={int(first_large_n_match['total_objects_n'])} large-N point only becomes matched-or-better at tolerance `{float(first_large_n_match['accuracy_tolerance']):.3f}` or above."
                )
        interpretation.append("P6 therefore shows large-N matched-or-better speedup against the best-accuracy non-WPU baseline, but not Pareto dominance over every baseline or real energy evidence.")
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSV:",
        "",
        f"- `{args.input.as_posix()}`",
        "",
        "Derived CSV:",
        "",
        f"- `{args.out_csv.as_posix()}`",
        "",
        "| tolerance | N | matched-or-better | acc gap | speedup | WPU acc | baseline acc |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    del strict_rows
    for row in rows:
        lines.append(
            f"| {float(row['accuracy_tolerance']):.3f} | {int(row['total_objects_n'])} | "
            f"{row['matched_accuracy']} | {float(row['accuracy_gap']):.6f} | "
        f"{float(row['matched_speedup']):.6f} | {float(row['wpu_accuracy']):.6f} | "
            f"{float(row['baseline_accuracy']):.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
