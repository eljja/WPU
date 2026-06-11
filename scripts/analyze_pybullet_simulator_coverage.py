from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("docs/experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit simulator-backed WPU coverage across committed PyBullet evidence."
    )
    parser.add_argument("--out-csv", type=Path, default=ROOT / "pybullet_simulator_coverage.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "pybullet_simulator_coverage_results.md")
    parser.add_argument("--out-ko-md", type=Path, default=ROOT / "pybullet_simulator_coverage_results.ko.md")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    rows.append(_cup_7seed_row())
    n256 = _n256_screen_row()
    if n256 is not None:
        rows.append(n256)
    n512 = _n512_row()
    if n512 is not None:
        rows.append(n512)
    rows.append(_shift_row())
    rows.append(_closed_loop_row())
    rows.append(_objectification_row())
    rows.append(_system_profile_row(ROOT / "pybullet_system_profile.csv", "system_profile_cpu"))
    cuda = _system_profile_row(ROOT / "pybullet_system_profile_cuda.csv", "system_profile_cuda")
    if cuda is not None:
        rows.append(cuda)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    args.out_md.write_text(_render_markdown(rows, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_render_markdown(rows, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _cup_7seed_row() -> dict[str, object]:
    path = ROOT / "pybullet_cup_benchmark_7seed.csv"
    rows = _read_rows(path)
    return _coverage_row(
        axis="cup_7seed_baseline_complete",
        source=path,
        seed_count=len(_values(rows, "seed")),
        model_count=len(_values(rows, "model")),
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects_n"),
        horizon_max=1,
        branch_count_max=3,
        corruption_count=1,
        baseline_complete=_has_wpu_and_baseline(rows),
        notes="7-seed cup benchmark with WPU, graph, and token baselines; accuracy claims remain limited to this cup-task protocol.",
    )


def _n512_row() -> dict[str, object] | None:
    path = ROOT / "pybullet_cup_benchmark_n512.csv"
    if not path.exists():
        return None
    rows = _read_rows(path)
    return _coverage_row(
        axis="cup_n512_wpu_only_extension",
        source=path,
        seed_count=len(_values(rows, "seed")),
        model_count=len(_values(rows, "model")),
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects_n"),
        horizon_max=1,
        branch_count_max=3,
        corruption_count=1,
        baseline_complete=_has_wpu_and_baseline(rows),
        notes="Large-background WPU-only extension. The graph-transformer baseline did not finish under the attempted 20-minute run, so this is systems feasibility evidence, not matched baseline superiority evidence.",
    )


def _n256_screen_row() -> dict[str, object] | None:
    path = ROOT / "pybullet_cup_benchmark_n256_baseline_screen.csv"
    if not path.exists():
        return None
    rows = _read_rows(path)
    return _coverage_row(
        axis="cup_n256_baseline_screen",
        source=path,
        seed_count=len(_values(rows, "seed")),
        model_count=len(_values(rows, "model")),
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects_n"),
        horizon_max=1,
        branch_count_max=3,
        corruption_count=1,
        baseline_complete=_has_wpu_and_baseline(rows),
        notes="Low-training 5-seed N_bg=256 screen with WPU, graph, and token baselines; useful for matched large-N feasibility, not for strong accuracy-superiority claims.",
    )


def _shift_row() -> dict[str, object]:
    path = ROOT / "pybullet_shift_generalization.csv"
    rows = _rows_of_type(_read_rows(path), "summary")
    mechanisms = _values(rows, "eval_mechanism")
    shifted = sorted(mechanism for mechanism in mechanisms if mechanism != "nominal")
    return _coverage_row(
        axis="mechanism_shift_generalization",
        source=path,
        seed_count=_seed_count_from_summary(rows),
        model_count=len(_values(rows, "model")),
        mechanism_count=len(mechanisms),
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects_n"),
        horizon_max=1,
        branch_count_max=3,
        corruption_count=1,
        baseline_complete=_has_wpu_and_baseline(rows),
        notes=f"Nominal plus {len(shifted)} shifted mechanism families: {', '.join(shifted)}.",
    )


def _closed_loop_row() -> dict[str, object]:
    path = ROOT / "pybullet_closed_loop_rollout.csv"
    rows = _read_rows(path)
    return _coverage_row(
        axis="closed_loop_rollout",
        source=path,
        seed_count=len(_values(rows, "seed")),
        model_count=len(_values(rows, "model")),
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "background_objects") + 5,
        horizon_max=_max_int(rows, "horizon"),
        branch_count_max=3,
        corruption_count=1,
        baseline_complete=_has_wpu_and_baseline(rows),
        notes="Multi-step delta-overlay rollout diagnostic; finite-corrected safety is tracked in the separate state-integrity audit.",
    )


def _objectification_row() -> dict[str, object]:
    path = ROOT / "pybullet_objectification_quality.csv"
    all_rows = _read_rows(path)
    rows = _rows_of_type(all_rows, "summary")
    return _coverage_row(
        axis="objectification_quality",
        source=path,
        seed_count=_seed_count_from_all_rows(all_rows),
        model_count=0,
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects"),
        horizon_max=1,
        branch_count_max=3,
        corruption_count=len(_values(rows, "corruption")),
        baseline_complete=True,
        notes="Objectification-contract audit over clean and corrupted simulator-derived state; it measures input quality, not model superiority.",
    )


def _system_profile_row(path: Path, axis: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    all_rows = _read_rows(path)
    rows = _rows_of_type(all_rows, "summary")
    return _coverage_row(
        axis=axis,
        source=path,
        seed_count=_seed_count_from_all_rows(all_rows),
        model_count=0,
        mechanism_count=1,
        background_min=_min_int(rows, "background_objects"),
        background_max=_max_int(rows, "background_objects"),
        total_n_max=_max_int(rows, "total_objects"),
        horizon_max=1,
        branch_count_max=_max_int(rows, "branch_count"),
        corruption_count=1,
        baseline_complete=True,
        notes="Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.",
    )


def _coverage_row(
    *,
    axis: str,
    source: Path,
    seed_count: int,
    model_count: int,
    mechanism_count: int,
    background_min: int,
    background_max: int,
    total_n_max: int,
    horizon_max: int,
    branch_count_max: int,
    corruption_count: int,
    baseline_complete: bool,
    notes: str,
) -> dict[str, object]:
    return {
        "axis": axis,
        "source": source.as_posix(),
        "seed_count": seed_count,
        "model_count": model_count,
        "mechanism_count": mechanism_count,
        "background_min": background_min,
        "background_max": background_max,
        "total_n_max": total_n_max,
        "horizon_max": horizon_max,
        "branch_count_max": branch_count_max,
        "corruption_count": corruption_count,
        "baseline_complete": baseline_complete,
        "notes": notes,
    }


def _render_markdown(rows: list[dict[str, object]], *, korean: bool) -> str:
    title = "# PyBullet Simulator Coverage Audit" if not korean else "# PyBullet 시뮬레이터 Coverage Audit"
    intro = (
        "This audit separates simulator grounding breadth from superiority claims. "
        "A row can increase coverage while still being unusable as a matched-baseline "
        "accuracy claim when `baseline_complete=False`."
        if not korean
        else "이 audit는 simulator grounding의 범위와 우월성 주장을 분리한다. "
        "`baseline_complete=False`인 행은 coverage를 넓히더라도 matched-baseline accuracy claim으로 사용할 수 없다."
    )
    table_header = (
        "| Axis | Seeds | Models | Mechanisms | N_bg max | N max | Horizon max | Corruptions | Baseline complete |"
        if not korean
        else "| 축 | Seeds | Models | Mechanisms | N_bg max | N max | Horizon max | Corruptions | Baseline complete |"
    )
    lines = [
        title,
        "",
        intro,
        "",
        "Source CSVs:",
    ]
    for source in sorted({str(row["source"]) for row in rows}):
        lines.append(f"- `{source}`")
    lines.extend(["", table_header, "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in rows:
        lines.append(
            f"| {row['axis']} | {row['seed_count']} | {row['model_count']} | "
            f"{row['mechanism_count']} | {row['background_max']} | {row['total_n_max']} | "
            f"{row['horizon_max']} | {row['corruption_count']} | {row['baseline_complete']} |"
        )
    if korean:
        lines.extend(
            [
                "",
                "## 해석",
                "",
                "- 현재 PyBullet evidence는 cup benchmark, mechanism shift, closed-loop rollout, objectification corruption, CPU/CUDA systems profile까지 포함한다.",
                "- `cup_n256_baseline_screen`은 N_bg=256, total N=261에서 WPU, graph, token baseline을 모두 완료한 matched large-N screen이지만, 저훈련 설정이므로 강한 accuracy superiority claim에는 쓰지 않는다.",
                "- `cup_n512_wpu_only_extension`은 N_bg=512, total N=517까지 WPU가 실행된다는 evidence지만, dense graph baseline이 같은 protocol에서 완료되지 않았으므로 accuracy superiority evidence가 아니다.",
                "- P3의 다음 병목은 단일 PyBullet cup family를 넘어서는 mechanism 다양성, baseline-complete large-N comparison, 그리고 perception/state adapter를 포함한 end-to-end objectification이다.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Current PyBullet evidence covers cup prediction, mechanism shift, closed-loop rollout, objectification corruption, and CPU/CUDA systems profiling.",
                "- `cup_n256_baseline_screen` completes WPU, graph, and token baselines at N_bg=256 and total N=261, but it is a low-training screen and should not be used as a strong accuracy-superiority claim.",
                "- `cup_n512_wpu_only_extension` shows WPU execution at N_bg=512 and total N=517, but it is not accuracy-superiority evidence because the dense graph baseline did not complete under the same protocol.",
                "- The next P3 bottleneck is not another small cup run; it is mechanism diversity, baseline-complete large-N comparison, and end-to-end objectification through a perception/state adapter.",
            ]
        )
    lines.extend(["", "## Row Notes" if not korean else "## 행별 메모", ""])
    for row in rows:
        lines.append(f"- `{row['axis']}`: {row['notes']}")
    return "\n".join(lines) + "\n"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_of_type(rows: list[dict[str, str]], row_type: str) -> list[dict[str, str]]:
    if not rows or "row_type" not in rows[0]:
        return rows
    typed = [row for row in rows if row.get("row_type") == row_type]
    return typed or rows


def _values(rows: list[dict[str, str]], column: str) -> set[str]:
    return {row[column] for row in rows if row.get(column, "") != ""}


def _min_int(rows: list[dict[str, str]], column: str) -> int:
    return min(int(float(row[column])) for row in rows if row.get(column, "") != "")


def _max_int(rows: list[dict[str, str]], column: str) -> int:
    return max(int(float(row[column])) for row in rows if row.get(column, "") != "")


def _has_wpu_and_baseline(rows: list[dict[str, str]]) -> bool:
    models = _values(rows, "model")
    has_wpu = any(model.startswith("wpu-") for model in models)
    has_baseline = any(not model.startswith("wpu-") for model in models)
    return has_wpu and has_baseline


def _seed_count_from_summary(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 0
    if "seed_count" in rows[0]:
        return max(int(float(row["seed_count"])) for row in rows if row.get("seed_count", "") != "")
    return len(_values(rows, "seed"))


def _seed_count_from_all_rows(rows: list[dict[str, str]]) -> int:
    seeds = {row["seed"] for row in rows if row.get("seed", "") not in {"", "all"}}
    return len(seeds)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
