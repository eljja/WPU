from __future__ import annotations

import re
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
BACKTICK_PATH = re.compile(r"`([^`]+\.(?:md|tex|pdf|csv|png|svg|py|ps1|docx))`")
SOURCE_CSV = re.compile(r"Source CSV:\s*`([^`]+)`")
SOURCE_CSVS_SECTION = re.compile(r"Source CSVs:\s*\n((?:\s*-\s*`[^`]+`\s*\n?)+)")
SOURCE_CSV_BULLET = re.compile(r"-\s*`([^`]+)`")
LATEX_GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*])?\{([^}]+)\}")
LATEX_CITE = re.compile(r"\\cite\{([^}]+)\}")
LATEX_BIBITEM = re.compile(r"\\bibitem\{([^}]+)\}")
MARKDOWN_TABLE_ROW = re.compile(r"^\|\s*(.*?)\s*\|$")


def _is_external(target: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target))


def _should_check_backtick_path(target: str) -> bool:
    normalized = target.replace("\\", "/")
    if normalized.startswith("artifacts/"):
        return False
    return normalized.startswith(
        (
            "docs/",
            "scripts/",
            "tests/",
            "wpu/",
            "README",
        )
    )


def _is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def _markdown_table_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MARKDOWN_TABLE_ROW.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _round6(value: str) -> str:
    return f"{float(value):.6f}"


def _round_decimals(value: str, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def test_markdown_local_references_exist() -> None:
    missing: list[str] = []
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", ".venv", "artifacts"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")

        for match in LOCAL_LINK.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith("#") or _is_external(target):
                continue
            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

        for match in BACKTICK_PATH.finditer(text):
            target = match.group(1)
            if not _should_check_backtick_path(target):
                continue
            candidate = (ROOT / target.replace("\\", "/")).resolve()
            if not candidate.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing local documentation references:\n" + "\n".join(missing)


def test_experiment_source_csv_references_are_nonempty() -> None:
    issues: list[str] = []
    for path in (ROOT / "docs" / "experiments").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        source_targets = [match.group(1) for match in SOURCE_CSV.finditer(text)]
        for section in SOURCE_CSVS_SECTION.finditer(text):
            source_targets.extend(match.group(1) for match in SOURCE_CSV_BULLET.finditer(section.group(1)))

        for target in source_targets:
            if target.replace("\\", "/").startswith("artifacts/"):
                issues.append(f"{path.relative_to(ROOT)} -> uncommitted artifact source {target}")
                continue
            if not target.replace("\\", "/").startswith("docs/experiments/"):
                issues.append(f"{path.relative_to(ROOT)} -> source CSV outside docs/experiments {target}")
                continue
            candidate = ROOT / target.replace("\\", "/")
            if not candidate.exists():
                issues.append(f"{path.relative_to(ROOT)} -> missing {target}")
                continue
            if not _is_git_tracked(candidate):
                issues.append(f"{path.relative_to(ROOT)} -> untracked source CSV {target}")
                continue
            with candidate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            if len(rows) < 2:
                issues.append(f"{path.relative_to(ROOT)} -> empty {target}")

    assert not issues, "Invalid experiment Source CSV references:\n" + "\n".join(issues)


def test_current_v2_evidence_reports_declare_source_csvs() -> None:
    readme = (ROOT / "docs" / "experiments" / "README.md").read_text(encoding="utf-8")
    current_section = readme.split("Historical or preliminary reports:", 1)[0]
    referenced_reports = re.findall(r"`([^`]+\.md)`", current_section)

    missing: list[str] = []
    for report in referenced_reports:
        if not report.startswith("wpu_v2_"):
            continue
        if not (report.endswith("_results.md") or report.endswith("_analysis.md")):
            continue
        path = ROOT / "docs" / "experiments" / report
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "Source CSV" not in text:
            missing.append(report)

    assert not missing, "Current v2 evidence reports must declare Source CSVs:\n" + "\n".join(missing)


def test_selector_report_tables_match_summary_csvs() -> None:
    cases = [
        (
            ROOT / "docs" / "experiments" / "wpu_v2_composition_variant_selector_results.md",
            ROOT / "docs" / "experiments" / "wpu_v2_composition_variant_selector_summary.csv",
            "selector criterion",
            "criterion",
            "delta loss vs static learned",
            "delta_loss_vs_static_learned",
            {
                "lowest other-seed loss": "lowest_other_seed_loss",
                "highest other-seed accuracy": "highest_other_seed_accuracy",
            },
        ),
        (
            ROOT / "docs" / "experiments" / "wpu_v2_diagnostic_variant_selector_results.md",
            ROOT / "docs" / "experiments" / "wpu_v2_diagnostic_variant_selector_summary.csv",
            "criterion",
            "criterion",
            "delta loss vs static base",
            "delta_loss_vs_static_base",
            {
                "min CV delta": "min_cv_delta",
                "max CV win then delta": "max_cv_win_then_delta",
                "best train loss delta": "best_train_loss_delta",
            },
        ),
    ]
    issues: list[str] = []

    for report_path, csv_path, table_criterion, csv_criterion, table_delta, csv_delta, criterion_labels in cases:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            csv_rows = {
                (row["causal_k"], row[csv_criterion]): row
                for row in csv.DictReader(handle)
            }

        table_rows = _markdown_table_rows(report_path)
        header_index = next(
            index for index, row in enumerate(table_rows)
            if row[:2] == ["K", table_criterion]
        )
        headers = table_rows[header_index]
        for row in table_rows[header_index + 1:]:
            if len(row) != len(headers) or not row[0].isdigit():
                break
            values = dict(zip(headers, row))
            criterion_key = criterion_labels.get(values[table_criterion], values[table_criterion])
            key = (values["K"], criterion_key)
            source = csv_rows.get(key)
            if source is None:
                issues.append(f"{report_path.relative_to(ROOT)} -> table row missing in CSV {key}")
                continue
            comparisons = [
                ("loss", "loss"),
                ("accuracy", "accuracy"),
                (table_delta, csv_delta),
                ("excess over generated oracle", "excess_over_generated_oracle"),
            ]
            for table_column, csv_column in comparisons:
                if _round6(values[table_column]) != _round6(source[csv_column]):
                    issues.append(
                        f"{report_path.relative_to(ROOT)} -> {key} {table_column} "
                        f"table={values[table_column]} csv={source[csv_column]}"
                    )

    assert not issues, "Report table values do not match source CSVs:\n" + "\n".join(issues)


def test_clipped_diagnostic_report_table_matches_summary_csv() -> None:
    report_path = ROOT / "docs" / "experiments" / "wpu_v2_clipped_diagnostic_probe_results.md"
    csv_path = ROOT / "docs" / "experiments" / "wpu_v2_clipped_diagnostic_probe_summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = {
            _round_decimals(row["residual_clip"], 2): row
            for row in csv.DictReader(handle)
            if _round_decimals(row["compute_cost"], 2) == "0.05"
        }

    table_rows = _markdown_table_rows(report_path)
    header_index = next(
        index for index, row in enumerate(table_rows)
        if row[:2] == ["residual clip", "regret pearson"]
    )
    headers = table_rows[header_index]
    comparisons = [
        ("regret pearson", "regret_pearson"),
        ("regret R2", "regret_r2"),
        ("dense rate", "dense_rate"),
        ("policy loss", "policy_loss"),
        ("loss delta", "policy_delta_vs_sparse"),
        ("oracle excess", "policy_excess_over_oracle"),
    ]
    issues: list[str] = []

    for row in table_rows[header_index + 1:]:
        if len(row) != len(headers) or not row[0].replace(".", "", 1).isdigit():
            break
        values = dict(zip(headers, row))
        source = csv_rows.get(values["residual clip"])
        if source is None:
            issues.append(f"{report_path.relative_to(ROOT)} -> table row missing in CSV {values['residual clip']}")
            continue
        for table_column, csv_column in comparisons:
            if _round_decimals(values[table_column], 3) != _round_decimals(source[csv_column], 3):
                issues.append(
                    f"{report_path.relative_to(ROOT)} -> clip={values['residual clip']} {table_column} "
                    f"table={values[table_column]} csv={source[csv_column]}"
                )

    assert not issues, "Clipped diagnostic report table values do not match source CSV:\n" + "\n".join(issues)


def test_latex_graphics_and_citations_resolve() -> None:
    issues: list[str] = []
    for path in ROOT.rglob("*.tex"):
        if any(part in {".git", ".venv", "artifacts"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")

        for match in LATEX_GRAPHICS.finditer(text):
            target = match.group(1)
            if _is_external(target):
                continue
            candidates = [
                (ROOT / target.replace("\\", "/")).resolve(),
                (path.parent / target.replace("\\", "/")).resolve(),
            ]
            if not Path(target).suffix:
                candidates.extend(
                    candidate.with_suffix(suffix)
                    for candidate in list(candidates)
                    for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".svg")
                )
            if not any(candidate.exists() for candidate in candidates):
                issues.append(f"{path.relative_to(ROOT)} -> missing graphic {target}")

        cited_keys = {
            key.strip()
            for match in LATEX_CITE.finditer(text)
            for key in match.group(1).split(",")
            if key.strip()
        }
        bib_keys = {match.group(1) for match in LATEX_BIBITEM.finditer(text)}
        for key in sorted(cited_keys - bib_keys):
            issues.append(f"{path.relative_to(ROOT)} -> missing bibitem {key}")

    assert not issues, "Invalid LaTeX references:\n" + "\n".join(issues)
