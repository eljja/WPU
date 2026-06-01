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


def _should_check_root_backtick_path(target: str) -> bool:
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


def _resolve_backtick_path(source_path: Path, target: str) -> Path | None:
    normalized = target.replace("\\", "/")
    if normalized.startswith("artifacts/") or _is_external(normalized):
        return None
    if _should_check_root_backtick_path(normalized):
        return (ROOT / normalized).resolve()
    return (source_path.parent / normalized).resolve()


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


def _assert_table_matches_csv(
    report_path: Path,
    csv_path: Path,
    table_header_prefix: list[str],
    csv_key_columns: list[str],
    table_key_columns: list[str],
    comparisons: list[tuple[str, str, int]],
    key_value_maps: dict[str, dict[str, str]] | None = None,
    key_rounding: dict[str, int] | None = None,
    csv_row_filter: dict[str, str] | None = None,
) -> list[str]:
    key_value_maps = key_value_maps or {}
    key_rounding = key_rounding or {}
    csv_row_filter = csv_row_filter or {}

    def normalize_key_value(column: str, value: str) -> str:
        mapped = key_value_maps.get(column, {}).get(value, value)
        if column in key_rounding:
            return _round_decimals(mapped, key_rounding[column])
        return mapped

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = {}
        for row in csv.DictReader(handle):
            if any(_round_decimals(row[column], 6) != _round_decimals(value, 6) for column, value in csv_row_filter.items()):
                continue
            key = tuple(normalize_key_value(column, row[column]) for column in csv_key_columns)
            csv_rows[key] = row

    table_rows = _markdown_table_rows(report_path)
    header_index = next(
        index for index, row in enumerate(table_rows)
        if row[: len(table_header_prefix)] == table_header_prefix
    )
    headers = table_rows[header_index]
    issues: list[str] = []
    for row in table_rows[header_index + 1:]:
        if len(row) != len(headers):
            break
        values = dict(zip(headers, row))
        if not values[table_key_columns[0]].replace(".", "", 1).isdigit():
            break
        key = tuple(
            normalize_key_value(column, values[column])
            for column in table_key_columns
        )
        source = csv_rows.get(key)
        if source is None:
            issues.append(f"{report_path.relative_to(ROOT)} -> table row missing in CSV {key}")
            continue
        for table_column, csv_column, digits in comparisons:
            if _round_decimals(values[table_column], digits) != _round_decimals(source[csv_column], digits):
                issues.append(
                    f"{report_path.relative_to(ROOT)} -> {key} {table_column} "
                    f"table={values[table_column]} csv={source[csv_column]}"
                )
    return issues


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
            candidate = _resolve_backtick_path(path, target)
            if candidate is None:
                continue
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


def test_claim_ledgers_have_matching_claim_ids() -> None:
    expected_ids = [f"C{index}" for index in range(1, 11)]
    issues: list[str] = []
    for ledger_path in (ROOT / "docs" / "claims.md", ROOT / "docs" / "claims.ko.md"):
        ids = [
            row[0]
            for row in _markdown_table_rows(ledger_path)
            if row and re.fullmatch(r"C\d+", row[0])
        ]
        if ids != expected_ids:
            issues.append(f"{ledger_path.relative_to(ROOT)} ids={ids}")

    assert not issues, "Claim ledgers must keep matching C1-C10 rows:\n" + "\n".join(issues)


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
        issues.extend(
            _assert_table_matches_csv(
                report_path=report_path,
                csv_path=csv_path,
                table_header_prefix=["K", table_criterion],
                csv_key_columns=["causal_k", csv_criterion],
                table_key_columns=["K", table_criterion],
                comparisons=[
                    ("loss", "loss", 6),
                    ("accuracy", "accuracy", 6),
                    (table_delta, csv_delta, 6),
                    ("excess over generated oracle", "excess_over_generated_oracle", 6),
                ],
                key_value_maps={table_criterion: criterion_labels},
            )
        )

    assert not issues, "Report table values do not match source CSVs:\n" + "\n".join(issues)


def test_clipped_diagnostic_report_table_matches_summary_csv() -> None:
    report_path = ROOT / "docs" / "experiments" / "wpu_v2_clipped_diagnostic_probe_results.md"
    csv_path = ROOT / "docs" / "experiments" / "wpu_v2_clipped_diagnostic_probe_summary.csv"
    issues = _assert_table_matches_csv(
        report_path=report_path,
        csv_path=csv_path,
        table_header_prefix=["residual clip", "regret pearson"],
        csv_key_columns=["residual_clip"],
        table_key_columns=["residual clip"],
        comparisons=[
            ("regret pearson", "regret_pearson", 3),
            ("regret R2", "regret_r2", 3),
            ("dense rate", "dense_rate", 3),
            ("policy loss", "policy_loss", 3),
            ("loss delta", "policy_delta_vs_sparse", 3),
            ("oracle excess", "policy_excess_over_oracle", 3),
        ],
        csv_row_filter={"compute_cost": "0.05"},
        key_rounding={"residual_clip": 2, "residual clip": 2},
    )

    assert not issues, "Clipped diagnostic report table values do not match source CSV:\n" + "\n".join(issues)


def test_invariant_set_scorer_report_table_matches_source_csv() -> None:
    report_path = ROOT / "docs" / "experiments" / "wpu_v2_invariant_set_scorer_results.md"
    csv_path = ROOT / "docs" / "experiments" / "wpu_v2_retriever_invariant_set_scorer.csv"
    policy_labels = {
        "static learned interaction": "static_learned_interaction",
        "invariant set scorer": "invariant_set_scorer",
        "train-selected mechanism": "train_selected_mechanism",
        "seed-stable mechanism": "seed_stable_selected_mechanism",
        "risk-adjusted mechanism": "risk_adjusted_selected_mechanism",
        "candidate oracle": "generated_plus_composition_oracle",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    def mean_for(feature_variant: str, causal_k: str, policy: str, column: str) -> float:
        values = [
            float(row[column])
            for row in csv_rows
            if row["feature_variant"] == feature_variant
            and row["causal_k"] == causal_k
            and row["policy"] == policy
        ]
        if not values:
            raise AssertionError(f"Missing rows for {(feature_variant, causal_k, policy, column)}")
        return sum(values) / len(values)

    table_rows = _markdown_table_rows(report_path)
    header_index = next(
        index for index, row in enumerate(table_rows)
        if row[:3] == ["variant", "K", "policy"]
    )
    headers = table_rows[header_index]
    issues: list[str] = []
    for row in table_rows[header_index + 1:]:
        if len(row) != len(headers) or row[0] not in {"role_geometry_family", "role_geometry_only"}:
            break
        values = dict(zip(headers, row))
        variant = values["variant"]
        causal_k = values["K"]
        policy = policy_labels[values["policy"]]
        static_loss = mean_for(variant, causal_k, "static_learned_interaction", "loss")
        expected = {
            "loss": mean_for(variant, causal_k, policy, "loss"),
            "accuracy": mean_for(variant, causal_k, policy, "accuracy"),
            "oracle match": mean_for(variant, causal_k, policy, "oracle_match_rate"),
        }
        expected["delta vs static"] = expected["loss"] - static_loss
        for table_column, expected_value in expected.items():
            if _round_decimals(values[table_column], 6) != _round_decimals(str(expected_value), 6):
                issues.append(
                    f"{report_path.relative_to(ROOT)} -> {(variant, causal_k, values['policy'])} "
                    f"{table_column} table={values[table_column]} csv={expected_value}"
                )

    assert not issues, "Invariant set scorer report table values do not match source CSV:\n" + "\n".join(issues)


def test_regret_distillation_report_table_matches_source_csv() -> None:
    report_path = ROOT / "docs" / "experiments" / "wpu_v2_retriever_regret_distillation_results.md"
    csv_path = ROOT / "docs" / "experiments" / "wpu_v2_retriever_regret_distillation.csv"
    policy_labels = {
        "static learned interaction": "static_interaction",
        "regret-distilled retriever": "regret_distilled_retriever",
        "generated oracle": "generated_oracle",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    def mean_for(causal_k: str, policy: str, column: str) -> float:
        values = [
            float(row[column])
            for row in csv_rows
            if row["causal_k"] == causal_k and row["policy"] == policy
        ]
        if not values:
            raise AssertionError(f"Missing rows for {(causal_k, policy, column)}")
        return sum(values) / len(values)

    table_rows = _markdown_table_rows(report_path)
    header_index = next(
        index for index, row in enumerate(table_rows)
        if row == ["K", "policy", "loss", "accuracy", "excess over generated oracle"]
    )
    headers = table_rows[header_index]
    issues: list[str] = []
    for row in table_rows[header_index + 1: header_index + 10]:
        values = dict(zip(headers, row))
        policy = policy_labels[values["policy"]]
        for table_column, csv_column in (
            ("loss", "loss"),
            ("accuracy", "accuracy"),
            ("excess over generated oracle", "excess_over_generated_oracle"),
        ):
            expected = mean_for(values["K"], policy, csv_column)
            if _round_decimals(values[table_column], 6) != _round_decimals(str(expected), 6):
                issues.append(
                    f"{report_path.relative_to(ROOT)} -> {(values['K'], values['policy'])} "
                    f"{table_column} table={values[table_column]} csv={expected}"
                )

    assert not issues, "Regret distillation report table values do not match source CSV:\n" + "\n".join(issues)


def test_readme_v2_summary_tables_match_invariant_scorer_csv() -> None:
    csv_path = ROOT / "docs" / "experiments" / "wpu_v2_retriever_invariant_set_scorer.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    def mean_for(causal_k: str, policy: str, column: str) -> float:
        values = [
            float(row[column])
            for row in csv_rows
            if row["feature_variant"] == "role_geometry_family"
            and row["causal_k"] == causal_k
            and row["policy"] == policy
        ]
        if not values:
            raise AssertionError(f"Missing README source rows for {(causal_k, policy, column)}")
        return sum(values) / len(values)

    expected_by_k = {
        causal_k: {
            "Static learned loss": mean_for(causal_k, "static_learned_interaction", "loss"),
            "Risk-adjusted mechanism loss": mean_for(causal_k, "risk_adjusted_selected_mechanism", "loss"),
            "Accuracy gain": (
                mean_for(causal_k, "static_learned_interaction", "accuracy"),
                mean_for(causal_k, "risk_adjusted_selected_mechanism", "accuracy"),
            ),
        }
        for causal_k in ("8", "16", "32")
    }

    issues: list[str] = []
    for readme_path in (ROOT / "README.md", ROOT / "README.ko.md"):
        table_rows = _markdown_table_rows(readme_path)
        header_index = next(
            index for index, row in enumerate(table_rows)
            if row == ["K", "Static learned loss", "Risk-adjusted mechanism loss", "Accuracy gain"]
        )
        for row in table_rows[header_index + 1: header_index + 4]:
            values = dict(zip(table_rows[header_index], row))
            expected = expected_by_k[values["K"]]
            if _round_decimals(values["Static learned loss"], 6) != _round_decimals(
                str(expected["Static learned loss"]), 6
            ):
                issues.append(f"{readme_path.relative_to(ROOT)} K={values['K']} static loss")
            if _round_decimals(values["Risk-adjusted mechanism loss"], 6) != _round_decimals(
                str(expected["Risk-adjusted mechanism loss"]), 6
            ):
                issues.append(f"{readme_path.relative_to(ROOT)} K={values['K']} risk-adjusted loss")
            expected_gain = (
                f"{expected['Accuracy gain'][0]:.6f} -> {expected['Accuracy gain'][1]:.6f}"
            )
            if values["Accuracy gain"] != expected_gain:
                issues.append(
                    f"{readme_path.relative_to(ROOT)} K={values['K']} accuracy gain "
                    f"table={values['Accuracy gain']} csv={expected_gain}"
                )

    assert not issues, "README v2 summary tables do not match source CSV:\n" + "\n".join(issues)


def test_paper_v2_tables_match_source_csvs() -> None:
    regret_csv = ROOT / "docs" / "experiments" / "wpu_v2_retriever_regret_distillation.csv"
    invariant_csv = ROOT / "docs" / "experiments" / "wpu_v2_retriever_invariant_set_scorer.csv"

    with regret_csv.open(newline="", encoding="utf-8") as handle:
        regret_rows = list(csv.DictReader(handle))
    with invariant_csv.open(newline="", encoding="utf-8") as handle:
        invariant_rows = list(csv.DictReader(handle))

    def regret_mean(causal_k: str, policy: str, column: str) -> float:
        values = [
            float(row[column])
            for row in regret_rows
            if row["causal_k"] == causal_k and row["policy"] == policy
        ]
        if not values:
            raise AssertionError(f"Missing regret rows for {(causal_k, policy, column)}")
        return sum(values) / len(values)

    def invariant_mean(causal_k: str, policy: str, column: str) -> float:
        values = [
            float(row[column])
            for row in invariant_rows
            if row["feature_variant"] == "role_geometry_family"
            and row["causal_k"] == causal_k
            and row["policy"] == policy
        ]
        if not values:
            raise AssertionError(f"Missing invariant rows for {(causal_k, policy, column)}")
        return sum(values) / len(values)

    expected_regret = {
        causal_k: [
            causal_k,
            f"{regret_mean(causal_k, 'static_interaction', 'loss'):.6f}",
            f"{regret_mean(causal_k, 'regret_distilled_retriever', 'loss'):.6f}",
            f"{regret_mean(causal_k, 'static_interaction', 'accuracy'):.6f}",
            f"{regret_mean(causal_k, 'regret_distilled_retriever', 'accuracy'):.6f}",
        ]
        for causal_k in ("8", "16", "32")
    }
    expected_risk_adjusted = {
        causal_k: [
            causal_k,
            f"{invariant_mean(causal_k, 'static_learned_interaction', 'loss'):.6f}",
            f"{invariant_mean(causal_k, 'risk_adjusted_selected_mechanism', 'loss'):.6f}",
            f"{invariant_mean(causal_k, 'static_learned_interaction', 'accuracy'):.6f}",
            f"{invariant_mean(causal_k, 'risk_adjusted_selected_mechanism', 'accuracy'):.6f}",
        ]
        for causal_k in ("8", "16", "32")
    }

    def check_markdown_table(path: Path, header: list[str], expected: dict[str, list[str]]) -> list[str]:
        table_rows = _markdown_table_rows(path)
        header_index = next(index for index, row in enumerate(table_rows) if row == header)
        issues: list[str] = []
        for row in table_rows[header_index + 1: header_index + 4]:
            if row != expected[row[0]]:
                issues.append(f"{path.relative_to(ROOT)} K={row[0]} table={row} csv={expected[row[0]]}")
        return issues

    issues: list[str] = []
    for paper_path in (ROOT / "docs" / "paper" / "state_is_all_you_need.md", ROOT / "docs" / "arxiv" / "state_is_all_you_need_ko.md"):
        issues.extend(
            check_markdown_table(
                paper_path,
                ["K", "Static learned interaction loss", "Regret-distilled loss", "Accuracy before", "Accuracy after"],
                expected_regret,
            )
        )
        issues.extend(
            check_markdown_table(
                paper_path,
                ["K", "Static learned loss", "Risk-adjusted mechanism loss", "Accuracy before", "Accuracy after"],
                expected_risk_adjusted,
            )
        )

    tex_text = (ROOT / "docs" / "arxiv" / "state_is_all_you_need_en.tex").read_text(encoding="utf-8")
    for expected in (expected_regret, expected_risk_adjusted):
        for row in expected.values():
            latex_row = " & ".join(row) + r" \\"
            if latex_row not in tex_text:
                issues.append(f"docs/arxiv/state_is_all_you_need_en.tex missing row {latex_row}")

    assert not issues, "Paper v2 tables do not match source CSVs:\n" + "\n".join(issues)


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
