from __future__ import annotations

import re
import csv
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
            candidate = ROOT / target.replace("\\", "/")
            if not candidate.exists():
                issues.append(f"{path.relative_to(ROOT)} -> missing {target}")
                continue
            with candidate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            if len(rows) < 2:
                issues.append(f"{path.relative_to(ROOT)} -> empty {target}")

    assert not issues, "Invalid experiment Source CSV references:\n" + "\n".join(issues)


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
