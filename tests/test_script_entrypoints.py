from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_BOOTSTRAP = "sys.path.append(str(Path(__file__).resolve().parents[1]))"


def test_scripts_importing_script_modules_bootstrap_repo_root() -> None:
    offenders: list[str] = []
    for path in (ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        first_script_import = min(
            [
                index
                for pattern in ("from scripts.", "import scripts.")
                if (index := text.find(pattern)) != -1
            ],
            default=-1,
        )
        if first_script_import == -1:
            continue
        bootstrap_index = text.find(ROOT_BOOTSTRAP)
        if bootstrap_index == -1 or bootstrap_index > first_script_import:
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, "Scripts importing scripts.* must bootstrap repo root:\n" + "\n".join(offenders)


def test_staged_regret_hybrid_help_runs_as_direct_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/staged_regret_hybrid.py", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
