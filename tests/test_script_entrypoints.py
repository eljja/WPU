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
    _assert_help_runs("scripts/staged_regret_hybrid.py")


def test_readme_and_current_evidence_script_help_runs() -> None:
    scripts = [
        "scripts/train_object_physics.py",
        "scripts/eval_object_physics.py",
        "scripts/route_sweep.py",
        "scripts/robust_experiment_suite.py",
        "scripts/analyze_n_sweep.py",
        "scripts/analyze_b_sweep.py",
        "scripts/analyze_step_sweep.py",
        "scripts/analyze_controlled_stress.py",
        "scripts/causal_working_set_experiment.py",
        "scripts/retriever_regret_distillation_probe.py",
        "scripts/retriever_cross_seed_invariant_set_scorer_probe.py",
    ]

    for script in scripts:
        _assert_help_runs(script)


def test_readme_core_scripts_run_as_direct_scripts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "object_physics_smoke.pt"
    commands = [
        [
            sys.executable,
            "scripts/train_object_physics.py",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--background-objects",
            "2",
            "--hidden-dim",
            "16",
            "--checkpoint",
            str(checkpoint),
        ],
        [
            sys.executable,
            "scripts/eval_object_physics.py",
            "--samples",
            "4",
            "--batch-size",
            "2",
            "--background-objects",
            "2",
            "--hidden-dim",
            "16",
            "--checkpoint",
            str(tmp_path / "missing_checkpoint.pt"),
        ],
        [
            sys.executable,
            "scripts/route_sweep.py",
            "--samples",
            "4",
            "--batch-size",
            "2",
            "--background-sizes",
            "0",
            "2",
        ],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    assert checkpoint.exists()


def _assert_help_runs(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, f"{script}\n{result.stderr}"
