from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
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
        "scripts/analyze_candidate_oracle_gap.py",
        "scripts/analyze_candidate_noharm_gate.py",
        "scripts/analyze_candidate_regret_gate.py",
        "scripts/analyze_state_integrity.py",
        "scripts/audit_v2_priority_dashboard.py",
        "scripts/causal_working_set_experiment.py",
        "scripts/retriever_regret_distillation_probe.py",
        "scripts/retriever_cross_seed_invariant_set_scorer_probe.py",
        "scripts/retriever_cross_seed_candidate_regret_gate_probe.py",
    ]

    for script in scripts:
        _assert_help_runs(script)


def test_documented_python_scripts_expose_help() -> None:
    scripts = _documented_python_scripts()
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(_run_help, scripts))

    failures = [
        f"{script}\n{stderr}"
        for script, returncode, stderr in results
        if returncode != 0
    ]
    assert not failures, "Documented scripts must expose --help:\n" + "\n".join(failures)


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


def test_objectification_relation_repair_probe_runs(tmp_path: Path) -> None:
    output = tmp_path / "relation_repair.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/objectification_relation_repair_probe.py",
            "--samples",
            "4",
            "--train-samples",
            "8",
            "--learned-steps",
            "2",
            "--near-distractors",
            "2",
            "--background-objects",
            "2",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "no_repair" in text
    assert "ungated" in text
    assert "type_gated" in text
    assert "learned_scorer" in text
    assert "in_distribution" in text
    assert "dense_distractors" in text
    assert "aliased_types_with_roles" in text
    assert "aliased_types_without_roles" in text
    assert "downstream_branch_accuracy" in text
    assert "downstream_branch_loss" in text


def test_object_history_hidden_mechanism_probe_runs(tmp_path: Path) -> None:
    output = tmp_path / "hidden_mechanism.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/object_history_hidden_mechanism_probe.py",
            "--train-samples",
            "16",
            "--eval-samples",
            "8",
            "--train-steps",
            "4",
            "--candidates",
            "4",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "history_scorer" in text
    assert "hidden_field" in text
    assert "relation_precision" in text
    assert "downstream_loss" in text


def test_object_history_hidden_mechanism_probe_multiseed_summary_runs(tmp_path: Path) -> None:
    output = tmp_path / "hidden_mechanism_multiseed.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/object_history_hidden_mechanism_probe.py",
            "--train-samples",
            "16",
            "--eval-samples",
            "8",
            "--train-steps",
            "4",
            "--candidates",
            "4",
            "--seeds",
            "3",
            "5",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "summary,all,hidden_field,history_scorer" in text
    assert "seed_count" in text
    assert "hidden_mechanism_probe row_type=summary" in result.stdout


def test_object_relation_law_probe_runs(tmp_path: Path) -> None:
    output = tmp_path / "object_relation_law.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/object_relation_law_probe.py",
            "--train-samples",
            "16",
            "--eval-samples",
            "8",
            "--train-steps",
            "4",
            "--candidates",
            "4",
            "--seeds",
            "3",
            "5",
            "--eval-mechanisms",
            "hidden_inverse",
            "hidden_inverse_far",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "summary,all,hidden_inverse,history_relation_law" in text
    assert "summary,all,hidden_inverse_far,history_relation_law" in text
    assert "delta_mse" in text
    assert "object_relation_law_probe row_type=summary" in result.stdout


def test_object_relation_law_revision_probe_runs(tmp_path: Path) -> None:
    output = tmp_path / "object_relation_law_revision.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/object_relation_law_revision_probe.py",
            "--train-samples",
            "16",
            "--calibration-samples",
            "8",
            "--eval-samples",
            "8",
            "--train-steps",
            "4",
            "--candidates",
            "4",
            "--seeds",
            "3",
            "5",
            "--mechanisms",
            "hidden_inverse_gain_shift",
            "hidden_power_shift",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "summary,all,hidden_inverse_gain_shift,form_revised_history_law" in text
    assert "summary,all,hidden_power_shift,gain_calibrated_history_law" in text
    assert "calibration_mse" in text
    assert "object_relation_law_revision_probe row_type=summary" in result.stdout


def _assert_help_runs(script: str) -> None:
    _, returncode, stderr = _run_help(script)

    assert returncode == 0, f"{script}\n{stderr}"


def _run_help(script: str) -> tuple[str, int, str]:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    return script, result.returncode, result.stderr


def _documented_python_scripts() -> list[str]:
    refs: set[str] = set()
    for path in [*ROOT.glob("README*.md"), *(ROOT / "docs").rglob("*.md")]:
        if any(part in {"artifacts", ".venv", ".git"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        refs.update(match.group(1) for match in re.finditer(r"`(scripts/[^`]+?\.py)`", text))
    return sorted(refs)
