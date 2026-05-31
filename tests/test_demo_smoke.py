from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robot_cup_demo_prints_required_trace_sections() -> None:
    result = subprocess.run(
        [sys.executable, "demos/robot_cup_demo.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    for section in (
        "Event:",
        "Initial frontier:",
        "Scheduler path:",
        "Model path:",
        "Frontier trace:",
        "Changed objects:",
        "Branches:",
        "Memory estimate:",
    ):
        assert section in output
