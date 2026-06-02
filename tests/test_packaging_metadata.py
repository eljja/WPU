from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_uses_wheel_build_backend_requirements() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_requires = set(metadata["build-system"]["requires"])

    assert "wheel" in build_requires
    assert any(requirement.startswith("setuptools>=") for requirement in build_requires)


def test_pyproject_license_metadata_is_setuptools_compatible() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["license"] == {"file": "LICENSE"}
    assert "License :: OSI Approved :: GNU Affero General Public License v3" in project["classifiers"]


def test_ci_builds_and_smoke_installs_wheel() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m pip wheel . --no-deps --wheel-dir dist" in ci
    assert "python -m pip install --force-reinstall --no-deps dist/*.whl" in ci
    assert "wpu.create_model('wpu-cws-indexed'" in ci
