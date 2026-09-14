"""Installer compatibility, offline policy, and real local-wheel resolution."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile

from packaging.specifiers import SpecifierSet
import pytest

import setup_env


@pytest.mark.parametrize("minor", [10, 11, 12, 13, 14])
def test_runtime_policy_matches_project_metadata(minor):
    project = tomllib.loads((setup_env.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    supported = f"3.{minor}" in SpecifierSet(project["project"]["requires-python"])
    info = {"version": [3, minor, 0], "bits": 64, "implementation": "cpython"}
    if supported:
        setup_env.validate_runtime(info)
    else:
        with pytest.raises(ValueError, match="64-bit CPython"):
            setup_env.validate_runtime(info)


@pytest.mark.parametrize("change", [{"bits": 32}, {"implementation": "pypy"}])
def test_rejects_unsupported_runtime_architecture(change):
    with pytest.raises(ValueError, match="64-bit CPython"):
        setup_env.validate_runtime({"version": [3, 12, 4], "bits": 64, "implementation": "cpython", **change})


def test_existing_incompatible_venv_is_not_modified(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"existing interpreter")
    wheels.mkdir()
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {"version": [3, 14, 0], "bits": 64, "implementation": "cpython"})
    monkeypatch.setattr(setup_env, "run", lambda _: pytest.fail("must not modify unsupported existing environment"))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels)]) == 1
    assert python.read_bytes() == b"existing interpreter"


def test_offline_gpu_policy_applies_to_every_install_step(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "local wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.touch()
    wheels.mkdir()
    nested = wheels / "cp313"
    nested.mkdir()
    (nested / "candidate.whl").touch()
    commands, verification = [], []
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {"version": [3, 13, 1], "bits": 64, "implementation": "cpython"})
    monkeypatch.setattr(setup_env, "run", commands.append)
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(["setuptools>=77", "wheel"])))
    monkeypatch.setattr(setup_env, "verify", lambda p, **k: verification.append((p, k)))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels), "--offline", "--gpu"]) == 0
    assert len(commands) == 2
    for cmd in commands:
        assert cmd[0] == str(python)
        assert "--no-index" in cmd and "--disable-pip-version-check" in cmd
        assert str(wheels) in cmd and str(nested) in cmd
        assert "--upgrade-deps" not in cmd
    assert "--no-build-isolation" in commands[-1]
    assert commands[-1][-1] == f"{setup_env.PROJECT_ROOT}[dev,gpu]"
    assert verification == [(python, {"gpu": True})]


def test_nonempty_non_environment_directory_is_preserved(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    env.mkdir()
    wheels.mkdir()
    (env / "notes.txt").write_text("keep me")
    monkeypatch.setattr(setup_env, "run", lambda _: pytest.fail("must not replace directory"))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels)]) == 1
    assert (env / "notes.txt").read_text() == "keep me"


def _wheel(directory, name, version, tag="py3-none-any"):
    stem = f"{name}-{version}"
    dist = stem + ".dist-info"
    entries = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.11\n",
        f"{dist}/WHEEL": f"Wheel-Version: 1.0\nGenerator: installer-test\nRoot-Is-Purelib: true\nTag: {tag}\n",
    }
    entries[f"{dist}/RECORD"] = "\n".join(f"{p},," for p in [*entries, f"{dist}/RECORD"])
    path = directory / f"{stem}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def test_pip_installs_compatible_local_wheel_from_nested_folder_offline(tmp_path):
    root = tmp_path / "mixed wheels with spaces"
    nested = root / "other directory"
    nested.mkdir(parents=True)
    _wheel(nested, "temsim_setup_fixture", "1.0")
    _wheel(root, "temsim_setup_fixture", "9.0", "cp399-cp399-win_amd64")
    _wheel(root, "temsim_unrelated_fixture", "1.0")
    target = tmp_path / "installed"
    result = subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "install",
        *setup_env.wheel_sources(root, offline=True), "--ignore-installed", "--target", str(target),
        "temsim_setup_fixture>=1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "temsim_setup_fixture" / "__init__.py").read_text() == "VERSION = '1.0'\n"
    assert not (target / "temsim_unrelated_fixture").exists()
    assert "http" not in result.stdout
