"""Check the failing TEM expectation against the pre-fix transport modules."""
from importlib import import_module
from pathlib import Path
import os
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for name in ("first_order", "record_plane"):
        path = f"src/temsim/physics/{name}.py"
        source = subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True, encoding="utf-8")
        module = import_module(f"temsim.physics.{name}")
        exec(compile(source, f"HEAD:{path}", "exec"), module.__dict__)
    import pytest
    return pytest.main(["-o", "addopts=", "-q", "--tb=short",
        "tests/test_wave_imaging.py::test_defocused_image_wave_uses_symplectic_specimen_canonical_transfer"])


if __name__ == "__main__":
    raise SystemExit(main())
