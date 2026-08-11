"""Create the project virtual environment and install its dependencies.

Run this file once from PyCharm with a Python 3.12 interpreter. After it
finishes, select ``.venv/Scripts/python.exe`` as the project interpreter.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"

DEPENDENCIES = (
    "numpy",
    "scipy",
    "numba",
    "matplotlib",
    # PySide6 6.11.1 fails to load QtCore with the current Windows/Conda
    # Python runtime. Keep this known-good Qt release until compatibility is
    # verified before upgrading it.
    "PySide6==6.8.3",
    "pyqtgraph",
    "PyOpenGL>=3.1.7,<4",
    "ase>=3.29,<4",
    "abtem==1.0.10",
    "imageio>=2.37,<3",
    "pillow>=11,<13",
    "tifffile>=2025,<2027",
    "pytest",
    "pytest-qt",
    "tomli-w",
)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def verify(python: Path) -> None:
    imports = (
        "import matplotlib, numba, numpy, pytest, PySide6, scipy, temsim; "
        "from PySide6 import QtCore, QtGui, QtWidgets; "
        "import pyqtgraph; "
        "print('Environment verification passed.'); "
        "print('Python:', __import__('sys').version.split()[0]); "
        "print('PySide6:', PySide6.__version__); "
        "print('Qt:', QtCore.qVersion()); "
        "print('PyQtGraph:', pyqtgraph.__version__); "
        "print('TEM Simulator:', temsim.__version__)"
    )
    run([str(python), "-c", imports])


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print(
            "ERROR: Run this script with 64-bit Python 3.12. "
            f"Current version: {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return 1

    try:
        if venv_python().exists():
            print(f"Reusing existing environment: {VENV_DIR}")
        else:
            print(f"Creating environment: {VENV_DIR}")
            base_python = Path(
                getattr(sys, "_base_executable", sys.executable)
            ).resolve()
            run(
                [
                    str(base_python),
                    "-m",
                    "venv",
                    "--upgrade-deps",
                    str(VENV_DIR),
                ]
            )

        python = venv_python()
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ]
        )
        run([str(python), "-m", "pip", "install", *DEPENDENCIES])
        run([str(python), "-m", "pip", "install", "--editable", ".[dev]"])
        verify(python)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"\nERROR: Environment setup failed: {error}", file=sys.stderr)
        return 1

    print("\nEnvironment setup completed successfully.")
    print(f"Set the PyCharm interpreter to: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
