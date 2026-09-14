"""Install the project using local wheels and, unless offline, the package index.

Python 3.12 is the validated baseline; 64-bit CPython 3.11-3.13 is accepted.
Run ``python setup_env.py --help`` for alternate environments and offline setup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
WHEEL_DIR = PROJECT_ROOT / "wheels"
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 14)  # PySide6 6.8.3 requires Python <3.14.


def venv_python(directory: Path = VENV_DIR) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str]) -> None:
    print(f"\n> {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def runtime_info(python: Path) -> dict:
    code = (
        "import json, struct, sys; "
        "print(json.dumps({'version': list(sys.version_info[:3]), "
        "'bits': struct.calcsize('P')*8, 'implementation': sys.implementation.name}))"
    )
    result = subprocess.run([str(python), "-c", code], cwd=PROJECT_ROOT,
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def validate_runtime(info: dict) -> None:
    version = tuple(info["version"][:2])
    if not MIN_PYTHON <= version < MAX_PYTHON or info["bits"] != 64 or info["implementation"] != "cpython":
        raise ValueError(
            "Use 64-bit CPython 3.11, 3.12 (recommended), or 3.13. "
            "abTEM requires >=3.11; pinned PySide6 6.8.3 requires <3.14. "
            f"Selected runtime: {info}. Use --venv to create a separate environment."
        )


def wheel_sources(directory: Path, offline: bool) -> list[str]:
    if not directory.is_dir():
        raise ValueError(f"Wheel directory does not exist: {directory}")
    # Each directory is a candidate pool. pip checks platform, ABI, Python and
    # dependency versions; passing every *.whl as a requirement would not.
    paths = {directory, *(p.parent for p in directory.rglob("*.whl"))}
    options = ["--no-index"] if offline else []
    for path in sorted(paths):
        options.extend(["--find-links", str(path)])
    return options


def verify(python: Path, *, gpu: bool = False) -> None:
    run([str(python), "-m", "pip", "--disable-pip-version-check", "check"])
    imports = (
        "import abtem, ase, cloudpickle, manifold3d, matplotlib, numba, numpy, "
        "pytest, PySide6, scipy, temsim, xraylib; "
        "from PySide6 import QtCore, QtGui, QtWidgets; "
        "import OpenGL, pyqtgraph; "
        + ("import cupy; " if gpu else "")
        + "print('Environment verification passed.'); "
        "print('Python:', __import__('sys').version.split()[0]); "
        "print('PySide6:', PySide6.__version__); "
        "print('Qt:', QtCore.qVersion()); "
        "print('PyQtGraph:', pyqtgraph.__version__); "
        "print('TEM Simulator:', temsim.__version__)"
    )
    run([str(python), "-c", imports])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=VENV_DIR,
                        help="Environment directory (default: project .venv). Existing environments are reused, never replaced.")
    parser.add_argument("--wheel-dir", type=Path, default=WHEEL_DIR,
                        help="Local wheel directory, including subdirectories (default: project wheels).")
    parser.add_argument("--offline", action="store_true", help="Use only installed packages and local wheels; never use the package index.")
    parser.add_argument("--gpu", action="store_true", help="Also install the project's optional CUDA 12.x CuPy dependency.")
    args = parser.parse_args(argv)
    directory, wheel_directory = args.venv.resolve(), args.wheel_dir.resolve()

    try:
        sources = wheel_sources(wheel_directory, args.offline)
        python = venv_python(directory)
        if python.exists():
            info = runtime_info(python)
            validate_runtime(info)
            print(f"Reusing existing environment: {directory}")
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"{directory} is not an executable virtual environment and is not empty. Choose another --venv directory.")
            base_python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
            info = runtime_info(base_python)
            validate_runtime(info)
            print(f"Creating environment: {directory}")
            # ensurepip is local. --upgrade-deps would access the network before
            # our wheel/offline options could take effect.
            run([str(base_python), "-m", "venv", str(directory)])

        print("Selected Python:", ".".join(map(str, info["version"])))
        if tuple(info["version"][:2]) != (3, 12):
            print("Python 3.12 remains the reference for the CPU validation lock; other versions resolve compatible dependencies.")
        print(f"Local wheels: {wheel_directory} ({sum(1 for _ in wheel_directory.rglob('*.whl'))} files)")
        # Read build requirements and runtime dependencies from the project.
        # This runs inside the selected interpreter (which supplies tomllib).
        metadata = subprocess.run(
            [str(python), "-c", "import json,tomllib; print(json.dumps(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))"],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        bootstrap = ["pip>=23.1", *json.loads(metadata.stdout)]
        install = [str(python), "-m", "pip", "--disable-pip-version-check", "install", *sources]
        run([*install, *bootstrap])
        extras = "dev,gpu" if args.gpu else "dev"
        # Build dependencies were installed with the same wheel/index policy.
        # Avoid a second isolated build environment bypassing that setup.
        run([*install, "--no-build-isolation", "--editable", f"{PROJECT_ROOT}[{extras}]"])
        verify(python, gpu=args.gpu)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"\nERROR: Environment setup failed: {error}", file=sys.stderr)
        if args.offline:
            print("Offline setup requires compatible wheels for all missing dependencies, including pip/setuptools/wheel. Add them to wheels/ or run without --offline.", file=sys.stderr)
        return 1

    print("\nEnvironment setup completed successfully.")
    print(f"Set the PyCharm interpreter to: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
