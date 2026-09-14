# Local dependency wheels

Put `.whl` files here. Subdirectories such as `cp311`, `cp312`, `cp313`,
`windows` or `linux` are allowed; `setup_env.py` discovers them recursively.
Different Python versions and platforms can coexist. pip selects compatible
files and resolves versions against `pyproject.toml`; it does not blindly install
every wheel, downgrade an installed package, or bypass dependency constraints.
Only required dependencies (including dev dependencies and, with `--gpu`, the
GPU extra) are installed. Unrelated wheels can be stored here for later use.

From the project root:

```powershell
python setup_env.py
python setup_env.py --offline
python setup_env.py --gpu
python setup_env.py --venv .venv311 --wheel-dir D:\dependency-wheels
```

Normal setup considers both local wheels and the configured package index;
pip chooses a satisfying version. `--offline` disables the index for every
installation step, including build tools. An empty or incomplete wheel folder
is fine online; offline setup fails explicitly if a required package is missing.
Existing installed dependencies that satisfy the requirements are retained.

To prepare an offline **Python 3.12 CPU validation** wheel collection, use a
connected machine with the same OS, architecture and Python version:

```powershell
python -m pip download --only-binary=:all: --dest wheels "pip>=23.1" "setuptools>=77" wheel
python -m pip download --only-binary=:all: --dest wheels -r requirements/validation-cpu-lock.txt
```

The CPU lock is specifically the Python 3.12 validation baseline. Do not force
it onto Python 3.11: some locked versions, including SciPy 1.18, require 3.12.
Ordinary setup uses the compatible ranges in `pyproject.toml`. Python 3.13 is
accepted by dependency metadata but is not the project's full validation baseline.

Wheels are local binary dependency artifacts and are ignored by Git. This README
keeps the folder in the repository; share the binaries separately when needed.
Simulator source remains installed editable from the working tree, so a stale
simulator wheel stored here cannot replace current project code.

Validation on Windows x64 (2026-09-14): Python 3.11.9 fresh setup, import checks,
and 79 installer/vacuum GUI/3D geometry/EDS tests passed. Python 3.12.4 reused
the existing environment successfully in offline mode; its 11 installer tests
passed. The CPython 3.13 Windows wheel dependency plan resolved successfully,
but a Python 3.13 interpreter and scientific acceptance were not executed.
The existing Python 3.12 CPU CI lock remains unchanged.
