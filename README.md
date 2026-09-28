# TEM Simulator v2

A Python/PySide6 desktop application for exploring transmission electron
microscopy, electron optics and simulated detector signals.

## AI for experiments

The simulator provides an offline environment for learning instrument controls,
exploring parameter responses and developing experimental workflows. This can
help researchers make better use of precious **instrument time**, by preparing
and testing ideas before using a microscope. It also provides a foundation for
future **AI for experiments**, including AI-guided tuning and experiment design.
The simulator does not currently provide autonomous microscope control.

## Features

- Editable microscope geometry and hardware settings, with electron emission,
  extraction, acceleration, lenses, deflectors, stigmators and apertures.
- Interactive ray diagrams, transverse beam plots, flight-time colouring,
  magnetic-field views and individually adjustable virtual electron paths.
- Hardware tuning with beam measurements and comparison against a saved baseline.
- Specimen scattering, detector signals, STEM scanning, EDS and optional
  energy-filter modelling.
- Live tuning with user-selected calculation endpoints, cached upstream states
  and resumable particle calculations.
- Compressed result export/import (`.temresult`), reusable startup results and
  separate virtual-electron sessions (`.temdiag`).

The current focus is classical particle transport; coherent tip-to-column wave
development remains paused. Models target physical mechanisms and qualitative
parameter trends, rather than a calibrated commercial instrument or a fully
validated microscope.

## Setup and launch

On Windows, install **64-bit Python 3.12** (recommended), then run these commands
from the project folder:

```powershell
py -3.12 setup_env.py
.venv\Scripts\python.exe main.py
```

Setup creates or reuses `.venv`, installs dependencies and checks the environment.
Python 3.11–3.13 is supported; the interface uses PySide6 6.8.3. Internet access
is normally needed for missing dependencies. Run `py -3.12 setup_env.py --help`
for offline installation and optional GPU settings.

For subsequent launches, run only the second command. In PyCharm, select
`.venv\Scripts\python.exe` as the interpreter and run `main.py`.

[MIT License](LICENSE).
