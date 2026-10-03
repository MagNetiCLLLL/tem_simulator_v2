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
- A separate coherent-beam development page with explicit tip-boundary inputs,
  cached exact-Z observations and clear unsupported-operator limits.
- Hardware tuning with beam measurements and comparison against a saved baseline.
- Specimen scattering, detector signals, STEM scanning, EDS and optional
  energy-filter modelling.
- Live tuning with user-selected calculation endpoints, cached upstream states
  and resumable particle calculations.
- Compressed result export/import (`.temresult`), reusable startup results and
  separate virtual-electron sessions (`.temdiag`).

Classical particle transport remains the default. Independent coherent
development resumed on 2026-10-02; the default tip-to-image chain and arbitrary
Z within the gun or material are not yet qualified. See
[coherent-beam inputs and limits](docs/COHERENT_BEAM.md). Models target physical
mechanisms and qualitative parameter trends, rather than a calibrated
commercial instrument or a fully validated microscope.

Use **Run high-accuracy once** to update Ray Diagram. Then set specimen or
detector parameters and click **Calculate** on that page; compatible executed
beam states before the specimen are reused, and downstream rays update with the
result. Tab changes do not calculate. Live tuning retains its separate cutoff
and continuation controls.

For fast CIF-dependent HAADF/BF/DF images, choose **Projected atoms (fast
approximation)** in Scanning Image, set the scan pixel size and click
**Calculate**. This non-wave model combines actual CIF positions/occupancies,
the executed probe footprint and physical downstream detector acceptance.
It estimates thin-sample independent-atom contrast; BF/DF do not include Bragg
interference or channeling. A broad probe or coarse sampling still removes
atomic detail. **Material particle paths** retains the finite-volume particle
transport readout. Auto contrast changes display limits only; actual intensity
ranges remain visible. Loading a CIF does not enable coherent imaging.

The Sample page starts with **Vacuum sample**. To add material, import your own
**CIF / MCIF**, set its dimensions and orientation, then click **Calculate
Sample**. This calculates specimen interactions and updates downstream rays;
STEM images and EDS spectra use their own page buttons. No built-in reference
CIF library or automatic material selection is provided.

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

[Project map / 项目地图](docs/PROJECT_MAP.md): files, features, UI, tests and refactoring candidates.

[MIT License](LICENSE).
