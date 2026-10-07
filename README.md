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
- Ray Diagram conjugate-plane searches from a pinned Z, with magnification,
  rotation and separate point-image / line-focus diagnostics.
- **Calculate beam** on the **Electron beam** page for a captured Tip and
  instrument state, with intensity, simulated electron arrivals, per-mode phase
  and probability-flow observations, cached exact-Z planes and named
  emission-state intensity overlays.
- Small-angle installation offsets/tilts of post-gun lenses, deflectors,
  stigmators, corrector fields and circular apertures, using shared ray/wave
  geometry and fields. Finite tilted aperture plates need declared thickness;
  coherent spherical aberration uses a bounded small-angle thin-lens approximation.
- Hardware tuning with beam measurements and comparison against a saved baseline.
- Specimen scattering, detector signals, STEM scanning, EDS and optional
  energy-filter modelling.
- Live tuning with user-selected calculation endpoints, cached upstream states
  and resumable particle calculations.
- Compressed result export/import (`.temresult`), reusable startup results and
  separate virtual-electron sessions (`.temdiag`).

The **Electron beam** page has its own **Calculate beam** button. The top
**Run high-accuracy once** button prepares classical particle results for Ray
Diagram and the specimen/detector pages. Apply supported Tip emission/phase
parameters explicitly; opening the page or calculating never enables a source
model or applies a draft.
The default tip-to-image chain and arbitrary Z within the gun or material are
not yet qualified. See [electron-beam inputs and limits](docs/COHERENT_BEAM.md).
Models target physical mechanisms and qualitative parameter trends, rather than
a calibrated commercial instrument or a fully validated microscope.

Choose the current Tip, a saved state or an intensity overlay on **Electron
beam**, then press **Calculate beam**. Switch observation views without
propagating again. Electron arrivals sample the calculated screen probabilities,
including electrons that do not reach that screen; phase belongs to one coherent
mode, and probability-flow arrows show local current directions rather than
measured electron trajectories.
For a same-input classical comparison, enable **Compare classical rays** under
the advanced settings before Calculate beam. **Update rays** remains a fast
particle preview; **Run high-accuracy once** (also available as **Simulation →
Calculate classical rays**) prepares the classical Ray Diagram and the incident
beam required by Scanning Image and the specimen/detector page calculations.
Those pages reuse compatible executed particle states. Tab changes do not
calculate. Live tuning retains its cutoff and continuation controls.

In **Ray Diagram**, select an axial Z and open **Conjugate planes**. Click
**Find conjugate planes** to pin that reference and search the captured optics.
Click a row to inspect its Z; the reference stays pinned. To search from another
plane, choose **Use selected Z**, then **Find conjugate planes**. The column
transfer is cached until a new result is published or optics become stale.
Hover the selected-plane status, Selected Z line, or a conjugate candidate to
see its symbolic transfer equation, classification rule and reference convention.
These are nominal-energy, first-order real-plane diagnostics in the supported
straight column; they do not establish beam transmission or crystal diffraction
intensity. Near-tip non-paraxial transport and the curved energy-filter branch
are outside this search.

**EDS** groups **Spectrum**, **Interactions 3D** and **Parameters**. Its local
interaction view highlights scattering by type and can focus on the material;
the spectrum and detailed paths keep separate explicit calculation buttons.

New instrument states start with both **AC** and **Descan** raster drives enabled;
loaded profiles retain their saved drive settings.

For **FEG** and **FEG + Mono**, **Gun Aperture** (also called **Extractor
Aperture**) is a separate component after Gun Lens and before Accelerator.
Its default centre is 24 mm from the tip, before the optional Wien unit; this
is adjustable simulator geometry, not a manufacturer dimension. It intercepts
electrons outside the opening and marks the default differential-pumping
boundary between gun-side and column-side vacuum. Vacuum pressures remain
specified inputs rather than predictions from aperture conductance or pumping
speed. The supplied SFEG illustration provides structural context; it does
not specify dimensions or change the current cold-field-emitter model.

High-accuracy particle calculations default to **3,000 rays**. The top toolbar's
**Compute** control is the single CPU/GPU selection for all pages. **Auto** uses
a usable GPU for supported operations, including small jobs, and otherwise uses
CPU. CPU-only preparation remains on CPU; completed results report the actual
backend. Individual pages do not override this preference.
For particle calculations, a CUDA driver without Numba's CUDA compiler also
triggers CPU fallback under **Auto**; the result retains the reason.
**Require GPU** continues to reject an unavailable GPU backend.

For fast CIF-dependent HAADF/BF/DF images, first run **Run high-accuracy once**
with the current instrument settings. Then choose **Projected atoms (fast
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
