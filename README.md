# TEM Simulator v2

TEM Simulator v2 is a Python desktop application for exploring transmission
electron microscopy and electron optics. It provides an interactive environment
for studying how an electron source, microscope geometry and operating settings
influence beam propagation and recorded signals.

The project is intended for teaching, experimentation and model development.
Instrument definitions are stored in editable TOML files, with a PySide6 interface
for configuring and visualising the microscope. Models include explicit
approximations and are not a calibrated replica of a commercial instrument.

The current development focus is classical particle transport. Coherent
tip-to-column wave development is paused; existing wave models and historical
results remain available within their documented limits.

Cold field emission now uses a coupled electrode potential for flat and
continuously curved tips. Extraction, the gun lens, all accelerating electrodes
and the grounded outlet participate in one field solution. Ray Diagram marks
electrode locations; they do not bound the fringe fields. Electric fields are
cached separately from emitted particles and rebuilt when their physical or
numerical inputs change. See the [field repair and validation](docs/development/gun-electrode-field-repair-2026-09-26.md).

## Getting started

**Ray Diagram → Magnetic field → Display** switches between combined 2D
field strength and **3D field lines**, including magnetic lens, stigmator,
corrector and deflector contributions. Rotate Ray Diagram to rotate both views;
the drawing fills the available space and follows the visible axial range.
Main controls share one row; map import and display-reference settings are in
Advanced. No electron recalculation is needed for projection/pan/zoom. See the
[combined field guide](docs/development/magnetic-field-lines-3d-2026-09-26.md)
for the equivalent deflector-field and near-axis/map domain limits.

The spatial field and electron views have physical **Z** and **projected U**
axes with automatic length units. Wheel over the plot to zoom both axes,
or over an axis to zoom only that direction, using the Ray Diagram's wheel
response and pointer anchor. **Link Ray Diagram** links both physical ranges;
uncheck it for independent pan/zoom. Rechecking adopts the current Ray Diagram
range. The choice is retained in workspace layouts; viewing never retraces a
particle. The 2D magnetic-strength axis remains in tesla.
See the [navigation and response verification](docs/development/test-electron-navigation-performance-2026-09-26.md)
for the measured speed improvements, numerical equivalence and test scope.

Choose **Electron trajectories** in the same display menu, then **Electrons…**.
**Virtual electrons** is a dockable panel, also available in **View**. Drag its
title bar to dock, tabify or float it; closing it preserves the electron list
and calculated paths. Narrow docks stack the list above a scrollable editor.
The list keeps independent electron parameters, colours, calculation status
and actual endpoints. **Add electron** starts at the captured tip; **Duplicate**
copies the selected parameters for comparison. Click a row to edit initial
energy in eV, X/Y/Z, polar angle in mrad (with a degree readout), azimuth and maximum path length. New electrons
use the tip's local emission energy (0.3 eV for the default tip).
The polar slider provides fine adjustment over 0–5 mrad; numeric input steps
by 0.01 mrad and preserves larger explicit angles. **Physical angle** reports
initial/final 3D direction angles, independently of the enlarged plot slopes.
Extraction, the electrostatic gun lens, all accelerating stages and magnetic
optics act on that trajectory; the readout shows how its kinetic energy changes.
**Reset to tip emission** restores the captured tip defaults.

Use **Selected electron** to inspect one path, or **Overlay checked** to compare
the checked rows. Colours match the list and legend. Switching rows, changing
visibility or renaming an electron reuses its stored trajectory. Calculations
share one fixed E+B scene and run serially; editing one electron only invalidates
that electron. Identical settings can share the same executed path.
Captured-field preparation and electron integration run in a persistent hidden
process with one numerical CPU worker, leaving the interface responsive. Fields
remain prepared for subsequent edits; first use includes process startup time.
The shared CPU admission still prevents competing calculations from multiplying
the configured budget. This isolates the computation without changing its physics.
Typing a number submits on Enter or focus loss. Sliders update while held:
the previous path remains dashed while actual integrated prefixes and completed
parameter samples arrive. An earlier sample is labelled separately and never
reported as the current exact result. Releasing requests the final selected value;
rapid intermediate values are coalesced. Changing fields clears the old paths.
Compiled field evaluation and integration accelerate supported captured scenes
without changing their tolerances or hardware stops; other scenes retain the
complete general solver. First use includes compilation/cache-loading overhead.
See the [continuous response verification](docs/development/continuous-electron-response-2026-09-26.md).
Apertures, electrodes and walls stop the path; specimen, detector and
electron-electron interactions are excluded. This diagnostic does not change
the instrument source or create a continuation checkpoint. The coloured paths
follow the Ray Diagram projection. First use may prepare an extended electric
field once in the background; unchanged settings reuse cached trajectories.
Numerical controls and explicit unsupported-field stops are described in the
[virtual electron guide](docs/development/magnetic-test-electron-2026-09-26.md).
Linked field plots also match the Ray Diagram's actual horizontal pixel bounds
after docking, scrolling or resizing. Electron lines are thin, magnetic lines
are more visible, and field arrows keep a fixed screen size. See the
[dock and responsiveness verification](docs/development/virtual-electron-dock-response-2026-09-26.md).

Use 64-bit Python 3.12 on Windows:

```powershell
python setup_env.py
.venv\Scripts\python.exe main.py
```

## Saving and opening results

Use **Export result…** (`Ctrl+S`) to save the displayed classical calculation
with its calculated settings and exact continuation state in a `.temresult`
file. **Open result…** (`Ctrl+O`) restores those settings and displays the saved
results without running another calculation. A later edit is checked against
the saved upstream dependencies before any segment is reused.

For reusable demonstrations, use **File → Save current as named result…**,
then choose it in **File → Startup result**. The next launch opens that completed
result instead of calculating an initial preview. No example result is marked
complete without having been calculated. Parameter-only profiles remain under
**File → Settings**.

Exports use lossless compression and exact array deduplication; particle
precision and sampling are preserved. File size still depends on the amount of
retained trajectory, scattering and detector data. Named results are stored
separately from disposable calculation caches.

Open **Ray Diagram → Live tuning** for both **Calculation** and **Working points**.
Calculation owns the cutoff plane and the Open/Export result controls. Working
points shows the record list and restore controls; expand **Advanced** for
record import/export, comparisons, portable inputs and numerical checks.

## Beam plots and ray colours

The **Plot** selector in Beam analysis and Ray Diagram's **Colour by** control
share one colour quantity. Source position and emission direction keep each
path's launch colour through the column. **Time of flight** uses a gradient of
saved cumulative flight time along each ray, on the same scale as both transverse
plots. Hover still reports the arrival delay at the selected plane. Missing
clocks are grey. This view reads the captured trajectories; it does not run
transport, measure geometric path length or calculate coherent phase.

**Plot sizes…** provides one common width and height for the two beam plots;
their internal plotting rectangles also match. The colour legend has its own
size. Plot sizes are retained with the workspace layout.

**Auto-adjust condensers** changes C1/C2/C3 to improve transmission through the
projection-chamber entrance. The result shows their before/after values and the
validation population. This optical reference defers specimen signals and does
not calibrate probe/image focus. Use **Live tuning → Working points → Advanced →
Undo last apply** to undo the latest applied adjustment.

**Acceleration gaps** toggles amber annotations from the displayed result's
captured gun geometry. Analytic-field bands mark the potential transitions;
solved-field markers indicate electrode positions only. These marks help locate
the small bends at acceleration stages; toggling them does not recalculate or
smooth the trajectories. The calculated-range bar also supports completed
optical validation, without presenting it as a resumable particle checkpoint.

## Hardware tuning

Open **Hardware tuning** beside Ray Diagram. Select an alignment in the left
column, then edit its associated hardware parameters directly in the right
column. Press Enter or leave a numeric field to apply it. The search box filters
the alignment list; long hardware groups scroll and the column divider is saved
with the workspace layout.

This page provides manual hardware controls independently of Direct Alignment.
Tasks sharing a deflector or lens edit the same instrument value, and the
existing parameter editors stay synchronized. Units and unavailable hardware
are shown explicitly. Component validation rejects invalid or conflicting
drives before changing the instrument. Accepted edits use the normal result
invalidation and calculation policy; selecting a task performs no calculation.

There are controls for the gun, illumination, focusing, projection, two-fold
stigmation, the hexapole probe corrector, scan/descan static drives and beam
wobble. Standalone three-fold stigmators are marked unsupported. Ideal Optics
keeps hexapole field controls inactive. Pure beam shift/tilt and calibrated
focus remain distinct from manually changing the listed drives.

## Documentation

See [docs/](docs/) for workflows, model assumptions and development notes, and
[configs/](configs/) for instrument definitions and operating settings.
The [working-point and convergence workflow](docs/working-points-and-convergence.md)
explains portable captured results, constrained alignment, independent sampling
comparisons, resource controls, resumable parameter/geometry experiments and
their current validation limits.

## License

[MIT License](LICENSE).
