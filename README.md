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

## Getting started

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

## Documentation

See [docs/](docs/) for workflows, model assumptions and development notes, and
[configs/](configs/) for instrument definitions and operating settings.
The [working-point and convergence workflow](docs/working-points-and-convergence.md)
explains portable captured results, constrained alignment, independent sampling
comparisons, resource controls, resumable parameter/geometry experiments and
their current validation limits.

## License

[MIT License](LICENSE).
