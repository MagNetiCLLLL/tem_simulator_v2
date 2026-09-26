# Combined magnetic-field views — 2026-09-26

## User workflow

Open **Ray Diagram → Magnetic field**, then choose **2D field strength** or
**3D field lines**. Both show the captured combined field. Selecting a component
elsewhere does not isolate its field or change the displayed population.

The main controls occupy one row: display mode, 3D line density, Fit and
Advanced. Map import, source/reference metadata and display-reference settings
are in Advanced. The drawing fills the remaining area. The original per-lens
plot selector and independent field-camera rotations have been removed.

Rotate **Ray Diagram** to rotate the field projection. Both use
`U = X cos(theta) + Y sin(theta)`, with Z horizontal and positive U upwards.
The orthogonal transverse coordinate remains depth, not an independent orbit.
Ray Diagram's visible axial range also controls the field view. Dragging the
3D field pans; the wheel zooms; Fit resets that local pan/zoom. Transverse
coordinates automatically fill the available height, with the actual display
enlargement disclosed in the footer. This changes no physical field geometry.

3D colour and seed density use the same fixed, six-decade logarithmic |B|
scale. Weak transverse fields remain visible alongside stronger lens fields.
The reference initializes once from the combined sampled field and stays fixed
across new snapshots. Advanced → Use current peak explicitly changes it.
Keep this reference unchanged when comparing hardware adjustments. Density is
qualitative and budget-limited, not a calibrated number of webers per line.
Arrows follow +B; these are magnetic field lines, not electron trajectories.

2D shows combined Bz, projected Bu/Bv and the RMS transverse field on a small
transverse ring. An ideal stigmator can have zero on-axis field and a nonzero
ring RMS; displaying Bz alone would conceal it. The diagnostic records the
actual local ring radius and validity, including any reduction required by
finite map support. Projection changes reuse sampled Cartesian vectors.

## Field ownership and physical limits

`magnetic_field_scene.py` combines enabled captured round-lens fields,
stigmators, magnetic corrector multipoles and deflectors. Existing finite gun
magnetic providers are used directly, including blanking and an installed
crossed-field velocity selector's magnetic contribution. Dynamic column
scan/deflector drives are sampled at the captured simulation time.

Column deflectors currently transported as instantaneous angular kicks use
explicit **display-only equivalent finite-coil fields**. Their signed field
integrals reproduce those kicks at the captured reference momentum, using the
configured effective thickness. This does not infer a fringe profile or
replace particle transport. Layout-only virtual controls have no independent
magnetic field. Electrostatic extractor, gun lens and accelerator fields are
not misrepresented as magnetic fields. The post-column energy filter's bent
coordinate system is outside this straight-column scene.

Joint nonlinear lens circuits are counted once. Existing finite registered
maps retain their valid volume. Analytic fields are limited to their near-axis
validity region; lines do not reconstruct unmodelled return fields in yokes.
Overlapping nonzero analytic contributions must all be valid; a line cannot
continue using a silently incomplete sum. Exactly unpowered analytic controls
do not impose an artificial smaller validity radius on neighbouring fields.

Generated FEM providers require a current dependency-checked captured cache.
3D display does not initiate a FEM solve or an electron calculation. Missing
or stale provider state is reported explicitly. No source inputs, particle
transport, detector participation or coherent physics are changed here.

## Rendering and execution

`magnetic_field_lines.py` uses bounded bidirectional RK2 integral curves of
`dr/ds = B/|B|`. Deterministic section seeds sample the **total** field; no
component receives an independent strength normalization. The default budget
is 640 candidate lines with 64 integration steps per direction. Lines stop
at domain boundaries, zero/weak fields or the display-step limit.

The 2D profile and 3D lines are prepared asynchronously using one numerical
CPU thread and shared process admission. New snapshots cancel pending stale
work. Geometry uses a bounded four-entry memory cache. Switching views,
projection rotation and range changes reuse completed data; they do not run
upstream particle calculations. The software canvas uses cached colour-batched
paths, collision-aware labels and bounded arrows, without mandatory OpenGL.

## Validation

Final combined run: **113 passed, zero failed, zero skipped** (51.965 s).

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH='src;.'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TEMSIM_CPU_THREADS='1'
.venv\Scripts\python.exe -m pytest -p pytestqt.plugin tests/test_magnetic_field_3d.py tests/test_magnetic_field_canvas.py tests/test_magnetic_field_lines.py tests/test_magnetic_field_scene.py tests/test_incremental_magnetic_scene.py tests/test_lazy_ray_panels.py tests/test_scientific_output_gui.py tests/test_gui_shell.py::test_magnetic_field_and_ray_diagram_share_the_axial_axis tests/test_gui_shell.py::test_ray_plot_marks_every_component_centre_and_detected_crossover -q --tb=short --junitxml=tmp/magnetic-combined-final.xml
```

Coverage includes signed field/kick equivalence, exact multipole composition,
zero-axis stigmator fields, captured scan time, gun blanking and crossed-field
magnetic providers, no-FEM admission, zero-control domain preservation,
fixed-reference weak-field visibility, registered maps and joint ownership.
GUI checks cover cached angle/range changes, async cancellation/stale rejection,
one-row controls, Advanced controls, first-3D-to-2D fitting, continued map-import
behavior and the actual MainWindow selection/diagnostic update path.

Offscreen screenshots were inspected at 1280×760 and 900×350, with configured
lens providers plus nonzero stigmator, beam-deflector and captured scan drives.
The screenshot script evaluates fields; it does not claim a new electron run.
The last run prepared the profile and 241 lines / 19,329 segments in **1.985 s**
on one numerical CPU thread. Its fixed reference was 1.586939714 T. Canvas
heights were **709/760 px (93.3%)** and **299/350 px (85.4%)**, including the
canvas's single footer. This is a display preparation measurement, not a full
simulation speedup or an end-to-end frame-rate claim.

A separate 20,666-segment canvas measurement found cached angle reprojection
6.47–8.14 ms and range clipping/reprojection 2.45–8.25 ms. These software-only
measurements preceded the final exactly-zero component domain fix; the final
GUI screenshot timing above used the final implementation.

Local evidence:

- `tmp/magnetic-combined-final.xml`: final 113-test result.
- `tmp/render_magnetic_combined.py`, `tmp/magnetic-combined-render.json`.
- `tmp/magnetic-combined-full.png`, `tmp/magnetic-combined-compact.png`.
- `tmp/magnetic-combined-2d.png`, `tmp/magnetic-combined-advanced.png`.
- `tmp/magnetic-combined-active-check.json`: active component field values.
- `tmp/magnetic-projection-canvas-timing.json`: cached canvas timing.

Compilation and `git diff --check` passed. This was an affected-component and
GUI integration run, not the complete project suite or full microscope
qualification. Generated images and numerical arrays remain local under
`tmp/`. Restart the application to load these interface changes.
