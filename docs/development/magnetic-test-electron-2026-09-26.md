# Virtual electron trajectories in electric and magnetic fields — 2026-09-26

## Use

Open **Ray Diagram → Magnetic field**, select **Electron trajectories**,
then use **Electrons…**, or open **View → Virtual electrons**. Selecting the mode
with the mouse opens this dockable panel. Drag its title bar to place it beside
or in a tab with Instrument setup or Live tuning; it can also float separately.
Closing the panel retains the list and calculated trajectories. Workspace
layouts retain its placement. Narrow panels stack the list above the scrollable
editor; wide panels use two columns. Advanced numerical controls remain inside
that editor. See the [dock and response verification](virtual-electron-dock-response-2026-09-26.md).

The electron list and selected electron's editor are separate. **Add electron**
creates another tip-default electron; **Duplicate** copies the selected
parameters; **Remove** deletes the selected list entry. Each entry retains its
own parameters, stable identity, colour, calculation status and actual endpoint.
Removing all entries leaves an empty view; adding another starts at the tip.

**Selected electron** displays the active row only. **Overlay checked electrons** displays
the checked rows, using matching colours in the list and plot legend. Selecting
a row changes the editor and highlights that path; it does not change which
rows are checked. In overlay mode an unchecked row can be edited while its path
remains hidden. Checking it requests any missing calculation. Renaming, row
selection and visibility changes preserve existing trajectories.

The default electron starts at the captured physical tip, with its local
emission energy. For the default flat tip this is **XYZ=(0,0,0), 0.3 eV**,
zero polar angle and the full supported column path. The energy input is in
**eV**, not the nominal accelerating voltage or the eventual exit energy.
**Reset to tip emission** restores the captured tip position, emission energy
and forward direction. It does not modify the instrument source.

Editable inputs are initial kinetic energy (eV), initial X/Y (µm), initial Z
(mm), polar angle (mrad from +Z, with the equivalent degrees beside it), azimuth (degrees from +X towards +Y), and
maximum travelled path (mm). Numeric entries and sliders control the same
values. X/Y sliders cover ±0.01 µm (±10 nm), while numeric entries permit a
wider range. Energy and path sliders use logarithmic spacing. A 90° polar
angle launches transversely and 180° launches upstream. These are physical
3D directions, not angles in the enlarged drawing. One degree is approximately
17.4533 mrad; the maximum polar angle is pi × 1000 mrad.

For near-axis tuning, the **polar slider covers 0–5 mrad**, with 0.0025 mrad
per increment; the number editor steps by 0.01 mrad. Larger explicitly entered
angles remain available and are not silently clipped when switching electrons.
Polar angle controls the tilt away from +Z. Azimuth controls the initial tilt
direction in X/Y: 0° towards +X, 90° towards +Y. At zero polar angle the azimuth
does not change the launch direction. It is separate from Ray Diagram projection.

The energy readout also reports **Physical angle: initial → final mrad**, using
the integrated 3D direction. Its tooltip gives the maximum angle along the
calculated path. During calculation the endpoint is labelled live. A zero
velocity has no defined direction. The plot footer explicitly says angles are
not to scale: transverse enlargement does not change physical positions or
these direction readouts.

Dragging a slider samples new settings continuously. The previous trajectory
stays visible as a dashed comparison while the solver publishes actual accepted
path points. A live prefix has a circular moving endpoint; a complete current
trajectory has its usual square endpoint. Earlier parameter samples are labelled
as such, and only a finished result matching the current settings becomes current.
Release requests the final value. Numeric typing still commits on Enter or focus
loss. No geometric interpolation invents paths between calculated samples.
See the [continuous response verification](continuous-electron-response-2026-09-26.md)
for measured response, compilation overhead and numerical limits.

**Start at view centre** explicitly copies the current axial view midpoint
into initial Z. It leaves the other diagnostic inputs unchanged. An edited
position is never silently shifted into the field. Subsequent captured
snapshots preserve user edits but invalidate their trajectories; Reset to tip
emission applies the new captured tip defaults when desired.

Each coloured curve follows one electron in chronological order. A circle
marks its start and a square its endpoint. Separate paths are never connected.
The plot status and parameter editor report the selected electron's
initial/final kinetic energy, travelled path, time of flight and step count.
The parameter window also reports the energy range along the path. Extraction
and acceleration can change kinetic energy; a static magnetic force does no
work. Background magnetic field lines are optional, and their +B arrows need
not follow electron motion.

Projection angle follows Ray Diagram. Physical **Axial Z** and **Projected U**
axes show automatically selected length units. Here U = X cos(theta) +
Y sin(theta), with the Ray Diagram projection angle theta. Internal range
values are millimetres, while stored XYZ positions remain metres. Display
enlargement is excluded from the tick values and reported separately in the
footer.

**Link Ray Diagram**, enabled initially, shares the visible Z and U intervals
in both directions. Disable it for independent pan/zoom; the current view is
retained. Re-enabling it adopts the current Ray Diagram ranges. This choice
is saved with the workspace layout. The 2D combined magnetic-field profile
retains its tesla Y axis and only links axial position.
Both field presentations also align their actual horizontal plot boundaries
with Ray Diagram, including after resizing, scrolling and dock changes. Thin
electron curves and brighter magnetic lines share the same physical scale;
magnetic arrowheads have a fixed screen size and do not stretch with the axes.

Wheel over the plot to zoom both directions around the pointer. Wheel over
the horizontal or vertical axis to zoom only that direction, with the same
wheel factor and anchoring as Ray Diagram. Dragging an axis pans only its
direction. Fit, pan, zoom and projection reuse stored trajectories. They do
not change any physical coordinates or launch a calculation. Switching to
ordinary 3D field lines removes the electron overlay and preserves its cache
for return. Independent ranges survive mode, background and window-size changes.
Fit includes visible magnetic lines and electrons together. If a manually
chosen transverse range is too narrow for the magnetic lines, Fit reveals the
combined extent; the magnetic geometry is never independently rescaled.

**Advanced numerical settings** exposes the maximum spatial step, step
budget, relative tolerance and position tolerance. Defaults are **1 mm**,
**20,000 steps**, **1e-4** and **0.001 nm**, respectively. Local field scales,
forces and error control may require smaller actual steps. A smaller maximum
step or tighter tolerance permits convergence checks; exceeding the budget
is explicitly reported as an incomplete path.

## Physical scope

These are independent classical test electrons in frozen electric and
magnetic fields, without electron-electron interactions or a space-charge solve.
Their initial parameters never become a microscope source,
transport checkpoint, particle archive or input to the main particle run.
The instrument's source, upstream state and current calculation are unchanged.
Coherent development remains paused.

The electric provider includes the existing tip/extractor field,
electrostatic gun focusing, every accelerator electrode and the declared
grounded downstream enclosure. An installed velocity selector contributes
its existing electric potential and E field, including fringe components;
its B contribution is counted once in the combined magnetic scene. Magnetic
lens, stigmator, corrector and deflector fields retain their existing model
and validity restrictions.

Physical tip return, annular electrodes, body bores, grounded liner walls and
apertures can terminate the electron at their first contact. The C1 mechanism
uses its active circular or slit transmission rule; an inactive circular
setting does not override an open slit. Interception is chronological and
supports forward, backward and transverse trajectories.

Specimen and detector interactions are excluded. This deterministic field
trajectory does not execute residual-gas scattering, specimen signals,
detector readout, or dynamically varying scan/blanking drives. Time-dependent
hardware inputs are frozen at the captured instant. The main calculation's
corresponding capabilities and controls are unchanged.

There are explicit limits on fields that have no matching continuous model:

- The bent post-column energy filter is outside the straight-column field
  coordinates. When installed, this diagnostic stops at its entrance.
- An active electrostatic beam blanker with only an integrated kick model has
  no continuous E provider for this trajectory. The diagnostic stops at the
  plate entrance, or rejects preparation if that entrance is at the tip.
  Its field is not replaced by free drift or an invented force.
- Existing column magnetic deflector fields may be equivalent finite-coil
  models. Their signed integrals are bound to the captured reference momentum
  and control time. Changing the virtual electron's energy changes its
  response to that fixed B; it does not recalibrate B.
- Missing radial or registered-map support of an active magnetic source is
  unknown and stops the trajectory. Only declared axial gaps with no active
  source are treated as zero B. A field-domain stop is not a physical wall.

## Electric-field preparation and numerical provenance

The controller prepares the electromagnetic scene only when the electron
view becomes active. It uses an isolated gun clone, preserving shared
immutable field products where available. Existing dependency-bound memory
and disk caches are reused. A cache miss may require a one-time electric
field solve in the background; changing only electron parameters does not
solve the field again.

For the default flat or continuously curved tip, the requested full-column
path can extend the numerical electric domain through the same physical
grounded liner. Electrode positions, voltages and physical enclosure inputs
are retained. This is a newly resolved numerical field on the extended mesh,
**not the exact original short-domain array**. The solver does not replace
the field by zero at the old gun exit or force a target final energy.

A bounded comparison used 901 points on the shared on-axis tip-to-gun-exit
interval, with the same mechanics and voltages. Extending the field's
numerical endpoint from 550 to 3026.4 mm changed potential by at most
**58.5018 V** and axial E by at most **16751.2 V/m**, or **1.177%** of the old
peak axial field. The potential difference at the gun-exit point was below
6e-11 V. This measures sensitivity to the changed numerical boundary/mesh;
it is not proof that the two fields are interchangeable, an off-axis error
bound, or full mesh/domain convergence. Evidence is retained in
`tmp/test-electron-electric-domain-comparison.json`.

## Integration, termination and reuse

`test_electron_scene.py` combines the captured B providers, complete gun E
provider, scalar potential and hardware intersections. SI coordinates are
right-handed global XYZ, with +Z downstream, E in V/m, B in tesla and potential
in volts. The tip launch uses its actual surface convention; there is no
arbitrary downstream launch offset.

`magnetic_test_particle.py` integrates
`dx/dt = p/(gamma*m)` and `dp/dt = -e*(E + v cross B)` with a symmetric
Gonzalez discrete-gradient step and step-doubling error control. It preserves
the static electron invariant **K_eV − phi_V** to iteration tolerance without
momentum rescaling or energy reset. Absolute invariant error in eV and
relative error are recorded separately from physical changes in kinetic
energy. A pure-B specialization uses the symmetric Boris update.

The calculation permits zero longitudinal velocity and reversal. Elapsed
time is accumulated from accepted time steps; travelled distance integrates
changing speed with Simpson quadrature. Neither is derived from the rendered
polyline. Step control resolves electric impulse near sub-eV emission,
gyromotion, source support and local field resolution. Grid-face handling
avoids repeated vanishing steps without stepping over a physical field
region. Hardware contacts are refined on the two chronological half-steps.

The status distinguishes requested path completion, field-domain exit,
unsupported-field boundaries, hardware/aperture interception, return to the
tip, invalid initial position, step-budget exhaustion and numerical failure.
A truncated or cancelled trajectory is not reported as a complete requested
path. No diagnostic result is automatically archived as a microscope run.

`gui/magnetic_test_electron.py` uses one numerical CPU worker under shared
process admission. Each record owns its latest executed trajectory. An additional
eight-entry cache holds recent exact scene/settings combinations, allowing
identical duplicates and restored parameters to reuse a calculation. Cache
eviction never removes the current trajectory of an existing list record.
Records are held for the application session, not added to microscope archives.

Parameter edits coalesce for 60 ms independently for each record. Keyboard
entry commits on Enter or focus loss; sliders submit samples while held, with
up to a 350 ms execution window before a newer value supersedes that sample.
This includes backend admission and display latency so continuous input cannot
starve real path updates. Actual accepted prefixes are published periodically.
The previous same-field trajectory stays visible separately, labelled and dashed.
Enter does not activate Reset to tip or another action. Current results require
the matching scene generation, stable record ID, revision and parameters; earlier
active samples may appear only as explicitly labelled previous previews.
Changing fields or deleting a record rejects its old samples. Editing one electron invalidates only that record;
removing a record discards its unfinished work. A changed captured field clears
all trajectories while preserving the list and user parameters. Failed attempts
remain visible and do not block calculations for other electrons or retry in a
loop. Only the selected electron or checked overlay entries are queued.
Hiding the view cancels pending publication and tracing. An
already-running field solve may finish before its cancelled result is
discarded. Preparation/provider errors are visible without an automatic
retry loop.

Field-line and electron caches are independent. Electron parameter changes
do not rebuild field-line geometry; density and colour-reference changes do
not retrace the electron. Camera operations reuse cached coordinates. The
software QPainter canvas adds no OpenGL dependency.

## Multi-electron integration evidence

The final targeted run passed **243 checks**, with zero failures, errors or
skips, in 71.424 s: 43 solver, 25 scene, 34 controller GUI, 55 canvas and 86
related field/application checks. XML evidence is
`tmp/multiple-test-electrons-regression.xml`. Compilation and whitespace
checks passed. The same test command in the baseline section below includes
these expanded suites. This is affected-feature validation, not a full-project
qualification.

GUI coverage includes real Add/Duplicate/Remove button dispatch, separate
parameter records, exact-result reuse, overlay membership, deleting or hiding
in-flight records, serial queue continuation after a failure, stale snapshot
rejection, and per-record edit delays. Existing records keep their results
after the extra history cache evicts them. Tests reproduced and repaired a
precision defect: a full widget roundtrip during an energy edit had rounded
unmodified tip coordinates and numerical settings. Every control now updates
only its corresponding stored field or XYZ coordinate, retaining untouched
values exactly. Resetting an already matching tip state is also a no-op.

The local GUI exercise `tmp/render_multiple_test_electrons.py` uses the actual
default configured E+B providers and the unchanged relativistic trajectory
solver. Three tip-origin electrons each travelled 550 mm, starting at 0.3 eV:
polar/azimuth angles (5°, 45°), (10°, 45°), and (5°, 135°). All reached the
requested path with 3,566 accepted steps and approximately 300000.3 eV final
kinetic energy. Maximum static total-energy errors remained below 1.2e-8 eV.

Instrumentation recorded **one electromagnetic-scene preparation and exactly
three trajectory executions**. Duplicating the first electron before editing
its angle reused its exact result object. Row selection, switching between
selected/overlay modes, hiding/revealing the second path, renaming and rotating
the view added no trajectory or field executions. All three record results
remained available after these operations.

The initial GUI preparation plus first path took 11.641 s; the next two new
parameter sets took 9.765 s and 9.547 s. These are local one-thread diagnostic
measurements, not full particle-population benchmarks or guarantees for other
inputs. Inspected screenshots show matching list/plot colours, per-electron
markers and a single-row legend. The drawing retained 709/760 px height in the
large view and 299/350 px at 900×350. Evidence is local under
`tmp/multiple-test-electrons-{render.json,list.png,overlay.png,selected.png,compact.png}`.

## Baseline E+B validation, before the multi-electron list

Final affected-feature validation: **213 checks passed**, zero failures or
skips: **43 solver tests** in `tmp/test-electron-em-solver-final.xml` and
**170 scene/GUI/display checks** in `tmp/test-electron-em-gui-regression.xml`.
These are targeted checks, not full-project or complete-microscope
qualification. Run with the project virtual environment, offscreen Qt and
one numerical worker:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH='src;.'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TEMSIM_CPU_THREADS='1'
.venv\Scripts\python.exe -m pytest -p pytestqt.plugin tests/test_magnetic_test_particle.py tests/test_test_electron_scene.py tests/test_magnetic_test_electron_gui.py tests/test_magnetic_field_3d.py tests/test_magnetic_field_canvas.py tests/test_magnetic_field_lines.py tests/test_magnetic_field_scene.py tests/test_incremental_magnetic_scene.py tests/test_lazy_ray_panels.py tests/test_scientific_output_gui.py tests/test_gui_shell.py::test_magnetic_field_and_ray_diagram_share_the_axial_axis tests/test_gui_shell.py::test_ray_plot_marks_every_component_centre_and_detected_crossover -q --tb=short
```

Numerical coverage includes signed uniform-B curvature, relativistic radius,
constant magnetic energy, independent nonuniform-field DOP853 comparison,
step convergence, sub-eV electric acceleration, analytic work/flight time,
retarding-field turns, crossed fields, hard-stop refinement, thin fields,
native map resolution, field-domain exit and cancellation. Scene tests cover
electric/B composition, optional velocity-selector fields, physical first
contacts, C1 slit selection, unsupported fields and unchanged source inputs.
GUI tests cover eV conversion, physical tip defaults/reset, variable-energy
readout, lazy preparation, stale/cancelled jobs, exact-cache reuse, errors,
debounce and compact projection/mode integration.

The final actual default on-axis diagnostic used the configured electric and
magnetic providers from the physical tip to **3.0264 m**, with initial kinetic
energy **0.3 eV** and the default 20,000-step budget. Both runs completed the
requested path:

| Maximum step | Runtime, one worker | Accepted steps | Final kinetic energy | Maximum invariant error |
|---|---:|---:|---:|---:|
| 1 mm | 9.319 s | 8,799 | 300000.3000000185 eV | 1.962e-8 eV |
| 0.5 mm | 10.857 s | 10,191 | 300000.2999999852 eV | 1.484e-8 eV |

Both paths remain on axis, as expected for the default symmetric launch.
Their time of flight agrees to the quoted **14.193718 ns**. That close
agreement is specific to this on-axis case and does not establish off-axis
convergence or certify the full instrument. The recorded **0.178 s** scene preparation reused existing
field products and is not a cold-cache solve benchmark. Arbitrary launch
angles, tighter tolerances, curved tips, mapped fields and first-time solves
can take longer. Continuous controls stay responsive through background
execution, but a full new tip-to-column trajectory is not an instant update.

An actual offscreen Qt run completed the initial default view, including
profile, field lines, electromagnetic scene and trajectory, in **19.297 s**.
A second tip launch with polar angle **5°**, azimuth **45°** and a **550 mm**
path completed in **11.125 s**, with 3,566 steps and maximum absolute X/Y
excursions of **16.98/18.00 µm**. Its final kinetic energy was
300000.30000000645 eV, with invariant error 1.11e-8 eV. Changing only initial
energy to **0.6 eV** completed in **7.437 s**, gave
300000.60000001057 eV, and changed flight time from approximately
3556.1 to 3554.9 ps. The same prepared scene was reused for all these edits.
Reset returned to the tip position, 0.3 eV and forward direction while
preserving the user's chosen path length.

The inspected Qt views retained a **709/760 px** canvas height at the larger
size and **299/350 px** in the compact view (93.3% and 85.4%). The GUI report
and pictures are `tmp/test-electron-em-render.json` and
`tmp/test-electron-em-{default,controls,angle,compact}.png`. These measurements
use the actual configured fields and diagnostic solver, not synthetic worker
results. The local render report stores numeric energies independently of
the text-label formatting; the final labels retain enough digits to show the
0.3/0.6 eV contribution above the accelerating energy.

Evidence: `tmp/test-electron-em-actual.json`,
`tmp/test-electron-em-step-0.001.npz`, `tmp/test-electron-em-step-0.0005.npz`,
`tmp/test-electron-electric-domain-comparison.json`, and the two XML reports
above. Generated numerical arrays and screenshots remain local under `tmp/`
and are excluded from Git. Restart the application to load the interface.
