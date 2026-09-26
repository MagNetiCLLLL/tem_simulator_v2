# TEM Simulator v2 — Project Handoff

Last updated: **2026-09-26**. Current checkpoint:
**classical particle transport and qualitative scientific trends**.

Latest continuation: **Fine polar-angle controls and small-angle audit**
([evidence and scope](docs/development/virtual-electron-polar-angle-audit-2026-09-26.md)).
The polar slider now covers 0–5 mrad with 0.0025 mrad increments; numeric steps
are 0.01 mrad and larger explicitly entered angles remain intact. The existing
energy panel reports initial/final physical direction angles and the plot footer
states that angles are not to scale. Polar/azimuth tooltips distinguish initial
direction from projection rotation. 168 affected GUI/display checks pass.

Full GUI startup reconstruction matches the screenshot's 0.866371 mrad case:
9,512 steps, Z3026.392986 mm, TOF14193.718 ps. Its actual final direction angle
is 1.813557 mrad and radial displacement 1.216858 mm. The 294x drawing ratio
makes the projected final angle appear about 27.3 degrees. Small-angle response
is approximately linear; tighter integration changes final radius by about
0.293%, and compiled/reference sampled positions agree within 1.64e-13 m.
No numerical or physical model changes were needed for this control repair.
See the audit for the bounded scope and difference from bare default_state.

Previous continuation: **Continuous virtual-electron trajectories and mrad controls**
([verification and scope](docs/development/continuous-electron-response-2026-09-26.md)).
Sliders now publish actual integrated prefixes and completed sampled settings
while held. The previous same-field path stays visible as a dashed comparison;
stale/partial paths never become exact current results. Release requests the
latest value. The primary polar input is mrad with adjacent equivalent degrees.
Numeric typing still commits on Enter/focus loss. Captured fields, process and
exact-result caches are reused; unchanged magnetic backgrounds now reuse one
raster layer across electron updates.

Supported captured E+B models use compiled field sampling, adaptive integration
and chronological hardware contacts at the original tolerances. Reference
comparisons preserve step counts, stopping reasons and energy invariants.
Unrepresented/custom field laws and geometry retain the complete reference path;
exact-type and original-method checks prevent incorrect Gaussian substitution.
Pure warm integration measured 7.52 s to 0.202 s for the default full path and
4.69 s to 0.096 s for the 2.61-degree aperture case. First compilation remains
an extra startup cost; coherent development remains paused.

Actual Qt dock/process/canvas test: 80 distinct slider inputs over 1.574 s,
6 complete path updates plus 7 partial path updates, zero empty display states,
and 0.200 s from release to the final latest-settings trajectory. The 10 ms
timer's p95/max gaps were 23.1/39.6 ms. One process and field preparation were
reused and the process exited on close. These are measured diagnostic cases,
not a monitor frame-rate guarantee or full-instrument qualification.

Final combined affected-feature regression: **447 passed**, zero failures,
errors or skips, 199.494 s (`tmp/continuous-electron-final.xml`). Includes delayed
IPC input-starvation, exact-cache races, mrad extrema/layout, field-raster cache
invalidation, real compiled/reference fields, hardware stops, cancellation,
process lifecycle, docking/navigation and CPU admission. Compile and diff
checks passed. Local evidence: `tmp/continuous-electron-live-20260926/`.
Restart the application to load changes. Generated reports/caches remain local.

Previous continuation: **Dockable virtual electrons and responsive field display**
([verification and scope](docs/development/virtual-electron-dock-response-2026-09-26.md)).
Virtual electrons is a native dock with View-menu access, tabifying/floating,
named-layout and restart restoration. Narrow panels stack their list and
scrollable editor; closing the panel preserves cached trajectories. Linked
spatial/2D field axes now match the Ray Diagram's actual horizontal pixels
through dock, sidebar, scroll and resize changes. Electron strokes are 1.0/1.4
pixels; magnetic lines use 75% opacity and fixed-size screen arrowheads, removing
stretched arrow spikes. Fit retains a common physical scale for both populations.

Captured-field preparation and unchanged E+B transport now use one persistent,
hidden, single-CPU process. Parent-side numerical admission still serializes it
with other numerical jobs; scene/result caches avoid repeat work. Two real
transport cases retained bitwise identical saved arrays. The measured Qt timer
maximum gap changed from about 39–40 ms to 16–18 ms; solve time was similar,
and first use includes process startup. This improves responsiveness without
lowering physical or numerical accuracy. An actual MainWindow diagnostic run
reused one field preparation for two trajectories and needed no additional
execution for overlay, docking, floating or resize; axial pixel errors were
below 0.02 logical pixels. These are diagnostic and GUI checks, not full-column
scientific qualification. Restart the app to load these changes.

Final affected-feature regression: **332 passed**, zero failures/errors/skips,
178.143 s (`tmp/virtual-electron-dock-response-final.xml`). Thirteen process
tests cover exact parity, cold startup, scene ownership, CPU admission,
cancellation/recovery and the actual Windows compute-process PID/exit. A final
MainWindow run confirmed the same PID ownership and exit, two real E+B paths,
one shared prepared scene and no layout-triggered integration. Compile and
diff checks passed. Generated reports/arrays/screenshots remain local; no
commit or push was requested.

Previous continuation: **Linked trajectory axes and faster parameter response**
([verification and scope](docs/development/test-electron-navigation-performance-2026-09-26.md)).
Spatial field/electron plots now have physical Z/U axes, Ray Diagram wheel
anchoring and single-axis zoom/pan. **Link Ray Diagram** is optional, works in
both directions and is retained with workspace layouts. Independent ranges
survive resizing and new trajectories; Fit establishes an explicit range.
The 2D field-strength axis remains in tesla.

Numerical edits commit on Enter/focus loss and slider release, avoiding work
for intermediate values. Enter no longer triggers Reset to tip accidentally.
Equivalent scalar field interpolation, axial hardware candidate filtering
and dependency-complete Gaussian peak caching reduced the measured default
3.0264 m integration from 13.219 to 7.132 s and a 550 mm off-axis integration
from 7.919 to 3.794 s. All saved trajectory arrays were bitwise identical.
These timings exclude first field preparation; numerical settings and physical
models are unchanged. Restart the application to load the modified interface.

Final affected-feature regression: **309 passed, one existing default-provider
assertion deselected**, 97.15 s. The excluded failure was independently reproduced
with all three old numerical implementations restored. Real Qt plot/axis wheels,
linked/independent navigation and resize passed a stored-trajectory replay;
readable captures and exact array digests are retained locally. Compilation and
fatal-error lint checks passed. See the linked verification note for evidence
paths, scientific limits and the excluded test. No commit or push was requested.

Previous continuation: **Multiple virtual electrons and trajectory overlays**
([guide](docs/development/magnetic-test-electron-2026-09-26.md)).
The Magnetic field menu now offers **Electron trajectories → Electrons…**.
An electron list supports adding tip-default entries, duplicating a selected
entry, editing its independent parameters, renaming it and removing it.
**Selected electron** shows one row; **Overlay checked** shows checked rows in
matching colours with independent start/end markers and direction arrows.
Clicking a row selects its editor without changing overlay membership.

One captured E+B scene serves the list. A serial one-thread queue calculates
only requested missing paths. Each record retains its current result; an
additional eight-entry exact-settings cache permits reuse between identical
electrons. Row selection, visibility and name changes do not retrace physics.
Edits invalidate only their own record. Stable IDs, per-record revisions and
scene generations prevent deleted or stale work from being published. A new
captured field clears all results while preserving list parameters. These
remain independent test particles without electron-electron interactions;
the existing E+B solver, tip defaults and hardware stops are unchanged.
The list is session-local and does not create production particle archives.

Final validation: **243 targeted checks passed**, zero failures/errors/skips
(43 solver, 25 scene, 34 controller GUI, 55 canvas and 86 related checks).
An actual configured E+B GUI run executed three distinct 550 mm tip-origin
paths with **one field preparation and three trajectory calculations**.
Exact duplication, row selection and visibility changes reused the results.
Screenshots at 1280×760 and 900×350 were inspected; the canvas retained 93.3%
and 85.4% of panel height. Local evidence is under `tmp/multiple-test-electrons-*`.
The regression also caught and fixed unrelated controls rounding exact stored
parameters: each edit now updates only that field or XYZ coordinate. No legacy
single-path display adapter remains. Compilation and whitespace checks passed.
Restart the application to load the new interface; no commit or push requested.

Previous continuation: [Virtual electron from the physical tip in E+B fields](docs/development/magnetic-test-electron-2026-09-26.md).
**Magnetic field → Single electron trajectory → Electron…** now starts at the
captured physical tip with its local emission energy: default XYZ=(0,0,0),
0.3 eV and the full supported column path. Initial energy is entered in eV;
XYZ, polar/azimuth direction and path remain editable, and **Reset to tip
emission** restores the tip state. Extraction, electrostatic focusing, all
accelerator electrodes, optional installed velocity-selector E+B, magnetic
optics and physical aperture/electrode/wall stops participate. The readout
shows changing kinetic energy and distinguishes hardware interception,
unsupported fields, field-domain limits and numerical truncation. Specimen
and detector interactions are excluded; this is a deterministic frozen-field
diagnostic, not a residual-gas scattering or time-dependent scan calculation.

The electric provider is prepared lazily on an isolated gun clone, with
existing dependency-bound caches. A full-column diagnostic may extend the
numerical E domain through the existing grounded liner. Physical settings
remain unchanged, but the extended mesh is a different numerical solution,
not the original short-domain field array. The bounded on-axis comparison
found at most 58.5 V potential difference and 1.18% of the old peak axial E
field; see the guide before treating these arrays as interchangeable. The
bent energy filter and an active electrostatic blanker without a continuous
field provider cause explicit upstream stops; no field-free bypass is used.

Static E+B transport uses adaptive discrete-gradient Lorentz integration
without forced exit energy or momentum rescaling. Time and travelled path
are integrated with changing speed. Default controls allow a 1 mm maximum
step, 20,000 steps and advanced tolerances. One numerical CPU thread, 150 ms
input coalescing, latest-only publication, cancellation and eight exact cached
trajectories keep controls responsive. Camera and field-line changes reuse
paths. Main source settings, instrument particle runs, checkpoints and
archives are unchanged; coherent development remains paused.

Final validation: **213 affected checks passed** (43 solver tests plus 170
scene/GUI/display regressions). The actual default on-axis tip-to-3.0264 m
case completed in **9.319 s**, 8,799 steps, from **0.3 to 300000.3000000185 eV**,
on one numerical worker. Halving the maximum step completed in 10.857 s with
10,191 steps; both reached the requested path and conserved K−eφ to below
2e-8 eV. An actual Qt initial view completed in 19.297 s including profile,
field geometry and trajectory. A 5°/45° tip launch over 550 mm completed in
11.125 s, and changing initial energy to 0.6 eV took 7.437 s using the same
prepared scene. The canvas retained 85.4–93.3% of tested panel height; local
GUI evidence is `tmp/test-electron-em-render.json` and matching PNGs.
This is a single-electron diagnostic benchmark, not a full-instrument
simulation or a guarantee for arbitrary launch conditions. Local evidence:
`tmp/test-electron-em-actual.json` and
`tmp/test-electron-em-gui-regression.xml`; generated data remain excluded from
Git. Restart to load the interface. No commit or push in this continuation.

Previous continuation: [Stable instrument navigation layout](docs/development/direct-alignment-layout-2026-09-26.md).
Direct Alignment's variable-height content now scrolls inside the page, with
Cancel fixed below it. Navigation tabs do not pass width-dependent height
requests into the setup layout. Setup area, tab positions and user splitter/
floating-dock dimensions remain stable when switching pages or showing long
messages, including 420 px width. Existing controls and layout identities are
retained; restart requires no layout reset. All 34 targeted layout/alignment
checks passed. Local before/after evidence is under `tmp/alignment-layout-*`.

Previous continuation: [Combined magnetic field views](docs/development/magnetic-field-lines-3d-2026-09-26.md).
Both 2D/3D views now show the combined lens, stigmator, corrector and deflector
field. Per-lens display selection and independent orbit controls are removed.
Ray Diagram rotation uses the same U/Z projection in both field views, with
shared physical axial ranges. A single toolbar leaves 85–93% of tested panel
height for the canvas; map import and reference/details are in Advanced.
2D adds total projected transverse components and finite-ring RMS to expose
zero-axis multipole fields. Captured finite gun providers are reused directly;
column angular kicks use explicitly labelled finite-coil display equivalents.
Electrostatic fields and the bent post-column filter are outside this magnetic
scene. Exactly-zero controls do not truncate neighbouring field domains.
One-thread async sampling shares its frozen scene with the cached 3D renderer;
rotation/pan/zoom do not rerun fields or electron transport. Async diagnostic
completion refreshes the currently selected hardware's parameter information.
Missing/stale FEM cache fails before diagnostic helpers can start a new solve.
Final 113 targeted checks passed (zero failures/skips), including MainWindow
selection, map controls, projection, stale work, field direction and domains.
The final active-field render prepared 241 lines/19,329 segments in 1.985 s on one
numerical thread; this is not a full-simulation benchmark. Final screenshot and
timing evidence remain under `tmp/magnetic-combined-*`. Restart to use the new
interface. Other uncommitted gun repair and hardware tuning work is preserved.
No coherent development, commit or push in this continuation.

Previous continuation: [Production gun-electrode field repair](docs/development/gun-electrode-field-repair-2026-09-26.md).
The default classical flat/continuous-curvature cold FEG now executes a coupled
electrode field with explicit downstream electrical closure. Existing ten
accelerator stages, extractor, gun lens, original tip emission, apertures,
magnetic controls and optional velocity selector remain active. Grounded liner
wiring and selector housing common bias are declared simulator assumptions.
No artificial exit source, drawing smoothing or final-energy reset is used.
Flat mesh/domain qualification and actual curved/selector traces passed;
261 distinct targeted regression checks have final passing outcomes. This is
not full-project or complete microscope qualification. The independent 193-ray
16-to-32 mesh envelope difference is 0.2016%; production step halving changes
maximum exit position by 0.5904 nm. A completed 5,000-particle gun trace took
108.829 s on one CPU thread, with 19.43 ms exact cached-result reuse and
4.831e-9 eV maximum exit energy residual. It is not a full-column timing or an
equivalent old/new speed comparison. Local reports and raw accepted histories
are in `tmp/gun-repair-20260926/`; the before/after figure and independent
five-case evidence are in `tmp/closed-gun-qualification/`.
Snapshots exclude generated field objects, archive electrostatic provenance,
and invalidate execution reuse when consumed field inputs/implementations
change. Obsolete analytic field controls are hidden for geometry fields;
Ray Diagram marks actual accelerator electrodes. Restart and recalculate;
old matched lens/selector drives may need retuning. The historical explicitly
selected advanced tip-surface provider remains separate and was not newly
qualified by the flat outlet study. No coherent work, commit or push.

Previous continuation: [Accelerator turning mechanism and proportional reference](docs/development/accelerator-turning-mechanism-2026-09-26.md).
Twelve new 193-particle tip-to-450 mm cases isolate analytic transition width,
reference contour, mesh, step and outlet sensitivity. Strong repeated transverse
turns arise in the compact analytic stage ramps before rendering; smaller steps
retain them. No longitudinal reversal was observed in these admitted forward
cases. A bounded patent-proportional reference uses six divider nodes and five
intervals, with explicit planar-cathode/extractor-enclosure limitations. Shaped
versus flat intermediate electrodes show resolved physical focusing differences.
The 80-340 mm result is stable to tested mesh/outlet changes, while final energy
remains outlet-sensitive: do not promote the reference to the default provider.
123 related tests plus five renderer/arrival checks passed. Source states, IDs
and weights match exactly; generated archives/figures remain local under
`tmp/accelerator-mechanism-20260926/`. Production/GUI defaults are unchanged.
No coherent work, commit or push. True reflected-ray drawing remains a separate
chronological-history limitation of the common-Z display.

Previous continuation: [Gun electrode potentials and patent cross-section review](docs/development/gun-electrode-patent-review-2026-09-26.md).
The outlet energy variation is explained by the changed potential at the fixed
450 mm observation plane. The default analytic lens amplitude and the diagnostic
electrode voltage have different meanings; the report records both, including
the 0.3 eV final-energy convention difference. Original patent cross-sections
identify extraction, electrostatic control, staged acceleration and grounding.
The next specification separates physical conductors/electrical nodes from the
numerical domain and proposes explicitly wired downstream shielding using
existing geometry. It has not been implemented or qualified. This continuation
changes documentation/reference figures only; no new particle run or default
solver change, and no commit or push.

Previous continuation: [Coupled reference gun field with an explicit planar cathode](docs/development/planar-gun-coupled-field-2026-09-26.md).
An opt-in diagnostic now solves the extractor, gun lens and all accelerator
electrodes in one vacuum field, starting trajectories at the unchanged flat
emission samples. Eleven 193-particle cases and 101 targeted/regression tests
completed. The 16-to-32 mesh envelope difference is 0.23045%; executed cutoff
save/load/continuation checks at 26/30/34 mm passed. Static field caches are
compressed and dependency checked. Production defaults and GUI are unchanged.
Do not promote this model yet: extending the fixed-potential outlet changes
the energy observed at 450 mm by up to 9.518 keV despite good energy conservation.
Cathode geometry and outlet electrical closure/shielding still require explicit
definitions. Diagnostic archives are not GUI result files; generated arrays
remain ignored under `tmp/planar-gun-20260926/`. No commit or push performed.

Previous continuation: [Independent Hardware tuning tab](docs/development/hardware-tuning-tab-2026-09-25.md).
The new main-workspace tab has a searchable alignment list on the left and
inline hardware controls on the right. Its 27 task entries resolve to current
instrument objects, with functional names, explicit units and availability.
It is independent of Direct Alignment and introduces no automatic solver or
new physical state. Scalar and component validation happen before live edits;
accepted changes use existing invalidation/preview and synchronize other editors.
All 97 targeted cases passed, including five invalid-drive regressions first
reproduced before the validation fix. Compact and main-window screenshots were
inspected. The prior default-cache native crash recurred in an existing test;
the full affected test group passed with an isolated compilation cache. Details
and scope are in the linked note. Restart to load the new tab. No commit/push
or user-application restart was performed for this feature.

Previous continuation: [Condenser adjustment and accelerator annotations](docs/development/condenser-adjustment-display-2026-09-25.md).
The Ray Diagram action is now Auto-adjust condensers, with captured C1/C2/C3
before/after values and explicit optical-validation scope. A persisted
Acceleration gaps toggle draws snapshot-based stage annotations without changing
trajectories. Completed optical validation supplies its actual straight-column
extent; it does not acquire a resumable particle checkpoint. All 116 distinct
targeted cases passed. A native 193-source-sample candidate passed application,
undo and GUI publication using an isolated compilation cache; both screenshots
were inspected. The note records the earlier default-cache native crash and
verification limits. No solver smoothing, coherent development, commit/push or
user-application restart was performed. Restart to load the changes.

Previous repair: [Match transport result signatures](docs/development/match-transport-signatures-2026-09-25.md).
The transport result producer now copies immutable manifest signatures into the
current result dictionary contract. Two new tests reproduced the user's exact
error before the fix; all 33 targeted checks pass afterward. No physical solver
or compatibility policy changed. Restart the application to load the repair.

Previous continuation: [Shared beam plot sizes and ray colours](docs/development/unified-beam-colours-2026-09-25.md).
The two transverse plots share fixed dimensions and axis gutters. Plot choices
now synchronize Ray Diagram colours in both directions. Source/angle colours
stay fixed per emitted identity; TOF uses a saved cumulative-time gradient with
one common result-wide palette, while relative arrival delay remains in hover.
The distance-versus-time clarification was left open; cumulative time was the
explicit implementation assumption. Geometric path length was not added.
Gradient rendering retains stop clipping, missing-clock grey paths, rotation
and the existing screen-detail caches. Numerical transport was not changed.
Focused validation covers 252 distinct passing cases; see the linked note for
scope and the existing old-archive source-identity rejection. No commit/push
or user-application restart was performed. Coherent work remains paused.

Previous continuation: [Working points inside Live tuning](docs/development/live-tuning-working-points-2026-09-22.md).
The independent Working Points workspace tab has been removed. Live tuning now
contains Calculation and Working points subpages; infrequent record tools,
captured details and convergence checks are in a default-collapsed Advanced area.
Calculation has one Open/Export pair and scrollable cutoff/range controls. The
same records, signals, restoration and continuation paths remain connected.
New/reset Results layouts open the nested Working points page, while existing
custom layouts are preserved. All 190 distinct related tests passed. Source and
isolated installed GUI checks each passed 16/16 with calculations forbidden;
all four tested page states fit 900 × 700. The final wheel matches all 449 current
Python modules. Compilation and whitespace checks passed. This is targeted
verification; the prior complete-suite live-slider timing issue below remains
unresolved. Restart to use the changed interface. No commit or push was made.

Previous continuation: [Compact result files and startup results](docs/development/result-files-and-startup-2026-09-22.md).
The toolbar and File menu now provide Open result / Export result (Ctrl+O / Ctrl+S),
named saved results and an optional startup result. Files preserve calculated
inputs, full-precision executed state and dependency-checked continuation;
opening restores editable settings and saved views without transport or an
automatic Preview. Exports use lossless compression and exact array deduplication;
a real 49-particle fixture shrank 39.53%. Internal automatic archives retain
their low-overhead storage policy. Generated result packages remain excluded
from Git, and the named library is separate from disposable calculation caches.

Result-file, archive and library checks pass, including actual Preview/High
accuracy file loading with physical calculations forbidden. Full regression
executed 6,000 cases: 5,998 passed, one historical skip, one live-slider timing
failure. Following the final menu optimization, all 31 result/archive GUI cases
(including two new menu checks) passed as part of a 38-case run: 37 passed,
the same slider final-frame timeout failed. Final collection has 6,002 cases, all covered by
the full run or affected-module follow-up. The unchanged slider single case
then passed on both HEAD and current code with matched one-thread budgets, but
the intermittent module timing failure remains unresolved: do not claim a
fully green regression or proven new transport regression. Compilation,
dependency checks, final wheel byte comparison (449 Python modules), and
isolated installed GUI startup checks passed. Restart the application to use the new controls;
no GUI restart, commit or push was performed for this feature.
Coherent development remains paused.

Previous continuation: [Project audit repairs](docs/development/project-audit-fixes-2026-09-21.md).
The five independently reproduced defects are fixed: all filter-plane plots use
executed physical crossings, archive reuse checks verified file identity,
ordinary sample/EDS/STEM cancellation reaches existing safe boundaries, camera
pixels must be positive, and Experiments layout restoration refreshes its status.
Wheel builds use fresh private staging to exclude deleted source modules.
Fresh full regression: **5,926 passed, one historical-reference skip, zero
failures or unexecuted cases**. Production inputs remained fixed throughout;
two test-fixture corrections were followed by restarting their entire test group.
Final 1,056-input identity verification, compilation, dependency and whitespace
checks passed. The directly built wheel matches all 447 current Python modules,
and isolated installed GUI startup passed. Restart the application to load the
changes; no GUI restart, Git commit or push was performed. Coherent development
remains paused. This result supersedes the earlier test counts below.

Previous continuation: [Unified transverse tracking plots](docs/development/transverse-unified-plots-2026-09-20.md).
The normal Plot selector now fixes both coordinates and colour: source position
maps to plane position, emission direction maps to plane angles, and TOF maps
to plane position with per-path delay colours. The second coordinate selector
and its callback were deleted. Independent combinations are explicitly Advanced
diagnostics. Both plots retain shared identity/path colours; mode changes reuse
cached transport. All 179 related tests passed; compilation and whitespace checks
passed. An offscreen synthetic display fixture was visually checked. Restart the
GUI to use these changes. Coherent development remains paused.

Previous continuation: [Current interface and persistence audit](docs/development/current-contract-audit-2026-09-20.md).
Current accepted coverage is **5,858 passed and one skipped**, from a complete
5,859-case regression followed by full affected-module retests (25/94/98 cases).
There are no unexecuted current cases or remaining failures. The 581 production
files remained unchanged during this final verification; the single skip uses
obsolete C2 aperture-position numerical reference data. `compileall`, static
explicit-call checks and `git diff --check` passed.

Current State/profile/particle-archive schemas are **78 / 12 / 2**. Removed
fields, former component aliases and old schemas are rejected, not automatically
migrated. Quaternion sample orientation, held scan calibration, independent
stigmator controls, physical slit state, ten multipole field/calibration records,
tip quadrature and accelerator stage controls preserve their current ownership
through saves and loads. General and assembly-triggered condenser calibration
use detached state; only accepted controls reach the live instrument.

This round also fixed in-flight cache clearing, stale GUI callbacks, filter
control edits overwriting mode state, near-pole Euler conversion and a small
complex64 FFT norm drift while retaining physical absorption/band losses.
Coherent development remains paused. Restart the running GUI to use this code;
no application restart, Git commit or push was performed. This latest audit
supersedes earlier compatibility and unresolved-test summaries for active code;
those older entries remain a historical record.

Previous continuation: [Ray Diagram navigation](docs/development/ray-navigation-performance-2026-09-20.md).
Screen-resolution rendering retains original paths, stops, colours, full-data
Fit/export and physical results. A single-thread compiled simplifier with a
NumPy fallback and bounded scale caches removes repeated dense-line painting.
Actual saved-trajectory offscreen redraws fell from 420 to 9 ms for pan and
378 to 12 ms for zoom; first new zoom levels were at most 63 ms in this fixture.
These are isolated plot timings, not a measured full desktop frame rate.
Final targeted tests passed 58/58; the broader selection had 188 passes and
11 existing GUI-shell failures described in the report, so it is not all-green.
Restart is needed to use the new UI code. Coherent development remains paused.

Previous continuation: [EDS dose and spectral readout cache](docs/development/eds-readout-cache-2026-09-19.md).
Executed per-electron expectations now support safe dose, spectral range/bin,
FWHM and Poisson readout updates in full and section continuations. Static
frame-period/dwell changes reuse transport only when both scans are off;
material, incidence and detector geometry remain dependencies. Complete
records and coefficients survive archives, including JSON mapping reordering.
Duplicate event-ledger work is removed, and startup native compilation caches
are isolated by complete source version. Coherent development remains paused.
All **434 related regression cases passed**; the report records separate final
progress-label checks and the measured/final implementation identities.
In one actual 5,000-particle loaded-archive dose-halving comparison, the whole
call fell from **110.294 s to 11.777 s (89.3%)**; its EDS stage fell from
**99.627 s to 1.086 s**. Physical records and ledgers agreed within the declared
tolerance, with zero repeated gun/material/EDS physics on replay. Noise-only
and bin/FWHM updates took **1.428 / 1.580 s** in memory. Full archive load/save
still cost about **45 / 39 s**; this does not accelerate the first EDS execution.
Restart the application after updating code; historical results remain readable
but old solver identities are not rebound to make their calculation caches fit.

Previous continuation: [Persistent exact EDS results](docs/development/eds-persistent-cache-2026-09-19.md).
Complete EDS products now survive ordinary/full and cutoff-plane particle
archives. Both continuation paths reuse them only after actual incident,
material, model, acquisition-call and independent EDS dependency checks.
Historical missing EDS stays readable without claiming a completed spectrum.
All **257 related regression cases passed**, including a real nine-particle
tip-origin archive/continuation test. The actual 5,000-material-hit archive
round-trip preserved the complete EDS records exactly; a loaded downstream
lens edit took **13.339 s** with zero repeated gun/material/EDS executions,
against **118.209 s** when the same saved EDS was deliberately recomputed.
Complete EDS records matched exactly. Including larger-archive IO and checks,
the controlled pair took **201.757 / 93.831 s** (recompute / reuse).
See the report for settings, separate IO costs and the single-pair scope.

Previous continuation: [Classical gun batches, archive identities and timings](docs/development/gun-batches-and-timing-2026-09-19.md).
Bounded compiled batches retain the existing accepted steps, physical event
handlers and history cadence. The current controlled 5,000-particle pair at
sixteen threads took 84.00 s before and 78.76 s after (6.23% reduction), with
identical full-output digests. Immutable array identities are shared between
archive validation and manifests. Cached signals now shows current timings
and reuse details; archive status distinguishes save/load/verification and
historical measurements. See the report for final validation and the separate
actual material-hit continuation evidence. Coherent work remains paused.
All 316 main related regression cases and the final 49-case UI/controller
selection passed (overlapping selections, not a summed total). The actual
5,000-particle material/loaded-lens continuation passed with zero repeated gun
or elastic-material executions; eight branches reused their saved prefixes.

Previous continuation: [Fixed Transverse beam picture sizes](docs/development/transverse-plot-sizes-2026-09-19.md).
The Plot sizes dialog sets source, selected-plane and colour-legend dimensions
independently. Pictures remain fixed inside a local scroll area and follow
named workspace layout autosave/restore. All 101 related cases passed in two
runs (86 + 15), plus syntax and visual checks. No transport calculation changed.

Previous continuation: [User-defined cutoff and Ray Diagram extent](docs/development/ray-cutoff-display-2026-09-19.md).
The existing section/save/load/continuation workflow is confirmed. A narrow
spatial strip below the ray plot now distinguishes completed coverage, the
requested cutoff and the available restart plane. It follows the physical Z
axis while panning/zooming and marks earlier results after input changes.
It does not report time percentages or turn a requested plane into completion.
The report records the supported straight-column range and final UI checks:
139 related cases passed in two runs (133 + 6), plus syntax and visual checks.
The extended GUI test cleanup now drains delayed Qt widget deletion within
each test, avoiding accumulation at a later hidden-panel transition.

Previous continuation: [Classical gun loop workspace optimization](docs/development/gun-loop-workspace-2026-09-19.md).
At the same sixteen-thread budget, a new controlled pair of complete
5,000-particle gun runs took 95.84 s before and 79.79 s after this round
(16.74% less time). Trajectory, TOF, identities, weights and history digests
remain identical. Execution-local buffers and field preparation are reused;
the original physical operations and numerical error budgets remain in place.
This is a warmed gun-only measurement, not a full-microscope timing. The
latest report records final regression and archive-continuation verification.
The final related regression passed all 223 cases, including real small-source
archive/continuation and clock checks. A separate Numba-unavailable process
passed a bounded general-path smoke test; syntax and diff checks passed.

Previous continuation: [Particle speed, cutoff planes and automatic restart archives](docs/development/particle-speed-and-autosave-2026-09-19.md).
Numerical CPU work now uses at most 50% of available logical CPUs. The complete
5,000-particle gun benchmark improved from 161.05 s at four threads to 130.14 s
at the same budget and 85.75 s at sixteen threads, retaining identical outputs.
High accuracy now accepts cutoff planes and retains full-precision restart
state. Completed current particle calculations are automatically archived in
the background with explicit endpoint, quality, population, timestamp and path.
An actual 5,000-particle section took 86.91 s, saved/loaded in about 2.1/2.0 s,
and continued to the full axial destination in 23–24 s without any gun retrace.
Exact saved-gun state equality and final full-state persistence were verified;
the report records the separate storage recovery and finite-specimen-miss scope.
The 403-test integration run had 401 passes and two outdated disabled-filter
progress fixtures; after correcting only those fixtures, both affected test
files passed all 36 tests. Separate gun coverage passed 76 tests.

Previous continuation: [Particle section tuning and live detector signals](docs/development/particle-section-live-tuning-2026-09-19.md), with
[measured performance and next gun optimizations](docs/development/particle-performance-2026-09-19.md).
Main-window Preview/Medium now use the classical material/detector path. Users
can select a section and participating components, reuse executed upstream
states, save/load them, and extend material branches from their checkpoints.
Current-pixel counts and classical 2D scan products update with accepted results.
The default new assembly omits the energy filter; explicit saved choices remain.

Previous continuation: [Classical time-of-flight tracking](docs/development/transverse-time-of-flight-2026-09-19.md).
All three source/plane presets are available. TOF uses executed clocks; missing
history and aggregate finite-loss paths remain explicitly unknown. The [High accuracy Stage 2 performance and cancellation repair](docs/development/high-accuracy-performance-2026-09-19.md)
is retained.
The preceding [detector response and current accounting batch](docs/development/qualitative-third-batch-2026-09-19.md) remains in place.

Read [AGENTS.md](AGENTS.md), then the
[detailed current handoff](docs/development/HANDOFF_2026-09-19.md).
Coherent tip-to-column development is **paused**. The previous root document is
preserved in [the historical archive](HANDOFF_HISTORY_2026-09-13.md); its old
continuation and shutdown requests are not active instructions.

## Current objective and boundaries

The simulator must respond correctly to parameter changes within declared
physical regimes, using its own mechanical structure. Matching a real
microscope's numerical settings or absolute performance is not required.
Use scientific/functional equipment names in the UI; preserve original records,
reference URLs and compatibility identifiers as evidence.

Keep the physical tip → extraction → acceleration → focusing/apertures → column
chain. Do not add an independently configurable downstream source, renormalize
away current losses or remove interactions when a readout is disabled. Vacuum
remains opt-in/default-off. Integrate the energy filter last; pre-filter paths
remain upstream and unsupported downstream paths must not silently bypass it.

## Local repository state

- Workspace: `F:\tem_simulator_v2`; branch: `master`.
- HEAD: `7c52fa0593dfe73b36bcb12c042f55ba09f3334d`.
- Local `origin/master` points to the same commit; the live remote was not checked
  for this handoff.
- Scientific naming, all three qualitative batches and the performance repair are **uncommitted**. Preserve
  unrelated changes, especially `tests/test_stem_cuda_pipeline.py` and the
  untracked user brief `CODEX_OPTIMIZATION_ROUND2.md`.
- The latest detector batch includes code changes and bounded classical checks.
  No commit, push, application restart or shutdown was performed.

## Completed work

| Area | Delivered behavior | Detailed evidence |
| --- | --- | --- |
| Scientific names | Functional component/camera labels; original record IDs and numerical settings preserved | [Scope and naming](docs/development/qualitative-scientific-scope-2026-09-18.md) |
| Tip, gun, lenses, apertures | Parameter meanings/units, bounded current/aperture/lens checks; corrected nonzero-launch-potential energy handoff and cache identity | [First batch](docs/development/qualitative-first-batch-2026-09-18.md) |
| Stigmators, correctors, scan | Shared explanations; float64 first-order observations; correct on-axis linearization; filter-boundary guards; actual tip-origin local stigmator checks | [Second batch](docs/development/qualitative-second-batch-2026-09-19.md) |
| Active/inactive controls | Source, physical, numerical and historical metadata separated in the existing registry | [Parameter inventory](docs/development/qualitative-parameter-inventory-2026-09-18.md) |
| Detector response/current | Preserve zero weights and moved-detector collection in diagnostics; retain upstream losses in current budgets; shared response/readout meanings and bounded dose/PSF checks | [Third batch](docs/development/qualitative-third-batch-2026-09-19.md) |
| High accuracy gun performance | Same-error-budget compiled CPU/parallel gun update and impulse bound; High cancellation reaches physical gun steps; cache version includes numerical implementation | [Performance repair](docs/development/high-accuracy-performance-2026-09-19.md) |
| Source/plane tracking | Real source scatter; position/launch-angle presets; fixed ancestry colours and display cache | [Source tracking](docs/development/transverse-source-tracking-2026-09-19.md) |
| Classical TOF | Executed per-path clocks through gun, column, zero-loss specimen paths and physical filter planes; checkpoint/cache persistence; incomplete times grey | [Time-of-flight tracking](docs/development/transverse-time-of-flight-2026-09-19.md) |
| Particle sections and live signals | Executed gun/column/material prefix reuse; exact selected stop; checked section persistence; current-pixel detector counts and bounded classical STEM frames | [Section workflow and scope](docs/development/particle-section-live-tuning-2026-09-19.md) |

The model identifiers are `launch-potential-handoff-v2` for analytic gun
energy, `all-active-field-step-doubled-boris-compiled-v2` for analytic stepping,
and `axis-linear-float64-checkpoints-v2` for first-order response.
Changed calculations invalidate dependent caches; historical files and held scan
calibrations remain readable. No physical defaults were retuned to pass tests.

## Verification and open failures

- Latest TOF regression: **152 display tests**, **103 physical-transport tests**
  and **54 cache/restart tests** passed in separate completed runs. The final
  filter/source-routing/progress group also passed **54 tests** (receipt:
  `tmp/filter-tof-pipeline-20260919.xml`). Receipts
  are in `tmp/tof-20260919/` and the [TOF report](docs/development/transverse-time-of-flight-2026-09-19.md).
  This includes actual small CUDA execution, real tip-origin cache restarts,
  unknown loss clocks and malformed-clock rejection. An actual 49-particle
  tip-to-specimen run supplied the offscreen three-panel preview; its coarse
  column step does not establish whole-instrument convergence. Missing
  event-resolved finite-loss timing remains explicit.
  After correcting gun first-crossing geometry/time alignment, a final **39-test
  gun/cache/restart/reference group passed** (`tmp/tof-20260919/final-gun-cache.xml`).
- Latest source-view regression: **155 passed**, exit 0. Actual default-emission
  samples were rendered offscreen; no gun/column run was performed for the image.
  Full GUI exploration also found nine failures: eight reproduced before this
  feature, while one timeout passed separately on both versions and remains
  unexplained in the combined run. See the source-tracking report; full GUI
  acceptance is not claimed.
- Latest declared classical software lane: **259 passed**, exit 0; source and
  configuration hashes unchanged during execution and all 965 final inventory
  hashes match. Performance-repair run: `c5a31d64a673452988bacf33f6e7b82a`.
  Full simulator: **UNQUALIFIED**.
- Final performance/physical-source regression: **58 passed**. Complete 49-ray
  gun reference/compiled comparisons retain particle identity, energy, current
  and stops. A 15,000-ray local update-plus-impulse benchmark measured about
  **5.35x** acceleration with four CPU threads; this is not a whole-High timing.
  A final 15,000-ray default-gun diagnostic reached 33,335 steps in a 360-second
  budget and was deliberately cancelled before completion. Large-bundle total
  runtime and history/progress overhead remain open; earlier whole-High speed
  has not been demonstrated.
  The original running application was not restarted and retains its old code.
- Second-batch final control regression: **60 passed**, no skips. The actual-chain column
  refinement met unchanged 10 nm / 1 microradian budgets, with measured
  differences 0.134 nm / 0.400 microradians. This is a bounded local check.
- Expanded dependencies: initially 88 passed / 6 failed / 1 skipped. After
  scoped fixture corrections, **90 cases have passing evidence, four remain
  failing and one is skipped**. This is not a single clean 95-case rerun.
- The four failures reproduce on unchanged HEAD: a requested 60 mrad setting
  reached about 51.22657 mrad; requested 1.5, 2.0 and 2.2 micrometre illumination
  diameters reached about 1.10469 micrometres. Retain them as unresolved historical
  target qualifications; do not force source/geometry/presets to fit them.
- Third-batch detector regression: **262 passed, 2 explicitly deselected**; editor
  integration: **77 passed**. The two deselected coherent-image tests also fail
  on unchanged HEAD at source admission; neither the tests nor gate was bypassed.
  Actual 49-particle tip-to-detector runs retain current balance and the same
  stop fractions at 0.125/0.0625 mm column steps, including with readout disabled.
- Counts overlap; do not sum them or claim the full repository passed.
- Final serial Python compilation, whitespace and documentation-link checks passed.

Receipts remain locally under `tmp/qualitative-20260918/` and
`tmp/qualitative-20260919/`, plus `tmp/qualitative-detectors-20260919/`.
See the batch reports for exact runs, limitations,
unchanged-HEAD reproductions and commands. Generated arrays/caches stay out of Git.
Native interactive GUI, full-image physics, full-chain GPU performance and
filter-reaching transport have not been qualified by these batches.

## Next step

The [EDS cache plan](docs/development/eds-cache-plan-2026-09-19.md) now has its
exact-result persistence and per-electron response phases implemented. Dose,
spectral binning/broadening and Poisson sampling can use the executed response
when their physical dependencies match.
Incident direction remains a physical dependency through material and photon
paths; changing it cannot be reduced to changing electron counts.

Profile the first EDS calculation's substages before changing its algorithm.
Also reduce archive JSON/record reconstruction overhead without dropping
executed records or weakening provenance. The response cache does not speed up
first acquisition; repeated dose/readout edits now avoid EDS physics. Gun history,
prepared workspaces and bounded same-step batches are already implemented;
grouped adaptive stepping still needs separate numerical validation.

Earlier particle-section validation: **265 passed** in 187.11 seconds, receipt
`tmp/particle-sections-delivery-20260919.xml`. The group includes short physical
transport fixtures, real small tip-origin persistence, material probability/TOF,
actual 2x2 classical detector arrays, default no-filter startup, and GUI/controller
lifecycle tests. Expected existing Pydantic deprecation and pyqtgraph teardown
warnings were emitted. This is a focused suite, not the entire repository suite.
Do not add overlapping earlier agent totals. A stronger persistence-only rerun
passed **11 tests** in 16.23 seconds and checks restored material-branch
checkpoints (`tmp/particle-section-persistence-delivery-20260919.xml`). Compilation
passed. A fresh isolated 49-particle no-filter classical pipeline completed in
6.625 s after these checks; see the performance report for conditions and scope.
Coherent calculations remain disabled in particle tuning. Current scan material
contrast uses one reference exit distribution; per-pixel material Monte Carlo
and coherent atomic contrast remain outside this change.

The third Transverse Beam preset now reads executed per-path time. Remaining
timing work is **event-resolved finite-loss transport** and explicitly defined
arbitrary filter sections; see the [TOF report](docs/development/transverse-time-of-flight-2026-09-19.md).
Do not invent missing event depths, infer clocks from drawing paths or equate
TOF with wave phase. Coherent development remains paused.

Apply the performance repair by saving the current desktop configuration/results
and restarting the application; this also loads the new source views. Recheck the same High request; live inspection
showed the old job eventually left the gun for EDS. See the performance report
for bounded-run limitations and remaining history/progress-label overhead.

Continue **detector sampling/PSF support and loss presentation** across bounded
parameter ranges. The third-batch mean-response checks do not qualify extreme
PSF widths, arbitrary pixel sampling or charge-sharing noise covariance. Keep
physical interception, electronic response loss and virtual diagnostics distinct.
Use actual tip-origin transport for integration evidence and separately label
isolated checks. Keep coherent work paused and integrate the filter last.

Then continue remaining multipole semantics and bounded sweep/convergence
presentation. Existing working-point and numerical-qualification workflows should
be extended, not rebuilt. Historical alignment targets and earlier Round 2 gaps
remain separate open items in the
[detailed resumption record](docs/development/HANDOFF_2026-09-19.md) and
[Round 2 history](docs/development/ROUND2_OPTIMIZATION_PROGRESS.md).
