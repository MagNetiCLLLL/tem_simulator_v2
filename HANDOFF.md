# TEM Simulator v2 — Project Handoff

Last updated: 2026-09-07

## Purpose

This file is the persistent handoff for continuing development. Keep it focused
on current behaviour, confirmed design decisions, provisional assumptions and
the next concrete work. `README.md` remains the user/developer overview and
`CHANGELOG.md` remains the release history.

## Parameter definitions and calculation use (2026-09-07)

- `parameter_semantics.py` is the shared, Qt-free source of parameter labels,
  meaning categories and conservative declared provenance. `DimensionSpec`
  and `ManifestField` expose this metadata without changing field paths or
  TOML value-column contracts. Source evidence is independent of physical
  meaning; topology-only photographs do not establish measured dimensions.
- `dimension_audit.py` and the File/3D-editor audit dialog inspect saved catalog
  values without generating meshes or modifying configuration. The initial
  report under `docs/reports/` covers 11 modules, 482 definitions, 4,480 numeric
  dimensions and 280 structural review items. Most dimensions lack established
  field-specific evidence; this is a review queue, not an automatic sizing fix.
- `parameter_impact.py` traces original geometry/operating/material routes for
  the active simulation mode and explicit field recipe. Mixed vacuum/magnetic
  length effects are kept separate, missing setup is disclosed, and imported
  maps do not imply automatic field regeneration. `model_3d` parameters remain
  excluded from beam/field calculations, including child/shared bodies.
- MainWindow supplies active mode, field descriptors and assembly context to
  the inspectors. The 3D Use column and selected-parameter details update
  without discarding drafts. Calculation-controller lifecycle events drive
  stale/running/current/failure state; drafts override claims of being current.
  External module files do not borrow active simulation results. No solver or
  cache identity is changed by these descriptions. See
  `docs/PARAMETER_DEFINITIONS.md` for definitions and limitations.
- Source revision checks prevent obsolete displayed dimensions from claiming
  to be current. Clean documents reload when the assembly refreshes; pending
  drafts remain intact. Save-copy and module-selection changes rebuild runtime
  aperture context. Audit jumps resolve module/port paths and missing fields
  without substituting a component dimension, and active component navigation
  synchronizes the other inspectors.
- Verification: 130 editor/impact/source/manifest integration regressions passed;
  the wider 629-test model/geometry run exposed one lost vacuum-editing tooltip,
  which was restored and passed in the integration rerun. Further source and
  audit boundary checks are in `test_parameter_source_refresh.py` and
  `test_dimension_impact_editor.py` (19 passed after the final source/context
  fixes). Visuals under `tmp/parameter_definitions_*.png`
  use disposable files; the user's EnergyFilter TOML remains byte-identical.

## File-backed 3D model editing (2026-09-07)

- `part_model_features.py` implements part-local `model_3d` schema v1: existing,
  box or elliptic-cylinder bases; X/Y scale, Euler XYZ rotations and offsets;
  stable-ID cylindrical holes and rounded slots with X/Y/Z axes and depths.
  Manifold3D Boolean provenance carries surface IDs and parameter paths into
  semantic edges. The mechanical CAD mesh does not replace beam-clearance or
  axisymmetric field models. New fields stay in the original part's TOML.
- The 3D source selector now follows the actual opened document and its label
  includes the selected part key. Face/edge picking highlights related rows in
  Dimensions and All parameters; Features manages editable cuts and
  base changes. Existing module/file draft guards still apply.
- PhysicalLayoutView retains its object identity and original plot API. Its
  `tabs` now contain `section_page` (2D section) and `model_editor`
  (PartModelEditorPage). Project context and selected components are wired by
  MainWindow; files are loaded lazily when the 3D page is shown.
- Physical Layout 2D double-clicks resolve the plotted component/label first,
  then the nearest axial/radial envelope for empty space, and reveal it in 3D.
  Single clicks keep the current page, including sample/Energy Filter parts.
  Ray Diagram retains its axial cursor without navigation; component highlights
  survive manual tab switches. Explicit 3D reveals refit the selection while
  retaining rotation, and source-file draft guards still apply.
- `part_model_document.py` owns file snapshots, cross-component drafts,
  dimension-array updates, fixed-centre length edits, undo/redo and source-change
  detection. Active-module saves reuse the catalog transaction and assembly
  reload; independent instrument TOMLs can be edited and saved separately.
  Save-copy destinations are kept outside the catalog to preserve its one-to-one
  module-file authority. Existing 2D/TOML table drafts survive project saves.
- `part_model_3d.py` builds meshes and exposes existing dimension fields without
  Qt. Actual annuli, pole sections, radial profiles and parent-owned split
  intervals are represented; unsupported geometry is explicitly an envelope or
  omitted if no outer extent exists. Array-defined apertures preview one
  existing diameter by index, without inventing perforation coordinates.
- `gui/part_model_view.py` uses Numba/software triangle rasterization and one
  shared depth/selection buffer. It supports arbitrary trackball rotations,
  panning, zoom, an axial view and display-only section clipping without requiring
  an OpenGL context. The section exposes existing surfaces, not new cut solids.
- Material snapshots live in each part's optional `material_regions` mapping,
  with `body` as a default and `upper`/`lower` for real split bodies. The existing
  `material_class` role is retained. `part_materials.py` provides sourced iron
  and explicit static nonmagnetic approximations plus their application scope.
  Linear/nonlinear geometry solvers and material identities consume assignments;
  non-solver components retain metadata only. No conductivity, thermal,
  scattering, analytic-field or beam-cutoff changes are implied.
- Three-dimensional editing currently opens existing instrument module TOMLs;
  STEP/STL import and arbitrary face subdivision are not implemented. Parametric
  holes and slots are supported as described above. Do not treat envelope
  displays as detailed OEM CAD models.
- Verification: `tmp/part_model_regressions.py` ran 636 tests successfully on
  2026-09-07, covering new models/materials/UI plus existing manifest, geometry,
  field and selected GUI checks. Two existing extreme-ray numeric warnings
  remain. Actual Fusion/dark-theme view: `tmp/physical_layout_3d_preview.png`.
  User EnergyFilter TOML remained byte-identical, including the 180 mm IL coil.

## Graphical mechanical dimensions (2026-09-07)

- Select a simple excitation coil, housing or yoke through Physical Layout or
  the Mechanical tree, then open Saved mechanical dimensions > Edit dimensions….
  The separate editor shows an equal-scale 2D axisymmetric section. Numeric
  length/ID/OD/radial-thickness controls and six drag handles edit the same draft.
  The axial centre and its fractional position within an asymmetric envelope
  remain fixed; surrounding gaps may change. Thickness edits explicitly retain
  either ID or OD, with thickness derived as `(OD - ID) / 2`.
- Undo/Redo/Revert operate on the preview. Apply is the persistence boundary:
  full manifest/catalog validation runs before saving and geometry reload
  preserves runtime lens strengths. Invalid dimensions or assembly collisions
  appear inline and leave the saved assembly unchanged. Opening the editor
  reads saved dimensions, without importing uncommitted TOML table edits.
  Apply preserves existing table drafts, including invalid text, and shows a
  notice that those values remain unsaved while the geometry summary is current.
- Vacuum diameter is the beam passage, displayed read-only alongside the
  material class. Projector vacuum-ID mismatch errors now direct simple-layer
  users to Edit dimensions and the mechanical ID/OD fields for material
  thickness changes. A material-class label does not provide a complete material
  assignment editor; existing FEM permeability/B-H configuration stays separate.
- `src/temsim/part_geometry.py` owns the neutral annular-cylinder representation;
  `src/temsim/gui/part_geometry_editor.py` owns its 2D interaction. The primitive
  can support future 3D rendering, but this stage is not a full 3D CAD system.
  Shaped radial profiles and shared/segmented structures are explicitly outside
  the graphical editor's supported scope rather than approximated as cylinders.
- Manifest validation now checks finite positive layer length, ordered
  mechanical ID/OD, vacuum clearance and non-empty material classes for the
  three layer profiles. Simple concentric layers with the same optical owner
  cannot intersect both axially and radially; contact within the existing
  tolerance is allowed. Split material intervals retain their physical gaps,
  and complex/shared geometry is not forced through the simple-annulus check.
  Existing legacy sizing recipes and other assembly constraints remain active.
- Preserve user-edited TOML values when extending this feature. In the current
  workspace, EnergyFilter's Intermediate Lens coil is 180 mm long, at local
  Z 162.5..342.5 mm with centre 252.5 mm; do not restore the former 162 mm value.
- Focused coverage lives in `tests/test_part_geometry.py`,
  `tests/test_part_geometry_editor.py`, `tests/test_part_geometry_validation.py`,
  `tests/test_part_geometry_integration.py`
  and the existing manifest/editor integration tests. Validation tests read all
  current instrument manifests without modifying them and exercise safe radial
  edits, overlap rejection, contact, complex-profile handling and useful errors.
- Final geometry, catalog, input-policy and selected GUI regression run:
  457 passed (2026-09-07), with existing extreme-ray numeric warnings.
  `tmp/part_geometry_regressions.py` isolates QSettings and loads the installed
  Segoe UI font for Windows offscreen sizing checks. Actual editor screenshots:
  `tmp/part_geometry_editor_preview.png` and `tmp/part_geometry_editor_900.png`.

## TOML part length editing (2026-09-07)

- Editing a part's `length_mm` in ParameterPanel updates the staged start/end
  coordinates immediately after the cell is committed. The centre and its
  fractional position within an asymmetric envelope remain fixed. The edit is
  written only through Validate and save TOML; loading the panel never changes
  staged coordinates. Standalone ManifestEditor length-only saves use the same
  rule, while explicit endpoint updates retain strict consistency validation.
- The user chose fixed centres with variable gaps. Projector housing clearance
  now checks finite, non-negative gaps instead of enforcing the nominal 5 mm
  design gap at every pair. Bore, collision, provenance and other assembly
  validations remain active. No default configuration dimensions were changed.
- The reported IL housing 230 -> 225 mm produces local endpoints 140/365 mm;
  IL coil 162 -> 168 mm produces 168.5/336.5 mm. Both retain centre 252.5 mm,
  reload through the catalog, and leave every other part at its original Z.
- Validation: 156 combined regressions passed across editor, catalog, layout,
  field-provider and GUI paths. After tightening length type validation even
  for explicit endpoints, 45 editor/UI cases passed in a focused rerun. Invalid
  or conflicting input preserves the original file bytes. Compile and diff
  checks pass; the checked-in instrument TOMLs retain their original values.

## Inspection fixes (2026-09-07)

- Wave-STEM integrates shared and per-probe detector masks by multiplying and
  summing the final two diffraction axes. Recording plans include downstream
  static kicks and transport scan-time kick deltas to each physical stop;
  sample-position transport and the old detector-centre shifts are not added
  twice. Ordinary first-order/TEM callers retain their previous event defaults.
- New 4D-STEM calibrations persist the acquisition-time raster. Disk/RAM
  detector replay uses those times and rebuilds derived scan calibration on
  private components, including when the input state came from a detached bank
  snapshot. Legacy cubes retain their original calibration digest and remain
  readable, but dynamic recording replay requires recorded scan times.
- STEM request/product identities include the corrected recording schema;
  incident, elastic, EDS and TEM wave identities remain reusable.
- Operating-profile format 4 records nullable editable fields in `none_values`,
  preserving automatic Cs/Cc estimation. Readers support formats 1-4, validate
  null markers, and validate sample values before applying device changes.
- Artifact quota admission counts the retained object, replacement reference
  and fixed store overhead before publishing or evicting. Rejected oversize
  writes preserve old identities and shared objects; successful writes retain
  normal LRU eviction. Tests now use structured geometry authority rather than
  requiring a particular sentence in non-OEM provenance prose.
- Corrected two additional legacy test fixtures exposed by verification: the
  single-plane Collins test retracts downstream apertures, and the synthetic
  specimen transport initialises its cached field supports. All existing
  symplecticity, map agreement and trajectory-reversal assertions are retained.
- Validation: 410 distinct CPU cases across 34 test files passed across the
  combined run and targeted reruns; 6 CUDA cases were skipped because the CuPy
  backend is unavailable. The final physical-routing implementation passed
  66 focused cases, wave imaging passed 17 (1 CUDA skip), and the aperture /
  six-stage files passed 17. Changed Python files compile and `git diff --check`
  passes. Coverage is focused on the changed paths and their integrations.

## Stage 1: incremental ray and magnetic graphics

- New ray results reuse unchanged component, specimen, wall and crossover layers;
  ray and stop groups update in place. Removed layers release their items,
  labels, legend entries and old ray payloads. Only the initial waiting notice
  requires a full scene clear.
- Mechanical/aperture/detector drawing inputs are value-keyed independently from
  physical result identities. Same-object republication still invalidates
  projected-array caches. Geometry edits preserve user X/Y ranges; Fit remains
  explicit.
- Cursor identity and in-flight position survive publication. The latest
  detector-versus-Z focus choice is retained. Pending rotation and result
  publication cannot leave aperture/detector graphics at an old angle.
- MagneticFieldView reuses curves, markers, legends and selected support. Field
  and rotation providers still evaluate the latest snapshot; no new physics
  memoization or preset calculation was introduced.
- The implementation, benchmark scope and outstanding 50 ms end-to-end target
  are documented in `docs/RAY_INTERACTION_PERFORMANCE.md`. Stages 2-6 of
  `docs/ENHANCEMENT_PLAN_2026-09-06.md` have not been started.
- Final combined validation: 110 tests passed (288.84 s); another 16 selected
  existing GUI-shell checks passed separately. The isolated synthetic benchmark
  measured 94.87 ms median same-geometry publication versus 263.38 ms fresh-scene
  construction. Rotation/Z event-inclusive p95 values were 217.37/126.22 ms:
  the 50 ms complete-response target is still open. Raw output is in
  `docs/benchmarks/RAY_SCENE_2026-09-07.json`; no native/GPU FPS claim is made.

## Unified TEM/STEM result viewers

- Ray Diagram > Cached signals now contains the detached physical-detector
  table, not duplicate TEM/STEM image widgets. The normal Illuminating Image
  and Scanning Image > Images viewers have independent Result source selectors:
  Current calculation / Advanced bank.
- Switching a source is presentation-only. It cannot request a solve/readout or
  apply bank settings to live State. Both main and bank products remain retained;
  new main results do not replace a displayed bank image. Pending/failed bank
  readouts retain previous images with an explicit status; missing products are
  unavailable rather than borrowing current images.
- InteractiveReadout carries the detached readout-state snapshot for provenance
  and detector geometry. Main scan playback/paused frames remain independent of
  static bank presentation. Geometry/4D-STEM and parameter/aberration controls
  continue to refer to the current instrument. Bank coordinates and approximation
  notes are available in source-status tooltips.
- Removed cachedSignalTabs saved layout entries are safely ignored by the
  existing name-based layout restoration; other tab/splitter names are retained.
- Validation: 110 focused tests passed, 3 production integration cases deselected
  in 103.92 s. Coverage includes shared routing, TEM/STEM source isolation,
  paused/main frames, bank failure/debounce handling, layout restoration,
  analytical bank clipping, scan and 4D-STEM regressions. No full high-accuracy
  imaging or preset solve. Changed Python files compile; whitespace checks pass.
  Offscreen screenshots use explicitly synthetic arrays, only for UI inspection.
  The existing non-fatal pyqtgraph destroyed-signal warning remains at teardown.

## Background requests and lazy ray panels

- MainWindow Preview/Medium uses CalculationController.submit_background.
  CapturedCalculationRequest owns copied editable values and small live-geometry
  components; only recursively frozen installed assembly data is shared. Workers
  do not read or serialize the live State. Full snapshot reconstruction and exact
  identities are prepared off-thread; cache dispatch remains on the GUI thread.
- Preserve both identities: original live model signature can differ from a
  TOML-normalized calculation snapshot. The new capture carries the former's
  geometry/anchors instead of replacing it with the normalized model signature.
- One generation/lifecycle spans preparation, cached delivery and ray tracing.
  Latest-value live scheduling remains; obsolete preparations cannot launch
  solvers. Cache completion is generation-checked after result callbacks too,
  preventing reentrant requests from being finished by an older cache hit.
- Physical Layout, Magnetic Field and Transverse keep a newest-only pending
  result while hidden. Visibility applies current result/focus/Z/projection.
  Magnetic X follows current Ray X; user ranges survive deferred refresh.
  Pending magnetic diagnostics are labelled, never silently served as current.
  Shared scientific publication and high-accuracy retained products remain eager.
- Explicit High accuracy still uses the prior synchronous request contract.
  Full static/dynamic scene separation on each new ray result is a later step.
  MainWindow close invalidates pending work. No preset strengths or physical
  calculation approximations changed. See scripts/benchmark_request_preparation.py.
- Physical Layout label packing now updates the ViewBox transform before
  measuring labels and repacks on viewport resize, including first activation
  after a hidden update.
- Validation: 182 focused tests passed across three separate selections
  (97 backend/cache, 45 presentation, 40 main-window/live/layout). The real
  small-bundle CPU drag test uses the background path; controlled solver tests
  explicitly keep their synchronous test boundary. Compilation/whitespace
  checks passed. No full high-accuracy image or preset solve was run.
- Serial warm request benchmark (8 repeats): GUI preparation 530.39 ms before
  versus 5.57 ms capture now, plus 545.13 ms moved to background. Exact snapshots
  and signatures match. These numbers are request preparation only, not solver
  speed or display frame-rate claims. See docs/RAY_INTERACTION_PERFORMANCE.md.

## Ray interaction and bounded caches (2026-09-06)

- Simulation > Performance and cache provides persistent, modeless retention
  controls: high accuracy up to 8 GiB/32 entries by default, live tuning 1 GiB/128
  entries, ray display 512 MiB and disk checkpoints 16 GiB. RAM defaults adapt to
  hardware; the dialog caps aggregate managed caches at half physical RAM.
- Preview/Medium now have exact-signature LRU histories with shared-buffer memory
  accounting. Latest compatible seeds and asynchronous generation guards remain.
  High-accuracy results are retained independently; no interpolated optics or
  automatic preset/image calculation was introduced.
- Scalar live edits validate the full batch without a pre-edit instrument
  snapshot and refresh existing parameter widgets in place. Worker snapshots
  remain detached. Analytical Objective reference-plane roots have a bounded,
  exact-parameter scalar cache to avoid repeated solves during state construction.
- Ray rotation reuses clipped X/Y bases and existing scene items; pan/zoom reuses
  sampled projected-slope summaries. Explicit result publication invalidates
  derived display data. User plot ranges, clipping endpoints and scan offsets
  retain their existing semantics. See docs/RAY_INTERACTION_PERFORMANCE.md.
- Cache preferences do not change physical state, submit calculations, cancel
  workers or erase visible results. Disk quota enforcement is deferred to the
  next managed checkpoint write; disk storage does not yet persist all images.
- Final combined regression: 126 tests passed in 149.10 s across cache
  preferences/integration, live scalar edits, live slider updates, interactive
  calculations, cache capacity, Objective plane caching and ray-display caching.
  Includes the real small-bundle solver during sustained slider motion. Additional
  controller/artifact and ray/scan regression selections passed separately.
  No full high-accuracy image run or preset recalculation. A pyqtgraph teardown
  destroyed-signal disconnect warning remains; tests exited successfully.

## Sourced lens material defaults (2026-09-06)

- JEOL polepiece/yoke/objective documentation supports a generic pure-iron
  magnetic-body default and the existing copper winding model. This is an
  explicit reference design, not a Titan material identification. Research,
  sources, scope and workflow: docs/LENS_MATERIAL_DEFAULTS.md.
- configs/materials/lens_defaults.toml selects the existing FEMM Pure Iron
  curve by stable key. Linear permeability 14872 is the pinned block's constant
  Mu_x/Mu_y value, not a fit to its B-H points. Source SHA-256, both constants
  and all 21 B-H pairs were verified against the pinned remote source.
- Model Inspector pre-fills unconfigured material drafts and provides Use
  default material. Applying still requires an explicit coil input. Saved
  numeric/material snapshots and material-class overrides are preserved;
  custom material selections no longer leak into another unconfigured lens.
  Linear reference attribution is persisted unless its value is overridden.
- Material configuration is included in wheel data files. No assembly geometry,
  excitation percentage, high-accuracy settings or preset strengths changed.
  No recipe is silently installed and no setup/convergence guard is bypassed.
- Validation: 37 material/inspector/nonlinear tests passed, followed by 2
  linear-field/profile regressions (39 total). Existing Pydantic deprecation
  warnings remain. Changed Python files compile and offscreen material controls
  were visually inspected with Segoe UI explicitly loaded for the headless
  renderer. No full suite, high-accuracy column run or OEM validation was run.

## Named workspace layouts and per-state panel sizes (2026-09-06)

- View > Layouts supports Default and user-named layouts, Save current layout,
  and Save layout as. Changes autosave after 600 ms idle, before switching
  layouts, and on close; the last active layout is restored at startup.
  Existing legacy main-window/live-control preferences migrate without a reset.
- Versioned QSettings snapshots retain main-window and native dock geometry,
  tabbed/floating/hidden dock state, 13 named application splitters, selected
  presentation tabs and magnetic/transverse/Advanced-bank visibility. Ray
  splitter sizes are stored independently for each magnetic/transverse switch
  combination. Hidden pages restore on Show after Qt allocates their space.
- Layout restoration blocks presentation-tab signals and never applies model
  parameters or clears results. The component-selection handler no longer
  enlarges the instrument editor automatically. Reset only changes the active
  named layout; other saved layouts remain intact.
- Scope and workflow: docs/WORKSPACE_LAYOUTS.md. Stable widget names identify
  application panels, not fragile list indices or ambiguous third-party names.
  Physical settings, cached calculations and plot camera/zoom are not layout
  data. Normal Qt screen/minimum-size safeguards still constrain restoration.
- Validation: 39 focused GUI/layout/scheduling regressions passed (138.26 s);
  the unchanged real-ray integration case was deselected. This includes eight
  new persistence/isolation/autosave tests and existing floating-dock checks.
  Menu rendering was inspected offscreen. No high-accuracy or unchanged physics
  suite was run; native pointer dragging and physical multi-monitor/DPI changes
  were not exercised.

## Live tuning dock and shared Ray Diagram (2026-09-06)

- Removed the top-level Interactive Calculation tab and its duplicate live ray
  plot. The only live plot is now Ray Diagram > Rays, including Medium support
  outlines, projection, transverse view and the user's fixed plot range.
- Live tuning is a native closable/movable/floatable dock, initially hidden and
  tabified beside Instrument setup and parameters. View > Live tuning and the
  Ray Diagram button share its toggle action. The aligned range/reference and
  slider columns remain together; the optional Advanced bank is retained.
- First opening allocates a readable control width when space permits. Later
  openings preserve the user's width. Dock visibility, placement and floating
  size use the main workspace settings; internal control widths have a new
  key so the obsolete three-pane saved state is not misapplied.
- Detached bank detector readouts are in Ray Diagram > Cached signals; TEM/STEM
  now share the main viewers via Result source (see the newer entry above).
  Live status is separate and cannot relabel or redraw a cached signal.
  Closing/reopening the dock does not destroy ranges, controls, caches or
  current results, and does not start or cancel calculation. Component/Z
  navigation and starting live tuning reveal the shared Rays subpage.
- Validation: 32 focused Qt/layout/scheduling regressions passed (112.97 s),
  including real CPU small-bundle frames during sustained slider motion. The
  final first-open width/persistence change passed all four dock regressions
  (24.70 s). Changed-file compilation and offscreen layout inspections passed.
  No high-accuracy specimen calculation or unchanged physics suite was run;
  native Windows pointer docking/dragging was not exercised.

## Live optical tuning before one final signal calculation (2026-09-06)

- Mixed-range slider fix: detector draft rows no longer reject the entire live
  plan. Lens/aperture sliders remain aligned and usable; detector references
  display Advanced bank and are excluded from live updates. Detector bounds
  are still retained and fully validated when building the detached bank.
  A detector-only live request gives a concise explanation without starting work.
  Validation: reproduced the reported C1/DF-Z/Objective failure before the fix;
  24 focused UI/scheduling tests passed (66.53 s), with the unchanged real-ray
  integration case deselected. Offscreen mixed-row layout inspection and
  changed-file compilation passed. No high-accuracy physics calculation run.
- Adding a unit reads its current live value into a copyable, read-only
  Reference column; it does not invent bounds or alter the capture snapshot.
  The middle controls now use identity-matched table rows with linked heights,
  vertical placement and scrolling. Resizing and Advanced-bank expansion keep
  alignment. Draft edits preserve active bank/live controls, with unmatched
  controls explicitly labelled Active only rather than paired with a wrong unit.
  Validation: 17 focused GUI/scheduling regressions passed (58.91 s), plus
  offscreen layout inspection and changed-file compilation. The unchanged
  real-ray integration test was deselected for this UI-only change.
- Continuous slider refresh: replaced the page's restart-on-every-edit debounce
  with a 50 ms latest-value throttle and release flush. Main-window live tuning
  bypasses the second debounce and lets one ray frame finish, retaining only
  the latest pending settings. This avoids cancellation starvation during drag.
  Intermediate frames are explicitly labelled Updating and never restore older
  lens values. Ordinary model edits still invalidate workers; High accuracy
  clears pending live work. Completed high-accuracy products remain retained.
  Validation includes sustained motion with the real CPU small-bundle solver,
  changed ray coordinates, final-value readback and controlled queue/error tests.
  All 15 selected layout, live-refresh and controller regressions passed
  (66.13 s); changed Python files passed compilation checks.
  No high-accuracy specimen/image calculation or unchanged physics suite is run.
- Interactive Calculation now uses three independently resizable columns:
  range selection, live/cached controls, and ray/signal views. Component names
  sit to the left of their sliders; controls scroll together with range rows.
  Three-pane widths use a separate QSettings key from the former two-pane layout.
  This is a UI-only change: existing tuning and bank signals are unchanged.
  Validation: seven focused layout/control tests and an offscreen visual check;
  no unchanged physics suite or high-accuracy calculation was rerun.
- The reported 60%/3-of-5 stop was an explicit 6 GiB bank limit, not a hung
  solver. Live Preview/Medium tuning now avoids building a high-accuracy bank
  for each focus value. Advanced banks remain optional, collapsed by default,
  and an oversized first-point storage projection stops construction early.
- The toolbar selects Preview (49 rays, 1 mm maximum step) or Medium (160
  interior source samples plus 33 zero-current support probes, 0.25 mm step).
  Static column fields, selected physical-model tier and clipping remain active;
  sample scattering, spectra, multislice, raster frames and full transfer
  diagnostics are deferred. Medium shading is sampled support, not density or
  a guaranteed outer envelope. A low-count trace may miss a narrow aperture.
- Interactive Calculation requires explicit endpoints for continuous live lens
  and aperture operating controls. These intentionally update current settings
  and Ray Diagram; detector geometry remains detached Advanced-bank readout.
  Bank sample counts/RAM settings are hidden until Advanced bank is expanded.
  The final action flushes the last slider value and submits one High accuracy
  request, with the already selected signal products. No automatic lens preset.
- Existing RK4 equations are reused by a serial Numba tuning kernel in Auto;
  NumPy fallback and explicit backend choices remain. Obsolete queued requests
  are dropped, results are generation-checked and cancellation is checked around
  integration segments. Native kernels are not force-killed. Tuning snapshots
  and prefix seeds remain separate from completed High accuracy products.
- Preview/Medium cannot erase completed images/spectra even if sparse rays miss
  the sample. Changed results are marked stale, and user Ray Diagram limits
  remain fixed. Last live sliders, source flags, cache isolation, analytical
  support/clipping and serial/NumPy equivalence have regression coverage.
- Validation: 55 targeted tests passed (85.21 s), followed by a 30-test final
  UI/range regression (71.60 s) and all 11 new tuning tests (25.87 s), including
  the added Objective-edit/prefix-reuse case. These selections overlap.
  Default assembled CPU/Auto worker
  timings after warmup were Preview 0.404 s and Medium 0.923 s (initial runs
  1.721 s / 2.446 s). These exclude GUI/snapshot overhead and are not universal
  real-time guarantees. Full acquisition, nonlinear/FEM benchmarks and the full
  repository suite were not run. No preset solve, Git push or shutdown requested.
- Final offscreen page layout was inspected; the quality selector is only in
  the main toolbar and the new plots use explicit mm labels (no SI-prefix/mm
  concatenation). Changed Python files compile and whitespace checks pass.
- See `docs/INTERACTIVE_CALCULATION.md`. This supersedes the former bank-first
  default described in the following historical checkpoint.

## Interactive range calculation and independent readout (2026-09-06)

- New English **Interactive Calculation** tab captures a detached state and
  requires explicit Minimum/Maximum endpoints, optical sample counts and a
  pinned RAM limit. Lens and upstream-aperture axes form up to 256 exact nodes;
  downstream-aperture and inserted-detector controls are continuous readout axes.
  No interpolation, extrapolation, automatic preset solve or live geometry edit.
- Completed main results seed dependency-compatible nodes without being replaced.
  Bank replacement is transactional; failure, cancellation and changed external
  files retain the old complete bank. Latest-wins readout discards superseded
  responses. Main calculations/alignment and range workers cannot overlap.
- Ray replay preserves emitted-source weights, pre-sample losses and wall stops,
  then re-evaluates ordered apertures and recording planes. Zero openings block
  even on-axis rays. TEM retains pre-pupil complex configurations for reopening
  the Objective aperture and repropagates affected downstream waves.
- Wave STEM retains a bounded RAM diffraction-probability cube. Compatible P2
  nodes and later detector/aperture readouts reuse this cube rather than repeat
  specimen calculations. This remains angle-resolved first-order routing, not
  arbitrary-plane coherent STEM. Uncached high-angle tails disable changed-stop
  STEM output explicitly. EDS-array range controls are not included.
- Validation: 77 related tests passed (104.88 s), followed by a final rerun of
  all 16 new tests (42.50 s), including an added analytical detector-Z test using
  the production camera component. Four focused offscreen GUI tests also passed.
  Small CPU production cases verify source-checkpoint reuse, TEM replay against
  a fresh projection and exactly one STEM specimen call for two P2 nodes.
  These runs overlap; they are not a full-suite count or a production speed claim.
  Offscreen layout was inspected. Full acquisition/GPU benchmarking was not run.
- See `docs/INTERACTIVE_CALCULATION.md`. Banks are session-local; JSON export
  contains the plan only. Existing unrelated worktree changes were preserved.
  No preset calculation, commit, push or shutdown was requested for this task.

## Fixed optical apertures (2026-09-06)

- Apertures navigation includes gun/anode, projection-chamber DPA and
  spectrometer-entrance stops. The optional NanoPulser stop exposes its original
  TOML editor, not a disposable runtime copy. EELS internals remain in Energy Filter.
- Fixed stops are always inserted; legacy disabled flags cannot retract them.
  Installation remains separate, including coherent-wave intermediate masks.
  Movable condenser/objective/selected-area apertures retain insertion switches.
- DPA is now a single active runtime hard stop, shared by ray/record-plane and
  coherent-wave transport. Diameter and X/Y are operating controls; its TOML
  axial plane and default bore can change independently of P2 and the literature
  reference. Refresh/snapshot/profile operations preserve the active opening.
- Existing 0.2 mm family-reference bore and zero-thickness approximation remain.
  No lens presets were recalculated. This supersedes the mechanical-only DPA
  implementation described in the historical 2026-08-30 checkpoint below.
- Validation: 102 targeted tests passed (83.41 s), covering fixed-stop controls,
  snapshots/profile round trips, identical cache signatures after worker copies,
  manifest edits, optional installation, record-plane/coherent masking and GUI
  navigation. Changed Python files compile; whitespace checks pass. No full
  acquisition, lens preset solve, commit, push or shutdown was requested.
- A broader test selection exposed an existing unrelated fixture failure:
  `test_vector_specimen_flight_transverse_field_and_plane_reversal` constructs
  `SpecimenFieldTransport` via `__new__` without `_provider_supports`. Its test
  fixture and production transport were not changed in this aperture task.

## STEM angular coverage and registered contrast (2026-09-06)

- Images distinguish covered, partial, outside-grid and unchecked angular
  bands. Outside-grid channels no longer appear as physical black images;
  raw arrays remain in the cached result. An undersampled illumination disk
  suppresses all displayed detector images. No BF/DF grayscale inversion.
- **Match detector sampling** proposes a grid-only change with an area/storage
  estimate and confirmation. It does not start a calculation, change lens or
  detector geometry, change FOV, or discard existing frames. Stale proposals
  and proposals exceeding the current grid/potential limits cannot be applied.
- Fixed two reproduced coordinate defects: corner-origin inverse-FFT probes
  versus centred potential axes, and a second half-box shift on finite CIF
  potentials. CPU, CUDA and the STEM incident-wave helper now agree. Finite
  boxes use even lateral dimensions to avoid a half-pixel origin error.
- Wave/STEM/4D-STEM cache signatures include a new coordinate schema. Old wave
  products must be recalculated; incident, elastic, EDS and local sample-region
  cache signatures remain reusable. Existing unrelated worktree edits remain.
- Final focused regression: **153 passed, 17 warnings**, exit 0 (96.63 s).
  Includes independent abTEM thin-phase contrast/registered-atom checks,
  actual CUDA/CPU comparisons, finite/periodic atom positions, recording-plane
  routing, shared caches and offscreen GUI pause/sampling controls. Pydantic
  deprecation, small CUDA-grid occupancy and pyqtgraph teardown warnings remain.
  Changed-file `compileall` and `git diff --check` passed.
  No full repository suite, full user acquisition, preset solve or Git push.
- See `docs/STEM_SAMPLING_AND_CONTRAST.md` for assumptions and limits. A covered
  detector band is not a thick-crystal contrast or projector-calibration
  certificate. The user's exact screenshot configuration was not rerun.

## Spectrum-only EDS and shared sample views (2026-09-06)

- EDS now contains only its spectrum, short status and hover energy/counts.
  Removed its trajectory/projection tabs, line table and projection redraw
  timer. Line contributions remain in the scientific result, not another UI.
- The existing settings widget is hosted once under **Sample Interactions 3D >
  Parameters** (collapsed initially). Support, EDS response/acquisition and
  detailed-region controls retain their original state and shared-cache wiring.
- **3D / X-Z / Y-Z** switch renderers over one immutable interaction scene.
  Orthogonal 2-D views include the same selected electron, X-ray and event
  categories and use local nm with +Z downward. Each view retains its range or
  camera during switching; no transport/EDS calculation is triggered. Without
  OpenGL, both projections remain available and 3D is explicitly disabled.
- The detailed-path calculation has one button on the sample page; the
  relocated parameter panel does not add a second copy of that action.
- Related offscreen regression: **139 passed, 16 warnings**, exit 0 (298.96 s).
  The first run exposed one obsolete line-table assertion; it was updated to
  check the spectrum-only no-current state before the complete related rerun.
  Changed-file compilation passed. Layout renders use a synthetic test scene;
  the 3D camera round-trip uses a renderer API fixture, not live GPU validation.
  Third-party Pydantic deprecations and a pyqtgraph teardown warning remain.
  No full repository suite, preset solve, Git push or shutdown was performed.

## Specimen transport performance checkpoint (2026-09-06)

- Removed long trial flights near material interfaces and needless searches
  through virtual vacuum supports. Curved magnetic intersections, scattering
  physics and boundary tolerances remain active. Field support metadata are
  resolved once per calculation.
- Local deterministic benchmarks: 64 axial rays through 10 nm Si with vacuum
  support improved from 13.05 s to 0.46 s; 16 tilted rays improved from 5.39 s
  to 0.19 s. Event counts, energy and weights match exactly; position/path
  differences are at floating-point rounding scale. These are local transport
  timings, not full application or STEM scan speedups.
- Checkpoint-only candidates now reach the solver after raster/lens edits
  invalidate all complete-product signatures. Source and common-plan checks
  still decide which prefix can be reused; stale products are not authorized.
- Related offscreen regression: **150 passed, 16 warnings**, exit 0 (120.43 s).
  No full repository suite, GPU benchmark or preset recalculation was run.
  Details, reproduction commands and remaining limitations are in
  `docs/PERFORMANCE_2026-09-06.md`.
- Preview display replacement, stale EDS display policy and the STEM wave CPU
  fallback remain separate work. Existing uncommitted changes are preserved;
  no Git push or shutdown was requested or performed this round.

## Magnetic convergence and R-Z checkpoint (2026-09-05)

- **Model Inspector > Field validation** now runs three mesh cases and two
  boundary cases on detached snapshots, with completed-case progress and
  cooperative cancellation. Expansion retains every existing interior node.
  It never applies a refined mesh, changes current/presets or clears images.
- Comparisons use fixed physical reference planes, explicit absolute/relative
  targets and separate unavailable states. Bz, finite-segment paraxial focal
  length, signed Larmor rotation and a collimated test radius are checked.
  The local RK4 transfer has step-refinement, determinant and uniform-field
  analytic tests; it is not a specimen/image model. Cs/Cc and external-software
  validation remain **Not checked**, not inferred from axial-field convergence.
- **Magnetic R-Z** shows the final study case's material masks, field magnitude
  and R*A_phi flux contours. Tab/layer changes reuse the cached scene. Controls
  are split over two rows to preserve small-window usability; plot/table panes
  are user-resizable. Flux overlays are removed correctly when redrawn.
- The four-study LRU is independent of image/ray results. Refined FEM data use
  their own twelve-entry cache; completed base solves can be reused. Relevant
  geometry, current, material, voltage and options key each study. Distant
  unrelated linear-lens and sample/image edits retain compatible studies.
  Late results for earlier inputs are cached without replacing the current view.
- Exported JSON contains the original study inputs, units, field curves,
  material provenance and diagnostics. It is not a FEMM-native model or a claim
  of an external comparison. See `docs/MAGNETIC_FIELD_VALIDATION.md` for the
  numerical method, limits, source references and remaining work.
- An offline synthetic shared-coil study correctly fails the 1% target on its
  coarse grid; a solved FEM system is not automatically marked numerically
  validated. Validation was also run on a synthetic joint B-H case. No OEM
  material/geometry, desktop OpenGL/GPU, Cs/Cc or external FEMM validation is
  claimed. Final related regression: **156 passed, 16 warnings**, exit 0
  (173.50 s), covering validation, Model Inspector, nonlinear FEM, circuits,
  field providers, model modes and GUI shell. Final targeted verification:
  **23 passed, 16 warnings**, exit 0 (11.40 s). Counts overlap. Changed-file
  compilation and whitespace checks passed. Warnings are third-party Pydantic
  deprecations and a pyqtgraph teardown disconnect warning. Offscreen inspection
  covered convergence curves, material masks and flux/field views; the separate
  small-window regression checks the corrected minimum width.
- Pre-existing uncommitted work is retained. No preset recalculation, Git push,
  shutdown or production parameter changes were performed.

## Static nonlinear magnetics checkpoint (2026-09-05)

- The configured **Nonlinear Material Field** tier now runs a static isotropic
  axisymmetric B-H solve; Coupled Multiphysics remains unavailable. Model
  Inspector provides a referenced pure-iron table and explicit B-H CSV import.
  Curves, source URLs and hashes are embedded in saved recipes. No material,
  ampere-turn value or physical dimension is inferred as a Titan calibration.
- All configured B-H coil channels are solved jointly at the complete current
  vector. Shared circuits require every member recipe. Active foreign coils
  inside the solve domain must join it or be explicitly disabled. One provider
  carries the total field, while member entries add nothing a second time;
  diagnostics identify this bookkeeping role rather than labelling it Gaussian.
- Damped Newton uses an assembled-residual convergence check and strict final
  B-H range validation. Unsupported geometry, genuine coil/iron overlap,
  nonconvergence and conflicting material assignments fail explicitly.
  Algebraic convergence does not establish mesh/domain convergence or measured
  accuracy. No hysteresis, thermal feedback, 3-D iron or GPU FEM was added.
- Existing projector tapered poles now have matching material masks. Objective
  coil/yoke masks share Physical Layout's TOML upper/lower intervals; current
  density uses their actual winding cross-section. No mechanical dimensions or
  operating strengths were changed to make the field solve succeed.
- Mapped-lens guards prevent duplicate empirical ray Cs. Generated Objective
  fields also suppress an additional empirical ray Cc kick; energy-dependent
  momentum already enters their spatial-field transport. Saved options and
  explicit Manual/Field-derived wave-aberration selection are retained.
- Prepared meshes and completed joint operating points have bounded caches.
  Current, material and geometry edits invalidate the affected result; returning
  to a retained point reuses it. Frozen plans keep their original field. Old
  completed diagnostics are not presented as current after an input change.
- Current persisted solver identity: `temsim-solver-2026-09-static-bh-v1`.
  State schema 76 and profile format 3 are unchanged. Earlier persisted seeds
  are rejected without deleting user data. See
  `docs/STATIC_NONLINEAR_MAGNETICS.md` for source provenance and numerical scope.
- Initial related regression: **147 passed, 16 warnings**, exit 0 (144.94 s).
  Follow-up material/cache/UI/manifest regression: **45 passed, 16 warnings**,
  exit 0 (70.12 s). Final regression: **155 passed, 16 warnings**, exit 0,
  covering nonlinear FEM, diagnostics, modes, Model Inspector, persisted
  manifests, downstream transport and GUI shell. These runs overlap; counts are
  not additive. Warnings are third-party Pydantic deprecations, with an additional
  pyqtgraph teardown disconnect warning. Changed-file compilation passed.
  Offscreen rendering verified the B-H controls at 1100 x 680; no desktop
  OpenGL/GPU or measured instrument validation was performed.
- No preset optimisation, Git upload or shutdown was performed. Pre-existing
  uncommitted work is preserved.

## Simulation model levels checkpoint (2026-09-05, historical)

The static B-H checkpoint above supersedes this checkpoint's nonlinear,
projector/Objective geometry and persisted-solver limitations.

- **Simulation** now sits beside File and View. New windows start in Ideal
  Optics; legacy states/profiles without a selection remain Custom. The
  available alternatives are Analytical Field, explicitly configured Linear
  Geometry Field, and Custom / Per-lens Models. Nonlinear Material Field and
  Coupled Multiphysics are disabled, not analytic substitutes.
- This selector controls column-lens physics, independently of ray count,
  integration step and backend. Ideal retains finite-length paraxial focusing,
  magnetic rotation and physical interception; column focusing uses reference
  energy and excludes spherical/hexapole/extra chromatic kicks. Wave-lens
  coefficients are zero except configured defocus. Gun, dedicated Energy Filter
  transport, specimen scattering and local specimen flights keep their models.
- Each mode retains round-lens excitation/polarity, field recipes and aberration
  options. Geometry, installation, specimen and numerical controls are shared.
  Switching models does not calculate a preset or clear other models' results.
  Active model identity separates workers, stage results and ray checkpoints;
  inactive shelves do not invalidate a returning mode's compatible cache.
- Linear Geometry requires material/current recipes for every enabled round
  lens and does not fall back to analytic fields. Readiness does not certify
  geometry or convergence. The shipped projector tapered-pole profile remains
  unsupported by the linear solver; the existing Objective coil/iron-overlap
  rejection also remains. Production geometry was not changed to enable a tier.
- State schema 76 and operating-profile format 3 persist the mode and shelves;
  profile versions 1/2 remain readable. Current persisted solver identity:
  `temsim-solver-2026-09-simulation-modes-v1`. Earlier seeds are rejected without
  deleting the saved results. See `docs/SIMULATION_MODES.md`.
- Verification: related optics/provider/cache/GUI regression **183 passed,
  16 warnings**, exit 0 (233.86 s). Final model/controller/persisted-artifact
  regression **60 passed, 16 warnings**, exit 0 (91.24 s), including all 14
  model-level cases. These suites overlap; their counts are not additive.
  Warnings are third-party Pydantic deprecations and a pyqtgraph teardown
  disconnect warning. Offscreen rendering confirms menu selection and disabled
  tier contrast. Synthetic FEM tests are not material/OEM/GPU validation.
- No preset optimisation, Git upload or shutdown was performed. Pre-existing
  uncommitted changes remain in the workspace.

## Magnetic-circuit stage 1 checkpoint (2026-09-05)

- See `docs/MAGNETIC_CIRCUIT_MODELS.md` for schema, numerical boundaries and
  the next stage. Optical controls and physical magnetic circuits are now
  separate: an explicit circuit may contain independent poles, shared poles,
  an air-core coil, or a profiled monolithic saturation insert.
- D/I/P1/P2 dimensions, strengths and presets are unchanged. Their shipped
  independent-pole topology remains an engineering assumption. The previous
  reference to FEI US9595359B2 Figure 6 as evidence for those dimensions and
  topology has been corrected; no Titan OEM section drawing was supplied.
- Shared passive bodies affect all member-channel geometry/cache identities
  without becoming extra optical sources. Linear shared-coil responses drive
  only their assigned current channel. Material, mesh and boundary settings
  must agree across generated responses, including cached ones.
- Optional radial profiles use the same knots in Physical Layout and FEM
  material masks. Explicit air-core projector installation no longer restores
  missing legacy pole drawings. Existing named condenser/objective assembly
  constraints still require compatible installation-specific manifests.
- Model Inspector has a compact read-only Magnetic circuits subtab. No view
  refresh starts a field solve. The unresolved mixed-material C1/C2 carrier
  remains a shared dependency, not an invented homogeneous iron volume.
- Monolithic saturation inserts reject the constant-permeability FEM solver.
  Imported maps for that topology are locked to their reference excitation
  and polarity; no nonlinear B-H solver or material calibration is claimed.
  Analytic fallback remains provisional, not a geometry-derived solution.
- Related offscreen regression: **184 passed, 16 warnings**, exit 0,
  258.49 s. After the final shared-operator guard, affected magnetic/provider/
  inspector/physics tests passed again: **37 passed, 16 warnings**, exit 0,
  19.50 s, including three new mismatch cases. Warnings are third-party
  Pydantic deprecations; the broader run also emitted a pyqtgraph teardown
  disconnect warning after completion. These are synthetic/offscreen checks,
  not EOD/MEBS/COMSOL, measured-material, desktop OpenGL or GPU validation.
- At this checkpoint the solver identity was `temsim-solver-2026-09-magnetic-circuits-v2`;
  old implementation seeds are rejected without deleting saved results.
- Next: sourced B-H tables, a coupled nonlinear circuit solve at the complete
  current vector, and independent numerical/material benchmarks. No preset
  optimisation, Git upload or shutdown was performed in this phase.

## Six-stage physics checkpoint (2026-09-05)

- See `docs/SIX_STAGE_PHYSICS_IMPLEMENTATION.md` for current numerical scope
  and verification. Older checkpoints below are historical, not current test counts.
- Final related offscreen regression: **302 passed, 17 warnings**, exit 0,
  274.58 s. Compilation and whitespace checks pass. This is not a full preset
  recalibration, measured-column validation or desktop OpenGL/GPU certification.
- Specimen magnetic flights and downstream reference matching now use the
  registered vector-field providers and the shared relativistic Boris kernel;
  the exact uniform-Bz helper remains an analytical test reference.
- Model Inspector offers an explicit linear axisymmetric A-phi FEM recipe from
  resolved pole/yoke/coil geometry. Relative permeability and ampere-turns must
  be supplied; no alloy calibration, nonlinear B-H curve or preset is invented.
  C1/C2 generated finite fields in an offline check. Current Objective coil/iron
  overlap is rejected; its geometry was not silently changed.
- TEM Camera propagation can retain complex waves through intermediate physical
  aperture planes. Masks remove probability; they do not renormalise it away.
- Probe/Image aberrations have mutually exclusive Manual and Field-derived
  modes. Field fits use production rays without empirical Cs kicks and exclude
  first-order focus already carried by transport. Worker/result caches share
  the fit; fit RMS must not be presented as a corrected beam size.
- Design Explorer accepts multiple runtime-parameter axes on detached recipes
  (GUI Cartesian limit: 64 points). TOML geometry must still be explicitly
  edited/validated before capture; unrestricted geometry sweep overlays are
  not implemented.
- Project-facing additions are English. The living specification and historical
  projector/EDS research have been translated, retaining all prior requirement
  IDs, dated outcomes, citations and physical qualifications.
- No automatic preset optimisation, Git upload or shutdown was performed in
  this phase. Pre-existing uncommitted work remains in the workspace.

## Generic EDS elastic-trajectory checkpoint (2026-08-31)

- The subsystem, GUI, state and results are named **EDS**, not Ultra-X.
  Installed angular geometry now resolves through
  `configs/detectors/eds/EDS.toml` with `eds_system_key = "eds"`. Product
  names remain provenance only.
- `configs/specimen_supports/catalog.toml` and `specimen/support.py` add a
  3.05 mm circular grid model with Cu, Au or virtual vacuum and ten commercial
  50–500 square meshes. Offset and rotation are continuous; zero offset is a
  mesh-opening centre.
- `specimen/elastic_transport.py` now generates seeded, event-driven 3-D
  trajectories in the finite sample and continuous square-grid geometry. It
  resolves specimen faces, mesh openings/sidewalls, bars, annular rim and
  circular outside boundary; samples exponential flights and scattering atoms;
  and records forward, reverse, lateral and safety-limit outcomes.
- The current offline provider uses the Demers et al. relativistic
  screened-Rutherford total/differential cross section for 100–300 keV. It is
  explicitly provisional for Z>30 and is not called ELSEPA, full Mott or
  crystal channeling. A future legally sourced ELSEPA provider can replace the
  public cross-section functions without changing the transport geometry.
- `detector/eds_atomic.py` evaluates the checked-in NIST Bote–Salvat K/L/M
  fits for Z=1–99. `detector/eds_signal.py` converts Monte Carlo-averaged or
  caller-supplied elastically scattered material segments into direct shell
  vacancies, xraylib characteristic lines, emitting-layer self absorption,
  configured solid-angle collection, ideal scalar efficiency, energy
  broadening and optional Poisson counts.
- EDS is a dedicated spectrum-only top-level page. Support, path mode,
  seed/event guard and acquisition controls are hosted in **Sample Interactions
  3D > Parameters**. **Update point EDS** is explicit and synchronous;
  editing a mechanical/support component never auto-runs a spectrum or
  recomputes a lens preset. A straight-primary reference remains selectable.
- There is no independent elastic trajectory count. Each point acquisition
  uses every upstream ray that survives to the physical sample plane, carrying
  its X/Y, tx/ty (including accumulated rotation), energy offset and
  source-current weight. Emitted-current counts are multiplied by the upstream
  survival fraction, then conditional ray weights are applied once.
- Specimen trajectories are shown only in Sample Interactions 3D, with 3D,
  X-Z and Y-Z views. Ray Diagram still synchronizes its arbitrary rotation
  with Transverse X-Y, but no longer redraws duplicate EDS trajectory plots.
- Scanning Image is a fixed horizontal split, not a nested full ScanControlView
  tab. Its left pane owns Scanning Parameters / Probe Aberrations; its right
  pane owns Geometry / Images. The existing scan controller still owns all
  signals, plots and image data, while only its panels are reparented.
- The Images pane has **Pause refresh**. It freezes HAADF/DF/BF on the previous
  complete frame, including across a newly calculated replacement frame, but
  does not stop the scan clock or Ray Diagram playback. Resuming switches to
  the current cached frame and line progress.
- Sample `Mode` is now the only structure-source selector. Real sample exposes
  only user-imported CIF/MCIF; Virtual sample owns the simulator TOML reference
  list (Silicon [110], Gold [001], etc.) plus idealised ray-interaction rows.
  The duplicate Real structure-source combo and `atomic_structure_source` state
  field were removed in schema 71. Schema/profile migration maps retired CIF
  ownership to Real and retired preset ownership to Virtual. Dormant values may
  be retained for non-destructive mode switching, but every calculation uses
  only the source owned by the active mode.
- Current-change validation: all 392 collected tests outside
  `tests/test_direct_alignment.py` pass offscreen. The 34-test Direct Alignment
  module has 30 passes and four existing Nanoprobe convergence-range failures
  after the earlier condenser geometry changes; this specimen-source change
  does not modify that solver or relax its optical thresholds. Targeted
  Sample/profile/wave/EDS/scan/GUI groups pass. `compileall` and
  `git diff --check` pass.
- Checkbox indicators are now global theme assets rather than Fusion defaults.
  Unchecked, checked, partial, hover and disabled states use nine packaged SVGs
  under `src/temsim/gui/assets`; `pyproject.toml` includes them as package data.
  The checked state is cyan with a white tick and the unchecked state is a
  bright outline on the dark background.
- This is not yet a complete coupled electron/photon Monte Carlo. The initial
  beam is the calculated geometrical sample-plane phase space; it does not
  convert a coherent probe wave into unique classical paths. Bulk-density transport has no
  crystallographic channeling/coherent diffraction and is not derived from
  multislice. It also omits electron slowing, elastic recoil, inelastic angular
  deflection, bremsstrahlung, vacancy cascades, cross-layer photon
  absorption, secondary fluorescence, full 3-D holder/pole shadowing and a
  measured energy-dependent detector efficiency. Absolute counts remain
  provisional.
- Validation on 2026-08-31: the focused elastic/EDS/sample tests passed, then
  the non-recalibrating full suite completed as `408 passed, 1 skipped,
  6 deselected`. The six deselections are the two known Nanoprobe C2/C3
  calibration families intentionally not rerun. `compileall`, import smoke,
  `pip check` and `git diff --check` also passed.

## Projection-chamber differential-pumping aperture checkpoint (2026-08-30, historical)

- Both recording TOMLs now contain a distinct
  `projection_chamber_dpa_aperture` at local Z 772.5 mm, exactly at the P2
  housing / projection-chamber boundary and 7.25 mm upstream of the HAADF
  active surface. The display name is **Projection-Chamber
  Differential-Pumping Aperture**; the downstream
  `energy_filter_entrance_aperture` remains a separate Iliad component.
- The configured 0.2 mm bore is documented only as a Tecnai/Talos-family
  reference. Titan-specific bore, material and axial plate thickness remain
  unverified; the TOML therefore stores a zero-length physical boundary and
  Physical Layout supplies only schematic drawing thickness.
- This DPA is fixed, non-retractable mechanical hardware whose diameter can be
  edited as a TOML design variable. It is not in runtime `APERTURE_KEYS`, has
  no optical-reference coordinate, does not clip rays, does not assert a
  conjugate-plane class and does not recalculate any preset lens strength.

## EDS clearance and post-P2 detector-section checkpoint (2026-08-30)

- The EDS active-face/housing polygons no longer intersect the resolved
  Objective pole-piece silhouette. Their position is solved from the larger
  upper/lower pole OD plus a 1 mm display-only clearance. This is not an OEM
  detector dimension or a validated 3-D collimator/shadowing model; angular
  centre/acceptance lines remain metadata and may cross the axisymmetric view.
- Both recording TOMLs contain a mechanical-only
  `post_projector_detector_chamber` from the P2 housing end at local Z 772.5 mm
  through the HAADF, main screen, DF and BF active planes. Its 1100 mm end and
  180/200 mm ID/OD are adjustable non-OEM drawing dimensions. Camera remains
  downstream.
- Current P2-end to active-plane gaps are HAADF 7.25 mm, screen 127.25 mm,
  DF 217.25 mm, BF 287.25 mm and camera 399.75 mm. Public Titan diagrams
  support HAADF being first in the viewing/detector section but do not publish
  the 7.25 mm absolute spacing.
- No detector active plane, Objective/projector geometry, magnetic field,
  optical preset or warm start was changed or recalculated. Manifest and GUI
  regressions enforce the chamber boundary/order and EDS solid-polygon
  clearance.

## Start and verify

- Workspace: `F:\tem_simulator_v2`
- Python: `.venv\Scripts\python.exe`
- Application entry point: `main.py`
- Run: `.venv\Scripts\python.exe main.py`
- Tests: `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest -q`
- Latest non-recalibration result: **394 passed, 1 skipped, 6 deselected** from
  **401 collected** on 2026-08-30. The six deselected cases are the explicitly
  retained Nanoprobe live-solve points whose C2 calibration was not recomputed
  after mechanical changes. Serial `compileall`, `pip check`, `main.py` import
  and the offscreen `MainWindow` show/close smoke check passed.

## Layered aberration checkpoint (2026-08-20)

- `optics/aberrations.py` is the shared intrinsic/system aberration authority.
  Round-lens missing Cs/Cc values resolve to labelled provisional focal-scale
  estimates; explicit values remain authoritative and no unavailable value is
  presented as a physical zero.
- Full effective coefficients live only at probe/specimen and Objective/image:
  C1, A1, B2, A2, C3, S3, A3, C5 and Cc. Non-axisymmetric terms carry an
  azimuth. State schema 65 persists probe/image overrides.
- The Aberrations page compares identical relay optics with nonlinear
  hexapoles disabled/enabled. TEM CTF and STEM probe phase use the same active
  effective set through fifth order. Cc remains a first-order energy-dependent
  defocus term rather than an incorrect single-energy coherent phase.
- This is a principle-correct, non-OEM model. No real instrument calibration
  or energy-filter aberration curve is claimed.

## Transverse X-Y direction-colour checkpoint (2026-08-20)

- The former four discrete initial-quadrant colours are replaced by a
  continuous cyclic HSV direction map, with +X = 0 degrees and angle increasing
  counter-clockwise toward +Y = 90 degrees.
- An on-page DPC-style colour wheel uses the exact ray colour mapping and labels
  all four signed cardinal axes. Colour represents initial polar direction
  only, not radius, energy, intensity or survival state.
- Initial angles are measured about the ray bundle's own start-plane centroid,
  so off-axis translation does not corrupt the rotation diagnostic. A ray
  exactly at that centroid has undefined direction and is shown in neutral
  grey.
- Component selection displays its centre Z, except recording devices, which
  display their upstream top-surface signal Z. Go to Z, axial-plot selection
  and the movable cyan cursor display an arbitrary Z; cursor movement refreshes
  the transverse slice live, and whichever source was selected last remains
  authoritative across recalculation.

## Detector point-spread checkpoint (2026-08-20)

- Camera, Fluorescent Screen and BF/DF/HAADF now carry TOML-authoritative
  `point_spread_*` fields: model, sigma X/Y in detector-plane mm, principal-axis
  rotation, status and source. Current values are deliberately labelled
  `provisional_model_parameter`, not real-instrument calibration.
- `detector/point_spread.py` performs only forward convolution with a unit-sum,
  rotatable anisotropic Gaussian. `detector_response_image()` accepts rays with
  the component's square/disk/annulus hit mask, applies the PSF, reapplies the
  finite sensitive-area mask and exposes accepted, response and retained
  weights. Zero padding permits physically meaningful edge loss.
- Selecting one of those recording devices overlays its response beneath the
  original direction-coloured points in Transverse X-Y. Selecting an arbitrary
  Z or a non-recording component clears the response, so an ordinary geometric
  slice is never presented as a detector image.
- This stage does not alter ray trajectories or the specimen-to-Objective CTF,
  and it does not yet connect Zebra/EELS or EFTEM output readout PSFs.

## Permanent Energy Filter checkpoint (2026-08-20)

- Instrument Setup selects only the gun and column. The Iliad Energy Filter is
  permanent hardware and is the sole selectable recording system.
- `NoEnergyFilter.toml` remains catalogued as non-selectable historical
  geometry. Applying or loading a legacy no-filter selection normalises it to
  `Energy Filter`, while the runtime optical branch can still be disabled.
- The active catalog has 15 selectable gun/column assemblies; validation still
  audits all 10 module TOMLs and 480 variant-scoped part definitions.

## Latest Direct Alignment checkpoint (2026-08-10)

- The left assembly navigator now has Optical, Mechanical and Direct Alignment
  pages. Direct Alignment exposes four TOML-defined operator controls:
  Nanoprobe convergence, Microprobe illuminated diameter, Image magnification
  and effective Diffraction camera length.
- Nanoprobe and Microprobe dynamically solve C2/C3 together. Nanoprobe targets
  the current-weighted 95% radial semi-angle relative to the 3-D chief ray and
  constrains the waist to the sample. Microprobe targets the 95%-current sample
  diameter while constraining radial wavefront curvature and a 0.5 mrad maximum
  semi-angle. Common Larmor rotation cannot change these radial metrics.
- Image dynamically solves Objective/D/I/P1/P2 as one preset against the active
  recording stop. It no longer fixes an Objective intermediate image or assigns
  equal/independent per-lens magnifications. The total sample-to-recording ABCD
  condition is `B=0`, and displayed magnification is `|A|`. Diffraction remains
  a separate distributed-field D/I/P1/P2 solve which relays the live Objective
  back focal plane and reports effective camera length.
- `src/temsim/optics/equivalent_image_lenses.py` derives a non-OEM thin-lens
  power from each isolated post-sample `integral(Bz**2 dz)` and retains its
  signed Larmor rotation as a separate event. The same exact-Z events drive the
  optimiser, production first-order trace and ordinary ray diagram; this is not
  a display-only multiplier. Simulation metrics separately expose signed
  magnification, upright/inverted state and Larmor rotation.
- Image has distinct TOML seed branches. LM covers 10x through 1,000x with the
  Objective nearly bypassed. Normal/HM covers targets above 1,000x with the
  Objective active. Seeds at 10/100/1,000/10,000/100,000/1,000,000x are merely
  deterministic starting currents; arbitrary values such as 65.7x, 333x,
  2,500x, 25,000x, 250,000x and 750,000x were also live-solved successfully.
- The old unused I/P1/P2, X-only magnification prototype is now only a
  compatibility facade over the single Direct Alignment implementation. It can
  no longer write a failed local solution into live state.
- Solves run in a dedicated Qt worker on a canonical state snapshot. A result is
  committed atomically only if validation succeeds and the live state token is
  unchanged; manual edits, assembly/mode changes, failed targets and stale
  background results leave every lens unchanged.
- The regression working points include 30 mrad Nanoprobe, 2.0 um Microprobe,
  the complete decade grid from 10x through 1,000,000x Image, and
  0.01/0.05/0.1/0.5/1/2 m Diffraction. Microprobe is validated over 0.5-2.2 um.
  Fine-grid diagnosis rejected the earlier apparent 1M distributed-field
  solution as a 0.1 mm integration artefact: it failed at 0.05/0.025 mm. No
  implausible multi-tesla projector rating was committed. The equivalent focal
  model is an engineering calibration, not a Talos OEM current table.
- On 2026-08-10 the GUI startup and compatible-assembly reload paths were fixed
  to apply the active TOML operating-mode pair instead of trusting only the
  `State` mode labels. Projector Direct Alignment now uses bounded logarithmic
  continuation plus validation-grid refinement. This removes the false
  approximately 0.10865 m camera-length plateau without changing field ratings.
  The current 5 m request reaches approximately 2.59 m with a small BFP relay
  residual but saturates P2 at its existing 100% limit; 5 m therefore remains
  an explicit field/geometry calibration task, not a claimed working point.
- Optimisation uses a 0.1 mm grid and 0.05 mm validation for the condenser and
  Image controls. The cancellation-sensitive Diffraction relay is optimised at
  0.05 mm and validated at 0.025 mm. The 0.05 m working point has a 6.19 um BFP
  relay residual at the validation step.
- Fine validation uses the production nonlinear propagation path, including
  upstream deflector/corrector kicks, exact saved aperture planes and column
  walls. Invalid ray weights are rejected; an entirely blocked sample bundle
  reports unavailable beam metrics without crashing the mechanical trace.
  Every control also rejects optimiser/validation observable spread above 1%.

## TOML authority checkpoint (2026-08-09)

- Instrument structure now has one final authority: the resolved selection of
  one gun, one column and one project/recording TOML. `main.py` and the Qt
  composition root contain no instrument geometry.
- Every active part has a stable definition ID of the form
  `<module path>::parts[<canonical key>]`; runtime optics, gun components,
  sample, Objective and Energy Filter retain that non-serialised provenance.
- The former gun geometry re-read through the process-default manifest root
  was removed. `AssemblyCatalog(root=...)` now remains authoritative through
  final runtime application, including gun exit, ordinary component geometry
  and Energy Filter/slit geometry.
- Catalog validation rejects duplicate module files, module keys and selection
  signatures; module validation rejects duplicate part keys/orders. Runtime
  layout/state key collisions and missing structural TOML fields fail instead
  of silently taking the last Python object or a class default.
- There are 10 module TOMLs, 480 variant-scoped definitions, 196 logical part
  keys, 284 intentional cross-variant repetitions and 30 collision-free
  selectable assemblies. Cross-variant repetitions are mutually exclusive,
  never an override order.
- Saved profiles omit all TOML-owned positions and structural attributes. The
  canonical Objective Stigmator key is now `objective_stigmator`; `obj_stig`
  remains input-only migration compatibility.
- Energy Filter geometry, energy-slit mechanics, default sample Z, sample
  diameter and stage envelope are TOML-derived. Python continues to own
  algorithms and runtime operating controls, not a second structural model.

## Latest compact projector and recording-surface checkpoint (2026-08-20)

- D, I, P1 and P2 retain the validated optical centres at local Z = 82.5,
  252.5, 432.5 and 635.0 mm. Their complete housing/yoke envelopes are now
  32.5-132.5, 137.5-367.5, 372.5-492.5 and 497.5-772.5 mm, leaving exactly
  5 mm between neighbouring physical assemblies.
- The whole projector stack uses a 20 mm electron-accessible vacuum ID. Pole
  mechanical bores are 21.5 mm so the declared 0.75 mm liner remains outside
  the electron path. Geometry is tagged as a user-defined non-OEM principle
  model; do not present it as a production drawing.
- Camera, Fluorescent Screen and BF/DF/HAADF collect signal on the upstream
  top surface. Their optical reference equals `local_start_z_mm`; runtime
  detector Z, Physical Layout callouts, Ray Diagram markers and Transverse X-Y
  selection all point to this signal surface, while body centres remain
  available as mechanical geometry.
- The exact axial grid now merges coordinate-rounding-sized terminal
  intervals. This prevents a near-zero RK4 interval and singular gradient
  when a restored optical centre lies an ulp away from a regular grid point.

## Historical projector-lens mechanical checkpoint (2026-08-09)

- Both recording-system TOMLs now use the same D-I-P1-P2 Titan/Talos-class
  engineering reconstruction supplied by the user.  Each lens is explicitly
  tagged `engineering_reconstruction_not_oem`; no production drawing or OEM
  measurement is claimed.
- The magnetic-yoke defaults are D = D162 x 68 mm, I = D158 x 60 mm,
  P1 = D158 x 62 mm and P2 = D165 x 70 mm.  The independent TOML coil rows
  now use the supplied ID/OD/axial lengths: 56/140/44, 60/138/38,
  58/138/40 and 66/145/45 mm respectively.
- Pole shoulder OD / bore / gap are now D = 54/10/4, I = 55/12/6,
  P1 = 54/10/5 and P2 = 62/15/7 mm.  The supplied 12/12/13/14 mm nose
  lengths, nominal 63 deg cone metadata, 3 mm face land and R2-R4 fillet
  range are TOML-owned.  Pole-face tip OD remains a clearly disclosed
  schematic value because it was not present in the supplied public data.
- A 0.75 mm vacuum-liner wall gives clear IDs of 8.5, 10.5, 8.5 and 13.5 mm.
  The outer non-magnetic housings add 1 mm radial clearance and a 2 mm shell
  around the reference yokes, so their total ODs are 168, 164, 164 and
  171 mm, all inside the supplied engineering ranges.
- Lens and optical-reference centres remain exactly 82.5, 252.5, 432.5 and
  635.0 mm.  No excitation, field profile, polarity, preset or aberration
  parameter changed.  Python fallback definitions now read these mechanical
  values from the authoritative recording TOML instead of retaining the old
  D250-D300 hard-coded envelopes.
- Physical Layout consumes the explicit pole-nose axial length and reports
  the cone/land metadata in pole tooltips.  Validation checks provenance,
  liner-to-pole clearance, paired pole details, radial housing/yoke/coil
  nesting and unchanged centre/reference coupling in both recording systems.

## Latest exact specimen-interface checkpoint (2026-08-09)

- `src/temsim/physics/core.py` no longer rounds a requested propagation stop
  to the nearest axial grid point. Full RK4 intervals retain the requested
  `step_mm`; only the final interval is shortened so both endpoints are exact.
  CPU, Numba CPU and CUDA now consume the same per-interval step array.
- For the default FEG / C3 + Probe Corrector / Energy Filter preview, the
  incident bundle now ends at the authoritative sample Z = 1599.2 mm and the
  outgoing bundle starts at the same Z. The former 1599.2--1600.0 mm overlap
  is zero, and X/Y are continuous across the single shared specimen plane.
- Plotting was deliberately not used to hide or clip the defect. The physical
  propagation arrays themselves now satisfy `incident.z[-1] == sample.z_mm`
  and `branch.z[0] == sample.z_mm`, so downstream diagnostics that use the
  incident endpoint also recover the correct specimen plane.
- Completing the formerly omitted final 0.03885 mm exposed that the probe
  two-hexapole calibration had been fitted before the true specimen plane.
  The exact-plane calibration is now HP2 = `5.08073490e5 m^-3`, HP1/HP2 =
  `0.59513503`; the HP1 orientation and every mechanical coordinate are
  unchanged. At 0.1 mm its residual is 3.53% of the positive round-lens Cs
  contribution, versus 26.6% with the stale values.
- The probe-corrector residual remains below the existing 20% limit at 0.1,
  0.05 and 0.025 mm (3.53%, 11.27% and 13.86%). It is 43.2% at 0.2 mm, so the
  2.5 mm interactive preview remains a ray-layout schematic; quantitative
  corrector assessment requires the 0.1 mm high-accuracy path or finer.
- Regression coverage includes the previous +0.8 mm overshoot and -0.2 mm
  shortfall cases, the real default preview interface, CPU/Numba parity and a
  non-divisible 900.3 mm CUDA endpoint compared with CPU.

## Latest signed optical-transfer checkpoint (2026-08-09)

- `src/temsim/physics/first_order.py` defines the full paraxial state ordering
  `(x, y, theta_x, theta_y)` in the right-handed column frame and returns one
  signed 4x4 transfer in 2x2 blocks. The observable position relation is plain
  text: `r_plane = J_img @ r_sample + J_diff @ theta_sample`; `J_img` is
  dimensionless and `J_diff` is in m/rad.
- One reference ray plus four transverse basis rays are traced together.
  Subtracting the reference removes affine beam shifts from the Jacobian.
  Spherical-aberration kicks and hexapole nonlinearities are deliberately off
  for this first-order derivative; the ordinary nonlinear ray trace is
  unchanged.
- Ray-simulation metrics now expose both matrices, rotation, handedness,
  anisotropy and image/diffraction conjugacy residuals. Equivalent scalar
  magnification and camera length come from `sqrt(abs(det(matrix)))`, so the
  full signed X-Y map remains available instead of being discarded by `abs`.
- The **Optical Transfer** GUI page shows named objective/recording planes and
  can capture one Image state plus one Diffraction state at the same plane. It
  reports the normalised diffraction-vector-to-image-direction transform,
  including rotation, mirroring and anisotropy. The mapped reciprocal vector
  is normal to lattice planes; it is not a direct-lattice length map.
- Both recording-system TOMLs now own camera-axis rotation, U/V flips,
  uncertainty, status and source. Current values are
  `uncalibrated_identity` with 180 deg uncertainty. The GUI therefore labels
  the result model-only and refuses to imply an absolute hardware crystal
  orientation; current lens polarity provenance is also shown as provisional.
- The matrices cover the straight axial column and expose the Energy Filter
  entrance as a chain boundary. They do not yet include the curved sector,
  M01--M10 or Zebra detector axes. No mechanical coordinate was changed.
- Validation includes the exact field-free drift matrix, signed
  rotation/reflection/anisotropy decomposition, calibrated and uncalibrated
  detector-frame relations, TOML rejection cases, simulation metrics and the
  two-mode Qt capture workflow. A separate read-only audit produced finite
  transfer matrices for all 30 assemblies.

## Latest magnetic-field polarity checkpoint (2026-08-09)

- All 52 configured optical magnetic-lens definitions now own
  `field_polarity = +1/-1`, `field_polarity_status` and
  `field_polarity_source` in their selected instrument TOML. The sign is the
  effective Bz direction in the source-to-detector +z coordinate and remains
  independent of the non-negative 0–100% excitation magnitude.
- The former `fei_column_polarity.py` Python tables and every automatic
  reapplication of those tables were removed. Constructors use the default
  manifest, and selecting a new assembly applies the selected manifest with
  `preserve_operating_parameters=False`.
- Ordinary recalculation continues to preserve a user's runtime `polarity`
  override. Microprobe/Nanoprobe Mini Condenser reversal is now an explicit
  `field_polarity` entry in `configs/operating_modes/catalog.toml`, not an
  illumination-mode branch in Python.
- Manifest validation rejects missing, Boolean, floating-point or non-unit
  signs and requires controlled provenance status plus a non-empty source.
  Magnetic Field tooltips report the selected sign, status and source.
- All present common-column, projector and corrector signs are tagged
  `provisional_model_assumption`. A negative AutoScript/FLC raw value must not
  be used as proof of negative Bz; replace a provisional sign only with service
  coil/current mapping or an absolute image/diffraction rotation calibration.
- The repeatable migration/check is
  `scripts/migrate_field_polarity_to_manifests.py`. A catalog-wide audit matched
  384 selected magnetic-lens instances across all 30 assemblies to their TOML
  signs without changing any mechanical coordinate.

## Latest layout-to-editor navigation checkpoint (2026-08-09)

- Physical Layout and the separate Energy Filter branch now treat component
  names, centre markers and drawn bodies as single-click navigation targets.
  Dashed leaders remain visible guides but are deliberately not click targets,
  so a leader crossing a housing cannot steal that housing's selection. Each
  selectable item resolves to the existing canonical component key; no
  duplicate display-only component registry was introduced.
- A plot selection reopens/raises the instrument dock, switches to the correct
  Optical filter or Mechanical page, selects the tree item and expands the
  parameter half of the left splitter. Runtime targets open on **Operating**;
  static assembly parts open on **TOML**. Re-clicking the already selected
  element deliberately refreshes the editor context.
- Iliad XO / Optional EFTEM Energy Slit exposes only its live `inserted`,
  requested centre-loss
  and requested energy-width controls. Centre and width edits go through
  `configure_energy_window`, so blade centre and physical gap stay consistent
  with calibrated dispersion and travel limits. Mechanical slit geometry and
  derived blade positions remain TOML-owned/read-only in this interface.
- GUI regressions exercise a Physical Layout label click, Iliad slit and
  dynamic-focus-quadrupole label clicks, an M01 housing-body click, TOML
  routing for mechanics without a field model, and the complete slit
  navigation/editing path.

## Latest Iliad public-topology checkpoint (2026-08-09)

- `EnergyFilter.toml` no longer stores all branch geometry on the synthetic
  `energy_filter` interface. That row now owns only the branch interface and
  public-topology metadata. One large tapered prism, M01--M10, XO/optional
  EFTEM slit, dynamic-focus electrostatic quadrupole, bias tube, fast shutter,
  camera deflector, optional EFTEM output plane and Zebra each have one unique
  TOML row and canonical key.
- Validation requires exactly one prism and ten multipoles. It records that
  most multipoles are publicly described as dodecapoles but keeps each
  numbered pole assignment `not_public`; no guessed production BOM is
  presented as fact. M01--M10 are explicitly simulator model indices because
  public evidence does not expose production labels or exact internal order.
- The previous 135 mm radius / 90 deg bend and all carrier/device coordinates
  remain usable but are now explicitly `provisional_parameterized_non_oem`.
  The 90 deg value is tagged as a patent-example starting point, not a
  confirmed Iliad product angle. Unknown prism yoke, multipole bore/envelope,
  electrode and detector-package dimensions remain editable rather than
  manufacturer-labelled.
- M01--M03 retain 22 mm provisional housings and M04--M10 retain 28 mm, with
  independent per-element TOML ownership and 20 mm magnetic supports.
  Validation rejects support/housing inconsistencies, overlaps and wrong path
  ordering. Runtime multipoles retain the definition ID of their own TOML row.
- The entrance aperture uses the reported 5 mm experimental condition without
  treating it as the only installed mechanism size. Zebra now correctly uses
  28.672 x 0.800 mm for each 2048-pixel strip. The separate 256 x 2048
  alignment area is 3.584 x 28.672 mm; the old model incorrectly used its
  3.584 mm height as the strip acceptance. Strip pitch and package remain
  explicitly unknown; the current 1.0 mm pitch is provisional and editable.
- The dedicated Energy Filter view now draws the entrance aperture, prism
  clear path, ten hollow carriers, XO/slit, provisional electrostatic
  envelopes, EFTEM output and the Zebra active plane. Curvilinear-only parts
  are excluded from main-column axial markers and Physical Layout records.
  X and Z remain independently zoomable.
- The dynamic-focus electrostatic quadrupole is present as a four-electrode
  mechanical placeholder with
  `mechanical_layout_only_dynamic_focus_field_not_implemented`; it does not
  silently add an unvalidated field or an eleventh member to the confirmed
  ten-multipole system. The slit and shutter remain separate components with
  separate functions.

## Latest wave-optics checkpoint (2026-08-09)

- The symmetric split-operator multislice engine accepts both total 2-D
  projected potentials and explicit `(Z, Y, X)` finite-projection slices. It
  supports rectangular grids and independent X/Y sampling.
- NumPy complex128 remains the CPU reference. Optional CuPy complex64 supports
  multislice and TEM/STEM FFTs with a complete CPU retry after CUDA, driver,
  allocation or FFT failure. Install with `pip install -e ".[gpu]"`.
- Angle-resolved STEM now has one device-resident compound CUDA pipeline:
  scan positions and all potential configurations upload once; probe formation,
  multislice, diffraction FFT, frozen-phonon intensity averaging and detector
  masks remain on device; one stacked detector array returns to the host.
- Each resident multislice calculation builds one reusable CUDA plan. The
  reciprocal-frequency grid, 2/3-bandwidth mask and uniform or nonuniform
  slice-geometry propagators remain cached across all probe batches and all
  potential configurations. Potential shape/finite checks and maximum-phase
  scans run once per configuration instead of once per batch.
- Any resident-pipeline failure discards every partial CUDA result and reruns
  the complete STEM observable on the NumPy complex128 reference. Metrics make
  residency, fallback, upload count, batch count and result-transfer bytes
  visible.
- Silicon [110] and gold [001] now own TOML atomistic definitions. Optional
  abTEM 1.0.10 + ASE build unstrained, commensurate periodic supercells and
  Lobato--Van Dyck finite Z-slice neutral-atom IAM potentials. Install with
  `pip install -e ".[atomistic]"`.
- Frozen phonons use reproducible independent isotropic Gaussian (Einstein)
  displacements. The preset or user value is the one-axis RMS sigma. TEM and
  STEM average configuration intensities, not complex amplitudes, and report
  a finite-ensemble relative standard error.
- The model intentionally excludes bonded charge redistribution, correlated
  phonons, absorptive/inelastic potentials, magnetic specimen fields and spin.
  The STEM wave path does not add a separate Rutherford/TDS tail, so frozen
  phonon scattering is not double counted.
- The Sample parameter page exposes atomistic/frozen-phonon toggles,
  configuration count, RMS sigma and random seed. Dependent controls disable
  when multislice, atomistic potential or frozen phonons are inactive. Saved
  operating profiles round-trip all five fields.
- Source-qualified thermal defaults are 0.085 angstrom for Si near room
  temperature and a rounded 0.080 angstrom 300 K Debye estimate for Au. Both
  references are stored in their specimen TOMLs.
- Numerical regression checks crystal number density, abTEM transmission-phase
  units, seeded reproducibility, TEM/STEM incoherent averaging, rectangular
  sampling, integrated intensity, reusable-plan identity, nonuniform slice
  geometry, and CPU/CUDA agreement for atomic slices.
- Warm CUDA benchmark (96 x 96, 12 slices, 64 scan positions, two potential
  configurations, two detectors, batch 8): resident median 0.1157 s versus
  0.1484 s for the former per-configuration host round trip (1.28x). Bulk
  array transfers fell from 32 / 13.5 MiB to 1 / 1.5 KiB. This is a local
  software benchmark, not a universal hardware-performance claim.
- Isolated warm propagation benchmark for the same 96 x 96, 12-slice,
  64-probe, two-configuration workload: rebuilding device grids and
  propagators for every call took a 0.0677 s median; one plan reused for all
  16 calls took 0.0505 s including its build and one-time potential checks
  (1.34x, relative L2 difference 0). The plan retained 189 KiB of cached
  arrays. This is likewise a local software benchmark rather than a hardware
  guarantee.

## Scan / descan checkpoint (2026-08-11)

- Sample now has an explicit inserted/retracted state. Retraction preserves the
  exact sample Z as the optical probe-reference plane but disables ray
  diffraction/diffuse broadening and sets interacting wave thickness to zero.
  A dormant or invalid CIF path is not touched while the holder is retracted.
  Ray Diagram and Physical Layout both distinguish the parked holder from an
  inserted specimen.
- AC Scan is one shared command driving physical upper and lower foils. The
  lower-foil X/Y map is recalculated from the active signed first-order column
  optics so the combined sample-plane angular response is zero. A field-free
  equal-and-opposite pair is used only if the lower response is singular.
- AC raster scale is now specified by X/Y pixel counts and one square
  specimen-plane pixel pitch. The control domain is 0.001 nm through 1 mm and
  the displayed FOV is exactly `count x pitch`; raster coordinates denote
  pixel centres, so centre-to-centre span is `(count - 1) x pitch`. A second
  signed 2x2 calibration maps those requested specimen axes through the active
  optics to coil commands. Singular transfers and demands above the physical
  coil limit fail explicitly instead of silently changing the FOV.
- AC Scan and AC Descan now expose exactly the same raster and two-foil
  controls. Shared pixel count, line count, pitch, and frame period are kept
  synchronized; the derived lower-foil control is read-only for both pairs.
  All five column TOMLs place their foil-pair centres at equal distances on
  opposite sides of the sample and keep identical foil length, gap, and
  effective thickness. Manifest validation rejects geometry that breaks this
  symmetry.
- Scan geometry and STEM acquisition use the same physical foil planes and
  signed 2x2 response model; neither path substitutes one kick at a mechanical
  centre. Descan receives the exact negative of the calibrated AC command.
  Its lower-foil 2x2 coupling is solved through the current post-sample optics
  so the combined AC plus Descan displacement vanishes at the Selected Area
  Aperture image-reference station. Singular or unreachable solutions fail
  transactionally and restore the previous state.
- Objective Aperture and Selected Area Aperture are reported as physical
  diffraction- and image-reference stations. They are not forced to carry
  those ideal labels under arbitrary lens settings: the sample-to-plane
  position and angle Jacobians classify each current plane as `image`,
  `diffraction`, or `mixed` and expose both residuals. Objective first-image
  and first-diffraction coordinates are refreshed after voltage, sample, or
  objective-lens changes. Propagation grids include requested plane Z values
  exactly instead of reading them from a coarser display-history sample.
- One active AC raster produces exactly one HAADF, DF and BF frame by
  integrating the physical detector acceptance at every probe position.  The
  interactive Preview uses the geometric detector-interception approximation;
  High accuracy can use the wave/multislice specimen model.
- Wave STEM applies descan as a first-order, per-probe shift of each physical
  detector's equivalent angular acceptance.  This is an explicit approximation
  and does not claim a full time-dependent post-specimen wave propagation.
- The calculated frame is cached in the STEM page.  While AC Scan
  remains enabled, a GUI timer repeatedly plays its raster-line acquisition;
  stopping scan stops the timer and retains the last complete frame.  Playback
  never launches repeated physics calculations.
- Pausing image refresh is distinct from stopping scan: HAADF/DF/BF retain the
  previous complete frame while the timer and Ray Diagram continue. A newly
  calculated frame remains hidden until refresh resumes.
- The calculation also caches AC/Descan first-order response bases on every
  displayed branch Z grid. Each playback tick adds only the current scan
  displacement to the cached rays. The View Angle projection remains live, so
  the diagram can be rotated while a frame is playing without retracing the
  column.
- Each BF/DF/HAADF panel reports its instrument-TOML Z, inner/outer active size
  and derived collection-angle interval. The angle uses the full active signed
  sample-to-detector transfer and exposes anisotropic min/max ranges when the
  two singular values differ materially.

## Specimen-mode checkpoint (2026-08-12)

- A central Sample tab now owns insert/retract, Real/Virtual mode, finite X/Y
  size and thickness, sample centre and scan origin. Its immutable geometry
  snapshot is shared by the renderer and STEM calculation and carries the
  finite box, scan FOV, calculation ROI, current probe, orientation and region
  state. OpenGL displays atoms/cell/box/+Z beam/FOV/ROI when supported; Qt
  offscreen/minimal or missing OpenGL uses the safe 2-D view.
- The Sample tab is also the sole GUI owner of specimen presets, TEM/STEM wave,
  multislice, frozen-phonon, Virtual interaction and probe-convolution
  controls. The duplicate `sample` entry and quick controls were removed from
  the left instrument tree; clicking the specimen in a diagnostic layout
  activates the central Sample tab instead.
- Sample owns specimen state/structure only and no longer duplicates the
  BF/DF/HAADF images. A custom CIF is orthogonalised with abTEM and expanded
  periodically through the finite-sample/current-ROI intersection. The view
  uses ASE covalent-neighbour bonds, reduced covalent-radius balls, ASE/Jmol
  colours and a beside-view element legend. Its default 2,500-atom soft limit
  may reduce only a clearly reported display window; the multislice ROI is
  unchanged. Above 3,000 user-selected atoms OpenGL uses point-sphere level of
  detail; the safe 2-D fallback remains a coloured ball-stick projection.
- The former Scan / Descan top-level tab is now STEM with exact Geometry and
  Images subtabs. Detector images are placed on centre-derived physical pixel
  edges in laboratory micrometres, their X/Y unit aspect is locked to one, and
  normal PyQtGraph pan/zoom remains enabled. A model notice explains that
  `geometric_detector_interception` polygons are detector-clipping boundaries,
  not specimen contrast; CIF multislice requires High accuracy with wave/
  multislice enabled. Sampling diagnostics compare FOV with the finite sample
  and pixel pitch with half the shortest periodic CIF atom spacing.
- `atomic` (UI: Real sample) accepts only a user CIF/MCIF. `virtual` owns one
  instrument TOML reference specimen and the separately configured idealised
  angular channels. One canonical `(w,x,y,z)` unit quaternion controls the
  physical orientation. A direct-lattice zone axis maps to laboratory +Z, a
  non-collinear in-plane direction maps to +X, and numeric or explicit mouse
  edit mode updates the same quaternion. Camera orbit remains the mouse
  default and draft physical edits require Apply.
- Custom CIF structures are loaded with ASE and orthogonalised with abTEM.
  Potential construction generates only the periodic neighbourhood needed by
  `scan ROI + probe padding` intersected with the finite specimen before exact
  rotation/cropping. It never expands a macroscopic sample in full. Outside
  the finite X/Y envelope is explicit vacuum; the 5,000,000-atom safety limit
  applies to the ROI-local pre-crop structure.
- A Real custom CIF requires atomistic IAM and multislice. It never borrows the
  dormant Virtual TOML reference's atoms, material constants or thermal
  displacement and never silently falls back to another material. Frozen
  phonons accept a global user RMS or an explicit per-element RMS table.
- Real sample Ray Diagram calculations never synthesize `+g/-g` or diffuse
  diffraction branches, even if legacy ray-preview fields remain in a loaded
  profile. Coherent CIF or Virtual-TOML elastic diffraction belongs to the
  high-accuracy TEM/STEM wave/multislice calculation. Separate Real-CIF
  populations represent zero-loss, plasmon/low-loss, core ionisation, optional
  other inelastic loss and plural events. They use absolute Poisson
  probabilities, representative energy offsets and characteristic-angle
  quadrature; they are not user-authored diffraction beams. Only Virtual mode
  can create user-defined angular interaction branches.
- Ray Diagram hue is keyed to canonical interaction kind (incident, Real
  zero-loss/plasmon/ionisation/plural, or Virtual direct/diffraction/diffuse/
  arbitrary/Rutherford). Within each hue, five dark-to-bright
  bins encode the exact 3-D sample-plane convergence semi-angle relative to
  that branch's current-weighted chief ray. A common interaction kick is thus
  not counted as convergence; brightness saturates at the incident bundle's
  calculated 99%-current convergence semi-angle.
- Selecting/dragging an axial Z cursor calculates a current-weighted plane
  interaction budget using every ray weight (not the 48-ray display subset).
  It reports sample-conditional probabilities, source fractions at Z,
  composition at Z, effective absorption and pre/post-sample stops; these
  categories close to source probability one. Elastic wave redistribution can
  coexist with every inelastic energy state and is never presented as an
  exclusive collision label.
- `virtual` mode owns extensible diffraction spot/ring, Gaussian diffuse,
  arbitrary angular, user screened power-law, physical screened relativistic
  Rutherford and absorption rows. All probabilities are absolute and are
  rejected above one; they are not silently normalised. The direct beam is the
  exact remainder. The physical row integrates `2*pi*sin(theta) dtheta` and
  uses `P=1-exp(-N_areal*sigma)`; it is explicitly not a Mott calculation.
- Virtual rectangles, ellipses and NPY/PNG/TIFF grayscale maps define density
  inside the finite slab. Outside is vacuum. The per-pixel interaction
  probability is convolved with the calculated probe when enabled.
- High-accuracy wave intensity is integrated only over strict reciprocal-space
  support and is not renormalised when a detector extends outside it. An
  optional, separately reported Rutherford approximation begins strictly
  beyond that support and scales the wave channel to preserve probability;
  it is disabled by default. Bonding charge, absorptive/inelastic multislice
  potentials, energy-differential dielectric/EELS spectra, magnetic scattering
  and full Mott elastic scattering remain out of scope. Stochastic inelastic
  populations are instead transported by the separate IMFP model.
- Each STEM result now carries detector fraction images, pA, expected electrons
  per pixel, optional reproducible Poisson counts, dwell, uncollected/absorbed/
  truncated channels, separate high-angle-tail images, laboratory axis/order
  metadata and the sample-plane ProbeState. Physical detector `hit_mask`
  geometry is evaluated after the full signed 2x2 transfer in axial order.
- Operating profiles are format v2 for quaternion/zone metadata, interaction
  and region tables, map paths and per-element RMS values. Format v1 remains
  readable and its two legacy relative-weight controls migrate once to
  absolute-probability rows. Detector geometry remains instrument-TOML owned.

## Current validated state

- The instrument catalog contains 10 module TOMLs, 480 part definitions and 15
  selectable Energy Filter assembly combinations.
- High-accuracy defaults target a 32 GiB workstation and use a conservative
  24 GiB application-memory preflight limit.
- Magnetic-lens excitation is limited to 0–100%. A lens needing a stronger
  field must raise its calibrated 100% field value rather than exceed 100%.
- Column/vacuum walls only cut off rays. They do not clip propagation or the
  mathematical support of a lens magnetic field.
- Ray stops use the earliest physical intersection among vacuum walls,
  apertures and recording devices.
- Ray Diagram supports continuous transverse viewing angle. Rotating X/Y keeps
  the current zoom, Z scale and Z=0 position unchanged.
- Physical Layout names use one dynamic screen-space callout packer. Ordinary
  components, upper/lower Objective mechanics, stage/holder and recording
  devices can occupy many rows above or below the column; a dashed leader ties
  every visible name to the component centre/outer mechanical edge. Relayout
  changes display coordinates only and never alters TOML geometry.
- Gun trajectories retain equal-laboratory-time snapshots and per-ray arrival
  times for future wavefront/arrival-arc overlays at important planes.
- Operating-mode storage provides calculated condenser `micro_probe`/
  `nano_probe` and projector `imaging`/`diffraction` lens/aperture values.

## Mechanical lens model

Every configured round magnetic lens currently has one optical parent plus
independent mechanical children:

```text
<lens>                         optical/field parent; no material drawing
├── <lens>_housing             non-magnetic outer housing
├── <lens>_yoke                soft-magnetic yoke
├── <lens>_excitation_coil     insulated copper winding package
├── <lens>_upper_pole          upstream pole piece, where applicable
└── <lens>_lower_pole          downstream pole piece, where applicable
```

Mechanical-only parts use these TOML fields:

- `local_start_z_mm`, `local_center_z_mm`, `local_end_z_mm`, `length_mm`
- `vacuum_inner_diameter_mm`
- `mechanical_inner_diameter_mm`, `mechanical_outer_diameter_mm`
- `mechanical_profile`, `mechanical_part_role`, `material_class`
- `mechanical_only = true`, `parent_key`
- `mechanical_overlap_group`, `mechanical_overlap_role`,
  `mechanical_overlap_reason`
- Excitation coils additionally use `field_source_key = "<parent lens>"`.

The mechanical children never create optical elements or additional magnetic
field sources. Existing lens strengths, apertures and field formulae were not
changed by the split.

Canonical profiles and key helpers are in
`src/temsim/mechanical_profiles.py`. Assembly validation is in
`src/temsim/module_manifest.py`. The repeatable migration is
`scripts/migrate_lens_mechanical_layers.py`.

## Confirmed and provisional topology

Confirmed:

- C1 and C2 are adjacent hollow cylindrical lens assemblies.
- The C1–C2 interface has one C1 lower pole and one C2 upper pole facing each
  other. Their gap midpoint is the confirmed target region for the C1–C2
  crossover.
- A three-condenser system gives C3 its own two-pole/single-gap assembly.
- Diffraction Lens and Intermediate Lens mechanical envelopes do not overlap;
  they retain at least 5 mm axial clearance for future pole-piece configuration.
- The innermost continuous wall is called the **vacuum liner**. Use
  **alignment tube** only for a separately adjustable alignment component.

Provisional:

- Housing/yoke/coil radial dimensions are schematic initial ratios derived
  from each existing lens envelope, not measurements of FEI hardware.
- Diffraction Lens, Intermediate Lens, P1 and P2 are presently represented as
  independent two-pole/single-gap lenses.
- A real FEI projector column may share a central pole piece or magnetic yoke
  between adjacent excitation stages. Public information does not yet confirm
  that every software-named projector lens owns two mechanically independent
  pole pieces.
- Pole geometry currently affects the mechanical drawing only; it does not
  reshape the solver magnetic field.
- Vacuum-liner wall thickness is a provisional 0.25 mm.

Do not label provisional projector topology as an exact FEI/Titan mechanical
reconstruction without a service drawing, section drawing or measured part.

## Ultra-X EDS geometry checkpoint

Every selectable column now includes one `eds_detector_system` aggregate at
the sample plane. It is a transverse, mechanical-only child of the Objective
assembly and is explicitly excluded from axial vacuum-wall ownership and the
electron-optical layout. Physical Layout projects two opposing azimuths of the
six-segment array and draws its angular acceptance; the head size and distance
are display-only schematics.

There is exactly one installed product definition:
`configs/detectors/eds/UltraX.toml`. Each of the five column TOMLs owns only
the local placement row and references that definition. There is no Super-X
TOML, second EDS aggregate or detector selector; Super-X remains research
context only.

The retained evidence-bearing values are `>4.45 sr` unshadowed and `4.04 sr`
with the analytical double-tilt holder. Six segments are supported by a
published instrument report and the previously inspected six-stream user EMD,
not by the current OEM datasheet. The 32.06 degree take-off value is likewise
single-instrument user metadata. Ultra-X active area, sensor distance, crystal
shape and package dimensions remain `not_public` and must not inherit the
Super-X 30 mm2 value. Full sources and the projector-lens research record are
in `docs/TEM_PROJECTOR_AND_EDS_GEOMETRY_RESEARCH_2026-08-30.md`.

## Real-part photo intake

Photos can be used to replace provisional dimensions. For a useful measurement
handoff, request:

- Front, side and top views, photographed as orthogonally as possible.
- A ruler, caliper or other known dimension in the same plane as the part.
- At least one confirmed dimension such as bore diameter, flange diameter or
  mounting-hole spacing.
- Part name/number and the lens/stage it came from.
- Disassembled or section views when internal geometry is relevant.

For each inferred dimension, record:

```text
value_mm = ...
uncertainty_mm = ...
evidence = "photo/drawing/measurement identifier"
confidence = "confirmed | measured | estimated | provisional"
```

A single unscaled oblique photo supports shape and ratio estimates only. It
cannot establish reliable absolute dimensions, hidden bores, winding data,
material grade, permeability or saturation field.

## Next work

1. Measure absolute camera/display axes and image/diffraction rotation using a
   known specimen/stage reference, then replace the identity calibration and
   provisional field polarities with evidence-backed values and uncertainties.
2. Extend the first-order transfer chain through the curved Energy Filter
   sector, M01--M10 and Zebra/output detector before using EFTEM/EELS output
   coordinates for crystallographic orientation.
3. Add thickness-dependent frozen-phonon convergence studies and report how
   detector/image uncertainty changes with configuration count. Do not encode
   one universal "converged" count.
4. Add a real amorphous atomistic builder and an explicitly separate
   absorptive/inelastic model before making quantitative HAADF or EELS claims.
5. Replace the analytic Objective field envelope with a source-qualified
   upper-pole/gap/lower-pole field distribution or an imported field map.
6. Continue ingesting real-part photographs or section drawings and replace
   provisional housing/yoke/coil ratios with evidence-backed dimensions.

## Guardrails for future changes

- Do not alter all lens strengths while crossover locations are incomplete.
- Keep every operating excitation at or below 100%.
- Do not use mechanical lens dimensions to truncate a lens field.
- Do not make a mechanical-only child contribute a second magnetic field.
- Preserve TOML as the authority for static geometry and optical references.
- Preserve instrument TOML as the authority for default magnetic-field
  polarity and its provenance; do not reconstruct Bz signs from signed raw
  hardware-control values.
- Preserve signed 2x2 optical maps until the final scalar display. Never use a
  single absolute transfer coefficient to infer rotation or handedness.
- Do not claim absolute crystal orientation while detector axes or contributing
  magnetic-field polarities remain uncalibrated/provisional.
- Keep frozen-phonon scattering and any empirical Rutherford/TDS tail mutually
  exclusive unless a validated coupling explicitly prevents double counting.
- Never call the IAM result a bonded-charge, inelastic or magnetic specimen
  calculation.
- Validate all 15 selectable assemblies and run the full test suite after
  topology edits; retain manifest validation for the historical no-filter TOML.
- Update this file whenever a provisional assumption becomes measured or
  confirmed, or when the next-work ordering materially changes.

## Known documentation mismatch

No known catalog-count mismatch remains at this checkpoint.
