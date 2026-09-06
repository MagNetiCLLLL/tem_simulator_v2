# Ray Diagram and Live Tuning

The **Live tuning** dock opens from **View > Live tuning** or the **Live tuning**
button in **Ray Diagram**. It replaces the former Interactive Calculation tab.
Live sliders change current component settings and update the single Ray Diagram. The
optional **Advanced bank** remains a detached experiment and never changes live
settings or replaces a main-window High accuracy result. Neither mode solves
a lens preset automatically.

The dock contains two resizable columns: range selection and excitation/live controls.
Each control has its component name on the left and its
slider/value on the right. Range and control rows share their vertical position,
height and scrolling, including after resizing or expanding Advanced bank.
It is initially closed and shares the left dock area with Instrument setup and
parameters. It can be shown, closed, moved, tabbed or floated using normal Qt
dock controls. Visibility, placement and floating size use the existing saved
workspace layout; internal column widths are remembered separately. Closing
the dock only hides it: ranges, slider values, completed banks and readouts stay
in memory, and running work is not cancelled or restarted. It does not change
the main ray plot's user-defined zoom or projection.
These preferences belong to the selected **View > Layouts** entry; see
[Workspace layouts](WORKSPACE_LAYOUTS.md) to keep different arrangements.
Ray Diagram contains **Rays** and **Cached signals** subpages. The latter hosts
the detached Advanced-bank physical-detector table, with a separate status from
live tuning. TEM and STEM have no duplicate viewers there: use **Result source**
in **Illuminating Image** and **Scanning Image > Images**, respectively.
Each viewer independently selects **Current calculation** or **Advanced bank**.
Preview frames do not redraw or relabel those cached signals.
Adding a unit fills a read-only **Reference** from the current settings, not a
stale capture. It can be selected and copied. Minimum/Maximum remain explicit;
adding a reference does not change lens values or request a calculation.
Controls from an existing live plan/bank that are absent from an edited draft
remain available below the matched rows, labelled **Active only**.

## Recommended workflow: tune first, calculate signals once

1. Select **Preview: fast rays** or **Medium: sampled beam** in the toolbar.
   Existing left-side lens controls update continuously; no bank is required.
2. Open **Live tuning**. For bounded sliders, use **Capture current settings**, add lens or aperture
   operating parameters, enter
   explicit **Minimum / Maximum**, then select **Start live tuning**. All live
   values are continuous, including lens excitation. The small ray bundle is
   recalculated at the requested value; lens interpolation is not used.
   Detector geometry belongs to the detached Advanced bank, not live tuning:
   it is TOML-owned and must not be silently lost during a worker snapshot.
   Mixed lists are supported: detector rows display **Advanced bank**, while
   lens/aperture sliders remain usable and aligned. Detector draft bounds are
   ignored for live tuning, but remain required and validated for bank builds.
3. Inspect **Ray Diagram > Rays**. User plot limits stay fixed.
   Dragging updates continuously: edits are sampled at most once every 50 ms,
   with one ray calculation in flight and only the latest pending settings.
   Intermediate complete frames are labelled **Updating**. Release flushes the
   final value. Display rate depends on solver and drawing time, not mouse speed;
   50 ms is an input throttle, not a promised frame time.
4. After tuning, select **Run high-accuracy once at current settings**. This
   flushes the final slider value and submits one current-state calculation,
   not a sweep. Existing TEM/STEM/EDS enablement still determines its products.

Preview traces 49 deterministic source samples with a maximum 1 mm integration
step. Medium traces 160 interior samples plus 33 source-support probes at a
maximum 0.25 mm step. The support probes span source-position and angle azimuths,
plus the central ray; their current weights are zero. Interior samples retain
the configured truncated Gaussian source and energy distribution. These are
sampling/resolution tiers, independent of the Simulation menu's physical-model
tiers; they do not silently replace a selected field model with ideal optics.

Both tuning tiers omit specimen scattering, multislice, spectra, scan-frame
generation and full transfer diagnostics. They retain the selected column field
model, static deflection and physical clipping. Dynamic raster drive is paused
in the detached tuning snapshot only. Medium shading joins sampled limits: it
is not electron density or guaranteed beam support. Outer rays alone cannot
determine interior aperture losses or nonlinear caustics. A small bundle may
miss a narrow opening; quantitative signals still require the final calculation.

Auto can use a serial Numba instance of the existing RK4 equations for small
tuning bundles, with NumPy fallback. CPU/GPU preferences remain available.
Ordinary model edits reject obsolete workers. During live slider motion, the
current ray frame finishes before calculating the latest accumulated settings;
intermediate frames never write their older values back to the live controls.
An explicit High accuracy request clears pending live work. The detached
Advanced-bank readout keeps its existing debounce and generation guards.
Cancellation checks bracket integration segments; an active native kernel is
not force-killed. Previous high-accuracy images/spectra remain stored and are
marked stale after edits, never erased because a low-count preview missed a stop.
Nonlinear/FEM field solves and the first JIT compilation can still be slow.

Repeated exact Preview/Medium settings now use a bounded result history. Rotation
and pan reuse display data; scalar edits no longer rebuild the instrument before
worker submission. Retention limits are available under **Simulation > Performance
and cache...**. See [Ray interaction and cache settings](RAY_INTERACTION_PERFORMANCE.md)
for budgets, persistence and benchmark scope.

Preview/Medium freezes a small independent request before returning control to
the GUI, then prepares the full snapshot and cache identities in the background.
The same request token covers preparation and tracing; edits during preparation
do not cancel every frame or mix parameter versions. Hidden Physical Layout,
Magnetic Field and Transverse panels consume only the latest result on reopening.
Their delayed presentation does not delay or clear shared scientific products.

## Optional advanced multi-point bank

1. Configure the sample, installed/inserted components and requested TEM/STEM
   wave products in their existing pages. Set High-accuracy rays and Step in
   the toolbar. Finish any running calculation or alignment.
2. Select **Capture current settings**. Existing complete high-accuracy results
   are offered as dependency-checked seeds, not modified.
3. Add controls and explicitly enter **Minimum** and **Maximum** for each.
   Blank, non-finite, reversed, duplicate and invalid component ranges fail
   before construction. Units are shown on each row; aperture sizes are diameters.
4. For **Precompute** controls, set the number of exact samples. Endpoints are
   included. Multiple optical axes form a Cartesian grid (maximum 256 points).
   **Readout** controls are continuous and do not multiply the optical grid.
5. Expand **Advanced bank**, set the pinned RAM limit and select
   **Build high-accuracy bank (advanced)**. Optical combinations
   and actual per-stage progress are reported. Expensive unmodified products use
   the existing dependency-scoped pipeline. Construction is transactional:
   failure, cancellation or changed external inputs never publishes a partial bank.
   A conservative first-point storage projection stops an oversized request
   early. It can overestimate shared storage; live tuning avoids this bank cost.
6. Use **Cached controls** in the dock's right column and inspect
   the detector table in **Ray Diagram > Cached signals**. For images, select
   **Advanced bank** in the usual TEM/STEM image viewer. The controls belong to the
   completed bank, not to subsequently edited draft ranges or live settings.
   Optical choices select calculated nodes; readout controls update after a
   short debounce. A superseded response is discarded.

The splitter size is retained through QSettings. The plan can be exported as
JSON; export describes ranges only and is not a portable simulation checkpoint.
The RAM bank is session-local. Capture/build again after changing the sample,
assembly, field map or an unlisted parameter. No automatic extrapolation occurs.

### Shared image viewers

- **Current calculation** displays the main calculation result. **Advanced bank**
  displays the last completed bank readout at its selected optical node and
  declared readout values. Both use the same parameter schema and physical
  pipeline, but a bank retains settings from capture time rather than following
  later edits to the main instrument.
- Selecting a result source only selects stored products. It does not submit
  propagation, rebuild a bank, reproject a wave, recollect detector data or write
  cached parameters back into the instrument. Changing **Cached controls** still
  requests the existing bank readout operation; that is a separate action.
- Main and bank results remain independent. New main results do not overwrite
  an image currently displaying the bank. Pending/failed bank requests retain
  the previous readout with an explicit status. A missing bank TEM/STEM product
  shows as unavailable, never as a main result under the bank label.
- Sample, scan and aberration controls still edit the **current instrument**.
  The STEM source selector applies only to **Images**; **Geometry** and
  **4D-STEM** keep their current-data workflows. Bank images are complete static
  frames, not new acquisitions. Main scan playback and its paused-frame state
  are kept separately. Detailed bank coordinates and model limitations are in
  the source-status tooltip.

The unified-viewer regression selection passed 110 tests on 2026-09-06, including
presentation-only source switches and retained main/paused/bank products. Three
production integration cases were excluded; no full high-accuracy imaging or
preset solve was run. UI inspection used synthetic stored arrays, not microscope
simulation results.

## Advanced-bank calculation boundaries

| Control | Work required |
| --- | --- |
| Lens excitation; upstream aperture diameter/X/Y | Precomputed optical nodes, using compatible source/specimen checkpoints |
| Downstream aperture diameter/X/Y | Re-evaluate ray masks; repropagate coherent waves |
| Inserted recording-detector Z, supported X/Y, width or annular inner diameter | Re-evaluate sequential interception; update wave/angle-resolved readout |

Only controls implemented by each component are listed. Retracted apertures and
detectors must first be inserted in the main settings and captured again. Detector
Z must remain downstream of the sample and within the retained column extent.
The page does not move housings or claim mechanical clearance for experimental
readout-plane displacements. TOML geometry is not rewritten by these controls.

## Physical models and limits

- Ray positions remain in metres internally; stop geometry and UI coordinates
  use millimetres. Propagation is along +Z. Retained trajectory interpolation
  has the existing history-grid accuracy; it is not a new exact integrator.
- Raw downstream coordinates are independent of detector/aperture survival.
  Replay starts with authoritative pre-sample losses and cached column-wall
  stops. Ordered physical stops are re-evaluated, preserving original source
  weights. Reopening an aperture never resurrects an upstream or wall loss.
  A zero-diameter aperture blocks even an exactly on-axis numerical ray.
- TEM stores separate pre- and post-Objective-pupil complex configurations.
  Aperture changes reapply the pupil and downstream coherent propagation;
  frozen-phonon intensities are averaged incoherently. The existing paraxial
  Objective-pupil and intermediate-plane wave approximations are retained.
  Legacy results lacking a pre-pupil checkpoint need one TEM calculation.
- Wave STEM automatically retains an in-memory diffraction-probability cube
  when requested by the captured scan settings. No duplicate file-output
  setting is needed. Allocation is checked against the remaining bank and
  application budgets before the cube is created. Incomplete cubes are rejected.
- BF/DF/HAADF recollection uses the existing angle-resolved first-order physical
  routing model, including ordered apertures/detectors. It is not arbitrary-plane
  coherent STEM imaging: intensity data cannot reconstruct discarded phase.
  Source-current and tracked inelastic-population factors are preserved, and
  bandwidth losses are not renormalised. An uncached added Rutherford tail is
  not silently omitted: changed-stop STEM output is unavailable in that case.
- New source/Objective changes that affect the incident beam or sample field
  still require the affected specimen work at their precomputed nodes. This
  release does not implement PRISM/S-matrix probe synthesis, intensity-image
  interpolation, extrapolation, or arbitrary coupled-geometry interpolation.
- EDS detector-array geometry is not an interactive range control in this
  release. Main EDS results are left untouched. Ray readout is not labelled as
  a specimen TEM/STEM contrast image or an EDS photon spectrum.

Reference background: [abTEM detector storage](https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/scan_and_detect.html)
distinguishes diffraction intensities from full complex waves;
[abTEM aperture transfer](https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html)
describes pupil filtering before image formation.

## Verification

`tests/test_interactive_calculation.py` covers finite ranges, exact-node lookup,
source weighting, shrink/reopen, detector-Z readout, first-detector precedence, upstream/wall losses,
zero opening, cancellation, stale inputs, RAM bounds, incomplete cubes, UI
required fields and retained-bank transaction semantics. Small CPU production
tests verify source-checkpoint reuse, TEM replay versus a fresh projection, and
one STEM specimen calculation shared by two P2 nodes and changed detector readout.
These are limited-grid software tests, not instrument calibration or a claim of
interactive frame rate for large production calculations.

`tests/test_optical_tuning.py` separately verifies source-support weights,
omission of expensive specimen/scan solvers, serial/NumPy RK4 equivalence and
fallback, cancelled requests, preserved high results/view limits, continuous
sliders, and a single final high-accuracy submission.

`tests/test_live_tuning_dock.py` checks View/plot toggle consistency, native dock
save/restore, one shared live plot, and retained controls/cache across hide/show.
These offscreen Qt checks do not verify native pointer dragging on Windows.
