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
in **Illuminating Image > Stored wave / reference** and **Scanning Image > Images**, respectively.
Each viewer independently selects **Current calculation** or **Advanced bank**.
**Illuminating Image > Camera / screen** separately displays captured physical
particle reception and explicitly labelled scan previews; see
[Receiver imaging](RECEIVER_IMAGING.md).
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
4. After tuning, select **Run high-accuracy once at current settings** in the dock. This
   flushes the final slider value and submits one current-state calculation,
   not a sweep. Existing TEM/STEM/EDS enablement still determines its products.

Preview normally traces 49 source samples with a maximum 1 mm integration
step; Medium normally uses a 193-ray budget at a maximum 0.25 mm step, including
zero-current support probes. Source-specific strata may require a larger bundle.
An explicitly configured Tip position × direction × energy quadrature retains
its complete population and weights in both tiers; it is never repartitioned
or replaced by support probes. These are
sampling/resolution tiers, independent of the Simulation menu's physical-model
tiers; they do not silently replace a selected field model with ideal optics.

The Live tuning particle path retains the selected source, column fields,
scan drive, physical clipping, specimen scattering and independently enabled
EDS transport. It updates particle detector signals for the current scan pixel;
it does not acquire a complete STEM raster or request legacy TEM/STEM wave
products. Medium shading joins sampled limits: it is not electron density or
guaranteed beam support. A small bundle may miss a narrow opening; quantitative
signals still require the final calculation.

All calculations use the toolbar's single CPU/GPU selection. Auto can use an
available GPU; the reported backend records actual execution, including fallbacks.
Ordinary model edits reject obsolete workers. During live slider motion, the
current ray frame finishes before calculating the latest accumulated settings;
intermediate frames never write their older values back to the live controls.
An explicit High accuracy request clears pending live work. The detached
Advanced-bank readout keeps its existing debounce and generation guards.
Cancellation checks bracket integration segments; an active native kernel is
not force-killed. Previous high-accuracy images/spectra remain stored and are
marked stale after edits, never erased because a low-count preview missed a stop.
Nonlinear/FEM field solves and the first JIT compilation can still be slow.

### Electron beam during Live tuning

First use **Calculate beam** in **Electron beam** to obtain a successful wave
result. Subsequent Live slider edits retain that image as previous data, cancel
obsolete wave work, and wait until edits have settled for 300 ms and the latest
particle preview has completed. They then submit one **Calculate beam** action
with the latest optics and observation Z. This also honours the selected saved
state/overlay mode and advanced classical-comparison option. There is no second
source or backend control, and no interpolation between complex wave fields.
The existing wave pipeline owns source admission, propagation and cache reuse.

A particle-only session does not automatically start its first wave calculation.
Unapplied Tip drafts are preserved and block automatic refresh. Explicit
**Calculate beam**, **Cancel**, other input changes and a failed particle preview
supersede a queued refresh. After a failure, use **Calculate beam** to retry;
there is no repeated automatic retry. The 300 ms interval coalesces edits, not
a guaranteed wave frame rate. Full wave propagation may still take much longer.

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

1. Configure the sample and installed/inserted components in their existing
   pages. Set High-accuracy rays and Step in
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
   The current toolbar compute backend is captured when **Build** is selected
   and stays fixed for every point in that build, even if the toolbar changes
   while the job is queued or running. Changing the backend does not require
   recapturing physical settings and does not rewrite previous result provenance.
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

Bank builds use the captured full-calculation workflow. They do not inherit the
currently selected page's calculation workflow or execute the **Electron beam**
development pipeline. Requests requiring new coherent TEM/STEM images remain
blocked by production source admission, including when the shared tip has
coherence parameters. Completed historical images can still be displayed.

### Shared image viewers

- **Current calculation** displays the main calculation result. **Advanced bank**
  displays the last completed bank readout at its selected optical node and
  declared readout values. A bank retains physical settings from capture time
  rather than following later edits to the main instrument. Its execution
  backend is selected separately at build start; completed products retain
  their original execution provenance.
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
| Downstream aperture diameter/X/Y | Re-evaluate ray masks; changed-stop TEM output is unavailable while coherent source admission is closed |
| Inserted recording-detector Z, supported X/Y, width or annular inner diameter | Re-evaluate sequential ray interception; recollect retained STEM angular intensities when supported |

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
- An unchanged completed historical TEM recording may be reused. Changed
  stops request TEM reprojection, which currently fails the production source
  gate and is reported as unavailable. A retained Objective checkpoint or
  configured coherent tip does not qualify a new coherent image. The separate
  **Electron beam** workflow does not supply an Advanced-bank wave product.
- Completed STEM products may contain an in-memory diffraction-probability
  cube for changed-stop recollection. New wave STEM production remains
  source-gated. Existing memory limits and incomplete-cube rejection apply;
  a bank never substitutes missing raw angular data with a rendered image.
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
required fields and retained-bank transaction semantics. The TEM admission test
explicitly imports the Si CIF fixture because the startup specimen is vacuum;
TEM/STEM admission tests prohibit transport when the source is unqualified.
The separate bounded particle-bank test covers source-checkpoint reuse.
`tests/test_bank_readout_bridge.py` covers the current toolbar backend at build
dispatch, fixed policy while queued, unchanged captured physical inputs and
preserved previous-bank provenance. These software tests do not qualify a
coherent source-to-image calculation or claim an interactive frame rate for
large production calculations.

`tests/test_optical_tuning.py` separately verifies source-support weights,
omission of expensive specimen/scan solvers, serial/NumPy RK4 equivalence and
fallback, cancelled requests, preserved high results/view limits, continuous
sliders, and a single final high-accuracy submission.

`tests/test_live_tuning_dock.py` checks View/plot toggle consistency, native dock
save/restore, one shared live plot, and retained controls/cache across hide/show.
These offscreen Qt checks do not verify native pointer dragging on Windows.
