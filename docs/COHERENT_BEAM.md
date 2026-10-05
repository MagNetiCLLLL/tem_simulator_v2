# Electron beam and continuous Z observation

The **Electron beam** page has its own **Calculate beam** button. It captures
the applied instrument Tip, electrodes, lenses, apertures, specimen
and detectors for the development wave pipeline. Intensity, electron arrivals,
phase and probability flow are observations of its executed state. Optional
**Compare classical rays** under the advanced settings adds a matched classical
calculation; it is off by default. This does not invent phase from particle
positions or flight times, or accept a configurable gun-exit or specimen-plane
source. Existing stored source settings remain unchanged until explicitly applied.

The main toolbar's **Run high-accuracy once** button runs classical particles
and prepares the incident beam required by Scanning Image and the
specimen/detector pages. It is also available as **Simulation → Calculate
classical rays**. Run that calculation before using those pages' calculation
buttons; **Update rays** provides a fast particle preview.

The electron-wave workflow assumes ideal vacuum: it does not simulate residual-gas
scattering or pressure attenuation. Electric and magnetic fields, specimen
interactions, apertures, column walls and physical detector absorption remain
active. Ideal vacuum does not mean ideal lenses or removal of aberrations.

**Source selection:** stored classical defaults remain unchanged. A projected
emission width, metal apex radius and back-projected virtual-source width are
different quantities. The historical 5 nm / 0.3 eV inputs are editable example
values, not required constants or measured tip performance. For a planar Gaussian
boundary, **Driven emission · non-paraxial near tip** executes a driven near-field
segment before continuing through the gun and column. **Forward emission ·
paraxial** retains its historical forward/paraxial validity checks and can reject
these small, low-energy inputs. Saved inputs retain their declared model;
selecting and applying another model is an explicit physical change. The curved metal-tip
model uses the separate non-paraxial near-field and two-way round-gun solver;
its full-chain numerical convergence is under development.

## Tip parameters

There is one Tip. Its geometry, current, emission positions, energy and angular
distributions, phase and coherence parameters belong to the instrument Tip.
Ray and wave calculations read that applied record; neither owns a separate
source. **Tip parameters…** opens its editor, including phase and coherence
options. Inline controls on Electron beam edit these same Tip parameters.
**Apply tip parameters** validates and publishes pending edits for every
calculation. Merely opening a page or calculating never applies a hidden source.

Gaussian-Schell emission and constant-phase surface emission are physical
distribution models for the Tip, not different Tips or switches that merely
start wave calculation. Changing the model can change the emission law and
invalidate calculated results. Numerical budgets remain calculation settings.

For the 5 nm / 0.3 eV / 0.3 eV example, enable **Gaussian-Schell emission**,
choose **Driven emission · non-paraxial near tip**, press **Apply tip parameters**,
and then **Calculate beam**. Application validates and publishes the
Tip; calculation still executes and checks the actual fields, apertures and
requested observation path. No emission width or energy is enlarged to pass
the old forward-source check.

This driven model prescribes a complex Gaussian injection at a planar cathode.
Its first 4 micrometres retain longitudinal reflection and evanescent components
in the accelerating field; the executed state then continues through extraction,
acceleration and the installed gun optics. It is not a metal-tip tunnelling model.
Particles read the same positions, energy law and phase-gradient directions;
wave diffraction is not reinterpreted as random classical launch momentum.
Near-field spectra, normal derivatives, phase references and transmission remain
part of the stored executed state. Full microscope qualification requires more
than passing this source check or a single successful calculation.

### Compute device and memory

The main toolbar's **Compute** choice is captured with the applied Tip and
optics for coherent column and material calculations. This is the only editable
CPU/GPU selection; STEM and coherent pages consume the same captured preference.
**Auto** prefers an available supported GPU even for small jobs; **CUDA GPU**
and **Prefer GPU** request it and report an allowed
CPU fallback. **Require GPU** stops if the requested GPU execution cannot be
provided. Gun propagation remains on CPU.

**Advanced calculation options → GPU working limit** defaults to **24 GiB**.
It is separate from **Working limit per stage** and the RAM/disk cache limits;
raising it does not create additional device memory. Device admission also
checks available memory. Editing a numerical budget requires a new explicit
calculation and leaves the previous complete image visible.

The completed result readout reports the actual **Column** and **Material**
devices, precision and any fallback reason. It reads executed records, not the
toolbar selection; older records without this evidence say that the execution
device was not recorded. Coherent GPU operators retain **complex128 / float64**.
The CPU timings below remain CPU evidence, not GPU speed measurements.

After restarting the application, choose **Compute → CUDA GPU** (or **Require
GPU** when an unavailable GPU must be reported as an error), keep **GPU working
limit = 24 GiB** on the tested 32 GB device, and press **Calculate beam**.
The optional classical comparison uses that same captured device selection;
it does not define different Tip parameters. An already running task retains its original
settings. Inspect the completed **Column** and **Material** backend labels to
confirm which stages actually ran on GPU.

The complete small-source Si example below has now also run through the public
pipeline on an RTX 5090 in **343.24 s**, with unchanged physical inputs,
discretisation and complex128 precision, resolving the same eight surrounding
spots at **Z = 1605.1934995 mm**. The earlier CPU run took 1473.33 s on a different
source revision. A bounded comparison on the same current revision measured
49.41 s CPU versus 11.27 s GPU for one actual column step, with relative complex
error below 9.3 × 10⁻¹⁶. These are measured local results, not a guarantee for
other grids or mode counts. See [the validation record](COHERENT_DIFFRACTION_VALIDATION.md)
for inputs, timing scope, numerical comparison and untested boundaries.

The 2026-10-04 CPU execution check used the real Coherent beam page's capture
and background worker with the startup optics, 5 nm emission FWHM, 0.3 eV mean
energy, 0.3 eV RMS-equivalent FWHM, and all nine energy modes. It completed at
Z = 450 mm in 85.54 s on one numerical thread without a propagation-cache hit;
the transmitted fraction was 0.99822827349. The source and captured optics were
unchanged. This was an offscreen Qt calculation through extraction, acceleration
and gun optics, not native desktop visual acceptance or a specimen diffraction
check. Main-window publication/dispatch and historical forward-model rejection
were checked separately. Halving the near-tip step and energy-step limit changed
the worst normalized mode shape by 0.221% and total transmission by 0.0823%; the
small reflected fraction is not established as quantitatively converged.

## Small-source Si result and GUI inputs — 4 October 2026

The complete CPU pipeline produced a central beam and eight surrounding Si
diffraction peaks at **Z = 1605.1934995 mm**, using the following small-source
settings. Source-to-specimen execution took 568.16 s and the specimen/remaining
column observation took 903.66 s; the complete local run including readout took
1473.33 s (about 24.6 minutes). The actual Coherent beam Qt page displayed that
published result with its ordinary **128 × 128 display bins** and **Log intensity**;
both full view and zoom visibly resolve the eight spots. This was offscreen Qt
result handling, not native-desktop click-through verification.

These settings are explicit edits to a fresh startup; they have not been
written into the optical preset or made automatic source overrides. The exact
input, request and result records remain in the ignored local directory
`outputs/realistic-tip-optics-20261004/si-public-final/`. See
[the validation record](COHERENT_DIFFRACTION_VALIDATION.md) for evidence and limits.

1. Keep the startup cold FEG, three-condenser column, **Nanoprobe** and
   **Diffraction (objective back focal plane)** preset, with 300 kV acceleration.
   Set the main toolbar **Step (mm)** to **0.2**. In the instrument **Optical**
   tree / **Live lens control**, enter the four changed excitations below.
   Leave all other lens settings and field directions at their startup values;
   in particular, Objective remains **68.98010%**. Do not subsequently apply
   the stored lens preset, which would replace the entered excitations.

   | Lens | Excitation (%) |
   | --- | ---: |
   | C1 | 16.36398 |
   | C2 | 12.96266 |
   | C3 | 37.15894 |
   | Mini Condenser | 63.09399 |

2. Select **C2 Aperture** in the instrument controls, keep it enabled and
   inserted, and set **Opening diameter to 200 µm**. This explicit optical change reduces severe hard-edge
   truncation of the candidate illumination. Other apertures, column walls and
   physical components remain active according to their captured settings.
3. In **Sample**, choose **Imported CIF** and open the user's `Si.cif`:
   Fm-3m, cell parameter **3.82166108 Å**, four Si sites per conventional cell.
   Enable **Inserted**, select **Circular disk**, diameter **10 nm**, thickness
   **5 nm**, and keep the centre and scan origin at zero. Set zone axis
   **[0, 0, 1]**, in-plane axis **[1, 0, 0]**, and press **Align CIF zone axis**.
   Under **Wave imaging settings**, enable **Multislice propagation** and
   **Lobato IAM potential**; enter **Grid 96**, **Field of view 40 Å** explicitly,
   **Target slice thickness 10 Å**, and **Additional defocus 0 nm**. Keep the
   bandwidth fraction at its exact default **2/3**, frozen phonons off and the
   approximate Rutherford tail off. This candidate retains the imported-CIF
   default unspecified/disabled inelastic and absorption rates; those stored
   zero rates mean inactive channels, not a physical zero mean free path.
   Grid 96 is an explicit qualitative setting. The earlier Grid 192 run
   successfully reached the specimen, but preserving its complete 137.758 nm
   incident field required a 13228 × 13228 potential grid, with an estimated
   41.72 GiB working allocation above the declared 24 GiB limit. Grid 96 uses
   the explicit 40 Å reference field to retain that same incident field on
   a minimum 3308 × 3308 wave grid and 6616 × 6616 potential grid. The material
   planner rounds these upward to FFT-efficient 3360 × 3360 and 6720 × 6720
   grids on the same physical field. Sampling becomes slightly finer; nothing
   is cropped. All retained potential slices, copies, incoming waves and
   single-mode propagation scratch are budgeted together. Even the earlier
   2 Å / 25-slice setting has an allocation-free estimate of 21.20 GiB within
   the explicit 24 GiB limit; this candidate uses 10 Å / five slices to reduce
   execution cost. Slice-thickness and intensity convergence remain unverified.
   Potential preparation uses this declared coherent budget; ordinary TEM/STEM
   calls retain their previous default guards. It supports the
   first (200)/(220) reciprocal orders for this CIF; it does not establish
   converged diffraction intensities or retain every higher reciprocal order.
   The full incident field is retained; it is not cropped to fit the budget.
   These Sample controls publish their values as they are edited. The disabled
   historical **TEM image / diffraction** checkbox is not required by Electron
   beam, and **Calculate sample** is not a prerequisite for this wave request.
4. Open **Electron beam**, enable **Gaussian-Schell emission**, select
   **Driven emission · non-paraxial near tip**, and set **Physical emission
   FWHM 5 nm**, **Tip mean kinetic energy 0.3 eV**, **Energy spread
   (RMS-equivalent FWHM) 0.3 eV**, and **Minimum kinetic energy 0.01 eV**.
   Keep emission-centre X/Y, all three wavefront curvatures, both tilts and
   incoherent angular RMS at **zero**; current is **10000 nA**. Press
   **Apply tip parameters** before calculating. The metal apex geometry is
   unchanged; the 5 nm value describes this planar emission intensity, not
   a replacement metal radius or a measured FEG performance claim.
5. Open **Advanced calculation options** and enter the following. **Energy
   samples 3** reduces quadrature cost while retaining the physical 0.3 eV
   spread; it does not make the source monoenergetic. The normal page default
   for nonzero spread is **9**, and energy-quadrature convergence remains a
   separate check.

   | Control | Candidate entry |
   | --- | ---: |
   | Initial grid pixels | 128 |
   | Maximum refined grid pixels | 32768 |
   | Energy samples | 3 |
   | Column step | 0.2 mm |
   | Gun field step | 0.05 mm |
   | Gun bore sampling step | 1 mm |
   | Gun energy change per step | 5% |
   | Material trajectories per mode | 32 |
   | Steps per segment | 128 |
   | Working limit per stage | 24 GiB |
   | RAM cache limit | 8 GiB |
   | Disk cache limit | 192 GiB |

6. Disable **Follow Ray Z**, enter **Z = 1605.19350 mm**, select **Log
   intensity**, then press **Calculate beam**. No prior particle,
   Sample or STEM calculation is required for this independent wave path.
   For the eventual matched comparison, enable **Compare classical rays** in
   the advanced settings before **Calculate beam**, with the same applied Tip
   and optical settings; separate earlier particle
   results are not evidence of matching inputs.

The target Z is the Ray Diagram objective diffraction plane for these actual
settings. An independent axial-field calculation predicts 1605.1934964 mm;
the declared 0.2 mm coherent column discretisation predicts 1605.1935413 mm.
This is numerical agreement within the model, not nanometre instrument
calibration. The user's FCC [001] cell predicts the first reciprocal-lattice
spot centres near ±(1.122, 39.872) µm and ±(−39.872, 1.122) µm relative to the
beam centre. The eight window maxima in the completed wave readout are
0.497–0.526 µm from the independent 200/220 predictions on 0.25 µm readout bins.
These are window-maximum comparisons, not fitted peak centroids or converged
intensity predictions. An unchanged incident-wave vacuum control lacks these
distinct surrounding spots. Final retained probability is 0.98981707 per tip
electron, versus 0.99818353 at specimen entrance; numerical bandwidth loss is
kept separate from physical absorption.

The completed CPU request sets the **column** working limit to 24 GiB, while its
gun limit remains 8 GiB. The GUI's single working-limit control applies 24 GiB
to every stage, so the recorded requests are not byte-identical. This is a
resource-budget difference; the native GUI calculation-button path remains
unverified for this complete run. The offscreen page readout reused the exact
completed result and scheduled no new propagation. The historical 28.39 µm / 30 eV
example below uses a different declared source and is not this candidate.

## Multiple emission states in common optics

The Electron beam controls include a session-local **electron state list**.
Every entry captures an already applied physical-tip emission state. It is an
input record, not a downstream source or an executed checkpoint. The normal
tip-to-observation pipeline still executes extraction, acceleration, installed
fields, apertures, specimen interactions and physical absorption.

1. Edit the Tip parameters and press **Apply tip parameters**, then **Add
   current state**. Repeat with different supported energy, position, tilt or
   other emission inputs. Unapplied drafts cannot be captured.
2. Rename rows and set relative weights. Choose **Selected state** to inspect
   one row, or **Overlay checked states** to show their combined intensity.
3. Press the same **Calculate beam** button used for the current Tip. All selected states execute in the current
   common optics. Identical states share one calculation. Move observation Z
   afterwards to query the same sessions; completed planes are reused.
4. To edit a saved state, select it, press **Apply selected state**, edit and
   **Apply tip parameters**, then **Replace selected**. Loading explicitly changes
   the instrument source for both particles and waves; selecting a row does not.

Changing names, weights or visibility only changes the readout. Missing state
results require Calculate; they are never silently omitted from an overlay.
Lens or specimen edits require Calculate again, using the saved emission inputs
and the new common optics. A change of physical tip geometry, current or source
family requires recapturing states instead of silently reinterpreting them.
The list is held for this application session; it is not added to result-file
export. A current-Tip calculation and its optional classical comparison remain
available under **Current tip parameters**. The rows store initial emission states of this Tip;
they do not create separate physical Tips.

For independent electron states the displayed probability density is
`sum(w_i * I_i) / sum(w_i)`. Weights represent relative population fractions,
not additional physical current, exposure or particle drawing counts. State
images are conservatively rebinned into the same physical X/Y grid and must
refer to exactly the same Z. Each member retains its complex modes and phase
references; the mixture has no single phase and does not interfere between
independent states. Selecting a single row views it at unit weight.

Calculations run sequentially through the shared resource coordinator, with one
latest observation target. The display cache budget covers the whole list;
retained mode/session memory is also included in shared admission. If one state
fails, the previous complete image remains, with the failure identified. An
overlay does not claim to match the single source currently shown in Ray
Diagram. Apply a row, select **Current tip parameters**, enable **Compare classical
rays**, then press **Calculate beam** for that comparison.
This feature does not extend non-paraxial solver support or establish full-chain
physical convergence; unsupported source states and planes remain explicit.

## Curved metal tip

Open **Tip parameters…**, select **Curved metal tip with electrode
fields**, and select **Constant-phase surface emission**
under **Show phase and coherence parameters**. This selection uses the existing Tip
dimensions; edit apex radius, cone angle and emitting-cap angle through
**Physical Layout**. Patch diameter, area and depth are derived from these
dimensions, never independent downstream-source inputs.

There is one total kinetic-energy distribution: a positive gamma law specified
by mean and RMS, or a monoenergetic law when RMS is zero. Both particle sampling
and wave energy quadrature consume that same record. Selecting the shared
boundary from a classical surface model retains its energy mean and RMS, but
explicitly selects the supported smooth cap profile and local-normal emission.
It does not claim to preserve a previous independent angular distribution.
The amplitude tapers as cosine squared across the cap, so flux tapers as cosine
to the fourth power; sampling includes the actual spherical surface-area measure.
The current is prescribed incident/injected reference flux. Reflection and
aperture loss may reduce the executed escaping wave current.

The particle representation is the geometric-optics local-normal ray limit of
this prescribed boundary. It does not contain the diffraction-induced angular
spread of the complex wave. This is an explicit approximation for comparing
the methods, not a reconstruction of quantum phase from particle trajectories.
Only constant surface phase is currently supported; nonzero phase gradients are
rejected until their tangential current and injection boundary are implemented.
This model does not yet predict metal tunnelling, brightness or space charge.

The joined near-field numerical default now covers the complete retained radial
basis (Q2 elements, 385 by 97 nodes, numerical radius four times the emitting-cap
radius); this does not enlarge the physical source. It resolves an interface
coverage defect, not full gun convergence. A bounded diagnostic with unchanged
100 nm apex radius, 10 degree cap and 0.3 eV monoenergetic emission still changed
the gun-exit current fraction from 0.18535 to 0.09734 when increasing radial
modes from 32 to 48. Those outputs are not accepted as converged physical
predictions. Refinement of long gun propagation and aperture projections remains
necessary; side-wave validity checks and unsupported-field checks stay active.

Historical surface reservoirs remain readable with their original energies and
phase. Replacing one requires the explicit source-editor action; merely opening
the record cannot convert it into a new particle source.

## Workflow

1. Configure the instrument and import a CIF, orientation and thickness in Sample.
2. Open Electron beam. Its source controls read the active instrument tip,
   including the current emission width, mean energy and energy spread. Select
   its emission and coherence model and edit these controls as a draft. Press
   **Apply tip parameters** to validate and publish the source for both particles
   and waves. Applying a source marks previous results stale but starts no
   calculation. The 28.39 micrometre / 30 eV monochromatic demonstration below
   is an explicit historical example, never a startup or fallback source.
3. Review observation Z and press **Calculate beam** on the Electron beam
   page. It uses the selected current Tip or saved-state display mode.
   For a matched comparison of the current Tip, enable **Compare classical rays**
   in the advanced settings first. This freezes the applied source and optics
   once, executes classical transport and active specimen interactions, then
   starts the wave calculation from those same inputs. It uses the toolbar's
   particle count/step and this page's separate wave numerical budgets.
   **Update rays** and **Run high-accuracy once** (also available as
   **Simulation → Calculate classical rays**) remain explicit classical
   operations and do not establish a matched pair. Unapplied source
   edits are rejected. Failed source-domain checks do not change tip dimensions,
   energy or energy spread.
4. After Calculate, return to **Ray Diagram**, enable **Beam analysis**, and
   select **Beam observation** in the right panel. Keep **Follow Ray Z** checked:
   dragging the cyan Z cursor updates the beam observation beside the rays.
   **Classical rays** restores the classical transverse view. **Controls…** opens
   the full Electron beam settings; both screens share one calculation session.
   Alternatively, drag the **Z range** slider or enter Z on the full page.
   **Start** and **End** beside the slider set its
   browsing range only; narrow them to inspect a focus in small increments.
   Exact display-cache hits are immediate. A running query is allowed to finish
   while new positions replace a single pending target, so continuous motion
   cannot indefinitely cancel/restart propagation. Its completed image is labelled
   with the actual displayed Z while the latest target is calculated automatically.
   An already displayed exact target takes priority over an older in-flight query.
   Cancel, changed inputs and closed sessions reject their late outputs.
   The previous complete image stays visible while a new plane is calculated.
   Previously visited planes are immediate while retained in the bounded display
   cache; a new Z still needs propagation from a verified upstream checkpoint.
   Large grids, many modes or crossing the specimen can therefore take longer.
   No interpolation of intensities is presented as an exact physical plane.
   A separate status line distinguishes **Requested Z**, **Calculating Z** and
   **Displayed Z**. The readout reports the actual complex-wave grid as well as
   the smaller display-bin count; a 128 x 128 image can represent a much larger
   executed wave.
   The first completed image fits its wave area once. Later Z changes preserve
   each view's physical XY zoom and pan, even when the wave domain changes.
   **Fit full** fits the currently displayed wave area on demand without
   starting propagation. The full page and Ray sidebar retain separate views.
5. After changing source, hardware, specimen or numerical settings, press
   **Calculate beam** again to establish a new captured session.

### Observations of the same calculated beam

The observation selector changes the readout without running propagation again:

- **Intensity** displays the weighted probability density at the completed Z.
  Independent modes and saved states add intensities, not complex amplitudes.
- **Electron arrivals** samples a virtual exposure from the current display-bin
  probabilities. **Emitted electrons** and **Seed** affect only those events,
  not the source, propagated wave, classical ray count or physical detector.
  Electrons not reaching the screen remain absent. Dots are independent screen
  events; dense exposures show all sampled counts per bin. This view does not
  simulate detector efficiency, point-spread response or electronic noise.
- **Phase** selects one executed Tip state and one coherent mode. The overlay
  has no aggregate phase; different independent modes do not acquire a common
  phase reference merely by being shown together.
- **Probability flow** displays local transverse current directions at that
  plane, including the applicable magnetic-field contribution. These arrows
  are not complete paths or measured trajectories of individual electrons.
  **Beam current**, **Angular distribution** and interaction diagnostics also
  read retained modes; angular distributions retain their stated gauge convention.

Observation changes retain the captured source and actual displayed Z. Missing
complex modes remain unavailable instead of being inferred from an intensity
image or from classical ray positions.

## Optional classical comparison

Each pair has its own request token and captured input identity. The controller
checks the actual particle result's source/optics and requested particle
numerics, then checks the wave capture and returned result identities. Input
changes, cancellation, an independent replacement result or a late result from
an older request cannot become a current matched comparison. Paired execution
requires ideal vacuum for both methods; enabling residual-gas transport causes
an explicit refusal rather than silently switching it off.

With the classical comparison enabled, the Electron beam page shows a same-Z table of current, emitted-current
fraction, mean energy, energy RMS, laboratory X/Y centroid, X/Y RMS widths and
radial RMS. Statistics use all retained weighted particles and the propagated
complex mode grids, not the plotted ray subset or the display image bins. Each
completed comparison labels its actual Z; a newer cursor position does not
relabel an older result. After a material sample, a bare optical-reference ray
bundle is not accepted as the specimen-exit comparison.

The two energy columns have different definitions: particle values are executed
local kinetic energies, interpolated between retained trajectory planes; wave
values are the forward modes' reference kinetic energies. The latter are not
a local kinetic-energy map inside a transverse electric field. Missing retained
particle energies remain unavailable. Interference and the methods' declared
approximations can produce different intensities and widths even with matching
inputs. Phase remains attached to individual coherent modes; no particle phase
or aggregate phase of an incoherent mixture is invented.

Pairing is sequential execution with the same physical inputs, not one shared
particle/wave numerical solution or interchangeable trajectory/complex-field
cache. The methods use the shared field providers/store and retain their own
dependency-checked executed checkpoints. The pairing controller shares the
existing artifact store and resource coordinator, and does not add a second
particle-history RAM cache. Within the particle path, specimen continuation
reuses the executed incident checkpoint. Within the wave path, moving Z reuses
valid complex-wave checkpoints and exact completed readouts.

Continuous quadratic column steps can share one complex-field transform when
every intermediate sampled grid satisfies the angular-spectrum bounds and fits
strictly inside its physical bore. All original field integration nodes, affine
forces, scalar actions and continuous phase lifts remain in the composition.
An aperture, discrete impulse, nonlinear multipole, possible wall interception,
unresolved phase lift or sampling-chart boundary stops this optimization and
retains the stepwise operators. Material and recording boundaries remain owned
by the stage router. The optimization neither reduces the wave grid nor replaces
an executed field with a fitted source. It changes the numerical carrier
schedule; comparisons must use complex fields on common physical coordinates,
including their absolute phase, rather than comparing raw array indices.

If Calculate reports **Tip source check rejected; propagation did not start**,
the calculation has stopped before launching a wave worker. **Source check
details** contains a read-only, copyable report of that attempt: requested tip
inputs, the actual source-domain bounds and observation Z. Editing inputs marks
the report as previous; it does not silently replace its values or a retained
completed image. A successful source check still does not qualify propagation.

The classical default of 5 nm emission FWHM, 0.3 eV mean kinetic energy and
0.3 eV RMS-equivalent energy FWHM is not an executable default for the present paraxial Gaussian
wave route. Its 0.01 eV lower energy boundary gives a conservative transverse
momentum/total momentum support bound of about 2.78989 at the declared tail
probability of 1e-8. The current 1% generator-error budget requires a bound no
larger than about 0.198997. These are probability-region bounds, not measured
electron angles or a claim that every emitted electron travels backwards.
Reducing High-accuracy rays, selecting a different Z or merely running classical
transport first cannot change this check. Removing energy spread alone leaves
the default 5 nm, 0.3 eV source outside that paraxial budget. Review and explicitly
edit the tip boundary, use the documented Si input design below, or use a future
wider-domain gun model; no low-energy modes are discarded and no parameters are
automatically widened or accelerated to bypass the check.

The nonabsorbing virtual screen displays the weighted real-space intensity
density of the modes, in probability per tip-reference electron per square
micrometre. Display binning preserves the retained original complex grids; this
is not calibrated detector counts. Inserted physical detectors still absorb the
beam. Observation at the exact detector Z denotes its incident state.

## Physical information required from the user

| Information | Origin and units | Purpose |
|---|---|---|
| Tip geometry, emitting region and current | Existing source settings; nm, angles, nA | Physical emission boundary and intensity scale, not a downstream waist |
| Tip kinetic energy and energy spread | eV; specify FWHM or RMS | Wavelength and energy mixture; accelerated energy follows electrode potentials |
| Spatial coherence model | Explicit tip mutual intensity or coherent surface boundary | Defines which amplitudes interfere |
| Mean tilt, offset and relative wavefront curvature | Existing Gaussian-Schell tip fields | Spatial phase at the tip, not an exit source |
| Electrode potentials, lens/deflector/stigmator settings, apertures and positions | Existing instrument | Phase evolution, focusing, interception and conjugate planes |
| CIF, orientation, occupancy, thickness and thermal displacement settings | Imported structure and Sample | Crystal scattering and diffraction through multislice, not predefined spots |
| Observation Z | mm | Intensity at the actual propagated plane |

For a **Gaussian-Schell tip**, the active emitter is the only source definition.
Particle diagnostics sample its positive Gaussian Wigner distribution, including
position/momentum correlations; wave modes represent the same mutual intensity
and positive energy law. Source edits are applied transactionally to the active
tip, then each calculation captures that instrument. The historical serialized
field `virtual_source_fwhm_nm` denotes prescribed intensity FWHM at emission,
not a measured virtual source or metal apex diameter. `energy_spread_fwhm_ev`
is an RMS-equivalent width (2.35482 times energy RMS), not a guarantee of the
non-Gaussian spectrum's measured FWHM. Mean kinetic energy is a separate input
before extraction and acceleration. Saved records retain their original values.
Incoherent angular RMS is given in mrad. Zero means one
spatially coherent mode; nonzero energy spread still needs multiple energy
samples. Diffraction's angular spread follows spatial width and wavelength, so
the independent classical angular distribution cannot also be imposed as the
same constraint. Position and angle probabilities do not uniquely determine a
wavefunction or coherence.

The Apply operation rejects an unsupported Gaussian source before
changing the current instrument. This includes the current 5 nm, low-energy
startup source: unifying ownership does not supply its missing wider-angle
near-tip wave physics. Constant-phase curved-surface emission provides a
geometric-optics particle representation of the same Tip parameters; full
wave convergence is still unqualified. Historical wave-only reservoirs remain
readable as historical records and require explicit replacement in Tip parameters
before a matched particle/wave calculation. These limits do not permit replacing
the source, discarding energy tails or
bypassing extraction, acceleration, apertures or other installed components.

The intensity display can use linear or logarithmic colour limits. This does
not change complex fields, probabilities or the calculation session.

For a **physical surface tip**, mean kinetic energy and RMS width belong to its
emission parameters. The supported constant-phase model adds no separate energy
inputs. It supplies one axisymmetric spatial mode per energy; arbitrary spatial
partial coherence and metal tunnelling are not implemented. Normal/tangential
probabilities alone do not define phase. Curved geometry cannot silently become
a flat Gaussian source.

No individual-electron phases or meaningless shared absolute phase are required.
Each mode retains its complex field and phase reference. Incoherent modes and
energies contribute weighted intensities, not summed amplitudes or one invented
aggregate beam phase.

## Speed and numerical settings

- Cost follows grid size, spatial modes, energies and propagation operators,
  rather than the classical particle count. Source-domain and memory checks
  precede wave execution; numerical controls are grouped under Advanced.
- Material progress reports the current trajectory, slice and operation, including
  saved-slice reuse. A trajectory can contain many expensive slices; the previous
  trajectory count alone was not a measure of whether execution was advancing.
- For each source mode and frozen-phonon configuration, the first complete
  material history establishes whether the collision operator was exactly the
  identity at every executed slice energy. Only then are identical requested
  histories represented by that executed complex field with their summed weight.
  Any nonzero scattering/absorption probability, including rare events, keeps all
  independent histories. A sampled zero-loss outcome is not proof of identity.
  Slice checkpoints retain this proof for cancellation and continuation. Elastic
  potential, installed column fields, apertures, clocks and phase still execute.
- Actual executed gun, specimen and column states are reused with full complex
  state and dependency identities. Relevant physical or numerical changes
  invalidate them. Display operations never create a new source.
- Each live observation session captures and restores its private instrument
  once in the first worker, then reuses that prepared state and its immutable
  input identity. Subsequent queries and checkpoint commits still verify full
  implementation bytes, external file contents and the input inventory. This
  avoids repeatedly reconstructing the entire instrument for a small Z change;
  it does not replace extraction, acceleration or propagation with a new source.
  Source, hardware or numerical edits require a new Calculate session.
- Movement can resume from the nearest successfully committed upstream
  observation or intermediate column segment in the current process. The
  bounded index refers to existing checked disk checkpoints or the bounded RAM
  cache, without duplicating wave arrays. Detector absorption occurs exactly
  once. Backward movement uses an earlier valid executed stage, never a later
  wave propagated backwards through absorbing material.
- Segmented column optics and specimen multislice avoid an atomically resolved
  three-dimensional mesh over the whole metre-scale column. Continuous phase
  paths and phase carriers are retained across foci, not only endpoint matrices.
- Inside a column segment, independent wave modes can run concurrently, with
  at most eight workers and no more than the active CPU budget. BLAS and each
  worker's Numba kernels remain serial. Shared input arrays are reserved first;
  the remaining working-memory budget is divided between bounded mode slots,
  including finished outputs awaiting ordered storage. Checkpoints publish only
  after every mode succeeds. If a worker's memory partition is too small, the
  workers stop and join, the unfinished checkpoint is discarded, and the same
  segment retries serially with the remaining full budget. Pixel-limit errors,
  invalid physical operators and user cancellation are not retried this way.
  No mode, phase, aperture event or transmitted-current loss is omitted.
- Material modes remain sequential. The Hermitian Galerkin operator uses
  SciPy FFTs with at most four workers, further limited by the active CPU
  budget. Fourier origins, unitary normalization and the exponential error
  check remain unchanged. FFT-friendly grid rounding only adds samples on
  the same physical field, and the rounded grid must pass the memory budget.
- Galerkin material propagation retains the analytic quadratic phase carrier
  `U`. A scalar potential commutes with that carrier, so its Hermitian matrix
  acts on the complete envelope in the moving `U × Fourier` basis without
  expanding a large chirp onto the envelope grid. Its numerical **Bandwidth
  fraction** projects residual Fourier frequencies: the physical projector is
  `U P U†`. The carrier and resulting momentum remain intact, and removed
  numerical probability is recorded. This is a different finite-basis
  projection from the historical laboratory-frequency cutoff; it requires
  basis convergence and is not a physical detector or angular aperture.
  Sampled-phase material propagation retains its historical laboratory
  projection and carrier-expansion guard. Physical apertures, column fields
  and detector absorption keep their existing operators in both methods.
- The gun grid also limits fractional kinetic-energy change, including the
  rapid acceleration immediately after emission. **Gun energy change per step**
  is a numerical percentage, not a new source-energy setting. Reducing only the
  maximum spatial step can leave the first acceleration interval unresolved.
- Continuous Z uses at most one executing request and one latest pending position,
  with a 40 ms input throttle. This is a scheduling interval, not a guaranteed
  frame rate: a new range or a large mixed-mode wave still needs propagation.
  Previously completed frames remain visible with their true Z. Up to 64 exact-Z
  displays are retained within the selected RAM budget. Generated segmented wave states remain local and
  excluded from Git. Numerical jobs respect the global half-logical-CPU ceiling.
- Enabled physics is never dropped to meet a budget. Insufficient resources or
  sampling produce explicit errors. Grid, slice thickness, energy quadrature and
  mode truncation need separate convergence checks; defaults are not proof of
  convergence.

Primary references for the mechanisms:
[multislice propagation](https://abtem.github.io/doc/user_guide/walkthrough/multislice.html),
[partial coherence](https://abtem.readthedocs.io/en/latest/user_guide/tutorials/partial_coherence.html),
and [sampling and antialiasing](https://abtem.github.io/doc/user_guide/appendix/antialiasing.html).
These are not validation reports for this simulator's complete tip-to-detector chain.

## Current development boundaries

This increment connects the independent page, explicit tip settings, shared
electromagnetic transport, specimen multislice, exact post-gun Z observations
and the paired calculation interface for admitted source conditions.
**It does not qualify the default tip through the complete instrument to every
arbitrary Z.**

- Targets must be at or beyond the gun exit. Gun-interior observations are not
  yet exposed.
- Nonvacuum specimen entrance and exit are supported. Arbitrary interior cuts
  need a truncated multislice interface and are currently rejected.
- The default narrow, low-energy Gaussian tip can violate the paraxial source
  domain. Its energy, size or probability support is not automatically changed.
  The physical-surface route has a two-way gun solver, but boundary, radial-mode
  and column-handoff convergence remain to be established.
- The new planar/accelerating near-field work is an explicit experimental API,
  not a selectable GUI source or an automatic fallback for rejected inputs.
  `build_planar_tip_boundary` and `execute_accelerating_tip` retain complex
  boundary traces, normal derivatives and declared flux information within
  their bounded domains. Their outputs remain `NOT_A_GUN_CHECKPOINT` and
  `INCOMPLETE`: a phase-preserving checkpoint through the full installed gun
  and its coupling to the downstream column are not connected for this route.
  Evanescent near-field components have no positive classical ray
  representation. These APIs neither remove the default-source guard nor
  establish a completed particle/coherent physical chain.
- The Gaussian route uses the captured electrostatic field's quadratic
  transverse expansion, including residual acceleration after the gun exit,
  together with axial lens fields, normal/skew stigmators and finite magnetic
  dipoles. Complex phase follows the canonical path. Nonpolynomial imported
  field maps, unsupported time-energy blanking and the older surface/radial
  route retain their explicit admission boundaries. Opening the page is not
  evidence that any path has been executed or converged.
- Reaching an assembled energy filter's entrance remains rejected until its wave
  operators are integrated last.
- Production TEM/STEM source admission remains closed. This development page
  does not waive physical or numerical qualification requirements.

Next physics priorities: bounded execution and convergence for the physical tip
and gun; validation for the shared distributed electromagnetic fields; specimen
interior observation; standard CIF diffraction and coherence checks; energy
filter integration last.

## Explicit Si diffraction example

The 3 October 2026 development run uses the user's **Fm-3m monatomic FCC Si**
CIF, with cell parameter 3.82166108 Å. It is not silently replaced by diamond
Fd-3m Si. Use [001] along Z, [100] along X, a 10 nm specimen diameter and 5 nm
thickness, with the stored nanoprobe/diffraction optical settings. Physical
recording detectors are explicitly retracted for this nonabsorbing virtual-screen
example. The complete tip-origin calculation produced resolved periodic peaks at
**Z = 1605.1939574922505 mm**. This Z belongs to those held optics; changing lens
settings changes the appropriate observation plane.

| Physical tip boundary input | Example value |
| --- | ---: |
| Gaussian intensity FWHM | 28390.10000542304 nm |
| Initial kinetic energy | 30 eV |
| Energy FWHM and incoherent angular RMS | 0; one spatial/energy mode |
| Relative wavefront curvature Qxx / Qyy | 624.3690658100246 / 624.369065810008 m⁻¹ |

### Manual GUI entry from a fresh start

No working-point import, result loading, script or configuration-file edit is
required. Restart the application to load the new centre/XY-curvature controls.
All the archived geometry/configuration file hashes were checked against the
current checkout on 3 October 2026 and match. The default assembly and
nanoprobe/diffraction preset reproduce the archived lens and aperture settings;
this equality is specific to that checked configuration, not every future build.

1. In **Instrument setup and parameters**, use the cold field-emission gun,
   three-condenser column with probe corrector, no monochromator, no optional
   beam blanker, no image corrector and no energy filter. Keep **Nanoprobe** and
   **Diffraction (objective back focal plane)**. The extractor is 4 kV; the gun
   lens is 1.2 kV relative to the extractor; the accelerator is 300 kV. Keep the
   flat tip (mechanical curvature 0) and the shared analytic electrode/column
   fields. The prescribed tip current is 10000 nA. The source controls below
   now apply to the instrument Tip; they do not change electrode or lens
   settings. Retained particle results must be recalculated for comparison.
2. Check lens excitation percentages in the instrument **Optical** tree. These
   values already match a fresh start with that preset. If necessary enter them
   in **Live lens control**. Use the existing geometry; do not run transport
   matching or lens recalibration after entering the values.

| Lens | Excitation (%) | Field direction |
| --- | ---: | --- |
| C1 | 51.666666666666664 | +Z |
| C2 | 25.25772717253661 | +Z |
| C3 | 21.273622908333973 | +Z |
| ADL (Adapter Lens) | 86.0039434207 | +Z |
| TL22 | 60 | -Z |
| TL21 | 60 | +Z |
| TL12 | 60 | +Z |
| Mini Condenser | 41.5570797718 | -Z |
| Objective | 68.9801 | +Z |
| Diffraction | 14.577617197168571 | +Z |
| Intermediate | 0.035082657259397326 | +Z |
| Projector P1 | 32.1132572718703 | +Z |
| Projector P2 | 9.826874422815507 | +Z |

The default gun and column deflectors/stigmators have zero applied strengths;
scan/descanning raster drives are off. Retain the default probe-corrector
elements. C2 opening diameter is 100 micrometres, C3 is 4000 micrometres; both
are inserted. Objective and selected-area apertures are retracted. To match the
virtual-screen example, select each recording plane in the instrument tree and
uncheck **Inserted** (HAADF, DF, BF, fluorescent screen and pixelated camera).
Existing non-retractable apertures and column walls remain active.

3. In **Sample**, select **Imported CIF** and **Open CIF...** to load the user's
   Fm-3m FCC Si (cell parameter 3.82166108 Angstrom), not a different Si structure.
   Check **Inserted (interactions enabled)**; choose **Circular disk**, diameter
   10 nm, thickness 5 nm; sample centre and scan-origin X/Y are zero. Enter zone
   axis **0, 0, 1**, in-plane axis **1, 0, 0**, then **Align CIF zone axis**. Under
   **Wave imaging settings**, enable **Multislice propagation** and **Lobato IAM
   potential**; set **Grid = 256**, **Field of view = 50 Angstrom**, **Target slice
   thickness = 1 Angstrom**, **Additional defocus = 0 nm**. Retain the exact
   default bandwidth fraction 2/3; the display rounds it to 0.66667, so there is
   no need to retype that rounded value. Disable frozen phonons and the
   approximate Rutherford tail. Retain the example's default unspecified
   plasmon/ionisation MFPs and disabled other/absorption channels: all their
   stored MFP controls are zero, which denotes an inactive channel here, not a
   physical zero mean free path. Material IMFP + Poisson event transport remains
   selected; the explicitly inactive rates do not remove elastic multislice.
4. In **Electron beam → Tip parameters**, select **Gaussian-Schell emission**
   and enter all source values below. They prescribe the emission boundary at
   the physical tip, before extraction and acceleration;
   they never define a downstream source. Decimal entries are provided because
   the standard spin boxes do not require scientific-notation text.

| Tip parameter | Manual entry |
| --- | ---: |
| Tip emission FWHM | 28390.10000542304 nm |
| Tip mean kinetic energy | 30 eV |
| Minimum kinetic energy | 0.01 eV |
| Energy spread FWHM | 0 eV |
| Tip emission centre X | -0.066960908718241 nm |
| Tip emission centre Y | -0.072669533872498 nm |
| Wavefront curvature X | 624.3690658100246 m^-1 |
| Wavefront curvature XY | -0.000000000009349 m^-1 |
| Wavefront curvature Y | 624.369065810008 m^-1 |
| Wavefront tilt X | -0.000041806037718 mrad |
| Wavefront tilt Y | -0.000045370130384 mrad |
| Incoherent angular RMS | 0 mrad |

5. Press **Apply tip parameters**, then disable **Follow Ray Z** and enter
   **Z = 1605.19395749225 mm**. In **Advanced
   numerical budgets**, set the values below. Select **Log intensity**, then
   **Calculate beam**. No preceding high-accuracy particle run or
   Calculate Sample/Calculate STEM is needed for this independent wave path.
   The three-panel comparison figure requires separate fine, coarse and vacuum
   runs; a single GUI run shows the current plane, without the independent cyan
   reciprocal-lattice prediction overlay. GUI decimals are not a promise of
   bitwise-identical re-execution.

| Advanced control | Fine Si example |
| --- | ---: |
| Initial grid pixels | 128 |
| Maximum refined grid pixels | 4096 |
| Energy samples | 1 |
| Material trajectories per mode | 1 |
| Column step | 1 mm |
| Gun field step | 0.2 mm |
| Gun bore sampling step | 10 mm |
| Gun energy change per step | 0.025% |
| Steps per segment | 128 |
| Working limit per stage | 8 GiB |
| RAM cache limit | 1 GiB |
| Disk cache limit | 20 GiB |

The restored Sample settings include requested grid 256, field of view 50 Å
and target slice thickness 1 Å. Multislice and the atomistic potential remain
enabled in the independent pipeline; the older TEM/STEM production readout
checkboxes are not the activation controls for this page. Material rates in this
example are explicitly zero. Do not interpret one trajectory as adequate for a
general specimen with nonzero rates. The default remains 32, now exposed rather
than silently overridden.

The bounded settings use source grid 128, one energy sample, maximum gun step
0.2 mm, maximum gun energy change **0.025%**, column step 1 mm, automatic
refinement capped at 4096 cells per axis in the recorded request, 8 GiB working
budget and one material trajectory. The refined-grid cap, trajectory count and
stage budgets are explicit Advanced controls; request metadata is not
automatically applied. That trajectory is exact for this example's explicitly zero
inelastic rates; nonzero rates require separate trajectory convergence.
The material comparison holds those upstream inputs fixed and jointly changes
the requested sample grid 192→256 and slice thickness 2→1 Å. Automatic refinement
and potential quadrature can create larger executed grids. This comparison does
not establish convergence of every source, gun, field or column discretisation.

The wavefront curvature is a spatial phase correlation at the tip, distinct
from mechanical tip curvature. The example uses the simulator's idealised
planar cathode field with a Gaussian coherent emission boundary. It does not
predict a nanometre metal tip's tunnelling, space charge, brightness or current,
and does not calibrate commercial instrument performance. Acceleration follows
the captured electrode potentials; there is no configurable accelerated source.

The 28.39 micrometre FWHM came from a quadratic transport design with a target
1.5 nm specimen waist, with the existing gun/column controls held fixed, followed
by the actual complete wave calculation. It is a width of prescribed intensity
at a planar cathode, not a metal apex radius, emission-patch size derived from
nanometre tip geometry, or a measured cold-field-emission virtual source. This
large width and 30 eV launch energy are idealised demonstration inputs. The
result establishes this declared idealised source/column/material case; it does
not establish coherent emission and near-field transport from a realistic metal
FEG. Making a physical nanometre tip coherent requires the corresponding source
and near-field model, rather than shrinking this input or bypassing its guards.

The final screen retains per-mode complex fields and their phase references.
Material bandwidth loss is reported separately from physical absorption;
intensity plots are never renormalised to hide that loss. Sampled material guards
operate on envelope spectrum plus analytical carrier, so a genuine wave node is
not mistaken for aliasing. Galerkin material retains its carrier in the declared
moving basis described above. Alternative resolved propagation charts are attempted
within the same budget; cancellation and genuine sampling failures remain errors.
An optional covariance-based phase-carrier change may require a much larger
grid than the physical propagation itself. If that representation change
exceeds the declared grid/memory budget, the original complete complex field
and carriers are retained and the physical propagator checks its own sampling.
This does not bypass an unresolved physical operator or raise the memory cap.

On failure, both coherent views identify the failed task's Z and the retained
image's Z. A retained image upstream of the specimen cannot show specimen
diffraction; it is not a result for the failed downstream request. For the Si
example above, use the specified downstream Z and **Log intensity** to inspect
weak diffraction peaks. That plane depends on the lens settings, not on a
universal fixed diffraction position.
Offscreen screenshots and CPU regressions are not native desktop or GPU evidence.
