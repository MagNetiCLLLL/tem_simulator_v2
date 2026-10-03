# Coherent beam and continuous Z observation

The user explicitly resumed coherent development on 2026-10-02. Classical
particles remain the default workflow. The separate **Coherent beam** page uses
the same captured instrument's tip, electrodes, lenses, apertures, specimen and
detectors through the development wave pipeline. It does not invent phase from
particle positions or flight times, or accept a configurable gun-exit or
specimen-plane source.

This workflow assumes ideal vacuum: it does not simulate residual-gas
scattering or pressure attenuation. Electric and magnetic fields, specimen
interactions, apertures, column walls and physical detector absorption remain
active. Ideal vacuum does not mean ideal lenses or removal of aberrations.

## Workflow

1. Configure the instrument and import a CIF, orientation and thickness in Sample.
2. Open Coherent beam and explicitly select **Use a coherent boundary at the
   physical tip**. Wave settings affect this page's detached captured instrument;
   they do not rewrite the active particle instrument.
   For a flat tip without saved coherence, the source controls now start with
   the idealised diffraction example: 28390.10000542304 nm intensity FWHM,
   30 eV launch energy, zero energy spread/incoherent angular RMS and its exact
   designed centre, tilt and curvature values listed below. One energy sample
   avoids repeating the same monochromatic mode. Existing coherent inputs and
   curved surface boundaries are retained. The checkbox stays off until selected.
3. Review the physical tip inputs and observation Z, then press the green
   **Calculate coherent beam** button. Failed source-domain checks do not allocate
   wave arrays or automatically change tip dimensions, energy or energy spread.
4. After Calculate, return to **Ray Diagram**, enable **Beam analysis**, and
   select **Coherent XY** in the right panel. Keep **Follow Ray Z** checked:
   dragging the cyan Z cursor updates the coherent screen beside the rays.
   **Particle rays** restores the classical transverse view. **Controls…** opens
   the full Coherent beam settings; both screens share one calculation session.
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
   Calculate again to establish a new captured session.

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
0.3 eV energy FWHM is not an executable default for the present paraxial Gaussian
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

For a **Gaussian-Schell tip**, the page loads saved coherent source inputs when
present. Otherwise a flat tip uses the user-selected idealised example defaults
described above; the classical instrument's width and energy law are unchanged.
Explicit tip width, energy, energy spread,
wavefront curvature and tilt edits apply to its detached wave calculation;
current and hardware remain captured from the instrument. These are launch
boundary inputs, never an independently configured downstream probe.
Incoherent angular RMS is given in mrad. Zero means one
spatially coherent mode; nonzero energy spread still needs multiple energy
samples. Diffraction's angular spread follows spatial width and wavelength, so
the independent classical angular distribution cannot also be imposed as the
same constraint. Position and angle probabilities do not uniquely determine a
wavefunction or coherence.

The intensity display can use linear or logarithmic colour limits. This does
not change complex fields, probabilities or the calculation session.

For a **physical surface tip**, first use needs an explicit reservoir mean kinetic
energy and RMS width in eV, plus any existing relative edge-to-apex phase. The
current model supplies one axisymmetric spatial mode per energy; it is not an
arbitrary partially coherent spatial source or a prediction of metal tunnelling.
Classical normal/tangential probabilities cannot define a complex boundary. The
continuous-curvature particle tip cannot silently become a flat Gaussian source.

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
electromagnetic transport, specimen multislice and exact post-gun Z observations.
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
   fields. The prescribed tip current is 10000 nA. The independent coherent
   controls below do not mutate these active particle inputs or fields.
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
4. In **Coherent beam**, select **Use a coherent boundary at the physical tip**
   and enter all source values below. They prescribe the emission boundary at
   the tip of this independent session, before extraction and acceleration;
   they never define a downstream source. Decimal entries are provided because
   the standard spin boxes do not require scientific-notation text.

| Coherent beam source control | Manual entry |
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

5. Disable **Follow Ray Z** and enter **Z = 1605.19395749225 mm**. In **Advanced
   numerical budgets**, set the values below. Select **Log intensity**, then
   **Calculate coherent beam**. No preceding high-accuracy particle run or
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
intensity plots are never renormalised to hide that loss. Numerical guards
operate on envelope spectrum plus analytical carrier, so a genuine wave node is
not mistaken for aliasing. Alternative resolved propagation charts are attempted
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
