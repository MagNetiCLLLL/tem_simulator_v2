# TEM Simulator v2

Clean PySide6 reconstruction of the TEM simulator.

## Development setup

1. Run `setup_env.py` with a 64-bit Python 3.12 interpreter.
2. Select `.venv\Scripts\python.exe` as the PyCharm project interpreter.
3. Run `main.py`.

The default high-accuracy calculation is sized for a 32 GiB workstation. A
24 GiB application memory budget is checked before submission so extreme
ray-count/step combinations fail safely instead of exhausting system memory.
The calculation toolbar exposes `Auto (GPU / CPU)`, `CPU`, `Numba CPU`, and
`CUDA GPU` backends. CUDA accelerates the independent-ray RK4 column
propagator. With the optional CuPy extra installed, it also accelerates
multislice propagation and the TEM/STEM image-formation FFTs. Angle-resolved
STEM keeps probe batches, potential configurations, FFTs and detector
integration resident on CUDA and copies only the final detector arrays. One
reusable multislice plan caches the reciprocal grid, anti-alias mask and
slice-geometry propagators across every probe batch and frozen-phonon
configuration in that calculation.
Install that extra with
`.venv\Scripts\python.exe -m pip install -e ".[gpu]"` on a CUDA
12.x system. Unavailable drivers, dependencies, devices, JIT/FFT failures, or
GPU memory failures fall back to the CPU reference without aborting the run.
The status bar, wave-image summary and calculation log report the backends
actually used. Tiny previews stay on CPU in Auto mode because accelerator
launch and transfer overhead is larger than the useful work.

The standard project environment includes ASE and abTEM for finite-projection
Lobato--Van Dyck independent-atom potentials. Silicon and gold use
commensurate crystal builders, while a user CIF/MCIF is orthogonalised and
cropped directly to the scan ROI plus probe padding inside the finite X/Y/Z
sample envelope. A macroscopic sample therefore does not create a macroscopic
supercell. The canonical orientation is a unit quaternion; zone-axis `[uvw]`,
an independent in-plane direction and numeric/mouse rotations all update that
same physical state. A custom CIF never silently falls back to a different
material model. IAM includes sampled coherent elastic diffraction but not
bonding charge, magnetic scattering or a full Mott treatment. Real-specimen
inelastic transport is a separate probability-conserving model: material IMFP
anchors generate zero-loss, plasmon/low-loss, core-ionisation and plural-event
populations, while optional effective absorption removes current from the
tracked transmitted beam. It is not an absorptive multislice potential or a
full energy-differential EELS/dielectric calculation.
Magnetic-lens excitation is consistently expressed on a 0–100% scale; lenses
that require stronger fields own correspondingly higher 100% field
calibrations instead of using over-100% excitation values.
Every optical magnetic-lens part also owns `field_polarity`,
`field_polarity_status` and `field_polarity_source` in its selected instrument
TOML. `field_polarity` is the effective signed Bz direction along the
source-to-detector +z axis and is independent of the non-negative excitation
percentage. The current common-column and corrector signs are explicitly
provisional model assumptions, not manufacturer-confirmed coil directions.
Selecting a new assembly resets these defaults from TOML; ordinary
recalculation preserves a user's runtime direction override.

## Current MVP

- Loads the FEG, FEG + monochromator and thermionic gun TOMLs.
- Loads five C2/C3/corrector column arrangements with the Energy Filter
  recording system permanently installed; Instrument Setup no longer exposes
  a recording-system selector.
- Validates all 10 module TOMLs, 466 part definitions and 15 selectable catalog
  assembly combinations at startup. The legacy no-filter TOML remains
  validation-only historical geometry.
- Rejects magnetic-lens manifests that omit a signed `field_polarity` or its
  provenance, and reports that provenance in Magnetic Field diagnostics.
- Exposes the complete signed sample-to-plane `J_img` and `J_diff` 2x2
  matrices instead of reducing image magnification and camera length to
  unsigned X-only scalars. The matrices retain rotation, reflection and
  anisotropy in the common column X-Y frame.
- Shows every active part, its derived assembly anchor and absolute positions.
- Edits operating parameters and saves/loads complete operating profiles as
  TOML.
- Adds a left-side **Direct Alignment** page with four user-level coupled
  controls. Nanoprobe convergence and Microprobe illuminated diameter solve
  C2/C3 together. Image magnification solves Objective/D/I/P1/P2 as one preset;
  Diffraction camera length independently solves D/I/P1/P2 against the
  Objective back-focal plane. The individual low-level controls remain
  editable.
- Defines Direct Alignment targets, ranges, coupled-device lists, tolerances
  and calibration provenance only in `configs/operating_modes/catalog.toml`.
  Solves run on a background state snapshot and commit all coupled values only
  if the target and conjugate constraint pass fine-step validation and the live
  state has not changed. Failed, unreachable or stale solves change no lens.
- Edits existing module and part TOML values with validation and atomic
  rollback when an assembly becomes invalid.
- Recalculates a direct-beam preview after a lens excitation change without
  blocking the GUI.
- Marks every active component centre on the ray plot, keeps lens centres
  labelled, and reveals other component labels progressively while zooming.
- Uses distinct high-contrast markers for apertures and draws paired
  deflectors at both TOML-defined U/L interaction planes; coincident virtual
  planes remain explicitly identified rather than being given a false gap.
- Provides an optional auto-zoom mode that focuses both axes on the selected
  component while retaining mechanical and neighbouring ray-path context.
- Stops every displayed ray at its first physical intercept and reports the
  maximum physical X angle plus the transverse display magnification, so the
  schematic's deliberately unequal axis scales cannot be mistaken for a
  90-degree electron deflection.
- Applies an exact position-dependent circular vacuum-wall cutoff in X/Y and
  keeps the earliest stop when walls, apertures and recording devices compete.
  Every part owns `vacuum_inner_diameter_mm`; uncovered drift spaces use their
  module's `vacuum_drift_inner_diameter_mm`.
- Treats the vacuum wall strictly as a mechanical cutoff. Propagation and lens
  fields remain valid beyond the last defined wall segment and are not clipped
  to a lens body's mechanical dimensions.
- Draws the stepped vacuum walls at physical scale and marks ray stops by cause with
  clickable Z/X/Y/radius diagnostics.
- Draws an aperture's mechanical body centre and effective optical stop as
  separate references when the TOML intentionally assigns different planes,
  including the Objective Aperture stop at the nominal back focal plane.
- Renders each active aperture stop as two solid blocking segments separated
  by a blank X-opening derived from the current radius and offset controls;
  disabled apertures retain only a dotted, non-blocking reference.
- Detects and labels gun and lens crossovers in both interactive previews and
  one-shot calculations, including their axial position and RMS beam radius.
- Provides separate **Physical Layout** and **Magnetic Field** pages. The
  layout uses hollow-cylinder projections, resolved vacuum bores, optical
  references and dynamically packed multi-row name callouts with dashed
  leaders to the corresponding hardware; the field page plots solver-identical
  total and per-lens Bz, field support, peak field and focal length with
  tree/plot selection linking.
- Provides a separate **Optical Transfer** page with this plain-text mapping:
  `r_plane = J_img @ r_sample + J_diff @ theta_sample`. It reports both
  matrices, conjugacy residuals, equivalent magnification/camera length,
  rotation, handedness and anisotropy at objective and recording planes.
  Capturing one Image state and one Diffraction state at the same plane gives
  the normalised diffraction-vector-to-image-direction transform.
- Makes every resolved name, centre marker and drawn hardware body on
  **Physical Layout** and **Energy Filter** a navigation target. Dashed
  leaders remain visual guides and do not intercept clicks intended for the
  hardware beneath them. A
  single click opens the left instrument dock, selects the matching Optical or
  Mechanical category and shows its Operating or TOML parameter page. Energy
  the Iliad XO / optional EFTEM slit therefore opens directly on its inserted
  state, selected loss and energy-window controls.
- Models C1, C2 and C3 as lens assemblies with explicit upper/lower tapered
  pole-piece children, matching the Objective assembly's mechanical hierarchy.
- Runs a one-shot full calculation with a configurable ray count and axial
  integration step.
- Runs the optional TEM wave-imaging backend during a high-accuracy calculation
  and displays the image and diffraction pattern on a dedicated page.
- Provides a central **Sample** page with insert/retract and Real/Virtual mode
  controls, specimen presets, TEM/STEM wave and multislice settings, finite
  sample/scan/ROI overlays, +Z beam, zone-axis alignment and dual mouse
  behaviour. Sample parameters are not duplicated in the left instrument
  tree; clicking the specimen in a layout opens this page directly. It
  contains specimen state and structure only;
  detector images live only on the STEM page. A custom CIF is orthogonalised
  with abTEM and periodically expanded through the finite-sample/current-ROI
  intersection. ASE covalent neighbours form the sticks, ASE/Jmol colours and
  reduced covalent radii form the element-specific balls, and a legend beside
  the view names every displayed element. A user-adjustable soft atom limit
  crops only an explicitly reported rendering window for macroscopic volumes;
  it never reduces the calculation ROI. Mouse drag orbits the camera by
  default; explicit edit mode changes a draft physical orientation which must
  be applied. PyQtGraph OpenGL/PyOpenGL is used when supported and a labelled
  two-dimensional ball-stick projection is used safely otherwise.
- Gives AC Scan and AC Descan the same raster and two-foil controls because
  they are the same deflector principle at different axial positions. Their
  TOML geometries mirror one another about the sample. The active signed
  first-order optics calibrate AC for a pure sample shift; Descan receives the
  exact opposite 2x2 scan command and derives its lower-foil coupling so that
  the scanned chief ray is stationary at the Selected Area Aperture
  image-reference station. Pixel count and a square pixel pitch from 0.001 nm
  to 1 mm define `FOV = count x pitch`; unreachable coil demands fail
  explicitly. One HAADF/DF/BF frame is calculated from the signal intercepted
  at every probe position and then replayed from cache. The same within-frame
  scan position animates the Ray Diagram without rerunning ray physics, and
  the diagram remains freely rotatable during playback. Preview uses the fast
  geometric detector approximation; High accuracy can use wave/multislice
  detector integration with first-order descan acceptance shifts.
- Treats Objective Aperture and Selected Area Aperture as physical
  diffraction- and image-reference stations, respectively, rather than
  asserting that their current optical state is ideal. Sample-to-plane
  position and angle Jacobians classify every reported station as `image`,
  `diffraction`, or `mixed`; objective first-image and first-diffraction
  locations are recomputed whenever sample, voltage, or objective excitation
  changes. Exact requested observation Z values are included in propagation,
  avoiding display-grid interpolation errors at these planes.
- Names the scan workspace **STEM**, with separate **Geometry** and **Images**
  subtabs. BF/DF/HAADF images use physical scan coordinates, keep one X unit
  equal to one Y unit, and retain unrestricted interactive pan/zoom. The image
  notice distinguishes geometric detector clipping from virtual-sample and
  CIF/multislice signals. It also warns when the FOV leaves the finite sample
  or the pixel pitch is coarser than half the shortest periodic CIF atom
  spacing. Polygon/wedge patterns from `geometric_detector_interception` are
  therefore identified as preview acceptance boundaries, not atomic contrast.
- Keeps the optical sample Z as a probe-reference plane when **Sample
  inserted** is cleared. Retracting the holder removes diffraction, diffuse
  ray broadening and atomistic/wave interaction; retained CIF settings are
  dormant until the specimen is inserted again.
- Supports two explicit specimen modes. **Real sample (CIF / crystal)** uses a
  TOML crystal or custom CIF for high-accuracy finite IAM/multislice. It never
  creates artificial `+g/-g` or diffuse diffraction beams. Instead, measured
  material IMFP anchors and independent-event Poisson statistics create
  absolute zero-loss, plasmon, core-ionisation and plural-inelastic ray
  populations with representative energy loss and characteristic scattering
  angle. User MFP/loss overrides support measured films and custom CIFs;
  effective absorption is disabled unless explicitly supplied. **Virtual sample** has
  extensible diffraction-spot/ring, Gaussian/diffuse, arbitrary-angle,
  user-screened-power-law, physical screened-relativistic-Rutherford and
  absorption rows. Probabilities are absolute, are never normalised, and must
  sum to at most one; the remainder is direct beam. Rectangles, ellipses and
  NPY/PNG/TIFF grayscale maps define local density inside the finite slab,
  with vacuum outside and optional convolution by the calculated probe.
- In Ray Diagram, hue denotes the interaction type: Real energy-loss state or
  Virtual user channel, with neutral hues for incident/vacuum/zero-loss paths.
  Five dark-to-bright shades denote each
  ray's 3-D sample-plane convergence semi-angle relative to its branch's
  weighted chief ray; the brightest shade is reached at the calculated
  99%-current convergence angle. Real inelastic characteristic-angle kicks are
  removed before calculating this brightness. Selecting any axial Z opens a
  source-normalised table of sample interaction probability, fraction reaching
  that Z, local composition, energy loss, absorption and hardware stops, with a
  conservation check. The Transverse X-Y page retains its separate
  spatial/quadrant colour encoding.
- Reports each installed BF/DF/HAADF detector's TOML-authoritative axial
  position, inner/outer active size and collection angle. Collection angle is
  calculated through the active signed sample-to-detector transfer, including
  rotation or anisotropy rather than assuming angle = radius / axial distance.
- Integrates wave and virtual angular intensity through each detector's actual
  `hit_mask` after the complete signed 2x2 sample-to-detector transfer, in
  axial order so upstream interception cannot be double counted. A STEM frame
  reports source fractions, pA, expected electrons per dwell, optional seeded
  Poisson counts, uncollected/absorbed/truncated fractions and a separately
  identified optional high-angle tail; it does not retain a full 4D cube.
- Provides a complex128 CPU-reference symmetric split-operator multislice
  engine and an optional complex64 CuPy CUDA path with safe CPU fallback,
  explicit angstrom/inverse-angstrom FFT conventions, 2/3 anti-alias
  bandwidth limiting, per-slice phase diagnostics and integrated-intensity
  checks. Silicon [110] and gold [001] provide explicit 3-D atomic potential
  slices; vacuum and presets without an atomistic definition retain the
  explicitly labelled continuous-column model.
- Treats the resident STEM CUDA calculation as one atomic operation. Any
  allocation, propagation, FFT or detector-integration failure discards all
  partial GPU output and recomputes the complete observable with the NumPy
  complex128 reference. Metrics report residency, fallback reason, potential
  uploads, batch count, reusable-plan builds/uses/cache bytes and bulk
  host-transfer bytes.
- Provides reproducible frozen-phonon configurations with preset or user RMS
  displacement and a user-visible random seed. TEM image/diffraction and STEM
  detector signals average configuration intensities, never complex exit-wave
  amplitudes. A relative standard-error diagnostic helps judge finite-ensemble
  convergence without claiming a universal configuration count.
- Treats the atomistic potential as a neutral-atom IAM and the thermal motion
  as independent isotropic Gaussian (Einstein) displacements. Bonding charge,
  correlated phonons, absorptive/inelastic *potentials* and magnetic specimen
  fields are not included. Stochastic inelastic energy-loss populations are
  transported separately from this conditional zero-loss coherent wave. Strict
  multislice angular support is the default
  and omitted intensity is not renormalised. An optional screened relativistic
  Rutherford approximation can complete only angles beyond that support; it
  uses explicit Z/areal-density/screening inputs, is reported separately, and
  is not described as Mott or silently blended into the wave-supported range.
- Defines the STEM probe semi-angle as the weighted 99% angular containment
  about the three-dimensional chief ray and reports RMS/95%/edge angles
  separately instead of using RMS convergence as a hard pupil radius.
- Defines the operator-facing Nanoprobe convergence control explicitly as the
  current-weighted 95% radial semi-angle about the chief ray. This preserves
  the calibrated 20-40 mrad operating convention and is invariant to a common
  Larmor rotation; it is intentionally distinct from the wave pupil's more
  conservative weighted 99% containment. Microprobe area is the corresponding
  95%-current diameter and is constrained by sample-plane radial wavefront
  curvature and a 0.5 mrad maximum semi-angle.
- Keeps gun ray paths on a strict shared-Z grid while retaining equal-time
  electron snapshots and per-ray plane-arrival times for future wavefront/arc
  overlays at important optical planes.
- Samples the cold-FEG energy tail on a strictly positive kinetic-energy
  distribution while preserving its requested mean, FWHM and deterministic
  low-discrepancy ray ordering.

The `configs/instruments` files remain the authority for static mechanical
geometry and optical references. The **Anchors** tab must be checked whenever
a component is added, removed or restructured. A changed manifest is accepted
only after the complete catalog and active assembly validate.

## Instrument-definition authority

- `configs/instruments/catalog.toml` selects exactly one gun, one column and
  one project/recording module. It does not contain a second copy of part
  geometry.
- Every active part has one variant-scoped definition ID:
  `<module TOML>::parts[<canonical key>]`. Runtime components expose this ID
  for diagnostics and tests.
- The selected module TOMLs own part membership, canonical display names,
  parentage, axial/mechanical geometry, vacuum geometry, optical references,
  magnetic-field polarity provenance, detector orientation and the Objective
  and Energy Filter structural models.
- `configs/operating_modes` owns calculated mode presets. Runtime state owns
  user excitation, alignment, insertion and solver choices; saved state omits
  all manifest-owned structural attributes.
- Python owns component behaviour, validation, drawing and propagation
  algorithms. Any bootstrap geometry needed to instantiate a class is read
  from TOML and is replaced by the already-resolved selected assembly; it is
  not a second authority.

The 466 part rows contain 192 logical keys and 274 intentional repetitions
across mutually exclusive hardware variants. They are not applied as an
override stack. Catalog validation rejects duplicate files, module keys,
selection signatures, per-file part keys/orders, runtime keys and any active
assembly collision, then applies all 15 selectable Energy Filter assemblies in
regression tests. The no-filter module remains catalogued only so its historical
geometry continues to be validated.

## Direct Alignment reachability

The current 300 kV C3 + Probe Corrector non-OEM model has regression-validated
working points at 30 mrad Nanoprobe convergence, 2.0 um quasi-parallel
Microprobe diameter, 10x/100x/1,000x/10,000x/100,000x/1,000,000x physical
Image magnification and
0.01/0.05/0.1/0.5/1/2 m effective camera length. The validated Microprobe area
range is 0.5-2.2 um
while retaining its parallel constraint and keeping the default C2 solution in
the 30-70% window.

Image now covers the complete requested 10x-1,000,000x interval, including
non-table values validated by local solves. It does not assign five independent
"lens magnifications." Instead, each magnetic lens is reduced to an engineering
thin-lens focal event derived from its isolated `integral(Bz**2 dz)`, with its
signed Larmor rotation retained separately. The five events are composed as
one ABCD transfer from sample to the active recording stop: `B=0` is the image
condition and `|A|` is the displayed magnification. LM targets through 1,000x
use a separate branch with Objective nearly bypassed; Normal/HM targets switch
to an Objective-on five-lens branch. The TOML preset table supplies only
deterministic starting currents; every requested value is still solved and
validated against the production ray tracer.

This focal-event calibration is explicitly non-OEM. It avoids the numerically
unstable and physically implausible alternative of inflating Gaussian peak
fields into multi-tesla oscillatory branches. Simulation metrics retain
`signed_image_magnification`, `image_inversion` and
`image_larmor_rotation_deg`, while the user-facing magnitude remains `|A|`.
Diffraction remains a separate distributed-field/BFP calculation and never
uses a fictitious Diffraction-lens image magnification.

The Diffraction input range remains 0.01-5 m. It is a request range, not a claim
that every current non-OEM field setting is reachable. If any solve fails its
target or conjugate tolerance, the GUI reports the candidate and restores the
exact previous state.
The desktop window applies the selected TOML operating-mode pair at startup and
after a compatible assembly reload. Large projector changes use bounded
logarithmic continuation followed by a validation-grid refinement, preventing
the previous false camera-length plateau near 0.10865 m. With the current
non-OEM P2 rating, the 5 m request follows the continuous solution branch to
about 2.59 m and reaches the P2 upper excitation limit, so 5 m is not yet a
validated working point.
The physical sample-to-recording-plane magnification is not relabelled as a
manufacturer-calibrated nominal magnification. Absolute detector/display scale
still requires an evidence-backed detector calibration.

## Image/diffraction orientation calibration

The camera part in each recording-system TOML owns
`detector_axis_rotation_deg`, `detector_flip_x`, `detector_flip_y`,
`detector_orientation_uncertainty_deg`, `detector_orientation_status` and
`detector_orientation_source`. The shipped values are deliberately marked
`uncalibrated_identity` with 180 deg uncertainty. They are coordinate
placeholders, not measured camera mounting or display-pipeline calibration.

The **Optical Transfer** page therefore distinguishes a model relation from an
absolute hardware calibration. A crystal-orientation claim requires measured
detector axes and evidence-backed magnetic-field polarities; the full 2x2
matrix must be used because a scalar rotation alone cannot represent mirroring
or anisotropy. The reported reciprocal-vector direction is normal to lattice
planes and is not automatically a direct-lattice vector.

Current `J_img`/`J_diff` records cover the straight axial column through the
Energy Filter entrance. The curved sector/M01--M10/Zebra branch is explicitly
excluded until its own first-order chain is derived and validated.

## Energy Filter physical layout

The axial Diffraction, Intermediate, P1 and P2 lens housing/yoke/coil layers
are ordinary TOML mechanical parts and appear in **Physical Layout**. Iliad's
curved post-column path is shown only in the separate **Energy Filter** page;
its internals are never flattened into coincident main-column markers.

The public functional topology is represented explicitly as one large tapered
prism, ten independently powered multipoles, an XO crossover/optional EFTEM
energy-slit assembly, a separately named fast shutter, a confirmed but not yet
field-modelled dynamic-focus electrostatic quadrupole, MultiEELS bias tube,
Zebra camera deflector, optional EFTEM output plane and the Zebra detector.
Each component has one TOML row and one canonical key. Public evidence says
most of the ten multipoles are dodecapoles, but does not identify the pole
family of every numbered element or publish production labels/exact order.
M01--M10 are therefore explicitly marked as stable simulator model indices,
not manufacturer part names.

Published numeric anchors are kept distinct from provisional mechanics. The
entrance aperture defaults to the reported 5 mm experimental condition. Each
Zebra strip has a 28.672 x 0.800 mm active area (2048 pixels at 14 um in the
dispersive direction), while the 2-D alignment area is 28.672 x 3.584 mm.
Strip pitch, inter-strip gaps, detector package, prism radius/bend/gap, all
multipole coordinates and envelopes, and electrostatic-element dimensions are
still unknown. Current values such as the 135 mm/90 deg prism reference, 22 mm
M01--M03 housings and 28 mm M04--M10 housings are adjustable non-OEM starting
parameters with explicit provenance, not manufacturer measurements.

The branch view draws the TOML-backed hollow carriers and provisional
electrostatic envelopes, labels unknown external packages instead of inventing
them, and keeps X/Z scaling independent. Clicking a label, centre marker or
drawn body opens the same component in the left editor. Runtime controls use
**Operating**; structural values and evidence status use **TOML**.
