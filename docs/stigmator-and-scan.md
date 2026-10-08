# Stigmators and scan/descan

## CM, objective and scan design

**Design Explorer > CM / objective / scan > Calculate CM / scan design** compares
the current installed field model at fixed CM excitation magnitude and opposite
signs. Microprobe uses the positive control sign, Nanoprobe the negative sign.
Switching the stored operating preset preserves the applied CM magnitude; the
other preset controls still follow their existing definitions. Optical off does
not remove the CM field or set its drive to zero. An explicitly disabled or zero
CM is not accepted as this mode-switching mechanism. The default magnitude is
owned by `configs/operating_modes/catalog.toml`, not a second GUI control.

[EPFL's explanation](https://www.epfl.ch/labs/lsme/tem-ray-diagrams/) describes
reversed CM polarity coupling to the objective pre-field, while explicitly
identifying its thin-lens drawings as schematic and the TFS implementation as
different in detail. The project sign convention is not an OEM winding/current
calibration. Percent drive is not amperes.

The design calculation holds the Tip, objective and all other fields fixed.
It uses the same installed-field transverse tracer as the rest of the program.
For an axisymmetric magnetic example, radial focusing in the Larmor frame
depends on `(e B_total / (2 p))²`. Consequently, the spatial cross term
`2 B_CM B_objective` matters; reversing an isolated lens does not turn it into
a diverging lens. Peak field alone cannot determine the required placement.

CM candidates translate its existing analytic profile within the space between
the AC downstream body face and the upper-objective upstream pole face. Mapped
CM fields are rejected for relocation because their translated validity has not
been established. This checks axial envelopes, not full three-dimensional
collision or a coupled magnetostatic solution. The live geometry is unchanged.

The table ranks a first-order necessary condition using the mechanical
coordinates `(x, y, theta_x, theta_y)` and `M = [[A, B], [C, D]]`:

- Nano point focus requires `A_n + B_n K = 0`.
- Micro parallel output requires `C_m + D_m K = 0`.
- The same incoming direction-position correlation `theta = K r` must serve
  both modes. The displayed difference is the spectral norm of the two required
  K matrices, when the solves are well conditioned.

K can contain an antisymmetric part inside magnetic fields; it is not generally
a scalar wavefront curvature or a phase Hessian. A small mismatch is not finite
probe acceptance. The Python analysis API also accepts an explicit incident
covariance and RMS limits; the GUI diagnostic does not fabricate those inputs
from an unrelated retained beam.

AC calculations use both transverse axes and the complete combined-field
transfer at the selected CM candidate. They solve for physical upper/lower kick
matrices producing the applied scan FOV while holding either mechanical or
canonical sample direction fixed. Canonical direction includes the local vector
potential. Choosing this diagnostic constraint does not alter the production
scan calibration. Static AC alignment is reserved in the drive-limit budget;
the response is linearized about a neutral AC reference, not qualified for
arbitrary large displaced/tilted operating rays. A current in amperes requires
an independently established coil field/current calibration.

The optional three-point AC position study preserves centre separation and
searches the user-specified upper-coil interval. Its ranking is by required
drive and conditioning, not proof of mechanical installability. No candidate
is installed automatically. The pivot check tests the complete two-axis
response (each axis normalized to its own FOV), including local refinement;
otherwise it reports only closest approach. A scan pivot is not an individual
probe waist, and neither scan constraint alone verifies detector stationarity.

On the default geometry, the legal axial CM centre range is 1521.5–1524.5 mm,
with only about 0.72–0.97% normalized CM/objective field overlap. Five-position
diagnostics at 10% and 90% put the smallest sampled input-correlation mismatch
at 1524.5 mm, but neither establishes a qualified dual-mode probe. The narrow
current analytic CM field has negligible amplitude near the upper-objective
field centre. Matching real integrated-lens behaviour therefore requires a
mechanically consistent CM cavity/pole geometry and corresponding coupled field
profile; moving the existing CM into the solid upper pole is not an allowed fix.

## Stigmator controls

Select **Condenser Stigmator**, **Objective Stigmator** or **Diffraction Stigmator**
in the optical component list. All three use independent X/Y quadrupole channels
with a 0/45-degree structural basis (`field_model = "normal_skew"`). The obsolete
rank-one X-minus-Y law is not a selectable or loadable model. Explicit unsupported
model identifiers are rejected rather than mapped to a different physical law.
No stored percent strengths or lens presets are rescaled or recalculated.

X and Y are two quadrupole bases, not independent focusing lenses along the
screen X and Y axes. With the supplied 0/45-degree basis, their coefficients are
`q_normal = scale * X / 100` and `q_skew = scale * Y / 100`. Their local transport
tensor is `K = [[q_normal, q_skew], [q_skew, -q_normal]]`, multiplied by the existing
axial Gaussian. Units are m^-2; the transverse equation is `theta' = -K r`.
The principal-axis angle is `atan2(q_skew, q_normal) / 2`. Reversing a pure
component (or the entire tensor) swaps focusing axes by 90 degrees. With both
channels active, reversing just one changes their vector sum and does not in
general rotate the axes by 90 degrees. A zero tensor has no defined principal axis.

The basis is defined relative to column X/Y, not the rotated Ray Diagram view.
Structural TOML owns basis angles and physical positions. Effective field widths
are initialised from TOML and remain editable through the existing runtime validator.
The Gaussian FWHM is the effective length; it is not a measured coil support.
Percent scales `max_strength_m2`, not amperes. The manifests explicitly identify
the co-located analytic model as **not a measured coil map**. Existing housing
dimensions, excitation presets and gun stigmation remain unchanged.

The two-element, 45-degree arrangement is documented in the original
[FEI Tecnai User Interface Manual](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_alignments_doc.pdf),
printed page 17, Stigmators. That reference supports the principle, not this
simulator's coil dimensions or absolute strength calibration.
TOML also declares the generic eight-coil topology: two interleaved four-pole
windings. [JEOL's stigmator explanation](https://www.jeol.com/words/semterms/20201020.111014.php)
illustrates this principle. Housing rendering is unchanged; this declaration
does not add measured coil CAD, turns, materials or a finite-element field solve.

**Direct Alignment > Condenser twofold beam shape** offers a bounded solve for
the independent model. The target is zero normal/skew position covariance at the
specimen entrance. Set current, effective-sample and maximum-D95 gates before
running. The existing background candidate/apply/undo path is used. A failed,
cancelled or stale solve cannot change the live instrument. Numerical search
bounds are not hardware ratings. This geometric task does not measure or certify
wave astigmatism, full convergence or atomic-resolution imaging.

## Single-frame STEM images

The startup ray preview and later Preview/Medium updates do not acquire STEM
frames, even when raster and image generation are already enabled. Until the
first acquisition, the detector image panels remain empty. Later ray previews
retain the previous STEM frame and indicate that it needs updating.

Prepare the particle result with **Run high-accuracy once**. Enabling the AC
raster and **Generate STEM detector images** requests one frame when the second
of these two options is switched on. The completed HAADF, DF and BF arrays are
revealed once, then remain displayed. There is no repeating refresh or Pause
refresh control. Short frame periods display the complete frame immediately.

Click **Calculate STEM (single frame)** to request another single scan. Valid
upstream particle results and matching frame data can be reused; replaying a
stored result never changes the Poisson seed or generates new detector counts.
Changing input values retains the previous complete frame until a new result
arrives. Tab, result-source, contrast and display-quantity changes do not start
another acquisition or another playback pass.

## Scan/descan controls

The Descan entry is a control channel of the **Image/Diffraction Deflectors**.
Static image/diffraction-shift and dynamic descan commands drive the same upper
and lower coils. Their drives are summed once at the host's interaction centres;
Descan has no independent solid, vacuum restriction or collision envelope.
This relationship follows the
[FEI Tecnai manual, sections 3.3.3–3.3.4](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_alignments_doc.pdf).

The upstream Beam Shift/Tilt pair and downstream AC Scan pair remain separate.
In a probe-corrected column the scan must remain after the corrector and before
the objective, as described by [CEOS](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR).
Shared control channels do not imply that every deflection function along the
column uses the same hardware. Axial coordinates and coil dimensions remain
adjustable simulator values, not measured OEM geometry.

Older held calibrations used an independent Descan position. Their saved host
identity prevents old matrices from silently driving the relocated shared coils.
Use **Calibrate Descan (keep AC)** when retaining a valid held AC calibration,
or **Calibrate and hold** when intentionally recalculating both pairs.
Automatic calibration uses the current host geometry. Disabling the physical
Image/Diffraction pair suppresses all its drives while retaining each channel's
stored operating settings.

In **Scanning Image > Scanning Parameters > Scan / descan calibration**:

1. Enable the required AC/Descan drives and select the specimen reference.
2. Select an installed **Descan observation plane**, for example Fluorescent
   Screen. Each choice displays the physical component name and current Z in
   millimetres. This is separate from the specimen plane used for probe scanning.
3. Set the desired raster and current optics; click **Calibrate and hold** to
   recalculate both pairs using the production mechanical-angle AC constraint.
4. Adjust **AC pivot X/Y** or diffraction-lens excitation. The held ratios stay
   fixed, so residual scan angle and pattern displacement remain observable.
5. Recalibrate explicitly when wanted, or select **Automatic recalibration** to
   restore automatic coupling.

The target is an explicit component key, defaulting to
`selected_area_aperture`. There is no automatic replacement by another station
if it is missing. The former `legacy_image_reference` value is unsupported and
must be replaced with a deliberate physical selection. **Specimen centre** and
**Specimen entrance** remain current, valid scan references; neither is a
compatibility mode. A physical station is not automatically a conjugate image
plane: the current transfer determines its image, diffraction or mixed role.

Both pairs use one raster clock. Each pair has two declared axial X/Y dipole
stations. For the upper/lower AC angular responses `Du`, `Dl`, calibration solves
`Dl R = -Du`; physical optics determine `R`. The command matrix maps the requested
specimen FOV to the actual response. Pivot offsets add to the diagonal of `R`.
Descan uses the opposite command and matches the AC position response at the
chosen observation plane. Equal timing and symmetric positions alone do not
force equal coil ratios when intervening lenses change the transfer.

The production response is linearized from the installed finite-length coil
fields, including the lens field across each coil. A centre-plane angle kick is
an explicitly separate diagnostic approximation; its algebraic residual is not
the residual of the finite-field execution. The CM / scan design page uses the
finite-coil response. A common scan pivot describes the intersection of scanning
chief rays and does not establish that each finite probe is focused.

The specimen direction constraint in the production AC calibration remains the
mechanical incident direction. Inside the objective magnetic field this differs
from the canonical momentum used to classify conjugate planes. Neither a zero
mechanical-angle residual nor a canonical pivot, by itself, establishes a fixed
pattern at every detector. Descan must use the physical station that needs to
remain stationary; holding the beam at the selected-area aperture can increase
its motion on a downstream screen.

### Direction in the objective field

The scan result reports the captured specimen reference (entrance or centre),
its Z, calibration mode and **mechanical X/Y slope span in mrad**. This is the
peak-to-peak scan-induced first-order slope over the sampled pixel centres,
not the convergence semi-angle, an absolute beam tilt, or the angle of the
weighted mean unit velocity of a finite probe. It reuses the position observer's
finite-coil response without a second transport solve. A missing historical
readout is unavailable, not zero. Held drive records do not currently encode
their original angle constraint, so a held record is not labelled canonical
or mechanical merely from its mode.

For electrons, `p_mechanical = p_canonical + e A_vector`. In an axial field,
using the symmetric gauge and the same reference longitudinal momentum `p`,
`theta = u + G r`, where `u = p_canonical,transverse / p` and
`G = e Bz / (2p) * [[0, -1], [1, 0]]`. Therefore a scan at fixed canonical
direction has a mechanical direction change `delta theta = G delta r`.
The ray equations and wave probability current use this same conversion;
holding a phase gradient constant does not hold mechanical current direction
constant inside a magnetic field. This is consistent with the canonical and
kinetic momentum distinction in
[Floettmann and Karlovets, Physical Review A 102, 043517 (2020)](https://arxiv.org/html/2006.12948v2).

For a fixed conservative first-order map to a field-free screen, write
`r_screen = A_mech r_sample + B theta_sample`. The position Poisson brackets
require
`A_mech B^T - B A_mech^T + B (G - G^T) B^T = 0`.
With nonzero specimen `Bz` and invertible two-axis angular map `B`, setting
`A_mech = 0` violates this relation. Thus exactly unchanged mechanical direction
over a two-dimensional scan and an exactly stationary direct beam cannot both
be imposed on that fixed map. This restriction is not a failure to find the
right projector currents. It does not exclude approximate performance over a
small FOV, a field-free specimen, one-axis special cases, or compensation by
scan-dependent downstream deflection.

Canonical diffraction conjugacy is `A_can = A_mech + B G` approximately zero.
It is a valid optical coordinate statement, not a claim of mechanically
parallel scanning at the specimen. The schematic front-focal-plane pivot is
likewise a centroid constraint; focus of each finite probe remains a separate
check. The software retains the mechanical-angle automatic AC calibration and
the existing canonical plane classifier. Neither is silently substituted for
the other to certify screen stability.

Held mode stores the calibration's captured input identity, ratios and reference
FOV. Changing raster extent scales commands without refitting optics. After a
lens change, the labels remain **requested** FOV; actual sampled positions can
differ. Changing observation plane measures the same held drives there; it does
not move a detector or silently recalibrate. Changing specimen reference requires
a new calibration. Singular or over-limit calibration fails without partial edits.

### Projector switching with held AC

With the condenser mode unchanged, selecting the other **Image / Diffraction**
mode and applying it changes only the stored D/I/P1/P2 strengths. Tip, condenser,
CM, Objective, scan controls, held record and receiver insertion remain as set.
Applying the same mode selections again reapplies both complete presets. The
existing Apply button's tooltip explains this distinction.

Stored projector strengths do not solve conjugacy for the current Objective or
chosen receiver. The switch reports that condition as **unqualified** and marks
previous particle/wave results stale. An Image label alone does not establish a
focused specimen image on the physical Fluorescent Screen (`flu_screen`).

To retain an accepted AC calibration while changing downstream optics:

1. Keep **Hold saved calibration** selected. A saved canonical-angle AC pivot
   retains its matrices; **Calibrate and hold** would replace it with a new
   mechanical-angle AC solution.
2. Apply the intended projector settings and select the required physical
   **Descan observation plane**. Receiver insertion remains a separate action.
3. Enable AC scan, Descan scan and their physical hosts, then click
   **Calibrate Descan (keep AC)**. The API is
   `physics.scan_geometry.recalibrate_descan_only(state)`.
4. Recalculate and inspect the result at that receiver. Turning Descan off for
   a diffraction observation retains its saved settings; stationarity still
   depends on the actual diffraction conjugacy and the accepted AC constraint.

The Descan-only operation restores the held AC ratios with the current raster
extent scaling, solves only downstream position compensation and saves the
updated held record. It leaves lenses and the AC angle constraint unchanged;
it does not impose a downstream angle constraint. Merely selecting another
target or changing a projector does not perform this operation. Retaining a
canonical pivot also requires keeping its upstream fields and specimen reference
fixed. First-order compensation is not finite-FOV probe or image qualification.
This operation requires zero logical Descan static X/Y bias, because that bias
uses the same coil coupling being recalibrated. It rejects a nonzero bias without
changing it; separate physical-host alignment remains supported and retained.

STEM display magnification relates the displayed scan width to the specimen
raster FOV (`N * pitch`). It is distinct from the physical sample-to-screen
magnification determined by the column transfer at a specimen image plane.
Changing display zoom, or reducing the requested raster FOV, does not establish
a new optical magnification or change the projector's conjugate plane.

Targets at/after the energy-filter entrance are not offered: the current
first-order scan observer cannot replace the actual filter response with drift.
Unavailable or unsupported target selections fail explicitly.
This does not remove the main simulator's energy-filter transport.
The same boundary applies to scan-position previews and ray-playback response
matrices. Unsupported downstream planes are listed as unavailable; ray playback
does not invent zero motion or an unfiltered continuation past the entrance.

The first-order observer now uses the same signed transfer calculation for
scan pivot, scan scale, descan and plane classification. Analytic hexapole and
spherical terms have zero derivative on the column axis and are excluded from
this diagnostic linear map; the finite-particle forward calculation keeps them.
Mapped fields use small central differences, including their affine reference
trajectory. Exact requested planes use float64 integration checkpoints. Compact
intermediate plotting samples remain display data. This is an on-axis paraxial
observer, not a nonlinear scan-distortion or off-axis corrector calibration.

Pixel pitch is a **request**: footprint FOV is `N * pitch`, and the span between
first and last pixel centres is `(N - 1) * pitch`. At fixed raster phase, changing
frame period changes time and dwell without changing this geometric path. There
is no flyback dead time in the current clock model. The lower-foil gain scalar is
only a representative readback; the full 2x2 matrix owns cross-axis coupling.
Operating profiles and current State records store the upper pair gain and
calibration record, and omit the derived lower gain. Loading it as an operating
input is rejected so it cannot silently change the primary upper gain. Complete
executed snapshots still preserve the exact matrices and their readbacks.

Independent wave calculations resolve these same finite-coil scan controls on
their captured worker state. Repeated Z observations and scan dwells reuse the
resolved calibration for that request. Paired calculations transfer the executed
particle calibration to the wave worker; matching includes the actual drive
matrices and emission time, as well as the instrument snapshot. The original
request identity still determines whether a result belongs to the current UI
settings. A retained result without a matching drive identity can be viewed, but
cannot be presented as a matched particle/wave pair.

## Corrector parameter meanings

Quadrupole `strength_m2` is an effective coefficient in m^-2, with positive
strength focusing column X and defocusing column Y. Hexapole `strength_m3` is
in m^-3: its normal/skew components are the signed strength multiplied by
`cos(3 * orientation_rad)` and `sin(3 * orientation_rad)`. These are paraxial
equation coefficients, not magnetic gradients or measured coil currents.
The local hexapole force is quadratic in transverse position. A 120-degree
rotation repeats the pattern; a 60-degree rotation is equivalent to reversing
its signed strength. A pair's correction depends on relay optics and the incoming
beam, so increasing either strength is not guaranteed to improve probe/image size.

Effective Gaussian FWHM, physical body length and orientation are distinct.
Changing FWHM at fixed peak coefficient also changes its integral. Toggling one
field does not remove the installed relay lenses or physical apertures. Ideal
mode omits nonlinear hexapole action; the parameter description now states this.

The Tecnai manual's beam-shift pivot and diffraction-focus discussion (printed
pages 19 and 52) supports their coupled effect. The implemented response is
first-order transport with the installed finite coil fields; no measured
electronic lag, hysteresis or commercial calibration is claimed.

## Readouts and evidence

In **Working Points > Sampling and Convergence**, enable **Show weighted
phase-space moments** for a retained point. The table shows means and the 4x4
covariance of `(x, theta X, y, theta Y)`. Covariance units are row-unit times
column-unit. Projected RMS emittances are geometric, not relativistically
normalised. Values use retained current weights and never restore optics or run
transport. Empty/metadata-only data do not receive fabricated moments or phase.

The new tensor participates in cache identities and persistent restart content.
Historical scalar-only incident seeds remain on disk and readable as historical
data, but do not supply new tensor restarts. Changing a basis/model invalidates
affected propagation. Other cached high-accuracy results are retained as stale
evidence rather than silently becoming current.

See [the Round 2 progress receipt](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/04e87584ecb88a802813e2f109d67a5719615399/docs/development/ROUND2_OPTIMIZATION_PROGRESS.md)
for historical tests and limitations at that revision, not current execution status.
The [second qualitative batch](https://github.com/MagNetiCLLLL/tem_simulator_v2/blob/04e87584ecb88a802813e2f109d67a5719615399/docs/development/qualitative-second-batch-2026-09-19.md)
records subsequent observer fixes, local symmetry checks and tip-origin evidence.
