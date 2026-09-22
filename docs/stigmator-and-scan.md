# Stigmators and scan/descan

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

## Scan/descan controls

In **Scanning Image > Scanning Parameters > Scan / descan calibration**:

1. Enable the required AC/Descan drives and select the specimen reference.
2. Select an installed **Descan observation plane**, for example Fluorescent
   Screen. Each choice displays the physical component name and current Z in
   millimetres. This is separate from the specimen plane used for probe scanning.
3. Set the desired raster and current optics; click **Calibrate and hold**.
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

Held mode stores the calibration's captured input identity, ratios and reference
FOV. Changing raster extent scales commands without refitting optics. After a
lens change, the labels remain **requested** FOV; actual sampled positions can
differ. Changing observation plane measures the same held drives there; it does
not move a detector or silently recalibrate. Changing specimen reference requires
a new calibration. Singular or over-limit calibration fails without partial edits.

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
ideal first-order transport with instantaneous kicks; no measured electronic
lag, hysteresis or commercial calibration is claimed.

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

See [the Round 2 progress receipt](development/ROUND2_OPTIMIZATION_PROGRESS.md)
for executed tests and remaining work. Coherent tip propagation remains paused.
The [second qualitative batch](development/qualitative-second-batch-2026-09-19.md)
records subsequent observer fixes, local symmetry checks and tip-origin evidence.
