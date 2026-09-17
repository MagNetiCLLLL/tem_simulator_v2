# Stigmators and scan/descan

## Stigmator controls

Select **Condenser Stigmator**, **Objective Stigmator** or **Diffraction Stigmator**
in the optical component list. In **Field model**, explicitly choose
**Independent X/Y (0/45 deg)**. Old profiles and the default instrument retain
**Legacy X-Y (rank 1)**. No preset strengths are recalculated.

X and Y are two quadrupole bases, not independent focusing lenses along the
screen X and Y axes. With the supplied 0/45-degree basis, their coefficients are
`q_normal = scale * X / 100` and `q_skew = scale * Y / 100`. Their local transport
tensor is `K = [[q_normal, q_skew], [q_skew, -q_normal]]`, multiplied by the existing
axial Gaussian. Units are m^-2; the transverse equation is `theta' = -K r`.
The principal-axis angle is `atan2(q_skew, q_normal) / 2`. Thus changing one
channel's sign swaps focusing axes by 90 degrees, while combining both channels
allows continuous orientation. A zero tensor has no defined principal axis.

The basis is defined relative to column X/Y, not the rotated Ray Diagram view.
Structural TOML owns basis angles, physical positions and effective lengths.
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
   Screen. This is separate from the specimen plane used for probe scanning.
3. Set the desired raster and current optics; click **Calibrate and hold**.
4. Adjust **AC pivot X/Y** or diffraction-lens excitation. The held ratios stay
   fixed, so residual scan angle and pattern displacement remain observable.
5. Recalibrate explicitly when wanted, or select **Automatic (legacy)** to
   restore automatic coupling.

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
Historical target selections remain readable but fail explicitly if unsupported.
This does not remove the main simulator's energy-filter transport.

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
