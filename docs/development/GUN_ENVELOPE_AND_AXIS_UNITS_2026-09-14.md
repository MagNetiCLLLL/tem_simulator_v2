# Gun envelope and Ray Diagram units

## Scope

Classical particle diagnostics only. Coherent tip-to-column development remains
paused. No installed geometry, voltages, source defaults, profiles or calculation
caches were changed. The comparisons below use the current default assembly,
not unsaved settings in a running application.

## Display correction

Ray Diagram stored and plotted coordinates remain millimetres. Axis tick values
are now converted to SI metres before PyQtGraph selects a prefix. This gives
`20 nm` for `0.000020 mm`, rather than `20 µmm`; micrometre and millimetre scales
also remain correct. This is not a string replacement: the tick multiplier and
unit are consistent. Cursor coordinates, cached arrays, projection geometry and
user-selected view ranges are unchanged.

Beam analysis already uses explicitly converted units with automatic prefixing
disabled. Its position/angular/phase-space display contracts remain unchanged.
Correcting units does not increase the common-Z sampling density of a cached
Ray Diagram; a highly magnified near-tip drawing can still interpolate between
coarse stored display samples.

## Installed reference geometry and potentials

The default tip apex is Z = 0 mm and electrons travel along +Z.

| Component | Mechanical interval (mm) | Reference centre (mm) | Potential |
| --- | --- | --- | --- |
| Tip | -1 to 0 | Apex 0 | -300 kV |
| Extractor | 6 to 10 | 8 | -296 kV |
| Electrostatic gun lens | 14 to 22 | 18 | -294.8 kV |
| Accelerator housing | 30 to 370 | 200 | Ten staged electrode potentials |
| Accelerator electrodes | First centre 42; last centre 362 | Individual stages | Last stage 0 V |
| Gun exit boundary | 450 | 450 | 0 V |

These are reference dimensions, not validated OEM dimensions. The extractor
and gun lens clear bores are 4 mm and 8 mm in diameter respectively.

Mechanical intervals are not hard field cutoffs. The current axisymmetric
Laplace solve includes all these electrode boundaries simultaneously; fields
extend into the gaps and interact. For example, the default on-axis potential
rise above the tip is approximately 4.000 kV at Z = 1 mm, 4.022 kV at Z = 8 mm,
5.352 kV at Z = 18 mm and 17.050 kV at Z = 30 mm. An electron does not drift
unaccelerated all the way to the extractor centre.

## Executed position sensitivity checks

Command (project interpreter):

```powershell
.\.venv\Scripts\python.exe scripts/check_gun_envelope.py --rays 49 --case default --case lens-nearer --case front-nearer
```

Each case preserves extraction, acceleration, lens electric fields and gun
apertures. Only in-memory electrode positions are changed in the two sensitivity
cases; body lengths, bores, voltages and emission inputs remain unchanged.
The accelerator stays at its original location.

The diameter below is twice the maximum radius about the optical axis among
reaching positive-weight rays, not a FWHM or a current-containing percentile.
X projection span is a different observable. These are small 49-ray diagnostics,
not high-accuracy imaging or experimentally calibrated predictions.

| Case | Extractor centre (mm) | Gun-lens centre (mm) | Envelope diameter at that lens centre (mm) | Envelope diameter at fixed Z = 30 mm (mm) | Rays transmitted through gun exit |
| --- | --- | --- | --- | --- | --- |
| Default | 8 | 18 | 4.6481 | 5.1144 | 20 / 49 |
| Lens nearer | 8 | 14.5 | 3.7382 | 4.7552 | 0 / 49 |
| Both front electrodes nearer | 4 | 11 | 2.8065 | 4.5504 | 0 / 49 |

The default X projection span at Z = 18 mm is 3.7534 mm, consistent with a
roughly 4 mm span in the reported screenshot. At Z = 8 mm the axis-centred
envelope diameter is already 2.2375 mm.

Shortening the distance reduces the width at the relocated lens, but it does
not improve the complete gun transmission with unchanged voltages and
accelerator placement. These are not proposed new operating presets.
The fixed-Z comparison avoids attributing the entire difference to observing
the same diverging bundle at an earlier plane.

The script executes the gun trace directly. Ray Diagram additionally applies
the assembled column/drift-wall envelope; the listed exit counts are gun-only
counts, not complete column or sample currents. This distinction does not
change the reported lens-plane result, inside the gun lens's bore.

## Why the old source looked different

The old classical parameter record uses a planar Gaussian source with a 5 nm
spatial FWHM, angular RMS 1 mrad and angular cutoff 3 mrad. The active curved-tip
model does not consume those angular or virtual-source-width inputs: it emits
over a 10-degree spherical cap with a local normal/tangential energy law.

The electric-field implementation also differs. The old analytic gun-lens
profile applies `potential_scale = 4.22125` to the 1.2 kV control. Its on-axis
potential rise at Z = 18 mm is 9.0655 kV. The grounded model instead sets the
actual annular lens electrode to 5.2 kV above the tip and solves the combined
field, producing 5.3524 kV on axis there. An unchanged displayed voltage is
therefore not an unchanged old/new field or focal action. This audit did not
reinstate the legacy multiplier in the physical electrode model.

Additional controls at unchanged electrode geometry:

- Suppressing angular spread about each local surface normal still gives a
  4.0438 mm envelope diameter at Z = 18 mm (RMS radius 1.7327 mm vs 1.7451 mm).
  The broad envelope is not solely a few high-angle random launch samples.
- Reducing only the emitting-cap half-angle from 10 to 1 degree gives a
  3.9029 mm envelope diameter there. This is an artificial sensitivity control;
  at unchanged surface flux density it also reduces emitted current to
  1.0025 nA. It is not a calibrated fix or a claim of preserved brightness/current.

These two controls were read at exact common-Z grid nodes (8, 18, 30 mm). The
final diagnostic script uses original equal-time histories for arbitrary-plane
crossings, so nanometre-scale readouts do not interpolate coarse display lines.

## Numerical qualification: not converged

```powershell
.\.venv\Scripts\python.exe scripts/check_gun_envelope.py --rays 49 --case default --case refined-field
```

At identical physical inputs and the same 0.2 mm maximum trace step:

| Metric | Default grid | Refined grid |
| --- | --- | --- |
| Actual field grid nodes (radial x axial) | 166 x 461 | 326 x 888 |
| Apex cells per radius | 20 | 40 |
| Maximum emission-surface mesh displacement | 4.763 nm | 2.263 nm |
| Envelope diameter at Z = 18 mm | 4.6481 mm | 4.0236 mm |
| RMS radius about the axis at Z = 18 mm | 1.7451 mm | 1.4509 mm |
| Gun-exit transmitted rays | 20 / 49 | 41 / 49 |

The diameter changes by about 13.4% and RMS radius by about 16.9%. Exit energy
invariants pass in both cases (maximum error below 3e-9 eV, against a 1e-3 eV
budget), but that does not establish trajectory convergence. The small linear
Laplace residual also does not certify the field interpolation or launch mesh.
The grounded implementation reports this lack of convergence certification.

Conclusion: finite travel distance contributes to the envelope growth in this
model, but the absolute 4 mm result is not yet a validated physical prediction.
First qualify the tip-boundary representation, field mesh/interpolation and
particle-step sensitivity; then jointly constrain electrode locations, bores,
voltages and emission distribution. Do not fit mechanical geometry to compensate
for unconverged numerical transport. No physical solver fix is claimed here.

## References and limits

[JEOL's FEG description](https://www.jeol.com/words/semterms/20121024.062458.php)
distinguishes the approximately 100 nm tip curvature radius, small virtual
source and relatively broad emission angles. It also distinguishes extraction
and acceleration anodes. It does not validate this simulator's 8/18/30 mm
geometry or predict a 4 mm beam at this gun lens.

The model neglects space charge and tunnelling-current prediction, uses
reference annular electrodes with insulating radial/back numerical boundaries,
and is not an OEM gun reconstruction. Geometry sensitivity checks alone are
not experimental validation.

## Validation completed

- 119 distinct focused checks passed in separate processes/groups: 35 axis and
  Beam analysis checks, 21 ray display-cache checks, 24 incremental scene checks,
  12 ray-colour checks, 12 lazy-panel checks, 11 cached wave-display checks and
  4 diagnostic crossing-readout checks. The wave-display checks use small cached
  fixtures, not a coherent gun solve.
- An initial combined 80-test GUI run stalled after 58 completed items and was
  stopped. The affected next test passed alone, and all five files subsequently
  passed separately. The combined-run stall remains unqualified; a complete
  combined-suite pass is not claimed.
- Existing Pydantic deprecation and Qt signal-disconnect warnings remain.
- The new diagnostic endpoint readout was corrected for floating-point rounding
  at the exit plane and tested against a truly missing crossing (which raises,
  rather than extrapolating). Default/refined diagnostics were rerun successfully.
- Offscreen plot rendering was inspected at nanometre scale: both axes show
  `nm` with consistent ticks. Syntax compilation and whitespace checks passed.
- No full TEM/STEM image, full-suite, OEM field or experimental qualification was
  performed. No existing application process was closed; only the stalled test
  subprocess was stopped.
