# Probe-corrector reference and validation

The user-supplied photographs of an older **Probe Corrector DCOR (Service)**
interface are references for separating hardware channels, calibrated controls,
and correction diagnostics. They are not a magnetic-field calibration or a
production drawing of the simulated corrector. Those older screenshots show
**STEM @ 200 kV**. Additional photographs identified by the user as the
**Iliad Ultra CEOS probe corrector** show **DCORPRIME SERVICE UI**, operating
at **STEM @ 80 kV**. The current default simulation uses **300 kV**.

## Reading the DCOR screenshots

- **Internal Single Channels / Total Output** shows absolute channel outputs,
  with units including A, mA and microampere. The two main HP channels are near
  105 mA; transfer-lens channels are near 0.75 and 1.03 A. These values alone
  do not specify magnetic field, integrated multipole strength, effective
  length, polarity, or excitation dependence.
- **Calibrated Alignment Tools** shows adjustments associated with aberrations
  and alignment effects, such as StigA1, StigA2, B2, C3, BeamShift and BeamTilt.
  Zero displayed adjustments can coexist with nonzero absolute channel outputs.
  A calibrated control can require coordinated changes to several channels.
- The displayed **Phase Plate** is a phase diagnostic in angular space. It is
  not an image of the real-space probe. Its outer angular extent does not by
  itself specify the working convergence angle or probe diameter.

The general optical roles relevant to the simulator are primary hexapoles for
the nonlinear correction, quadrupole trims for nonsymmetric adjustment,
dipole steering for alignment, and round transfer lenses for the relay between
correction elements. Exact assignments of ambiguous labels such as DPH, HP and
auxiliary channels require additional manufacturer documentation. Neither
element order nor polarity should be inferred from a screenshot label alone.

In a two-stage hexapole corrector, the combined hexapole action can supply
negative spherical aberration while compensating the unwanted threefold
distortion. Cancellation depends on the complete relay and operating condition;
the presence of two enabled hexapoles does not establish correction. See
[JEOL's Cs corrector explanation](https://www.jeol.com/words/emterms/20121023.021059.php)
and [aberration terminology](https://www.jeol.com/words/emterms/20231204.php).

The simulator therefore retains field strengths and excitation settings in its
own units. No A-to-field conversion, OEM accuracy, or manufacturer's calibrated
aberration-control response is inferred from the photographed current values.

## Additional DCORPRIME reference

The two additional user photographs are preserved without alteration:

- [Service alignment tools, 80 kV](references/ceos_dcorprime/service_alignment_tools_80kv.jpg)
- [Internal single channels, 80 kV](references/ceos_dcorprime/internal_single_channels_80kv.jpg)

These extend the interface reference with three distinct kinds of information:

1. **Internal single channels:** named hardware outputs and their units.
   The photographed page includes HP1/HP2, QPH1/QPH2, DPH1/DPH2, transfer
   lenses, steering channels, and channels labelled QPC and HPC. Record these
   as UI labels; their field geometry, longitudinal positions and calibration
   matrices are not specified by the photographs.
2. **Service alignment tools:** named coordinated adjustments, including
   AT_A1, AT_A1fine, AT_A2, AT_A2fine, AT_B2, AT_B2fine, AT_A3C, AT_C3,
   AT_S3, AT_A4, AT_B4, AT_B4pure, AT_D4, AT_A5C and AT_ApDist.
   Other visible controls include HP1_P/HP2_P, QPH1_P/QPH2_P,
   TL2asym/TL2sym/TL2SymFoc, DP2asym/DP2sym, BeamShift, TableauTilt,
   BeamTilt, TiltOL, ShiftOL, UserBeamShift and RestoreShift.
   These controls belong to a different layer from absolute hardware outputs;
   their displayed zero offsets do not establish zero element excitation.
3. **Optical state:** a phase-plate view, an estimated probe-shape view,
   microscope properties and several separate probe-size measures. A future
   simulator interface should preserve these distinctions and identify the
   basis of each displayed quantity.

Both photographs display the following working-state values (transcribed
from the UI, not independently measured or adopted as simulator defaults):

| Display label | Displayed value |
|---|---:|
| STEM voltage | 80 kV |
| Probe semi aperture | 20.2 mrad |
| Probe current | 131 pA |
| Chromatic aberration | 1.35 mm |
| Energy spread FWHM | 0.4 eV |
| Diffraction limit | 126 pm |
| + Source size | 123 pm |
| + Cc × dE | 0.54 eV mm |
| Optimum D50 | 180 pm |
| Total D50 | 180 pm |
| D FWHM | 157 pm |
| I FWHM | 41.76% |

The screenshot's probe metrics are labelled D50 and FWHM. Preserve those
definitions separately from this simulator's geometric RMS radius and D95;
the displayed numbers are not interchangeable. The photographs alone do not
specify the vendor's full probe-size calculation or how its individual
contributions are combined.

The optical-state panel also shows **Exclude 1st & 2nd order**,
**Correction: None** and **No geometric aberrations**. These are the displayed
diagnostic selections/status, not evidence that the physical instrument has
zero residual aberrations or that its corrector elements are switched off.
The pictured probe and phase plate should therefore be retained together with
those settings when used as visual references.

This reference addition does not change the current simulator geometry,
300 kV calibration, field settings or correction algorithm. In
particular, QPC/HPC labels alone do not justify adding field sources or
assigning their physical parameters.

## Inspecting a calculated probe

Run a calculation, then open **Scanning Image → Probe Aberrations**. The sample
probe summary uses all finite surviving incident rays and their current weights
at the physical sample plane. It reports geometric RMS radius, 95%-current
diameter D95, twofold and threefold shape moments, and a local linear estimate
of the waist offset. Positive waist offset means downstream of the sample.
These values need not equal statistics computed from the decimated points
displayed in the transverse ray plot.

For centered positions `z = (x - <x>) + i(y - <y>)`, the shape moments are

`M_n = |<z^n>| / <|z|^n>`, for `n = 2, 3`, with current-weighted averages.

Both range from zero to one and are invariant to translation, rotation and
uniform scaling. An equilateral three-lobed pattern can have `M_2 = 0` and
`M_3 = 1`: a circular covariance ellipse does not guarantee a round spot.
Finite-source sampling also contributes to these moments. Low values alone
do not exclude higher-order structure, broad radial tails, or residual
aberrations, and the moments are not A1 or A2 coefficients.

The coefficient table is a separate, compact reference-ring diagnostic. It
estimates a C3 residual ratio by comparing transverse ray errors with nonlinear
fields off/on. The errors are **positions**, reported in nm, not angles in rad.
It does not fit the full aberration set to the actual illumination bundle.

An em dash **—** means that a residual coefficient has **not been measured**.
It does not mean a measured zero. An explicitly supplied coefficient is marked
as configured, and the existing wave model retains its configured/default
coefficients independently of the geometric shape measurements. A circular
wave image produced with default-zero nonsymmetric coefficients does not prove
that the traced corrector has eliminated those aberrations.

## Acceptance of a corrected operating point

Evaluate these properties together, on the same physical sample plane:

1. **Numerical convergence:** hold optical settings and deterministic source
   rays fixed while refining the integration step. Compare RMS radius, D95,
   waist offset and both shape moments. A setting fitted at one step must remain
   acceptable at finer steps; numerical drift must be small relative to the
   claimed improvement and probe dimensions.
2. **Focus and illumination:** verify the waist relative to the sample, the
   requested current-weighted convergence angle, and surviving current. Enlarging
   a defocused spot, shrinking the pupil, or rejecting most rays can conceal
   asymmetric errors without delivering the requested probe.
3. **Shape and size:** compare the same-scale probe distribution, twofold and
   threefold moments, radial containment and tails. Repeat with adequate source
   sampling; do not select an operating point solely by its covariance or one
   normalized moment.
4. **Aberration scope:** distinguish the reference-ring C3 estimate from a fit
   over the actual pupil. Assess a centered-source bundle as well as the finite
   source, so source broadening cannot hide deterministic optical errors.
5. **Wave interpretation:** geometric ray containment is not a diffraction-limited
   FWHM. A quantitative intensity-probe claim additionally requires converged
   wave sampling and the relevant pupil, residual phase, source coherence and
   energy distribution.

Acceptance tolerances and the validated voltage, aperture, assembly and lens
settings should accompany each calibration. A good default working point does
not certify arbitrary later channel edits or another instrument configuration.

## Recalibrated default and reproduction

The stored **Nanoprobe** preset now restores C3 focus and both principal
hexapoles. It is calibrated for FEG, the C3 + Probe Corrector column, 300 kV,
100 µm C2 aperture at absolute Z 765 mm, and sample Z 1599.2 mm. The current
column geometry achieves about 24.625 mrad at the nominal 25 mrad working point.
The preceding stored 30 mrad result predated the aperture relocation and must
not be interpreted as a measurement on the current geometry.

With 1,000 deterministic emitted rays and a 0.05 mm integration step:

| Quantity | Result |
|---|---:|
| Geometric RMS radius | 0.10435 nm |
| 95%-current diameter D95 | 0.44693 nm |
| Twofold / threefold shape moments | 0.0726 / 0.0109 |
| C3 excitation | 21.2736229083% |
| HP2 / HP1 strength | 626910.5744 / 374822.5253 m⁻³ |
| HP2 / HP1 azimuth | 0 / −0.04231957336 rad |

Refining the step from 0.05 to 0.025 mm changes the RMS radius by less than
0.000001 nm and the local waist position by 0.03042 nm. Source sample size
changes the reported moments and containment radii slightly. The physical
calibration tests also compare the same rays against the linear column and
against spherical aberration without hexapoles, and require unchanged pupil
survival. They do not accept apparent circularity obtained by broadening or
blocking the beam.

A separate 15,000-ray validation of the installed preset (without refitting)
gives RMS radius **0.10665 nm**, D95 **0.44588 nm**, and moments
**0.00743 / 0.02680** at 0.05 mm. It retains 7,891 rays, or **52.6067%** of
the emitted current. The measured 95%-current semi-angle is **24.2381 mrad**;
the difference from the 1,000-ray figure is source sampling, not step drift.
The 0.025 mm result gives RMS radius 0.10665 nm and changes the local waist
by only 0.03042 nm.

In the application, restart to load the updated catalog, select **Nanoprobe**,
click **Apply calculated lens preset**, and run a high-accuracy
calculation. Inspect the exact sample plane (default Z = 1599.200 mm) and
**Scanning Image → Probe Aberrations**. Use a 0.05 mm or finer integration
step for comparing calibration figures. The axial Z control now displays six
decimal places in mm and steps by 1 nm, so nearby defocus planes remain
distinguishable from the exact sample plane.

To refit a candidate using the installed source and lens preset:

```powershell
.venv/Scripts/python.exe scripts/calibrate_nanoprobe_corrector.py --rays 1000 --step 0.05 --target 25 --keep-convergence
```

This writes a report, ray data and comparison plot under
`outputs/probe_calibration`; it does not overwrite instrument configurations.
Acceptance requires final focus and convergence checks as well as step
refinement. `--skip-focus` saves diagnostic candidates only. The field fit
minimises nonlinear position errors, while C3 independently restores sample
focus. Current screenshots provide no OEM current-to-field conversion.

The 12–240 µm Direct Alignment warm-start path has been recomputed for the
relocated aperture with an approximate 0.25 mrad/µm reference. It remains a
starting estimate followed by physical validation. In particular, requesting
60 mrad with a 240 µm aperture can produce roughly 55.6 mrad within the existing
8% convergence tolerance. The default hexapole calibration does not guarantee
a round probe after changing convergence, source, voltage, aperture or relay
lenses; those working points require their own correction.

## Integration consistency

Post-gun propagation advances canonical transverse momenta with RK4 and
evaluates the continuous magnetic and multipole fields at the actual interval
nodes and midpoints. This removes numerical differentiation of Bz from the ray
equations. Thin lens and aberration actions occur at their physical planes.
NumPy, Numba and CUDA use the same step formula; public results retain physical
ray slopes. A finer step now verifies the fitted optics instead of moving the
focus by a large discretisation error.
