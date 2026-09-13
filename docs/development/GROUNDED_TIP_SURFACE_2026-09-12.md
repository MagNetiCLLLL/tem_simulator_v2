# Grounded cold-FEG surface reference

Status: implemented classical reference model; **not complete coherent imaging**.

The user authorised a new source parameterisation on 2026-09-12, including
replacement of the old imposed source widths/energies. This supersedes the
earlier requirement to keep those numbers in a newly designed source. No
historical profile is silently converted. No independently adjustable gun-exit
or specimen-plane source has been introduced.

## Use

1. Open **Model Inspector → FEG tip emission…**.
2. Select **Grounded tip surface model (reference)**.
3. Adjust the four surface inputs and apply. Existing extractor and high-tension
   controls set electrode voltages; there is no second extraction-voltage box.
4. Save the operating profile to retain the selected model and its parameters.

Reference geometry lives in `configs/sources/cold_feg_tip.toml`; both FEG module
TOMLs point to that one file. Edit it and use **Reload reference TOML** to capture
a new draft explicitly. Applying a draft does not retune the column. The initial
reference is a 100 nm spherical apex tangent to a 5-degree cone with a 1 mm shank.
Only the approximate apex radius and W(310) material have the listed manufacturer
reference; cone/shank dimensions, current, patch and distributions are explicit
uncalibrated model choices, not OEM specifications.

The historical source editor is hidden while the new model is selected. Its
independent FWHM, angular cutoff, Young and Boersch controls do not enter the
surface calculation. Its original parameters/results remain readable. Merely
opening the editor does not change the active model.

## Voltage convention and fields

- Final accelerating-anode metal and downstream numerical boundary: **0 V**.
- Tip: `-high_tension_kv * 1000` V.
- Extractor: `tip_voltage + extraction_voltage`.
- Reference gun-lens annulus: `extractor_voltage + lens_bias_voltage`.
- Accelerator ring potentials interpolate from extractor to ground using the
  existing stage voltage fractions. At the final fraction of one the metal is
  exactly grounded. No source kinetic energy is subtracted from these voltages.

The installed extractor, gun-lens and accelerator geometry defines annular
Dirichlet conductors. The tip is a spherical-cap/tangent-cone conductor.
A nonuniform axisymmetric vacuum Laplace solve includes their mutual electric
influence. It no longer imposes a field-free 50 micrometre region after the tip,
or an independently adjustable start/end of the extraction field. The old gun
lens `potential_scale` is not used. Electrode movement changes the boundary
problem rather than sliding an artificial potential ramp.

The radial and upstream numerical boundaries are insulating (zero normal field),
not physical grounded enclosures. The downstream truncation is a grounded plane.
These explicit boundaries and idealised annuli require sensitivity checks before
instrument calibration. In particular, mechanical outer diameters do not prove
the true electrical contour of a commercial gun. Space charge is neglected;
there is no self-consistent emission current or image-charge/tunnelling barrier.

The sparse cylindrical finite-volume solve shortens metal-intersecting stencil
edges to the analytic cap/cone surface. Potential interpolation and its gradient
share the same bilinear representation. Field arrays are cached by the consumed
geometry, voltage and numerical inputs, separately from emission/trajectory data.
Changing only current or initial energy reuses the electrostatic solve, but not
the emitted beam/trajectory result. Changing an electrode position or voltage
invalidates the field cache. Its bounded eight-entry cache contains executed
solutions, not parameter-labelled substitute beams.

## Surface input contract

| Editable input | Meaning | Related outputs, not additional inputs |
| --- | --- | --- |
| Surface current (nA) | Prescribed total outgoing flux | Current density = current / patch area; downstream current includes losses |
| Emission cap half-angle (deg) | Uniform-area emitting patch on the spherical cap | Patch dimensions from the TOML apex radius; not a virtual-source size |
| Mean normal kinetic energy (eV) | Positive exponential distribution along the local normal | Total mean and spectral RMS |
| Mean tangential kinetic energy (eV) | Exponential transverse energy, uniform azimuth | Angular distribution about the local normal |

Normal and tangential energies are independent in this **declared reference
law**, not universal properties of field emission. Their sum is total local
kinetic energy. Mean energy is `mean_normal + mean_tangential`; energy RMS is
`sqrt(mean_normal**2 + mean_tangential**2)`. RMS is deliberately not labelled
FWHM for this non-Gaussian spectrum. Direction follows from the energy split and
surface normal; it cannot be independently assigned an incompatible angular
cutoff. The reference 0.2/0.1 eV component means are examples, not an inference
from the commonly quoted 0.3 eV cold-FEG spectral width.

Emission is an outgoing boundary condition, not a Fowler-Nordheim current
prediction. Its numerical surface is represented on the solved metal grid; the
inward mesh displacement is recorded and must remain below 10% of apex radius.
That is a coarse geometry guard, not a converged source claim. Surface shape,
emitted distribution and field require refinement together.
Returning rays are absorbed by this same discretised tip metal, not allowed
to pass through the conductor. The launch boundary itself is not an absorber.

The distribution defines **classical flux, not complex amplitudes**. A necessary
emittance/uncertainty check is not a phase reconstruction. A coherent request
with this model fails explicitly before any old Gaussian or downstream source
can be substituted. Coherent surface emission, tunnelling/near-field matching,
and its complete column/specimen/scanned-detector wave chain remain open.
The separate analytical EELS readout currently assumes Gaussian source width.
It rejects this new non-Gaussian source explicitly instead of applying the
inactive historical FWHM or treating an RMS-derived width as its full spectrum.
Connecting the transported spectrum through the physical energy filter remains
later work; the historical EELS workflow remains unchanged.

## Executed trajectories and numerical validation

The selected model is connected to the existing gun tracing/column source entry.
It retains the gun magnetic deflectors/stigmator, installed Wien fields and
apertures, body bores, DPA and exit/C1 stop. Curved-surface coordinates and full
directions are retained rather than treating all emission as a z=0 disk.

A relativistic, symmetric discrete-gradient Lorentz update preserves `K-e*phi`
to its nonlinear residual; the magnetic force does no work. Step doubling
controls trajectory error. No momentum normalisation to an expected energy is
performed. The exit event is re-integrated to the physical plane; interpolating
momentum across the end of the electric field would lose part of its work.
The exit-energy acceptance budget is 0.001 eV, independent of the result.

Energy conservation alone does not certify the field shape, orbit, emission
model or images. Tests also cover an analytic graded-grid Laplace solution,
potential/field consistency, geometry boundaries, flux and energy statistics,
cache invalidation, profile/snapshot round trips, GUI isolation, and independent
DOP853/step convergence for the particle integrator.

Reproduce the reference field refinements and a nine-ray gun trace:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_grounded_tip.py --output <new-directory> --render-editor
```

The output includes a loadable profile, actual cached ray arrays and a JSON
report. No STEM/TEM specimen image is generated or claimed by this script.

### Recorded results

- [Focused regression report](evidence/20260912T213504Z-grounded-tip-final-b8db24ce/report.json):
  75 passed, zero failed/skipped. Covers the new surface, field, integrator,
  profile/snapshot and UI changes and adjacent source-admission checks.
- [Historical-source compatibility report](evidence/20260912T213707Z-grounded-tip-legacy-compat-8a295b42/report.json):
  18 passed, zero failed/skipped, including original gun timing and historical
  effective-source storage/admission behaviour.
- [Readout and tuning guard report](evidence/20260912T214033Z-grounded-tip-readout-guards-92a800f2/report.json):
  nine passed, zero failed/skipped. The original analytical EELS tests remain
  green; the new surface cannot use its old Gaussian-width fallback, and medium
  tuning does not replace surface rays with historical Gaussian support probes.
- [Default assembled-gun reference](evidence/20260912-grounded-tip-final-reference/report.json):
  nine emitted rays, four passed the exit aperture, 100 nA emitted and 44.44 nA
  transmitted in this very small diagnostic bundle; about 29.54 s on CPU.
  Maximum exit work-energy error was 3.03e-9 eV against the independent 0.001 eV
  budget. This is not a converged transmission estimate or speed benchmark.
- The field report retains 1x/2x/4x grid results. Apex field changes by about
  1.3% from 2x to 4x; the default mesh's maximum source displacement is 4.86 nm.
  Neither mesh independence nor an OEM electrical geometry is certified.
- [Loadable reference profile](evidence/20260912-grounded-tip-final-reference/grounded-tip-profile.toml)
  and [offscreen editor rendering](evidence/20260912-grounded-tip-final-reference/tip-editor.png).
  Loading the profile is an explicit choice and does not tune the column.

The earlier [broader run](evidence/20260912T212028Z-grounded-tip-regressions-00a96f0f/pytest.log)
was **not green**: five historical coherent-gun tests failed at the existing
paraxial source-domain gate; one profile-version assertion still expected 7.
The assertion now expects the intentional version 8 and passes above. The
five coherent failures were not removed, relaxed, or made to pass by changing
the emission parameters. They remain unresolved; the focused green reports
do not constitute full coherent TEM/STEM acceptance. The full project suite,
GPU execution, physical microscope comparison and full-column new-source
image generation were not run in this change.

## Sources

- [JEOL field-emission gun](https://www.jeol.com/words/semterms/20121024.062458.php):
  tungsten tip, approximate apex radius, extraction/acceleration distinction.
- [JEOL cold field emission](https://www.jeol.com/words/emterms/20121023.105957.php):
  tunnelling and energy-width terminology; not justification for a launch-energy mean.
- [Gonzalez, numerical methods and original papers](https://web.ma.utexas.edu/users/og/numerics.html):
  discrete-gradient conservation framework.
- [Relativistic charged-particle dynamics](https://www.unige.ch/~hairer/preprints/relcpd.pdf):
  independent discussion of relativistic energy-conserving discretisation.
