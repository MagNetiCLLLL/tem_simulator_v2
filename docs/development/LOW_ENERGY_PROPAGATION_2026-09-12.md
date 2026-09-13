# Low-energy propagation and material-grid retry

Status: **development operators, not physical-tip STEM acceptance**.

Later specimen-stage update: [non-circular potential propagation](GALERKIN_SPECIMEN_2026-09-12.md)
resolves the two atomistic execution failures described below with a new
numerical method. This note and its failed runs retain the earlier evidence;
the low-energy gun integration remains incomplete.

## Preserved source and instrument

The user's choice is to preserve the current source parameters and extend
propagation, not retune the source into a paraxial acceptance domain. No
profile, default source value, gun component, or production admission gate
was changed in this step.

Current default values read from `default_state()` on 2026-09-12:

| Input | Value |
| --- | --- |
| Physical tip radius | 100 nm |
| Emission-region intensity FWHM | 5 nm |
| Emission kinetic energy | 0.3 eV |
| Energy-spread FWHM | 0.3 eV |
| Minimum kinetic energy | 0.01 eV |
| Young decay width / Boersch sigma | 0.2 eV / 0.1 eV |
| Extractor voltage | 4 kV |
| Extractor field transition | 0.05 to 6 mm |

The legacy default has no selected `TipCoherence` law. Selecting the existing
Gaussian--Schell tip law still does not admit full imaging with these values.
Historical results and the existing classical gun remain available.

## Added low-energy boundary operator

`src/temsim/physics/low_energy_boundary.py` solves the stationary,
nonrelativistic scalar equation for a **transversely uniform, layered real
electrostatic potential**. Each transverse Fourier component satisfies

\[
\psi''(z)+k_z^2\psi(z)=0,\qquad
k_z^2=2m_e e E_{\rm kin}(z)/\hbar^2-|k_\perp|^2.
\]

Energies are in eV, lengths in metres, and wavenumbers in rad/m. They include
the already specified electrostatic energy gain; they are not new source
parameters. The right boundary is outgoing, or decaying if evanescent. The
left boundary specifies the **total complex boundary field**. The operator
solves its normal derivative; it does not assume a forward-only launch.

For right admittance `Y = psi'/psi` and a layer of width `d`, define
`c = cos(k_z*d)` and `s = sin(k_z*d)/k_z`. Exact backward transfer is

\[
Y_L=\frac{Y_R c+k_z^2 s}{c-Y_R s},\qquad
\frac{\psi_R}{\psi_L}=\frac{1}{c-Y_R s}.
\]

Exponentially scaled trigonometric functions avoid overflow in thick
forbidden layers. The exact `k_z=0` limit uses `s=d`. A factored difference
of squares avoids catastrophic cancellation at grazing incidence. An
admittance pole is reported explicitly; a two-component nodal chart is not
implemented. Subdivision of a smooth potential must be converged separately.

The returned boundary field, derivative, complex logarithmic transfer, and
current retain reflection and evanescent coupling. Current is

\[
j_z=(\hbar/m_e)\operatorname{Im}(\psi^*\partial_z\psi),
\]

not `|psi|^2`. A finite evanescent entrance region can couple to a propagating
exit. Its boundary current must not be assigned zero merely because its
local `k_z` is imaginary. No component is clipped or renormalised. Underflow
of extremely small transmission is accompanied by its retained logarithm.
The logarithm's imaginary part is a branch-dependent complex phase, not an
unwrapped classical action or arrival-time reference.

The nonrelativistic kernel rejects energies above 1 keV. This is an explicit
development scope limit, not a relativistic accelerating-gun solution. It
does **not** execute radial focusing, magnetic gun operations, apertures,
curved emitter boundary/tunnelling physics, or spin. It is therefore **not
wired as a substitute for the full gun**, and creates no gun-exit cache or
independently configurable downstream beam.

## Connected material-stage retry

The existing zero-loss and conditional-inelastic specimen adapters now retry
typed sampling failures from the same executed specimen-entrance checkpoint.

- The wave envelope is Fourier-embedded at fixed period, without fitting a
  new source or discarding represented frequencies.
- The potential is regenerated from the same physical specimen on a finer
  lattice. An interpolated coarse potential is not treated as new detail.
- The original calculation window, specimen orientation, source state,
  phonon seed, and conditional-trajectory random counters stay unchanged.
- Budgets and cancellation are checked before the next potential allocation.
  Unsupported physics is not caught as a numerical retry.
- Partial inelastic slices bind the actual potential arrays, coordinates,
  slice thicknesses, and refinement factor. Coarse slices cannot contaminate
  a fine-grid retry. Failed final writers remain unpublished; completed
  slices and upstream checkpoints are not deleted.

This is sampling-error recovery, **not independent potential convergence**.
The existing strict bandwidth guard is unchanged. Two actual atomistic
operator fixtures still reach the potential builder's 8 GiB work limit at
4096 pixels. Raising this limit alone is not established as a solution.
The retry mechanism does not make those fixtures, or the full image, pass.

## Evidence and remaining work

All checks are offline. No real microscope acquisition or GPU throughput
claim was made. The evidence directories contain input hashes and raw test
outputs; failed runs are retained.

- `evidence/20260912T135239Z-low-energy-boundary-35807d19/`: 43 passed,
  4 failed. Two newly introduced kernel-check issues were subsequently
  corrected (grazing-incidence cancellation and insufficient field-layer
  subdivision). The two atomistic-grid failures remain unresolved.
- `evidence/20260912T135612Z-boundary-refinement-749d4d29/`: 92 passed.
  Includes analytical uniform-medium/step checks, evanescent coupling,
  current conservation, independent adaptive-ODE comparison and convergence,
  numerical retry contracts, and unchanged source-admission constraints.
- `evidence/20260912T140349Z-low-energy-validated-493d5c98/`: 111 passed,
  0 failed, 0 skipped. Includes finite-barrier tunnelling against its analytic
  transmission probability and an inelastic retry after a coarse slice has
  already committed. Repeated execution preserves the same mode fields,
  energies, and random histories; final cache reuse avoids preparation.
- `evidence/20260912T140650Z-material-compatibility-c2816c5a/`: the two
  actual atomistic operator fixtures were rerun against the final code;
  both still fail at the existing 8 GiB potential-builder work limit. These
  failures are not skipped, retuned, or counted among the 111 passing checks.

The full existing FEG field was inspected, not only its extractor component.
Its on-axis potential and electric field are zero from the tip through
`z=0.05 mm` at the current defaults. At `z=0.1 mm` its potential gain is
approximately `0.0234384 V`, with axial electric field `-1400.36 V/m`.
The radial extension is quadratic in radius. This is not a resolved curved
tip extraction field, even though the tip radius and work function exist
as parameters. The new layered operator cannot supply that missing field.

The next physical integration needs a jointly specified curved tip/extractor
boundary model and transverse, nonparaxial accelerating-wave transport,
including all existing gun operations. Its boundary field/current law must
be compatible with the retained energy distribution. Matching into an
accelerated paraxial stage is allowed only after actual upstream execution
and domain checks. A label, normalised angular histogram, or ray-fitted pupil
does not meet this requirement. Full dynamic-scan/inelastic/detector imaging
and convergence evidence are still required before opening production STEM.

## Reference

Krecinic and Ernstorfer, *Wave-Mechanical Electron-Optical Modeling of
Field-Emission Electron Sources*, Physical Review Applied 15, 064031 (2021),
[doi:10.1103/PhysRevApplied.15.064031](https://doi.org/10.1103/PhysRevApplied.15.064031).
The paper motivates treating curved emission surfaces and immediate
inhomogeneous acceleration together. The layered boundary solver above is
an independently derived limited-domain operator, not an implementation or
validation of that paper's complete source model.
