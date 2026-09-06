# Six-stage physics implementation

This work preserves ideal continuous controls. Geometry changes never run lens
preset optimisation implicitly. A completed calculation is not calibration.

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Shared vector-field specimen transport and reference-plane matching | Implemented |
| 2 | Axisymmetric geometry-to-field calculation | Implemented: linear and static B-H materials, supported axisymmetric profiles |
| 3 | Coherent propagation through intermediate aperture planes | Implemented: sampled first-order canonical maps |
| 4 | Exclusive manual / field-derived aberration selection | Implemented: finite-pupil field-ray fit |
| 5 | Model Inspector with evidence and limitations | Implemented |
| 6 | Multi-parameter Design Explorer | Implemented: allowlisted runtime parameters |

All project-facing text is English. Numerical tolerances and unsupported cases
must be explicit; no synthetic calibration or silent replacement of user data.

## Current scope and checks

- The specimen kernel uses registered XYZ fields and continuous quadrupole /
  hexapole fields. Static magnetic flights preserve momentum magnitude. Arc
  chord errors are bounded relative to the existing geometry epsilon. Once a
  ray leaves the finite specimen envelope, downstream transport owns it.
- Geometry fields use P1 axisymmetric A-phi finite elements with explicit
  relative permeability or a static B-H curve and ampere-turns at 100% excitation.
  Passive neighbouring magnetic parts inside the finite domain enter the
  geometry identity. Mesh and remote-boundary convergence are not implied by
  a successful solve; see the later [B-H extension](STATIC_NONLINEAR_MAGNETICS.md).
- The current C1 and C2 assemblies produced finite fields in an offline check.
  The later geometry correction makes Objective material masks use the existing
  TOML upper/lower intervals, removing the old drawing-envelope overlap without
  changing user geometry. Genuine coil/iron overlaps still fail explicitly.
- Intermediate coherent waves use an affine lattice and an analytical
  quadratic phase carrier. Aperture masks remove probability without
  renormalisation. Singular mixed conjugacy and undersampled residual chirps
  are reported as unsupported sampling, not replaced by an intensity preview.
- Field-derived coefficients fit production-ray intercept gradients over a
  finite pupil. No manual coefficient or empirical Cs kick enters the fit.
  C1 is already in first-order transport; Cc is estimated with symmetric energy
  perturbations. Unmapped round lenses cannot provide missing high-order field
  information. Fits are approximate and report residuals, not calibration.
- Design Explorer accepts additional parameter rows and executes the Cartesian
  product (up to 64 points) on detached snapshots. Existing result reuse,
  cancellation, tolerance checks and multivariate sensitivity fitting remain
  authoritative. The live microscope settings are not overwritten.
  TOML-owned geometry is deliberately not a runtime sweep parameter: edit and
  validate the assembly first, then capture the design. Arbitrary geometry
  overlays and automatic geometry optimisation are not implemented here.

## Model controls

Open **Model Inspector**, choose a lens, enter relative permeability and
ampere-turns, then select **Use geometry field**. The next calculation builds
the field. **Use analytic field** explicitly removes the selected map/recipe.
Geometry recipes are serialized in `lens_field_map_descriptors` and reused by
their geometry/settings identity. Changing strength scales a fixed linear
field and does not rebuild the magnetostatic mesh.

Choose **Manual** or **Field-derived** separately for Probe and Image
aberrations. Manual values remain stored when the mode changes, but are not
added to the field-derived wave coefficients.
Field fits are prepared in the calculation worker and retained on the result
snapshot. Image views and diagnostics share those fits; opening an aberration
page never starts a new field-derived solve. A fit residual measures how well
the polynomial represents field-ray errors. It is not the RMS size after an
actual corrector adjustment. Reference/difference columns are therefore hidden
in this mode; a separate before/after field-corrector experiment is not implied.

## Validation scope

Final related regression on 2026-09-05: **302 passed, 17 warnings**, exit code 0,
in 274.58 s, using the project virtual environment and offscreen Qt. The run
covered the six-stage tests, Model Inspector, elastic/axial/local/downstream
transport, 3-D view, design planning/execution/GUI, aberrations, TEM waves,
field providers, vector transport, GUI shell, calculation reuse, manifests /
artifacts and calculation controller. This was not a whole-project preset
recalibration or a desktop OpenGL/GPU validation. A pyqtgraph teardown
disconnect warning was also emitted; no test failed.

Compilation and diff-whitespace checks passed. An English-content scan over
project Python, TOML and Markdown found no remaining Chinese text. Translation
checks retained every original UR/DA ledger ID and every distinct URL in the
historical projector/EDS research; UR-027/028 were appended.

- Analytic checks cover reversible vector-field plane matching, a manufactured
  A-phi solution with mesh refinement, and the centre field of a finite air-core
  solenoid. The latter differs from its analytic radial-average reference by
  about 1.47% on the tested finite domain/grid; this is not a calibrated C1/C2 result.
- Coherent-wave checks cover identity/Fourier maps, absolute probability after
  masking, and the physical Camera path with an open or fully closed Objective
  aperture. A closed aperture gives zero signal, without renormalisation.
- Field-fit checks recover known Cs/A2 gradients and exercise production vector
  rays; cache/UI checks prevent duplicate fits and false correction claims.
- Geometry, sample/EDS/downstream, field providers, waves, design sweeps, result
  cache/manifests, controllers and offscreen GUI are included in related regressions.
- Mesh/domain convergence for a particular installed lens, arbitrary 3-D pole
  geometry, measured B-H data, experimental calibration and desktop OpenGL/GPU
  behaviour are not established by these offline tests. Finite wave-grid coverage
  and aperture-edge resolution still require convergence checks for each design.
- Preset strengths were not optimised. Existing manual settings and mechanical
  geometry were not changed to force model validity.

The subsequent [magnetic-circuit stage](MAGNETIC_CIRCUIT_MODELS.md) adds explicit
material sharing, channel-owned coils, shared radial profiles and fixed-point
guards for saturation-dependent maps. The subsequent static B-H extension adds
single-valued nonlinear material response, not hysteresis or multiphysics.

The current solver implementation identity is `temsim-solver-2026-09-static-bh-v1`
following the [static B-H extension](STATIC_NONLINEAR_MAGNETICS.md).
Older incompatible disk seeds are rejected. Within the current identity,
dependency-scoped in-memory and persistent reuse remains active.

## Numerical references

The model uses the conventional axisymmetric vector-potential weak form, not
FEMM's specialised modified-flux element. See the
[FEMM magnetostatics manual](https://www.femm.info/doku/lib/exe/fetch.php?media=upload%3Afiles%3Amanual.pdf)
for magnetic boundary conditions and the distinction between linear material
inputs and measured nonlinear B-H curves. Sparse linear systems use
[SciPy spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
