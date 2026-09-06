# Static nonlinear lens fields

Implementation checkpoint: 2026-09-05. This is a reference-material design
model, not a reconstruction or calibration of a Titan magnetic circuit.

## Reference data

The first library entry is **Pure iron - FEMM reference**, with 21 B/H pairs
copied numerically from the `Pure Iron` block of the pinned FEMM library:

- [Material data, pinned mirror revision](https://raw.githubusercontent.com/cenit/FEMM/7d9e8ed90772ddb939a3256f976e231bcb5cca97/bin/matlib.dat)
- Source-file SHA-256: `ba0ca80ffae56440a9ad785c9089b0e7f63b45e1c7a73dd5c0bdc338beeb63b7`.
- [FEMM soft-magnetic reference notes](https://www.femm.info/doku/doku.php?id=softmagneticmaterials)
  explain the handbook provenance of its scanned-curve collection. The direct
  scanned ZIP was unavailable during this work; the actual data used here are
  identified by the pinned library block, not an assumed scan equivalence.
- [FEMM numerical cautions](https://www.femm.info/doku/doku.php?id=FAQ)
  distinguish interpolation, extrapolation and finite-boundary limitations.

Only the numerical B/H facts are included; no FEMM solver code or proprietary
material database is incorporated. Other material properties from the library
are not inferred, copied into a loss model, or used as a Titan assignment.
These data have no verified batch, heat treatment or temperature certificate
for this instrument. It is now the generic material default for unconfigured
Model Inspector drafts; existing recipes are not rewritten and application
still requires **Use geometry field**. See [lens material defaults](LENS_MATERIAL_DEFAULTS.md).

The bundled table is `configs/materials/magnetic/femm_pure_iron.toml`.
Values use tesla and A/m. There is no amplitude rescaling. Our constitutive
interpolation is piecewise-linear **H(B)**, not FEMM's spline interpolation;
agreement at the tabulated points does not certify identical solver outputs.

User CSV imports require exactly `B_T,H_A_per_m`, including (0,0), followed by
strictly increasing, finite values. The full validated curve, source URI and
SHA-256 are embedded in the recipe. Editing/deleting the external CSV does not
silently alter an existing result; explicitly re-import it to change the model.

## User workflow

1. Open **Simulation > Configure lens models...**.
2. Select a round lens, then **Nonlinear B-H** and a reference material, or
   **Import B-H CSV...**. Enter its total positive ampere-turns at 100%.
3. Choose mesh/boundary settings and **Use geometry field**. This saves the
   recipe and selects Custom; it does not solve a field or calculate a preset.
4. Configure the other channels in a shared magnetic circuit, including disabled
   coils. All B-H channels currently use one common material/mesh/solver setup;
   their ampere-turns and operating percentages/polarities may differ.
5. Calculate explicitly. The global **Nonlinear Material Field** tier becomes
   available when every enabled column round lens has a valid B-H recipe.

The menu's readiness check is not a geometry, domain-interaction or convergence
certificate. Active foreign coils inside a nonlinear solve domain are rejected:
include them in the joint B-H solve or explicitly disable them. The UI does not
change their enabled state on the user's behalf. Linear/analytic/native-map
mixing is not silently treated as a saturated compound-lens solution.

Advanced recipes can use `material_bh_overrides` by the physical part's
`material_class`; otherwise the explicitly chosen reference curve applies to
magnetic bodies. Non-magnetic space remains vacuum permeability. Linear recipes
retain `material_permeabilities`. Conflicting overlapping material assignments
fail rather than using whichever part happened to be visited last.

## Field and optical scope

- Static, isotropic, single-valued B-H magnetostatics in an axisymmetric R-Z
  domain. No hysteresis, remanence, eddy currents, temperature or deformation.
- All configured B-H channels are solved at their **complete current vector**.
  Nonlinear saturated fields are not computed separately and then added.
- One enabled control entry carries the joint field; the remaining entries
  contribute zero additional field. The inspector identifies the joint owner.
  These are bookkeeping roles, not claims that the other lenses produce no
  magnetization, nor independently measurable per-lens field decompositions.
- Registered vector-field transport consumes the total field once. Existing
  mapped-lens guards suppress empirical Cs kicks for those channels. Generated
  linear/nonlinear Objective fields also suppress the additional empirical ray
  Cc kick: particle momentum already controls their energy-dependent focusing.
  Saved aberration options are retained for other modes. Wave-image
  Manual/Field-derived aberration selection remains explicit; it is not changed
  automatically and does not add a second geometrical Cs term to the rays.
- Axisymmetric fields cannot represent arbitrary off-axis/asymmetric iron,
  general 3-D winding geometry or nonlinear multipole-iron feedback. The gun and
  dedicated Energy Filter keep their separate models. Coupled Multiphysics stays
  unavailable; opening its menu entry does not run an approximation substitute.

## Geometry corrections

The existing `tapered_bore_pole` shape is now supported by the material mask,
with its existing taper length, bore and outer/tip dimensions unchanged.
Objective coil/yoke masks now use the same TOML-defined upper/lower material
intervals as Physical Layout. The coil ampere-turn normalization uses the total
actual winding cross-section of both intervals, not the old drawing envelope.
True coil/iron overlaps still fail. No housing, pole dimensions, source strength
or preset was edited to make a solve succeed.

## Numerical method and validation

The P1 A-phi weak form integrates `H(B) . curl(v) r dA = J_phi v r dA`;
`Br = -dA/dz`, `Bz = dA/dr + A/r`. Three-point triangle quadrature and centroid
material assignment are explicit discretization choices. The tangent is
`nu I + (dH/dB - nu) b_hat b_hat^T`, including the finite origin limit.
Damped Newton uses backtracking and a relative assembled-residual criterion.
Axis and remote boundaries have A=0. No result is returned on nonconvergence.

Newton trial steps may temporarily use a vacuum-slope extension of H(B) to
avoid undefined iterates. Every accepted final solution is checked against each
material's supplied B range at quadrature points; an out-of-range solution is
rejected, not clamped or extrapolated. The included curve ends at 2.56 T.
This range check is distinct from mesh and boundary convergence.

`tests/test_nonlinear_magnetostatics.py` covers the linear limit, a manufactured
nonlinear potential under mesh refinement, reference-point reproduction,
polarity reversal, zero current, saturation/non-superposition, failure guards,
joint-field non-duplication, frozen plans, changing-current cache identity,
physical masks, reference import and offscreen menu/state behavior. Existing
finite-solenoid and linear refinement tests still apply to the shared FEM core.
There is no claimed full TEAM-13, EOD/MEBS/FEMM comparison, measured-material
validation, GPU nonlinear solver or arbitrary-3D verification in this stage.

## Cache

The later [Field validation workflow](MAGNETIC_FIELD_VALIDATION.md) adds detached
mesh/boundary comparisons and cached R-Z material/flux views. It does not imply
that a particular installed lens, high-order aberrations or an external FEMM
comparison has passed. Numerical validation is explicitly requested per study.

Bounded prepared meshes are reused independently of current. Completed joint
maps are keyed by physical geometry, all channel excitations, full material
snapshots and numerical settings. Returning to a retained operating point can
reuse its map. A new current does not scale an old nonlinear field; stale live
providers reject changed operating points while frozen propagation plans retain
their original immutable fields. Existing stage/result caches remain intact.
The persisted solver identity is `temsim-solver-2026-09-static-bh-v1`; previous
solver seeds are rejected without deleting saved data.
