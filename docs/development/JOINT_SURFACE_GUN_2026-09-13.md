# Joint physical-tip / grounded-gun development

Status: **TEM and STEM images have not been generated or qualified by this
work. Production source admission remains closed.** Actual three-energy
coherent gun calculations now execute, but their output is not yet converged
in radial basis, axial step, electrode mesh and domain. Current conservation
alone is not an imaging-accuracy certificate.

## Physical inputs and boundaries

The diagnostic explicitly selects the existing coherent curved-cap reservoir:
100 nm apex radius, 10 degree emitting cap, 100 nA incoming reservoir current,
0.3 eV mean and 0.1 eV RMS energy. The existing reference assembly, 4 kV
extraction, electrostatic gun lens, 300 kV acceleration, bores and apertures
are retained. No default TOML, historical profile, lens strength or source
width is retuned. Voltages remain relative to the final grounded anode.

`build_surface_gun_checkpoint` solves the tip-adjacent finite-element field
jointly with the downstream two-way load. The top trace is an unknown of this
joint solve, not a separately specified exit source or a fitted ray field.
Its result carries the actual gun/source inputs, shared-column snapshot,
implementation identity, numerical settings, and every energy mode's complex
field and axial phase reference. Numerical Laguerre-frame widths are coordinate
choices, not configurable downstream source parameters.

The new gun model solves scalar stationary relativistic orbital propagation.
It retains extraction, acceleration, round electrostatic focusing, the
consumed shared-column axial magnetic field and installed gun masks.
Unsupported non-axisymmetric gun fields, Wien/slit optics and offset gun
apertures fail explicitly. Spin, self-consistent space charge and material
tunnelling are not implemented by this scalar model.

## Numerical changes

- Quadratic body-fitted axisymmetric finite elements are available for the
  curved near field. The original linear-element path remains available.
- The full grounded potential is integrated in a moving radial Laguerre
  basis. Analytic aperture moments and piecewise `r^2` potential moments now
  replace repeated quadrature. The relativistic coordinate products retain
  two extra modes before projection. Independent positive quadrature remains
  a reference; no quadratic potential substitution is used for the wave.
- The moving-basis derivative includes its finite-subspace closure term.
  Forward and backward complex channels use simultaneous covariant scattering
  operators, with current-metric orthogonality and separately evaluated small
  transverse phases.
- Potential interpolation is even in radius (`r^2`), removing the unphysical
  axial cusp of a linear-radius interpolant. The electric field is the
  derivative of the same interpolated potential. Electrode-grid convergence
  remains a separate requirement.
- Analytic aperture integration fixes false attenuation from underintegrating
  a wide clear bore. Finite-basis mask losses still
  include unresolved modes; the recorded projection-closure defect makes
  this limitation explicit.
- Roundoff-equivalent axial nodes are merged within eight ULPs. Each physical
  mask plane is scheduled once; coincident physical components are all kept.
- The rapidly varying longitudinal carrier now uses an exact Liouville
  coordinate change, **not a WKB truncation**. Both its scalar correction and
  derivative jumps at potential knots are retained. See the general
  [change-of-variable identities in NIST DLMF](https://dlmf.nist.gov/1.13#iv).
- The curved finite-element domain can retain the radial carrier analytically
  as `psi = exp(i*gamma*r^2/2)*u`, using the executed radial frame's `gamma`.
  Both covariant-gradient terms are retained in the weak equation. The
  physical source boundary is transformed by the same phase, and the full
  complex wave is restored at output. This is a numerical coordinate change,
  not an independently specified source or fitted optical correction.
  The interface Gram defect is recorded; neither it nor the wave is
  normalized to force a match. Direct phase coordinates remain selectable
  for independent comparisons.
- The joined face is checked against the full analytic radial basis before
  executing the gun. A basis extending beyond that face is rejected, not
  normalized or treated as absorption. Its numerical coordinate width can be
  independently refined; this never changes the physical emitting cap.

For `D_z^2 f + (k^2 I + H) f = 0`, set `alpha = k/k_ref`, `ds/dz = alpha`,
and `f = alpha^(-1/2) u`. The transformed connection is `G_s = G_z/alpha`,
and the residual is `(H + a''/a I)/alpha^2`, where `a = alpha^(-1/2)`.
For `q = k^2`, `a''/a = 5(q'/q)^2/16 - q''/(4q)`. Changes in `q'` at
potential knots require real derivative-jump interfaces. Omitting either
term would change the equation. The inverse transformation is applied to
both amplitude and covariant derivative before exporting physical flux.

## State preservation and limits

The physical axial carrier is held in `AxialWaveReference` and removed from
the transverse envelope exactly once. There is no aggregate phase for the
incoherent energy mixture. The radial and Cartesian gun-exit representations
describe the same executed mode; their integrated probabilities are checked.

Near-field side escape is a numerical-domain channel, **not absorption**.
The full-state pipeline refuses to discard a significant side channel.
Standalone diagnostics retain the near complex arrays alongside the gun
arrays. They are not automatically admitted as full-image source caches.

`radial_column_wave` is an experimental full-complex axisymmetric reduction
that retains spherical aberration and physical masks. Its domain/sampling
guards exposed a refinement error near C1: interpolating a previously clipped
field creates Gibbs tails. Refinement now re-evaluates the executed gun
expansion and replays all intervening operations, including hard stops. Its
loss ledger is replaced with that of the refined execution. The actual
reference calculation passes C1, but the full column remains unqualified.
It is not yet connected as the production replacement for the Cartesian
column operator. Extending a grid does not recover missing physical detail.

Float64/complex128 CPU and explicitly selected CUDA FFTLog backends now share
the same transform and unchanged sampling guards. The bounded cache contains
numerical Mellin multipliers only. CUDA unavailability or insufficient device
memory is explicit, never a silent approximation or fallback. See the
[CuPy FFT API](https://docs.cupy.dev/en/stable/reference/fft.html). GPU kernel
agreement and partial-prefix timing are not full-image GPU acceptance.

## Evidence and reproducibility

`scripts/inspect_surface_column.py` executes the actual reference gun and
writes an exclusive new evidence directory. It can stop at the gun or attempt
the round column prefix. `scripts/compare_surface_gun.py` compares matching
energy modes in physical cylindrical coordinates without fitting phase,
curvature, scale or source parameters. Neither script claims an image.

Current evidence includes:

- `20260913T001551Z-grounded-axis-regularity-c491194b`: 66 scoped passes.
- `20260913T003323Z-joint-round-events-e91239eb`: 55 scoped passes.
- `20260913T003736Z-liouville-coupled-ramp-1bb6d4c2`: original Airy-refinement
  failure retained (512 intervals did not meet the unchanged 2e-6 target).
- `20260913T003810Z-liouville-ramp-converged-2a39face`: 48 scoped passes.
  Independent Airy ramp comparison reaches approximately 9.85e-7 absolute
  complex error at 1024 intervals and shows second-order convergence.
  A free moving-frame calculation separately converges against a direct
  nonparaxial Bessel angular-spectrum integral, without fitted phase.
- `20260913-liouville-gun16`: executed gun, approximately 48.6734 nA exit
  current and 6.64e-11 unresolved side fraction. This is not a calibrated
  efficiency or converged beam.
- `20260913-liouville-gun32` and `20260913-liouville-axial16`: output-current
  changes remain approximately 15.2% and 9.31%, respectively; full complex
  changes remain large. These are **failed convergence comparisons**, not
  evidence of successful TEM/STEM imaging.
- `20260913T011806Z-radial-replay-cuda-99ec616c`: 52 scoped passes, including
  actual CUDA complex-field/roundtrip comparisons and replayed hard stops.
- `20260913-analytic-domain5-64` and `20260913-analytic-domain5-128`:
  unconverged output currents 29.3109 nA and 18.2428 nA. The cylindrical
  complex comparison restores the recorded far-gun carrier and rejects
  mismatched physical or near-phase references; no phase is fitted.
- `20260913-replay-column-prefix` and `20260913-cuda-replay-column-prefix`:
  explicitly interrupted partial runs, with reasons and existing NPZ arrays
  retained. Both passed C1. Reaching Z = 546.5 mm took 348.03 s on CPU and
  138.13 s with CUDA in these partial runs, including gun execution/replay.
  Neither produced a complete prefix checkpoint or an image.
- `20260913T012511Z-radial-phase-coarse-69384848`: original 65-node strong
  phase-coordinate analytic failure retained (7.47e-4 versus 5e-4 target).
- `20260913T012526Z-radial-phase-refined-43e2e774`: 34 scoped passes after
  refining to 129 nodes, without relaxing the error target. Includes exact
  plane-wave comparisons, source/cache compatibility, and actual CUDA tests.
- `20260913-radial-phase-gun16` and `20260913-radial-phase-gun64`:
  complete three-energy gun diagnostics after phase factoring. The 16-mode
  interface Gram defect is 0.002788 (previous direct-coordinate defect about
  0.18); the 64-mode defect is still 0.026779. Exit currents are 48.7395 nA and
  29.2748 nA: **full gun convergence is still not established**.
- `20260913T013325Z-radial-phase-domain-1ced15c0`: 51 scoped passes after
  phase-coordinate, analytic-integral and compatibility changes. This does
  not replace a complete image acceptance run.
- `20260913-fine-interface-coordinate02` and
  `20260913-fine-interface-coordinate01`: 769 radial by 257 axial near-field
  nodes, 64 radial gun modes, identical physical inputs. Changing only the
  numerical coordinate width/cap-radius ratio from 0.2 to 0.1 changes output
  current from 29.2522 nA to 58.1572 nA and produces a relative complex L2
  difference of 1.239. These are **failed coordinate-convergence checks**.
  The width-0.2 interface Gram defect improves to 0.001904, so fixing the
  interface alone does not establish full-gun accuracy. Both reports retain
  the complete instrument snapshot and gun record.
- `20260913T014348Z-joint-phase-replay-final-127830b7`: 106 scoped passes,
  zero failures and zero skips, with input hashes unchanged throughout the
  run. Includes the actual CUDA kernel checks. Subsequent serial compilation
  of `src`, `scripts` and `tests` completed. No full-suite or image acceptance
  is implied.

Older attempts, including interrupted large radial grids, remain in their
separate evidence directories. No test failures have been deleted, no source
admission gate relaxed, and no full-suite, full-chain GPU or hardware
validation is claimed.

## Remaining work

1. Converge the executed gun's full complex output, including aperture basis
   closure, independent electrode and wave grids, and numerical boundaries.
   The global finite Laguerre expansion remains strongly dependent on its
   coordinate choice after physical gun apertures. Do not use the improved
   near-field interface, conserved current, an Ideal Optics selection or a
   successful isolated test to admit this unconverged state as image input.
2. Resolve high-dynamic-range column propagation without deleting Cs, clipping
   unsupported waves or replacing them with a fitted source.
3. Carry this same executed state through real CIF elastic/inelastic specimen
   transport, timed scan coils, downstream lenses and physical detectors.
4. Produce actual TEM and multichannel STEM images, verify their convergence
   and detector ordering, then connect the qualified route to shared GUI
   calculation/cache controls. A detector before the energy filter does not
   traverse it; a requested path reaching the filter must not bypass it.
