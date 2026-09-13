# Transversely coupled low-energy boundary propagation

Status: **development electric-field operator; full physical-tip imaging is not complete**.

## What changed

The earlier stratified solver treated each transverse frequency independently.
`physics/coupled_low_energy.py` now solves a coupled transverse boundary problem,
including reflection, evanescent components and their conversion between
channels. `physics/electrostatic_wave_channels.py` builds its electric-potential
matrices from the installed FEG's existing field providers and geometry.

The 5 nm emission-region FWHM, 0.3 eV emission energy, existing energy
distribution, components, operating profiles and production source guards
are unchanged. No gun-exit or specimen-plane source was added. Neither new
operator publishes a `BeamState` or `TipGunCheckpoint`, nor is either used as
a shortcut around the existing gun in TEM/STEM imaging.

## Equation, boundary data and current

For a fixed orthonormal transverse basis, the internal kernel solves

```
f''(z) + Q(z) f(z) = 0
Q = (2 m_e e / hbar^2) V(E_tip + phi - phi_tip) - K_transverse^2
```

`Q` is Hermitian and has units of inverse square metres. `V` is the matrix of
a scalar field in the transverse basis; energy is numerically expressed in
eV and electric potential in V in this expression. The left **total complex
field**, including any reflected part, is prescribed. The right half-space
has the outgoing/decaying logarithmic derivative `i sqrt(Q_exit)`. Its zero
eigenvalue has zero derivative; the exponentially growing solution is excluded.
Positive, negative and zero eigenvalues are retained, not passed through a
forward-angle cutoff.

This is a boundary-value problem, **not an emission current law**. Its internal
left boundary is useful for independent operator tests. It is not permission
to configure a source at an interior plane. Current is solved as
`(hbar/m_e) Im(f* f')`, summed over the transverse basis, rather than treating
`sum(abs(f)^2)` as current. Complex amplitude and derivative are retained at
every interface. No fitted phase, intensity normalisation or fabricated axial
flight time is added.

## Stable layer composition

Each axial interval uses a constant Hermitian `Q` and its exact eigensolution.
The arbitrary positive numerical wavenumber `kappa` defines coordinates
`f = u + v`, `f' = i kappa (u - v)`. It is not an injection energy or an
independently configurable physical reference plane.

For a slab of width `d`, let `k = sqrt(q)`, `c = cos(k d)`, `s = sin(k d)/k`
in its eigenbasis. The symmetric slab scattering coefficients are

```
D = c - i (kappa + q/kappa) s / 2
r = i (q/kappa - kappa) s / (2 D)
t = 1 / D
```

Scaled trigonometric functions avoid constructing `exp(+abs(Im(k d)))`.
An entire sinc expansion handles grazing channels. A backwards reflection
recursion composes noncommuting slabs; a forwards reconstruction recovers the
fields and derivatives. Independent changes of the numerical `kappa` leave
the tested physical answer unchanged.

Local scattering unitarity, linear-solve residuals and interface-current
balance are checked separately. Singular or ill-conditioned solves fail
explicitly. A coupled transmission below the normal floating-point range
requires a future log-scaled coupled representation; it is rejected, never
replaced by zero. The existing diagonal solver can retain scalar logarithmic
transmission, but that does not yet solve the general coupled case.

The implementation is a dense CPU reference, with approximate cost
`O(layers * channels^3)`. The channel limit defaults to 1,024 and the configurable
working-memory estimate defaults to 72 GiB, also checked against available
memory. This is not a GPU/performance upgrade or a measured total-RSS cap.
Cancellation is checked between stages and before result publication.

## Sampling the existing gun field

The adapter captures a detached, geometry-inclusive gun graph and its source
implementation identity. It does not apply a new preset. Validation that
would alter captured settings is rejected. Numerical arrays are detached too;
changing the live gun or solver code during field preparation rejects the
result. Returned arrays and records are immutable.

The sampled electric potential includes the existing extractor, electrostatic
gun lens and accelerator provider, plus the analytic Wien **electric** part
when installed. Unsupported providers are rejected. The nonrelativistic
adapter accepts tip energy components in `(0, 1000] eV` and sampled local
kinetic energies in `[-1000, 1000] eV`; negative values are barriers. A path
outside this domain is not made usable by omitting the accelerator.

Transverse Fourier-Galerkin matrices use actual field samples at independent
quadrature factors 2--16 over a fixed physical period. Retained frequency
differences do not wrap circularly. The kinetic term uses the affine lattice's
reciprocal metric, including the `2*pi` and SI conversion factors. The finite
transverse domain is **periodic**; it is not a physical gun bore. Basis size,
physical period, potential quadrature and axial subdivision all need separate
convergence checks. Current conservation alone does not establish accuracy.

The final electric-field sample is a matrix for future boundary matching.
It is **not** evidence that the remaining physical gun is a uniform outgoing
medium. A test connecting that matrix to an outgoing half-space checks only
the declared interior operator problem.

## Remaining physical integration

The existing analytic axial/radial-order-r-squared potential is not a solved
curved-tip extraction field. In the current default model it is zero from
the tip to the start of the extractor transition at 0.05 mm. Sampling it more
finely does not create the missing near-tip field.

Before full tip-to-image admission, the implementation still needs:

1. A defined tip flux/coherence boundary consistent with the preserved spatial
   and energy distribution, and the curved tip/extractor electric field.
2. Nonparaxial magnetic covariant derivatives, including the installed gun
   deflectors, stigmator, Wien magnetic part and shared column-field tails.
3. Physical bores, apertures and absorption in that same executed chain.
4. Validated matching to high-energy propagation, carrying each mode's complex
   field, phase reference and current into existing column/specimen stages.
5. Full TEM/STEM, dynamic scan and inelastic convergence and acceptance checks.

No item on this list is marked complete by the electric-only boundary kernel.
The energy-filter integration order and physical detector interception rules
remain unchanged.

## Verification and evidence

All checks here are offline CPU tests. The references include exact uniform
propagating/grazing/evanescent waves, the independent earlier diagonal solver,
a dense first-order transfer exponential for short noncommuting slabs, and an
adaptive DOP853 solution for a smoothly varying coupled matrix. Direct
exponential quadrature checks odd/even Fourier indexing. Actual FEG field
samples check radial coupling and response to a changed extractor setting;
these interior test fields are explicitly not physical tip emission fixtures.

The smooth-field complex-amplitude error for 16, 32, 64 and 128 intervals is
approximately `1.97e-4`, `4.91e-5`, `1.23e-5`, `3.07e-6`, respectively.
The initial 64-interval run missed the unchanged `1e-5` tolerance. Axial
refinement, not tolerance relaxation, resolves that test. Actual extractor
potential quadrature factors 2, 4, 8 and 16 give successive matrix differences
of approximately `1.58e14`, `3.67e13`, `9.05e12 m^-2`; these are convergence
observations at a fixed test domain, not full-gun precision guarantees.

- `evidence/20260912T192250Z-coupled-boundary-initial-6d4aa69f/`:
  6 passed, 1 failed. The axial convergence failure above is retained.
- `evidence/20260912T192745Z-coupled-electric-field-ad571dc4/`:
  25 passed, 0 failed, 0 skipped, 7.55 s. Initial coupled/actual-electric-field
  and earlier diagonal boundary checks; raw XML stores convergence values.
- `evidence/20260912T193306Z-coupled-final-ff4436a4/`:
  162 passed, 0 failed, 0 skipped, 118.12 s. Includes 17 new coupled/field
  cases, detached-input checks, the actual atomistic specimen cases and
  existing source-guard, grid, inelastic, segmented-cache, detector/scan and
  CLI regressions. Source/configuration inputs did not change during the run.
- `evidence/20260912T193528Z-coupled-compatibility-2ab5a413/`:
  23 passed, 2 failed, 0 skipped, 18.53 s. Both historical full-gun variants
  still fail at the preserved 6.972% paraxial generator-error bound versus
  the 1% limit. The new electric-only kernel does not bypass that limit.
  These files overlap the focused run; their counts are not added together
  as a unique-test total.

Final regression evidence is recorded in `ACCEPTANCE_MATRIX.json`. None of
these results is a hardware acquisition, full physical-tip image, or GUI
imaging demonstration. Historical gun-source failures are not skipped or
retuned into passes.

## References and distinction

Multichannel scattering with open and closed channels has a current-unitarity
formulation; see Kazinski and Korolev,
[Multichannel scattering for the Schrodinger equation on a line with different
thresholds at both infinities](https://arxiv.org/abs/2307.00473).
The locally derived reflection recursion above is tested independently; it
does not claim to reproduce that paper's complete analytical construction.

For an alternative discretisation, Simoni, Viel and Launay describe a
spectral-element multichannel boundary problem in
[Application of the spectral element method to the solution of the multichannel
Schrodinger equation](https://arxiv.org/abs/1705.04102).
The present implementation is piecewise-constant scattering composition,
not their spectral-element method or its reported convergence order.
