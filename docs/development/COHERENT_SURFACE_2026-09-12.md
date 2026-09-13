# Coherent grounded-tip near-field connection

Status: **a computed development near-field segment, not a completed TEM/STEM
imaging chain**. The new curved-tip model is explicitly selected; historical
source parameters and results are not converted. Final accelerating anode is
still 0 V, tip is `-HT`, and extractor is `-HT + extraction voltage`.

## Use in the application

1. Open **Model Inspector → FEG tip emission…**.
2. Select **Grounded tip surface model**, then **Coherent surface reservoir
   (development)**. Define current, cap angle, total kinetic-energy mean/RMS
   and surface edge phase. The classical normal/tangential energy inputs are
   hidden and inactive for this choice.
3. Open **Preview coherent tip near field… → Calculate near field**.
   This calculates the draft in a separate window without changing existing
   ray or image results. Numerical controls specify radial/axial resolution,
   domain size and energy quadrature. Cancellation waits for a running sparse
   factorisation to return; failed calculations retain the previous result.
4. Inspect the energy-mixture density and select an energy component for
   phase. **Export complex fields…** saves an exclusive-create, pickle-free
   NPZ containing every complex mode, coordinates, potential, fluxes, phase
   references, numerical choices and calculation identities.
5. **Apply** in the source editor explicitly saves the chosen source in the
   instrument; a subsequent operating-profile save uses version 9. Version 8
   classical surface records remain classical. Ray sampling of this new
   coherent reservoir is not implemented and fails explicitly; it is not
   replaced with the historical ray source.

The near-field window is not a camera readout. Its intensity is a weighted
wave-density diagnostic, not counts; the energy mixture has no single phase.
Source parameters remain at the curved metal surface, never at the numerical
top boundary, gun exit or specimen.

## Source and equation

The opt-in `coherent-cap-reservoir-v1` boundary is a prescribed post-emission
reservoir, **not a Fowler–Nordheim/material tunnelling calculation**. At each
energy it is one fully coherent, axisymmetric spatial mode. If `s` is arc
distance along the spherical tip and `s_max` the emission cap radius in arc
length, its incoming amplitude is proportional to

`cos²(pi*s/(2*s_max)) * exp(i*edge_phase_rad*(s/s_max)²)`

inside the cap, and zero outside. Current sets the incoming reservoir flux,
not net transmitted current after reflection. Angular spread is a consequence
of this wave boundary and propagation, not another independent classical
angular distribution. Total local kinetic energy has a positive gamma law
specified by mean and RMS, or is monochromatic for zero RMS. Different
energies are incoherent. Orthonormal Laguerre Jacobi-matrix quadrature avoids
Gamma-function overflow without changing narrow source widths.

The executed domain solves stationary, scalar, nonrelativistic Schrodinger
propagation in cylindrical `(r,z)`, with azimuthal index zero:

`[Laplacian + (2*m*e/hbar²)*(E + phi - phi_tip)] psi = 0`.

The actual grounded-electrode Laplace solve supplies `phi - phi_tip`. The
analytical curved metal surface has zero potential rise; interior values use
the electrode-grid interpolation. Wave and electrode discretisations require
**separate** refinement. No ray-derived phase, fitted source width or adjusted
electrode voltage is used.

P1 triangular finite elements use cylindrical measure `2*pi*r dr dz`.
Consistent mass/potential matrices use Gauss quadrature; no mass lumping.
The tip is a driven local Robin port, top and side are local Sommerfeld ports:

`(K - alpha*(E*M + P) - i*(B_tip + B_top + B_side))*psi = -2*i*B_tip*a`.

Normalize `a` to unit incoming flux **before** solving. Reflected flux is
`(psi-a)^H B_tip (psi-a)`, top/side fluxes are `psi^H B psi`. Outgoing fields
are never rescaled to force transmission to one. The discrete flux balance is
an algebraic identity of this lossless finite-domain problem; it does not
prove physical or discretisation convergence.

## Interfaces and cache

`simulate_tip_wave(state, TipWaveRequest(stop="tip_near_field", surface=...))`
now reaches this executed segment through the existing tip-wave entry point.
It checks the installed column field plan, walls, aperture/coil events and
energy-filter entrance before using the near-field cache. Gun magnetic/Wien
fields and electrode/aperture intersections are also checked. Unsupported
fields or stops in this domain are rejected, never silently omitted. All
later gun components remain installed; this stopping request does not claim
to have propagated through them.

The bounded in-memory cache includes the actual gun inputs, source, numerical
choices, grounded-field definition and implementation hash. Shared-column
guards run before reuse. The returned checkpoint also records the captured
instrument and column-plan identities. Read-only exported files are not an
independently configurable source or automatic full-gun restart file.

For an explicitly saved coherent-surface profile:

```powershell
.\.venv\Scripts\python.exe scripts/trace_tip_wave.py `
  --profile path\to\profile.toml --stop tip_near_field `
  --surface-radial-nodes 193 --surface-axial-nodes 385 `
  --energy-samples 3 --output path\to\new-result-directory
```

This saves the request profile, complex NPZ, receipt and a scientific plot.
`--describe` only reports configuration. `--scan` cannot produce images from
this stopping stage.

## Numerical evidence and limits

Reproducible reference driver: `scripts/inspect_coherent_surface.py`.
It explicitly selects the new source and preserves the installed default
assembly, R = 100 nm, 10-degree cap, mean = 0.3 eV, RMS = 0.1 eV and the
existing electrode voltages. It does not rewrite the default TOML.

The actual reference comparison uses a fixed observation region
`0 <= r <= 15 nm`, `0 <= z <= 1 nm`, cylindrical-weighted L2 errors, and no
fitted global phase. The initial complete refinement study is in
`evidence/20260912-coherent-surface-reference/report.json`; the final-code
rerun is in `evidence/20260912-coherent-surface-final-reference/report.json`.
Complex-field comparisons sum errors over the matching energy modes; they do
not construct a phase for the energy mixture.

| Independent refinement | Density change | Complex-field change |
| --- | ---: | ---: |
| Wave mesh 49×97 → 97×193 | 4.31% | 24.88% |
| Wave mesh 97×193 → 193×385 | 1.43% | 8.04% |
| Larger radial domain at comparable resolution | 0.105% | 0.688% |
| Top boundary 2 nm → 4 nm at comparable resolution | 0.316% | 0.798% |
| Electrode mesh and apex resolution doubled | 4.48% | 17.90% |
| Energy quadrature 3 → 5 | 0.0233% | Not a same-mode comparison |

Therefore **the reference wave/electric meshes are not yet converged to 1%**.
The approximately 75.5% top flux is a result of this finite-domain reservoir
model, not a measured gun efficiency. Sub-1% boundary sensitivity in the
tested region does not make a local Robin port an exact transparent boundary.
Independent analytical plane-wave and step-reflection fixtures check complex
propagation, retained reflection, phase linearity and current conservation.

The expanded regression at
`evidence/20260912T221702Z-coherent-surface-final-851c9b42/report.json` reports
100 passes and 9 failures. All nine are historical Gaussian-Schell source
tests rejected by the unchanged source-domain gate; the same named failures
are recorded in `evidence/20260912T121745Z-review-p0-b2639b47/pytest.log`.
They are not counted as passes, removed or given new source parameters.
Final targeted validation is recorded separately in
`evidence/20260912T222309Z-coherent-surface-scoped-3d311c7b/report.json`:
**142 passed, 0 failed, 0 skipped**, including actual near-field execution,
GUI worker completion/cancellation, failed-result retention, CLI export,
phase/flux checks, source-profile round trips and classical compatibility.
These are CPU/offscreen checks, not full-chain or hardware validation.

## Required next connection

1. Resolve the near-tip electric boundary and wave phase errors through
   independent field/wave refinement or a better-resolved curved-element
   discretisation. Do not tune source widths or voltages to match a target.
2. Replace the finite-domain outgoing port with a **two-way, flux- and
   phase-consistent interface** to the remaining relativistic/magnetic gun.
   A stored top field plus a label is not this missing transport. Downstream
   back-reflection cannot be reproduced by merely joining two forward runs.
3. Execute extraction, acceleration, electrostatic/magnetic focusing and all
   physical apertures before connecting the actual gun-exit checkpoint to
   the existing column, specimen, dynamic scan and detector wave stages.
4. Validate full TEM/STEM images and per-mode detector phase; integrate a
   requested energy-filter traversal last. Production admission remains closed.

This scalar near-field solver is limited to positive sampled kinetic energies
up to 1 keV. It excludes spin, space charge, material tunnelling and nonzero
magnetic fields. It is not a calibrated representation of a commercial CFEG.

## References

- [Yanagisawa et al., Scientific Reports 7 (2017)](https://www.nature.com/articles/s41598-017-12832-3):
  experimental tip-emission interference and wave-mechanical tip modelling.
  Motivation for treating the emission coherently, not validation of this
  prescribed reservoir, electrode geometry or full microscope.
- [SciPy `eigh_tridiagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html):
  the symmetric tridiagonal eigensolver used for positive gamma quadrature.
- Grounded geometry and its reference sources remain documented in
  [the preceding surface redesign](GROUNDED_TIP_SURFACE_2026-09-12.md).
