# Non-circular specimen potential propagation

Status: **development specimen operator; full physical-tip imaging remains unavailable**.

## Outcome and scope

The two previously failing real atomistic specimen tests now execute without
changing their physical inputs, expected outcomes or conservation tolerances.
The earlier 8 GiB allocation failures remain in the historical evidence. The
potential-builder memory limit was not raised to obtain this result.

The FEG's 5 nm emission-region FWHM, 0.3 eV emission energy, existing energy
distribution, physical components and production source-admission checks are
unchanged. This change affects the development tip-wave specimen adapters,
not the legacy STEM image engine or the source-to-gun transport model.

## Why refinement alone was ineffective

The guarded sampled method bounds the sum of the incoming envelope spectrum,
analytical carrier and complex transmission spectrum before multiplication.
It remains available as `specimen_phase_method="sampled"`, with the original
`0.8*pi` budget and `1e-12` spectral-tail allowance unchanged.

For the atomistic example the whole-domain transmission spectrum approaches
Nyquist even on finer grids. The sufficient global support-sum condition
therefore kept rejecting the operation. This happened even though the wave
probability near the window edge was extremely small. Repeated enlargement
eventually reported a memory error, but increasing memory alone did not
establish a converged calculation.

## Numerical method

The default development specimen method is now `galerkin`. A wave has `N`
Fourier modes per axis. The potential is regenerated from the unchanged atoms
on an independently controlled quadrature lattice with at least `2N` samples
per axis over the same physical period. It is not interpolated from a coarse
potential and presented as new atomic detail.

Let `R` isometrically embed the retained wave Fourier coefficients into the
larger quadrature grid. For the dimensionless real phase `phi = sigma V dz`
(including the split-step fraction), the finite-basis operator is

```
H = R* diag(phi) R
wave_out = exp(i H) wave_in
```

Here `R*` denotes the adjoint, not scalar conjugation. `H` is Hermitian. Its
Fourier convolution is non-circular: scattering beyond the retained basis
does not reappear at the opposite edge as a false low-angle peak. The
exponential preserves norm within that finite basis and does not renormalise
the propagated field. An independently constructed dense matrix exponential
is used as a small-grid reference, not the same FFT implementation twice.

The exponential uses a deterministic shifted/scaled Taylor action. For each
scaled action the spectral-norm radius is at most one; the prior remainder
bound is `exp(r) r^(degree+1)/(degree+1)!`. The composed bound is below the
declared `1e-13` evaluation tolerance. Roundoff is checked separately by the
unchanged `1e-10` lossless-norm test. No random norm estimator consumes the
physical trajectory RNG or changes an interrupted calculation's replay.

**This is not an infinite-bandwidth solution.** Finite-basis exclusion and
potential quadrature error require independent convergence checks; unitarity
alone does not prove accurate scattering. A very small basis can suppress
unresolved high-angle transport. The returned record explicitly marks basis
convergence as requiring independent refinement. Stronger/thicker specimens
or different incident beams are not qualified by the example below.

Analytical phase carriers must still pass the original sampling check before
expansion. A carrier-free band projection acts only on Fourier coefficients
already represented and cannot itself fold frequencies; it no longer uses a
phase-product guard when there is no phase product. Existing selected
bandwidth projection remains a separately recorded **numerical loss**, never
detector absorption. Physical column fields, finite specimen occupancy,
zero-loss attenuation, conditional inelastic histories and physical apertures
are not removed by the method selection.

## Independent numerical controls and cache reuse

`WaveGridNumerics` adds two consumed numerical inputs:

- `specimen_phase_method`: `galerkin` (default) or `sampled`.
- `specimen_quadrature_factor`: integer 2--16 (default 2).

The wave basis and independently regenerated potential quadrature can be
converged separately. Both stay within the existing pixel, working-memory and
available-memory limits. Odd and even grids retain the same physical origin.
The development command-line request exposes them as
`--specimen-phase-method` and `--specimen-quadrature-factor`; they are included
in saved requests and failure receipts. These flags do not admit a new source.

Changed specimen numerical inputs invalidate specimen and downstream reuse.
They do **not** invalidate executed upstream column segments, whose identity
contains only the column's consumed numerical inputs. Inelastic slice keys
also bind the actual regenerated material arrays. Regression tests exercise
real cache hits/misses, not only equality of labels. Existing completed caches
are not deleted. Source-code identity changes prevent active reuse of results
computed by an older implementation; historical viewing is retained.

## Convergence evidence

The independent specimen fixture uses the actual atomistic potential, 300 keV
electrons, 0.4 nm thickness and a fixed incoming complex Gaussian field. It is
**not an executed tip source**. No source-generation step is mocked and then
claimed as full-chain acceptance. The separate inelastic fixture prescribes
known loss outcomes to check energy-aware remaining-material propagation and
bit-exact interrupted replay; its atomistic potential and propagation are real.

In the real-potential convergence fixture, a fixed `8 nm^-1` circular
observation band is used at every grid. This is a Fourier diagnostic at the
sample exit, not a propagated physical detector and not a window that scales
with Nyquist. Complex coefficients are compared in a shared Fourier window
without fitting or removing global phase.

Preliminary measured weights (per reference electron) were:

| Wave basis | Potential quadrature factor | Fixed-band weight | Numerical band loss |
| --- | --- | --- | --- |
| 128 | 2 | 0.39872060 | 2.8344e-5 |
| 128 | 4 | 0.39868320 | 3.5295e-5 |
| 128 | 8 | 0.39867134 | 3.7680e-5 |
| 256 | 2 | 0.39856966 | 1.0602e-5 |
| 512 | 2 | 0.39854267 | 3.0745e-6 |

The 256-to-512 fixed-band change is approximately 0.0068%. The shared complex
coefficient difference decreases from `6.01e-4` (128 to 256) to `1.60e-4`
(256 to 512); independent quadrature refinement also decreases its complex
difference. These are case-specific observations, not a universal accuracy
certificate. The test preserves its measured values in XML properties.

Evidence directories preserve source/configuration hashes and raw outputs:

- `evidence/20260912T175546Z-galerkin-initial-194acd1a/`: 14 passed, including
  the two original atomistic cases without changing their inputs/assertions.
- `evidence/20260912T175957Z-galerkin-convergence-89106a26/`: 98 passed,
  including independent matrix/analytic references, actual material grid
  convergence, original sampled-phase counterexamples, and source guards.
- `evidence/20260912T180334Z-galerkin-final-c775c5ae/`: 145 passed, 0 failed,
  0 skipped, 114.24 s. Final numerical, source-guard, segmented-cache,
  detector/scan, CLI and actual specimen checks; no acceptance threshold
  was relaxed. XML properties retain the convergence measurements.
- `evidence/20260912T180709Z-galerkin-compatibility-9be11549/`: 23 passed,
  2 failed, 0 skipped, 16.89 s. The full pipeline/column/canonical-action
  files were run. Both remaining failures are the historical gun-exit test
  variants that request the current narrow source through a paraxial gun
  (6.972% generator-error bound versus the unchanged 1% limit). They were
  not retuned, skipped or counted as passes. The atomistic specimen case in
  the same file passes. This does not complete low-energy gun propagation.
- `evidence/20260912T180652Z-galerkin-compatibility-0c0f4460/` records an
  earlier command with nonexistent test paths: exit 4, no tests executed.
  It is retained for audit and contributes no passing checks.

All checks are offline CPU numerical/engineering tests. No GPU throughput,
full physical-tip image, or hardware acquisition claim is made. The outstanding
physical integration is still the curved tip/extractor field and transverse
nonparaxial accelerating-wave chain, followed by full scan/image convergence.

## References and distinction from this implementation

- [abTEM multislice source](https://abtem.github.io/doc/_modules/abtem/multislice.html)
  illustrates why anti-alias treatment belongs before transmission. The method
  here is a Hermitian potential exponential, **not** a claim to duplicate
  abTEM's bandlimited transmission-function implementation.
- Bao, Lin, Ma and Wang, [An extended Fourier pseudospectral method for the
  Gross-Pitaevskii equation with low regularity potential](https://arxiv.org/abs/2310.20177),
  motivates using an extended potential window. Its error theorems are not
  asserted as validation of this electron-optics pipeline. The implemented
  operator and deterministic exponential are derived and tested locally.
