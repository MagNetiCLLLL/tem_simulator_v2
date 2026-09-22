# EDS response caching: scientific boundaries and implementation order

Code/science audit recorded 2026-09-19. Phases 1 and 2 are implemented; see the
[persistent EDS cache report](eds-persistent-cache-2026-09-19.md) and
[dose/readout response report](eds-readout-cache-2026-09-19.md) for validation,
scope and measured timings. Phase 3 remains proposed work.
Classical tip-origin emission and transport remain authoritative; coherent
development remains paused. The previous measured EDS stages took 176.176 s
and 172.024 s in the declared finite-Si benchmark. These are complete EDS
stage times, not separately profiled photon-intersection timings.

## What can be reduced to electron dose times a response?

For fixed normalized incident phase-space distribution f, specimen/material
state G and detector/collection state D, define K[d,k](f,G,D) as the expected
detected counts in detector d and spectral bin k per electron reaching the
sample reference plane. In the current linear, stationary model:

    expected_counts[d,k] = N_plane * K[d,k](f,G,D)

K is an expected yield, not a Bernoulli probability constrained to at most
one photon per electron. It includes finite-specimen overlap and support
contributions, electron material paths/energies, vacancy and fluorescence
yields, photon absorption/shadowing and detector response. N_plane is physical
weighted dose, not the 5,000 numerical trajectories used to sample it.
Source-current and upstream transmission factors must each be applied once.

Electron incidence direction is relevant even when characteristic photon
emission is approximated as isotropic. It changes material entry/exit, path
length, scattering, emission positions and consequently photon escape and
absorption. For a straight path through a uniform plane-parallel foil away
from edges, L=t/cos(theta), with theta relative to the surface normal. This
limiting relation is not a global monotonic EDS-intensity law: collection,
absorption, edges and scattering still matter.

Holding f, G and D fixed permits dose-only scaling. A physical current control
that also changes source energy/angular distribution is not dose-only.
Changing numerical ray count at fixed physical current and dwell must not
scale the mean spectrum with that count; it changes the sampling estimate.
Poisson-sampled counts must be regenerated from the new expectation rather
than multiplied as an already-observed noisy spectrum.

## Findings before phase 1

- `detector/eds_signal.py:767`: weighted material tracks generate mean shell
  vacancies using n*sigma(E)*ds, then radiative line expectations. The current
  code already uses fractional/expected counts, not every physical photon.
- `detector/eds_signal.py:942`: each line becomes deterministic detector
  quadrature representatives. The earlier 82,592 entries are emission/line
  records, not 82,592 independently sampled physical photons.
- `detector/eds_photon_transport.py:1036`: a call-local 1,024-entry LRU already
  shares exact origin/direction geometry. It does not persist across calls;
  each representative still applies its energy-dependent attenuation and
  builds/account for its own result.
- `simulation_pipeline.py:432,724`: ordinary full calculations already reuse
  matching in-memory EDS products. The section route at `:1147` requests EDS
  afresh even when its material result is reused.
- `physics/particle_sections.py:51` and
  `physics/completed_particle_section.py:73`: material restart caches retain
  elastic/inelastic/downstream state but no complete EDS product.
- `particle_section_io.py`: the explicit persistence type whitelist and
  restored result omit the EDS product. Loading a full particle archive thus
  regenerates EDS, even after an otherwise compatible downstream-lens edit.

## Recommended implementation order

### 1. Persist and restore identical executed EDS results — implemented

Add an optional EDS product to the material checkpoint and preserve it from
both ordinary full and explicit section calculations. Missing fields in old
archives remain readable as no cached EDS; they do not claim a completed EDS
product. Include the nested spectrum, vacancy/line, material-quadrature and
photon-transport record types in the existing explicit type whitelist. Reuse
the immutable-array, checksum and atomic-file mechanisms already present.

Admit reuse only after the executed material cache, its incident digest,
actual acquisition point and EDS dependency signature match. In addition to
field/model/data identities, this includes actual incident positions, slopes,
energies, survival, source IDs, weights, clocks and numerical settings.
Keep original provenance and physical identities; no downstream source is
introduced. Restored products must participate in both the pipeline's reuse
decision and its retained-observable set, or they will be discarded/recomputed.

Start by reusing exactly identical dose/readout requests. Do not combine this
first change with a different scattering model or a rounded cache key.

### 2. Separate physics from dose and spectral readout — implemented

Retain a per-incident-electron expectation separately from absolute counts.
Preserve the complete executed records needed by the event ledger and saved
state. Dose-only changes scale all relevant expectation records consistently.
Detector response and random counting are downstream products with separate
identities: bin edges/FWHM update binning/broadening; a random seed update only
resamples counts. Keep current exact-result caching as the initial fallback.

The implementation uses immutable coefficients captured from a positive-dose
execution. Complete and per-electron response identities are separate; the
ordinary and saved-section pipelines both support the latter. Only static
(both scans off) frame-period changes are also admitted as dwell-only changes.

### 3. Reduce repeated work on the first calculation

Measure substage times and bounded cache hit/build counts first: material
track preparation, shell cross sections, emission positions, detector
quadrature, intersections, attenuation, spectrum formation and persistence.
The existing whole-EDS timer does not identify which of these dominates.

Reuse unit-weight transmission/collection coefficients for identical
origin/direction/photon-energy and material/geometry identity. Reuse detector
quadrature directions when their geometry permits it. Batch array operations
instead of repeatedly constructing the same intermediates for each line.
Preserve emission/source identity mappings and per-segment accounting.
Reordering/grouping accumulation may change roundoff; require a declared
tolerance and compare spectra, per-segment totals, absorption and ledgers.

Static atomic tables and material attenuation already have caches; audit and
extend those rather than add duplicate stores. Energy-grid interpolation or
spatial/angular buckets are later approximations, not exact caches. They need
error/convergence checks and fallback at elemental thresholds, finite edges
and shadow boundaries before becoming an optional fast path.

## Desired invalidation rules

| Change, with other relevant inputs fixed | Reuse | Recompute |
| --- | --- | --- |
| Downstream projector excitation that does not affect specimen fields or photon geometry | Complete EDS product | Affected downstream electron propagation |
| Pure dose/dwell scaling in the linear stationary regime | Trajectories, per-electron expectations | Absolute means and optional counting noise |
| Spectrum bins or Gaussian width | Material and collected line expectations | Spectral response |
| Poisson seed | All expectations | Sampled counts |
| Incident location/direction/energy/distribution | Unchanged atomic/material lookup data | Changed electron/material/EDS response |
| Specimen composition/thickness/shape/pose | Independent atomic tables | Affected material and photon paths |
| Detector collection geometry or photon-only occluder | Compatible electron histories/emitted lines | Photon collection/absorption and spectrum |
| Support geometry/material | Only proven independent data | Electron paths too if the support participates in them |
| Scan pixel | Static tables and truly matching conditional responses | Local beam/material response unless physically equivalent |

Scanning a heterogeneous specimen cannot use one universal spectrum multiplied
by per-pixel electron count. A homogeneous, uniform, non-shadowed limiting
case can share a response only after equivalence is established.

## Acceptance checks and model limits

1. Save/load an executed EDS result, change only an eligible downstream lens,
   and make any gun, material or EDS recomputation fail explicitly. Compare
   expectation spectra, per-segment totals, ledgers and restored provenance.
2. Independently vary incidence angle/position/energy, material and detector
   inputs. Relevant caches must invalidate even if electron count is unchanged.
3. In a fixed-distribution linear fixture, double dose: expected counts double;
   normalized spectral shape stays fixed. Zero dose stays zero. Changing the
   number of numerical histories is not an additional physical-dose factor.
4. Count-noise seeds change sampled results, not expectations. Changed bins or
   width leave pre-response collected line totals invariant in the declared
   spectral-window/normalization regime.
5. Compare cold execution, identical loaded reuse, dose-only changes and an
   angle/geometry invalidation separately. Report time/IO/memory and the actual
   recomputed stages. Keep numerical work at or below half available CPUs.

The present model is independent-atom classical transport with characteristic
lines. It does not include crystallographic channeling or a complete continuum/
detector-artifact spectrum. Electron material boundaries are currently axis
aligned while photon specimen boundaries accept orientation; the current code
therefore does not establish a fully shared arbitrary-tilt geometry model.
Pose must remain an invalidating input, and any broader tilt-trend validation
must address that existing mismatch. Caching must not conceal it or imply
that missing physics was implemented. No coherent calculations are proposed.

## Scientific references consulted

- [Ritchie, 2010: incidence angle, finite geometry and X-ray intensity](https://www.nist.gov/publications/using-dtsa-ii-simulate-and-interpret-energy-dispersive-spectra-particles).
- [Ritchie, 2017: fractional X-ray generation/detection with geometry sensitivity](https://www.nist.gov/publications/efficient-simulation-secondary-fluorescence-nist-dtsa-ii-monte-carlo).

These references support the dependency and expectation-method principles;
they do not validate this simulator or provide a speedup prediction. The
original audit changed documentation only; subsequent implementation tests
and numerical measurements are in the linked phase-1 report.
