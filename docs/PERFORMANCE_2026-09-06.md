# Specimen transport and checkpoint reuse: 2026-09-06

## Changes

- Magnetic material-boundary searches now bound a trial flight by a nearby
  tangent intercept before integrating the magnetic arc. The actual curved
  flight and chord intersection still determine the crossing. A missing
  tangent intercept never rules out a curved trajectory entering material.
- The local material envelope excludes vacuum supports. Once an electron has
  left all local matter in the outward direction, further magnetic transport
  belongs to the column solver. Physical sample/reference planes are unchanged.
- Field-provider support intervals are resolved once per calculation, not once
  per magnetic step. Geometry or lens edits create a new field context.
- The GUI can pass an earlier incident checkpoint to the solver even when an
  AC raster or lens edit invalidates every complete-product signature. The
  solver must still validate the source and common integration-plan prefix.
  Invalid sample, image or detector products are not reused.

Collision physics, random seeds, geometry epsilon, magnetic step limits, ray
counts, operating strengths and presets have not been relaxed or recalculated.

## Measured local benchmark

Single before/after runs on the same local Windows environment, using the
project virtual environment. Both use Ideal Optics, a 10 nm virtual Si [110]
sample, vacuum support, 300 keV electrons and seed 73. Only
`simulate_elastic_point_transport` is timed; startup and artifact writing are
excluded. These are prescribed local incident rays, not a full-column run.

| Incident fixture | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| 64 axial rays | 13.0515 s | 0.4623 s | 28.23x |
| 16 rays, 20 mrad slope | 5.3933 s | 0.1853 s | 29.10x |

Terminal identities, energies, weights, outcomes and event counts match
exactly. The largest terminal-position difference is 2.67e-15 nm, the largest
material-path difference is 3.56e-15 nm, and the largest direction-component
difference is 6.62e-24. Stored history shapes and offsets also match. These
differences are at floating-point rounding scale, not a changed approximation.

The recorded NPZ pairs are retained locally in `tmp/transport_*20260906.npz`.
To generate a new artifact from the project root (an existing output is refused):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_specimen_transport.py --rays 64 --output tmp/transport_new.npz
.\.venv\Scripts\python.exe scripts/benchmark_specimen_transport.py --rays 16 --tilt-mrad 20 --output tmp/transport_tilt_new.npz
```

The script saves terminal states, material paths, trajectory history and timing
metadata. Comparing performance requires the same fixture and environment.
The regression suite also keeps the previous boundary search as a test-only
numerical reference.

## Validation and limits

- Related offscreen regressions: **150 passed**, 16 third-party Pydantic
  deprecation warnings, exit 0, 120.43 s. A pyqtgraph disconnect warning was
  emitted during teardown. This was not a full repository test run.
- Coverage includes axial and tilted seeded scattering, a real copper mesh,
  a curved sidewall interception without a tangent hit, local fields, EDS,
  downstream transport, controller caches and segmented column reuse.
- The raster integration regression verifies that checkpoint-based propagation
  and a cold calculation produce identical incident arrays.
- No full application, large STEM scan, desktop rendering or GPU performance
  speedup is claimed. The local ratios cannot be applied to the user's earlier
  hour-long calculation. Real-support timing has not been benchmarked.
- STEM wave CPU fallback, Preview replacing the active ray display, and stale
  EDS curves retained after a sample edit are separate remaining issues. This
  change does not claim to resolve those display/backend policies.
