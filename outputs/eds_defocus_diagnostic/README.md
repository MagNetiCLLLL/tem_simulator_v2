# EDS objective-defocus diagnostic

The clarified observation is that **68.98055% still produces Si EDS, whereas
68.0% does not**. The reproducible saved-profile calculation agrees: at 68.0%
the sampled beam misses the 10 nm diameter specimen. Zero sampled material
paths do not prove that the underlying continuous beam has exactly zero
overlap, and increasing ray count alone does not guarantee adequate sampling
of this very small specimen.

`objective_beam_overlap.png` plots the actual saved rays, with the specimen
outline at true scale. If 10 nm is intended only as an atomistic calculation
window inside a larger foil, it must not also be used as the physical foil
diameter. The physical envelope and local wave/atomistic window serve different
purposes; this diagnostic leaves the user's selected dimensions unchanged.

## Self-consistent optical and elastic calculations

Baseline: `../haadf_dpa_clearance/si110_haadf_dpa_12mm.toml`, Si disk diameter
10 nm, thickness 5 nm, 300 kV, vacuum support. The original profile objective
value is already 68.9801%. Calculations use its 0.05 mm integration step,
production `run(..., optical_only=True)`, `incident_rays_from_simulation`,
and `simulate_elastic_point_transport`. No STEM raster, GPU multislice or
synthetic image is used. AC scanning/wobble are disabled; the production point
API centres the incident centroid on (0,0) without changing its spread,
direction or weight. The unshifted centroid and minimum radius are also saved.

| Objective excitation | Emitted rays | Reaching sample plane | RMS radius (nm) | Minimum centred radius (nm) | Si material hits | EDS material tracks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 68.9801% | 49 | 25 | 0.08510 | 0.01143 | 25 | 27 |
| 68.98055% | 49 | 25 | 1.60687 | 0.14766 | 25 | 27 |
| 68.0% | 49 | 25 | 3486.307 | 319.081 | 0 | 0 |
| 68.9801% | 1000 | 561 | 0.105951 | 0.000714 | 561 | 578 |
| 68.98055% | 1000 | 561 | 1.800852 | 0.002940 | 561 | 578 |
| 68.0% | 1000 | 561 | 3930.954 | 52.379 | 0 | 0 |

At 68.98055%, all arriving rays intersect Si: weighted material path is
5.00064818 nm per arriving electron. At 68.0%, that path is zero for both
sample counts. The emitted current reaching the sample plane remains 56.1%
for the 1000-ray cases; reaching the plane is distinct from hitting the tiny
piece of material on that plane. Elastic-scattering event count is also
distinct from material hits: a straight path can ionise Si and generate EDS
without a sampled elastic collision.

Evidence: `baseline_v2/diagnostic.json`, `objective_68/diagnostic.json`, and
their per-case NPZ files with individual positions, weights and material paths.
Reproduction script: `../../scripts/diagnose_eds_objective_defocus.py`.

## Actual cached App incident rays

`recent_actual_cached_incident.json` records three independently read App
incident caches. Their complete state, objective percentage and EDS settings
are not available, so no cache is definitively assigned to a specific
excitation. The broad `f4c443...` cache has 15000 emitted rays, 7635 reaching
the sample plane, RMS radius 3800.34 nm, zero sample-radius hits at its raw
positions and one after centroid translation. This resembles the broad-beam
regime of the explicit 68.0% test, without identifying its exact settings.
The narrower latest cache has RMS 1.7291 nm and all 7635 rays within 5 nm.

## Consistent hybrid EDS calculation

The cache sample plane is Z=1679.2 mm, but the saved baseline assembly sample
plane is Z=1599.2 mm. The **consistent hybrid** reanchors the saved sample-plane
phase space by -80 mm to the intact baseline assembly. This leaves its local
X/Y, slopes, energies and weights intact and keeps sample and objective poles
in the same coordinate system. It does not reconstruct the unknown live App
state. The specimen/field/detector/dose assumptions are from the baseline,
with the specimen-local objective field set to 68.0%.

| Broad cached beam | Si hits / arriving rays | Si tracks | Expected counts, angular order 1 |
| --- | ---: | ---: | ---: |
| Raw transverse position | 0 / 7635 | 0 | 0 |
| Centroid translated to (0,0) | 1 / 7635 | 1 | 0.607300135 |

The one centred hit carries conditional weight 1/7635 and has a 4.999995 nm
material path. This is extremely sparse material sampling. The emitted dose
assumed by this profile is 1.523805926e10 electrons during 0.000244140625 s;
7.756172165e9 electrons reach the specimen plane. These are **baseline model
assumptions, not measured or recovered user EDS settings**. The detector uses
4.04 sr aggregate installed-holder solid angle, efficiency 1, and no Poisson
sampling. Under that specific order-1 expected count, the probability of a
Poisson acquisition recording zero total counts would be about 54.48%.

A lightweight check reuses the same single track with angular quadrature
orders 1/3/5: expected totals are 0.607300135 / 0.404861044 / 0.364374897.
All are positive. Finite angular quadrature affects absolute collection and
pole shadowing; these checks do not demonstrate convergence. They are separate
from the zero material paths in the explicit 68.0% 49/1000-ray calculations.

Valid hybrid evidence is in `broad_cached_hybrid_consistent_center/`,
`broad_cached_hybrid_consistent_raw/`, and `photon_quadrature_consistent/`.
They contain JSON metrics, the saved cached sample-plane arrays, and expected
spectra. Scripts are `../../scripts/diagnose_eds_cached_incident.py` and
`../../scripts/diagnose_eds_photon_quadrature.py`.

The cached-incident script currently reads the original machine's cache manifest
path from `recent_actual_cached_incident.json`; that application cache is not
included in the repository. The valid hybrid directories preserve the required
sample-plane NPZ data, but the script does not yet accept those NPZ files as an
alternative input. Its original command therefore requires the recorded local
cache, rather than working unchanged in a fresh checkout. This limitation does
not affect inspection of the saved arrays and spectra or the independent
saved-profile optical calculations above.

## Invalid and incomplete exploratory outputs

**Do not use** `broad_cached_center/`, `broad_cached_raw/`,
`broad_cached_center_photon_review/`, or `photon_quadrature/` for physical
conclusions. They incorrectly moved only the sample to the cached global Z
while leaving baseline poles 80 mm away, creating artificial photon shadows.
Each is explicitly marked INVALID and retained locally for audit, excluded by
`.gitignore` and not included in the repository. The old q1/3/5
results are invalid for the same reason. `latest_cached_baseline_eds/` was
stopped after the user's clarification; `baseline/` stopped on a missing
low-level tuning label before recording any valid case. These two incomplete
directories are also local only and excluded from the repository. The valid
hybrid, baseline and objective diagnostics listed above are included.

## Implemented corrections and visible diagnostics

The point-interaction engine now resolves omitted coordinates to the same
scan origin for both Elastic and EDS before computing or reusing either
result. A previous call-order defect could reuse raw-centroid Elastic paths
for a differently centred EDS request. Seven new regression cases verify the
correction, including explicit/default coordinates and old cached requests.
Low-level upstream boundary extraction still preserves the original beam.

Elastic metrics now distinguish rays reaching the plane from positive-weight
trajectories crossing the specimen or support. They include conditional hit
weights, beam RMS radius and physical specimen dimensions. The EDS page shows
these values beside expected and Poisson-sampled counts. It separately explains
zero sampled material hits, zero Poisson counts, generated photons not collected,
and collected signal outside the displayed energy window. Missing older
diagnostics are not interpreted as zero. No artificial intensity floor is added.

The cache schema invalidates obsolete particle-point products while retaining
valid incident and wave calculations. New UI diagnostics appear after restarting
the app and running a new High accuracy / EDS calculation. Related physics,
cache, photon transport and UI regression tests pass. These diagnostic scripts
do not modify the user's profiles or the live App state.
