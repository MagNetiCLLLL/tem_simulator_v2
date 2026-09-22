# Transverse Beam: source position, direction and flight time

Read-only production-code assessment requested on 2026-09-19. The previous
High calculation finished but was slow. The next requested visualization scope
is three simple tracking modes. This assessment does not resume coherent work
or claim that the new interface has been implemented.

## Present behavior and the misleading upper panel

`InitialDirectionColourWheel.paintEvent` in `gui/diagnostic_tabs.py` draws a
fixed azimuth legend without reading the result. The upper panel headed
`Source position colour` or `Emission direction colour` contains that legend,
not a source-position scatter plot. Its regular XY appearance cannot diagnose
source position/direction correlation.

In `gui/beam_analysis.py`, `View` and `Colour by` are independent. Position XY
plots positions at the selected plane; Angular XY plots atan of its projected
slopes. Switching coordinates does not change the chosen colour quantity.
Actual emission-direction colours are correctly read from pre-extraction
`gun_trace.emission_reference`, through source ancestry, in
`physics/ray_identity.py`. Missing launch records remain unavailable/grey.

The current source-position and direction hue maps encode azimuth only.
`Emission angle to normal` separately encodes the polar angle. Neither single
colour choice currently encodes both components of a two-dimensional direction.

## What the source actually samples

The default flat classical tip samples position and angle from separate Halton
dimensions and their respective truncated Gaussian distributions. It does not
force outward velocity to follow radial position. Each numerical particle has
one launch position and direction; finite samples approximate the joint
distribution rather than enumerating every direction at every exact position.

Consequently, source XY coloured by launch direction should show statistical
mixing. Direction XY coloured by that same direction necessarily has an ordered
colour pattern. "Any direction" means within the selected outgoing angular
distribution, not equal probability throughout all solid angles. A curved tip's
local surface normals can produce genuine position/direction correlation in
global coordinates, which must not be removed for visual uniformity.

A figure using 3,072 actual samples from the current default emitter is saved
locally at `tmp/transverse-audit-20260919/source-position-vs-angle.png`.
It compares both coordinate spaces and both azimuth colour quantities. It is
source sampling only, not a column calculation or the user's saved instrument.

## Three-mode target

| Mode | Upper plot | Lower plot | Present readiness |
| --- | --- | --- | --- |
| Source position | Actual emitted XY, coloured by its original position | Selected-plane XY, retaining the same original-position colours | Launch data and lower tracking available; actual upper plot missing |
| Source angle | The same actual emitted XY, coloured by original launch direction | Selected-plane XY or angle XY, retaining original-direction colours | Launch direction/lineage available; actual upper plot and simplified controls missing |
| Time of flight | Launch positions coloured by each displayed path's arrival-time difference at the selected plane | Selected-plane particles using the matching arrival-time colours | Full-chain time data missing |

For the first two modes, preserve the complete emitted source in the upper
panel even after downstream clipping. Colours do not change with Z. Keep a
small explicitly labelled colour legend, separate from the actual source plot.
If both direction components are needed, hue can encode azimuth and a second
visual colour dimension encode polar angle, with a declared fixed scale.
Keep coordinate projection distinct from the tracked colour quantity.

For TOF, all trajectories start from the same declared emission event. Show
absolute flight time and/or a clearly defined reference-subtracted delay.
Unlike source labels, this colour depends on the selected destination plane.
Non-arriving trajectories have no arrival time and must not become zero-delay
points. Different descendants of one source particle can have different times;
their values cannot be collapsed to one time per source ID. Do not jitter
coincident launch positions to manufacture a mixed-colour appearance.

## TOF is not quantum phase

Flight time is the integral of distance divided by local speed. In forward
column coordinates it can be accumulated as
`dt = dz * sqrt(1 + tx**2 + ty**2) / v(K)` with consistent SI units and
relativistic speed. Acceleration and inelastic energy losses matter; dividing
the total path by the final speed is generally insufficient.

The gun already records equal-time histories and numerical arrival times at
several physical planes. However:

- `GunExitBundle`, column `Branch`, propagation checkpoints and `BeamPlaneData`
  have no complete per-trajectory flight-time field.
- Persistent incident-seed serialization omits the gun's equal-time history
  and plane arrivals, so those times can disappear on cache reload.
- Detailed sample terminal bundles and the energy-filter public interfaces do
  not preserve complete path timing and ancestry through their full transport.

The robust implementation should accumulate time within existing transport,
carry it through checkpoints/branches and version the cache codec. It need not
solve fields or emit particles a second time. Store requested observation-plane
values rather than duplicating every full history. Measure the added cost.
Estimates from existing sparse drawing paths must be labelled approximate and
cannot reconstruct missing sample or filter transport. Integrate the filter
last and do not bypass it for a requested downstream plane.

Quantum phase additionally requires a defined coherent state/reference and
electromagnetic phase evolution. It cannot generally be recovered by renaming
TOF or using a universal `phase = frequency * TOF` rule. The classical
`Phase space U / theta U` views are position-angle diagrams. The separate wave
reader can display a single already-calculated complex mode's phase; it does
not qualify the paused tip-to-column coherent chain. See the
[Feynman discussion of electron phase and electromagnetic potentials](https://www.feynmanlectures.caltech.edu/II_15.html).

## Implementation order and performance

1. Implement the first two actual-source plots and the three-mode presentation.
   Cache emission arrays, source-ID indexing and fixed colours once per result;
   use the same sample identities above and below. Current emission hover and
   brush rendering repeat ID sorting, which can be removed without retracing.
2. Add TOF at the existing integrator stages, then checkpoints and persistent
   caches. Enable it only for paths with complete timing provenance.
3. Check straight drift `L/v`, longer tilted paths, energy-dependent speed,
   aperture reachability, branch inheritance, cache round trips and time-step
   convergence. Small delay differences need stable accumulation; coarse
   float32 display paths are not phase-accuracy evidence.

Current focused UI/data regression: **81 passed**, exit 0
(`tmp/transverse-audit-20260919/ui-audit.xml`). These validate existing controls
and ancestry, not the unimplemented source plots or TOF. No production code,
physical settings or running application was modified for this assessment.
