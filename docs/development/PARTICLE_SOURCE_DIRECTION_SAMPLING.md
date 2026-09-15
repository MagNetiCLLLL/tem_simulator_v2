# Classical tip sampling and Ray Diagram audit

Date: 2026-09-14. Coherent source development remains paused.

## Controls and meaning

In the FEG tip settings, `Directions per position` controls numerical sampling.
New reference/assembly defaults use 8. It divides the existing ray budget
between surface sites and local direction/energy samples; it does not multiply
current or invent a downstream source. At 49 rays, 6 sites receive 8–9 samples
each. At 15,000 rays, 1,875 sites receive 8 samples each. Increase the total ray
budget to resolve more positions at the same angular sampling.

Surface sites sample equal cap area. Every site has equal total probability;
if the ray count does not divide evenly, per-ray weights compensate for the
different group sizes. Direction azimuth is stratified within each site.
The normal/tangential energy law or specified positive-energy law and local
angular cutoff still determine the physical distribution. At zero angular
spread, directions stay parallel to the normal: no display-only divergence is
added. Zero-current tuning probes are separate from the physical quadrature.

The cap normal tilts farther from +Z toward the cap edge. This geometric tilt
is distinct from the distribution of electron velocities about each normal.
No virtual centre or prescribed gun-exit divergence is used by this model.

Historical profiles without the sampling field retain one sample per position.
They are not silently converted; edit `Directions per position` explicitly to
adopt grouped sampling. The source and transport cache identities include the
sampling setting. Changing it does not invalidate the unchanged electrostatic
field solution. No calculated arrays are added to version control.

## Transverse panel

`Colour by` offers:

- `Source position`: original launch-position azimuth about the physical source
  axis. Historical results without launch metadata keep their centroid-based
  convention.
- `Emission direction (azimuth)`: original velocity azimuth about +Z, with +X
  at 0 degrees and +Y at 90 degrees. Full launch directions preserve the correct
  hemisphere, unlike slopes alone.
- `Emission angle to normal`: original angle to the local outward surface
  normal, using a fixed 0–90 degree colour scale.
- `Interaction type`: the existing recorded scattering/channel classification.

Position, angular and phase-space plots share these labels. Hover shows launch
angles when available. Colours retain source ancestry after clipping, focusing,
projection rotation and specimen scattering. Neither changing colour nor moving
Z retraces the beam. User plot ranges remain fixed. The angular plot coordinates
are **current** angles; emission colours describe **original** angles.

Actual launch positions, unit directions and normals are captured before the
field-mesh launch adjustment and before common-Z history resampling. They are
saved with incident cache seeds. Old caches without them remain readable and
show neutral grey for unknown launch angles; later slopes are not substituted.

## Bounded propagation check

Reproduce without writing a cache or changing saved settings:

```powershell
.venv/Scripts/python.exe scripts/check_particle_source_transport.py --rays 49 --step-mm 0.5
```

The normal optical-only pipeline executes the physical tip, extraction,
acceleration, gun optics, apertures and the column. No specimen signal or wave
calculation is requested. Changing condensers reuses the executed gun result.

Observed for the current default assembly and 100 nA reference emission:

| In-memory C1 / C2 | Gun-exit rays | Sample-reaching rays | Main post-gun stop | Maximum live column angle |
|---|---:|---:|---|---:|
| 90% / 35% | 20 / 49 | 0 | Column wall | 29.50 degrees |
| 5% / 5% | 20 / 49 | 0 | C2 aperture (20 rays) | 2.496 degrees |

All 49 emitted rays retain launch-angle metadata. All live sampled coordinates
were finite. Maximum transmitted gun energy error was 2.22e-9 eV, below the
existing 0.001 eV budget. The second optical-only calculation reused the gun;
the first was approximately 49 seconds, the second approximately 0.48 seconds
on this host. These are small-ray diagnostic timings, not imaging benchmarks.

These are **not accepted imaging presets**. Default fields produce large slopes
outside a reliable small-angle regime. Passing the wall at weaker excitation
does not imply passing the next physical aperture. Stopped-ray diagnostic
continuations still produce overflow warnings farther downstream; they are not
live electrons or valid extrapolated signals. The field's mesh launch displacement
was approximately 4.76 nm and no field-grid convergence certification was made.
Energy conservation alone does not certify source shape or optics accuracy.
Condenser matching, aperture acceptance and column approximation validity remain
separate work; no wall/aperture was removed or silently widened here.

## Validation

Focused tests cover sample counts, equal-area weights, multiple directions at
one site, zero-spread limits, energy moments, local-angle cutoffs, legacy profile
compatibility, cache invalidation/persistence, full gun energy checks, and GUI
colour invariance under projection and reordered scattering descendants.
GUI fixtures are synthetic/offscreen, not hardware or full TEM/STEM acceptance.

The combined focused run passed 253 tests in 186.05 seconds. Two additional
legacy/curved Medium-sampling checks passed. After the final presentation-only
changes, all 30 beam-analysis/colour tests passed again. Compilation and
`git diff --check` passed. The full repository suite and full high-accuracy
TEM/STEM imaging were not run. Existing stopped-ray overflow warnings described
above remain an explicit propagation limitation, not a passed accuracy claim.
