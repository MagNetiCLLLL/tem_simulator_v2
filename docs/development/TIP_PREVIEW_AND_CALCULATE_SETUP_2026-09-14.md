# Tip preview diagnosis and calculation setup

## Observed early stop

Reproduced the user's C1 stop with the default FEG, C3 + Probe Corrector,
operating-mode pair, Ideal Optics and 49 ordinary surface-emission samples.
No source position, aperture, voltage or lens strength was changed for the
comparison. Coherent calculations were not run.

| Setting | Gun-exit ordinary rays | Ordinary rays at specimen | C1 downstream stop range |
|---|---:|---:|---|
| Vacuum on, 1 mm column step | 21 / 49 | 0 | 510.987–519.217 mm |
| Vacuum off, 1 mm column step | 21 / 49 | 0 | 510.991–519.217 mm |
| Vacuum on, 0.1 mm column step | 21 / 49 | 0 | 510.985–519.214 mm |

The gun exit coordinates span approximately ±1.9 mm in X/Y with slopes up to
1.67 mrad. C1 focuses this sampled bundle to a crossover and it subsequently
intercepts the physical bore. Changing the column step by a factor of ten moves
these stops by less than 0.004 mm. Normal vacuum does not explain their loss.
This is not evidence that the entire continuous emission distribution has zero
transmission: a 49-point quadrature does not resolve a very small acceptance.
Nor does it qualify the existing lens preset for the new curved-tip source.

## Corrected diagnostic path

The curved-surface emission branch returned before the existing tuning support
probe hook. It therefore did not provide the support diagnostics advertised by
Medium tuning. The surface path now has its own physical-tip probes:

- Preview: 48 ordinary emission samples and one zero-current apex-normal probe.
- Medium: 160 ordinary samples and 33 zero-current probes on the cap, including
  the apex and successively smaller cap-area radii.
- High accuracy emission is unchanged and adds no diagnostic probes.

Every probe starts on the configured spherical tip, uses the source's mean
emission energy and traverses extraction, acceleration, real gun apertures and
the column. It is never injected at the gun exit or specimen plane. Probes have
zero current weight; the ordinary sample weights sum to one. Numerical probe
settings enter gun trace cache keys, and the calculation input schema changes
invalidate older tuning products.

The executed apex probe reaches the specimen and is absorbed by the inserted BF
detector downstream. It does not pass through an absorbing detector to the
camera. This confirms an open on-axis route; it does not establish a transmitted
physical current or repair condenser alignment. No bore enlargement, aperture
removal, lens retuning or artificial field gap was used.

Ray Diagram draws probes as white dashed paths and reports their surviving count
separately from ordinary emission samples. If no ordinary ray reaches the sample,
the heading reports unresolved current and points to alignment/sampling. Saved
zoom remains under user control; Fit shows the full currently traced route.

## Calculation settings and vacuum map

The top toolbar's Calculate setup dialog controls STEM frame output, EDS,
residual-medium transport and seeded Poisson counts through their actual shared
state fields. It does not change physical insertion states. TEM/coherent STEM
and 4D-STEM controls remain visibly paused; historical explicit requests can be
turned off without silently converting profiles. New states use classical
particle readouts. Settings persist in operating profiles and working points.

Vacuum map is now a standalone tab immediately after Sample with a horizontal
beam diagram. Its component-anchored region ends remain independent. Gas/vacuum
gaps receive analytic linear-pressure transitions; their clipped path integrals,
collision species and removal coordinates participate in transport. See
`VACUUM_MAP_2026-09-14.md` for the interpolation convention and model limits.

## Verification

83 distinct targeted regression tests pass in their latest runs, including
the executed tip-to-BF chain, gap integrals/removal coordinates, profile
round-trips, tab order, independent GUI boundaries, and dashed probe display.
Scalar evidence: `evidence/vacuum-setup-preview-checks-2026-09-14.json`.
Generated screenshots and raw test logs remain local under
`outputs/vacuum-validation/`. No numerical array outputs were added to Git.
