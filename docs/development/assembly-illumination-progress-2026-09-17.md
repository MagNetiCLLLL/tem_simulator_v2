# Assembly illumination progress — 2026-09-17

## Status

The requested all-assembly default bank is **not complete and not installed**.
Historical operating presets and the open application's state were not changed.
Classical particle transport was used throughout; coherent tip-wave work and
new TEM/STEM wave-image qualification remain paused.

The executed simulation mode was `custom` (Custom / Per-lens Models), with
vacuum participation disabled. The earlier Ideal Optics description was a
metadata error, not the mode used for these calculations.

## Current and aperture policy

- Specimen current: 2–200 pA.
- C2 physical aperture diameter: 20–250 um.
- Source size, angular distribution and energy distribution remain unchanged.
- The authorised total-flux ceiling is selected once after fitting, normally
  for a 100 pA interior setpoint, then frozen during all independent checks.
- Measured transmission is retained. Surviving particles are not renormalised
  to manufacture the requested current. The ceiling cannot exceed 100%.
- The retained total-flux field is an ideal scalar control, not a reconstruction
  of the physical C1 spot-number/brightness relationship or space-charge solve.

## Microprobe results

All rows below use a 100 um C2 aperture and 0.001% total-flux ceiling. At 1024
emitted particles they deliver 100 pA and D95 = 2.000 um at the upper surface.
Checks used 512/1024 emitted particles, column steps 0.05/0.025 mm, and gun
steps 0.2/0.1 mm. The maximum measured relative variation in angle, diameter
and transmission must each remain <=1%; the original physical targets also
must pass at every setting. These finite-budget checks are not an absolute
error proof or a wave-image resolution claim.

| Gun / upstream assembly | Blanker | Alpha95 (mrad) | Particle illumination status |
| --- | --- | ---: | --- |
| FEG / C2 | No | 0.181884 | Passed the stated checks; independent crossover baseline recorded |
| FEG / C3 | No | 0.214001 | Passed the stated checks; independent crossover baseline recorded |
| FEG / C3 | Yes | 0.213421 | Passed the stated checks; independent crossover baseline recorded |
| FEG / C3 + Probe Corrector | No | 0.220911 | Numerical checks passed; independent six-crossover microprobe baseline explicitly approved |
| FEG / C3 + Probe Corrector | Yes | 0.216509 | Passed the stated checks; independent crossover baseline recorded |

The historical five-crossover reference is explicitly a nanoprobe/diffraction
reference. The user explicitly approved the new no-blanker corrected microprobe's
independent six-crossover baseline on 2026-09-17. Its count and ordered component
intervals are recorded in `illumination_targets.toml` and required by future
validation. Previously measured values are unchanged; this permission update
did not rerun calculations. The nanoprobe five-crossover constraint is unchanged.

Complete propagation-plan, gun-input, wall and aperture comparisons show
upstream equality for each row's matching downstream image-corrector and
energy-filter variants. That gives 18 full assembly/microprobe combinations
with passed particle checks and an authorised crossover baseline.
It does not validate any post-specimen image or filter response.

## Remaining unresolved results

- FEG + C2 with blanker, microprobe: the 512-ray diameter is 2.02124 um;
  the 1% diameter/sampling limit is not passed.
- All six FEG nanoprobe assemblies were examined. At the fitted 1024-ray
  budget, several reach 30 mrad with nm/sub-nm geometric spots. None passed
  every independent check in this batch. C2 without blanker has approximately
  0.26 nm D95; diameter variation still exceeds the 1% numerical gate.
- Monochromated FEG candidates remain seriously under-resolved. Only 25/53
  particles reach the sample at 512/1024 emitted particles. With identical
  lens settings, C2 microprobe D95 changes from 1.1543 to 2.0000 um and C3
  nanoprobe alpha95 changes from 19.3316 to 30.0000 mrad. Both are rejected as
  defaults even though current remains within 2–200 pA.
- Thermionic bounded searches still produce micrometre-scale spots at 30 mrad
  or excessive microprobe convergence. A failed search is not a proof of
  physical impossibility. Strong pupil selection needs better weighted source
  sampling and source-specific condenser optimisation, not a smaller invented
  source or a flux rescaling presented as improved focus.

## Files and validation

The adjacent `assembly-illumination-evidence-20260917.json` archive contains
scalar observations, input strengths, source/implementation identities,
current bookkeeping, crossover roots and upstream-equivalence comparisons.
It contains no ray/wave arrays or generated calculation caches.

Relevant unit/regression checks: 56 illumination/current/search tests, 11
current/cache tests and 8 upstream-equivalence/baseline-policy tests passed.
The old dose-cache fixture initially failed at the deliberately closed
coherent-source boundary. It now explicitly isolates only the readout/cache
unit; a separate test confirms production wave admission remains closed.
These isolated tests do not certify new image production. No full-suite,
visible-GUI or hardware acceptance is claimed.

Next numerical work: resolve accepted-particle sampling for monochromated and
thermionic guns, refine unresolved nanoprobe branches, and validate larger
independent ray budgets. Only then assemble an input-bound default bank and
connect it to preset application without changing downstream user settings.

No calculation is left running. After approving the separate microprobe baseline,
the user requested direct Git publication followed by forced shutdown, without
repeating tests or calculations. No calculation caches are included in publication.
