# Grounded particle-gun field and matching audit — 2026-09-14

## Scope and decision

Classical tip emission only. No coherent-wave execution, downstream source,
source-size reduction, fitted field multiplier, aperture removal or cache-array
export was used. This is a small-ray numerical/design audit, not OEM calibration
or full TEM/STEM imaging acceptance. Emission remains 100 nA from the existing
100 nm-radius, 10-degree-cap tip with the existing local energy/direction law.

The requested matching bounds are extraction 4–5 kV and gun-lens control
1–1.2 kV. Moving a lens closer reduces its entrance footprint, but does not by
itself establish acceptable gun-exit transport. The tested poor-transmission
arrangements were not promoted to the default assembly.

## Corrected boundary discretisation

The old reconstruction moved launch points onto a staircase metal boundary:
the 49-ray default displacement reached 4.762621 nm. The field's tangential
fraction relative to the physical surface normal was approximately 7–17%.
The analytic surface itself could have a nonzero work potential in that map.

The production solve now uses boundary-conforming triangular finite elements
in `s = r^2, z`. The axisymmetric weak form is

`integral(4*s*phi_s*v_s + phi_z*v_z) ds dz = 0`.

Triangles are clipped against the spherical-cap/tangent-cone contour. Launch,
metal absorption, potential and its gradient share that represented boundary.
Uncut cells reconstruct the solved nodal potential bilinearly in `(s,z)`;
this preserves continuity at cell edges and avoids unnecessary diagonal force
jumps. Cut cells retain the conforming triangular potential. This is a change
to spatial discretisation, not an additional force or an accelerated launch.

The discrete-gradient relativistic integrator and its 0.001 eV exit energy
budget are unchanged. The field/cache identity was versioned. The old
finite-volume mathematical helper and historical interpolation fixtures remain
available; active surface-particle calculations use the new field.

## Mesh evidence, unchanged electrode positions and controls

Extraction 4 kV; gun lens 1.2 kV relative to the extractor. Extractor centre
8 mm, lens centre 18 mm, exit plane 450 mm. Same 49 tip samples and 0.2 mm
maximum gun integration step in every row.

| Requested radial/axial nodes | Actual grid | Max launch displacement (nm) | Diameter at lens (mm) | RMS radius at lens (mm) | Passed exit rays |
|---|---|---:|---:|---:|---:|
| 160 / 320 | 166 × 461 | 0.0000395864 | 4.134727 | 1.440392 | 35 / 49 |
| 320 / 640 | 326 × 888 | 0.00000593675 | 4.130037 | 1.438915 | 37 / 49 |
| 640 / 1280 | 646 × 1741 | 0.000000171352 | 4.128788 | 1.438941 | 38 / 49 |

The two successive lens-plane diameter and RMS changes are below the 1%
comparison gate. The roughly 4 mm footprint is therefore not explained solely
by the old launch-grid error. The exit-count changes do **not** meet a 1%
transmission-convergence claim; aperture-edge classification and larger sample
budgets still need separate qualification. Small residual and energy errors
are not substitutes for that test.

The measured tip-field tangential fractions for these samples decreased to
approximately 0.00924%, 0.00241%, and 0.00109%, respectively. Exit energy
errors in these runs were below 4e-9 eV. The scalar Laplace residual was below
1e-15, but that alone says nothing about mesh accuracy.

An intermediate all-triangle reconstruction took 257 s at the default grid;
the retained cut-cell/bilinear reconstruction took about 84 s, and the two
refined runs about 80 and 83 s. These wall times were measured with other
bounded diagnostics running and are not isolated performance benchmarks.

## Voltage reference is a physical input

The existing model added the gun-lens control to the extractor potential.
That convention was previously implicit. `voltage_reference` now explicitly
accepts `tip`, `extractor`, or `ground` for the solved surface-particle model.
The source editor displays the resulting electrode potentials relative to
final-anode ground. The operating parameter and saved profile carry the
reference; changing it invalidates field/transport reuse.

For HT = 300 kV, extraction = 4.5 kV and gun-lens control = 1.1 kV:

| Lens control reference | Tip (kV to ground) | Extractor | Gun lens | Final anode |
|---|---:|---:|---:|---:|
| Tip | -300 | -295.5 | -298.9 | 0 |
| Extractor (historical default) | -300 | -295.5 | -294.4 | 0 |
| Ground | -300 | -295.5 | +1.1 | 0 |

These are different physical boundary conditions, not interchangeable labels.
Radial focusing and axial work are derived from the same potential. A ground-
referenced +1.1 kV electrode is 301.1 kV above a -300 kV emitter.

Missing references in historical gun payloads, profiles and exact working-point
graphs retain the original additive extractor convention. Only this known
schema extension is accepted; the snapshot round-trip still verifies every
previously captured value. No saved graph is rewritten. Historical analytic
guns keep their original additive convention rather than silently ignoring a
new reference selection.

Both FEG assembly defaults still explicitly select `extractor`. The user's
question about a tip-referenced interpretation is not treated as proof of
the physical machine's power-supply reference.

## Position and reference comparisons

49 rays; extraction 4.5 kV and lens control 1.1 kV; all original electrode
lengths, bore sizes, accelerator stages and apertures retained. Only the stated
centres/reference changed in detached diagnostic states, not saved defaults.

| Lens reference | Extractor centre (mm) | Lens centre (mm) | Diameter at lens (mm) | Passed exit current fraction |
|---|---:|---:|---:|---:|
| Extractor | 8 | 18 | 4.139872 | 35.8796% |
| Extractor | 8 | 14.5 | 3.327715 | 8.3333% |
| Extractor | 4 | 11 | 2.502320 | 2.0833% |
| Tip | 8 | 18 | 5.050374 | 0% |
| Tip | 8 | 14.5 | 4.077246 | 0% |
| Tip | 4 | 11 | 3.212561 | 4.1667% |

In the tip-referenced 8/18 mm case the envelope contracts to 1.576 mm at
Z=30 mm, then expands after its crossover and is clipped downstream. This is
real focusing in the model, but not a usable gun/column match. A small local
spot must not be reported as successful full transport.

A further tip-referenced trial used extraction 4.5 kV, lens control 1.2 kV,
extractor centre 2.5 mm and lens centre 9 mm. Their retained electrode lengths
put the bodies at 0.5–4.5 mm and 5–13 mm, respectively. The diameter was
2.650309 mm at the lens and 0.335187 mm at Z=70 mm, but subsequent expansion
still caused exit-aperture losses: 27 of 49 rays passed, carrying 54.1667% of
the prescribed source current. The exit energy error was below 2e-9 eV.
This is a promising local contraction, not a qualified full-column match;
neither this geometry nor its voltage reference was promoted to defaults.

The command-line report separates reaching a plane from passing its aperture.
`passed_exit_rays` and `passed_exit_fraction` are the accepted output; the
plane-envelope row can also include particles stopped at that exact plane.
Gun-only tracing does not apply the separate column drift-wall clipping.

## Reproduction and limits

Use `scripts/check_gun_envelope.py` with `--rays`, `--mesh-scale`,
`--extractor-kv`, `--lens-kv`, `--lens-reference`,
`--extractor-center-mm`, and `--lens-center-mm`. Control values supplied to
this matching audit are constrained to the user's requested voltage ranges.
It prints lightweight JSON; it neither edits profiles nor writes ray arrays.

The numerical boundary/gradient, analytical harmonic-potential comparisons,
energy conservation, source/profile round trips, cache identity and historical
voltage preservation have focused regression coverage. Full coherent imaging
remains paused. Large-ray transmission convergence and a calibrated source-to-
specimen match are not claimed by this audit.

### Final focused validation

The frozen-code run passed all 110 tests across the cut-field, grounded-tip,
gun-envelope, voltage-reference, static-energy, tip-assembly, optional-profile,
working-point contract and working-point GUI restoration suites. Compilation
of the changed Python files and the diff whitespace check also passed. The
full repository suite and full TEM/STEM imaging chain were not run.

The working-point persistence fixture explicitly checks that an unavailable
beam statistic remains unavailable when no rays survive; it does not qualify
source-to-specimen transmission. That fixture still emitted overflow/invalid
warnings from the downstream column ray integrator. Those warnings remain an
unresolved column-validation issue, despite the passing persistence assertions.
Pydantic deprecation and Qt signal-disconnection warnings were also observed.

## Primary references and interpretation

- [JEOL: electrostatic lens](https://www.jeol.com/words/semterms/20121024.051158.php)
  explains that the extraction and acceleration electrodes jointly form an
  electrostatic lens, which can accelerate or decelerate electrons.
- [FEI patent application US20180323036A1, Fig. 2 description](https://patents.justia.com/patent/20180323036)
  gives an example with a 0.05–2 kV variable electrostatic lens, approximately
  4.5 kV extractor, approximately 4 kV anode and subsequent acceleration stages.
  Interpreting these as gun-local potentials is consistent with that voltage
  sequence; this does not identify the user's machine's electrical reference.
  The same design also includes permanent-magnetic source pre-focusing, absent
  from the present reference gun. It must not be assumed to describe every FEG.

No additional magnetic pre-lens or separate anode has been invented or silently
installed. Adopting that topology requires an explicit reference design and its
geometry, material/field parameters and electrical connections.
