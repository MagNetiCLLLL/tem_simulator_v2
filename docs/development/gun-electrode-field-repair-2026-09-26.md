# Coupled gun electrodes and downstream electrical closure — 2026-09-26

## Physical change and scope

The classical flat-cathode gun now has a single axisymmetric vacuum Laplace
field for the existing extractor, electrostatic gun lens and all ten
accelerator electrodes. Their mechanical centres, bores, outer diameters and
thicknesses define the conducting boundaries. Voltages are rises relative to
the original cathode: 0 V at the cathode, 4,000 V at the default extractor,
5,200 V at the default gun lens, and 300,000 V at the final accelerating anode.
The lens voltage comes from its declared voltage reference, not an analytic
field multiplier. Stage voltages remain the extractor voltage plus the
declared fraction of the remaining high tension.

The former independent smooth voltage ramps generated large alternating
radial fields near each stage. The repair evaluates the potential and electric
field of the coupled conducting geometry instead of smoothing plotted rays.
Because the physical field changes, upstream focusing and the eventual beam
size can change as well. This is not a promise to reproduce the old numerical
lens settings or to eliminate every physically possible transverse turn.

No emitted position, direction, energy offset, weight or ray identifier is
changed. Extraction, acceleration, gun-lens focusing, the physical apertures,
deflection, stigmation and any installed velocity selector remain required
operations. There is no downstream replacement source or energy reset.

## Electrical enclosure

The simulator explicitly assigns ground to the resolved vacuum liner from
the accelerator mechanical downstream face, 370 mm in the default geometry,
through the existing connected downstream tube. This assignment is a
simulator electrical design, not measured wiring or a commercial instrument
calibration. The upstream gun aperture is not silently grounded.

The actual liner changes diameter around the downstream gun components.
Those pieces are retained, including the existing tube from 450 to 1,534.5 mm
with a 5.76 mm inner diameter and 19.2 mm outer diameter. The radial faces
connecting steps in the liner are represented as ideal zero-thickness
grounded conductor faces. The full physical endpoints remain in the field
request; only the numerical domain is truncated for a solve.

The default field extends to 550 mm, beyond the physical gun-to-column exit
at 450 mm. Ground is imposed at that numerical cut through the continuing
grounded tube. The outer radial boundary is insulating, and the axis is
regular. The flat cathode is explicitly an ideal equipotential plane spanning
the radial domain, not the nanometre emission patch or a cone envelope.
Below that plane, constant cathode potential is available only as conductor
interior evaluation so a returning particle can be intercepted. Other
out-of-domain vacuum requests are rejected; they do not switch to an old
analytic field or clamp the final energy.

For an installed velocity selector, its existing housing is represented by
an axisymmetric common-potential shell tied explicitly to the gun-lens
potential. Its transverse electric and magnetic fields remain separately
composed. This is a declared common-bias model of the housing, not an
automatic local-velocity cancellation. The flat, no-selector qualification
below does not qualify the full selector path or a curved emitter.

## Numerical method and resource limits

`src/temsim/physics/closed_gun_field.py` reuses the axisymmetric finite-element
solver and evaluates a bilinear scalar potential in `(r², z)`. The electric
field is the exact negative gradient of that same interpolant. Every metal
face is represented at its physical coordinate. Generated floating-point
aliases are merged without shifting physical boundaries.

The electrode mesh resolves the bore at 8, 16 or 32 cells per radius in the
refinement study. Grounded-liner axial faces use half that resolution and a
local two-bore refinement neighbourhood; the vacuum side of each radial
liner face remains refined. Uniform stretches of grounded tube use a graded
axial grid. All three levels stay within the 1.5 million vertex limit, which
is checked before solving. Field builds execute with one numerical CPU
worker and serialized numerical-job admission.

The production accessor retains at most four fields in memory and uses a
256 MiB local generated-field disk cache. Compressed float64 arrays are
validated against archive, array and complete-request SHA-256 identities.
The request binds consumed electrode voltages, physical liner geometry,
electrical assignments, numerical settings, implementation files and library
versions. Emission count/current/angle sampling are not field dependencies
because this vacuum model has no space-charge feedback. They remain
dependencies of particle execution and its checkpoints. Generated field
caches are not particle archives and remain outside Git.

## Independent flat-particle qualification

The comparison uses the same 193 original emitted particles for every case.
An independent relativistic DOP853 trace executes from the original cathode
to 450 mm, with one-sided derivatives at field-cell boundaries, relative
tolerance `2e-9` and maximum step 0.25 mm. Exact physical aperture planes are
included as execution endpoints. Every saved state is checked for the
electrostatic invariant `K - potential_rise`, and every ray's final kinetic
energy is compared with its own initial kinetic energy plus 300 keV.

| Bore refinement | Numerical endpoint (mm) | Grid (r × z) | Exit RMS radius (µm) | Maximum final kinetic-energy residual (eV) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 550 | 163 × 841 | 0.300465108 | 1.90e-8 |
| 16 | 550 | 261 × 1,619 | 0.303262027 | 1.55e-8 |
| 32 | 550 | 436 × 3,156 | 0.304013982 | 1.54e-8 |
| 16 | 650 | 261 × 1,626 | 0.303263172 | 1.47e-8 |
| 16 | 450 | 258 × 1,548 | 0.303260060 | 1.13e-8 |

All five runs completed with all 193 rays forward and clear of the sampled
physical conductor/bore surfaces and both executed aperture planes. The
maximum energy-invariant error over all saved states and particles was less
than `1.92e-8 eV`. The final energy criterion of less than 1 eV was therefore
met without energy projection, resetting or a forced field-free handoff.
These diagnostic clearance samples are not qualification of arbitrary
large-angle collision or backstreaming events in the production integrator.

Refining 16 to 32 changes the maximum full-path RMS envelope by **0.2016% of
the reference peak**, the maximum ray position by **1.753 nm**, and the final
RMS radius by approximately **0.248%**. Extending the numerical endpoint from
550 to 650 mm at refinement 16 changes the full-path envelope by **0.00227%
of its peak**, the maximum ray position by **0.0197 nm**, and the maximum
flight time by **0.789 fs**. The physical liner remains unchanged in this
endpoint comparison. A relative slope metric normalized to the peak source
angle is not used to claim the same relative accuracy at a downstream focus.

The exit is naturally shielded: at 450 mm the solved potential along the
tested arriving rays is ground to numerical precision. The finite-domain
electric field can still change near the final electrode and liner entrance;
the trajectory comparison above quantifies that effect instead of inferring
global convergence from a small exit residual alone.

## Local evidence and before/after figure

- `F:\tem_simulator_v2\tmp\closed-gun-qualification\report.json`:
  executed per-case field identity, grid, CPU receipt, energy checks and clearances.
- `F:\tem_simulator_v2\tmp\closed-gun-qualification\comparisons.json`:
  mesh and numerical-domain trajectory comparisons.
- `F:\tem_simulator_v2\tmp\closed-gun-qualification\closed-gun-before-after.png`:
  physical X versus Z, with the same axis limits and ray colours before/after;
  the lower panel compares radial electric fields at X = 1 µm, Y = 0.
- `F:\tem_simulator_v2\tmp\closed-gun-qualification\plot-provenance.json`:
  baseline archive checksum, identical-emission checksum and new field identity.

The figure's baseline is the previously executed analytic 4 mm stage-ramp
case. Its archive checksum and state checksum are checked before loading;
the initial state, IDs and weights are compared bit for bit with the new
emission. The plot uses physical coordinates without the GUI transverse
display gain. It is not a new set of invented representative rays.

The final five-case rerun and refreshed figure bind these implementation
SHA-256 values, checked against the files after completion:

- `closed_gun_field.py`: `f94888445744ad33137261004b7ff23e2166129bd50c6fad814a8354c6e954ce`
- `axisymmetric_cut_field.py`: `32ae7fb03e20b154dcf67e3ea6561b981b6ea3bc90c7fe3f4bb13f39644dfd72`

## Scope of the independent reference

These results establish the stated flat, classical, vacuum gun case and its
declared refinement/domain checks. They do not validate a complete microscope,
arbitrary emission energies or angles, space charge, insulating materials,
coherent transport, a curved source, or an installed velocity selector.
Production event handling and additional checks are recorded below. Numerical
energy conservation alone is not evidence that an approximate field reproduces
a real instrument.

## Production integration, curved tips and physical events

The ordinary cold-field-emission path now executes the coupled field directly.
A flat tip uses `ClosedGunField`; nonzero continuous curvature uses
`ContinuousGunField`. Both use the existing original emission samples. The
curved conductor is the spherical cap and its tangent cone, clipped by the
declared mechanical shank. Its cut-conforming field mesh resolves that metal
surface. Launch positions and normals are not snapped to a mesh or replaced by
a gun-exit distribution. The existing explicitly selected advanced tip-surface
model retains its own grounded-tip field; it is not newly qualified by the flat
outlet study above.

A cancellation problem in very weak curvature was corrected at generated mesh
intersections: their axial coordinate is evaluated on the analytic conductor
contour, avoiding subtraction of nearly equal coordinates. This changes mesh
vertices, not emitted particles. In the 49-ray weak-curvature limit at
`1e-14 nm^-1`, the discrepancy from the flat provider decreases from 2.1126 nm
to 0.9594 nm when bore resolution doubles from 8 to 16. Both are below the
combined independently measured mesh changes, 9.6294 nm. This is a scoped
discretisation check. A finite curvature change need not produce an exactly
proportional exit displacement because it also changes the conductor field.

The compiled transport step evaluates the same scalar potential as its
reference implementation and retains the discrete electrostatic energy
invariant. It uses common relativistic constants throughout. Adaptive step
control, exact aperture-plane crossings, magnetic fields and any enabled
vacuum interactions remain active. Returning particles can be intercepted at
the actual cathode; grounded tube walls and their radial step faces also
absorb intercepted particles. Out-of-domain vacuum evaluation raises an error.
Exit kinetic energy comes from the integrated momentum, without a forced
300 keV reset or an average-emission-energy subtraction.

Actual nine-particle production traces for curvature `0.01 nm^-1` and
`1e-8 nm^-1` both reached the 450 mm exit with all particles. Maximum energy
invariant errors were respectively 3.26e-9 and 3.96e-9 eV. The original ten
emission arrays, emission reference and first equal-time history positions
were preserved exactly. At `0.01 nm^-1`, the finite field mesh gave a maximum
launch-potential discrepancy of 3.174e-6 V, recorded as numerical boundary
error rather than hidden by moving the launch positions.

The installed velocity-selector path was also exercised with nine particles.
Its housing common bias, transverse electric field, magnetic field and slit
were all active. After explicitly matching its E/B drive to the new local
kinetic energy, all nine reached the exit; maximum energy error was
3.376e-9 eV. Existing drives matched against the old field are not silently
retuned. This test does not establish energy resolution or a full selector
transfer map.

## Production step comparison and measured runtime

All timings below used one numerical CPU thread on a host exposing 32 logical
CPUs, with nested BLAS/Numba pools also limited to one. They are warm transport
measurements using an already prepared field and compiled step, and cover the
tip-to-450 mm gun path only. They are not a full specimen/detector run or a
measured speedup against an equivalent old implementation.

| Emitted particles | Maximum step (mm) | Transport time (s) | Reached exit | Exact same-result reuse (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 193 | 0.2 | 5.029 | 193 | 21.05 |
| 193 | 0.1 | 6.616 | 193 | 19.45 |
| 5,000 | 0.2 | 108.829 | 5,000 | 19.43 |

The 193-particle comparison binds identical original source arrays and field
identity. Halving the maximum step changes the actual integrated exit position
by at most 0.5904 nm, slope by 1.15e-9 rad and flight time by 1.56e-16 s. At the
120 mm and 445 mm physical aperture crossings, maximum position changes are
0.2560 nm and 0.5846 nm. All arriving populations agree. The 5,000-particle run
preserves the original emission and has a maximum exit energy residual of
4.831e-9 eV. Its completed result was saved before checking reuse; development
attempts cancelled while implementation identities changed are listed
separately and are not counted as completed timings.

The actual production exit is also compared directly with the independent
DOP853 reference using the identical field. Source positions, normalized
momenta, IDs and weights agree bitwise. Maximum exit-position discrepancies
are 0.8534 nm at step 0.2 mm and 0.4840 nm at step 0.1 mm; maximum slope
discrepancies are 1.66e-9 and 9.13e-10 respectively. This provides an independent
transport check in addition to step refinement and energy conservation.

The saved accepted integration histories, rather than display-resampled
polylines, show the average weighted X-slope sign reversals over 80–340 mm
decrease from 6.57 in the earlier compact analytic-ramp case to 0.233 in the
new field. Weak physical focusing events remain. There is no rule that
electron X or Y must vary monotonically while longitudinal motion is forward.

The default flat field has 422,559 vertices; its compressed disk cache is
about 2.92 MB. Disk loading in the 5,000-particle process took about 16 ms.
Caching avoids repeating an unchanged field solve or particle trace; the
108.8-second uncached particle trace still leaves room for further transport
performance work.

## Results, controls and regression coverage

The gun result now carries its electrostatic model and exit-energy report as
a declared field. Incident-state archives retain that report across save/load.
Instrument snapshots retain electrical geometry and numerical inputs but
exclude generated field solver objects. Execution identities include the new
provider and consumed boundary conditions, so an old-field checkpoint cannot
be reused for a new-field request. Historical results remain readable as their
captured results.

Unused analytic ramp widths, offsets and lens multipliers are no longer offered
as active controls for a geometry-field gun. Their historical stored values
do not alter its field. Ray Diagram's **Accelerator electrodes** overlay marks
physical electrode positions instead of fictitious narrow ramp bands.

The final recorded outcomes cover 223 distinct targeted tests across field
boundaries, compiled steps, gun transport, voltages, apertures, profiles,
history/TOF, artifact persistence, continuation and relevant GUI controls.
The initially failing weak-curvature boundary case passed after the contour
intersection repair. An additional 38 curvature/comparison/FEM checks passed,
including the updated, mesh-bounded weak-curvature limit. This is 261 distinct
targeted checks, not a run of the entire project suite.

The additional 38-case command was `python -m pytest
tests/test_continuous_tip_curvature.py tests/test_tip_curvature_comparison.py
tests/test_axisymmetric_cut_field.py -q` in the project virtual environment,
with one numerical thread and Qt offscreen. It exited 0 (21 + 13 + 4 cases);
its evidence is the completed tool execution record, not a saved JUnit file.
The new 24-case `test_closed_gun_field.py` is already included in the 223 count.

Reproducible local evidence:

- `tmp/gun-repair-20260926/production-transport.json`: completed 193-particle
  step comparison and exact physical-plane differences.
- `tmp/gun-repair-20260926/flat5000-step0p2.json`: completed 5,000-particle
  timing, source identity, cache reuse and energy report.
- `tmp/gun-repair-20260926/actual-source-verification.json`: original emission
  position and momentum compared bitwise with accepted history starts.
- `tmp/gun-repair-20260926/accepted-history-turns.json`: turn counts from
  accepted integration histories.
- `tmp/gun-repair-20260926/independent-endpoint-comparison.json`: source checks,
  archive hashes and actual exit differences against independent integration.
- `tmp/gun-repair-final.xml`, `tmp/gun-repair-continuation.xml` and
  `tmp/gun-repair-final-boundaries-ui.xml`: overlapping JUnit runs; latest
  outcome per test gives the 223 distinct passing tests above.
- `scripts/validate_closed_gun_transport.py`: production validation runner;
  generated arrays remain local under `tmp/`.

Restart the application and run a new calculation to use the repaired field.
The changed focusing can require new lens or velocity-selector matching.
Coherent development remains paused. No claim is made of full TEM/STEM
qualification, space-charge physics or reproduction of commercial settings.
