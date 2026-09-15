# Tip-origin probe focus at the specimen entrance

## Current decision

**No qualified new-source nanometre probe has been obtained.** The 769-ray
draft profiles are rejected diagnostic candidates, not operating presets.
They match angle and local focus at one particle budget but have micrometre
spots and fail a sampling check. The Custom candidate also changes the
historical crossover sequence; the Ideal candidate has not qualified it.
See [the beam-path audit](PROBE_CHAIN_AUDIT_2026-09-14.md) for the executed
historical reference and separate tip, C1-waist and specimen measurements.

## Request and conventions

The user confirmed **current-weighted alpha95 = 30 mrad**. This is not a
physical aperture-edge angle. Positive Z points downstream. The sample Z is
its centre, so the requested entrance plane is `sample.z_mm - thickness_nm/2e6`.
For the current default 5 nm specimen at Z=1599.2 mm, the entrance is
Z=1599.1999975 mm. The specimen and objective positions are not moved to obtain
that target.

Work is limited to classical tip-origin particles. The current physical tip,
emission distributions, extractor, accelerator, gun lens and real apertures
are retained. Coherent-source development remains paused; no image calculations
were run in this task.

The initial comparison uses the default Nanoprobe/Diffraction operating pair
in Ideal Optics. The previous executed transport solution supplies C1 =
8.012821601837457% and C2 = 84.12563904641223%; these are explicit starting
controls, not new factory defaults. C3 and objective excitation are searched.

## Acceptance and implementation

`optics/surface_probe_focus.py` measures the **full-precision** incident state
at the exact entrance. Checkpoints must also be requested as exact integration
planes: checkpoint selection alone snaps to existing grid nodes. Plotting
arrays cannot qualify nanometre-scale focus.

Three planes, separated by 100 nm and all at or upstream of the entrance,
give a one-sided derivative of the current-weighted radial variance. A
positive second derivative distinguishes a true local waist from an envelope
maximum. The phase-space position/angle covariance supplies a second focus
diagnostic. No free-space extension through the sample replaces objective
field propagation.

The fixed-budget gates are:

- alpha95 within 1% of 30 mrad;
- both local-variance and covariance waist offsets within 1 nm;
- positive variance curvature, at least 16 positive-current rays and effective
  weighted sample count of at least 16; nonzero source current is required;
- both 0.05 and 0.025 mm column steps pass.

The earlier diagnostic imposed 1% source transmission. This was removed
after the user clarified that **crossover count**, not electron count, must
be preserved. Current-weighted effective sample support replaces that hidden
threshold; an explicit minimum-current requirement remains optional. See the
[numerical repair evidence](GUN_AXIS_MATCHING_REPAIR_2026-09-14.md).

The refinement returns a detached candidate only when these gates pass. Its
status explicitly says **PASS_AT_FIXED_PARTICLE_BUDGET**, not sampling
convergence. The user's current state is not edited. Active residual-medium
transport is not yet qualified for this routine and is rejected explicitly.

## First comparison: why one small ray bundle is insufficient

One first-order proposal, checked with the actual production propagator,
gave C3 = 32.72507667778783% and objective = 25.260664949735187%.

| Tip samples | Passing rays | alpha95 | Local waist offset from entrance | RMS radius |
| --- | ---: | ---: | ---: | ---: |
| 193 | 55 | 29.999986721 mrad | -0.998449 nm | 2.358193 um |
| 385, same controls | 112 | 29.090061502 mrad | +2467.336275 nm | 2.291887 um |

These are 0.05 mm-step measurements, not independently refitted states.
Both retain nonzero propagation into the projection chamber. The 193-ray
proposal is **not a sampling-converged focus calibration**. Its small angle
residual alone would have hidden the sample-dependent focal shift.

The spot is micrometre scale, not an atomic-resolution probe. For the
193-ray selected pupil, the two gun-exit symplectic RMS emittances were about
2.479e-8 and 3.459e-8 m rad. This diagnoses a broad outgoing phase space for
this particular pupil; it is not a proof that every possible gun/column
operating point is unreachable. The probe current, source brightness and
aberrations matter as well as convergence angle; see the manufacturer's
[probe diameter definition](https://www.jeol.com/words/semterms/20121024.063858.php).

## Follow-up at 769 particles: rejected candidates

The same C1/C2 transport settings above were retained. The two detached
profiles use the following C3/objective settings; neither changes defaults.

| Optics mode | C3 excitation (%) | Objective excitation (%) | Surface RMS radius | Surface d95 |
| --- | ---: | ---: | ---: | ---: |
| Ideal | 32.53341911740971 | 25.271080456726324 | 2.416905 um | 8.601969 um |
| Custom | 32.49174967180758 | 25.275237294853614 | 2.453495 um | 8.904571 um |

Both have alpha95 = 30 mrad and pass the two-step local-focus checks at
769 particles. The Custom candidate transmits 241 rays, carrying 31.3802%
of source current, through the specimen entrance and on to projection-chamber
entrance Z=2586.9 mm. This is optical transport, not image validation.

At the **same** Custom controls with 385 particles, alpha95 becomes
30.671892 mrad and the local waist is 4671.033 nm downstream of the entrance.
For the Ideal candidate, the corresponding waist shift is 6916.64 nm.
Neither candidate is sampling-converged. Correct convergence angle, a local
waist and nonzero downstream current are insufficient to qualify the requested
probe. The later topology audit additionally rejects the Custom branch.

`probe_qualification()` keeps spot diameter, historical component intervals
and sampling as separate gates. It is a diagnostic helper, not a newly
qualified GUI alignment mode. The existing two-control refinement explicitly
reports `NOT_QUALIFIED` for full probe acceptance even if its angle/focus fit
passes at fixed sampling.

## Reproduction and limits

`scripts/check_surface_probe_focus.py` executes the current physical tip and
uses cached first-order maps only to propose settings. Production validation
retains extraction, acceleration and physical stops. It writes no numerical
arrays. No generated calculation caches are intended for version control.

Analytical tests cover the entrance-plane convention, actual minima versus
maxima, invalid derivatives, zero-current support exclusion, exact
full-precision plane selection, detached state changes and rejected
refinements. These tests are not substitutes for the executed gun runs.
