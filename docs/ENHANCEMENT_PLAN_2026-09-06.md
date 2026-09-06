# Enhancement plan: interaction, result integrity and geometry validation

Date: 2026-09-06

Stage 1 implementation and validation update: 2026-09-07.

Status: Stage 1 rendering changes implemented; end-to-end performance acceptance
remains open. Stages 2-6 remain proposed. See the
[stage 1 implementation evidence](RAY_INTERACTION_PERFORMANCE.md#stage-1-incremental-result-presentation).

## Objectives

Improve interactive response, preserve completed calculation results, and make
the physical consequences of mechanical changes verifiable. Extend the existing
pages and cache infrastructure instead of adding duplicate imaging interfaces.

## Delivery order

Complete stages 1-3 first for day-to-day interaction and result reliability.
Follow with persistent products in stage 4, then aberration validation in stage 5
before using those metrics to compare geometry variants in stage 6.

| Stage | Scope | Status |
| --- | --- | --- |
| 1 | Incremental Ray Diagram rendering | Implemented; performance target open |
| 2 | Unified result identity and history | Planned |
| 3 | Background high-accuracy preparation and stage scheduling | Planned |
| 4 | Additional persistent scientific products | Planned |
| 5 | Field-derived aberration convergence | Planned |
| 6 | Isolated mechanical geometry variants | Planned |

## 1. Incremental Ray Diagram rendering

Baseline: new calculation results rebuilt the ray scene. Separate mechanical
structure, labels, rays, magnetic fields and diagnostic markers into independent
layers. Update only the layers affected by a new result, and measure calculation
time separately from rendering time.

Acceptance criteria:

- Moving the selected Z plane, rotating or zooming does not start physics work.
- User-defined plot ranges and the beam coordinate origin remain unchanged.
- Unchanged geometry retains its static graphics when new rays are published.
- Repeated adjustments do not accumulate duplicate graphics.
- On a declared fixed benchmark, target a 95th-percentile response of at most
  50 ms for display-only operations. This is a proposed target, not a measured
  result or a promise of real-time field solving.

## 2. Unified result identity and history

Extend the existing result-source controls with calculation time, result identity,
accuracy, physical model and differences from the current parameters. Use concise
states such as `Current`, `Outdated` and `Approximate`. Allow users to retain and
compare results without introducing another imaging page.

Acceptance criteria:

- Viewing a retained result does not change live parameters or submit work.
- A result from different settings is never labelled as the current calculation.
- Relevant parameter differences and cache reuse limitations are available.
- Viewing a result and explicitly restoring its parameters are separate actions.

## 3. Background high-accuracy preparation and stage scheduling

Move expensive high-accuracy request preparation off the GUI thread while
preserving memory checks and duplicate-request protection. Before execution,
show which stages will `Reuse`, `Recalculate` or remain `Unavailable`. Report
actual stage progress and improve cancellation without discarding completed work.

Acceptance criteria:

- Preparing a request does not block normal GUI interaction.
- The same task is submitted only once.
- Only dependency-affected stages are recalculated.
- Cancelled or superseded work cannot overwrite a complete result.
- Remaining-time estimates, where available, are labelled as estimates rather
  than presented as measured completion progress.

## 4. Additional persistent scientific products

Extend the existing disk cache with versioned products for specimen exit waves,
projection-system checkpoints, TEM images and EDS spectra. Large angle-resolved
STEM products are opt-in. Check input identity, model compatibility and integrity
before loading retained products.

Acceptance criteria:

- Compatible results can be restored after restart without repeating their
  completed source or specimen calculations.
- Changed inputs or models invalidate the appropriate dependent products.
- Stored scientific arrays retain their values and precision.
- Cache corruption, interrupted writes or quota exhaustion do not destroy a
  previously complete result.
- Users can control retained products and storage limits; saving every large
  intermediate is not compulsory.

## 5. Field-derived aberration convergence

Extend **Model Inspector -> Field validation** rather than adding a new page.
Existing checks primarily cover magnetic fields and local paraxial quantities.
Add Cs/Cc convergence studies across field mesh, ray step, pupil sampling and
symmetric energy perturbations at unchanged physical settings.

Acceptance criteria:

- Report coefficient changes, fit residuals and validation status separately.
- Use declared tolerances, with absolute floors for near-zero coefficients.
- Distinguish numerical convergence from pupil-dependent fitting and independent
  reference validation; an unchecked comparison must not be marked as passed.
- Do not tune presets to force agreement or add empirical aberrations twice.
- Validation studies do not overwrite production images or live settings.

## 6. Isolated mechanical geometry variants

Extend **Design Explorer** with detached axisymmetric geometry variants, starting
with supported pole gaps, bores, tapers, thicknesses and lens positions. Resolve
each assembly, check geometry and model readiness, and reuse compatible cached
stages. Compare fields, throughput, beam size and validated aberration metrics.

Acceptance criteria:

- Baseline source files and live settings remain unchanged during a study.
- Physical drawing, field material masks and electron clipping use the same
  resolved geometry for each variant.
- Invalid bores, overlaps or incompatible circuit configurations fail before
  expensive calculation.
- Changes to shared magnetic structures invalidate every affected channel.
- Returning to an identical retained variant reuses compatible results.
- Applying a valid variant is an explicit user action.
- Initial support is limited to axisymmetric geometry represented by the current
  solver; it does not imply arbitrary three-dimensional design capability.

## Scope and implementation rules

- Keep all project UI and documentation in English.
- Keep visible descriptions short; place detailed explanations in tooltips or
  existing diagnostic views.
- Preserve user layouts, plot ranges and completed high-accuracy results.
- Keep TOML-backed geometry authoritative and separate from operating controls.
- Do not automatically recalculate lens presets after geometry changes. Preset
  optimisation requires an explicit request or the agreed final assembly step.
- Do not add tomography or ptychography in this plan. Coupled multiphysics remains
  a separate future phase.
- Validate changed and directly affected behaviour with focused regressions.
  Record actual measurements separately from planned acceptance targets.

## Related project documentation

- [Ray interaction performance](RAY_INTERACTION_PERFORMANCE.md)
- [Interactive calculation and result sources](INTERACTIVE_CALCULATION.md)
- [Workspace layouts](WORKSPACE_LAYOUTS.md)
- [Simulation modes](SIMULATION_MODES.md)
- [Magnetic field validation](MAGNETIC_FIELD_VALIDATION.md)
- [Magnetic circuit models](MAGNETIC_CIRCUIT_MODELS.md)
- [Existing six-stage physics implementation](SIX_STAGE_PHYSICS_IMPLEMENTATION.md)

This roadmap is distinct from the existing six-stage physics implementation.
Stage status distinguishes implemented behaviour from outstanding acceptance
targets; proposed later stages are not implied to be complete.
