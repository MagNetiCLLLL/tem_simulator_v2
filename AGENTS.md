# Physical source and cache requirements

These requirements were explicitly specified by the user on 2026-09-11.

- Preserve existing physical components and capabilities. A numerical method
  may change, but it must not skip extraction, acceleration, focusing, apertures,
  or other already modelled gun/column operations.
- For the FEG, custom electron-source inputs belong only to emission from the
  tip (current/brightness, spatial/angular distribution and energy spread).
  Do not introduce an independently configurable accelerated, gun-exit,
  specimen-plane or other downstream source.
- The user explicitly permits adding new FEG tip parameters, including a
  physically defined coherence/phase model. This does not permit defining an
  independently configurable state after extraction or acceleration.
- An equivalent beam state is a cache of an executed upstream calculation,
  not a new source. Its identity must include the consumed upstream optics,
  source, model and numerical inputs. Changing a relevant input invalidates
  reuse. A label, digest binding or manual recalibration is not transport.
- Keep historical results readable without admitting prohibited historical
  exit-source inputs to active calculations. Do not silently convert profiles.
- Missing coherent tip-to-gun-exit physics must remain explicit. Rejecting an
  unsupported request is not completion of that physics or full TEM/STEM
  acceptance. Isolated mathematical fixtures do not qualify the full chain.
- Preserve the full modelled electron/wave state across stages. Users may
  choose optional observables and numerical budgets, but deselecting a
  readout must not remove phase, physical interactions or detector absorption.
- Detector phase readout is explicitly requested. Preserve per-mode complex
  fields and phase references; do not invent one aggregate phase for an
  incoherent mixture or confuse simulated phase with direct hardware readout.
- Dynamic scan and outgoing inelastic waves must use that same tip-origin chain.
  Segment expensive execution and release completed wave buffers. The user
  permits higher configurable cache budgets for the 96 GB host; numerical
  resource choices must not silently remove modelled physical effects.
- Integrate the energy filter last. A detector physically before its entrance
  does not traverse it; a requested path reaching it must not bypass it.

## Current scope and generated data (2026-09-13)

- Coherent tip-to-column wave development is paused at the user's request.
  Do not restart long wave calculations or silently enable a coherent source.
  Keep historical wave code and profiles readable. Current work uses classical
  particle emission with editable physical tip geometry and local emission.
- Curvature radius, cone angle and emitting-cap angle are distinct quantities.
  Derived patch diameter, depth, surface area and arc length are not independent
  downstream source inputs. Preserve extraction, acceleration and apertures.
- Do not commit generated calculation caches or numerical array outputs.
  Keep them locally; retain lightweight reports, source code and input settings.
  Actual microscope acquisition records are not calculation caches and remain
  eligible for version control. Do not rewrite published Git history without
  explicit authorization.

## Vacuum calculation policy (2026-09-14)

- Vacuum scattering / attenuation is opt-in and disabled by default. Users
  normally choose it before the first Preview; later changes remain allowed.
- Changing vacuum participation or active vacuum settings may invalidate all
  calculation stages. This broad cache invalidation is explicitly permitted.
- Preserve explicit on/off choices in saved maps, profiles and snapshots.

## Scientific scope and naming (2026-09-18)

- The target is physically correct mechanisms and qualitative parameter-response
  trends for the simulator's own mechanical structure. Reproducing numerical
  settings or absolute performance of a commercial microscope is not required.
- Instrument records may inform topology, interactions and validation hypotheses.
  Do not import their currents, sensitivities or magnifications as authoritative
  settings for a different geometry. Compare trends only after matching coordinate
  conventions, operating regime and held/fitted controls.
- Use scientific or functional equipment names in the simulator's interface,
  component labels and explanatory text. Do not label simulated components with
  commercial instrument or product names. Name the sensor technology only when
  the implemented model supports that distinction.
- Preserve original acquisition metadata, reference URLs, historical files and
  compatibility identifiers. Present functional labels without rewriting the
  underlying evidence or changing device selection semantics.
- Define the range and controls held fixed for each trend check. Do not assume
  global monotonicity across crossovers, saturation or changes of optical mode.
  Qualitative agreement does not waive unit, conservation or numerical-convergence
  checks, and does not by itself qualify the complete microscope chain.

## Particle performance and continuation (2026-09-19)

- Numerical work may use at most half of the logical CPUs available to the
  process, with at least one worker and respect for explicitly lower limits.
  Apply the budget in calculation workers and prevent nested library pools
  or simultaneous numerical jobs from multiplying that budget.
- Keep user-selected cutoff planes, full-precision executed checkpoints and
  dependency-checked continuation available for classical particle work.
  A saved cutoff is an executed upstream state, never a configurable source.
- Automatically archive each accepted completed particle calculation locally.
  Show the actual calculated endpoint, resumable endpoint, quality, population,
  save time and path. Failed, cancelled or stale work must not be reported as
  a successful current archive. Generated archives remain excluded from Git.
