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
