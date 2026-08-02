# Changelog

## Unreleased

- Colour-coded the magnetic-field plot by the active solver formula and added
  formula names and expressions to its legend, tooltips and selection details.
- Rebased every magnetic-lens excitation range to a hard 100% maximum while
  preserving existing operating fields by increasing the affected peak-field
  calibrations.
- Prevented cold-FEG energy-tail samples from producing negative launch
  kinetic energies and isolated, step-sensitive electron trajectories.

- Removed the false ray kink at the gun/column boundary by resampling each
  time-integrated gun ray onto one strictly increasing common-Z path grid.
- Preserved the original equal-laboratory-time gun snapshots and per-ray
  arrivals at important gun planes, including an API for extracting the
  equal-time front at a plane's median arrival time.
- Treated the vacuum wall as a pure stop source: ray propagation remains valid
  outside the defined mechanical wall segments, while existing wall and aperture
  stops continue to compete by first physical intercept.
- Made every C2 catalog combination runnable through the full solver.
- Tuned the high-accuracy defaults for a 32 GiB workstation and added a
  conservative preflight memory-budget check for custom ray/step combinations.
- Made operating-profile application transactional and prevented profiles from
  overriding catalog-owned topology, installation state and lens geometry.
- Added a TEM wave-image/diffraction page for high-accuracy wave calculations.
- Included instrument and specimen TOML resources in built wheels.
- Replaced the single column diameter with TOML-owned, position-dependent
  vacuum inner diameters used by both ray clipping and visualization.
- Added stepped vacuum-tube projections to Ray Diagram and Physical Layout.
- Added explicit tapered upper/lower pole pieces for C1, C2 and C3.
- Removed the ambiguous global Column chamber diameter control; vacuum geometry
  is now edited on each component's TOML tab.

## 0.1.0 - Basic TOML assembly and ray-tracing MVP

- Replaced the PyCharm sample with an English-only PySide6 application.
- Migrated the validated headless optical, detector and simulation core.
- Added catalog selection for three gun, five column and two recording modules.
- Added generic operating-parameter and module-TOML editors.
- Added atomic TOML validation/rollback and per-part assembly-anchor inspection.
- Added debounced background ray previews for live lens adjustment.
- Added configurable one-shot high-accuracy calculations.
- Added TOML operating-profile load and save.

Current wave-imaging and STEM scattering models remain optional inherited
research backends. The MVP does not claim multislice or experimentally
calibrated quantitative HAADF accuracy.
