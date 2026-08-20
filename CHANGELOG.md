# Changelog

## Unreleased

- Made the Iliad Energy Filter permanent hardware. Instrument Setup now
  selects only the gun and column, the no-filter recording module is retained
  only as validated historical geometry, and legacy no-filter selections are
  normalised to the installed Energy Filter when applied or loaded.
- Added probability-conserving Real-sample inelastic transport without
  restoring artificial diffraction beams. Specimen TOMLs now own measured
  200-keV total/plasmon IMFP anchors and provenance for Si, Au and the carbon
  model. Relativistic log-angle scaling handles plasmon voltage dependence;
  BEB with the stated U=B approximation scales the measured residual
  core-ionisation rate. Independent Poisson statistics produce zero-loss,
  single plasmon, single ionisation, optional other loss and plural-event
  populations. Optional effective absorption/removal defaults off.
- Real energy-loss ray populations carry representative loss energies into
  chromatic and Energy Filter transport and sample rotationally balanced
  characteristic scattering angles. Ray hue identifies the energy-loss state;
  the interaction kick is removed before convergence brightness is computed.
  A new Ray Diagram panel reports, at any selected Z, sample probability,
  source fraction reaching the plane, local composition, loss energy,
  pre/post-sample stops, effective absorption and numerical conservation.
- Removed the artificial `000/+g/-g` and diffuse-jitter branch model from Real
  sample Ray Diagram calculations. Coherent elastic diffraction/scattering
  remains a high-accuracy wave/multislice result. Manually defined ray interactions are
  confined to Virtual sample mode. Ray hue now identifies the interaction
  type, while five dark-to-bright levels encode each ray's sample-plane
  convergence semi-angle relative to its branch's weighted chief ray, with
  brightness saturating at the calculated 99%-current semi-angle.
- Consolidated all specimen editing in the central **Sample** workspace. The
  duplicate sample node and quick controls were removed from the left
  instrument tree, layout clicks now open the central page, and previously
  side-only preset, TEM/STEM wave, multislice, frozen-phonon and Virtual
  interaction settings remain directly editable there. The startup preview
  now uses the cancellable debounce timer so it cannot race a navigation click
  and overwrite its status message.
- Added a TOML-driven **Direct Alignment** page with four background,
  transactional coupled controls: C2/C3 Nanoprobe convergence, C2/C3
  quasi-parallel Microprobe area, D/I/P1/P2 Image magnification and D/I/P1/P2
  effective camera length. Metrics use rotation-invariant current containment
  and complete transverse transfer matrices; fine-step conjugate validation,
  stale-state rejection and exact rollback prevent failed targets from changing
  live lens values. Requested projector ranges remain distinct from the
  presently verified non-OEM reachability. Production validation now includes
  affine kicks, exact aperture planes, column walls, strict ray weights and a
  configured optimiser/fine-grid stability limit.
- Rebuilt the Iliad branch manifest around the supplied public evidence: one
  large tapered prism, ten individually defined multipoles, XO/optional EFTEM
  slit, dynamic-focus electrostatic quadrupole, bias tube, fast shutter,
  camera deflector, EFTEM output plane and Zebra now have unique TOML rows and
  click-to-edit names. Unpublished coordinates and envelopes remain explicit
  adjustable non-OEM parameters. Corrected each Zebra strip to a 28.672 x
  0.800 mm active area while retaining 28.672 x 3.584 mm for the separate 2-D
  alignment region, and kept the unmodelled electrostatic-quadrupole field
  visibly disabled rather than fabricating its optics.
- Made the selected instrument TOMLs the enforceable final authority for TEM
  structure. Removed the gun's default-root re-read, propagated custom catalog
  roots through runtime geometry, assigned every active part a unique
  variant-scoped source ID, canonicalised Objective Stigmator identity and
  stopped saved state from retaining manifest-owned structural fields.
- Added catalog-wide uniqueness checks, required structural-field validation,
  TOML-derived Energy Filter/sample bootstrap geometry and regression coverage
  that instantiated all 30 recording-system variants supported at that stage
  without an override or fallback path.
- Rebuilt the D-I-P1-P2 mechanical geometry in both recording-system TOMLs
  from the supplied FEI public-reference engineering reconstruction.  Added
  independent yoke, coil, pole shoulder/bore/gap/nose and 0.75 mm liner
  dimensions with explicit non-OEM provenance, while preserving every lens
  centre, optical reference and excitation parameter.
- Made projector-lens Python fallbacks read their mechanical values from the
  authoritative TOML, updated Physical Layout pole-nose rendering and added
  validation/regression coverage for the reconstructed radial and axial
  hierarchy.
- Made every ray-optics RK4 propagation hit its requested physical endpoint
  exactly. CPU, Numba and CUDA retain full requested steps and use a shortened
  final interval, eliminating the finite blue/yellow bundle overlap at the
  specimen without display-only clipping.
- Recalibrated the probe two-hexapole pair at the exact specimen plane while
  preserving its orientation and all mechanical coordinates. Added endpoint,
  sample-interface and accelerated-backend regression coverage.
- Added full signed 2x2 `J_img` and `J_diff` sample-to-plane outputs, including
  rotation, reflection, anisotropy and conjugacy diagnostics. Scalar image
  magnification and effective camera length now derive from the determinants
  of these coupled X-Y maps instead of taking one unsigned X coefficient.
- Added an **Optical Transfer** page that can capture one Image state and one
  Diffraction state at the same plane and calculate their normalised relative
  direction map. Camera-axis rotation, flips, uncertainty and provenance are
  TOML-owned; shipped identity values are explicitly uncalibrated and cannot
  support an absolute hardware crystal-orientation claim.
- Moved every magnetic-lens effective Bz sign from Python tables into the
  selected instrument TOMLs as validated `field_polarity` values with explicit
  status/source provenance. Microprobe/Nanoprobe Mini Condenser reversal now
  comes from the operating-mode TOML, while ordinary recalculation preserves
  runtime overrides.
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
