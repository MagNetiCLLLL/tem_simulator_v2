# Classical particle sections and live detector signals

The user-selected scope remains classical particles. Coherent tip-to-column
development is paused. Source emission, extraction, acceleration, focusing,
apertures and physical detector absorption remain active on the requested path.
This change targets qualitative physical behavior for the simulator's geometry.

## Workflow

1. In Live tuning, add the physical component controls and their allowed ranges.
   Stigmator axes, deflector offsets, and installed quadrupole/hexpole strengths
   and orientations now join lens excitations. Scan drive/calibration and
   mechanical geometry are not silently adjusted by these controls.
2. Enable **Section tuning (optional)**. Choose a named plane or an axial Z,
   and check all range components that will participate. Start live tuning.
3. The calculation stops at that plane. The unchanged prefix before the earliest
   physical field support/kick is eligible for reuse. A magnetic lens's center
   alone is not a safe restart boundary. Changed upstream inputs invalidate reuse.
4. After the matching calculation completes, **Save section…** writes a
   `.temsection` file. Choose another plane and participants to continue, or
   load the section in a later session using the same implementation and inputs.
   Loading a cache does not overwrite the live instrument or create a downstream
   source. A mismatched upstream stage is recalculated.

Ordinary Preview/Medium from the main window now runs the physical particle
path, including material interactions and requested detector products. The
lower-level optical diagnostic route remains available explicitly for existing
callers; it is marked optical-only and cannot report material detector signals.
Preview/Medium retain their displayed numerical particle budgets. They are not
a 5,000-particle High accuracy result or a convergence certificate.

## Physical products and limits

- With an inserted configured specimen, a downstream request computes finite
  specimen elastic transport and the existing aggregate inelastic distribution,
  even with scanning and EDS off. The current pixel uses the actual deflected
  incident centroid. Retracting the specimen removes its material interactions.
- Every elastic history now retains its actual material path length. A ray that
  misses the finite specimen receives a vacuum zero-loss channel. This metadata
  is independent of the 256-history display limit. Historical terminals without
  this information retain an explicitly identified legacy approximation.
- EDS is independently selected. Disabling it does not disable electron scattering.
- The default new assembly has no energy filter. Explicit saved filter choices
  remain valid. Full physical requests run an installed filter last; an explicit
  axial section ends no later than its entrance. The filter's bent coordinates
  are never extrapolated from an axial cursor.
- **Cached signals → Current pixel** shows weighted simulated electron counts
  and electrons/second from physical detector interceptions. These are distinct
  from measured counts over an exposure. Missing/not-reached/disabled readouts
  display a dash; an actually calculated zero remains zero. Filter detector rows
  use the executed filter's absorption products. Readout selection never removes
  physical detector absorption.
- Scanning remains off by default. When enabled, the existing classical STEM
  raster generates 2D HAADF/ADF/BF signals for inserted enabled readouts. A section
  ending before an inserted scan detector cannot publish a completed scan frame.
  Current classical scan contrast transports the shared specimen-exit distribution
  across the raster; it is not a new per-pixel atomic Monte Carlo or coherent model.
  Old products are cleared or explicitly detached when the requested scope changes.
  If an active scan kick is actually consumed, small auxiliary reference matrices
  may extend to its calibration plane; unconsumed AC/descan stages are not solved.
  These command calibrations do not propagate the particle population beyond the section.
- The physical state preserves IDs, source probabilities, energy, survival and
  float64 phase space/TOF. Aggregate finite-loss channels still lack event-depth
  timing, so their clocks remain unknown, not fabricated coherent phase.
- Vacuum scattering is opt-in. Without saved random/collision state, the column
  is conservatively recalculated when vacuum participation is on.

## Cache and persistence

Upstream gun and column checkpoints bind the consumed source, actual optics,
geometry, model implementation and numerical inputs. Material caches additionally
bind the actual incident phase space and scattering inputs. Outgoing material
branches retain their own executed plans and float64 checkpoints, so a later
projector edit or extension can reuse the unaffected post-specimen prefix.
Detector and aperture stops are reapplied without restoring absorbed electrons.

Files use checksummed JSON and numeric NPY arrays in the existing atomic package
writer, with a fixed data-class allowlist and no pickle/file-selected imports.
They remain local generated calculation data and must not be committed. Loaded
seeds are governed by the existing tuning cache memory budget. Existing coherent
code, historical profiles and original instrument evidence remain readable.

## Performance evidence

See [the measured performance report](particle-performance-2026-09-19.md).
The 5,000-particle no-filter benchmark used the actual default 0.5 mm column step
and was stopped at a 180-second diagnostic budget before gun transport returned.
It did not establish a complete-runtime measurement. The last 170.316-second
receipt recorded 35,336 accepted step calls and 112.686 seconds within their
step/retry work. No sample, EDS or filter stage had completed.

The implemented filter reference cache reduced a separate 49-particle filter
call from 13.690 to 6.135 seconds while preserving measured trajectories, clocks
and lineage. It cannot accelerate an uninstalled filter or an uncached gun.
Next proposed gun work is lossless history storage, compiled shared history
processing and reusable workspaces; grouped adaptive stepping needs separate
convergence checks before adoption. No looser error thresholds are introduced.

## Validation

The focused integration suite passed **265 tests** in 187.11 seconds
(`tmp/particle-sections-delivery-20260919.xml`). The strengthened persistence
suite passed **11 tests** in 16.23 seconds, including actual restoration and
extension from the saved material-branch endpoints
(`tmp/particle-section-persistence-delivery-20260919.xml`). These groups overlap;
do not sum them. Compilation and whitespace checks passed. An offscreen section
control render using an explicitly loaded Arial font was inspected; this does
not certify native desktop interaction. Tests distinguish real small-bundle transport,
analytical short-column fixtures, and GUI publication/lifecycle doubles. The
5,000-particle diagnostic is incomplete. No full instrument qualification,
production-sized scan timing, coherent acceptance, or guaranteed live frame rate
is claimed. A final isolated 49-particle no-filter complete pipeline took 6.625 s;
see the separate versioned performance receipt. The user's running application
has not been restarted.
