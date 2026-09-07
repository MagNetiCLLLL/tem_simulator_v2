# Finite CIF region and wave sampling

## Three independent sizes

- **Specimen envelope**: physical diameter/width and thickness in Sample. It
  controls where material exists; it is not the scan field or FFT window.
- **Atom region**: the overlap of that envelope with the padded illumination /
  scan window. A disk clips actual atom positions, not just displayed pixels.
- **Wave window**: the full illuminated/scanned area including vacuum and probe
  padding. It is not clipped to the specimen. Electrons missing the specimen
  remain part of wave propagation and detector readout.

For STEM, the window includes the traced beam centre, baseline scan correction,
scan positions and the existing diffraction/defocus support estimate. This is
still a finite-window approximation: check padding and grid convergence for a
quantitative image. Potential, beam and detector coordinates remain separate
from display limits. A larger window does not change the physical sample.

## What changes when focus changes

The configured **Field of view / Grid** defines the reference spatial spacing.
When the required STEM window grows, the execution grid grows to retain that
spacing; the saved input values are not overwritten. No automatic coarsening,
interpolation of the specimen image, or removal of vacuum electrons is used.
The result records `wave_sampling_plan`, `wave_window_bounds_nm`, and
`atom_generation_bounds_nm`. The last field is a bounding box; a disk further
clips atoms to its circular envelope. Detector sampling proposals map back to
the reference Grid so that subsequent runs do not enlarge it twice.

Window planning checks memory before atom construction. Requests above the
existing 4 GiB potential-storage limit report the window, grid, spacing and
estimated storage. An additional 8 GiB conservative wave-working-set limit
reserves CPU probe batches, FFT/mask buffers, potential copies and retained
configuration waves, even for vacuum. Neither estimate includes the full
column or all other application memory; the existing solver guards still apply.
Very large defocus can remain
infeasible even for a small specimen, because the surrounding vacuum still
requires a finely sampled wave. The existing detector angular-coverage warnings
remain necessary; preserving spacing alone does not certify an image.

## CIF construction and safety

Finite CIF generation no longer repeats a large covering cube before cropping.
Inverse-rotated bounds determine lattice columns, and each column's allowed
interval is solved against the physical/ROI faces. Batches contain at most
65,536 candidate sites. The 5,000,000-atom limit applies to retained atoms;
separate 20,000,000 column and candidate-work limits bound enumeration cost.
Crystal phase, the user's imported unit cell and right-handed orientation are
preserved. Sites are sorted in the previous ASE lattice/basis order to preserve
the deterministic random-displacement ordering for unchanged rectangular ROIs.
Choosing a disk changes the retained sites, as intended.

An explicit coarse FOV/Grid combination can still be invalid. Before building
abTEM slices, a project-side check rejects a degenerate radial integration grid
with a clear sampling error instead of allowing backend division by zero. No
installed dependency, atom limit or physical scattering coefficient is patched.
The TOML reference retains its commensurate periodic crystal. Its atomic and
analytic potentials now retain lattice phase relative to the physical sample
centre when the calculation window moves; the crystal must not move with a
steered beam.

## Cached results

The new wave-specimen algorithm version invalidates automatic reuse of old
TEM/STEM/4D-STEM and EFTEM products, including complete-result shortcuts.
Unchanged incident-ray, particle-interaction, EDS and ordinary EELS cache keys
are retained. Historical data are not deleted. A failed request does not turn a
previous successful image into a result for the changed parameters.

## Focused validation

`test_cif_roi_enumeration.py` compares small cases against the previous independent
repeat-and-crop implementation, including arbitrary rotation, offsets, exact
faces, disk clipping, ordering, storage/work guards and the coarse-grid failure.
`test_wave_domain_planning.py` covers window/sampling separation and small wave
cases. `test_wave_specimen_cache_version.py` checks selective cache invalidation,
including on-disk reuse of unaffected incident rays.

No full 100 x 100 acquisition, user-setting replacement, or interactive GUI
restart is part of this validation.

The final combined run passed 149 targeted tests in 83.12 s. A small real
finite-CIF fixture (89 Si atoms, 3 x 3 scan) produced different focused and
30 nm defocused BF/DF signals. Bandwidth-filter losses were reported, not
forced to zero (largest measured wave-norm change in that fixture: 0.120209%).
Separate vacuum cases preserved total intensity to 1e-10. These are numerical
regressions, not experimental calibration or a full-resolution Si image claim.
