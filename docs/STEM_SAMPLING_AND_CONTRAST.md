# STEM sampling and contrast

## Reading a detector image

The detector name does not prescribe the sign of atomic contrast. Small-angle
BF and low-angle DF include coherent/dynamical contrast; thickness, defocus,
orientation and collection angle matter. Incoherent high-angle ADF normally
shows bright atomic columns. No image is inverted to enforce that expectation.
Each image uses an independent linear grayscale (larger signal is brighter).

The Images page now reports angular coverage for each detector:

- **Band covered:** the conservative acceptance bound fits inside the grid.
  This is necessary, but it is not a convergence certificate.
- **Partial band:** the displayed intensity only includes sampled angles.
  It is diagnostic, not a quantitative full-band signal.
- **Not simulated:** the detector lies outside the wave grid. The image is
  cleared instead of displaying a misleading physical zero. Raw cached arrays
  are retained. An enabled Rutherford tail is labelled as approximate tail only.
- **Unknown:** the transfer has no finite invertible angular bound. Inspect
  the detector's actual conjugate plane; enlarging the grid is not a guaranteed
  remedy.

If the illumination disk itself exceeds the grid, all detector images are
suppressed. Older wave frames without coverage metadata are marked unchecked.
Hover over a coverage label for the probe semi-angle and acceptance bounds.
The bounds include raster position, affine offsets, anisotropy and square
detector corners. They are conservative and do not assume that upstream
aperture clipping establishes sufficient wave sampling.

## Correcting insufficient coverage

Use **Match detector sampling** on the Images page. It proposes an increased
Sample wave **Grid**, while retaining the wave FOV, raster pitch/pixels, lens
strengths and detector geometry. It displays the grid-area growth and an
approximate potential-storage cost. This is not an estimate of total peak
memory or runtime. Confirmation changes the parameter only; run High accuracy
when ready. The usual memory preflight still applies.

Proposals exceeding 8192 pixels or the existing 4 GiB potential-storage limit
are disabled. A proposal cannot be applied to a different edited state or a
paused frame from an older state. No hidden automatic enlargement is performed.

For example, at 300 kV a roughly 256-pixel, 4 nm grid with a two-thirds
bandwidth cutoff reaches only about 42 mrad. A 60 mrad inner-angle HAADF
detector cannot be evaluated by that grid. Increasing the number of traced
column rays or changing the display grayscale does not recover missing angles.

## Real-space registration correction

Prepared potentials and waves use axes `(arange(N) - N//2) * spacing`.
CPU and CUDA STEM inverse FFTs now place the unshifted probe at the centre of
those axes. The STEM incident-wave helper shares this convention.

Periodic reference cells move their box-origin lattice site to the centred
grid. Finite CIF boxes already place specimen-local zero at the box centre;
they must not receive the same extra half-box shift. Finite boxes use even
lateral grid dimensions so their centre is exactly on a grid point. The
finite-envelope mask is consequently applied in the same coordinates as the
potential and beam.

Affected wave/STEM/4D-STEM cache identities are versioned. Pre-correction wave
products must be recalculated, but incident-ray, elastic-event, EDS and local
sample-interaction cache identities are not invalidated by this correction.

## Validation and limits

`tests/test_stem_contrast_reference.py` checks independently specified ASE atom
positions against the potential grid, including finite and periodic boxes. A
weak isolated phase-object limiting case produces bright ADF and dark BF at
the registered atom position. Its intensities are compared against independent
abTEM probe construction, propagation and annular detection (`rtol=0.005`,
`atol=0.00002` in normalized intensity). This is not a universal contrast-sign
test for thick crystals. Existing tests compare actual CUDA and CPU STEM
results, including frozen-phonon ensembles.

Coverage, anisotropy, offsets, cache isolation, pause/refresh, raw-data
preservation and parameter-only sampling changes have dedicated tests in
`test_stem_sampling.py` and `test_stem_sampling_gui.py`.

The final related run on 2026-09-06 passed 153 tests (96.63 s, exit 0):

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' tests/test_stem_sampling.py tests/test_stem_sampling_gui.py tests/test_stem_contrast_reference.py tests/test_stem_cuda_pipeline.py tests/test_atomistic_specimen.py tests/test_wave_imaging.py tests/test_multislice.py tests/test_scan_system.py tests/test_fourdstem.py tests/test_fourdstem_cache_products.py tests/test_fourdstem_user_wiring.py tests/test_calculation_cache_reuse.py tests/test_record_plane.py tests/test_stem_detector_control.py -q --tb=short
```

The screenshot's exact acquisition has not been rerun. Quantitative thick-Si
images still require the saved specimen/beam state and convergence checks in
grid, slice thickness and thermal configurations. A covered band is not a
validation of the projector calibration or the underlying material model.

## References

- [abTEM: Scan and detect](https://abtem.readthedocs.io/en/main/user_guide/walkthrough/scan_and_detect.html):
  detector integration and the requirement that the simulated angular range
  cover the requested detector range.
- [Physical-optics Bloch-wave simulations (IUCr, 2025)](https://onlinelibrary.wiley.com/doi/full/10.1107/S2053273325000142):
  an explicit Si example with bright atomic contrast in BF and inverted
  ABF/MAADF contrast under its stated conditions.
