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

## Keeping the direct illumination disk out of DF

An annular detector is not dark field merely because its component key is
`df`. Its active area must lie outside the direct illumination disk for the
actual camera-length setting. In a centred, isotropic diffraction plane,
`direct_disk_radius ≈ effective_camera_length * probe_semiangle` and
`inner_collection_angle ≈ detector_inner_radius / effective_camera_length`.
The runtime model uses the full signed map instead:

```text
r_detector = J_img * r_sample + J_diff * theta_sample + offset
```

The projector excitations and detector Z determine `J_diff`; anisotropy,
affine offsets, specimen/scan position and time-dependent descan also matter.
Consequently, changing camera length can bring the direct disk into a DF
annulus whose physical dimensions have not changed.

**Images > Exclude direct beam…** proposes DF inner/outer diameters using the
current calculated state and its full raster. Review the proposed dimensions
and save them to the currently selected recording module's TOML. The action
keeps the detector Z and lens settings, includes a clearance margin, and
checks the available chamber bore. An unsuitable transfer or insufficient
mechanical clearance gives no applicable proposal. A bank result, stale frame,
or a camera-length/geometry change while reviewing cannot apply an old proposal.
The previous image remains a previous result until High accuracy is run again.

The fit only excludes the modelled direct disk. Upstream HAADF is allowed to
overlap the DF angular band and intercept those electrons first. The physical
sequential-stop masks determine the remaining DF signal; no signal is added
back to compensate for this shadowing. DF's standalone acceptance band and
the band that survives all upstream stops need not be identical.

For the archived Si [110] camera-length setting used in the Poisson display
example, DF's original 2/14 mm inner/outer diameters accepted approximately
1–7 mrad, within the approximately 24.8 mrad coherent illumination disk.
At its unchanged Z, 60/100 mm corresponds to approximately 30–50 mrad. These
dimensions are specific to that archived transfer, not universal DF defaults.
The separate `outputs/df_direct_beam_clearance` acquisition records its actual
geometry and preserved projector setting.

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

## Geometry preview display (2026-09-08)

New microscope states default to **Calculate STEM detector images (High accuracy)**
enabled. High accuracy honours this switch; explicitly saved disabled settings
remain disabled. With the STEM wave checkbox off, `geometric_detector_interception` displays
scan/descan detector interception. A finite-particle fallback may include
CIF-derived material composition/density, but it does not propagate the CIF
lattice independently at each scan pixel. The Images page distinguishes this
from multislice/thin-phase specimen images and provides **Enable CIF wave
imaging** followed by **Run High accuracy**. Enabling the setting alone does
not recompute or relabel the retained image. A paused display remains on its
previous complete frame until resumed.

Positive constant detector fractions now use mid-grey and a precise
**Constant signal** label. Varying geometric/particle previews use the fixed
0–1 emitted-current-fraction range. This prevents a one-ray change, such as
`3531/15000` to `3532/15000`, being stretched into an apparent black/white
specimen boundary. Nonconstant specimen wave images retain automatic contrast.
No raw image values or detector fractions are changed by display scaling.

Image coordinates remain in micrometres internally, with axis labels converted
to metres before a single automatic SI-prefix selection. At 32×32 pixels,
0.02 nm/pixel spans 0.64 nm and 0.05 nm/pixel spans 1.6 nm. Frame size, model,
scale and captured source context follow the displayed complete frame,
including paused and saved results, rather than later edits to live settings.

## Expected electrons and Poisson image display (2026-09-08)

**Scanning Image > Images > Display** selects three stored image quantities:

- **Ideal intensity**: detector fractions relative to the effective source
  electron flux, without detector counting noise.
- **Expected electrons**: the mean detected electrons per scan pixel,
  `fraction * effective_source_current_A * dwell_time_s / e`.
- **Poisson counts**: a seeded integer draw from that mean, representing
  finite-electron shot noise. It does not include detector electronics noise,
  drift, or beam damage.

In **Scanning Parameters > STEM image statistics**, checking **Generate seeded
Poisson counts** also selects the Poisson display. Run **High accuracy** to
produce a result with the enabled readout. A retained frame without the chosen
arrays shows **Unavailable**, rather than substituting an ideal image. With
**Pause refresh** checked, resume refresh to see the new result. Historical
bank results use their own stored arrays and metadata.

The notice above the images shows the displayed quantity, electrons/pixel,
captured dwell time, and the recorded seed for Poisson data. The tooltip also
includes captured effective source current when available. Dwell time is the
scan frame period divided by its pixel count. Numerical ray count is not the
physical electron dose. The signal fractions already include transport loss;
no second column-transmission factor is applied to the count conversion.

Expected and Poisson views share a fixed, zero-based count scale for each
detector, covering both full-frame arrays. Ideal intensity retains its existing
contrast rules. Switching quantity, changing tabs, or replaying scan lines
does not sample again: one completed frame contains one fixed realization.
Changing the seed and recalculating generates a reproducible new realization.
Angular-coverage and illumination-sampling masks still apply in count views;
adding counting noise does not supply missing simulated detector angles.

Changing only the Poisson switch/seed reuses cached STEM fractions and updates
the readout. Current scaling can also use this existing readout path. Frame
period remains a transport dependency because time-dependent scan/descan
settings can change trajectories; it is not unconditionally a dose-only edit.

`tests/test_stem_count_display_gui.py`, `tests/test_stem_poisson_statistics.py`,
and `tests/test_stem_dose_cache.py` cover stored-data display, pause/bank/line
playback, statistics, normalization and cache reuse. A readout-only Si [110]
comparison, based on archived corrected fractions, is documented in
`outputs/stem_poisson_display/README.md`; it is not a new specimen acquisition
or a convergence validation of those archived probabilities.

## References

- [abTEM: Scan and detect](https://abtem.readthedocs.io/en/main/user_guide/walkthrough/scan_and_detect.html):
  detector integration and the requirement that the simulated angular range
  cover the requested detector range.
- [Physical-optics Bloch-wave simulations (IUCr, 2025)](https://onlinelibrary.wiley.com/doi/full/10.1107/S2053273325000142):
  an explicit Si example with bright atomic contrast in BF and inverted
  ABF/MAADF contrast under its stated conditions.
