# Camera and fluorescent-screen reception

**Illuminating Image > Camera / screen** displays the selected installed receiver
in a captured calculation. Run **Run high-accuracy once**, then **Calculate
imaging** on this page. The latter reuses valid upstream particle calculations
and calculates sample interactions/downstream transport when required. It does
not request STEM detector acquisition, EDS, or a coherent-wave calculation.

Selecting a receiver changes the display only. It does not insert/retract the
screen or camera, or bypass an upstream detector. Retracted receivers, disabled
readout, incomplete trajectories, and missing calculations have explicit status
messages. A completed result with zero intercepted probability is a valid black
image, distinct from an uncalculated image. Editing the instrument marks the
retained image as a previous calculation; its receiver list and coordinates still
come from its own snapshot.

## Views

- **Current position** bins the actual first physical hits on the selected
  receiver, including available geometric specimen-exit scattering branches.
  It applies the configured detector point spread. Image/diffraction projector
  settings affect downstream propagation; the display does not invent a pattern
  from the mode label. Broad illumination follows the simulated source profile
  and aperture transmission rather than being forced uniform.
- **Single scan preview** averages translated copies of that retained distribution
  over one captured, calibrated AC/descan raster. It is calculated once and held.
- **Scan position preview** shows the moving beam at the selected raster pixel.
  Use the slider or **Next position** for manual inspection, or **Play scan**
  for timed playback. The cyan centroid trajectory is an optional geometric
  overlay; the yellow point marks the chosen position.
- **Frame build-up preview** accumulates the raster through the current position.
  Signal uses the original full-frame dwell weights, so a partial frame is not
  renormalized to look like a complete exposure. Contrast stays referenced to
  the complete frame during playback.

## Continuous camera observation

**Play scan / Pause scan**, **Reset scan**, **Loop**, and **Speed** control only
the camera/screen preview. With Loop off, playback stops at the last pixel and
retains the completed frame. With Loop on, the next preview frame starts after
the completed frame is displayed. Switching to Current position or Single scan
preview, dragging the position slider, or pressing Next position pauses playback.

**Follow STEM scan** follows the clock of the matching Scanning Image result.
It never connects one calculation's clock to a different instrument snapshot.
When the STEM frame completes, Loop can continue camera observation from the
retained data. HAADF/DF/BF acquisition remains single-frame and is not restarted.
Turning Follow STEM scan off allows independent local preview playback.

To compare compensation, enable **AC Scan Coils > Enable raster drive** and
disable **Descan control (Image/Diffraction) > Enable raster drive** in Scanning
Parameters. Calculate imaging, select an inserted receiver, and play Scan
position preview. Re-enable Descan or adjust its coupling, recalculate, and
compare the trajectory. Disabling the raster removes its time-varying command;
it does not disable the shared Image/Diffraction deflector or static offsets.

The preview does not change live scan settings. Input edits pause it and mark
the retained data stale; Calculate imaging is required before playing updated
settings. Hidden receiver pages stop image work and local timers; local playback
resumes when shown, while a followed scan retains only its latest position.

All scan views are explicitly **geometric previews**, not acquired physical
exposures. They use first-order displacement relative to the actual captured
simulation time, including descan and cross-axis rotation. They do not recalculate
scan-dependent specimen scattering or upstream clipping, and particles missed
by the static calculation cannot reappear. Moved pixels use a conservative
sensor-centre acceptance approximation; stationary accepted hits are preserved.
No detector current or physical electron counts are reported for a preview.
Curved energy-filter paths without a supported scan response remain unavailable.

## Units and data

Receiver coordinates are laboratory X/Y in millimetres. Pixels cover the full
active-area bounding square. **Fit receiver** and **Fit beam** change the view,
not the underlying sampling. Default images are 192 × 192 pixels. Contrast and
log display affect presentation only. Sparse particle sampling remains visible;
no artificial smoothing is added beyond the configured point-spread model.

Static intensity is probability per emitted electron per pixel. **Expected
incident electrons** uses the captured source current and the chosen static
exposure; it is not the count of numerical rays, a noisy camera ADU signal, or
an assumed quantum-efficiency model. NPZ export retains the raw ideal/response
probabilities, millimetre axes, exposure, capture provenance, and available count
expectation. Preview exports include trajectory/time arrays and their explicit
preview status, with no physical count array.

The implementation prepares the physical seed and scan geometry once per result
and receiver, then uses bounded CPU image shifts or accumulation. It renders
only visible pages, limits refresh to 10 Hz, and keeps a few display products
rather than all pixel images. Local frames shorter than 0.4 seconds at the
selected speed are slowed for visibility; this does not change physical raster
timing. Fast playback may skip displayed positions; Next position inspects each
retained pixel (the existing geometry may be decimated above 128 pixels/axis).
Following STEM uses its captured clock and coalesces display updates. No CUDA or
repeated particle, sample, or STEM calculation is needed for playback.

## Coherent imaging

Classical particles can show the direct beam and calculated scattering, but
cannot reproduce coherent Bragg interference by themselves. **Stored wave /
reference** retains the previous coherent-image display and separate specimen
exit-wave diffraction reference. Its existing source-admission gate is unchanged.
The specimen Fourier reference is not presented as the selected physical
camera's received image. A complete scan-dependent coherent sensor exposure
requires additional wave transport, not relabelling this geometric preview.
