# Classical parameter inventory: first qualification batch

This inventory describes the current simulator, not a commercial microscope.
Static structure remains owned by TOML; operating values remain independent.
The table is a traced implementation inventory, not a claim of complete physical
qualification. Source, numerical and field-model choices must accompany any
parameter-response result. All lengths in transport are converted explicitly
from UI/configuration units; particle momentum and position use SI.

## Tip, extraction and acceleration

| Controls | Meaning and consumer | Expected response and held conditions |
| --- | --- | --- |
| `emission_current_na` (nA) | Prescribed total emission; `emitter.emitted_current_a`, gun current accounting and downstream weighted flux | With fixed emission shape, sampling and external fields, scale current without changing trajectories. Self-consistent space charge is absent. |
| `virtual_source_fwhm_nm` (nm) | Historical name for the spatial distribution **at the tip**; `ColdFieldEmitter.emit` | Changes launch width, not the location of a virtual/downstream source. No global monotonic downstream spot-size claim. |
| `angular_rms_mrad`, `angular_cutoff_mrad` (mrad) | Local launch distribution and truncation; `ColdFieldEmitter.emit` | Hold geometry and cutoff/scale respectively fixed. The input Gaussian scale is not necessarily the RMS after truncation. |
| `curvature_nm_inv` (nm⁻¹) | Continuous analytic-tip geometry; `tip_curvature.curve_bundle` | Current model changes sag and local normals continuously; zero recovers planar launch. Projected spatial distribution and prescribed current stay fixed. Historical geometry discriminators are preserved. |
| `emission_energy_ev`, `energy_spread_fwhm_ev`, bounds and shape coefficients (eV) | Positive launch-energy distribution; `chromatic.cold_feg_energy_offsets` | Requested width is RMS-equivalent FWHM. Tail/Gaussian coefficients shape a subsequently normalized distribution; they are not independent predictions of interactions. |
| Surface-model apex radius, cone half-angle and emitting-cap angle (nm, deg) | Distinct geometry/emission quantities; `tip_assembly`, `tip_surface`, `grounded_tip_field` | Radius/cone affect conducting geometry; cap angle selects emitting area. At fixed prescribed flux density, current follows surface area. These are not independent downstream source settings. |
| Surface flux density (electrons/(nm² s)) or total current (nA) | Mutually exclusive prescriptions in `TipSurfaceModel` | Changing emitting area changes total current only in the fixed-density mode. No tunnelling-current inference. |
| `feg_extractor.voltage_kv` (kV) | Tip-relative extractor potential; `FegElectrostaticField` or connected electrode solution | Changes local acceleration and focusing. At fixed final potential, extraction voltage is not added again to final energy. No universal downstream beam-radius trend. |
| Gun-lens `voltage_kv`, `voltage_reference` | Electrostatic lens field; `ElectrostaticGunLens`, `grounded_tip_field.field_request` | Solved model supports tip/extractor/ground gauges. Analytic model preserves its historical extractor-reference convention. A voltage difference is not a focusing multiplier. |
| `potential_scale`, `soft_edge_mm` | Historical analytic gun-potential shape and scale | Not a measured voltage conversion. The connected surface-field model does not consume these controls and hides them. |
| `high_tension_kv` (kV) | Accelerator field and dependent transport | Preserve per-electron energy from the actual potential difference. The historical analytic field sets nominal final energy by subtracting mean launch energy from its voltage ramp; the solved surface model uses physical HT plus launch kinetic energy. Do not silently convert saved conventions. |
| `tip_radius_nm`, `tip_cone_half_angle_deg`, `emitter_material`, `work_function_ev`, `vacuum_pa` on the historical emitter | Metadata in the analytic launch path | These scalar values do not implement a new material-emission or residual-gas law. Actual continuous curvature and installed surface geometry have separate explicit owners. Vacuum participation comes from the opt-in vacuum map. |

## Magnetic lens and apertures

| Controls | Meaning and consumer | Expected response and held conditions |
| --- | --- | --- |
| Lens `percent` (%) | Excitation scale through the active runtime field provider; `physics.core.fields`, Lorentz/paraxial transport | In the isolated ideal fixed-shape limit, field amplitude is linear and focusing power is quadratic in amplitude. This is not a measured ampere scale or a saturation law. |
| Lens `polarity` (±1) | Signed magnetic field | Isolated axisymmetric lens: rotation changes sign, radial focusing is retained. Other overlapping fields can break the isolated comparison. |
| `b0_t` (T), `a_mm` (mm), Gaussian terms, pole geometry | Structural field support/calibration and provider-specific physical geometry | TOML/structural editor owns them. Ideal, geometric-material and imported-map providers have different dependencies. Geometry changes cannot leave a stale field map active. |
| `cs_mm`, `cc_mm` (mm) | Declared or provisional aberration coefficients | Distinguish geometric aberrations and chromatic defocus from wave-image resolution. Coherent tip-to-column development remains paused. |
| Aperture radius and offsets (mm), insertion state | `GunAperture.transmission_mask` and column `aperture_clipping.clip_segment` | Nested openings with the same centre and fixed incident ensemble have nondecreasing transmitted current. Offset sweeps are not globally monotonic. Stopped particles cannot return downstream. |
| Gun `trace_step_mm`, `drift_step_mm`, column `step_mm` | Numerical integration spacing | Independently refine these axes; none changes the physical source. |
| `history_step_mm`, `ray_count` | Saved observation spacing and source sampling budget | More samples do not multiply emitted current. History sampling is not an independent transport/source model. A finite-particle identity check is not ensemble convergence. |

## Stigmators, correctors and scan/descan (2026-09-19)

| Controls | Meaning and consumer | Held conditions / limits |
| --- | --- | --- |
| Stigmator X/Y (%) and `field_model` | Trace-free normal/skew tensor in fixed column coordinates; legacy difference remains rank one | Twofold axes depend on the vector sum. Pure reversal swaps axes, but changing just one mixed component need not rotate by 90 degrees. |
| `max_strength_m2`, `strength_m2` (m^-2) | Peak effective quadrupole coefficient | Positive normal coefficient focuses X and defocuses Y; it is not round-lens strength or coil current. |
| `strength_m3` (m^-3), `orientation_rad` (rad) | Signed hexapole equation coefficient and triple-angle normal/skew components | Fixed reference energy and relay optics. Local force is quadratic; 120-degree periodicity, 60-degree sign reversal. No universal correction-improvement trend. Inactive in ideal mode. |
| Stigmator `length_mm`, multipole `effective_length_mm` (mm) | Gaussian effective field FWHM, through existing runtime validators | Not material body length. Fixed-peak width changes also change the integrated coefficient. |
| Scan pitch (nm), X count, Y count | Shared requested raster at the selected specimen reference | FOV=N*pitch, centre span=(N-1)*pitch. Held optics changes can make actual pitch differ. |
| Frame/wobble period (s), wobble phase (degrees) | Time law of common paired-coil command | Compare paths at equal normalized time phase. No electronic lag, hysteresis or raster flyback dead time. |
| Pair gain, lower gain, pivot offsets | Dimensionless upper drive gain, representative lower readback, diagonal ratio perturbations | Full 2x2 coupling determines physical kicks. A lower scalar edit changes pair gain; it is not an independent lower-coil command. |
| Automatic/held mode, reference, physical target | First-order calibration or replay of saved drive ratios | Explicit recalibration replaces held ratios. Unsupported filter-reaching observers cannot bypass filter physics. |
| Historical scan amplitude scalars | Initial diagonal command compatibility values | Active calibrated scans use full command matrices, not direct edits to these scalars. |
| `effective_thickness_mm` on scan pairs | Stored validation/layout annotation | Does not move the separately defined kick planes or enable a distributed coil field. |

See the [second-batch evidence](qualitative-second-batch-2026-09-19.md).

## Boundaries

- Existing `parameter_registry` and `parameter_semantics` supply interface
  explanations; no parallel editable parameter store is introduced.
- Remaining general-multipole controls are still a later batch. Unknown
  controls remain unclassified and cache reuse stays conservative.
- The independent field-integral limit is an element test. Executing the tip,
  extraction, acceleration and first installed lens is a segment test. Neither
  establishes sample illumination, full-image, GPU or energy-filter acceptance.
- The existing source/assembly invariants in `AGENTS.md` remain in force,
  including paused coherent development and default-off vacuum participation.

## Detector response and current readout (2026-09-19)

| Controls | Meaning and consumer | Held conditions / limits |
| --- | --- | --- |
| Detector `inserted` | Physical collection/interception in `recording_clipping` and `record_plane` | Changes downstream survival; selecting a diagnostic does not insert a detector |
| `readout_enabled` | Electronic output selection in physical routing and `stem_signal._readout_view` | Disabling readout retains absorption/interception; it does not release particles downstream |
| Active width, inner diameter, centre offsets (mm) | Detector `hit_mask` at the installed physical plane | Fixed incident ensemble: nested larger active areas cannot reduce interception; a larger central hole cannot increase it. Offsets have no universal monotonic trend |
| Camera `pixels` | Native pixel count per side; existing camera projection has separate calculation sampling | Width/count sets native pitch; the ray PSF diagnostic uses its own display grid. Pixel count does not multiply dose |
| `point_spread_*` | Physical-mm Gaussian detector PSF and its provenance | Separate from optical blur, electron interception and integrated STEM collection. Virtual diagnostic arrays retain supplied weights; only the display copy is peak-normalized |
| Frame period, scan count | Existing `stem_signal` current/dwell conversion | At held collection fraction, electrons = current x frame period / (pixel count x elementary charge); a dynamic scan change still requires its own transport |
| `stem_poisson_enabled/seed` | Optional repeatable integrated-count noise | Changes realizations, not expected electrons, current or survival |
| `stem_fourdstem_response_mode` and efficiency | Ideal/adjustable supported pixel readout in `PixelatedDetectorResponse` | Efficiency scales mean signal, not physical absorption or DQE; no coherent source is admitted by enabling this response |
| Pixel spread (pixels), dark (electrons/pixel/exposure), read noise (electrons RMS) | Existing adjustable mean-response/noise model | Independent Poisson sampling follows expected-signal blur; event charge-sharing covariance is not modelled. Dark input is per exposure; clipped read noise biases near-zero means |
| Saturation (electrons), gain (counts/electron), offset (counts) | Existing post-noise upper clipping, scaling and offset | Saturation breaks linear dose scaling. Counts and clipped signal are not transported electron populations |

See the [third-batch evidence](qualitative-third-batch-2026-09-19.md). Current
checks qualify bounded mean response, explicit loss accounting and local stop
stability, not full camera images or arbitrary PSF/sampling regimes.

General reference checks: [magnetic lens principle](https://www.jeol.com/words/semterms/20121024.034959.php)
and [electrostatic energy conservation](https://www.phys.ufl.edu/~avery/course/2049/textbook/HRW_v10_ch24.pdf).
They support the limiting principles; they do not supply this simulator's
dimensions, currents, parameter signs or calibration tables.
