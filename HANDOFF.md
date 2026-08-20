# TEM Simulator v2 — Project Handoff

Last updated: 2026-08-20

## Purpose

This file is the persistent handoff for continuing development. Keep it focused
on current behaviour, confirmed design decisions, provisional assumptions and
the next concrete work. `README.md` remains the user/developer overview and
`CHANGELOG.md` remains the release history.

## Start and verify

- Workspace: `F:\tem_simulator_v2`
- Python: `.venv\Scripts\python.exe`
- Application entry point: `main.py`
- Run: `.venv\Scripts\python.exe main.py`
- Tests: `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest -q`
- Last full result: **311 passed** on 2026-08-20 in 427.3 seconds, with CUDA
  available and no skipped tests. Serial `compileall`, `pip check`, `main.py`
  import and the offscreen `MainWindow` smoke check passed.

## Layered aberration checkpoint (2026-08-20)

- `optics/aberrations.py` is the shared intrinsic/system aberration authority.
  Round-lens missing Cs/Cc values resolve to labelled provisional focal-scale
  estimates; explicit values remain authoritative and no unavailable value is
  presented as a physical zero.
- Full effective coefficients live only at probe/specimen and Objective/image:
  C1, A1, B2, A2, C3, S3, A3, C5 and Cc. Non-axisymmetric terms carry an
  azimuth. State schema 65 persists probe/image overrides.
- The Aberrations page compares identical relay optics with nonlinear
  hexapoles disabled/enabled. TEM CTF and STEM probe phase use the same active
  effective set through fifth order. Cc remains a first-order energy-dependent
  defocus term rather than an incorrect single-energy coherent phase.
- This is a principle-correct, non-OEM model. No real instrument calibration
  or energy-filter aberration curve is claimed.

## Transverse X-Y direction-colour checkpoint (2026-08-20)

- The former four discrete initial-quadrant colours are replaced by a
  continuous cyclic HSV direction map, with +X = 0 degrees and angle increasing
  counter-clockwise toward +Y = 90 degrees.
- An on-page DPC-style colour wheel uses the exact ray colour mapping and labels
  all four signed cardinal axes. Colour represents initial polar direction
  only, not radius, energy, intensity or survival state.
- Initial angles are measured about the ray bundle's own start-plane centroid,
  so off-axis translation does not corrupt the rotation diagnostic. A ray
  exactly at that centroid has undefined direction and is shown in neutral
  grey.
- Component selection displays its centre Z, except recording devices, which
  display their upstream top-surface signal Z. Go to Z, axial-plot selection
  and the movable cyan cursor display an arbitrary Z; cursor movement refreshes
  the transverse slice live, and whichever source was selected last remains
  authoritative across recalculation.

## Detector point-spread checkpoint (2026-08-20)

- Camera, Fluorescent Screen and BF/DF/HAADF now carry TOML-authoritative
  `point_spread_*` fields: model, sigma X/Y in detector-plane mm, principal-axis
  rotation, status and source. Current values are deliberately labelled
  `provisional_model_parameter`, not real-instrument calibration.
- `detector/point_spread.py` performs only forward convolution with a unit-sum,
  rotatable anisotropic Gaussian. `detector_response_image()` accepts rays with
  the component's square/disk/annulus hit mask, applies the PSF, reapplies the
  finite sensitive-area mask and exposes accepted, response and retained
  weights. Zero padding permits physically meaningful edge loss.
- Selecting one of those recording devices overlays its response beneath the
  original direction-coloured points in Transverse X-Y. Selecting an arbitrary
  Z or a non-recording component clears the response, so an ordinary geometric
  slice is never presented as a detector image.
- This stage does not alter ray trajectories or the specimen-to-Objective CTF,
  and it does not yet connect Zebra/EELS or EFTEM output readout PSFs.

## Permanent Energy Filter checkpoint (2026-08-20)

- Instrument Setup selects only the gun and column. The Iliad Energy Filter is
  permanent hardware and is the sole selectable recording system.
- `NoEnergyFilter.toml` remains catalogued as non-selectable historical
  geometry. Applying or loading a legacy no-filter selection normalises it to
  `Energy Filter`, while the runtime optical branch can still be disabled.
- The active catalog has 15 selectable gun/column assemblies; validation still
  audits all 10 module TOMLs and 466 variant-scoped part definitions.

## Latest Direct Alignment checkpoint (2026-08-10)

- The left assembly navigator now has Optical, Mechanical and Direct Alignment
  pages. Direct Alignment exposes four TOML-defined operator controls:
  Nanoprobe convergence, Microprobe illuminated diameter, Image magnification
  and effective Diffraction camera length.
- Nanoprobe and Microprobe dynamically solve C2/C3 together. Nanoprobe targets
  the current-weighted 95% radial semi-angle relative to the 3-D chief ray and
  constrains the waist to the sample. Microprobe targets the 95%-current sample
  diameter while constraining radial wavefront curvature and a 0.5 mrad maximum
  semi-angle. Common Larmor rotation cannot change these radial metrics.
- Image dynamically solves Objective/D/I/P1/P2 as one preset against the active
  recording stop. It no longer fixes an Objective intermediate image or assigns
  equal/independent per-lens magnifications. The total sample-to-recording ABCD
  condition is `B=0`, and displayed magnification is `|A|`. Diffraction remains
  a separate distributed-field D/I/P1/P2 solve which relays the live Objective
  back focal plane and reports effective camera length.
- `src/temsim/optics/equivalent_image_lenses.py` derives a non-OEM thin-lens
  power from each isolated post-sample `integral(Bz**2 dz)` and retains its
  signed Larmor rotation as a separate event. The same exact-Z events drive the
  optimiser, production first-order trace and ordinary ray diagram; this is not
  a display-only multiplier. Simulation metrics separately expose signed
  magnification, upright/inverted state and Larmor rotation.
- Image has distinct TOML seed branches. LM covers 10x through 1,000x with the
  Objective nearly bypassed. Normal/HM covers targets above 1,000x with the
  Objective active. Seeds at 10/100/1,000/10,000/100,000/1,000,000x are merely
  deterministic starting currents; arbitrary values such as 65.7x, 333x,
  2,500x, 25,000x, 250,000x and 750,000x were also live-solved successfully.
- The old unused I/P1/P2, X-only magnification prototype is now only a
  compatibility facade over the single Direct Alignment implementation. It can
  no longer write a failed local solution into live state.
- Solves run in a dedicated Qt worker on a canonical state snapshot. A result is
  committed atomically only if validation succeeds and the live state token is
  unchanged; manual edits, assembly/mode changes, failed targets and stale
  background results leave every lens unchanged.
- The regression working points include 30 mrad Nanoprobe, 2.0 um Microprobe,
  the complete decade grid from 10x through 1,000,000x Image, and
  0.01/0.05/0.1/0.5/1/2 m Diffraction. Microprobe is validated over 0.5-2.2 um.
  Fine-grid diagnosis rejected the earlier apparent 1M distributed-field
  solution as a 0.1 mm integration artefact: it failed at 0.05/0.025 mm. No
  implausible multi-tesla projector rating was committed. The equivalent focal
  model is an engineering calibration, not a Talos OEM current table.
- On 2026-08-10 the GUI startup and compatible-assembly reload paths were fixed
  to apply the active TOML operating-mode pair instead of trusting only the
  `State` mode labels. Projector Direct Alignment now uses bounded logarithmic
  continuation plus validation-grid refinement. This removes the false
  approximately 0.10865 m camera-length plateau without changing field ratings.
  The current 5 m request reaches approximately 2.59 m with a small BFP relay
  residual but saturates P2 at its existing 100% limit; 5 m therefore remains
  an explicit field/geometry calibration task, not a claimed working point.
- Optimisation uses a 0.1 mm grid and 0.05 mm validation for the condenser and
  Image controls. The cancellation-sensitive Diffraction relay is optimised at
  0.05 mm and validated at 0.025 mm. The 0.05 m working point has a 6.19 um BFP
  relay residual at the validation step.
- Fine validation uses the production nonlinear propagation path, including
  upstream deflector/corrector kicks, exact saved aperture planes and column
  walls. Invalid ray weights are rejected; an entirely blocked sample bundle
  reports unavailable beam metrics without crashing the mechanical trace.
  Every control also rejects optimiser/validation observable spread above 1%.

## TOML authority checkpoint (2026-08-09)

- Instrument structure now has one final authority: the resolved selection of
  one gun, one column and one project/recording TOML. `main.py` and the Qt
  composition root contain no instrument geometry.
- Every active part has a stable definition ID of the form
  `<module path>::parts[<canonical key>]`; runtime optics, gun components,
  sample, Objective and Energy Filter retain that non-serialised provenance.
- The former gun geometry re-read through the process-default manifest root
  was removed. `AssemblyCatalog(root=...)` now remains authoritative through
  final runtime application, including gun exit, ordinary component geometry
  and Energy Filter/slit geometry.
- Catalog validation rejects duplicate module files, module keys and selection
  signatures; module validation rejects duplicate part keys/orders. Runtime
  layout/state key collisions and missing structural TOML fields fail instead
  of silently taking the last Python object or a class default.
- There are 10 module TOMLs, 466 variant-scoped definitions, 192 logical part
  keys, 274 intentional cross-variant repetitions and 30 collision-free
  selectable assemblies. Cross-variant repetitions are mutually exclusive,
  never an override order.
- Saved profiles omit all TOML-owned positions and structural attributes. The
  canonical Objective Stigmator key is now `objective_stigmator`; `obj_stig`
  remains input-only migration compatibility.
- Energy Filter geometry, energy-slit mechanics, default sample Z, sample
  diameter and stage envelope are TOML-derived. Python continues to own
  algorithms and runtime operating controls, not a second structural model.

## Latest compact projector and recording-surface checkpoint (2026-08-20)

- D, I, P1 and P2 retain the validated optical centres at local Z = 82.5,
  252.5, 432.5 and 635.0 mm. Their complete housing/yoke envelopes are now
  32.5-132.5, 137.5-367.5, 372.5-492.5 and 497.5-772.5 mm, leaving exactly
  5 mm between neighbouring physical assemblies.
- The whole projector stack uses a 20 mm electron-accessible vacuum ID. Pole
  mechanical bores are 21.5 mm so the declared 0.75 mm liner remains outside
  the electron path. Geometry is tagged as a user-defined non-OEM principle
  model; do not present it as a production drawing.
- Camera, Fluorescent Screen and BF/DF/HAADF collect signal on the upstream
  top surface. Their optical reference equals `local_start_z_mm`; runtime
  detector Z, Physical Layout callouts, Ray Diagram markers and Transverse X-Y
  selection all point to this signal surface, while body centres remain
  available as mechanical geometry.
- The exact axial grid now merges coordinate-rounding-sized terminal
  intervals. This prevents a near-zero RK4 interval and singular gradient
  when a restored optical centre lies an ulp away from a regular grid point.

## Historical projector-lens mechanical checkpoint (2026-08-09)

- Both recording-system TOMLs now use the same D-I-P1-P2 Titan/Talos-class
  engineering reconstruction supplied by the user.  Each lens is explicitly
  tagged `engineering_reconstruction_not_oem`; no production drawing or OEM
  measurement is claimed.
- The magnetic-yoke defaults are D = D162 x 68 mm, I = D158 x 60 mm,
  P1 = D158 x 62 mm and P2 = D165 x 70 mm.  The independent TOML coil rows
  now use the supplied ID/OD/axial lengths: 56/140/44, 60/138/38,
  58/138/40 and 66/145/45 mm respectively.
- Pole shoulder OD / bore / gap are now D = 54/10/4, I = 55/12/6,
  P1 = 54/10/5 and P2 = 62/15/7 mm.  The supplied 12/12/13/14 mm nose
  lengths, nominal 63 deg cone metadata, 3 mm face land and R2-R4 fillet
  range are TOML-owned.  Pole-face tip OD remains a clearly disclosed
  schematic value because it was not present in the supplied public data.
- A 0.75 mm vacuum-liner wall gives clear IDs of 8.5, 10.5, 8.5 and 13.5 mm.
  The outer non-magnetic housings add 1 mm radial clearance and a 2 mm shell
  around the reference yokes, so their total ODs are 168, 164, 164 and
  171 mm, all inside the supplied engineering ranges.
- Lens and optical-reference centres remain exactly 82.5, 252.5, 432.5 and
  635.0 mm.  No excitation, field profile, polarity, preset or aberration
  parameter changed.  Python fallback definitions now read these mechanical
  values from the authoritative recording TOML instead of retaining the old
  D250-D300 hard-coded envelopes.
- Physical Layout consumes the explicit pole-nose axial length and reports
  the cone/land metadata in pole tooltips.  Validation checks provenance,
  liner-to-pole clearance, paired pole details, radial housing/yoke/coil
  nesting and unchanged centre/reference coupling in both recording systems.

## Latest exact specimen-interface checkpoint (2026-08-09)

- `src/temsim/physics/core.py` no longer rounds a requested propagation stop
  to the nearest axial grid point. Full RK4 intervals retain the requested
  `step_mm`; only the final interval is shortened so both endpoints are exact.
  CPU, Numba CPU and CUDA now consume the same per-interval step array.
- For the default FEG / C3 + Probe Corrector / Energy Filter preview, the
  incident bundle now ends at the authoritative sample Z = 1599.2 mm and the
  outgoing bundle starts at the same Z. The former 1599.2--1600.0 mm overlap
  is zero, and X/Y are continuous across the single shared specimen plane.
- Plotting was deliberately not used to hide or clip the defect. The physical
  propagation arrays themselves now satisfy `incident.z[-1] == sample.z_mm`
  and `branch.z[0] == sample.z_mm`, so downstream diagnostics that use the
  incident endpoint also recover the correct specimen plane.
- Completing the formerly omitted final 0.03885 mm exposed that the probe
  two-hexapole calibration had been fitted before the true specimen plane.
  The exact-plane calibration is now HP2 = `5.08073490e5 m^-3`, HP1/HP2 =
  `0.59513503`; the HP1 orientation and every mechanical coordinate are
  unchanged. At 0.1 mm its residual is 3.53% of the positive round-lens Cs
  contribution, versus 26.6% with the stale values.
- The probe-corrector residual remains below the existing 20% limit at 0.1,
  0.05 and 0.025 mm (3.53%, 11.27% and 13.86%). It is 43.2% at 0.2 mm, so the
  2.5 mm interactive preview remains a ray-layout schematic; quantitative
  corrector assessment requires the 0.1 mm high-accuracy path or finer.
- Regression coverage includes the previous +0.8 mm overshoot and -0.2 mm
  shortfall cases, the real default preview interface, CPU/Numba parity and a
  non-divisible 900.3 mm CUDA endpoint compared with CPU.

## Latest signed optical-transfer checkpoint (2026-08-09)

- `src/temsim/physics/first_order.py` defines the full paraxial state ordering
  `(x, y, theta_x, theta_y)` in the right-handed column frame and returns one
  signed 4x4 transfer in 2x2 blocks. The observable position relation is plain
  text: `r_plane = J_img @ r_sample + J_diff @ theta_sample`; `J_img` is
  dimensionless and `J_diff` is in m/rad.
- One reference ray plus four transverse basis rays are traced together.
  Subtracting the reference removes affine beam shifts from the Jacobian.
  Spherical-aberration kicks and hexapole nonlinearities are deliberately off
  for this first-order derivative; the ordinary nonlinear ray trace is
  unchanged.
- Ray-simulation metrics now expose both matrices, rotation, handedness,
  anisotropy and image/diffraction conjugacy residuals. Equivalent scalar
  magnification and camera length come from `sqrt(abs(det(matrix)))`, so the
  full signed X-Y map remains available instead of being discarded by `abs`.
- The **Optical Transfer** GUI page shows named objective/recording planes and
  can capture one Image state plus one Diffraction state at the same plane. It
  reports the normalised diffraction-vector-to-image-direction transform,
  including rotation, mirroring and anisotropy. The mapped reciprocal vector
  is normal to lattice planes; it is not a direct-lattice length map.
- Both recording-system TOMLs now own camera-axis rotation, U/V flips,
  uncertainty, status and source. Current values are
  `uncalibrated_identity` with 180 deg uncertainty. The GUI therefore labels
  the result model-only and refuses to imply an absolute hardware crystal
  orientation; current lens polarity provenance is also shown as provisional.
- The matrices cover the straight axial column and expose the Energy Filter
  entrance as a chain boundary. They do not yet include the curved sector,
  M01--M10 or Zebra detector axes. No mechanical coordinate was changed.
- Validation includes the exact field-free drift matrix, signed
  rotation/reflection/anisotropy decomposition, calibrated and uncalibrated
  detector-frame relations, TOML rejection cases, simulation metrics and the
  two-mode Qt capture workflow. A separate read-only audit produced finite
  transfer matrices for all 30 assemblies.

## Latest magnetic-field polarity checkpoint (2026-08-09)

- All 52 configured optical magnetic-lens definitions now own
  `field_polarity = +1/-1`, `field_polarity_status` and
  `field_polarity_source` in their selected instrument TOML. The sign is the
  effective Bz direction in the source-to-detector +z coordinate and remains
  independent of the non-negative 0–100% excitation magnitude.
- The former `fei_column_polarity.py` Python tables and every automatic
  reapplication of those tables were removed. Constructors use the default
  manifest, and selecting a new assembly applies the selected manifest with
  `preserve_operating_parameters=False`.
- Ordinary recalculation continues to preserve a user's runtime `polarity`
  override. Microprobe/Nanoprobe Mini Condenser reversal is now an explicit
  `field_polarity` entry in `configs/operating_modes/catalog.toml`, not an
  illumination-mode branch in Python.
- Manifest validation rejects missing, Boolean, floating-point or non-unit
  signs and requires controlled provenance status plus a non-empty source.
  Magnetic Field tooltips report the selected sign, status and source.
- All present common-column, projector and corrector signs are tagged
  `provisional_model_assumption`. A negative AutoScript/FLC raw value must not
  be used as proof of negative Bz; replace a provisional sign only with service
  coil/current mapping or an absolute image/diffraction rotation calibration.
- The repeatable migration/check is
  `scripts/migrate_field_polarity_to_manifests.py`. A catalog-wide audit matched
  384 selected magnetic-lens instances across all 30 assemblies to their TOML
  signs without changing any mechanical coordinate.

## Latest layout-to-editor navigation checkpoint (2026-08-09)

- Physical Layout and the separate Energy Filter branch now treat component
  names, centre markers and drawn bodies as single-click navigation targets.
  Dashed leaders remain visible guides but are deliberately not click targets,
  so a leader crossing a housing cannot steal that housing's selection. Each
  selectable item resolves to the existing canonical component key; no
  duplicate display-only component registry was introduced.
- A plot selection reopens/raises the instrument dock, switches to the correct
  Optical filter or Mechanical page, selects the tree item and expands the
  parameter half of the left splitter. Runtime targets open on **Operating**;
  static assembly parts open on **TOML**. Re-clicking the already selected
  element deliberately refreshes the editor context.
- Iliad XO / Optional EFTEM Energy Slit exposes only its live `inserted`,
  requested centre-loss
  and requested energy-width controls. Centre and width edits go through
  `configure_energy_window`, so blade centre and physical gap stay consistent
  with calibrated dispersion and travel limits. Mechanical slit geometry and
  derived blade positions remain TOML-owned/read-only in this interface.
- GUI regressions exercise a Physical Layout label click, Iliad slit and
  dynamic-focus-quadrupole label clicks, an M01 housing-body click, TOML
  routing for mechanics without a field model, and the complete slit
  navigation/editing path.

## Latest Iliad public-topology checkpoint (2026-08-09)

- `EnergyFilter.toml` no longer stores all branch geometry on the synthetic
  `energy_filter` interface. That row now owns only the branch interface and
  public-topology metadata. One large tapered prism, M01--M10, XO/optional
  EFTEM slit, dynamic-focus electrostatic quadrupole, bias tube, fast shutter,
  camera deflector, optional EFTEM output plane and Zebra each have one unique
  TOML row and canonical key.
- Validation requires exactly one prism and ten multipoles. It records that
  most multipoles are publicly described as dodecapoles but keeps each
  numbered pole assignment `not_public`; no guessed production BOM is
  presented as fact. M01--M10 are explicitly simulator model indices because
  public evidence does not expose production labels or exact internal order.
- The previous 135 mm radius / 90 deg bend and all carrier/device coordinates
  remain usable but are now explicitly `provisional_parameterized_non_oem`.
  The 90 deg value is tagged as a patent-example starting point, not a
  confirmed Iliad product angle. Unknown prism yoke, multipole bore/envelope,
  electrode and detector-package dimensions remain editable rather than
  manufacturer-labelled.
- M01--M03 retain 22 mm provisional housings and M04--M10 retain 28 mm, with
  independent per-element TOML ownership and 20 mm magnetic supports.
  Validation rejects support/housing inconsistencies, overlaps and wrong path
  ordering. Runtime multipoles retain the definition ID of their own TOML row.
- The entrance aperture uses the reported 5 mm experimental condition without
  treating it as the only installed mechanism size. Zebra now correctly uses
  28.672 x 0.800 mm for each 2048-pixel strip. The separate 256 x 2048
  alignment area is 3.584 x 28.672 mm; the old model incorrectly used its
  3.584 mm height as the strip acceptance. Strip pitch and package remain
  explicitly unknown; the current 1.0 mm pitch is provisional and editable.
- The dedicated Energy Filter view now draws the entrance aperture, prism
  clear path, ten hollow carriers, XO/slit, provisional electrostatic
  envelopes, EFTEM output and the Zebra active plane. Curvilinear-only parts
  are excluded from main-column axial markers and Physical Layout records.
  X and Z remain independently zoomable.
- The dynamic-focus electrostatic quadrupole is present as a four-electrode
  mechanical placeholder with
  `mechanical_layout_only_dynamic_focus_field_not_implemented`; it does not
  silently add an unvalidated field or an eleventh member to the confirmed
  ten-multipole system. The slit and shutter remain separate components with
  separate functions.

## Latest wave-optics checkpoint (2026-08-09)

- The symmetric split-operator multislice engine accepts both total 2-D
  projected potentials and explicit `(Z, Y, X)` finite-projection slices. It
  supports rectangular grids and independent X/Y sampling.
- NumPy complex128 remains the CPU reference. Optional CuPy complex64 supports
  multislice and TEM/STEM FFTs with a complete CPU retry after CUDA, driver,
  allocation or FFT failure. Install with `pip install -e ".[gpu]"`.
- Angle-resolved STEM now has one device-resident compound CUDA pipeline:
  scan positions and all potential configurations upload once; probe formation,
  multislice, diffraction FFT, frozen-phonon intensity averaging and detector
  masks remain on device; one stacked detector array returns to the host.
- Each resident multislice calculation builds one reusable CUDA plan. The
  reciprocal-frequency grid, 2/3-bandwidth mask and uniform or nonuniform
  slice-geometry propagators remain cached across all probe batches and all
  potential configurations. Potential shape/finite checks and maximum-phase
  scans run once per configuration instead of once per batch.
- Any resident-pipeline failure discards every partial CUDA result and reruns
  the complete STEM observable on the NumPy complex128 reference. Metrics make
  residency, fallback, upload count, batch count and result-transfer bytes
  visible.
- Silicon [110] and gold [001] now own TOML atomistic definitions. Optional
  abTEM 1.0.10 + ASE build unstrained, commensurate periodic supercells and
  Lobato--Van Dyck finite Z-slice neutral-atom IAM potentials. Install with
  `pip install -e ".[atomistic]"`.
- Frozen phonons use reproducible independent isotropic Gaussian (Einstein)
  displacements. The preset or user value is the one-axis RMS sigma. TEM and
  STEM average configuration intensities, not complex amplitudes, and report
  a finite-ensemble relative standard error.
- The model intentionally excludes bonded charge redistribution, correlated
  phonons, absorptive/inelastic potentials, magnetic specimen fields and spin.
  The STEM wave path does not add a separate Rutherford/TDS tail, so frozen
  phonon scattering is not double counted.
- The Sample parameter page exposes atomistic/frozen-phonon toggles,
  configuration count, RMS sigma and random seed. Dependent controls disable
  when multislice, atomistic potential or frozen phonons are inactive. Saved
  operating profiles round-trip all five fields.
- Source-qualified thermal defaults are 0.085 angstrom for Si near room
  temperature and a rounded 0.080 angstrom 300 K Debye estimate for Au. Both
  references are stored in their specimen TOMLs.
- Numerical regression checks crystal number density, abTEM transmission-phase
  units, seeded reproducibility, TEM/STEM incoherent averaging, rectangular
  sampling, integrated intensity, reusable-plan identity, nonuniform slice
  geometry, and CPU/CUDA agreement for atomic slices.
- Warm CUDA benchmark (96 x 96, 12 slices, 64 scan positions, two potential
  configurations, two detectors, batch 8): resident median 0.1157 s versus
  0.1484 s for the former per-configuration host round trip (1.28x). Bulk
  array transfers fell from 32 / 13.5 MiB to 1 / 1.5 KiB. This is a local
  software benchmark, not a universal hardware-performance claim.
- Isolated warm propagation benchmark for the same 96 x 96, 12-slice,
  64-probe, two-configuration workload: rebuilding device grids and
  propagators for every call took a 0.0677 s median; one plan reused for all
  16 calls took 0.0505 s including its build and one-time potential checks
  (1.34x, relative L2 difference 0). The plan retained 189 KiB of cached
  arrays. This is likewise a local software benchmark rather than a hardware
  guarantee.

## Scan / descan checkpoint (2026-08-11)

- Sample now has an explicit inserted/retracted state. Retraction preserves the
  exact sample Z as the optical probe-reference plane but disables ray
  diffraction/diffuse broadening and sets interacting wave thickness to zero.
  A dormant or invalid CIF path is not touched while the holder is retracted.
  Ray Diagram and Physical Layout both distinguish the parked holder from an
  inserted specimen.
- AC Scan is one shared command driving physical upper and lower foils. The
  lower-foil X/Y map is recalculated from the active signed first-order column
  optics so the combined sample-plane angular response is zero. A field-free
  equal-and-opposite pair is used only if the lower response is singular.
- AC raster scale is now specified by X/Y pixel counts and one square
  specimen-plane pixel pitch. The control domain is 0.001 nm through 1 mm and
  the displayed FOV is exactly `count x pitch`; raster coordinates denote
  pixel centres, so centre-to-centre span is `(count - 1) x pitch`. A second
  signed 2x2 calibration maps those requested specimen axes through the active
  optics to coil commands. Singular transfers and demands above the physical
  coil limit fail explicitly instead of silently changing the FOV.
- AC Scan and AC Descan now expose exactly the same raster and two-foil
  controls. Shared pixel count, line count, pitch, and frame period are kept
  synchronized; the derived lower-foil control is read-only for both pairs.
  All five column TOMLs place their foil-pair centres at equal distances on
  opposite sides of the sample and keep identical foil length, gap, and
  effective thickness. Manifest validation rejects geometry that breaks this
  symmetry.
- Scan geometry and STEM acquisition use the same physical foil planes and
  signed 2x2 response model; neither path substitutes one kick at a mechanical
  centre. Descan receives the exact negative of the calibrated AC command.
  Its lower-foil 2x2 coupling is solved through the current post-sample optics
  so the combined AC plus Descan displacement vanishes at the Selected Area
  Aperture image-reference station. Singular or unreachable solutions fail
  transactionally and restore the previous state.
- Objective Aperture and Selected Area Aperture are reported as physical
  diffraction- and image-reference stations. They are not forced to carry
  those ideal labels under arbitrary lens settings: the sample-to-plane
  position and angle Jacobians classify each current plane as `image`,
  `diffraction`, or `mixed` and expose both residuals. Objective first-image
  and first-diffraction coordinates are refreshed after voltage, sample, or
  objective-lens changes. Propagation grids include requested plane Z values
  exactly instead of reading them from a coarser display-history sample.
- One active AC raster produces exactly one HAADF, DF and BF frame by
  integrating the physical detector acceptance at every probe position.  The
  interactive Preview uses the geometric detector-interception approximation;
  High accuracy can use the wave/multislice specimen model.
- Wave STEM applies descan as a first-order, per-probe shift of each physical
  detector's equivalent angular acceptance.  This is an explicit approximation
  and does not claim a full time-dependent post-specimen wave propagation.
- The calculated frame is cached in the STEM page.  While AC Scan
  remains enabled, a GUI timer repeatedly plays its raster-line acquisition;
  stopping scan stops the timer and retains the last complete frame.  Playback
  never launches repeated physics calculations.
- The calculation also caches AC/Descan first-order response bases on every
  displayed branch Z grid. Each playback tick adds only the current scan
  displacement to the cached rays. The View Angle projection remains live, so
  the diagram can be rotated while a frame is playing without retracing the
  column.
- Each BF/DF/HAADF panel reports its instrument-TOML Z, inner/outer active size
  and derived collection-angle interval. The angle uses the full active signed
  sample-to-detector transfer and exposes anisotropic min/max ranges when the
  two singular values differ materially.

## Specimen-mode checkpoint (2026-08-12)

- A central Sample tab now owns insert/retract, Real/Virtual mode, finite X/Y
  size and thickness, sample centre and scan origin. Its immutable geometry
  snapshot is shared by the renderer and STEM calculation and carries the
  finite box, scan FOV, calculation ROI, current probe, orientation and region
  state. OpenGL displays atoms/cell/box/+Z beam/FOV/ROI when supported; Qt
  offscreen/minimal or missing OpenGL uses the safe 2-D view.
- The Sample tab is also the sole GUI owner of specimen presets, TEM/STEM wave,
  multislice, frozen-phonon, Virtual interaction and probe-convolution
  controls. The duplicate `sample` entry and quick controls were removed from
  the left instrument tree; clicking the specimen in a diagnostic layout
  activates the central Sample tab instead.
- Sample owns specimen state/structure only and no longer duplicates the
  BF/DF/HAADF images. A custom CIF is orthogonalised with abTEM and expanded
  periodically through the finite-sample/current-ROI intersection. The view
  uses ASE covalent-neighbour bonds, reduced covalent-radius balls, ASE/Jmol
  colours and a beside-view element legend. Its default 2,500-atom soft limit
  may reduce only a clearly reported display window; the multislice ROI is
  unchanged. Above 3,000 user-selected atoms OpenGL uses point-sphere level of
  detail; the safe 2-D fallback remains a coloured ball-stick projection.
- The former Scan / Descan top-level tab is now STEM with exact Geometry and
  Images subtabs. Detector images are placed on centre-derived physical pixel
  edges in laboratory micrometres, their X/Y unit aspect is locked to one, and
  normal PyQtGraph pan/zoom remains enabled. A model notice explains that
  `geometric_detector_interception` polygons are detector-clipping boundaries,
  not specimen contrast; CIF multislice requires High accuracy with wave/
  multislice enabled. Sampling diagnostics compare FOV with the finite sample
  and pixel pitch with half the shortest periodic CIF atom spacing.
- `atomic` (UI: Real sample) accepts either an instrument-selected TOML preset
  or a user CIF/MCIF. One canonical `(w,x,y,z)` unit quaternion controls the
  physical orientation. A direct-lattice zone axis maps to laboratory +Z, a
  non-collinear in-plane direction maps to +X, and numeric or explicit mouse
  edit mode updates the same quaternion. Camera orbit remains the mouse
  default and draft physical edits require Apply.
- Custom CIF structures are loaded with ASE and orthogonalised with abTEM.
  Potential construction generates only the periodic neighbourhood needed by
  `scan ROI + probe padding` intersected with the finite specimen before exact
  rotation/cropping. It never expands a macroscopic sample in full. Outside
  the finite X/Y envelope is explicit vacuum; the 5,000,000-atom safety limit
  applies to the ROI-local pre-crop structure.
- A custom CIF requires atomistic IAM and multislice. It never borrows a TOML
  preset's thermal displacement and never silently falls back to another
  material. Frozen phonons accept a global user RMS or an explicit per-element
  RMS table for a custom CIF.
- Real sample Ray Diagram calculations never synthesize `+g/-g` or diffuse
  diffraction branches, even if legacy ray-preview fields remain in a loaded
  profile. Coherent CIF/TOML elastic diffraction belongs to the high-accuracy
  TEM/STEM wave/multislice calculation. Separate material-derived Real
  populations represent zero-loss, plasmon/low-loss, core ionisation, optional
  other inelastic loss and plural events. They use absolute Poisson
  probabilities, representative energy offsets and characteristic-angle
  quadrature; they are not user-authored diffraction beams. Only Virtual mode
  can create user-defined angular interaction branches.
- Ray Diagram hue is keyed to canonical interaction kind (incident, Real
  zero-loss/plasmon/ionisation/plural, or Virtual direct/diffraction/diffuse/
  arbitrary/Rutherford). Within each hue, five dark-to-bright
  bins encode the exact 3-D sample-plane convergence semi-angle relative to
  that branch's current-weighted chief ray. A common interaction kick is thus
  not counted as convergence; brightness saturates at the incident bundle's
  calculated 99%-current convergence semi-angle.
- Selecting/dragging an axial Z cursor calculates a current-weighted plane
  interaction budget using every ray weight (not the 48-ray display subset).
  It reports sample-conditional probabilities, source fractions at Z,
  composition at Z, effective absorption and pre/post-sample stops; these
  categories close to source probability one. Elastic wave redistribution can
  coexist with every inelastic energy state and is never presented as an
  exclusive collision label.
- `virtual` mode owns extensible diffraction spot/ring, Gaussian diffuse,
  arbitrary angular, user screened power-law, physical screened relativistic
  Rutherford and absorption rows. All probabilities are absolute and are
  rejected above one; they are not silently normalised. The direct beam is the
  exact remainder. The physical row integrates `2*pi*sin(theta) dtheta` and
  uses `P=1-exp(-N_areal*sigma)`; it is explicitly not a Mott calculation.
- Virtual rectangles, ellipses and NPY/PNG/TIFF grayscale maps define density
  inside the finite slab. Outside is vacuum. The per-pixel interaction
  probability is convolved with the calculated probe when enabled.
- High-accuracy wave intensity is integrated only over strict reciprocal-space
  support and is not renormalised when a detector extends outside it. An
  optional, separately reported Rutherford approximation begins strictly
  beyond that support and scales the wave channel to preserve probability;
  it is disabled by default. Bonding charge, absorptive/inelastic multislice
  potentials, energy-differential dielectric/EELS spectra, magnetic scattering
  and full Mott elastic scattering remain out of scope. Stochastic inelastic
  populations are instead transported by the separate IMFP model.
- Each STEM result now carries detector fraction images, pA, expected electrons
  per pixel, optional reproducible Poisson counts, dwell, uncollected/absorbed/
  truncated channels, separate high-angle-tail images, laboratory axis/order
  metadata and the sample-plane ProbeState. Physical detector `hit_mask`
  geometry is evaluated after the full signed 2x2 transfer in axial order.
- Operating profiles are format v2 for quaternion/zone metadata, interaction
  and region tables, map paths and per-element RMS values. Format v1 remains
  readable and its two legacy relative-weight controls migrate once to
  absolute-probability rows. Detector geometry remains instrument-TOML owned.

## Current validated state

- The instrument catalog contains 10 module TOMLs, 466 part definitions and 15
  selectable Energy Filter assembly combinations.
- High-accuracy defaults target a 32 GiB workstation and use a conservative
  24 GiB application-memory preflight limit.
- Magnetic-lens excitation is limited to 0–100%. A lens needing a stronger
  field must raise its calibrated 100% field value rather than exceed 100%.
- Column/vacuum walls only cut off rays. They do not clip propagation or the
  mathematical support of a lens magnetic field.
- Ray stops use the earliest physical intersection among vacuum walls,
  apertures and recording devices.
- Ray Diagram supports continuous transverse viewing angle. Rotating X/Y keeps
  the current zoom, Z scale and Z=0 position unchanged.
- Physical Layout names use one dynamic screen-space callout packer. Ordinary
  components, upper/lower Objective mechanics, stage/holder and recording
  devices can occupy many rows above or below the column; a dashed leader ties
  every visible name to the component centre/outer mechanical edge. Relayout
  changes display coordinates only and never alters TOML geometry.
- Gun trajectories retain equal-laboratory-time snapshots and per-ray arrival
  times for future wavefront/arrival-arc overlays at important planes.
- Operating-mode storage provides calculated condenser `micro_probe`/
  `nano_probe` and projector `imaging`/`diffraction` lens/aperture values.

## Mechanical lens model

Every configured round magnetic lens currently has one optical parent plus
independent mechanical children:

```text
<lens>                         optical/field parent; no material drawing
├── <lens>_housing             non-magnetic outer housing
├── <lens>_yoke                soft-magnetic yoke
├── <lens>_excitation_coil     insulated copper winding package
├── <lens>_upper_pole          upstream pole piece, where applicable
└── <lens>_lower_pole          downstream pole piece, where applicable
```

Mechanical-only parts use these TOML fields:

- `local_start_z_mm`, `local_center_z_mm`, `local_end_z_mm`, `length_mm`
- `vacuum_inner_diameter_mm`
- `mechanical_inner_diameter_mm`, `mechanical_outer_diameter_mm`
- `mechanical_profile`, `mechanical_part_role`, `material_class`
- `mechanical_only = true`, `parent_key`
- `mechanical_overlap_group`, `mechanical_overlap_role`,
  `mechanical_overlap_reason`
- Excitation coils additionally use `field_source_key = "<parent lens>"`.

The mechanical children never create optical elements or additional magnetic
field sources. Existing lens strengths, apertures and field formulae were not
changed by the split.

Canonical profiles and key helpers are in
`src/temsim/mechanical_profiles.py`. Assembly validation is in
`src/temsim/module_manifest.py`. The repeatable migration is
`scripts/migrate_lens_mechanical_layers.py`.

## Confirmed and provisional topology

Confirmed:

- C1 and C2 are adjacent hollow cylindrical lens assemblies.
- The C1–C2 interface has one C1 lower pole and one C2 upper pole facing each
  other. Their gap midpoint is the confirmed target region for the C1–C2
  crossover.
- A three-condenser system gives C3 its own two-pole/single-gap assembly.
- Diffraction Lens and Intermediate Lens mechanical envelopes do not overlap;
  they retain at least 5 mm axial clearance for future pole-piece configuration.
- The innermost continuous wall is called the **vacuum liner**. Use
  **alignment tube** only for a separately adjustable alignment component.

Provisional:

- Housing/yoke/coil radial dimensions are schematic initial ratios derived
  from each existing lens envelope, not measurements of FEI hardware.
- Diffraction Lens, Intermediate Lens, P1 and P2 are presently represented as
  independent two-pole/single-gap lenses.
- A real FEI projector column may share a central pole piece or magnetic yoke
  between adjacent excitation stages. Public information does not yet confirm
  that every software-named projector lens owns two mechanically independent
  pole pieces.
- Pole geometry currently affects the mechanical drawing only; it does not
  reshape the solver magnetic field.
- Vacuum-liner wall thickness is a provisional 0.25 mm.

Do not label provisional projector topology as an exact FEI/Titan mechanical
reconstruction without a service drawing, section drawing or measured part.

## Real-part photo intake

Photos can be used to replace provisional dimensions. For a useful measurement
handoff, request:

- Front, side and top views, photographed as orthogonally as possible.
- A ruler, caliper or other known dimension in the same plane as the part.
- At least one confirmed dimension such as bore diameter, flange diameter or
  mounting-hole spacing.
- Part name/number and the lens/stage it came from.
- Disassembled or section views when internal geometry is relevant.

For each inferred dimension, record:

```text
value_mm = ...
uncertainty_mm = ...
evidence = "photo/drawing/measurement identifier"
confidence = "confirmed | measured | estimated | provisional"
```

A single unscaled oblique photo supports shape and ratio estimates only. It
cannot establish reliable absolute dimensions, hidden bores, winding data,
material grade, permeability or saturation field.

## Next work

1. Measure absolute camera/display axes and image/diffraction rotation using a
   known specimen/stage reference, then replace the identity calibration and
   provisional field polarities with evidence-backed values and uncertainties.
2. Extend the first-order transfer chain through the curved Energy Filter
   sector, M01--M10 and Zebra/output detector before using EFTEM/EELS output
   coordinates for crystallographic orientation.
3. Add thickness-dependent frozen-phonon convergence studies and report how
   detector/image uncertainty changes with configuration count. Do not encode
   one universal "converged" count.
4. Add a real amorphous atomistic builder and an explicitly separate
   absorptive/inelastic model before making quantitative HAADF or EELS claims.
5. Replace the analytic Objective field envelope with a source-qualified
   upper-pole/gap/lower-pole field distribution or an imported field map.
6. Continue ingesting real-part photographs or section drawings and replace
   provisional housing/yoke/coil ratios with evidence-backed dimensions.

## Guardrails for future changes

- Do not alter all lens strengths while crossover locations are incomplete.
- Keep every operating excitation at or below 100%.
- Do not use mechanical lens dimensions to truncate a lens field.
- Do not make a mechanical-only child contribute a second magnetic field.
- Preserve TOML as the authority for static geometry and optical references.
- Preserve instrument TOML as the authority for default magnetic-field
  polarity and its provenance; do not reconstruct Bz signs from signed raw
  hardware-control values.
- Preserve signed 2x2 optical maps until the final scalar display. Never use a
  single absolute transfer coefficient to infer rotation or handedness.
- Do not claim absolute crystal orientation while detector axes or contributing
  magnetic-field polarities remain uncalibrated/provisional.
- Keep frozen-phonon scattering and any empirical Rutherford/TDS tail mutually
  exclusive unless a validated coupling explicitly prevents double counting.
- Never call the IAM result a bonded-charge, inelastic or magnetic specimen
  calculation.
- Validate all 15 selectable assemblies and run the full test suite after
  topology edits; retain manifest validation for the historical no-filter TOML.
- Update this file whenever a provisional assumption becomes measured or
  confirmed, or when the next-work ordering materially changes.

## Known documentation mismatch

No known catalog-count mismatch remains at this checkpoint.
