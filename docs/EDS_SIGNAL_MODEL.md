# Generic EDS signal model

The simulator names this subsystem **EDS**. The currently installed six-segment geometry is an evidence-bounded reference configuration, not the identity of the signal engine. Product names may remain in provenance URLs or historical research notes, but are not used as the system key, GUI title, spectrum name or result label.

## Implemented trajectory and signal phase

- `configs/specimen_supports/catalog.toml` defines a 3.05 mm circular support, a nominal 25 um foil, Cu, Au and virtual-vacuum materials, and commercial 50–500 square-mesh dimensions.
- `temsim.specimen.support` classifies a point as opening, bar, rim, outside or virtual vacuum. The grid offset is the centre of an opening and rotation is continuous.
- `temsim.detector.eds_atomic` evaluates the Bote–Salvat K, L1–L3 and M1–M5 electron-impact ionisation fits for Z=1–99.
- `temsim.detector.eds_signal` accepts weighted electron track segments in any material. A segment can explicitly identify a primary or elastically scattered history; the vacancy and photon calculation is identical.
- `temsim.specimen.elastic_transport` generates event-by-event three-dimensional elastic trajectories through the finite rectangular or disk specimen and the downstream support. It resolves mesh openings, sidewalls, bars, the annular rim and material/vacuum boundaries.
- The explicit point calculation takes its history count directly from the upstream column result: every weighted ray surviving to the physical sample plane is transported once. There is no independent EDS trajectory-count input. Sample-plane X/Y, both incident slopes (including calculated rotation), source energy offset and source-current weight are retained. The point coordinate translates only the weighted beam centroid; it does not replace the calculated phase-space spread.
- **Straight primary reference** remains available as a deterministic diagnostic. Retained histories and collision points are displayed in **Sample Interactions 3D**, while every reaching ray contributes to the EDS calculation. Its `3D / X-Z / Y-Z` controls render the same cached scene without rerunning transport. **EDS** contains only the spectrum, short status and hover energy/counts; it has no separate trajectory plots or line table.
- Direct-vacancy fluorescence yield, radiative transition probability, line energy, atomic weight, elemental density and photon mass attenuation come from xraylib. Each result records the library version.
- The installed angular acceptance supplies the collection fraction. The user can select the holder-conditioned or unshadowed aggregate solid angle.
- The detector response currently supports an explicit ideal scalar efficiency, optional Gaussian energy broadening and reproducible Poisson counting.
- **Update point EDS** and the existing EDS/support settings are available in **Sample Interactions 3D > Parameters**. They reuse compatible shared High-accuracy products. Merely changing the view or opening the settings does not calculate a spectrum or lens preset.
- **Run sample-region high accuracy** is an explicit local calculation that reuses compatible shared products. Its
  adjustable entry plane samples the cached upstream column phase space; the
  finite material kernel runs about the sample; forward terminal electrons are
  reinjected at the sample reference and propagated through the actual
  objective/downstream lenses, apertures, recording planes and column wall. The
  configured exit plane is saved as an explicit handoff diagnostic. Editing a
  relevant parameter invalidates this cached result instead of silently
  recalculating it.
- The bounded result draws stored material electron paths, backscatter and the
  downstream continuation in the main Ray Diagram. Rutherford is not presented
  as a separate particle species: it is the current approximation used for the
  elastic events. Channeling remains owned by the coherent wave/multislice
  result and is never double-counted as an additional stochastic branch.
- For the 3-D display, representative characteristic photons are sampled uniformly on the sphere
  (`cos(theta)` uniform in `[-1,1]`, azimuth uniform in `[0,2 pi)`) and travel in
  straight lines. Public data do not provide a sensor face or distance, so the
  displayed endpoint is schematic. Detection uses an azimuth-partitioned
  elevation band whose aggregate spherical area exactly equals the selected
  holder-conditioned or unshadowed solid angle. This is explicitly an angular
  acceptance surrogate, not an invented detector intersection. The spectrum
  uses deterministic detector quadrature, not the finite display-ray population.

## Weighted overlap integration (2026-09-08)

**Sample Interactions 3D > Parameters > Weighted beam / sample overlap (EDS)**
is enabled by default. `sample.eds_overlap_sampling_points` defaults to 256
(32–4096); `sample.eds_overlap_sampling_enabled` can restore the original
discrete-ray estimate. Both settings are saved in operating profiles.

The automatic estimator addresses a small finite specimen illuminated by a
broad beam when fewer than 32 original histories cross the sample. It estimates
the continuous incident density using a weighted Gaussian kernel **mixture**
with Scott bandwidth, and integrates additional Sobol points over the finite
material intersection. Kernel-posterior selection retains an original ray's
direction and energy for each point. The local elastic solver still determines
whether each proposed path actually enters material, its length, and scattering.

For conditional incident density `f(x)`, proposal area `A` and `N` proposed
points, a point carries `q = f(x) A / N`. Rejecting vacuum points does not change
`N`. Although auxiliary elastic transport internally normalizes its inputs,
every resulting EDS material-path and emission-flight weight is multiplied by
the original `sum(q)` before ionisation. Thus dose remains
`source electrons × column survival`, and overlap enters exactly once through
path weights. The unsampled vacuum complement is not reassigned to the sample.

The original elastic histories and downstream terminal electrons remain
unchanged. `EDSSpectrum.material_quadrature` stores independent auxiliary
paths, all their emission flights, separate integration IDs and kernel-parent
source IDs. EDS ledger entries do not claim those integration IDs are original
electrons. Material paths replace the original EDS sample estimator; they are
not added to it. Only EDS and dependent combined-result cache identities change.

This first implementation is limited to vacuum supports, sufficiently sampled
two-dimensional incident density, a small thin specimen, small incident angles,
and a locally bounded analytic axial field. Material supports, degenerate or
poorly sampled density, resolved/focused illumination, thick/steep geometries,
imported field maps and unsupported local fields retain the original estimator
with a visible reason. The exact guards and their values are exported in
diagnostics. No intensity floor is imposed.

KDE bandwidth smooths unresolved structure and has Gaussian tails. This is an
explicit continuous-density approximation, not a unique reconstruction of the
true beam and not a coherent-wave solver. Increasing integration points tests
quadrature/trajectory stability; increasing upstream rays and checking the
beam model is still needed to assess density bias. Nonzero expected counts can
remain far below one, so Poisson-sampled spectra may still be zero. Coherent
BF/DF/HAADF retain their full wave domain, including vacuum propagation and
phase; EDS quadrature is not substituted for those images.

## Spectrum peak labels

**Peak labels** identifies positive simulated line contributions in the displayed
result. It follows retained/cached spectra, not the current sample draft.
Contributions from multiple vacancies or materials are grouped by exact element,
transition and energy; the hover tooltip names their sources.

The element selector provides an offline **Library** reference using the same
xraylib `LineEnergy` / `RadRate` tables as signal production (Z=1–99, where
transitions are tabulated). References are labelled **Ref** and are not evidence
of a detected element. H and He have no tabulated characteristic emission lines.
Neither selecting a reference nor hiding labels changes counts or runs physics.

At most 16 strong, visible labels are drawn; unresolved same-element neighbours
are condensed for readability. Hover retains nearby individual transition
energies alongside the existing spectrum energy/count readout. Branching rates
in the reference library are not abundance or predicted detector intensity.
Annotations do not change the user's plot range.

## Vacancy production and dose

For element Z and path segment length ds, the mean shell-vacancy count is

`lambda_Zs = N_e w_track n_Z sigma_Zs(E) ds`,

with `n_Z = rho w_Z N_A / A_Z`. The emitted line expectation is

`N_emitted = lambda_Zs omega_s R_line`.

Here `E` is in eV, `n_Z` in cm^-3, `sigma` in cm^2 and `ds` in cm. Vacancy expectations are not capped at one per electron. Fluorescence/Auger yields and an unresolved shell-transfer remainder share the same vacancy record. The event ledger reuses these records; it does not repeat the shell-ionisation calculation.

Default point dwell is `scan_frame_period_s / (scan_pixels_x * scan_lines)`. Source electrons are `effective_source_current_pa * 1e-12 * dwell_s / e`. In elastic Monte Carlo mode this is multiplied by the upstream survival fraction; surviving-ray weights are normalized conditionally. Column/aperture losses are therefore applied once. Explicit dwell/electron arguments override these defaults.

The selectable **Straight primary reference** uses nominal beam voltage and an axial path at the requested point. It does not use the upstream ray energy spread, incidence angles or survival fraction and is not equivalent to the default ray-driven acquisition.

Implementation: `src/temsim/detector/eds_signal.py` (`simulate_eds_point`, `simulate_eds_tracks`) and `src/temsim/specimen/interaction_engine.py`.

## Photon attenuation and collection

Normal state-aware acquisition transports each weighted photon representative through the configured geometry before forming the spectrum:

- Finite specimen intersections use the sample envelope, dimensions, material and orientation.
- Support absorption uses the selected material and mesh bar/rim encountered at the support mid-plane. This is a planar-layer approximation, not full photon tracking through every mesh sidewall.
- Objective pole intersections use the current assembly geometry as opaque shadows. No unsourced pole-alloy attenuation curve is assigned.
- Additional holder occluders and sourced detector faces are supported by the API. The current GUI does not supply these objects; it uses the aggregate angular acceptance.

For each material interval, `T = exp[-rho * (mu/rho)(E) * L_cm]`; interval transmissions multiply. Mixture mass attenuation is `sum_Z(w_Z * xraylib.CS_Total(Z, E_keV))`. A hard shadow has zero transmission. Unsupported attenuation-table energies are treated as absorbed, not extrapolated.

Without sensor-face dimensions, quadrature weights sum to `N_emitted * eta * Omega / (4 pi)` before attenuation. Default quadrature order is one representative per detector segment. This is an isotropic angular-acceptance surrogate, not a resolved physical sensor hit. With supplied faces, area quadrature uses `cos(theta) dA / r^2`; the global scalar efficiency and the face-relative response are each applied once.

Emission positions remain approximate. By default only up to 256 detailed elastic histories retain material flights. Vacancies with stored flights use one reproducible length-weighted representative position; those without flights fall back to the point-acquisition X/Y and specimen reference plane or support mid-plane. All histories still contribute their weighted material paths to vacancy production, but not all have fully resolved photon emission positions.

The lower-level `simulate_eds_tracks(state=None)` reference instead uses uniform-depth emitting-layer self-absorption:

`T_mean = [1 - exp(-tau)] / tau`, with `tau = rho * (mu/rho) * t_cm / sin(takeoff)`.

It has no cross-layer or pole shadow transport. In state-aware acquisition, transported counts replace this reference count; the old `self_absorption_transmission` line field remains diagnostic and is not multiplied into the final spectrum again.

Implementation: `src/temsim/detector/eds_photon_transport.py` and `src/temsim/detector/eds_signal.py`.

## Source energy and detector response

- Elastic Monte Carlo retains each arriving ray's `E_i = 1000 * beam_voltage_kv + energy_offset_ev[i]`. This affects the electron-impact ionisation cross section and line intensity. Electron energy is currently constant along that elastic history; independent inelastic branches do not provide continuous slowing to EDS tracks.
- Characteristic line energies come from atomic transition data. Source energy offsets are not added to photon energies or to the detector line width.
- `eds_energy_resolution_fwhm_ev` applies one energy-independent Gaussian to every line: `sigma = FWHM / (2 sqrt(2 ln(2)))`. Default zero means an ideal line placed in one energy bin; default bin width is 10 eV.
- Gaussian weights are sampled over approximately +/-5 sigma and normalized within the spectrum. This is a simplified response, not a calibrated spectral-edge-loss model.
- Optional Poisson sampling acts once on the final attenuated, broadened expected spectrum. It is disabled by default. The GUI displays sampled counts when present, otherwise expected counts.
- No separate Fano term, electronic readout-noise term, natural line width, escape peak, pile-up or dead-time model is implemented. A supplied Gaussian FWHM can represent an effective measured resolution, but its individual physical contributions are not calculated.
- `eds_detector_efficiency` is a constant scalar, default 1. `windowless` is product/geometry metadata, not a window-response calculation. Window/contamination absorption, detector dead layer and silicon thickness do not currently produce an energy-dependent sensor efficiency.

## Elastic trajectory kernel

For material mass density `rho`, element mass fraction `C_i`, atomic mass `A_i` and total elastic cross section `sigma_i`, the macroscopic event rate is

`1 / lambda_el = rho N_A sum_i(C_i sigma_i / A_i)`.

Each material flight is sampled as `L = -lambda_el log(R)`. If a finite geometry boundary occurs first, the trajectory stops exactly at that boundary, enters the adjacent material or vacuum, and resamples the memoryless free path there. At a collision, the element is selected in proportion to `n_i sigma_i`.

The current offline cross-section provider is the relativistic screened-Rutherford approximation from Demers et al. (their equations 10 and 11), with energy `E` in keV:

`sigma_i = 5.21e-21 (Z_i/E)^2 [4 pi / (delta_i (1 + delta_i))] [(E + 511) / (E + 1022)] cm2`,

`delta_i = 3.4e-3 Z_i^(2/3) / E`.

With `u = sin^2(theta/2)`, its normalized angular density and inverse sampler are

`p(u) = delta (1 + delta) / (u + delta)^2`,

`u = R delta / (1 + delta - R)`, with a uniform azimuth in `[0, 2 pi)`.

The direction rotation is performed in a local orthonormal frame and is tested to preserve unit norm and the requested polar angle. Elastic collision energy loss/nuclear recoil is currently zero, so kinetic energy remains constant along one trajectory. EDS path segments before the first collision are marked `straight_primary` (or `straight_primary_after_sample` in the grid); later segments are marked `elastic_scattered`.

## Explicit current limits

This phase is an elastic electron-trajectory and characteristic-line engine, not yet a complete coupled electron/photon Monte Carlo.

- The point EDS calculation now defaults to the event-driven elastic Monte Carlo; a straight primary path remains an explicit selectable reference.
- The present screened-Rutherford model is stated for 100--300 keV and the cited work warns that it is imprecise for `Z > 30`. Results that encounter heavier elements carry a warning. Quantitative work should replace the provider with ELSEPA/Dirac partial-wave differential cross sections; the project does not redistribute the restricted NIST SRD 64 tables.
- The specimen kernel uses bulk density and elemental mass fractions (independent-atom, amorphous-style transport). It does not reproduce crystallographic channeling or coherent diffraction. A crystalline multislice wave has no unique classical path, and this EDS calculation is not derived from or added to the multislice intensity.
- The initial trajectory ensemble is the geometrical-ray phase space calculated at the physical sample plane. It includes source position/angle sampling, lens/deflector transport, aperture and wall survival, Larmor-rotated X/Y slopes, energy offsets and source weights. It is still a classical ray ensemble: coherent aberrated probe-wave intensity and crystallographic channeling are not converted into unique classical paths.
- The elastic kernel uses axis-aligned finite foil boundaries and a support beginning at its downstream face. Photon specimen intersections use the supplied orientation. This is not yet one fully general, shared arbitrary-solid electron/photon transport model.
- Electron slowing, quantitative secondary-electron yield/energy transport, discrete multiple-ionisation transport, bremsstrahlung, vacancy cascades/Coster–Kronig transfer and secondary fluorescence remain unimplemented. The spectrum contains characteristic lines, not a complete experimental background or detector-artifact spectrum.
- The current geometry lacks published active area, sensor distance, crystal thickness, dead layer and window/contamination data. Absolute counts therefore remain a provisional model even though the shell cross sections, atomic relaxation data and configured aggregate solid angle are traceable.
- The existing aggregate core-ionisation mean free path is not used to scale element-specific EDS production.

## Data provenance

- Commercial grid dimensions: [PELCO TEM grids](https://www.tedpella.com/grids_html/Pelco-TEM-Grids.aspx).
- Electron-impact ionisation model: Bote and Salvat, *Atomic Data and Nuclear Data Tables* 95 (2009), [DOI 10.1016/j.adt.2009.08.001](https://doi.org/10.1016/j.adt.2009.08.001).
- Reference implementation/data origin: [NIST BoteSalvatICX.jl](https://github.com/usnistgov/BoteSalvatICX.jl), released as a United States Government public-domain work. The checked-in JSON is the machine-readable transcription published by [Temari](https://github.com/seto77/Temari).
- Atomic relaxation and attenuation data: [xraylib](https://github.com/tschoonj/xraylib).
- Complete-spectrum transport context and present accuracy limits: [NIST DTSA-II simulation paper](https://www.nist.gov/publications/simulating-electron-excited-energy-dispersive-x-ray-spectra-nist-dtsa-ii-open-source).
- Event-driven mean free path, exponential flights, screened-Rutherford formula and its heavy-element limitation: [Demers et al., *Microscopy and Microanalysis* 16 (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3165039/).
- Higher-accuracy elastic-scattering target: [NIST ELSEPA Dirac partial-wave publication](https://www.nist.gov/publications/elsepa-dirac-partial-wave-calculation-elastic-scattering-electrons-and-positrons-atoms) and [NIST SRD 64](https://www.nist.gov/srd/database-64-version-40).
