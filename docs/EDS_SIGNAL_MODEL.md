# Generic EDS signal model

The simulator names this subsystem **EDS**. The currently installed six-segment geometry is an evidence-bounded reference configuration, not the identity of the signal engine. Product names may remain in provenance URLs or historical research notes, but are not used as the system key, GUI title, spectrum name or result label.

## Implemented trajectory and signal phase

- `configs/specimen_supports/catalog.toml` defines a 3.05 mm circular support, a nominal 25 um foil, Cu, Au and virtual-vacuum materials, and commercial 50–500 square-mesh dimensions.
- `temsim.specimen.support` classifies a point as opening, bar, rim, outside or virtual vacuum. The grid offset is the centre of an opening and rotation is continuous.
- `temsim.detector.eds_atomic` evaluates the Bote–Salvat K, L1–L3 and M1–M5 electron-impact ionisation fits for Z=1–99.
- `temsim.detector.eds_signal` accepts weighted electron track segments in any material. A segment can explicitly identify a primary or elastically scattered history; the vacancy and photon calculation is identical.
- `temsim.specimen.elastic_transport` generates event-by-event three-dimensional elastic trajectories through the finite rectangular specimen and the downstream support. It resolves square-mesh openings and vertical sidewalls, crossed bars, the annular rim, the circular outside boundary and all upstream/downstream faces.
- The explicit point calculation takes its history count directly from the upstream column result: every weighted ray surviving to the physical sample plane is transported once. There is no independent EDS trajectory-count input. Sample-plane X/Y, both incident slopes (including calculated rotation), source energy offset and source-current weight are retained. The point coordinate translates only the weighted beam centroid; it does not replace the calculated phase-space spread.
- **Straight primary reference** remains available as a deterministic diagnostic. Retained histories and collision points are displayed in **Sample Interactions 3D**, while every reaching ray contributes to the EDS calculation. Its `3D / X-Z / Y-Z` controls render the same cached scene without rerunning transport. **EDS** contains only the spectrum, short status and hover energy/counts; it has no separate trajectory plots or line table.
- Direct-vacancy fluorescence yield, radiative transition probability, line energy, atomic weight, elemental density and photon mass attenuation come from xraylib 4.3.0.
- The installed angular acceptance supplies the collection fraction. The user can select the holder-conditioned or unshadowed aggregate solid angle.
- The detector response currently supports an explicit ideal scalar efficiency, optional Gaussian energy broadening and reproducible Poisson counting.
- **Update point EDS** and the existing EDS/support settings are available in **Sample Interactions 3D > Parameters**. They reuse compatible shared High-accuracy products. Merely changing the view or opening the settings does not calculate a spectrum or lens preset.
- **Run sample-region high accuracy** is a second, manual calculation. Its
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
- Representative characteristic photons are sampled uniformly on the sphere
  (`cos(theta)` uniform in `[-1,1]`, azimuth uniform in `[0,2 pi)`) and travel in
  straight lines. Public data do not provide a sensor face or distance, so the
  displayed endpoint is schematic. Detection uses an azimuth-partitioned
  elevation band whose aggregate spherical area exactly equals the selected
  holder-conditioned or unshadowed solid angle. This is explicitly an angular
  acceptance surrogate, not an invented detector intersection.

For element Z and path segment length ds, the mean shell-vacancy count is

`lambda_Zs = N_e w_track n_Z sigma_Zs(E) ds`,

with `n_Z = rho w_Z N_A / A_Z`. Expected line counts are

`N_line = lambda_Zs omega_s R_line T_self [Omega / (4 pi)] eta`.

The code retains both the shell optical depth and the expected vacancy count. It does not cap the expectation at one event per electron; a later event-sampling transport can draw discrete collisions from the same optical depth while updating energy between steps.

The current self-absorption term is the uniform-source-depth average within the emitting layer:

`T_self = [1 - exp(-mu rho t / sin(alpha))] / [mu rho t / sin(alpha)]`.

It does not yet trace each photon through arbitrary sample, support, holder and pole-piece solids.

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

Each simulated history carries its conditional sample-plane current weight. Source electrons are first multiplied by the calculated upstream survival fraction; the surviving histories are then normalized conditionally, so aperture/column losses are applied exactly once. Material paths retain the individual incident energy for the EDS vacancy calculation.

## Explicit current limits

This phase is an elastic electron-trajectory and characteristic-line engine, not yet a complete coupled electron/photon Monte Carlo.

- The point EDS calculation now defaults to the event-driven elastic Monte Carlo; a straight primary path remains an explicit selectable reference.
- The present screened-Rutherford model is stated for 100--300 keV and the cited work warns that it is imprecise for `Z > 30`. Results that encounter heavier elements carry a warning. Quantitative work should replace the provider with ELSEPA/Dirac partial-wave differential cross sections; the project does not redistribute the restricted NIST SRD 64 tables.
- The specimen kernel uses bulk density and elemental mass fractions (independent-atom, amorphous-style transport). It does not reproduce crystallographic channeling or coherent diffraction. A crystalline multislice wave has no unique classical path, and this EDS calculation is not derived from or added to the multislice intensity.
- The initial trajectory ensemble is the geometrical-ray phase space calculated at the physical sample plane. It includes source position/angle sampling, lens/deflector transport, aperture and wall survival, Larmor-rotated X/Y slopes, energy offsets and source weights. It is still a classical ray ensemble: coherent aberrated probe-wave intensity and crystallographic channeling are not converted into unique classical paths.
- The support is explicitly assumed to begin at the specimen downstream face. The crystal-orientation quaternion does not tilt the macroscopic rectangular foil envelope.
- Electron slowing, quantitative secondary-electron yield/energy transport, multiple ionisation event sampling, bremsstrahlung, vacancy cascades/Coster–Kronig transfer, secondary fluorescence, cross-layer absorption, holder/pole-piece photon shadowing and an energy-dependent measured SDD efficiency remain unimplemented. The Ray Diagram can show zero-weight, local, isotropic secondary candidates, but these qualitative markers never enter the downstream electron/current budget.
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
