"""Generic, detector-geometry-aware characteristic EDS signal model.

The signal layer converts electron track segments into shell vacancies,
characteristic photons and expected detector counts.  Track segments may be
caller supplied, the straight-primary reference, or Monte Carlo averages from
the finite-geometry elastic transport kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import xraylib

from temsim.detector.eds_atomic import (
    SUBSHELL_NAMES,
    bote_element_data,
    bote_ionisation_cross_section_cm2,
)
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_line_library import radiative_lines as _radiative_lines
from temsim.specimen.presets import (
    load_specimen_preset,
)
from temsim.specimen.scene import SpecimenScene
from temsim.specimen.source import (
    active_cif_path,
    selected_reference_preset_key,
    specimen_mode,
)


AVOGADRO_PER_MOL = 6.02214076e23
ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_G = 1.66053906660e-24


@dataclass(frozen=True, slots=True)
class EDSMaterial:
    key: str
    name: str
    density_g_cm3: float
    mass_fractions: tuple[tuple[int, float], ...]
    provenance: str

    def __post_init__(self) -> None:
        # Accept list-based inputs without retaining mutable composition in an
        # immutable material or making exact-value cache keys unhashable.
        object.__setattr__(
            self, "mass_fractions", tuple(tuple(row) for row in self.mass_fractions),
        )
        if (
            not math.isfinite(self.density_g_cm3)
            or self.density_g_cm3 <= 0.0
        ):
            raise ValueError("EDS material density must be finite and positive")
        if not self.mass_fractions:
            raise ValueError("EDS material needs at least one element")
        atomic_numbers = [item[0] for item in self.mass_fractions]
        fractions = [item[1] for item in self.mass_fractions]
        if (
            len(set(atomic_numbers)) != len(atomic_numbers)
            or any(not 1 <= value <= 99 for value in atomic_numbers)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in fractions
            )
            or not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-9)
        ):
            raise ValueError("EDS material mass fractions are invalid")

    def mass_fraction(self, atomic_number: int) -> float:
        return dict(self.mass_fractions).get(int(atomic_number), 0.0)

    def atom_number_density_cm3(self, atomic_number: int) -> float:
        z = int(atomic_number)
        fraction = self.mass_fraction(z)
        if fraction == 0.0:
            return 0.0
        return (
            self.density_g_cm3
            * fraction
            * AVOGADRO_PER_MOL
            / float(xraylib.AtomicWeight(z))
        )

    @lru_cache(maxsize=4096)
    def mass_attenuation_cm2_g(self, photon_energy_ev: float) -> float:
        # Immutable material + exact energy: reuse atomic data, not a rounded
        # energy or a path-dependent transmission. The cache is bounded.
        energy_kev = float(photon_energy_ev) * 1.0e-3
        if not math.isfinite(energy_kev) or energy_kev <= 0.0:
            raise ValueError("Photon energy must be finite and positive")
        result = 0.0
        for z, fraction in self.mass_fractions:
            try:
                coefficient = float(xraylib.CS_Total(z, energy_kev))
            except ValueError:
                # xraylib does not extrapolate below its attenuation table.
                # Treat those sub-100 eV photons as locally absorbed instead
                # of inventing an extrapolated transmission.
                return math.inf
            result += fraction * coefficient
        return result


@dataclass(frozen=True, slots=True)
class ElectronTrackSegment:
    """One material path traversed by a weighted electron population."""

    source_key: str
    material: EDSMaterial
    path_length_nm: float
    electron_energy_ev: float
    electron_weight: float = 1.0
    emitting_layer_thickness_nm: float | None = None
    history: str = "primary"
    source_ray_index: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.path_length_nm,
            self.electron_energy_ev,
            self.electron_weight,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("EDS electron-track values must be finite")
        if (
            self.path_length_nm < 0.0
            or self.electron_energy_ev < 0.0
            or self.electron_weight < 0.0
        ):
            raise ValueError("EDS electron-track values cannot be negative")
        if (
            self.emitting_layer_thickness_nm is not None
            and (
                not math.isfinite(self.emitting_layer_thickness_nm)
                or self.emitting_layer_thickness_nm < 0.0
            )
        ):
            raise ValueError(
                "EDS emitting-layer thickness must be non-negative"
            )
        if self.source_ray_index is not None and int(
            self.source_ray_index
        ) < 0:
            raise ValueError("EDS source-ray index cannot be negative")


@dataclass(frozen=True, slots=True)
class EDSVacancySignal:
    """One shell-vacancy contribution calculated from one electron track.

    This is the authoritative intermediate result for both radiative-line
    generation and the specimen event ledger.  It deliberately stores the
    already evaluated shell cross section so consumers never need a second
    ionisation pass.
    """

    vacancy_id: str
    track_index: int
    source_ray_index: int | None
    source_key: str
    material_key: str
    atomic_number: int
    subshell: str
    edge_energy_ev: float
    electron_history: str
    electron_energy_ev: float
    electron_weight: float
    path_length_nm: float
    ionisation_cross_section_cm2: float
    shell_optical_depth_per_electron: float
    expected_vacancies: float
    fluorescence_yield: float
    auger_yield: float
    unresolved_relaxation_yield: float

    def __post_init__(self) -> None:
        values = (
            self.edge_energy_ev,
            self.electron_energy_ev,
            self.electron_weight,
            self.path_length_nm,
            self.ionisation_cross_section_cm2,
            self.shell_optical_depth_per_electron,
            self.expected_vacancies,
            self.fluorescence_yield,
            self.auger_yield,
            self.unresolved_relaxation_yield,
        )
        if not str(self.vacancy_id).strip():
            raise ValueError("EDS vacancy ID cannot be empty")
        if int(self.track_index) < 0:
            raise ValueError("EDS vacancy track index cannot be negative")
        if self.source_ray_index is not None and int(
            self.source_ray_index
        ) < 0:
            raise ValueError("EDS vacancy source-ray index cannot be negative")
        if not 1 <= int(self.atomic_number) <= 99:
            raise ValueError("EDS vacancy atomic number must be Z=1..99")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("EDS vacancy values must be finite")
        if any(float(value) < 0.0 for value in values):
            raise ValueError("EDS vacancy values cannot be negative")
        if self.edge_energy_ev > self.electron_energy_ev:
            raise ValueError("EDS vacancy edge exceeds the electron energy")
        for name, value in (
            ("fluorescence", self.fluorescence_yield),
            ("Auger", self.auger_yield),
            ("unresolved relaxation", self.unresolved_relaxation_yield),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"EDS {name} yield must be in [0, 1]")
        if not math.isclose(
            self.relaxation_yield_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ValueError("EDS direct vacancy relaxation yields must sum to one")

    @property
    def expected_occurrences_per_incident_electron(self) -> float:
        """Vacancy rate per electron entering the sampled specimen plane."""

        return float(
            self.electron_weight * self.shell_optical_depth_per_electron
        )

    @property
    def relaxation_yield_sum(self) -> float:
        return float(
            self.fluorescence_yield
            + self.auger_yield
            + self.unresolved_relaxation_yield
        )


@dataclass(frozen=True, slots=True)
class EDSLineSignal:
    vacancy_id: str
    source_key: str
    material_key: str
    atomic_number: int
    subshell: str
    transition: str
    energy_ev: float
    ionisation_cross_section_cm2: float
    shell_optical_depth_per_electron: float
    expected_vacancies: float
    fluorescence_yield: float
    radiative_rate: float
    self_absorption_transmission: float
    expected_emitted_photons: float
    expected_detected_counts: float
    expected_counts_per_segment: tuple[float, ...]
    electron_history: str


@dataclass(frozen=True, slots=True)
class EDSMaterialQuadrature:
    """Auxiliary weighted material paths, never a downstream electron bundle.

    Independent quadrature IDs locate each emission flight. They are not
    identities of electrons traced through the column; their kernel parents
    are retained separately for provenance. Weights refer to the full current
    reaching the specimen plane, including the unsampled vacuum complement.
    """

    eds_tracks: tuple[ElectronTrackSegment, ...]
    material_flights: tuple[object, ...]
    quadrature_source_ray_indices: tuple[int, ...]
    parent_source_ray_indices: tuple[int, ...]
    metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class EDSSpectrum:
    energy_bin_centres_ev: np.ndarray
    expected_counts: np.ndarray
    sampled_counts: np.ndarray | None
    lines: tuple[EDSLineSignal, ...]
    metrics: dict[str, object]
    vacancies: tuple[EDSVacancySignal, ...] = ()
    elastic_transport: object | None = None
    photon_transport: object | None = None
    material_quadrature: EDSMaterialQuadrature | None = None

    def __post_init__(self) -> None:
        lines = tuple(self.lines)
        vacancies = tuple(self.vacancies)
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "vacancies", vacancies)
        vacancy_ids = [row.vacancy_id for row in vacancies]
        if len(vacancy_ids) != len(set(vacancy_ids)):
            raise ValueError("EDS vacancy IDs must be unique")
        missing = {
            line.vacancy_id for line in lines
        } - set(vacancy_ids)
        if missing:
            raise ValueError(
                "Every EDS line must reference its calculated vacancy"
            )

    @property
    def total_expected_counts(self) -> float:
        return float(sum(line.expected_detected_counts for line in self.lines))


def elemental_material(
    atomic_number: int,
    *,
    key: str | None = None,
    name: str | None = None,
    density_g_cm3: float | None = None,
    provenance: str = "xraylib elemental atomic weight and bulk density",
) -> EDSMaterial:
    z = int(atomic_number)
    if not 1 <= z <= 99:
        raise ValueError("EDS elemental materials require Z=1..99")
    return EDSMaterial(
        key=str(key or f"Z{z}"),
        name=str(name or xraylib.AtomicNumberToSymbol(z)),
        density_g_cm3=float(
            xraylib.ElementDensity(z)
            if density_g_cm3 is None
            else density_g_cm3
        ),
        mass_fractions=((z, 1.0),),
        provenance=provenance,
    )


def _normalised_mass_fractions(
    atomic_numbers: Iterable[int],
    atomic_masses: Iterable[float],
) -> tuple[tuple[int, float], ...]:
    totals: dict[int, float] = {}
    for atomic_number, mass in zip(
        atomic_numbers, atomic_masses, strict=True
    ):
        z = int(atomic_number)
        totals[z] = totals.get(z, 0.0) + float(mass)
    total_mass = sum(totals.values())
    if total_mass <= 0.0:
        raise ValueError("Atomic structure has no positive mass")
    return tuple(
        (z, mass / total_mass) for z, mass in sorted(totals.items())
    )


def material_from_cif(cif_path: str | Path) -> EDSMaterial:
    from temsim.specimen.rutherford import read_cif_composition

    composition = read_cif_composition(cif_path)
    path = Path(composition.source_path)
    if any(z > 99 for z, _fraction in composition.mass_fractions):
        raise ValueError(
            "The Bote-Salvat EDS model currently supports CIF elements Z<=99"
        )
    return EDSMaterial(
        key=f"cif:{path.name}",
        name=path.stem,
        density_g_cm3=composition.density_g_cm3,
        mass_fractions=composition.mass_fractions,
        provenance=composition.provenance,
    )


def material_from_sample(state) -> EDSMaterial | None:
    sample = state.sample
    if not bool(getattr(sample, "inserted", True)):
        return None
    if specimen_mode(sample) in {"atomic", "reference"}:
        cif_path = active_cif_path(sample)
        return material_from_cif(cif_path) if cif_path else None
    preset_key = selected_reference_preset_key(sample)
    preset = load_specimen_preset(preset_key)
    if preset.atomistic is not None:
        z = preset.atomistic.atomic_number
    else:
        atomic_numbers = {column.atomic_number for column in preset.columns}
        if not atomic_numbers:
            return None
        if len(atomic_numbers) != 1:
            raise ValueError(
                f"{preset.name} needs CIF or explicit EDS density/composition"
            )
        z = atomic_numbers.pop()
    return elemental_material(
        z,
        key=f"specimen:{preset.key}",
        name=preset.name,
        provenance=(
            f"{preset.source_path.name} elemental identity; "
            "xraylib atomic weight and bulk elemental density"
        ),
    )


def material_from_support_grid(grid) -> EDSMaterial | None:
    material = grid.material
    if material.is_vacuum:
        return None
    return elemental_material(
        material.atomic_number,
        key=f"support:{material.key}",
        name=material.name,
        density_g_cm3=material.density_g_cm3,
        provenance=(
            f"{grid.geometry_source_url}; support catalog "
            f"density status {material.status}"
        ),
    )


def _mean_self_absorption_transmission(
    material: EDSMaterial,
    photon_energy_ev: float,
    layer_thickness_nm: float,
    takeoff_angle_deg: float,
) -> float:
    thickness = float(layer_thickness_nm)
    if thickness <= 0.0:
        return 1.0
    sine = math.sin(math.radians(float(takeoff_angle_deg)))
    if sine <= 0.0:
        raise ValueError("EDS take-off angle must be positive")
    mass_attenuation = material.mass_attenuation_cm2_g(photon_energy_ev)
    full_path_cm = thickness * 1.0e-7 / sine
    optical_depth = (
        material.density_g_cm3 * mass_attenuation * full_path_cm
    )
    if optical_depth <= 1.0e-10:
        return 1.0 - 0.5 * optical_depth
    return -math.expm1(-optical_depth) / optical_depth


@lru_cache(maxsize=1024)
def _relaxation_yields(z: int, subshell_index: int) -> tuple[float, float, float]:
    """Reuse immutable atomic yields without changing vacancy arithmetic."""
    try:
        fluorescence_yield = float(xraylib.FluorYield(z, subshell_index))
    except ValueError:
        fluorescence_yield = 0.0
    try:
        auger_yield = float(xraylib.AugerYield(z, subshell_index))
    except ValueError:
        auger_yield = 0.0
    direct_yield_sum = fluorescence_yield + auger_yield
    if direct_yield_sum > 1.0:
        if direct_yield_sum > 1.0 + 2.0e-12:
            raise ValueError("xraylib vacancy-relaxation yields exceed one")
        fluorescence_yield /= direct_yield_sum
        auger_yield /= direct_yield_sum
    return (
        fluorescence_yield,
        auger_yield,
        max(1.0 - fluorescence_yield - auger_yield, 0.0),
    )


def _report_phase_progress(callback, start, end, completed, total, label):
    """Compose stage fractions; labels retain the actual physical work count."""
    if callback is None:
        return
    fraction = min(max(float(completed) / max(int(total), 1), 0.0), 1.0)
    callback(round(10_000 * (start + (end - start) * fraction)), 10_000, label)


def _line_response_kernel(centres, energy_ev, sigma_ev):
    """Reference Gaussian weights; callers may cache by exact line energy."""
    lower = max(int(np.searchsorted(centres, energy_ev - 5.0 * sigma_ev)), 0)
    upper = min(
        int(np.searchsorted(centres, energy_ev + 5.0 * sigma_ev)) + 1,
        centres.size,
    )
    weights = np.exp(-0.5 * ((centres[lower:upper] - energy_ev) / sigma_ev) ** 2)
    return lower, upper, weights, float(np.sum(weights))


def _empty_spectrum(
    *,
    energy_min_ev: float,
    energy_max_ev: float,
    energy_bin_width_ev: float,
    segment_count: int,
    metrics: dict[str, object],
) -> EDSSpectrum:
    edges = np.arange(
        energy_min_ev,
        energy_max_ev + energy_bin_width_ev,
        energy_bin_width_ev,
        dtype=float,
    )
    if edges.size < 2:
        edges = np.asarray((energy_min_ev, energy_max_ev), dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    expected = np.zeros_like(centres)
    centres.setflags(write=False)
    expected.setflags(write=False)
    metrics.update(
        {
            "detector_segment_count": int(segment_count),
            "total_expected_counts": 0.0,
            "total_expected_emitted_photons": 0.0,
            "line_count": 0,
        }
    )
    metrics.update(
        {
            "vacancy_contribution_count": 0,
            "total_expected_vacancies": 0.0,
            "total_expected_radiative_relaxations": 0.0,
            "total_expected_auger_relaxations": 0.0,
            "total_expected_unresolved_relaxations": 0.0,
            "shell_cross_section_evaluation_count": 0,
            "duplicate_shell_ionisation_passes": 0,
            # Efficiency application counts are configured before this helper
            # is called and retained in ``metrics`` for an empty result.
        }
    )
    return EDSSpectrum(centres, expected, None, (), metrics)


def _stable_unit_fraction(identifier: str) -> float:
    digest = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / float(1 << 64)


def _vacancy_emission_origin_mm(
    state,
    vacancy: EDSVacancySignal,
    elastic_transport,
    *,
    fallback_xy_nm: tuple[float, float],
    material_flight_index: dict[
        tuple[int, str, str, str],
        tuple[tuple[np.ndarray, np.ndarray, float], ...],
    ]
    | None = None,
) -> tuple[tuple[float, float, float], str]:
    """Resolve one reproducible emission point from the owning material flight."""

    matches: tuple[tuple[np.ndarray, np.ndarray, float], ...] = ()
    if vacancy.source_ray_index is not None and material_flight_index is not None:
        matches = material_flight_index.get(
            (
                int(vacancy.source_ray_index),
                str(vacancy.source_key),
                str(vacancy.material_key),
                str(vacancy.electron_history),
            ),
            (),
        )
    elif vacancy.source_ray_index is not None:
        rows = []
        for flight in tuple(
            getattr(elastic_transport, "material_flights", ())
        ):
            if (
                int(flight.source_ray_index) != int(vacancy.source_ray_index)
                or str(flight.source_key) != str(vacancy.source_key)
                or str(flight.material_key) != str(vacancy.material_key)
                or str(flight.history) != str(vacancy.electron_history)
            ):
                continue
            start = np.asarray(flight.start_nm, dtype=float)
            end = np.asarray(flight.end_nm, dtype=float)
            length = float(np.linalg.norm(end - start))
            if length > 0.0:
                rows.append((start, end, length))
        matches = tuple(rows)
    total_length = float(sum(row[2] for row in matches))
    if total_length > 0.0:
        target = _stable_unit_fraction(vacancy.vacancy_id) * total_length
        traversed = 0.0
        local_nm = matches[-1][1]
        for start, end, length in matches:
            if target <= traversed + length:
                fraction = (target - traversed) / length
                local_nm = start + fraction * (end - start)
                break
            traversed += length
        return (
            (
                float(local_nm[0]) * 1.0e-6,
                float(local_nm[1]) * 1.0e-6,
                float(state.sample.z_mm) + float(local_nm[2]) * 1.0e-6,
            ),
            ("weighted_overlap_material_flight"
             if isinstance(elastic_transport, EDSMaterialQuadrature)
             else "stored_elastic_material_flight"),
        )

    scene = SpecimenScene.from_state(state, include_eds_materials=False)
    x_nm, y_nm = (float(value) for value in fallback_xy_nm)
    if str(vacancy.source_key).startswith("support:"):
        local_z_nm = 0.5 * (scene.support_top_nm + scene.support_bottom_nm)
        status = "support_layer_centre_no_stored_flight"
    else:
        local_z_nm = 0.0
        status = "specimen_reference_plane_no_stored_flight"
    return (
        (
            x_nm * 1.0e-6,
            y_nm * 1.0e-6,
            scene.reference_z_mm + local_z_nm * 1.0e-6,
        ),
        status,
    )


def simulate_eds_tracks(
    tracks: Iterable[ElectronTrackSegment],
    detector_geometry: EDSDetectorArrayGeometry,
    *,
    incident_electrons: float,
    use_analytical_holder_solid_angle: bool = True,
    detector_efficiency: float = 1.0,
    energy_min_ev: float = 0.0,
    energy_max_ev: float = 40_000.0,
    energy_bin_width_ev: float = 10.0,
    energy_resolution_fwhm_ev: float = 0.0,
    poisson_enabled: bool = False,
    poisson_seed: int = 0,
    state=None,
    elastic_transport=None,
    photon_origin_xy_nm: tuple[float, float] = (0.0, 0.0),
    photon_detector_surfaces: Iterable[object] = (),
    photon_holder_occluders: Iterable[object] = (),
    photon_quadrature_order: int = 1,
    photon_maximum_stored_paths: int = 1024,
    photon_include_specimen: bool = True,
    photon_include_support: bool = True,
    progress_callback=None,
) -> EDSSpectrum:
    """Convert material track segments to one expected characteristic spectrum."""

    track_rows = tuple(tracks)
    incident = float(incident_electrons)
    efficiency = float(detector_efficiency)
    spectrum_values = (
        float(energy_min_ev),
        float(energy_max_ev),
        float(energy_bin_width_ev),
        float(energy_resolution_fwhm_ev),
    )
    if not math.isfinite(incident) or incident < 0.0:
        raise ValueError("Incident electron count must be non-negative")
    if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
        raise ValueError("EDS detector efficiency must be in [0, 1]")
    if (
        not all(math.isfinite(value) for value in spectrum_values)
        or spectrum_values[0] < 0.0
        or spectrum_values[1] <= spectrum_values[0]
        or spectrum_values[2] <= 0.0
        or spectrum_values[3] < 0.0
    ):
        raise ValueError("EDS spectrum axis/resolution is invalid")
    if int(poisson_seed) < 0:
        raise ValueError("EDS Poisson seed cannot be negative")
    detector_geometry.validate()
    _report_phase_progress(
        progress_callback, 0.0, 0.4, 0, len(track_rows),
        f"EDS ionisation tracks 0/{len(track_rows)}",
    )
    detector_surfaces = tuple(photon_detector_surfaces)
    holder_occluders = tuple(photon_holder_occluders)
    quadrature_order = int(photon_quadrature_order)
    maximum_stored_paths = int(photon_maximum_stored_paths)
    if quadrature_order <= 0:
        raise ValueError("EDS photon quadrature order must be positive")
    if maximum_stored_paths < 0:
        raise ValueError("EDS photon stored-path limit cannot be negative")
    solid_angle = (
        detector_geometry.analytical_holder_solid_angle_sr
        if use_analytical_holder_solid_angle
        else detector_geometry.minimum_unshadowed_solid_angle_sr
    )
    geometric_fraction = solid_angle / (4.0 * math.pi)
    metrics: dict[str, object] = {
        "system_name": "EDS",
        "signal_kind": "characteristic_x_ray",
        "incident_electrons": incident,
        "solid_angle_sr": solid_angle,
        "solid_angle_mode": (
            "installed_analytical_holder"
            if use_analytical_holder_solid_angle
            else "unshadowed_reference"
        ),
        "geometric_collection_fraction": geometric_fraction,
        "detector_efficiency": efficiency,
        "global_detector_quantum_efficiency": efficiency,
        "detector_efficiency_model": (
            "global_absolute_qe_x_surface_relative_response"
            if detector_surfaces
            else "global_absolute_qe"
        ),
        "surface_relative_response_applied": bool(detector_surfaces),
        "surface_relative_response_source": (
            "PlanarEDSDetectorSegment.efficiency"
            if detector_surfaces
            else None
        ),
        "global_detector_efficiency_application_count": 1,
        "surface_relative_efficiency_application_count": (
            1 if detector_surfaces else 0
        ),
        "detector_efficiency_application_count": (
            1 + (1 if detector_surfaces else 0)
        ),
        "detector_efficiency_application_semantics": (
            "one global absolute QE factor plus one sourced-face relative "
            "response factor"
            if detector_surfaces
            else "one global absolute QE factor"
        ),
        "ionisation_model": "Bote-Salvat K/L/M electron impact",
        "relaxation_database": f"xraylib {xraylib.__version__}",
        "vacancy_cascade_model": (
            "direct fluorescence/Auger yields with explicit unresolved "
            "shell-transfer remainder; secondary vacancy cascades omitted"
        ),
        "auger_energy_model": (
            "yield resolved; Auger kinetic energy and direction not assigned"
        ),
        "coster_kronig_model": (
            "not propagated; retained in unresolved relaxation remainder"
        ),
        "self_absorption_model": "uniform-depth emitting-layer average",
        "cross_layer_absorption": False,
        "electron_transport_model": "caller_supplied_weighted_track_segments",
        "elastic_trajectory_generation": False,
        "bremsstrahlung_included": False,
        "secondary_fluorescence_included": False,
        "poisson_noise_enabled": bool(poisson_enabled),
        "poisson_seed": int(poisson_seed) if poisson_enabled else None,
        "poisson_sampling_stage": "after_final_expected_spectrum",
        "track_segment_count": len(track_rows),
        "photon_transport_applied_to_main_spectrum": state is not None,
    }
    if incident == 0.0 or not track_rows:
        _report_phase_progress(
            progress_callback, 0.0, 1.0, 1, 1, "EDS spectrum: no incident tracks",
        )
        return _empty_spectrum(
            energy_min_ev=spectrum_values[0],
            energy_max_ev=spectrum_values[1],
            energy_bin_width_ev=spectrum_values[2],
            segment_count=detector_geometry.segment_count,
            metrics=metrics,
        )

    vacancies: list[EDSVacancySignal] = []
    lines: list[EDSLineSignal] = []
    maximum_shell_optical_depth = 0.0
    shell_cross_section_evaluation_count = 0
    for track_index, track in enumerate(track_rows):
        if track_index % max(1, len(track_rows) // 100) == 0:
            _report_phase_progress(
                progress_callback, 0.0, 0.4, track_index, len(track_rows),
                f"EDS ionisation tracks {track_index}/{len(track_rows)}",
            )
        if track.path_length_nm == 0.0 or track.electron_weight == 0.0:
            continue
        path_cm = track.path_length_nm * 1.0e-7
        layer_thickness_nm = (
            track.path_length_nm
            if track.emitting_layer_thickness_nm is None
            else track.emitting_layer_thickness_nm
        )
        for z, _mass_fraction in track.material.mass_fractions:
            number_density = track.material.atom_number_density_cm3(z)
            element_data = bote_element_data(z)
            for subshell_index, subshell in enumerate(
                element_data.subshell_names
            ):
                cross_section = bote_ionisation_cross_section_cm2(
                    z, subshell_index, track.electron_energy_ev
                )
                shell_cross_section_evaluation_count += 1
                shell_optical_depth = (
                    number_density * cross_section * path_cm
                )
                maximum_shell_optical_depth = max(
                    maximum_shell_optical_depth, shell_optical_depth
                )
                if shell_optical_depth <= 0.0:
                    continue
                expected_vacancies = (
                    incident
                    * track.electron_weight
                    * shell_optical_depth
                )
                (
                    fluorescence_yield, auger_yield, unresolved_relaxation_yield,
                ) = _relaxation_yields(z, subshell_index)
                ray_label = (
                    track.source_ray_index
                    if track.source_ray_index is not None
                    else "none"
                )
                vacancy = EDSVacancySignal(
                    vacancy_id=(
                        f"vacancy:track{track_index}:"
                        f"ray{ray_label}:"
                        f"Z{z}:{subshell}"
                    ),
                    track_index=track_index,
                    source_ray_index=track.source_ray_index,
                    source_key=track.source_key,
                    material_key=track.material.key,
                    atomic_number=z,
                    subshell=subshell,
                    edge_energy_ev=float(
                        element_data.edge_ev[subshell_index]
                    ),
                    electron_history=track.history,
                    electron_energy_ev=track.electron_energy_ev,
                    electron_weight=track.electron_weight,
                    path_length_nm=track.path_length_nm,
                    ionisation_cross_section_cm2=cross_section,
                    shell_optical_depth_per_electron=shell_optical_depth,
                    expected_vacancies=expected_vacancies,
                    fluorescence_yield=fluorescence_yield,
                    auger_yield=auger_yield,
                    unresolved_relaxation_yield=(
                        unresolved_relaxation_yield
                    ),
                )
                vacancies.append(vacancy)
                if vacancy.fluorescence_yield <= 0.0:
                    continue
                for transition, line_energy_ev, radiative_rate in (
                    _radiative_lines(z, subshell_index)
                ):
                    emitted = (
                        expected_vacancies
                        * vacancy.fluorescence_yield
                        * radiative_rate
                    )
                    transmission = _mean_self_absorption_transmission(
                        track.material,
                        line_energy_ev,
                        layer_thickness_nm,
                        detector_geometry.takeoff_angle_deg,
                    )
                    detected = (
                        emitted
                        * transmission
                        * geometric_fraction
                        * efficiency
                    )
                    lines.append(
                        EDSLineSignal(
                            vacancy_id=vacancy.vacancy_id,
                            source_key=track.source_key,
                            material_key=track.material.key,
                            atomic_number=z,
                            subshell=subshell,
                            transition=transition,
                            energy_ev=line_energy_ev,
                            ionisation_cross_section_cm2=cross_section,
                            shell_optical_depth_per_electron=(
                                shell_optical_depth
                            ),
                            expected_vacancies=expected_vacancies,
                            fluorescence_yield=vacancy.fluorescence_yield,
                            radiative_rate=radiative_rate,
                            self_absorption_transmission=transmission,
                            expected_emitted_photons=emitted,
                            expected_detected_counts=detected,
                            expected_counts_per_segment=tuple(
                                detected
                                / detector_geometry.segment_count
                                for _ in range(
                                    detector_geometry.segment_count
                                )
                            ),
                            electron_history=track.history,
                        )
                    )

    _report_phase_progress(
        progress_callback, 0.0, 0.4, len(track_rows), len(track_rows),
        f"EDS ionisation tracks {len(track_rows)}/{len(track_rows)}",
    )
    photon_transport = None
    if state is not None and lines:
        from temsim.detector.eds_photon_transport import (
            detector_quadrature_photons,
            transport_eds_photons,
        )

        vacancy_by_id = {
            vacancy.vacancy_id: vacancy for vacancy in vacancies
        }
        emission_keys = tuple(
            f"{line.vacancy_id}:line{line_index}:{line.transition}"
            for line_index, line in enumerate(lines)
        )
        origin_status_counts: dict[str, int] = {}
        material_flight_rows: dict[
            tuple[int, str, str, str],
            list[tuple[np.ndarray, np.ndarray, float]],
        ] = {}
        for flight in tuple(
            getattr(elastic_transport, "material_flights", ())
        ):
            start = np.asarray(flight.start_nm, dtype=float)
            end = np.asarray(flight.end_nm, dtype=float)
            length = float(np.linalg.norm(end - start))
            if length <= 0.0:
                continue
            flight_key = (
                int(flight.source_ray_index),
                str(flight.source_key),
                str(flight.material_key),
                str(flight.history),
            )
            material_flight_rows.setdefault(flight_key, []).append(
                (start, end, length)
            )
        material_flight_index = {
            key: tuple(values)
            for key, values in material_flight_rows.items()
        }
        origin_by_vacancy: dict[
            str, tuple[tuple[float, float, float], str]
        ] = {}

        def photon_rows():
            _report_phase_progress(
                progress_callback, 0.4, 0.9, 0, len(lines),
                f"EDS photon emissions 0/{len(lines)}",
            )
            for line_index, (line, emission_key) in enumerate(zip(
                lines, emission_keys, strict=True
            )):
                vacancy = vacancy_by_id[line.vacancy_id]
                if line.vacancy_id not in origin_by_vacancy:
                    origin_by_vacancy[line.vacancy_id] = (
                        _vacancy_emission_origin_mm(
                            state,
                            vacancy,
                            elastic_transport,
                            fallback_xy_nm=photon_origin_xy_nm,
                            material_flight_index=material_flight_index,
                        )
                    )
                    origin_status = origin_by_vacancy[line.vacancy_id][1]
                    origin_status_counts[origin_status] = (
                        origin_status_counts.get(origin_status, 0) + 1
                    )
                origin, _origin_status = origin_by_vacancy[
                    line.vacancy_id
                ]
                yield from detector_quadrature_photons(
                    emission_key=emission_key,
                    origin_mm=origin,
                    energy_ev=line.energy_ev,
                    expected_emitted_photons=(
                        line.expected_emitted_photons
                    ),
                    geometry=detector_geometry,
                    detector_surfaces=detector_surfaces,
                    use_analytical_holder_solid_angle=(
                        use_analytical_holder_solid_angle
                    ),
                    detector_efficiency=efficiency,
                    quadrature_order=quadrature_order,
                    source_key=line.source_key,
                    transition=line.transition,
                )
                completed_lines = line_index + 1
                if (
                    completed_lines % max(1, len(lines) // 100) == 0
                    or completed_lines == len(lines)
                ):
                    # Resuming after yield means all photons from this line
                    # have been transported, not merely generated.
                    _report_phase_progress(
                        progress_callback, 0.4, 0.9, completed_lines, len(lines),
                        f"EDS photon emissions {completed_lines}/{len(lines)}",
                    )

        photon_transport = transport_eds_photons(
            state,
            photon_rows(),
            detector_geometry,
            detector_surfaces=detector_surfaces,
            holder_occluders=holder_occluders,
            include_specimen=bool(photon_include_specimen),
            include_support=bool(photon_include_support),
            use_analytical_holder_solid_angle=(
                use_analytical_holder_solid_angle
            ),
            # The deterministic ray weights already contain the scalar
            # efficiency so exact-face and aggregate branches use it once.
            aggregate_detector_efficiency=1.0,
            maximum_stored_paths=maximum_stored_paths,
        )
        transported_lines = []
        for line, emission_key in zip(lines, emission_keys, strict=True):
            per_segment = tuple(
                photon_transport.expected_detected_weight_per_emission.get(
                    emission_key,
                    (0.0,) * detector_geometry.segment_count,
                )
            )
            transported_lines.append(
                replace(
                    line,
                    expected_detected_counts=float(sum(per_segment)),
                    expected_counts_per_segment=per_segment,
                )
            )
        lines = transported_lines
        metrics.update(
            {
                "photon_transport_model": photon_transport.metrics["model"],
                "photon_transport_mode": (
                    "sourced_detector_faces"
                    if detector_surfaces
                    else "aggregate_angular_fallback"
                ),
                "photon_transport_quadrature_order": quadrature_order,
                "photon_transport_geometry_complete": (
                    photon_transport.geometry_complete
                ),
                "photon_transport_origin_status_counts": dict(
                    sorted(origin_status_counts.items())
                ),
                "photon_transport_expected_unattenuated_counts": float(
                    sum(
                        photon_transport.quadrature_weight_per_emission.values()
                    )
                ),
                "photon_transport_expected_detected_counts": float(
                    photon_transport.metrics["total_detected_weight"]
                ),
                "photon_transport_blocked_ray_count": int(
                    photon_transport.metrics["blocked_photon_count"]
                ),
                "photon_transport_blocked_by_component_counts": dict(
                    photon_transport.metrics["blocked_by_component_counts"]
                ),
                "photon_transport_outside_acceptance_count": int(
                    photon_transport.metrics["outside_acceptance_count"]
                ),
                "photon_transport_stored_path_count": int(
                    photon_transport.metrics["stored_path_count"]
                ),
                "self_absorption_model": (
                    "finite specimen ray intervals from shared line-emission "
                    "origins; line field retains legacy uniform-depth reference"
                ),
                "cross_layer_absorption": bool(photon_include_support),
            }
        )

    edges = np.arange(
        spectrum_values[0],
        spectrum_values[1] + spectrum_values[2],
        spectrum_values[2],
        dtype=float,
    )
    if edges[-1] < spectrum_values[1]:
        edges = np.r_[edges, spectrum_values[1]]
    centres = 0.5 * (edges[:-1] + edges[1:])
    expected = np.zeros_like(centres)
    outside_counts = 0.0
    sigma_ev = spectrum_values[3] / (
        2.0 * math.sqrt(2.0 * math.log(2.0))
    )
    response_kernels = {}
    for line_index, line in enumerate(lines):
        if line_index % max(1, len(lines) // 100) == 0:
            _report_phase_progress(
                progress_callback, 0.9, 1.0, line_index, len(lines),
                f"EDS spectrum lines {line_index}/{len(lines)}",
            )
        if not spectrum_values[0] <= line.energy_ev < spectrum_values[1]:
            outside_counts += line.expected_detected_counts
            continue
        if sigma_ev <= 0.0:
            index = int(np.searchsorted(edges, line.energy_ev) - 1)
            index = min(max(index, 0), expected.size - 1)
            expected[index] += line.expected_detected_counts
            continue
        if line.energy_ev not in response_kernels:
            response_kernels[line.energy_ev] = _line_response_kernel(
                centres, line.energy_ev, sigma_ev,
            )
        lower, upper, weights, total_weight = response_kernels[line.energy_ev]
        if total_weight > 0.0:
            expected[lower:upper] += (
                line.expected_detected_counts * weights / total_weight
            )
    sampled = None
    if poisson_enabled:
        sampled = np.random.default_rng(int(poisson_seed)).poisson(
            np.maximum(expected, 0.0)
        )
        sampled.setflags(write=False)
    centres.setflags(write=False)
    expected.setflags(write=False)
    lines.sort(
        key=lambda row: (
            row.energy_ev,
            row.source_key,
            row.atomic_number,
            row.transition,
        )
    )
    vacancies.sort(
        key=lambda row: (
            row.track_index,
            row.atomic_number,
            row.subshell,
        )
    )
    metrics.update(
        {
            "detector_segment_count": detector_geometry.segment_count,
            "line_count": len(lines),
            "total_expected_counts": float(
                sum(line.expected_detected_counts for line in lines)
            ),
            "total_expected_emitted_photons": float(
                sum(line.expected_emitted_photons for line in lines)
            ),
            "spectrum_expected_counts": float(np.sum(expected)),
            "counts_outside_spectrum": outside_counts,
            "maximum_shell_optical_depth_per_electron": (
                maximum_shell_optical_depth
            ),
            "constant_energy_thin_track_assumption_credible": (
                maximum_shell_optical_depth <= 0.1
            ),
            "energy_resolution_fwhm_ev": spectrum_values[3],
            "energy_bin_width_ev": spectrum_values[2],
            "vacancy_contribution_count": len(vacancies),
            "total_expected_vacancies": float(
                sum(row.expected_vacancies for row in vacancies)
            ),
            "total_expected_radiative_relaxations": float(
                sum(
                    row.expected_vacancies * row.fluorescence_yield
                    for row in vacancies
                )
            ),
            "total_expected_auger_relaxations": float(
                sum(
                    row.expected_vacancies * row.auger_yield
                    for row in vacancies
                )
            ),
            "total_expected_unresolved_relaxations": float(
                sum(
                    row.expected_vacancies
                    * row.unresolved_relaxation_yield
                    for row in vacancies
                )
            ),
            "shell_cross_section_evaluation_count": (
                shell_cross_section_evaluation_count
            ),
            "shell_cross_section_reuse": (
                "single vacancy pass feeds radiative lines and event ledger"
            ),
            "duplicate_shell_ionisation_passes": 0,
            "photon_statistical_weight_conservation": (
                "detector quadrature weights are derived once from each "
                "line emission; attenuation/shadowing can only remove weight"
            ),
        }
    )
    _report_phase_progress(
        progress_callback, 0.9, 1.0, 1, 1, "EDS spectrum complete",
    )
    return EDSSpectrum(
        energy_bin_centres_ev=centres,
        expected_counts=expected,
        sampled_counts=sampled,
        lines=tuple(lines),
        metrics=metrics,
        vacancies=tuple(vacancies),
        elastic_transport=elastic_transport,
        photon_transport=photon_transport,
    )


def point_track_segments(
    state,
    *,
    x_nm: float,
    y_nm: float,
) -> tuple[ElectronTrackSegment, ...]:
    """Build the initial straight primary path through sample and support."""

    energy_ev = float(state.beam_voltage_kv) * 1000.0
    scene = SpecimenScene.from_state(state, include_eds_materials=True)
    rows: list[ElectronTrackSegment] = []
    for region in scene.axial_material_regions(x_nm, y_nm):
        rows.append(
            ElectronTrackSegment(
                source_key=region.source_key,
                material=region.material,
                path_length_nm=region.path_length_nm,
                electron_energy_ev=energy_ev,
                emitting_layer_thickness_nm=(
                    region.emitting_layer_thickness_nm
                ),
                history=(
                    "straight_primary"
                    if region.source_key == "sample"
                    else "straight_primary_after_sample"
                ),
            )
        )
    return tuple(rows)


def default_eds_dwell_time_s(state) -> float:
    scan = state.ac_deflector
    pixel_count = max(
        int(scan.scan_pixels_x) * int(scan.scan_lines), 1
    )
    return float(scan.scan_frame_period_s) / pixel_count


def default_eds_incident_electrons(state, dwell_time_s: float) -> float:
    from temsim.physics.beam_current import effective_source_current_pa

    current_pa = effective_source_current_pa(state)
    return current_pa * 1.0e-12 * float(dwell_time_s) / ELEMENTARY_CHARGE_C


def simulate_eds_point(
    state,
    detector_geometry: EDSDetectorArrayGeometry,
    *,
    simulation=None,
    x_nm: float | None = None,
    y_nm: float | None = None,
    dwell_time_s: float | None = None,
    incident_electrons: float | None = None,
    elastic_transport=None,
    incident_bundle=None,
    progress_callback=None,
    photon_detector_surfaces: Iterable[object] = (),
    photon_holder_occluders: Iterable[object] = (),
    photon_quadrature_order: int = 1,
    photon_maximum_stored_paths: int = 1024,
) -> EDSSpectrum:
    """Run an explicit point acquisition at the sample scan origin."""

    sample = state.sample
    x_value = (
        float(sample.scan_origin_x_nm) if x_nm is None else float(x_nm)
    )
    y_value = (
        float(sample.scan_origin_y_nm) if y_nm is None else float(y_nm)
    )
    dwell = (
        default_eds_dwell_time_s(state)
        if dwell_time_s is None
        else float(dwell_time_s)
    )
    if not math.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("EDS dwell time must be finite and positive")
    source_electron_count = (
        default_eds_incident_electrons(state, dwell)
        if incident_electrons is None
        else float(incident_electrons)
    )
    transport_mode = str(
        getattr(sample, "eds_transport_mode", "elastic_monte_carlo")
    ).strip().lower()
    material_quadrature = None
    overlap_metrics = {}
    if transport_mode == "elastic_monte_carlo":
        # Local import keeps the EDS track contract independent and avoids a
        # module-load cycle: elastic_transport consumes EDSMaterial and
        # ElectronTrackSegment, while only this explicit acquisition invokes
        # trajectory generation.
        from temsim.specimen.elastic_transport import (
            incident_rays_from_simulation,
            simulate_elastic_point_transport,
        )

        if incident_bundle is None:
            incident_bundle = incident_rays_from_simulation(
                state,
                simulation,
                target_x_nm=x_value,
                target_y_nm=y_value,
            )
        if elastic_transport is None:
            elastic_transport = simulate_elastic_point_transport(
                state,
                incident_rays=incident_bundle.rays,
                progress_callback=(
                    None if progress_callback is None else
                    lambda done, total, label: _report_phase_progress(
                        progress_callback, 0.0, 0.25, done, total, label,
                    )
                ),
            )
        tracks = elastic_transport.eds_tracks
        from temsim.specimen.overlap_sampling import build_overlap_sampling_plan

        plan = build_overlap_sampling_plan(state, incident_bundle, elastic_transport)
        overlap_metrics.update(plan.metrics)
        if plan.rays:
            # The elastic kernel uses conditional unit-sum weights internally.
            # Restore their *absolute* mass on every EDS track and flight before
            # ionisation. Source current and column survival are applied below
            # exactly as for the original histories, with no second overlap
            # multiplier. Retain all auxiliary flights for correct photon origins.
            mass = math.fsum(ray.weight for ray in plan.rays)
            auxiliary = simulate_elastic_point_transport(
                state, incident_rays=plan.rays,
                stored_trajectory_count=len(plan.rays),
                progress_callback=(
                    None if progress_callback is None else
                    lambda done, total, label: _report_phase_progress(
                        progress_callback, 0.25, 0.35, done, total,
                        "EDS overlap quadrature: " + label,
                    )
                ),
            )
            tracks = tuple(
                replace(track, electron_weight=track.electron_weight * mass)
                for track in auxiliary.eds_tracks
            )
            material_quadrature = EDSMaterialQuadrature(
                eds_tracks=tracks,
                material_flights=tuple(
                    replace(flight, electron_weight=flight.electron_weight * mass)
                    for flight in auxiliary.material_flights
                ),
                quadrature_source_ray_indices=tuple(ray.source_ray_index for ray in plan.rays),
                parent_source_ray_indices=plan.parent_source_ray_indices,
                metrics={
                    **plan.metrics,
                    "eds_overlap_material_weight_fraction": mass * float(
                        auxiliary.metrics["material_hit_weight_fraction"]
                    ),
                    "eds_overlap_material_hit_count": auxiliary.metrics["material_hit_trajectory_count"],
                    "eds_overlap_weighted_material_path_nm": mass * float(
                        auxiliary.metrics["mean_material_path_nm"]
                    ),
                    "eds_overlap_event_limit_weight": mass * float(auxiliary.metrics["event_limit_fraction"]),
                    "eds_overlap_path_limit_weight": mass * float(auxiliary.metrics["path_limit_fraction"]),
                    "eds_overlap_weight_reference": "full current reaching specimen plane; vacuum complement retained",
                    "eds_overlap_changes_terminal_electrons": False,
                },
            )
            overlap_metrics.update(material_quadrature.metrics)
        electron_count = (
            source_electron_count * incident_bundle.surviving_fraction
        )
        elastic_transport.metrics.update(
            {
                "emitted_ray_count": incident_bundle.emitted_ray_count,
                "reaching_sample_ray_count": (
                    incident_bundle.reaching_ray_count
                ),
                "sample_plane_surviving_fraction": (
                    incident_bundle.surviving_fraction
                ),
                "sample_plane_original_centroid_nm": (
                    incident_bundle.original_centroid_nm
                ),
                "sample_plane_target_centroid_nm": (
                    incident_bundle.target_centroid_nm
                ),
                "sample_plane_chief_angle_mrad": (
                    incident_bundle.chief_angle_mrad
                ),
                "sample_plane_energy_range_ev": (
                    incident_bundle.energy_range_ev
                ),
            }
        )
    elif transport_mode == "straight_primary":
        elastic_transport = None
        tracks = point_track_segments(state, x_nm=x_value, y_nm=y_value)
        electron_count = source_electron_count
    else:
        raise ValueError(
            "EDS transport mode must be elastic_monte_carlo or straight_primary"
        )
    result = simulate_eds_tracks(
        tracks,
        detector_geometry,
        incident_electrons=electron_count,
        use_analytical_holder_solid_angle=(
            str(
                getattr(
                    sample,
                    "eds_solid_angle_mode",
                    "installed_holder",
                )
            )
            != "unshadowed"
        ),
        detector_efficiency=float(
            getattr(sample, "eds_detector_efficiency", 1.0)
        ),
        energy_min_ev=0.0,
        energy_max_ev=float(
            getattr(sample, "eds_spectrum_max_energy_ev", 40_000.0)
        ),
        energy_bin_width_ev=float(
            getattr(sample, "eds_spectrum_bin_width_ev", 10.0)
        ),
        energy_resolution_fwhm_ev=float(
            getattr(sample, "eds_energy_resolution_fwhm_ev", 0.0)
        ),
        poisson_enabled=bool(
            getattr(sample, "eds_poisson_enabled", False)
        ),
        poisson_seed=int(getattr(sample, "eds_poisson_seed", 0)),
        state=state,
        elastic_transport=(material_quadrature if material_quadrature is not None else elastic_transport),
        photon_origin_xy_nm=(x_value, y_value),
        photon_detector_surfaces=photon_detector_surfaces,
        photon_holder_occluders=photon_holder_occluders,
        photon_quadrature_order=photon_quadrature_order,
        photon_maximum_stored_paths=photon_maximum_stored_paths,
        progress_callback=(
            None if progress_callback is None else
            lambda done, total, label: _report_phase_progress(
                progress_callback, 0.35, 0.99, done, total, label,
            )
        ),
    )
    result.metrics.update(
        {
            "acquisition_kind": "explicit_point",
            "sample_x_nm": x_value,
            "sample_y_nm": y_value,
            "dwell_time_s": dwell,
            "incident_electron_reference": (
                "explicit source-electron argument"
                if incident_electrons is not None
                else "emitted source current"
            ),
            "source_electrons_before_column_losses": source_electron_count,
            "electrons_reaching_sample_plane": electron_count,
            "requested_transport_mode": transport_mode,
        }
    )
    if elastic_transport is not None:
        result.metrics.update(elastic_transport.metrics)
        result = replace(result, elastic_transport=elastic_transport)
    result.metrics.update(overlap_metrics)
    result = replace(result, material_quadrature=material_quadrature)
    return result
