"""Generic, detector-geometry-aware characteristic EDS signal model.

The signal layer converts electron track segments into shell vacancies,
characteristic photons and expected detector counts.  Track segments may be
caller supplied, the straight-primary reference, or Monte Carlo averages from
the finite-geometry elastic transport kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
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

    def mass_attenuation_cm2_g(self, photon_energy_ev: float) -> float:
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
class EDSSpectrum:
    energy_bin_centres_ev: np.ndarray
    expected_counts: np.ndarray
    sampled_counts: np.ndarray | None
    lines: tuple[EDSLineSignal, ...]
    metrics: dict[str, object]
    vacancies: tuple[EDSVacancySignal, ...] = ()
    elastic_transport: object | None = None

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
    from ase.io import read

    path = Path(cif_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"CIF file does not exist: {path}")
    unit = read(path)
    if len(unit) == 0:
        raise ValueError(f"CIF file contains no atoms: {path}")
    volume_angstrom3 = abs(float(np.linalg.det(unit.cell.array)))
    if not math.isfinite(volume_angstrom3) or volume_angstrom3 <= 0.0:
        raise ValueError("CIF EDS material requires a finite 3-D unit cell")
    masses = np.asarray(unit.get_masses(), dtype=float)
    density = float(np.sum(masses)) * ATOMIC_MASS_UNIT_G / (
        volume_angstrom3 * 1.0e-24
    )
    fractions = _normalised_mass_fractions(unit.numbers, masses)
    if any(z > 99 for z, _fraction in fractions):
        raise ValueError(
            "The Bote-Salvat EDS model currently supports CIF elements Z<=99"
        )
    return EDSMaterial(
        key=f"cif:{path.name}",
        name=path.stem,
        density_g_cm3=density,
        mass_fractions=fractions,
        provenance=f"CIF unit-cell composition and crystallographic density: {path}",
    )


def material_from_sample(state) -> EDSMaterial | None:
    sample = state.sample
    if not bool(getattr(sample, "inserted", True)):
        return None
    if specimen_mode(sample) == "atomic":
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


_ORBITALS = (
    "K",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "Q1",
)


@lru_cache(maxsize=None)
def _radiative_lines(
    atomic_number: int, subshell_index: int
) -> tuple[tuple[str, float, float], ...]:
    origin = _ORBITALS[int(subshell_index)]
    rows = []
    for destination in _ORBITALS[int(subshell_index) + 1 :]:
        constant_name = f"{origin}{destination}_LINE"
        constant = getattr(xraylib, constant_name, None)
        if constant is None:
            continue
        try:
            energy_ev = 1000.0 * float(
                xraylib.LineEnergy(int(atomic_number), constant)
            )
            rate = float(xraylib.RadRate(int(atomic_number), constant))
        except ValueError:
            continue
        if energy_ev > 0.0 and rate > 0.0:
            rows.append((f"{origin}-{destination}", energy_ev, rate))
    return tuple(rows)


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
        }
    )
    return EDSSpectrum(centres, expected, None, (), metrics)


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
        "detector_efficiency_model": "ideal_scalar",
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
        "track_segment_count": len(track_rows),
    }
    if incident == 0.0 or not track_rows:
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
                try:
                    fluorescence_yield = float(
                        xraylib.FluorYield(z, subshell_index)
                    )
                except ValueError:
                    fluorescence_yield = 0.0
                try:
                    auger_yield = float(
                        xraylib.AugerYield(z, subshell_index)
                    )
                except ValueError:
                    auger_yield = 0.0
                direct_yield_sum = fluorescence_yield + auger_yield
                if direct_yield_sum > 1.0:
                    if direct_yield_sum > 1.0 + 2.0e-12:
                        raise ValueError(
                            "xraylib vacancy-relaxation yields exceed one"
                        )
                    fluorescence_yield /= direct_yield_sum
                    auger_yield /= direct_yield_sum
                unresolved_relaxation_yield = max(
                    1.0 - fluorescence_yield - auger_yield,
                    0.0,
                )
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
    for line in lines:
        if not spectrum_values[0] <= line.energy_ev < spectrum_values[1]:
            outside_counts += line.expected_detected_counts
            continue
        if sigma_ev <= 0.0:
            index = int(np.searchsorted(edges, line.energy_ev) - 1)
            index = min(max(index, 0), expected.size - 1)
            expected[index] += line.expected_detected_counts
            continue
        lower = max(
            int(
                np.searchsorted(
                    centres, line.energy_ev - 5.0 * sigma_ev
                )
            ),
            0,
        )
        upper = min(
            int(
                np.searchsorted(
                    centres, line.energy_ev + 5.0 * sigma_ev
                )
            )
            + 1,
            expected.size,
        )
        weights = np.exp(
            -0.5
            * ((centres[lower:upper] - line.energy_ev) / sigma_ev) ** 2
        )
        total_weight = float(np.sum(weights))
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
        }
    )
    return EDSSpectrum(
        energy_bin_centres_ev=centres,
        expected_counts=expected,
        sampled_counts=sampled,
        lines=tuple(lines),
        metrics=metrics,
        vacancies=tuple(vacancies),
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
    current_pa = max(
        float(state.electron_gun.emitted_current_a) * 1.0e12, 0.0
    )
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
    elastic_transport = None
    if transport_mode == "elastic_monte_carlo":
        # Local import keeps the EDS track contract independent and avoids a
        # module-load cycle: elastic_transport consumes EDSMaterial and
        # ElectronTrackSegment, while only this explicit acquisition invokes
        # trajectory generation.
        from temsim.specimen.elastic_transport import (
            incident_rays_from_simulation,
            simulate_elastic_point_transport,
        )

        incident_bundle = incident_rays_from_simulation(
            state,
            simulation,
            target_x_nm=x_value,
            target_y_nm=y_value,
        )
        elastic_transport = simulate_elastic_point_transport(
            state,
            incident_rays=incident_bundle.rays,
        )
        tracks = elastic_transport.eds_tracks
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
    return result
