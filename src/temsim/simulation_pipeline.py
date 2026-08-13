"""Application-facing simulation pipeline, independent of the Tk GUI."""
from dataclasses import dataclass

from temsim.detector.recording_system import ensure_recording_system
from temsim.detector.stem_signal import StemScanResult, acquire_stem_scan
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_names import normalise_component_names
from temsim.optics.corrector_structure import ensure_corrector_structure
from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter_raytrace import simulate_energy_filter
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import Simulation, run
from temsim.physics.scan_geometry import (
    ScanGeometryResult,
    calculate_scan_geometry,
    calculate_scan_ray_paths,
)
from temsim.physics.wave_imaging import (
    WaveImagingResult,
    simulate_wave_image,
    tem_wave_imaging_enabled,
)


@dataclass
class CalculationResult:
    simulation: Simulation
    energy_filter: object
    state_snapshot: object = None
    layout: object = None
    assembly: object = None
    wave_imaging: WaveImagingResult | None = None
    scan_geometry: ScanGeometryResult | None = None
    scan_ray_paths: object | None = None
    stem_scan: StemScanResult | None = None
    lens_crossovers: tuple[dict[str, object], ...] = ()
    aperture_stops: tuple[dict[str, object], ...] = ()


def aperture_stop_records(state) -> tuple[dict[str, object], ...]:
    """Capture the exact runtime hard-edge controls used by the solver."""
    records = []
    gun = state.electron_gun
    apertures = [
        getattr(gun, name)
        for name in ("dpa_aperture", "c1_aperture")
        if getattr(gun, name, None) is not None
    ]
    apertures.extend(state.apertures)
    seen = set()
    for aperture in apertures:
        if aperture.key in seen:
            continue
        seen.add(aperture.key)
        radius = getattr(
            aperture,
            "radius_mm",
            getattr(aperture, "effective_aperture_radius_mm", 0.0),
        )
        records.append(
            {
                "key": str(aperture.key),
                "name": str(aperture.name),
                "z_mm": float(aperture.z_mm),
                "radius_mm": max(0.0, float(radius)),
                "offset_x_mm": float(getattr(aperture, "offset_x_mm", 0.0)),
                "offset_y_mm": float(getattr(aperture, "offset_y_mm", 0.0)),
                "enabled": bool(getattr(aperture, "enabled", True)),
                "installed": bool(getattr(aperture, "installed", True)),
            }
        )
    return tuple(records)


def calculate_stem_scan_frame(state, simulation):
    """Calculate exactly one detector-signal frame when AC scan is active."""

    component = state.ac_deflector
    if not bool(component.enabled and component.scan_enabled):
        return None
    return acquire_stem_scan(simulation, state)


def calculate(state):
    """Normalise editable state and calculate all non-visual simulation results."""
    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    layout = apply_physical_layout_to_state(state)
    ensure_recording_system(state)
    ensure_corrector_structure(state)
    simulation = run(state, resolved_layout=layout)
    wave_imaging = (
        simulate_wave_image(state, simulation)
        if tem_wave_imaging_enabled(state)
        else None
    )
    energy_filter = simulate_energy_filter(state, simulation)
    scan_geometry = calculate_scan_geometry(state)
    scan_ray_paths = calculate_scan_ray_paths(state, simulation)
    stem_scan = calculate_stem_scan_frame(state, simulation)
    state.energy_filter_result = energy_filter
    lens_crossovers = detect_all_lens_crossovers(
        [simulation.incident, *simulation.branches.values()], state.lenses)
    state.all_lens_crossovers = lens_crossovers
    return CalculationResult(
        simulation=simulation,
        energy_filter=energy_filter,
        state_snapshot=state,
        layout=layout,
        assembly=state._resolved_assembly,
        wave_imaging=wave_imaging,
        scan_geometry=scan_geometry,
        scan_ray_paths=scan_ray_paths,
        stem_scan=stem_scan,
        lens_crossovers=tuple(lens_crossovers),
        aperture_stops=aperture_stop_records(state),
    )
