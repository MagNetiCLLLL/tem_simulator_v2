"""Exclusive ownership of the objective aperture in TEM wave propagation."""
from dataclasses import dataclass
import numpy as np

from temsim.physics.multiplane_wave import intermediate_apertures
from temsim.physics.recording_stop import active_tem_recording_plane


@dataclass(frozen=True)
class ObjectiveAperturePlan:
    strategy: str
    equivalent_mask: np.ndarray | None
    excluded_keys: tuple[str, ...]
    metadata: dict


def objective_aperture_plan(state, x_angstrom, y_angstrom, wavelength_angstrom):
    strategy = str(getattr(state.sample, "wave_objective_aperture_strategy", "physical_plane"))
    if strategy not in {"physical_plane", "equivalent_pupil"}:
        raise ValueError(f"Unknown objective aperture strategy: {strategy}")
    stop = active_tem_recording_plane(state)
    planes = intermediate_apertures(state, stop.z_mm)
    objective_key = str(state.objective_aperture.key)
    plane = next((p for p in planes if p.key == objective_key), None)
    metadata = {"strategy": strategy, "active": plane is not None,
                "physical_element_id": "aperture:" + objective_key,
                "enabled": bool(getattr(state.objective_aperture, "enabled", True)),
                "inserted": bool(getattr(state.objective_aperture, "inserted", True)),
                "installed": bool(getattr(state.objective_aperture, "installed", True)),
                "recording_z_mm": float(stop.z_mm),
                "reference_plane": "specimen_exit_canonical_momentum",
                "validity": "actual ordered physical plane" if strategy == "physical_plane"
                            else "Fourier-conjugate plane only; no earlier intervening physical aperture"}
    if plane is not None:
        metadata.update({"z_mm": plane.z_mm, "radius_mm": plane.radius_mm,
                         "offset_x_mm": plane.offset_x_mm, "offset_y_mm": plane.offset_y_mm})
    if plane is None or strategy == "physical_plane":
        return ObjectiveAperturePlan(strategy, None, (), metadata)
    if any(p.z_mm <= plane.z_mm and p.key != plane.key for p in planes):
        raise ValueError("equivalent_pupil unsupported: an earlier physical aperture requires ordered plane propagation")
    from temsim.physics.multiplane_wave import _canonical_map_and_offset
    matrix, offset = _canonical_map_and_offset(state, plane.z_mm)
    a, b = matrix[:2, :2], matrix[:2, 2:]
    # This is the Fourier-plane special case, not radius / axial distance.
    # Requiring a negligible A block avoids silently discarding the position
    # dependence of a finite illuminated specimen. Fixed tolerance is recorded.
    x, y = np.asarray(x_angstrom) * 1e-10, np.asarray(y_angstrom) * 1e-10
    wavelength_m = wavelength_angstrom * 1e-10
    frequency_step = min(1/(len(x)*(x[1]-x[0])), 1/(len(y)*(y[1]-y[0])))
    mapped_pixel = np.linalg.svd(b, compute_uv=False)[-1] * wavelength_m * frequency_step
    position_error = np.linalg.norm(a, ord=2) * max(np.max(np.abs(x)), np.max(np.abs(y)))
    metadata.update({"z_mm": plane.z_mm, "position_error_bound_m": float(position_error),
                     "allowed_position_error_m": float(max(mapped_pixel, 1e-30)*1e-8),
                     "transfer_matrix": matrix.tolist(), "offset_m": offset[:2].tolist()})
    if mapped_pixel <= 0 or position_error > max(mapped_pixel, 1e-30)*1e-8:
        raise ValueError("equivalent_pupil unsupported: objective aperture is not a Fourier-conjugate plane; use physical_plane")
    fx, fy = np.meshgrid(np.fft.fftfreq(len(x), x[1]-x[0]),
                         np.fft.fftfreq(len(y), y[1]-y[0]))
    xy = wavelength_m * np.einsum("ij,jyx->iyx", b, np.stack((fx, fy))) + offset[:2, None, None]
    mask = plane.transmission_mask(xy[0], xy[1]) if plane.radius_mm > 0 else np.zeros(fx.shape, bool)
    return ObjectiveAperturePlan(strategy, mask, (plane.key,), metadata)
