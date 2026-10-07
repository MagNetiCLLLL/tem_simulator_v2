"""Identity of a diagnostic path, distinct from field data and worker ownership.

Only executed field identities, supported hardware stop definitions, actual
initial conditions and integration inputs are consumed. These hashes never
create a source, authorize continuation or replace the worker's live token.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import numpy as np

from temsim.immutable_json import json_digest
from temsim.optics.model import Aperture
from temsim.optics.electron_gun.aperture import GunAperture
from temsim.optics.condenser_aperture import ContinuousApertureComponent
from temsim.optics.objective_aperture import ObjectiveApertureComponent
from temsim.optics.selected_area_aperture import SelectedAreaApertureComponent
from temsim.optics.energy_filter_entrance_aperture import EnergyFilterEntranceApertureComponent


_APERTURE_LAWS = {cls: getattr(cls, "transmission_mask", None) for cls in (
    Aperture, GunAperture, ContinuousApertureComponent, ObjectiveApertureComponent,
    SelectedAreaApertureComponent, EnergyFilterEntranceApertureComponent)}


def _aperture_inputs(aperture):
    method = getattr(aperture, "transmission_mask", None)
    if (type(aperture) not in _APERTURE_LAWS
            or getattr(method, "__func__", method) is not _APERTURE_LAWS[type(aperture)]):
        return None
    row = {"key": aperture.key, "model": type(aperture).__name__,
           "z_mm": float(aperture.z_mm), "radius_mm": float(aperture.radius_mm),
           "offset_x_mm": float(aperture.offset_x_mm), "offset_y_mm": float(aperture.offset_y_mm)}
    if type(aperture) is GunAperture:
        row["enabled"] = bool(aperture.enabled)
        slit = getattr(aperture, "_slit_profile", None)
        if getattr(aperture, "_slit_mode", False) and slit is not None:
            row["slit"] = {"inserted": bool(slit.inserted), "gap_um": float(slit.gap_um),
                           "centre_offset_um": float(slit.centre_offset_um),
                           "mechanical_bore_diameter_mm": float(aperture.mechanical_bore_diameter_mm)}
    return row


def transport_context_identity(scene):
    """Bind known field data to physical stops and actual diagnostic support.

    Unidentified/custom fields and aperture implementations remain unknown.
    A freshly generated process token must never fill that gap.
    """
    if not getattr(scene, "numerical_identity", None):
        return None
    apertures = [_aperture_inputs(value) for value in scene._apertures]
    if any(value is None for value in apertures):
        return None
    bores = [{"key": b.key, "lower_m": b.lower_m, "upper_m": b.upper_m,
              "inner_m": b.inner_m, "outer_m": b.outer_m if np.isfinite(b.outer_m) else None,
              "registration": asdict(b.registration) if b.registration is not None else None}
             for b in scene._bores]
    registrations = getattr(scene, "_aperture_registrations", ()) or (None,) * len(apertures)
    return json_digest({"schema": "diagnostic-transport-context-v2",
        "numerical_field": scene.numerical_identity,
        "bounds_m": np.asarray(scene.diagnostic_bounds_m).tolist(),
        "electric_bounds_m": np.asarray(scene.electric_bounds_m).tolist(),
        "apertures": apertures, "bores": bores, "unbounded_outer_radius": "None means infinity",
        "aperture_registrations": [asdict(value) if value is not None else None for value in registrations],
        "flat_cathode": scene._flat_cathode, "post_exit_ground": scene._post_exit_ground,
        "unsupported_stops": scene._unsupported_stops,
        "column_model": getattr(scene, "_column_identity", None),
        "column_handoff_z_m": getattr(scene, "_column_handoff_z_m", None)})


_IMPLEMENTATION_FILES = (
    "magnetic_test_particle.py", "test_electron_compiled.py", "test_electron_compiled_laws.py",
    "test_electron_intercepts.py", "physics/axis_field_interpolation.py",
    "test_electron_scene.py", "test_electron_sampling.py", "diagnostic_execution_identity.py",
    "physics/instrument_magnetic.py", "physics/compiled_magnetic_field.py",
    "physics/discrete_gradient.py", "physics/static_energy_lorentz.py",
    "physics/grounded_particle_step.py", "physics/instrument_electric.py",
    "physics/core.py", "physics/electrostatic_column_transport.py",
    "physics/posed_aberrations.py", "physics/posed_aberration_wave.py", "physics/vector_field_transport.py",
    "physics/ray_integrator.py", "physics/lens_field_provider.py",
    "test_electron_column.py",
    "lens_pose.py", "physics/column_wall.py", "physics/aperture_clipping.py",
    "physics/relativistic_lorentz.py", "optics/electron_gun/aperture.py",
    "optics/model.py", "optics/condenser_aperture.py", "optics/objective_aperture.py",
    "optics/selected_area_aperture.py", "optics/energy_filter_entrance_aperture.py",
)


@lru_cache(maxsize=1)
def _implementation_hash(stamps):
    root = Path(__file__).parent
    return json_digest({name: sha256((root / name).read_bytes()).hexdigest()
                        for name, _mtime, _size in stamps})


@lru_cache(maxsize=1)
def _library_versions():
    libraries = {"numpy": np.__version__}
    for name in ("scipy", "numba", "llvmlite"):
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libraries[name] = None
    return libraries


def trajectory_execution_identity(scene, settings, *, use_compiled=True):
    """One deterministic execution request; display edits are absent by design."""
    context = getattr(scene, "transport_identity", None)
    if context is None:
        return None
    from temsim.magnetic_test_particle import TestElectronSettings
    if not isinstance(settings, TestElectronSettings):
        raise TypeError("Execution identity requires diagnostic electron settings")
    root = Path(__file__).parent
    stamps = tuple((name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size)
                   for name in _IMPLEMENTATION_FILES)
    return json_digest({"schema": "diagnostic-electron-execution-v1", "context": context,
                        "settings": asdict(settings), "compiled_requested": bool(use_compiled),
                        "implementation": _implementation_hash(stamps), "libraries": _library_versions()})
