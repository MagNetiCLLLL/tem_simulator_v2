"""Recalculate condenser presets for the optional gun-to-column NanoPulser.

The ordinary three-module instrument retains its stored presets.  Installing
an extra module changes the source-to-C1 transfer and therefore requires a
full-ray checked C2/C3 solve, even with the blanking voltage switched off.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from temsim.operating_modes import direct_alignment_by_key
from temsim.optics.calibration_beam import transmitted_calibration_beam


CALIBRATION_RAYS = 512
_CACHE: dict[str, object] = {}


def _calibration_key(state, mode, definition) -> str:
    # Runtime serialization alone deliberately omits TOML-owned geometry.
    # Include the resolved manifests and absolute positions as well, so an
    # edited module length or field profile can never reuse an old result.
    assembly = getattr(state, "_resolved_assembly", None)
    geometry = [
        (part.key, part.start_z_mm, part.center_z_mm, part.end_z_mm,
         repr(dict(part.data)))
        for part in getattr(assembly, "parts", ())
    ]
    def transmitted_settings(value):
        if isinstance(value, dict):
            return {
                key: (False if key == "beam_blanked" else transmitted_settings(item))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [transmitted_settings(item) for item in value]
        return value

    # Inactive gun profiles may retain their last ordinary-blanker state.
    # Exposure gates do not describe transmitted-beam optics, but all saved
    # alignment fields and hardware parameters still participate in the key.
    payload = (
        transmitted_settings(state.to_dict()), geometry, mode.key,
        definition.targets,
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def recalibrate_nanopulser_condenser(state, mode, catalog):
    """Return freshly measured mode metadata and install validated C2/C3.

    Calibration virtually opens the ordinary gun-coil blanker and installed
    NanoPulser, retaining every physical aperture and alignment value. Both
    blanking selections and the ray count are restored even if the solve fails.
    Cache keys describe the transmitted beam, so toggling either exposure gate
    does not alone invalidate an otherwise identical condenser calibration.
    """
    key = ("nanoprobe_convergence" if mode.key == "nano_probe"
           else "microprobe_illumination")
    base_definition = direct_alignment_by_key(key, catalog)
    definition = replace(base_definition, targets={
        **base_definition.targets,
        "optimiser_step_mm": 0.1,
        "validation_step_mm": 0.05,
    })
    emitter = state.electron_gun.emitter
    original_rays = emitter.ray_count
    with transmitted_calibration_beam(state):
        try:
            emitter.ray_count = CALIBRATION_RAYS
            return _recalibrate_transmitted_condenser(state, mode, definition)
        finally:
            emitter.ray_count = original_rays


def _recalibrate_transmitted_condenser(state, mode, definition):
    from temsim.optics.direct_alignment import (
        apply_direct_alignment, _refine_nanoprobe_production_focus,
    )

    key = definition.key
    cache_key = _calibration_key(state, mode, definition)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        lenses = {lens.key: lens for lens in state.lenses}
        for lens_key in ("condenser_lens_2", "condenser_lens_3"):
            lenses[lens_key].percent = cached.devices[lens_key]["percent"]
        return cached

    if mode.key == "nano_probe":
        lenses = {lens.key: lens for lens in state.lenses}
        keys = ("condenser_lens_2", "condenser_lens_3")
        initial = np.asarray([lenses[key].percent for key in keys])
        # Keep an acceptable aperture-defined convergence while placing
        # its waist on the new sample plane.  A fresh layout should not
        # accept a merely nearby waist simply because the normal live
        # adjustment allows a wider operational focus tolerance.
        focus_definition = replace(definition, targets={
            **definition.targets, "maximum_waist_offset_mm": 1.0e-9,
        })
        focused, _ = _refine_nanoprobe_production_focus(
            state, focus_definition, definition.default_value,
            initial, definition.targets["validation_step_mm"],
        )
        for lens_key, percent in zip(keys, focused):
            lenses[lens_key].percent = float(percent)
    result = apply_direct_alignment(
        state, key, definition.default_value, definition=definition,
    )
    if not result.success:
        raise ValueError(
            "NanoPulser condenser preset recalculation failed: "
            + result.message
        )
    devices = {device_key: dict(values)
               for device_key, values in mode.devices.items()}
    for lens_key, percent in result.strengths.items():
        devices[lens_key]["percent"] = float(percent)
    # Stored source-probe and step-convergence metrics belong to the old
    # geometry. Only publish quantities actually checked by this solve.
    targets = {name: value for name, value in mode.targets.items()
               if not name.startswith(("achieved_", "step_"))
               and name not in ("calibration_ray_count", "calibration_step_mm",
                                "validation_step_mm")}
    targets.update({
        "achieved_convergence_sem_angle_mrad": result.convergence_95_mrad,
        "achieved_illumination_diameter_95_um": result.illumination_diameter_95_um,
        "calibration_ray_count": CALIBRATION_RAYS,
        "calibration_step_mm": 0.1,
        "validation_step_mm": result.validation_step_mm,
        "calibration_geometry_fingerprint": cache_key,
    })
    if mode.key == "nano_probe":
        targets["achieved_waist_offset_nm"] = result.constraint_value * 1.0e6
    else:
        targets["achieved_wavefront_curvature_per_m"] = result.constraint_value
    calibrated = replace(
        mode, devices=devices, targets=targets,
        calibration_status="computed_live_nanopulser_condenser_non_oem",
        calibration_reference=(
            "C2/C3 recomputed for the installed NanoPulser and current "
            "assembled geometry using 512 deterministic source rays. "
            "Full nonlinear propagation and physical clipping validate "
            "the sample illumination and focus constraint at 0.05 mm. "
            "Principal corrector fields retain the base-mode values; "
            "this is not a new aberration-corrector or OEM current calibration."
        ),
    )
    if len(_CACHE) >= 16:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[cache_key] = calibrated
    return calibrated
