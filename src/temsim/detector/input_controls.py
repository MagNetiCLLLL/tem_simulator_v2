"""Strict operating records for the current physical detector components."""

from collections.abc import Mapping
from numbers import Real
import math


def restore_detector_controls(component, data, *, fields, required=("inserted",)):
    if not isinstance(data, Mapping):
        raise ValueError("Detector controls must be a mapping")
    if data.get("key") != component.key:
        raise ValueError(f"Detector record requires current key {component.key!r}")
    unknown = set(data) - {"key", *fields}
    missing = set(required) - set(data)
    if unknown:
        raise ValueError(f"Unknown detector operating fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing detector operating fields: {', '.join(sorted(missing))}")
    for name in fields:
        if name not in data:
            continue
        if name in {"inserted", "readout_enabled"} and type(data[name]) is not bool:
            raise ValueError(f"Detector {name} must be a boolean")
        value = data[name]
        if name == "pixels" and (type(value) is not int or value < 1):
            raise ValueError("Camera pixels must be a positive integer")
        if name in {"centre_offset_x_mm", "centre_offset_y_mm"} and (
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        ):
            raise ValueError(f"Detector {name} must be a finite number")
        setattr(component, name, value)
    return component.validate()
