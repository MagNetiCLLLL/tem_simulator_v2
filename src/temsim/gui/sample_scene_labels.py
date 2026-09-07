"""Short, unit-explicit labels for physical and rendered sample geometry."""
from __future__ import annotations

import numpy as np


def _length(value_nm: float) -> str:
    value = float(value_nm)
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.5g} mm"
    if abs(value) >= 1.0e3:
        return f"{value / 1.0e3:.5g} µm"
    return f"{value:.5g} nm"


def _size(values) -> str:
    return " × ".join(_length(value) for value in values)


def sample_scene_labels(snapshot, *, completed_region: bool) -> tuple[str, str, str]:
    """Keep full material, requested local window and rendering cap distinct."""
    sx, sy, sz = snapshot.size_nm
    if snapshot.envelope_shape == "disk":
        full = f"Full sample — outline | Diameter {_length(sx)} | Thickness {_length(sz)}"
    else:
        full = f"Full sample — outline | {_size((sx, sy, sz))}"
    bounds = getattr(snapshot, "local_material_bounds_nm", None)
    if bounds is not None:
        size = np.diff(np.asarray(bounds, dtype=float).reshape(3, 2), axis=1).ravel()
        region = "Local calculation region" if completed_region else "Local structure preview"
        local = f"{region} | {_size(size)}"
    else:
        local = "Local region | No material intersection" if completed_region else "Local structure preview | No atomic region"
    count = len(snapshot.atomic_numbers)
    if not count:
        atoms = "Spheres | No explicit atoms"
    elif getattr(snapshot, "atom_display_capped", False):
        atoms = f"Spheres | {count:,} atoms shown | Display subset: {_size(snapshot.atom_display_size_nm)}"
    else:
        atoms = f"Spheres | {count:,} atoms in the local region"
    return full, local, atoms
