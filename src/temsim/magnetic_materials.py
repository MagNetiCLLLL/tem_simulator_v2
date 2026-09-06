"""Sourced, serializable SI material snapshots; no inferred OEM assignments."""

from copy import deepcopy
import csv
from hashlib import sha256
from pathlib import Path
import tomllib

import numpy as np

MU0 = 4e-7 * np.pi


def validate_bh_material(material: dict) -> dict:
    if not isinstance(material, dict):
        raise ValueError("B-H material must be a table")
    row = deepcopy(material)
    b, h = np.asarray(row.get("b_t", ()), float), np.asarray(row.get("h_a_per_m", ()), float)
    if (b.ndim != 1 or h.shape != b.shape or not 3 <= len(b) <= 100000
            or not np.all(np.isfinite(b)) or not np.all(np.isfinite(h))
            or b[0] != 0 or h[0] != 0 or np.any(np.diff(b) <= 0) or np.any(np.diff(h) <= 0)):
        raise ValueError("B-H data require at least three finite, strictly increasing B/H pairs starting at (0, 0), in T and A/m")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        slopes = np.diff(h)/np.diff(b)
    if not np.all(np.isfinite(slopes)) or np.any(slopes <= 0):
        raise ValueError("B-H differential reluctivity must be finite and positive")
    for key in ("label", "source_url", "source_sha256"):
        if not str(row.get(key, "")).strip():
            raise ValueError(f"B-H material requires {key}")
    digest = str(row["source_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("B-H source SHA-256 must contain 64 lowercase hexadecimal digits")
    if row.get("model", "isotropic_single_valued_bh") != "isotropic_single_valued_bh":
        raise ValueError("Only isotropic, single-valued B-H materials are supported")
    if row.get("interpolation", "piecewise_linear_h_of_b") != "piecewise_linear_h_of_b":
        raise ValueError("Unsupported B-H interpolation")
    if row.get("extrapolation", "reject_final_solution") != "reject_final_solution":
        raise ValueError("Out-of-range final B-H solutions must be rejected")
    row.update(b_t=b.tolist(), h_a_per_m=h.tolist(), model="isotropic_single_valued_bh",
               interpolation="piecewise_linear_h_of_b", extrapolation="reject_final_solution")
    return row


def reference_materials() -> tuple[dict, ...]:
    from temsim.paths import project_root
    root = project_root() / "configs" / "materials" / "magnetic"
    return tuple(validate_bh_material(tomllib.loads(path.read_text(encoding="utf-8")))
                 for path in sorted(root.glob("*.toml")))


def lens_material_defaults() -> dict:
    """Load design defaults, not current/geometry recipes or OEM assignments.

    Return independent snapshots so existing saved material inputs never change
    when the library or default selection changes. Linear permeability is the
    cited constant library value, not a derived or fitted B-H slope.
    """
    from temsim.paths import project_root
    path = project_root() / "configs" / "materials" / "lens_defaults.toml"
    defaults = tomllib.loads(path.read_text(encoding="utf-8"))
    if defaults.get("schema_version") != 1:
        raise ValueError("Unsupported lens material defaults schema")
    matches = [row for row in reference_materials() if row.get("key") == defaults.get("material_key")]
    if len(matches) != 1:
        raise ValueError("Lens material default must identify exactly one reference material")
    value = defaults.get("linear_relative_permeability")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
        raise ValueError("Default linear relative permeability must be finite and positive")
    if defaults.get("linear_source_fields") != ["Mu_x", "Mu_y"]:
        raise ValueError("Default linear permeability requires the cited Mu_x/Mu_y fields")
    material = matches[0]
    return dict(bh_material=material, linear_material_reference=dict(
        material_key=material["key"], label=material["label"],
        relative_permeability=float(value), source_url=material["source_url"],
        source_sha256=material["source_sha256"], source_block=material["source_block"],
        source_fields=defaults["linear_source_fields"], scope=defaults["scope"],
        selection_source_urls=defaults["selection_source_urls"],
    ))


def import_bh_csv(path: str | Path) -> dict:
    """Import explicit B_T,H_A_per_m columns; never sort, smooth or rescale data."""
    path = Path(path)
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if not rows or set(rows[0]) != {"B_T", "H_A_per_m"}:
        raise ValueError("B-H CSV headers must be B_T,H_A_per_m (tesla, A/m)")
    return validate_bh_material(dict(label=path.stem, source_url=path.resolve().as_uri(),
                                    source_sha256=sha256(raw).hexdigest(),
                                    scope="User supplied reference; material identity not independently verified",
                                    b_t=[float(row["B_T"]) for row in rows],
                                    h_a_per_m=[float(row["H_A_per_m"]) for row in rows]))


class BHCurve:
    """H(|B|) and differential reluctivity; no saturation clamp or hysteresis."""

    def __init__(self, material: dict):
        self.material = validate_bh_material(material)
        self.b = np.asarray(self.material["b_t"])
        self.h = np.asarray(self.material["h_a_per_m"])
        self.slopes = np.diff(self.h) / np.diff(self.b)

    def evaluate(self, magnitude_t, *, allow_iteration_extension=False):
        b = np.asarray(magnitude_t, float)
        if np.any(b < 0) or not np.all(np.isfinite(b)):
            raise ValueError("B-H queries require finite nonnegative magnitudes")
        if not allow_iteration_extension and np.any(b > self.b[-1]*(1+1e-12)):
            raise ValueError(f"B-H data range exceeded for {self.material['label']}: max {b.max():.6g} T, supported {self.b[-1]:.6g} T")
        index = np.clip(np.searchsorted(self.b, b, side="right")-1, 0, len(self.slopes)-1)
        slope = self.slopes[index]
        value = self.h[index] + slope*(b-self.b[index])
        # Only a safeguarded Newton trial may enter this extension. A returned
        # physical solution must pass the strict source-data range check.
        outside = b > self.b[-1]
        return (np.where(outside, self.h[-1]+(b-self.b[-1])/MU0, value),
                np.where(outside, 1/MU0, slope))
