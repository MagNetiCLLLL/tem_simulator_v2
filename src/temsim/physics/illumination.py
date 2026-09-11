"""Historical illumination metadata and explicit production-migration gates.

Independent-pupil numerical fixtures live in tests/fixtures/illumination.py.
This module must not reconstruct missing gun phase from particle diagnostics.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, asdict
from itertools import product
import math

import numpy as np

from temsim.physics.wave_flux import TEM_REFERENCE_PLANE

LEGACY_MODEL = "ray_conditioned_reduced_order"
EXPLICIT_MODEL = "specimen_entrance_pupil_modes"
ILLUMINATION_SCHEMA = "gun-source-policy-v2-reduced-order-transition"
MAX_SOURCE_MODES = 256


def _finite(values, shape, name):
    a = np.asarray(values, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


@dataclass(frozen=True)
class PupilState:
    """A conditional complex angular shape, not an extra physical stop.

    Coordinates are Cartesian paraxial direction angles in mrad. For analytic
    pupils, basis maps local angular coordinates into the laboratory frame;
    radii are local semi-axes. Sampled pupils use basis as the per-cell affine
    increment and offset as the coordinate of the central array index.
    Intensity input means sqrt(values) with the separately declared phase.
    No phase is inferred from an intensity measurement.
    """
    shape: str = "ellipse"
    semi_axes_mrad: tuple = (20., 20.)
    offset_mrad: tuple = (0., 0.)
    basis: tuple = ((1., 0.), (0., 1.))
    semantics: str = "amplitude"
    values: tuple = ()
    phase_rad: tuple = ()
    reference_plane: str = TEM_REFERENCE_PLANE
    provenance: str = "user_declared"

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("pupil must be an object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown pupil fields: {sorted(unknown)}")
        obj = cls(**data)
        obj.validate()
        # Own immutable data even if the caller supplied mutable nested lists.
        return cls(**{k: _tuples(v) for k, v in asdict(obj).items()})

    def validate(self):
        if self.shape not in {"ellipse", "rectangle", "sampled"}:
            raise ValueError("pupil shape must be ellipse, rectangle or sampled")
        if self.semantics not in {"amplitude", "intensity"}:
            raise ValueError("pupil semantics must be amplitude or intensity")
        if self.reference_plane != TEM_REFERENCE_PLANE or not self.provenance:
            raise ValueError("Declare the specimen-entrance reference and pupil provenance")
        axes = _finite(self.semi_axes_mrad, (2,), "pupil semi axes")
        if np.any(axes <= 0) or np.max(axes) >= 200:
            raise ValueError("Pupil semi axes must be positive and below 200 mrad (paraxial model)")
        _finite(self.offset_mrad, (2,), "pupil offset")
        basis = _finite(self.basis, (2, 2), "pupil basis")
        if abs(np.linalg.det(basis)) < 1e-12:
            raise ValueError("Pupil coordinate basis must be nonsingular")
        if self.shape == "sampled":
            values = np.asarray(self.values, dtype=float)
            if (values.ndim != 2 or min(values.shape) < 2 or max(values.shape) > 512
                    or not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0)):
                raise ValueError("Sampled pupil needs a nonnegative, nonzero 2-D array, 2–512 cells per axis")
            if len(self.phase_rad):
                _finite(self.phase_rad, values.shape, "pupil phase")
        elif len(self.values) or len(self.phase_rad):
            raise ValueError("Array values and phase require shape=sampled")
        if self.support_radius_mrad >= 200:
            raise ValueError("Pupil support exceeds the 200 mrad paraxial scope")

    @property
    def half_extent_local(self):
        if self.shape == "sampled":
            # Include the linear interpolation footprint around border cells.
            ny, nx = np.shape(self.values)
            return np.array((nx // 2 + 1, ny // 2 + 1), float)
        return np.asarray(self.semi_axes_mrad, float)

    @property
    def support_radius_mrad(self):
        if self.shape == "ellipse":
            return float(np.linalg.svd(np.asarray(self.basis) @ np.diag(self.semi_axes_mrad), compute_uv=False)[0])
        corners = np.array(list(product((-1., 1.), repeat=2))) * self.half_extent_local
        return float(np.max(np.linalg.norm(corners @ np.asarray(self.basis).T, axis=1)))

    def spectrum(self, *args, **kwargs):
        raise ValueError("Independent pupil generation is historical only; numerical fixtures live under tests/")

    def metadata(self):
        return {**asdict(self), "alpha_edge_rad": self.support_radius_mrad * 1e-3,
                "edge_semantics": "declared angular support about pupil centre; not a ray quantile",
                "normalization": "conditional source shape at specimen entrance; no upstream stop transmission inferred",
                "interpolation": "complex-amplitude bilinear, zero exterior" if self.shape == "sampled" else "analytic cell-centre sampling"}


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


@dataclass(frozen=True)
class SourceNode:
    position_nm: tuple
    tilt_mrad: tuple
    energy_offset_ev: float
    weight: float
    mode_id: str


def gaussian_quadrature(sigma, order=3):
    """Normalized Gauss-Hermite nodes for independent normal coordinates."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0) or not 1 <= int(order) <= 15 or int(order) != order:
        raise ValueError("Gaussian sigmas must be nonnegative; quadrature order must be 1–15")
    if int(order) == 1 and np.any(sigma > 0):
        raise ValueError("Nonzero spread requires at least two quadrature nodes")
    roots, weights = np.polynomial.hermite.hermgauss(int(order))
    dimensions = [[(0., 1.)] if s == 0 else list(zip(np.sqrt(2) * s * roots, weights / np.sqrt(np.pi))) for s in sigma]
    return [[*(float(p[0]) for p in points), math.prod(float(p[1]) for p in points)] for points in product(*dimensions)]


def default_illumination_config():
    return {"model": EXPLICIT_MODEL, "pupil": asdict(PupilState()),
            "positions_nm": [[0., 0., 1.]], "angles_mrad": [[0., 0., 1.]],
            "energies_ev": [[0., 1.]]}


def _nodes(data, dimensions, name):
    a = np.asarray(data, float)
    if (a.ndim != 2 or a.shape[1] != dimensions + 1 or not 1 <= len(a) <= MAX_SOURCE_MODES
            or not np.all(np.isfinite(a)) or np.any(a[:, -1] <= 0)
            or not np.isclose(np.sum(a[:, -1]), 1., rtol=0, atol=1e-10)):
        raise ValueError(f"{name}: rows need {dimensions} coordinates and a positive prior; priors must sum to 1")
    return a


def validate_illumination_config(config, voltage_kev=None):
    if not isinstance(config, dict):
        raise ValueError("wave_illumination must be an object")
    allowed = {"model", "pupil", "positions_nm", "angles_mrad", "energies_ev"}
    if set(config) - allowed:
        raise ValueError(f"Unknown illumination fields: {sorted(set(config) - allowed)}")
    model = config.get("model", LEGACY_MODEL)
    if model == LEGACY_MODEL:
        if set(config) - {"model"}:
            raise ValueError("Legacy illumination cannot silently ignore explicit pupil or source modes")
        return {"model": LEGACY_MODEL}
    if model != EXPLICIT_MODEL:
        raise ValueError(f"Unknown illumination model: {model}")
    pupil = PupilState.from_dict(config.get("pupil", {}))
    positions = _nodes(config.get("positions_nm", [[0, 0, 1]]), 2, "positions_nm")
    angles = _nodes(config.get("angles_mrad", [[0, 0, 1]]), 2, "angles_mrad")
    energies = _nodes(config.get("energies_ev", [[0, 1]]), 1, "energies_ev")
    if len(positions) * len(angles) * len(energies) > MAX_SOURCE_MODES:
        raise ValueError(f"At most {MAX_SOURCE_MODES} source × direction × energy modes per calculation")
    if np.max(np.linalg.norm(angles[:, :2] + pupil.offset_mrad, axis=1)) + pupil.support_radius_mrad >= 200:
        raise ValueError("Source directions plus pupil exceed the 200 mrad paraxial scope")
    if voltage_kev is not None and (not math.isfinite(voltage_kev) or np.any(voltage_kev + energies[:, 0] / 1000 <= 0)):
        raise ValueError("Every illumination energy must be positive")
    return {"model": model, "pupil": asdict(pupil), "positions_nm": positions.tolist(),
            "angles_mrad": angles.tolist(), "energies_ev": energies.tolist()}


def illumination_config(state):
    reference_energy = getattr(state, "_illumination_reference_energy_kev", state.beam_voltage_kv)
    config = validate_illumination_config(getattr(state.sample, "wave_illumination", {}), reference_energy)
    require_production_illumination(config)
    return config


def require_production_illumination(config):
    """One gate for GUI, profiles and numerical entry points.

    The old config parser remains available for historical metadata inspection.
    A label or a private source-node attribute cannot enable independent input.
    The retained ray-conditioned path is explicitly reduced-order, not a
    validated gun-to-specimen coherent-wave chain.
    """
    if config.get("model", LEGACY_MODEL) != LEGACY_MODEL:
        raise ValueError(
            "Independent specimen-entrance illumination is historical only. "
            "It cannot be used for a new calculation or restored as a gun source. "
            "Configure the electron gun and physical column instead."
        )


def explicit_illumination(state):
    return illumination_config(state)["model"] == EXPLICIT_MODEL


def source_nodes(config):
    config = validate_illumination_config(config)
    require_production_illumination(config)
    return (SourceNode((0., 0.), (0., 0.), 0., 1., "ray_reduced:0"),)


def state_for_source_node(state, node):
    raise ValueError("Independent source-mode generation is historical only; use the gun model")


def state_at_energy(state, energy_kev):
    """Private energy override; source high voltage and coil controls are fixed."""
    if not math.isfinite(energy_kev) or energy_kev <= 0:
        raise ValueError("Propagation energy must be positive and finite")
    result = copy(state)
    result._propagation_energy_kev = float(energy_kev)
    result.sample = copy(state.sample)
    return result


def illumination_ray_statistics(state, ray_stats):
    if not explicit_illumination(state):
        return ray_stats
    pupil = PupilState.from_dict(illumination_config(state)["pupil"])
    node = getattr(state, "_wave_source_node", source_nodes(illumination_config(state))[0])
    stats = dict(ray_stats)
    # The explicit reference is the sample plane. No ray-emittance-to-pure-mode
    # conversion and no implicit ray-derived defocus are applied to this mode.
    stats.update(mean_x_m=node.position_nm[0]*1e-9, mean_y_m=node.position_nm[1]*1e-9,
                 mean_tx_rad=(pupil.offset_mrad[0]+node.tilt_mrad[0])*1e-3,
                 mean_ty_rad=(pupil.offset_mrad[1]+node.tilt_mrad[1])*1e-3,
                 waist_offset_m=0., radial_wavefront_curvature_per_m=0.,
                 convergence_semiangle_rad=pupil.support_radius_mrad*1e-3,
                 explicit_pupil_support_rad=pupil.support_radius_mrad*1e-3)
    return stats


def explicit_pupil_spectrum(state, *args, **kwargs):
    raise ValueError("Independent pupil generation is historical only; use a gun-owned phase checkpoint")


def illumination_metadata(state, ray_stats=None):
    config = illumination_config(state)
    common = {"illumination_model": config["model"], "illumination_schema": ILLUMINATION_SCHEMA,
              "illumination_reference_plane": TEM_REFERENCE_PLANE,
              "illumination_convergence_status": "NOT_ASSESSED; compare independently refined source/energy quadratures",
              "illumination_config": config}
    if config["model"] == LEGACY_MODEL:
        stats = ray_stats or {}
        return {**common, "alpha_edge_rad": None,
                "alpha_95_current_rad": stats.get("convergence_95_rad"),
                "alpha_99_current_rad": stats.get("convergence_99_rad"),
                "sampled_max_angle_rad": stats.get("convergence_edge_rad"),
                "illumination_scope": "Ray-conditioned reduced-order coherent mode; chief ray, second moments (TEM) / 95%-current disk (STEM), specified aberrations retained. Arbitrary amplitude, higher phase and finite-emittance mixture not recovered."}
    nodes = source_nodes(config)
    pupil = PupilState.from_dict(config["pupil"])
    return {**common, "alpha_edge_rad": pupil.support_radius_mrad * 1e-3,
            "alpha_95_current_rad": None, "alpha_99_current_rad": None,
            "pupil_state": pupil.metadata(), "illumination_modes": [asdict(n) for n in nodes],
            "illumination_mode_count": len(nodes),
            "illumination_scope": "Declared specimen-entrance pupil; independent position, direction and energy priors; per-mode specimen/column propagation and incoherent intensity sum. No upstream source coherence or chromatic focus inferred from rays."}


def current_angle_quantiles(fx, fy, spectrum, wavelength_angstrom, centre_mrad=(0., 0.)):
    weight = np.abs(spectrum)**2
    angle = np.hypot(fx*wavelength_angstrom-centre_mrad[0]*1e-3,
                     fy*wavelength_angstrom-centre_mrad[1]*1e-3).ravel()
    order = np.argsort(angle)
    cumulative = np.cumsum(weight.ravel()[order])
    if cumulative[-1] <= 0:
        raise ValueError("Angular quantiles require nonzero current")
    return {**{f"alpha_{q}_current_rad": float(angle[order[min(np.searchsorted(cumulative, q/100*cumulative[-1]), len(order)-1)]]) for q in (95, 99)},
            "pupil_effective_bandwidth_rad": float(np.max(angle[weight.ravel() > weight.max()*1e-20])),
            "pupil_wave_grid_extent_inv_angstrom": [float(np.min(fx)), float(np.max(fx)), float(np.min(fy)), float(np.max(fy))]}


def convergence_report(values, orders, *, observable, unit, absolute_floor, relative_target=.01):
    """Compare declared observables; a sampling count alone is not convergence."""
    arrays = [np.asarray(v, dtype=float) for v in values]
    if (len(arrays) != len(orders) or len(arrays) < 3 or any(a.shape != arrays[0].shape for a in arrays)
            or not all(np.all(np.isfinite(a)) for a in arrays)
            or any(a >= b for a, b in zip(orders, orders[1:]))
            or not math.isfinite(absolute_floor) or absolute_floor < 0
            or not math.isfinite(relative_target) or relative_target < 0):
        raise ValueError("Convergence needs at least three increasing orders and finite matching observables/tolerances")
    comparisons = []
    for order, previous, current in zip(orders[1:], arrays, arrays[1:]):
        delta = float(np.max(np.abs(current-previous)))
        scale = float(np.max(np.abs(current)))
        bound = absolute_floor + relative_target*scale
        comparisons.append({"order": order, "max_absolute_change": delta, "reference_scale": scale,
                            "acceptance_bound": bound, "status": "PASS" if delta <= bound else "FAIL"})
    return {"observable": observable, "unit": unit, "orders": list(orders),
            "values": [v.tolist() for v in arrays], "relative_target": relative_target, "absolute_floor": absolute_floor,
            "reference": "next refined quadrature on the same specimen/wave/recording grid",
            "comparisons": comparisons, "status": "PASS" if all(r["status"] == "PASS" for r in comparisons[-2:]) else "INCONCLUSIVE",
            "scope": "fixture-local source/energy quadrature only; does not certify spatial-grid, phonon or material convergence"}
