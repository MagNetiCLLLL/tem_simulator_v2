"""Finite virtual specimens with explicit, probability-conserving channels.

User angular channels are phenomenological unless ``kind`` is
``physical_rutherford``.  The latter uses a screened relativistic Rutherford
approximation, not a Mott cross section.  Absolute channel probabilities are
never renormalised; their remainder is the direct beam and any absorption row
is reported as lost intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


PARAXIAL_VIRTUAL_MAX_MRAD = 500.0
CLASSICAL_ELECTRON_RADIUS_M = 2.8179403262e-15
ELECTRON_REST_ENERGY_EV = 510_998.95069


@dataclass(frozen=True, slots=True)
class VirtualScatteringBranch:
    name: str
    kick_x_rad: float
    kick_y_rad: float
    relative_weight: float
    kind: str


@dataclass(frozen=True, slots=True)
class VirtualInteractionComponent:
    name: str
    kind: str
    probability: float
    parameters: dict
    approximation: str | None = None


@dataclass(frozen=True, slots=True)
class VirtualAngularDistribution:
    angle_x_mrad: np.ndarray
    angle_y_mrad: np.ndarray
    probabilities: np.ndarray
    kinds: tuple[str, ...]
    direct_probability: float
    absorbed_probability: float
    components: tuple[VirtualInteractionComponent, ...]

    @property
    def scattered_probability(self) -> float:
        return float(np.sum(self.probabilities))

    @property
    def total_probability(self) -> float:
        return (
            self.direct_probability
            + self.scattered_probability
            + self.absorbed_probability
        )


def _finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _probability(name: str, value: float) -> float:
    probability = _finite(name, value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return probability


def relativistic_beta_gamma(beam_energy_kv: float) -> tuple[float, float]:
    kinetic_ev = _finite("Beam energy", beam_energy_kv) * 1.0e3
    if kinetic_ev <= 0.0:
        raise ValueError("Beam energy must be positive.")
    gamma = 1.0 + kinetic_ev / ELECTRON_REST_ENERGY_EV
    beta_squared = 1.0 - 1.0 / (gamma * gamma)
    return math.sqrt(max(beta_squared, 0.0)), gamma


def screened_relativistic_rutherford_dcs_m2_sr(
    theta_rad,
    *,
    atomic_number: int,
    beam_energy_kv: float,
    screening_angle_mrad: float,
):
    """Approximate screened relativistic Rutherford differential cross section.

    ``theta`` is the laboratory scattering angle.  Screening replaces the
    small-angle singularity by adding ``sin(theta_s/2)^2`` to the denominator.
    Spin, exchange, finite nuclear size and full Mott phase shifts are omitted.
    """

    z = int(atomic_number)
    if not 1 <= z <= 118:
        raise ValueError("Rutherford atomic number must be between 1 and 118.")
    screening_mrad = _finite("Rutherford screening angle", screening_angle_mrad)
    if screening_mrad <= 0.0:
        raise ValueError("Rutherford screening angle must be positive.")
    theta = np.asarray(theta_rad, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any(theta < 0.0) or np.any(theta > math.pi):
        raise ValueError("Rutherford scattering angles must be in [0, pi].")
    beta, gamma = relativistic_beta_gamma(beam_energy_kv)
    half_sine_squared = np.sin(0.5 * theta) ** 2
    screening_squared = math.sin(0.5 * screening_mrad * 1.0e-3) ** 2
    denominator = (half_sine_squared + screening_squared) ** 2
    prefactor = (
        z * CLASSICAL_ELECTRON_RADIUS_M / (2.0 * gamma * beta * beta)
    ) ** 2
    recoil_spin_factor = np.maximum(1.0 - beta * beta * half_sine_squared, 0.0)
    return prefactor * recoil_spin_factor / denominator


def integrate_screened_rutherford_cross_section_m2(
    *,
    atomic_number: int,
    beam_energy_kv: float,
    screening_angle_mrad: float,
    minimum_angle_mrad: float,
    maximum_angle_mrad: float,
    quadrature_points: int = 2048,
) -> float:
    minimum = _finite("Rutherford minimum angle", minimum_angle_mrad) * 1.0e-3
    maximum = _finite("Rutherford maximum angle", maximum_angle_mrad) * 1.0e-3
    if minimum < 0.0 or maximum <= minimum or maximum > math.pi:
        raise ValueError("Rutherford angular bounds must satisfy 0 <= min < max <= pi.")
    count = int(quadrature_points)
    if count < 32:
        raise ValueError("Rutherford integration needs at least 32 quadrature points.")
    # A geometric grid resolves a narrow screened peak while retaining the
    # exact 2*pi*sin(theta) solid-angle Jacobian.
    floor = max(minimum, 1.0e-12)
    theta = np.geomspace(floor, maximum, count)
    if minimum == 0.0:
        theta = np.r_[0.0, theta]
    dcs = screened_relativistic_rutherford_dcs_m2_sr(
        theta,
        atomic_number=atomic_number,
        beam_energy_kv=beam_energy_kv,
        screening_angle_mrad=screening_angle_mrad,
    )
    return float(np.trapezoid(dcs * (2.0 * math.pi * np.sin(theta)), theta))


def physical_screened_rutherford_probability(
    *,
    atomic_number: int,
    areal_density_atoms_nm2: float,
    beam_energy_kv: float,
    screening_angle_mrad: float,
    minimum_angle_mrad: float,
    maximum_angle_mrad: float,
) -> tuple[float, float]:
    density = _finite("Rutherford areal density", areal_density_atoms_nm2)
    if density < 0.0:
        raise ValueError("Rutherford areal density cannot be negative.")
    cross_section = integrate_screened_rutherford_cross_section_m2(
        atomic_number=atomic_number,
        beam_energy_kv=beam_energy_kv,
        screening_angle_mrad=screening_angle_mrad,
        minimum_angle_mrad=minimum_angle_mrad,
        maximum_angle_mrad=maximum_angle_mrad,
    )
    optical_depth = density * 1.0e18 * cross_section
    return -math.expm1(-optical_depth), cross_section


def _legacy_interactions(sample) -> list[dict]:
    """Migrate the original two relative sliders into absolute rows."""

    diffraction = max(
        _finite(
            "Virtual diffraction relative weight",
            getattr(sample, "virtual_diffraction_relative_weight", 1.0),
        ),
        0.0,
    )
    diffuse = max(
        _finite(
            "Virtual scattering relative weight",
            getattr(sample, "virtual_scattering_relative_weight", 0.2),
        ),
        0.0,
    )
    total = 1.0 + 2.0 * diffraction + diffuse
    return [
        {
            "name": "Migrated diffraction pair",
            "kind": "diffraction_spots",
            "enabled": diffraction > 0.0,
            "probability": 2.0 * diffraction / total,
            "angle_mrad": float(getattr(sample, "virtual_diffraction_angle_mrad", 5.0)),
            "azimuth_deg": float(getattr(sample, "virtual_diffraction_azimuth_deg", 0.0)),
            "spot_count": 2,
        },
        {
            "name": "Migrated diffuse ring",
            "kind": "diffuse_ring",
            "enabled": diffuse > 0.0,
            "probability": diffuse / total,
            "angle_mrad": float(getattr(sample, "virtual_scattering_angle_mrad", 20.0)),
            "width_mrad": 0.0,
            "azimuth_samples": int(getattr(sample, "virtual_scattering_azimuth_samples", 16)),
        },
    ]


def legacy_virtual_interaction_rows(sample) -> list[dict]:
    """Public V1-profile migration helper returning independent row tables."""

    return [dict(row) for row in _legacy_interactions(sample)]


def uses_legacy_virtual_controls(sample) -> bool:
    """Return true when a pre-table scalar was edited from its V64 default."""

    defaults = {
        "virtual_diffraction_angle_mrad": 5.0,
        "virtual_diffraction_azimuth_deg": 0.0,
        "virtual_diffraction_relative_weight": 1.0,
        "virtual_scattering_angle_mrad": 20.0,
        "virtual_scattering_relative_weight": 0.2,
        "virtual_scattering_azimuth_samples": 16,
    }
    return any(
        not math.isclose(
            float(getattr(sample, name, default)),
            float(default),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for name, default in defaults.items()
    )


def _legacy_scattering_branches(sample) -> tuple[VirtualScatteringBranch, ...]:
    diffraction_mrad = _finite(
        "Virtual diffraction angle",
        sample.virtual_diffraction_angle_mrad,
    )
    scattering_mrad = _finite(
        "Virtual scattering angle",
        sample.virtual_scattering_angle_mrad,
    )
    diffraction_weight = _finite(
        "Virtual diffraction relative weight",
        sample.virtual_diffraction_relative_weight,
    )
    scattering_weight = _finite(
        "Virtual scattering relative weight",
        sample.virtual_scattering_relative_weight,
    )
    if not 0.0 <= diffraction_mrad <= 200.0:
        raise ValueError(
            "Virtual diffraction angle must be between 0 and 200 mrad for the paraxial ray model."
        )
    if not 0.0 <= scattering_mrad <= 200.0:
        raise ValueError(
            "Virtual scattering angle must be between 0 and 200 mrad for the paraxial ray model."
        )
    if diffraction_weight < 0.0 or scattering_weight < 0.0:
        raise ValueError("Virtual scattering relative weights cannot be negative.")
    count = int(sample.virtual_scattering_azimuth_samples)
    if not 4 <= count <= 128:
        raise ValueError("Virtual isotropic scattering requires 4 to 128 azimuth samples.")
    azimuth = math.radians(_finite("Virtual diffraction azimuth", sample.virtual_diffraction_azimuth_deg))
    branches = [VirtualScatteringBranch("000", 0.0, 0.0, 1.0, "transmitted")]
    if diffraction_mrad > 0.0 and diffraction_weight > 0.0:
        magnitude = diffraction_mrad * 1.0e-3
        x = magnitude * math.cos(azimuth)
        y = magnitude * math.sin(azimuth)
        branches.extend(
            (
                VirtualScatteringBranch("virtual_+g", x, y, diffraction_weight, "diffraction_spot"),
                VirtualScatteringBranch("virtual_-g", -x, -y, diffraction_weight, "diffraction_spot"),
            )
        )
    if scattering_mrad > 0.0 and scattering_weight > 0.0:
        magnitude = scattering_mrad * 1.0e-3
        weight = scattering_weight / count
        branches.extend(
            VirtualScatteringBranch(
                f"virtual_ring_{index + 1:03d}",
                magnitude * math.cos(2.0 * math.pi * index / count),
                magnitude * math.sin(2.0 * math.pi * index / count),
                weight,
                "isotropic_ring",
            )
            for index in range(count)
        )
    return tuple(branches)


def resolve_virtual_interactions(
    sample,
    *,
    beam_energy_kv: float,
) -> tuple[VirtualInteractionComponent, ...]:
    rows = getattr(sample, "virtual_interactions", None)
    rows = (
        _legacy_interactions(sample)
        if uses_legacy_virtual_controls(sample)
        else list(rows)
        if rows
        else _legacy_interactions(sample)
    )
    result: list[VirtualInteractionComponent] = []
    total = 0.0
    aliases = {
        "ring": "diffuse_ring",
        "gaussian": "gaussian_diffuse",
        "arbitrary": "arbitrary_angular",
        "screened_power_law": "user_screened_power_law",
        "rutherford": "physical_rutherford",
        "lost": "absorption",
    }
    supported = {
        "diffraction_spots",
        "diffuse_ring",
        "gaussian_diffuse",
        "arbitrary_angular",
        "user_screened_power_law",
        "physical_rutherford",
        "absorption",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError("Each virtual interaction must be a table.")
        if not bool(raw.get("enabled", True)):
            continue
        kind = aliases.get(
            str(raw.get("kind", "diffuse_ring")).strip().lower(),
            str(raw.get("kind", "diffuse_ring")).strip().lower(),
        )
        if kind not in supported:
            raise ValueError(
                f"Virtual interaction {index + 1}: unsupported kind {kind!r}."
            )
        parameters = dict(raw)
        approximation = None
        if kind == "physical_rutherford":
            probability, cross_section = physical_screened_rutherford_probability(
                atomic_number=int(raw.get("atomic_number", 14)),
                areal_density_atoms_nm2=float(raw.get("areal_density_atoms_nm2", 0.0)),
                beam_energy_kv=beam_energy_kv,
                screening_angle_mrad=float(raw.get("screening_angle_mrad", 5.0)),
                minimum_angle_mrad=float(raw.get("minimum_angle_mrad", 0.0)),
                maximum_angle_mrad=float(raw.get("maximum_angle_mrad", 200.0)),
            )
            parameters["integrated_cross_section_m2"] = cross_section
            approximation = (
                "screened relativistic Rutherford; not Mott; independent single-event probability"
            )
        else:
            probability = _probability(
                f"Virtual interaction {index + 1} probability",
                raw.get("probability", 0.0),
            )
        total += probability
        result.append(
            VirtualInteractionComponent(
                name=str(raw.get("name", f"Interaction {index + 1}")),
                kind=kind,
                probability=probability,
                parameters=parameters,
                approximation=approximation,
            )
        )
    if total > 1.0 + 1.0e-12:
        raise ValueError(
            "Enabled virtual interaction and absorption probabilities sum to "
            f"{total:.6g}, above one. Probabilities are absolute and are not normalised."
        )
    return tuple(result)


def _polar_points(theta_mrad, azimuth_count, *, azimuth_offset_deg=0.0):
    count = int(azimuth_count)
    if not 1 <= count <= 4096:
        raise ValueError("Angular-channel azimuth samples must be between 1 and 4096.")
    phi = np.radians(float(azimuth_offset_deg)) + np.arange(count) * 2.0 * math.pi / count
    theta = np.asarray(theta_mrad, dtype=float)
    if theta.ndim == 0:
        theta = np.full(count, float(theta))
    if theta.shape != (count,):
        raise ValueError("Angular radius and azimuth sample counts do not match.")
    return theta * np.cos(phi), theta * np.sin(phi)


def _component_points(component: VirtualInteractionComponent):
    raw = component.parameters
    kind = component.kind
    if kind == "diffraction_spots":
        angle = _finite("Diffraction angle", raw.get("angle_mrad", 0.0))
        count = int(raw.get("spot_count", 2))
        x, y = _polar_points(
            angle,
            count,
            azimuth_offset_deg=float(raw.get("azimuth_deg", 0.0)),
        )
        weights = np.full(count, 1.0 / count)
    elif kind == "diffuse_ring":
        angle = _finite("Diffuse-ring angle", raw.get("angle_mrad", 0.0))
        width = _finite("Diffuse-ring width", raw.get("width_mrad", 0.0))
        count = int(raw.get("azimuth_samples", 64))
        # Deterministic symmetric radial offsets approximate a Gaussian ring
        # while keeping regression results seed independent.
        offsets = np.resize(np.asarray((-1.5, -0.5, 0.5, 1.5)), count)
        radius = np.maximum(angle + offsets * 0.5 * width, 0.0)
        x, y = _polar_points(radius, count, azimuth_offset_deg=float(raw.get("azimuth_deg", 0.0)))
        weights = np.exp(-0.5 * offsets**2)
        weights /= weights.sum()
    elif kind == "gaussian_diffuse":
        sigma = _finite("Gaussian diffuse sigma", raw.get("sigma_mrad", 10.0))
        if sigma <= 0.0:
            raise ValueError("Gaussian diffuse sigma must be positive.")
        rings = int(raw.get("radial_samples", 16))
        azimuths = int(raw.get("azimuth_samples", 64))
        if not 2 <= rings <= 256:
            raise ValueError("Gaussian radial samples must be between 2 and 256.")
        radius = np.linspace(0.0, min(4.0 * sigma, PARAXIAL_VIRTUAL_MAX_MRAD), rings + 1)[1:]
        theta = np.repeat(radius, azimuths)
        phi = np.tile(np.arange(azimuths) * 2.0 * math.pi / azimuths, rings)
        x = theta * np.cos(phi)
        y = theta * np.sin(phi)
        dr = radius[1] - radius[0]
        radial_weight = np.exp(-0.5 * (radius / sigma) ** 2) * radius * dr
        weights = np.repeat(radial_weight, azimuths)
        weights /= weights.sum()
    elif kind == "arbitrary_angular":
        lower = _finite("Arbitrary minimum angle", raw.get("minimum_angle_mrad", 0.0))
        upper = _finite("Arbitrary maximum angle", raw.get("maximum_angle_mrad", 20.0))
        if lower < 0.0 or upper <= lower:
            raise ValueError("Arbitrary angular bounds must satisfy 0 <= min < max.")
        radial = int(raw.get("radial_samples", 32))
        azimuths = int(raw.get("azimuth_samples", 64))
        radius = np.linspace(lower, upper, radial)
        theta = np.repeat(radius, azimuths)
        phi0 = math.radians(float(raw.get("minimum_azimuth_deg", 0.0)))
        phi1 = math.radians(float(raw.get("maximum_azimuth_deg", 360.0)))
        phi = np.tile(np.linspace(phi0, phi1, azimuths, endpoint=False), radial)
        x, y = theta * np.cos(phi), theta * np.sin(phi)
        weights = np.repeat(np.sin(radius * 1.0e-3), azimuths)
        weights = np.maximum(weights, 1.0e-15)
        weights /= weights.sum()
    elif kind in {"user_screened_power_law", "physical_rutherford"}:
        lower = max(_finite("Scattering minimum angle", raw.get("minimum_angle_mrad", 0.0)), 0.0)
        upper = _finite("Scattering maximum angle", raw.get("maximum_angle_mrad", 200.0))
        if upper <= lower or upper > PARAXIAL_VIRTUAL_MAX_MRAD:
            raise ValueError(
                f"Scattering bounds must satisfy min < max <= {PARAXIAL_VIRTUAL_MAX_MRAD:g} mrad."
            )
        radial = int(raw.get("radial_samples", 128))
        azimuths = int(raw.get("azimuth_samples", 64))
        radius = np.geomspace(max(lower, 1.0e-5), upper, radial)
        if kind == "physical_rutherford":
            density = screened_relativistic_rutherford_dcs_m2_sr(
                radius * 1.0e-3,
                atomic_number=int(raw.get("atomic_number", 14)),
                beam_energy_kv=float(raw["beam_energy_kv"]),
                screening_angle_mrad=float(raw.get("screening_angle_mrad", 5.0)),
            )
        else:
            screening = _finite("Power-law screening angle", raw.get("screening_angle_mrad", 5.0))
            exponent = _finite("Power-law exponent", raw.get("exponent", 2.0))
            if screening <= 0.0 or exponent <= 0.0:
                raise ValueError("Power-law screening angle and exponent must be positive.")
            density = (radius * radius + screening * screening) ** (-exponent)
        # Radial bin width and exact sin(theta) solid-angle Jacobian.
        dr = np.gradient(radius * 1.0e-3)
        radial_weight = density * np.sin(radius * 1.0e-3) * dr
        theta = np.repeat(radius, azimuths)
        phi = np.tile(np.arange(azimuths) * 2.0 * math.pi / azimuths, radial)
        x, y = theta * np.cos(phi), theta * np.sin(phi)
        weights = np.repeat(radial_weight, azimuths)
        weights /= weights.sum()
    else:
        raise ValueError(f"{component.name}: {kind} has no angular distribution.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(weights))):
        raise ValueError(f"{component.name}: angular distribution is not finite.")
    return np.asarray(x), np.asarray(y), np.asarray(weights)


def build_virtual_angular_distribution(
    sample,
    *,
    beam_energy_kv: float,
) -> VirtualAngularDistribution:
    components = list(resolve_virtual_interactions(sample, beam_energy_kv=beam_energy_kv))
    # Retain energy in the parameters so physical quadrature is reproducible
    # without a hidden global state.
    components = [
        VirtualInteractionComponent(
            item.name,
            item.kind,
            item.probability,
            ({**item.parameters, "beam_energy_kv": float(beam_energy_kv)} if item.kind == "physical_rutherford" else item.parameters),
            item.approximation,
        )
        for item in components
    ]
    x_values = []
    y_values = []
    probabilities = []
    kinds: list[str] = []
    absorbed = 0.0
    for component in components:
        if component.kind == "absorption":
            absorbed += component.probability
            continue
        if component.probability <= 0.0:
            continue
        x, y, weights = _component_points(component)
        x_values.append(x)
        y_values.append(y)
        probabilities.append(weights * component.probability)
        kinds.extend([component.kind] * x.size)
    total = absorbed + sum(component.probability for component in components if component.kind != "absorption")
    direct = max(1.0 - total, 0.0)
    x = np.concatenate(x_values) if x_values else np.empty(0)
    y = np.concatenate(y_values) if y_values else np.empty(0)
    weights = np.concatenate(probabilities) if probabilities else np.empty(0)
    for array in (x, y, weights):
        array.setflags(write=False)
    result = VirtualAngularDistribution(
        angle_x_mrad=x,
        angle_y_mrad=y,
        probabilities=weights,
        kinds=tuple(kinds),
        direct_probability=direct,
        absorbed_probability=absorbed,
        components=tuple(components),
    )
    if not math.isclose(result.total_probability, 1.0, rel_tol=0.0, abs_tol=2.0e-10):
        raise RuntimeError("Virtual angular distribution failed probability conservation.")
    return result


def _load_density_map(path_value: str) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Virtual density map does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix in {".png", ".tif", ".tiff"}:
        try:
            import imageio.v3 as iio

            array = iio.imread(path)
        except Exception:
            from PIL import Image

            array = np.asarray(Image.open(path))
    else:
        raise ValueError("Virtual density maps must be NPY, PNG, TIF or TIFF files.")
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        array = np.mean(array[..., :3], axis=-1)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Virtual density map must be a finite grayscale 2-D array.")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.clip(array, 0.0, 1.0)
    return np.clip(array, 0.0, 1.0)


def _sample_map_bilinear(array, u, v):
    height, width = array.shape
    x = np.clip(np.asarray(u, dtype=float) * (width - 1), 0.0, width - 1)
    # Image row zero is displayed at the top, while laboratory +Y is up.
    y = np.clip((1.0 - np.asarray(v, dtype=float)) * (height - 1), 0.0, height - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    return (
        array[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + array[y0, x1] * fx * (1.0 - fy)
        + array[y1, x0] * (1.0 - fx) * fy
        + array[y1, x1] * fx * fy
    )


def virtual_density_at_scan(
    sample,
    scan_x_um,
    scan_y_um,
    *,
    probe_sigma_nm: float = 0.0,
) -> np.ndarray:
    """Evaluate finite virtual-sample density at every probe centre."""

    # acquire_stem_scan already applies the user scan origin.  These are
    # absolute laboratory coordinates and must not be offset a second time.
    x = np.asarray(scan_x_um, dtype=float) * 1.0e3
    y = np.asarray(scan_y_um, dtype=float) * 1.0e3
    if x.shape != y.shape or x.ndim != 2 or x.size == 0:
        raise ValueError("Virtual density needs matching non-empty 2-D scan coordinates.")
    sample_x = x - float(getattr(sample, "centre_x_nm", 0.0))
    sample_y = y - float(getattr(sample, "centre_y_nm", 0.0))
    size_x = float(getattr(sample, "size_x_nm", 0.0))
    size_y = float(getattr(sample, "size_y_nm", 0.0))
    if not all(math.isfinite(value) and value > 0.0 for value in (size_x, size_y)):
        raise ValueError("Virtual sample X/Y sizes must be finite and positive.")
    finite_sample = (np.abs(sample_x) <= 0.5 * size_x) & (np.abs(sample_y) <= 0.5 * size_y)
    regions = list(getattr(sample, "virtual_regions", ()) or ())
    if not regions:
        density = finite_sample.astype(float)
    else:
        density = np.zeros_like(x, dtype=float)
        for index, raw in enumerate(regions):
            if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
                continue
            kind = str(raw.get("kind", "rectangle")).strip().lower()
            cx = float(raw.get("centre_x_nm", 0.0))
            cy = float(raw.get("centre_y_nm", 0.0))
            sx = float(raw.get("size_x_nm", size_x))
            sy = float(raw.get("size_y_nm", size_y))
            value = _probability(
                f"Virtual region {index + 1} density",
                raw.get("density", 1.0),
            )
            if sx <= 0.0 or sy <= 0.0:
                raise ValueError(f"Virtual region {index + 1} size must be positive.")
            rotation = math.radians(float(raw.get("rotation_deg", 0.0)))
            cosine, sine = math.cos(rotation), math.sin(rotation)
            dx = x - cx
            dy = y - cy
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            inside = (np.abs(local_x) <= 0.5 * sx) & (np.abs(local_y) <= 0.5 * sy)
            if kind == "ellipse":
                inside = (local_x / (0.5 * sx)) ** 2 + (local_y / (0.5 * sy)) ** 2 <= 1.0
                contribution = value * inside
            elif kind == "rectangle":
                contribution = value * inside
            elif kind == "map":
                path = str(raw.get("map_path", "")).strip()
                if not path:
                    raise ValueError(f"Virtual region {index + 1} needs a density-map path.")
                mapped = _sample_map_bilinear(
                    _load_density_map(path),
                    local_x / sx + 0.5,
                    local_y / sy + 0.5,
                )
                contribution = value * mapped * inside
            else:
                raise ValueError(f"Virtual region {index + 1}: unsupported kind {kind!r}.")
            density = np.maximum(density, contribution)
        density *= finite_sample
    sigma = float(probe_sigma_nm)
    if (
        bool(getattr(sample, "virtual_probe_convolution_enabled", True))
        and math.isfinite(sigma)
        and sigma > 0.0
        and min(x.shape) > 1
    ):
        dx_values = np.diff(x, axis=1)
        dy_values = np.diff(y, axis=0)
        step_x = float(np.median(np.abs(dx_values[dx_values != 0.0]))) if np.any(dx_values != 0.0) else math.inf
        step_y = float(np.median(np.abs(dy_values[dy_values != 0.0]))) if np.any(dy_values != 0.0) else math.inf
        if math.isfinite(step_x) and math.isfinite(step_y) and min(step_x, step_y) > 0.0:
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(
                density,
                sigma=(sigma / step_y, sigma / step_x),
                mode="constant",
                cval=0.0,
            )
    return np.clip(density, 0.0, 1.0)


def virtual_scattering_branches(sample, beam_energy_kv: float = 300.0) -> tuple[VirtualScatteringBranch, ...]:
    """Return a compact ray-diagram approximation of absolute channels."""

    if uses_legacy_virtual_controls(sample):
        return _legacy_scattering_branches(sample)

    distribution = build_virtual_angular_distribution(
        sample,
        beam_energy_kv=beam_energy_kv,
    )
    branches = [
        VirtualScatteringBranch(
            "000",
            0.0,
            0.0,
            distribution.direct_probability,
            "transmitted",
        )
    ]
    # Ray playback does not need thousands of quadrature branches.  Preserve
    # exact total probability while representing each enabled component by at
    # most 64 weighted angular samples. STEM integration uses the full grid.
    offset = 0
    for component in distribution.components:
        if component.kind == "absorption" or component.probability <= 0.0:
            continue
        x, y, weights = _component_points(component)
        stride = max(1, int(math.ceil(x.size / 64)))
        groups = [np.arange(start, min(start + stride, x.size)) for start in range(0, x.size, stride)]
        for index, group in enumerate(groups):
            group_weight = float(np.sum(weights[group])) * component.probability
            if group_weight <= 0.0:
                continue
            normalised = weights[group] / np.sum(weights[group])
            branches.append(
                VirtualScatteringBranch(
                    f"virtual_{offset + index + 1:03d}",
                    float(np.sum(normalised * x[group])) * 1.0e-3,
                    float(np.sum(normalised * y[group])) * 1.0e-3,
                    group_weight,
                    component.kind,
                )
            )
        offset += len(groups)
    return tuple(branches)
