"""Versioned, prescribed emission from a curved physical tip surface.

The default is classical outgoing flux; an explicit coherent reservoir is
optional. Neither predicts metal tunnelling. Geometry comes from a TOML record.
Kinetic energy is local to the surface; voltage zero is the final anode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib

import numpy as np

from temsim.paths import CONFIG_ROOT

SCHEMA = "grounded-cold-feg-surface-v1"
REFERENCE = CONFIG_ROOT / "sources" / "cold_feg_tip.toml"


def reference_path_for_gun(gun):
    return CONFIG_ROOT / getattr(gun.emitter, "_tip_reference_file", "sources/cold_feg_tip.toml")


@dataclass(frozen=True)
class TipGeometry:
    shape: str
    material: str
    apex_radius_nm: float
    cone_half_angle_deg: float
    shank_length_um: float

    def validate(self):
        if (self.shape != "spherical_cap_tangent_cone" or not isinstance(self.material, str)
                or not self.material.strip()):
            raise ValueError("Unknown tip geometry or missing material")
        _positive(self.apex_radius_nm, "Apex radius")
        _positive(self.shank_length_um, "Shank length")
        if not math.isfinite(self.cone_half_angle_deg) or not 0 < self.cone_half_angle_deg < 90:
            raise ValueError("Cone half-angle must be between 0 and 90 degrees")
        if self.shank_length_um * 1e3 <= self.apex_radius_nm:
            raise ValueError("Tip shank must extend beyond its spherical cap")
        return self

    def radius_m(self, z_m):
        """Metal radius at z relative to the apex (positive downstream)."""
        z = np.asarray(z_m, dtype=float)
        radius = self.apex_radius_nm * 1e-9
        angle = math.radians(self.cone_half_angle_deg)
        join_z = -radius * (1 - math.sin(angle))
        cap = np.sqrt(np.maximum(0., radius**2 - (z + radius)**2))
        cone = radius * math.cos(angle) + (join_z - z) * math.tan(angle)
        return np.where(z > 0, 0., np.where(z >= join_z, cap, cone))


@dataclass(frozen=True)
class SurfaceEmission:
    current_na: float
    cap_half_angle_deg: float
    normal_mean_energy_ev: float
    tangential_mean_energy_ev: float

    def validate(self, geometry, *, coherent=False):
        _positive(self.current_na, "Surface current", allow_zero=True)
        if not coherent:
            _positive(self.normal_mean_energy_ev, "Mean normal energy")
            _positive(self.tangential_mean_energy_ev, "Mean tangential energy", allow_zero=True)
        if (not math.isfinite(self.cap_half_angle_deg)
                or not 0 < self.cap_half_angle_deg < 90 - geometry.cone_half_angle_deg):
            raise ValueError("Emission patch must lie strictly inside the spherical tip cap")
        return self

    @property
    def mean_energy_ev(self):
        return self.normal_mean_energy_ev + self.tangential_mean_energy_ev

    @property
    def energy_sigma_ev(self):
        return math.hypot(self.normal_mean_energy_ev, self.tangential_mean_energy_ev)

    def area_nm2(self, geometry):
        from temsim.optics.electron_gun.tip_patch import patch_dimensions
        return patch_dimensions(geometry, self.cap_half_angle_deg)["surface_area_nm2"]


@dataclass(frozen=True)
class TipFieldNumerics:
    radial_nodes: int
    axial_nodes: int
    apex_cells_per_radius: int
    outer_radius_factor: float
    linear_residual_tolerance: float
    accelerator_ring_thickness_mm: float

    def validate(self):
        for name, lo, hi in (("radial_nodes", 32, 1024), ("axial_nodes", 64, 2048),
                             ("apex_cells_per_radius", 4, 200)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if not math.isfinite(self.outer_radius_factor) or self.outer_radius_factor <= 1:
            raise ValueError("Outer numerical boundary must enclose the electrodes")
        if not 0 < self.linear_residual_tolerance <= 1e-6:
            raise ValueError("Laplace residual tolerance must be in (0, 1e-6]")
        _positive(self.accelerator_ring_thickness_mm, "Accelerator ring thickness")
        return self


@dataclass(frozen=True)
class SurfaceCoherence:
    """An explicitly prescribed, axisymmetric coherent emitting reservoir.

    One spatial mode per energy; different energies form an incoherent mixture.
    Energies refer to the tip potential. Classical normal/tangential energies
    are not additional constraints on this quantum boundary.
    """
    mean_energy_ev: float = 0.3
    energy_rms_ev: float = 0.1
    edge_phase_rad: float = 0.0
    model: str = "coherent-cap-reservoir-v1"

    def validate(self):
        if self.model != "coherent-cap-reservoir-v1":
            raise ValueError("Unknown coherent surface boundary model")
        _positive(self.mean_energy_ev, "Coherent surface mean energy")
        _positive(self.energy_rms_ev, "Coherent surface energy RMS", allow_zero=True)
        if not math.isfinite(self.edge_phase_rad):
            raise ValueError("Surface edge phase must be finite")
        return self

    def energy_quadrature(self, count):
        """Positive gamma-law quadrature, or a single monochromatic component."""
        from scipy.linalg import eigh_tridiagonal
        self.validate()
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("Coherent energy samples must be an integer in [1, 64]")
        if self.energy_rms_ev == 0:
            return np.array([self.mean_energy_ev]), np.ones(1)
        if count < 2:
            raise ValueError("A nonzero energy width needs at least two energy samples")
        shape = (self.mean_energy_ev/self.energy_rms_ev)**2
        if not math.isfinite(shape) or shape <= 0:
            raise ValueError("This gamma quadrature needs a better-conditioned narrow-spectrum rule; no width was changed")
        # Orthonormal Laguerre Jacobi matrix (Golub-Welsch), already in eV.
        # Avoid Gamma(shape) overflow and division by very small gamma scales.
        scale = self.energy_rms_ev**2/self.mean_energy_ev
        n = np.arange(count, dtype=float)
        diagonal = self.mean_energy_ev + 2*n*scale
        off = np.sqrt(n[1:])*np.sqrt(self.energy_rms_ev**2+(n[1:]-1)*scale**2)
        nodes, vectors = eigh_tridiagonal(diagonal, off)
        weights = vectors[0]**2
        if np.any(nodes <= 0) or not np.all(np.isfinite(nodes)):
            raise ValueError("The positive source energy law is unresolved in floating point")
        return nodes, weights/weights.sum()


@dataclass(frozen=True)
class TipSurfaceModel:
    geometry: TipGeometry
    emission: SurfaceEmission
    field_numerics: TipFieldNumerics
    schema: str = SCHEMA
    status: str = "idealised_reference_not_calibrated"
    potential_reference: str = "final_accelerating_anode_ground_0V"
    geometry_reference: str = ""
    coherence: SurfaceCoherence | None = None

    def validate(self):
        if self.schema != SCHEMA or self.potential_reference != "final_accelerating_anode_ground_0V":
            raise ValueError("Unsupported tip surface model or voltage reference")
        self.geometry.validate()
        self.emission.validate(self.geometry, coherent=self.coherence is not None)
        self.field_numerics.validate()
        if self.coherence is not None:
            self.coherence.validate()
        return self

    def to_dict(self):
        result = asdict(self)
        if result["coherence"] is None:
            result.pop("coherence")
        return result

    @classmethod
    def from_dict(cls, data):
        try:
            row = dict(data)
            row["geometry"] = TipGeometry(**row["geometry"])
            row["emission"] = SurfaceEmission(**row["emission"])
            row["field_numerics"] = TipFieldNumerics(**row["field_numerics"])
            if row.get("coherence") is not None:
                row["coherence"] = SurfaceCoherence(**row["coherence"])
            return cls(**row).validate()
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid tip surface model: {error}") from error


def load_tip_surface_reference(path: str | Path = REFERENCE):
    with Path(path).open("rb") as stream:
        return TipSurfaceModel.from_dict(tomllib.load(stream))


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def emit_surface(model, count):
    """Positions [m], directions, local energies [eV], equal outgoing-flux weights.

    Uniform area on a spherical cap. Normal and tangential energy have
    exponential laws; azimuth is uniform in the tangent plane. This specifies
    one consistent energy/direction distribution without independent cutoffs,
    FWHM, virtual-source size or a second downstream source.
    """
    from temsim.optics.electron_gun.emitter import _halton_dimensions
    model.validate()
    if model.coherence is not None:
        raise ValueError("The coherent surface boundary must be propagated as a wave; classical ray sampling is not its phase-space distribution")
    if isinstance(count, bool) or int(count) != count or count < 9:
        raise ValueError("Surface emission requires at least 9 samples")
    u, a, en, et, b = _halton_dimensions(int(count), (2, 3, 5, 7, 11))
    p = model.emission
    from temsim.optics.electron_gun.tip_patch import sample_cap_frame
    positions, normals, tangent1, tangent2 = sample_cap_frame(model.geometry, p.cap_half_angle_deg, u, a)
    normal_energy = -p.normal_mean_energy_ev * np.log1p(-en)
    tangent_energy = -p.tangential_mean_energy_ev * np.log1p(-et)
    tangent = np.cos(2*np.pi*b)[:, None]*tangent1 + np.sin(2*np.pi*b)[:, None]*tangent2
    direction = np.sqrt(normal_energy)[:, None]*normals + np.sqrt(tangent_energy)[:, None]*tangent
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return positions, direction, normal_energy + tangent_energy, np.full(int(count), 1/int(count))


def surface_bundle(model, count):
    """Retain the curved launch surface and full direction, including dz < 0."""
    from temsim.optics.electron_gun.base import EmissionBundle
    p, d, energy, weight = emit_surface(model, count)
    # Slope is only a historical display coordinate, not the transport state.
    # An exactly transverse trajectory has no finite slope and is shown as NaN.
    slopes = np.divide(d[:, :2], d[:, 2, None], out=np.full_like(d[:, :2], np.nan),
                       where=d[:, 2, None] != 0)
    bundle = EmissionBundle(p[:, 0], p[:, 1], slopes[:, 0], slopes[:, 1],
                            energy-model.emission.mean_energy_ev, weight, np.arange(count))
    # Extra immutable-record attributes avoid changing historical bundle schema.
    object.__setattr__(bundle, "surface_position_m", p)
    object.__setattr__(bundle, "surface_direction", d)
    object.__setattr__(bundle, "surface_energy_ev", energy)
    return bundle
