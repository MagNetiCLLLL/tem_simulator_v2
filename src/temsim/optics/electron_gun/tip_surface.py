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
    current_na: float | None
    cap_half_angle_deg: float
    normal_mean_energy_ev: float
    tangential_mean_energy_ev: float
    flux_electrons_per_nm2_s: float | None = None
    maximum_angle_deg: float = 90.0
    energy_distribution: str = "normal_tangential_exponential"
    kinetic_mean_ev: float = 0.3
    kinetic_sigma_ev: float = 0.1

    def validate(self, geometry, *, coherent=False):
        if self.flux_electrons_per_nm2_s is None:
            _positive(self.current_na, "Surface current", allow_zero=True)
        else:
            if self.current_na is not None:
                raise ValueError("Choose surface flux density or total current, not both")
            _positive(self.flux_electrons_per_nm2_s, "Electron flux density", allow_zero=True)
        if isinstance(self.maximum_angle_deg, bool) or not math.isfinite(self.maximum_angle_deg) or not 0 <= self.maximum_angle_deg <= 90:
            raise ValueError("Maximum emission angle from the local normal must be in [0, 90] degrees")
        if self.energy_distribution not in {"normal_tangential_exponential", "gamma", "monoenergetic"}:
            raise ValueError("Unknown particle energy distribution")
        _positive(self.kinetic_mean_ev, "Mean kinetic energy")
        _positive(self.kinetic_sigma_ev, "Kinetic energy sigma", allow_zero=True)
        if self.energy_distribution == "gamma" and self.kinetic_sigma_ev == 0:
            raise ValueError("Gamma energy requires a positive sigma; use monoenergetic for zero width")
        if not coherent:
            _positive(self.normal_mean_energy_ev, "Mean normal energy")
            _positive(self.tangential_mean_energy_ev, "Mean tangential energy", allow_zero=True)
        if (not math.isfinite(self.cap_half_angle_deg)
                or not 0 < self.cap_half_angle_deg < 90 - geometry.cone_half_angle_deg):
            raise ValueError("Emission patch must lie strictly inside the spherical tip cap")
        return self

    @property
    def mean_energy_ev(self):
        if self.energy_distribution != "normal_tangential_exponential":
            return self.kinetic_mean_ev
        a, b = self.normal_mean_energy_ev, self.tangential_mean_energy_ev
        if self.maximum_angle_deg == 90:
            return a + b
        c = math.tan(math.radians(self.maximum_angle_deg))**2
        return a + (1+c)*a*b/(a*c+b) if b else a

    @property
    def energy_sigma_ev(self):
        if self.energy_distribution != "normal_tangential_exponential":
            return self.kinetic_sigma_ev if self.energy_distribution == "gamma" else 0.0
        return math.hypot(self.normal_mean_energy_ev, self.mean_energy_ev-self.normal_mean_energy_ev)

    def area_nm2(self, geometry):
        from temsim.optics.electron_gun.tip_patch import patch_dimensions
        return patch_dimensions(geometry, self.cap_half_angle_deg)["surface_area_nm2"]

    def total_current_na(self, geometry):
        if self.flux_electrons_per_nm2_s is None:
            return self.current_na
        return self.flux_electrons_per_nm2_s * self.area_nm2(geometry) * 1.602176634e-10


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

    @property
    def current_na(self):
        return self.emission.total_current_na(self.geometry)

    def validate(self):
        if self.schema != SCHEMA or self.potential_reference != "final_accelerating_anode_ground_0V":
            raise ValueError("Unsupported tip surface model or voltage reference")
        self.geometry.validate()
        self.emission.validate(self.geometry, coherent=self.coherence is not None)
        _positive(self.current_na, "Derived surface current", allow_zero=True)
        self.field_numerics.validate()
        if self.coherence is not None:
            self.coherence.validate()
        return self

    def to_dict(self):
        result = asdict(self)
        if result["coherence"] is None:
            result.pop("coherence")
        result["emission"] = {k: v for k, v in result["emission"].items() if v is not None}
        return result

    @classmethod
    def from_dict(cls, data):
        try:
            row = dict(data)
            row["geometry"] = TipGeometry(**row["geometry"])
            row["emission"] = SurfaceEmission(**{"current_na": None, **row["emission"]})
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
    if value is None or isinstance(value, bool) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def emit_surface(model, count):
    """Positions [m], directions, local energies [eV], equal outgoing-flux weights.

    Uniform area on a spherical cap. Normal and tangential energy have
    exponential laws conditioned on the local angular limit, or a specified
    positive total-energy law with uniform solid-angle directions. This specifies
    one consistent energy/direction distribution. No extra energy FWHM,
    virtual-source size or second downstream source is applied.
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
    if p.energy_distribution == "normal_tangential_exponential":
        a, bmean = p.normal_mean_energy_ev, p.tangential_mean_energy_ev
        normal_energy = -a * np.log1p(-en)
        tangent_energy = -bmean * np.log1p(-et)
        if p.maximum_angle_deg < 90 and bmean > 0:
            # Exact conditional exponential law T <= tan(theta_max)^2 N.
            # N = S + U, T = c U, with independent exponential S and U.
            c = math.tan(math.radians(p.maximum_angle_deg))**2
            u_energy = -a*bmean/(a*c+bmean) * np.log1p(-et)
            normal_energy += u_energy
            tangent_energy = c*u_energy
    else:
        energy = np.full(int(count), p.kinetic_mean_ev)
        if p.energy_distribution == "gamma":
            from scipy.special import gammaincinv
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                shape = np.square(np.float64(p.kinetic_mean_ev)/p.kinetic_sigma_ev)
            if not np.isfinite(shape) or shape <= 0:
                raise ValueError("Gamma energy distribution is unresolved at this mean and sigma")
            energy = gammaincinv(shape, en) * p.kinetic_sigma_ev**2/p.kinetic_mean_ev
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
            raise ValueError("Positive source energies are unresolved at this mean and sigma")
        cosine = 1-et*(1-math.cos(math.radians(p.maximum_angle_deg)))
        normal_energy, tangent_energy = energy*cosine**2, energy*(1-cosine**2)
    tangent = np.cos(2*np.pi*b)[:, None]*tangent1 + np.sin(2*np.pi*b)[:, None]*tangent2
    direction = np.sqrt(normal_energy)[:, None]*normals + np.sqrt(tangent_energy)[:, None]*tangent
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return positions, direction, normal_energy + tangent_energy, np.full(int(count), 1/int(count))


def surface_bundle(model, count, *, support_probes=0):
    """Retain the curved launch surface and full direction, including dz < 0."""
    from temsim.optics.electron_gun.base import EmissionBundle
    p, d, energy, weight = emit_surface(model, count)
    if support_probes:
        if support_probes not in (1, 33) or count <= support_probes:
            raise ValueError("Surface tuning requires 1 or 33 diagnostic probes plus emission samples")
        from temsim.optics.electron_gun.tip_patch import sample_cap_frame
        interior = count-support_probes
        # Diagnostic paths originate on the actual tip and traverse every gun
        # field/aperture. Zero weight prevents an axis probe inventing current
        # through an acceptance too small for the ordinary emission quadrature.
        area = np.zeros(support_probes)
        azimuth = np.zeros(support_probes)
        if support_probes == 33:
            area[:-1] = np.repeat([1., 1e-4, 1e-8, 1e-12], 8)
            azimuth[:-1] = np.tile(np.arange(8)/8, 4)
        p[interior:], d[interior:], _, _ = sample_cap_frame(
            model.geometry, model.emission.cap_half_angle_deg, area, azimuth)
        energy[interior:] = model.emission.mean_energy_ev
        weight[:interior] = 1/interior
        weight[interior:] = 0.
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
