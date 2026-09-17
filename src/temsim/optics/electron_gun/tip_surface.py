"""Versioned, prescribed emission from a curved physical tip surface.

The default is classical outgoing flux; an explicit coherent reservoir is
optional. Neither predicts metal tunnelling. Geometry comes from a TOML record.
Kinetic energy is local to the surface; voltage zero is the final anode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from temsim import input_io
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
    # Numerical quadrature, not an additional physical angular spread.
    directions_per_position: int = 1
    # Numerical importance quadrature only; the physical law remains uniform
    # emitting area. The narrow optical acceptance must not become a new source.
    spatial_sampling: str = "uniform_area"
    # Optional numerical budget priorities for the nine FULL-cap area strata.
    # Empty retains historical equal site counts. Never physical flux weights.
    spatial_stratum_allocation: tuple[int, ...] = ()
    # Numerical CDF refinement of the SAME local tangential Gaussian law.
    # The full tails retain their original weights; this is not beam tilt.
    angular_sampling: str = "uniform_cdf"
    angular_refinement_gain: float = 0.0
    angular_refinement_width_sigma: float = .1
    angular_stratum_allocation: tuple[int, ...] = ()

    def validate(self, geometry, *, coherent=False):
        if self.angular_sampling not in {"uniform_cdf", "tangent_stratified_v1", "tangent_stratified_v2"}:
            raise ValueError("Unknown tip angular quadrature")
        if self.angular_stratum_allocation and (
                self.angular_sampling != "tangent_stratified_v2" or coherent
                or len(self.angular_stratum_allocation) != 9
                or any(type(v) is not int or not 1 <= v <= 1000 for v in self.angular_stratum_allocation)):
            raise ValueError("Tangent allocation requires nine positive integer numerical priorities and classical v2 quadrature")
        if (not math.isfinite(self.angular_refinement_gain) or self.angular_refinement_gain < 0
                or not math.isfinite(self.angular_refinement_width_sigma)
                or self.angular_refinement_width_sigma <= 0):
            raise ValueError("Angular quadrature refinement must be finite and its width positive")
        if self.angular_sampling.startswith("tangent_stratified_") and (
                coherent or self.energy_distribution != "normal_tangential_exponential"
                or self.maximum_angle_deg != 90 or self.tangential_mean_energy_ev <= 0
                or self.directions_per_position < 9):
            raise ValueError("Tangent quadrature requires classical exponential emission, a 90-degree limit and nine directions per site")
        if self.spatial_sampling not in {"uniform_area", "apex_stratified_v1"}:
            raise ValueError("Unknown tip spatial quadrature")
        if self.spatial_stratum_allocation and (
                self.spatial_sampling != "apex_stratified_v1" or coherent
                or len(self.spatial_stratum_allocation) != 9
                or any(type(v) is not int or not 1 <= v <= 1000 for v in self.spatial_stratum_allocation)):
            raise ValueError("Cap allocation requires nine positive integer numerical priorities and classical full-cap strata")
        if type(self.directions_per_position) is not int or not 1 <= self.directions_per_position <= 256:
            raise ValueError("Directions per emission position must be an integer in [1, 256]")
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
    # Interpolation support, not an electrode size or source extent. Zero is
    # the raw-grid diagnostic reference. Nondefault values survive profiles.
    axis_core_fraction: float = .01
    # Local mesh resolution of electrode fringes; zero is the historical mesh.
    electrode_cells_per_bore: int = 8
    # Optional numerical grading at conductor corners, not rounded metal.
    electrode_corner_cells: int = 0

    def validate(self):
        if (type(self.electrode_corner_cells) is not int
                or self.electrode_corner_cells not in (0, *range(4, 65))):
            raise ValueError("Electrode corner cells must be zero or an integer in [4, 64]")
        if (isinstance(self.axis_core_fraction, bool) or not math.isfinite(self.axis_core_fraction)
                or not 0 <= self.axis_core_fraction <= .05):
            raise ValueError("Axis core fraction must be in [0, 0.05]")
        if (type(self.electrode_cells_per_bore) is not int
                or self.electrode_cells_per_bore not in (0, *range(4, 65))):
            raise ValueError("Electrode cells per bore must be zero or an integer in [4, 64]")
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
        if result["field_numerics"]["axis_core_fraction"] == .01:
            result["field_numerics"].pop("axis_core_fraction")
        if result["field_numerics"]["electrode_cells_per_bore"] == 8:
            result["field_numerics"].pop("electrode_cells_per_bore")
        if result["field_numerics"]["electrode_corner_cells"] == 0:
            result["field_numerics"].pop("electrode_corner_cells")
        if result["emission"]["spatial_sampling"] == "uniform_area":
            result["emission"].pop("spatial_sampling")
        if not result["emission"]["spatial_stratum_allocation"]:
            result["emission"].pop("spatial_stratum_allocation")
        if not result["emission"]["angular_stratum_allocation"]:
            result["emission"].pop("angular_stratum_allocation")
        for name,default in (("angular_sampling","uniform_cdf"),("angular_refinement_gain",0.),
                             ("angular_refinement_width_sigma",.1)):
            if result["emission"][name] == default:
                result["emission"].pop(name)
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
            if isinstance(row["emission"].spatial_stratum_allocation, list):
                from dataclasses import replace
                row["emission"] = replace(row["emission"], spatial_stratum_allocation=tuple(
                    row["emission"].spatial_stratum_allocation))
            if isinstance(row["emission"].angular_stratum_allocation, list):
                from dataclasses import replace
                row["emission"] = replace(row["emission"], angular_stratum_allocation=tuple(
                    row["emission"].angular_stratum_allocation))
            row["field_numerics"] = TipFieldNumerics(**row["field_numerics"])
            if row.get("coherence") is not None:
                row["coherence"] = SurfaceCoherence(**row["coherence"])
            return cls(**row).validate()
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid tip surface model: {error}") from error


def load_tip_surface_reference(path: str | Path = REFERENCE):
    with input_io.open_input(path) as stream:
        return TipSurfaceModel.from_dict(tomllib.load(stream))


def _positive(value, name, allow_zero=False):
    if value is None or isinstance(value, bool) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def emit_surface(model, count):
    """Positions [m], directions, local energies [eV], outgoing-flux weights.

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
    p = model.emission
    n = int(count)
    u, a, en, et, b = _halton_dimensions(n, (2, 3, 5, 7, 11))
    weights = np.full(n, 1/n)
    if p.directions_per_position > 1:
        # Spend the existing ray budget on shared surface sites and local
        # direction/energy samples, never multiply the physical current.
        directions = p.directions_per_position
        if p.spatial_sampling == "apex_stratified_v1":
            # A small preview must still represent EVERY cap stratum. Only
            # repartition previously unsupported small budgets; old valid
            # quadratures and physical source parameters remain unchanged.
            minimum = 81 if p.angular_sampling.startswith("tangent_stratified_") else 9
            if n < minimum:
                raise ValueError(f"Full-cap quadrature needs at least {minimum} current-carrying rays")
            directions = min(directions,n//9)
        sites = max(1, n // directions)
        sizes = np.full(sites, n // sites, dtype=int)
        sizes[:n % sites] += 1
        site_u, site_a, rotation = _halton_dimensions(sites, (2, 3, 13))
        u, a = np.repeat(site_u, sizes), np.repeat(site_a, sizes)
        starts = np.repeat(np.cumsum(sizes)-sizes, sizes)
        b = (np.arange(n)-starts+np.repeat(rotation, sizes))/np.repeat(sizes, sizes)
        weights = 1/(sites*np.repeat(sizes, sizes))
    if p.spatial_sampling == "apex_stratified_v1":
        # Cover the WHOLE cap with disjoint equal-area-coordinate annuli. Each
        # stratum's quadrature weights sum to its actual fraction of source
        # current, including outer rays that will be absorbed by real stops.
        # Refining this rule changes numerical sampling, not emitted flux or
        # the angular/energy distribution. Do not renormalise transmitted rays.
        if p.directions_per_position == 1:
            sites, sizes = n, np.ones(n, dtype=int)
            site_u, site_a, rotation = _halton_dimensions(sites, (2, 3, 13))
        from temsim.optics.electron_gun.tip_sampling import stratified_cap_area
        site_u, site_a, site_weight = stratified_cap_area(sites, p.spatial_stratum_allocation)
        u, a = np.repeat(site_u, sizes), np.repeat(site_a, sizes)
        weights = np.repeat(site_weight/sizes, sizes)
    from temsim.optics.electron_gun.tip_patch import sample_cap_frame
    positions, normals, tangent1, tangent2 = sample_cap_frame(model.geometry, p.cap_half_angle_deg, u, a)
    if p.angular_sampling.startswith("tangent_stratified_"):
        from temsim.optics.electron_gun.tip_sampling import stratified_tangent_momenta, tangent_cell_ids
        # The central refinement follows position only; it does not condition
        # away any physical angle/energy or depend on a downstream launch.
        first = np.cumsum(sizes)-sizes
        t1,t2 = np.empty(n),np.empty(n)
        sigma = math.sqrt(p.tangential_mean_energy_ev/2)
        sequence_offset = 0
        for start,size in zip(first,sizes):
            section = slice(start,start+size)
            centre = -p.angular_refinement_gain*math.hypot(*normals[start,:2])
            qn,a1,a2,cw = stratified_tangent_momenta(int(size),sigma_sqrt_ev=sigma,
                radial_centre_sigma=centre,halfwidth_sigma=p.angular_refinement_width_sigma,
                sequence_offset=sequence_offset,allocation=p.angular_stratum_allocation)
            if p.angular_sampling == "tangent_stratified_v2":
                # Increasing the site count must refine energy and local
                # momentum too, not repeat the same few directions forever.
                # v1 remains replayable for existing diagnostic records.
                sequence_offset += int(np.bincount(tangent_cell_ids(int(size),p.angular_stratum_allocation),minlength=9).max())
            en[section],t1[section],t2[section] = qn,a1,a2
            weights[section] *= size*cw
        normal_energy = -p.normal_mean_energy_ev*np.log1p(-en)
        direction = np.sqrt(normal_energy)[:,None]*normals+t1[:,None]*tangent1+t2[:,None]*tangent2
        direction /= np.linalg.norm(direction,axis=1)[:,None]
        return positions,direction,normal_energy+t1*t1+t2*t2,weights
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
    return positions, direction, normal_energy + tangent_energy, weights


def surface_bundle(model, count, *, support_probes=0, quadrature=None):
    """Retain the curved launch surface and full direction, including dz < 0."""
    from temsim.optics.electron_gun.base import EmissionBundle
    if quadrature is None:
        p, d, energy, weight = emit_surface(model, count-support_probes if support_probes else count)
    else:
        if support_probes or count != quadrature.total:
            raise ValueError("Surface product quadrature requires its complete declared population")
        from temsim.optics.electron_gun.tip_sampling import surface_product_samples
        p, d, energy, weight = surface_product_samples(model, quadrature)
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
        probe_p, probe_d, _, _ = sample_cap_frame(
            model.geometry, model.emission.cap_half_angle_deg, area, azimuth)
        p, d = np.vstack((p, probe_p)), np.vstack((d, probe_d))
        energy = np.r_[energy, np.full(support_probes, model.emission.mean_energy_ev)]
        weight = np.r_[weight, np.zeros(support_probes)]
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
