"""Structure-derived inputs for the approximate, incoherent high-angle tail.

The material calculation counts symmetry-expanded sites and every occupied
species. It is independent of the coherent atomistic-wave representation.
Screening follows the Thomas--Fermi/Moliere parameter in Geant4's Physics
Reference Manual, electron nuclear scattering equations 97--98. The existing
tail cross section remains an approximation, not a full Mott calculation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from scipy.constants import alpha, c, electron_mass, hbar, physical_constants
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import ncx2

from temsim.specimen.envelope import canonical_sample_envelope_shape
from temsim.specimen.source import active_cif_path
from temsim.specimen.virtual import (
    PARAXIAL_VIRTUAL_MAX_MRAD,
    VirtualAngularDistribution,
    VirtualInteractionComponent,
    _component_points,
    integrate_screened_rutherford_cross_section_m2,
    relativistic_beta_gamma,
)

SCREENING_REFERENCE_URL = (
    "https://geant4.web.cern.ch/documentation/dev/prm_html/"
    "PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html"
)
ATOMIC_MASS_UNIT_G = 1.66053906660e-24


@dataclass(frozen=True, slots=True)
class CIFComposition:
    source_path: str
    volume_nm3: float
    atoms_per_cell: tuple[tuple[int, float], ...]
    number_densities_atoms_nm3: tuple[tuple[int, float], ...]
    density_g_cm3: float
    mass_fractions: tuple[tuple[int, float], ...]
    partial_occupancy: bool
    mixed_occupancy: bool
    provenance: str


@dataclass(frozen=True, slots=True)
class TailElement:
    atomic_number: int
    number_density_atoms_nm3: float | None
    areal_density_atoms_nm2: float
    screening_angle_mrad: float


@dataclass(frozen=True, slots=True)
class TailMaterial:
    elements: tuple[TailElement, ...]
    material_source: str
    screening_source: str
    thickness_nm: float
    provenance: str
    warnings: tuple[str, ...]
    composition: CIFComposition | None = None


def _normal_interval_probability(lower, upper):
    """Stable standard-normal mass, including intervals in the positive tail."""
    return np.where(np.asarray(lower) >= 0., ndtr(-lower) - ndtr(-upper),
                    ndtr(upper) - ndtr(lower))


def finite_sample_gaussian_overlap(sample, x_um, y_um, *, probe_sigma_nm=0.):
    """Integrate a normalized isotropic Gaussian over the actual sample.

    Coordinates are absolute probe centres in micrometres; sample dimensions,
    centre and one-axis Gaussian sigma are nanometres. Each point is evaluated
    independently of raster size, ordering and spacing. This is a uniform
    finite-envelope approximation, not the coherent probe's intensity.

    A round disk uses the noncentral chi-square CDF. Rectangles separate into
    normal CDF intervals. A legacy disk with unequal diameters follows the
    shared envelope's ellipse semantics using one-dimensional quadrature.
    """
    shape = canonical_sample_envelope_shape(getattr(sample, "envelope_shape", "rectangle"))
    x, y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)
    if x.shape != y.shape or not x.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Sample overlap needs matching non-empty finite probe coordinates.")
    sizes = tuple(float(getattr(sample, name, 0.)) for name in ("size_x_nm", "size_y_nm"))
    centres = tuple(float(getattr(sample, name, 0.)) for name in ("centre_x_nm", "centre_y_nm"))
    sigma = float(probe_sigma_nm)
    if not all(math.isfinite(value) and value > 0 for value in sizes):
        raise ValueError("Sample overlap X/Y sizes must be finite and positive.")
    if not all(math.isfinite(value) for value in centres):
        raise ValueError("Sample overlap centre must be finite.")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Sample overlap Gaussian sigma must be finite and nonnegative.")
    with np.errstate(over="ignore", invalid="ignore"):
        dx, dy = x * 1e3 - centres[0], y * 1e3 - centres[1]
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        raise ValueError("Sample overlap coordinates exceed the finite nanometre range.")
    a, b = sizes[0] * .5, sizes[1] * .5
    if sigma == 0.:
        inside = ((np.abs(dx) <= a) & (np.abs(dy) <= b) if shape == "rectangle"
                  else np.hypot(dx / a, dy / b) <= 1.)
        return inside.astype(float)
    if shape == "rectangle":
        result = (_normal_interval_probability((-a-dx)/sigma, (a-dx)/sigma)
                  * _normal_interval_probability((-b-dy)/sigma, (b-dy)/sigma))
        return np.clip(result, 0., 1.)

    # Beyond twelve standard deviations the omitted Gaussian probability is
    # below 1e-32. This also avoids ill-conditioned enormous noncentralities
    # for macroscopic samples with a narrow probe deep inside or far outside.
    result = np.zeros(x.shape, dtype=float)
    far_inside = np.hypot((np.abs(dx) + 12*sigma)/a, (np.abs(dy) + 12*sigma)/b) <= 1.
    far_outside = np.hypot(np.maximum(np.abs(dx)-12*sigma, 0)/a,
                           np.maximum(np.abs(dy)-12*sigma, 0)/b) > 1.
    result[far_inside] = 1.
    pending = ~far_inside & ~far_outside
    if a == b and a / sigma <= 1e6:
        distances = np.hypot(dx[pending], dy[pending]) / sigma
        result[pending] = ncx2.cdf((a/sigma)**2, df=2, nc=distances**2)
    else:
        # Standardize the integration coordinate to keep narrow off-centre
        # probes resolved instead of integrating over a possibly huge body.
        for index in np.flatnonzero(pending):
            px, py = float(dx.flat[index]), float(dy.flat[index])
            lo, hi = max(-12., (-a-px)/sigma), min(12., (a-px)/sigma)
            if hi <= lo:
                continue
            def integrand(u):
                local_x = (px + sigma*u) / a
                half_y = b * math.sqrt(max(0., (1.-local_x)*(1.+local_x)))
                y_mass = float(_normal_interval_probability((-half_y-py)/sigma, (half_y-py)/sigma))
                return math.exp(-.5*u*u) / math.sqrt(2*math.pi) * y_mass
            result.flat[index] = quad(integrand, lo, hi, epsabs=2e-11, epsrel=2e-10, limit=150)[0]
    if not np.all(np.isfinite(result)):
        raise ValueError("Finite-sample Gaussian overlap could not be resolved numerically.")
    return np.clip(result, 0., 1.)


def composition_from_atoms(unit, *, source_path="") -> CIFComposition:
    """Count all species at each expanded ASE CIF site, never just its majority."""
    volume_nm3 = abs(float(np.linalg.det(unit.cell.array))) * 1.0e-3
    if not math.isfinite(volume_nm3) or volume_nm3 <= 0.0 or not len(unit):
        raise ValueError("CIF material requires atoms and a finite three-dimensional unit cell.")
    occupancies = unit.info.get("occupancy")
    kinds = unit.arrays.get("spacegroup_kinds")
    if occupancies is not None and (not isinstance(occupancies, dict) or kinds is None):
        raise ValueError("CIF occupancy metadata has no usable expanded-site mapping.")
    counts, masses = defaultdict(float), defaultdict(float)
    site_masses = unit.get_masses()
    partial = mixed = False
    for site, z in enumerate(unit.numbers):
        if occupancies is None:
            species = {chemical_symbols[int(z)]: 1.0}
        else:
            species = occupancies.get(str(int(kinds[site])))
            if not isinstance(species, dict) or not species:
                raise ValueError("CIF has incomplete site-occupancy metadata.")
        total = 0.0
        occupied = 0
        for symbol, raw_occupancy in species.items():
            try:
                element = atomic_numbers[str(symbol)]
                occupancy = float(raw_occupancy)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("CIF contains an invalid species or occupancy.") from exc
            if not 1 <= element <= 118 or not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
                raise ValueError("CIF site occupancies must be finite fractions in [0, 1].")
            total += occupancy
            occupied += occupancy > 0.0
            counts[element] += occupancy
            # Retain the ASE mass of an explicitly represented isotope; a
            # minority species uses the same ASE elemental mass library.
            mass = float(site_masses[site]) if element == int(z) else float(atomic_masses[element])
            masses[element] += occupancy * mass
        if total > 1.0 + 1.0e-8:
            raise ValueError("CIF occupancies at one site sum to more than one.")
        partial |= not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8)
        mixed |= occupied > 1
    counts = {z: count for z, count in counts.items() if count > 0.0}
    total_mass = sum(masses.values())
    if not counts or not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("CIF material has no positive occupied atomic mass.")
    provenance = f"CIF symmetry-expanded, occupancy-weighted unit cell: {source_path}"
    return CIFComposition(
        str(source_path), volume_nm3, tuple(sorted(counts.items())),
        tuple((z, n / volume_nm3) for z, n in sorted(counts.items())),
        total_mass * ATOMIC_MASS_UNIT_G / (volume_nm3 * 1.0e-21),
        tuple((z, masses[z] / total_mass) for z in sorted(counts)),
        partial, mixed, provenance,
    )


@lru_cache(maxsize=16)
def _read_composition(source_path: str, contents: bytes) -> CIFComposition:
    from ase.io import read

    try:
        unit = read(BytesIO(contents), format="cif", fractional_occupancies=True)
        return composition_from_atoms(unit, source_path=source_path)
    except (ValueError, TypeError, KeyError, IndexError, AssertionError) as exc:
        raise ValueError(f"Cannot derive CIF material from {source_path}: {exc}") from exc


def read_cif_composition(cif_path) -> CIFComposition:
    """Read a small unit cell; cache by actual bytes so file edits cannot go stale."""
    path = Path(cif_path).expanduser().resolve()
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"CIF file cannot be read: {path}") from exc
    return _read_composition(str(path), contents)


def estimate_screening_angle_mrad(atomic_number: int, beam_energy_kv: float) -> float:
    """Map Moliere A_s to the tail kernel's sin(theta_s / 2)**2 exactly.

    A_s=(hbar/(2*p*a_TF))**2 * [1.13+3.76*(alpha*Z/beta)**2],
    a_TF=(1/2)*(3*pi/4)**(2/3)*a_0/Z**(1/3). See reference eqs. 97--98.
    """
    z = int(atomic_number)
    if z != atomic_number or not 1 <= z <= 118:
        raise ValueError("Rutherford atomic number must be an integer from 1 to 118.")
    beta, gamma = relativistic_beta_gamma(beam_energy_kv)
    momentum = gamma * electron_mass * beta * c
    a_tf = 0.5 * (3.0 * math.pi / 4.0) ** (2.0 / 3.0) * physical_constants["Bohr radius"][0] / z ** (1.0 / 3.0)
    screening = (hbar / (2.0 * momentum * a_tf)) ** 2 * (1.13 + 3.76 * (alpha * z / beta) ** 2)
    if not 0.0 < screening < 1.0:
        raise ValueError("Beam energy is outside the supported Thomas-Fermi/Moliere tail approximation.")
    return 2.0 * math.asin(math.sqrt(screening)) * 1.0e3


def resolve_tail_material(sample, thickness_nm, beam_energy_kv) -> TailMaterial:
    """Resolve independent structure/manual material and Moliere/manual screening."""
    thickness = float(thickness_nm)
    if not math.isfinite(thickness) or thickness < 0.0:
        raise ValueError("Tail material thickness must be finite and nonnegative.")
    relativistic_beta_gamma(beam_energy_kv)
    material_source = str(getattr(sample, "real_tail_material_source", "structure"))
    screening_source = str(getattr(sample, "real_tail_screening_source", "moliere"))
    if material_source not in {"structure", "manual"} or screening_source not in {"moliere", "manual"}:
        raise ValueError("Tail sources must be structure/manual material and moliere/manual screening.")
    composition = None
    warnings = ["Screened Rutherford high-angle approximation; not a full Mott or multiple-scattering calculation."]
    if material_source == "structure":
        path = active_cif_path(sample)
        if not path:
            raise ValueError("Automatic tail material needs the current structure's CIF; choose a structure or explicit manual override.")
        composition = read_cif_composition(path)
        rows = [(z, n, n * thickness) for z, n in composition.number_densities_atoms_nm3]
        provenance = composition.provenance + f"; interacting thickness {thickness:g} nm."
        if composition.partial_occupancy or composition.mixed_occupancy:
            warnings.append("Composition includes partial/mixed occupancy. The current coherent atomistic wave cannot represent these sites; use an explicitly ordered structure for wave imaging.")
    else:
        raw_z = getattr(sample, "real_tail_atomic_number", 14)
        z = int(raw_z)
        density = float(getattr(sample, "real_tail_areal_density_atoms_nm2", 0.0))
        if z != raw_z or not 1 <= z <= 118 or not math.isfinite(density) or density <= 0.0:
            raise ValueError("Manual tail needs Z=1..118 and a positive finite areal density in atoms/nm2.")
        rows = [(z, None, density)]
        provenance = "Explicit manual single-element areal density; independent of structure thickness."
    if screening_source == "manual":
        angle = float(getattr(sample, "real_tail_screening_angle_mrad", 5.0))
        if not math.isfinite(angle) or not 0.0 < angle <= PARAXIAL_VIRTUAL_MAX_MRAD:
            raise ValueError("Manual tail screening angle must be in (0, 500] mrad.")
        angles = {z: angle for z, _, _ in rows}
        provenance += " Manual screening angle."
    else:
        angles = {z: estimate_screening_angle_mrad(z, beam_energy_kv) for z, _, _ in rows}
        provenance += f" Thomas-Fermi/Moliere screening at {float(beam_energy_kv):g} kV (Geant4 eqs. 97-98)."
        if float(beam_energy_kv) < 200.0:
            warnings.append("Below 200 keV, factorized screening/spin approximations need particular care, especially for heavy elements.")
    return TailMaterial(tuple(TailElement(z, n, area, angles[z]) for z, n, area in rows),
                        material_source, screening_source, thickness, provenance, tuple(warnings), composition)


def build_tail_angular_distribution(material: TailMaterial, *, beam_energy_kv,
                                    minimum_angle_mrad, maximum_angle_mrad) -> VirtualAngularDistribution:
    """One Poisson event law for the summed elemental optical depth.

    Element i contributes P(total event) * tau_i / sum(tau), rather than an
    independent probability whose sum could exceed one. Its conditional
    angular distribution uses the matching Z and screening parameter.
    """
    lower, upper = float(minimum_angle_mrad), float(maximum_angle_mrad)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower < upper <= PARAXIAL_VIRTUAL_MAX_MRAD):
        raise ValueError("High-angle tail requires 0 <= minimum < maximum <= 500 mrad.")
    inputs = []
    for element in material.elements:
        cross_section = integrate_screened_rutherford_cross_section_m2(
            atomic_number=element.atomic_number, beam_energy_kv=beam_energy_kv,
            screening_angle_mrad=element.screening_angle_mrad,
            minimum_angle_mrad=lower, maximum_angle_mrad=upper,
        )
        tau = element.areal_density_atoms_nm2 * 1.0e18 * cross_section
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError("Tail elemental optical depth must be finite and nonnegative.")
        inputs.append((element, cross_section, tau))
    total_depth = math.fsum(row[2] for row in inputs)
    probability = -math.expm1(-total_depth)
    components, xs, ys, weights, kinds = [], [], [], [], []
    for element, cross_section, tau in inputs:
        parameters = dict(atomic_number=element.atomic_number, beam_energy_kv=float(beam_energy_kv),
                          areal_density_atoms_nm2=element.areal_density_atoms_nm2,
                          screening_angle_mrad=element.screening_angle_mrad,
                          minimum_angle_mrad=lower, maximum_angle_mrad=upper,
                          integrated_cross_section_m2=cross_section, optical_depth=tau,
                          radial_samples=128, azimuth_samples=64)
        share = probability * tau / total_depth if total_depth else 0.0
        component = VirtualInteractionComponent(f"Tail {chemical_symbols[element.atomic_number]}",
                      "physical_rutherford", share, parameters, "screened Rutherford; one mixture event law, not Mott")
        components.append(component)
        if share:
            x, y, conditional = _component_points(component)
            xs.append(x)
            ys.append(y)
            weights.append(conditional * share)
            kinds.extend([component.kind] * len(x))
    arrays = [np.concatenate(items) if items else np.empty(0) for items in (xs, ys, weights)]
    for array in arrays:
        array.setflags(write=False)
    result = VirtualAngularDistribution(*arrays, tuple(kinds), 1.0 - probability, 0.0, tuple(components))
    if not math.isclose(result.total_probability, 1.0, rel_tol=0.0, abs_tol=2.0e-12):
        raise RuntimeError("Rutherford mixture failed probability conservation.")
    return result
