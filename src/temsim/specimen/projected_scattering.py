"""Incoherent independent-atom projection for qualitative STEM observables.

This is an intensity model, not a wave or a replacement particle checkpoint.
Actual CIF sites are rotated and cropped in the laboratory specimen envelope.
A normalized Gaussian with the *executed* incident XY covariance convolves
their occupancy-weighted projected density. No additional source or fitted
probe width is introduced. Within the finite material overlap q, the
thin-specimen mean-optical-depth closure is Psc=q*(1-exp(-sum(tau_Z)/q));
P_Z=Psc*tau_Z/sum(tau_Z), with vacuum probability 1-q. P_Z groups at least
one event into one angular draw; it is NOT a probability of exactly one event.
Exponentiating the probe-averaged optical depth is also an approximation,
not an exact average of atomic impact-parameter transmission. It omits channeling, coherent
diffraction, dynamical diffraction and plural elastic angular redistribution.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import math
from threading import RLock

import numpy as np

from temsim import input_io
from temsim.immutable_json import freeze_json, json_digest
from temsim.specimen.elastic_transport import (
    RUTHERFORD_MODEL_NAME,
    RUTHERFORD_REFERENCE_URL,
    incident_rays_from_simulation,
    screened_rutherford_parameter,
    screened_rutherford_total_cross_section_cm2,
)
from temsim.specimen.envelope import sample_envelope_shape
from temsim.specimen.geometry import quaternion_to_matrix, sample_orientation_quaternion
from temsim.specimen.rutherford import composition_from_atoms, finite_sample_gaussian_overlap
from temsim.specimen.source import active_cif_path

MODEL = "cif_independent_atom_projected_elastic_gaussian_probe_v1"
GAUSSIAN_RADIUS_SIGMA = 8.0
MAX_CANDIDATE_SITES = 8_000_000
MAX_PROJECTED_SITES = 2_000_000
MAX_PAIR_EVALUATIONS = 100_000_000
_MAP_CACHE_BYTES = 64 * 1024**2
_MAP_CACHE = OrderedDict()
_MAP_CACHE_SIZE = 0
_CACHE_LOCK = RLock()
_CANCEL_CHECK = ContextVar("temsim_projected_scattering_cancel_check", default=None)


def _check_cancelled():
    callback = _CANCEL_CHECK.get()
    if callback is not None:
        callback()


def _readonly(values, dtype=np.float64):
    array = np.ascontiguousarray(values, dtype=dtype)
    # bytes-backed arrays cannot be made writable by a downstream consumer.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ProjectedElement:
    atomic_number: int
    optical_depth: np.ndarray
    scattered_fraction: np.ndarray
    mean_cross_section_nm2: float
    energy_ev: np.ndarray
    cross_section_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class ProjectedScattering:
    elements: tuple[ProjectedElement, ...]
    optical_depth: np.ndarray
    direct_fraction: np.ndarray
    scattered_fraction: np.ndarray
    plural_event_fraction: np.ndarray
    material_overlap: np.ndarray
    direct_material_fraction: np.ndarray
    vacuum_fraction: np.ndarray
    probe_covariance_nm2: np.ndarray
    probe_centroid_nm: tuple[float, float]
    chief_angle_mrad: tuple[float, float]
    mean_energy_ev: float
    surviving_fraction: float
    identity: str
    metrics: object


@lru_cache(maxsize=8)
def _unit_cell(contents: bytes):
    """ASE's symmetry/occupancy interpretation, with no abTEM dependency."""
    from ase.io import read
    from ase.data import atomic_numbers, chemical_symbols

    unit = read(BytesIO(contents), format="cif", fractional_occupancies=True)
    composition_from_atoms(unit)  # Shared strict cell/species/occupancy validation.
    cell = np.asarray(unit.cell.array, dtype=float) * 0.1  # angstrom -> nm
    fractional = np.asarray(unit.get_scaled_positions(wrap=True), dtype=float)
    occupations = unit.info.get("occupancy")
    kinds = unit.arrays.get("spacegroup_kinds")
    basis, numbers, weights = [], [], []
    for index, number in enumerate(unit.numbers):
        species = ({chemical_symbols[int(number)]: 1.0} if occupations is None
                   else occupations[str(int(kinds[index]))])
        for symbol, fraction in sorted(species.items()):
            if float(fraction) > 0:
                z = int(atomic_numbers[symbol])
                if z > 99:
                    raise ValueError("Projected elastic scattering supports atomic numbers 1 to 99.")
                basis.append(fractional[index])
                numbers.append(z)
                weights.append(float(fraction))
    return (_readonly(cell), _readonly(basis), _readonly(numbers, np.int16),
            _readonly(weights))


@lru_cache(maxsize=4)
def _projected_sites(contents, geometry, roi, chief_slopes):
    """Enumerate the original (possibly oblique) lattice in bounded chunks.

    The integer bounding box is obtained by inverse-transforming the actual
    lab ROI. No orthogonalization, cell deformation, display atom limit, or
    hidden replacement by bulk material density is performed.
    """
    sx, sy, thickness, cx, cy, shape, quaternion = geometry
    cell, basis, numbers, occupations = _unit_cell(contents)
    rotation = quaternion_to_matrix(quaternion)
    lattice = cell @ rotation.T
    half = np.array((sx, sy, thickness)) * 0.5
    tilt = np.asarray(chief_slopes)
    # Project a tilted straight chief path back to the specimen centre plane.
    # The footprint is held fixed through the slab (thin projection model).
    lower = np.r_[np.maximum(-half[:2], np.asarray(roi[:2]) - [cx, cy]
                             - half[2] * np.abs(tilt)), -half[2]]
    upper = np.r_[np.minimum(half[:2], np.asarray(roi[2:]) - [cx, cy]
                             + half[2] * np.abs(tilt)), half[2]]
    if np.any(lower >= upper):
        return _readonly(np.empty((0, 2))), _readonly([], np.int16), _readonly([]), 0
    corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    inverse_corners = corners @ np.linalg.inv(lattice)
    parts_xy, parts_z, parts_w = [], [], []
    examined = retained = 0
    for frac, number, occupancy in zip(basis, numbers, occupations, strict=True):
        _check_cancelled()
        lo = np.floor(inverse_corners.min(axis=0) - frac).astype(np.int64) - 1
        hi = np.ceil(inverse_corners.max(axis=0) - frac).astype(np.int64) + 1
        counts = hi - lo + 1
        count = math.prod(int(value) for value in counts)
        examined += count
        if examined > MAX_CANDIDATE_SITES:
            raise ValueError(
                "CIF projected-scattering region exceeds the atomic enumeration budget; "
                "reduce the scan/material intersection or specimen thickness. "
                "No bulk or sharpened-probe substitute was used."
            )
        for start in range(0, count, 65_536):
            _check_cancelled()
            index = np.arange(start, min(start + 65_536, count), dtype=np.int64)
            cells = np.column_stack((index // (counts[1] * counts[2]),
                                      index // counts[2] % counts[1], index % counts[2])) + lo
            position = (cells + frac) @ lattice
            # Half-open faces assign a boundary lattice plane to exactly one
            # slab rather than double counting both periodic end faces.
            keep = np.all((position >= -half) & (position < half), axis=1)
            if shape == "disk":
                keep &= np.sum((position[:, :2] / half[:2])**2, axis=1) <= 1.0
            projected = position[:, :2] - position[:, 2, None] * tilt + [cx, cy]
            keep &= np.all((projected >= roi[:2]) & (projected <= roi[2:]), axis=1)
            found = int(np.count_nonzero(keep))
            retained += found
            if retained > MAX_PROJECTED_SITES:
                raise ValueError("Projected CIF exceeds the retained-atom budget; reduce the material/scan region.")
            if found:
                parts_xy.append(projected[keep])
                parts_z.append(np.full(found, number, dtype=np.int16))
                parts_w.append(np.full(found, occupancy))
    if not retained:
        return _readonly(np.empty((0, 2))), _readonly([], np.int16), _readonly([]), 0
    return (_readonly(np.concatenate(parts_xy)), _readonly(np.concatenate(parts_z), np.int16),
            _readonly(np.concatenate(parts_w)), retained)


def _gaussian_density(positions, occupancies, centres, covariance):
    """Occupancy/area in nm^-2; exact Gaussian sum inside the declared ROI."""
    if not len(positions):
        return np.zeros(len(centres))
    # Exact duplicate projected sites, e.g. one aligned atomic column, can be
    # summed without changing the Gaussian at any scan position.
    positions, inverse = np.unique(positions, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=occupancies, minlength=len(positions))
    if len(positions) * len(centres) > MAX_PAIR_EVALUATIONS:
        raise ValueError("Projected probe convolution exceeds its work budget; reduce scan pixels or material region.")
    inverse_covariance = np.linalg.inv(covariance)
    normalization = 1.0 / (2.0 * math.pi * math.sqrt(float(np.linalg.det(covariance))))
    result = np.zeros(len(centres))
    block = max(1, min(256, 524_288 // len(positions)))
    for start in range(0, len(centres), block):
        _check_cancelled()
        delta = centres[start:start + block, None, :] - positions[None, :, :]
        distance = np.einsum("...i,ij,...j->...", delta, inverse_covariance, delta)
        result[start:start + block] = np.exp(-0.5 * distance) @ weights * normalization
    return result


def _material_overlap(sample, points, covariance, chief_slopes):
    """Gaussian mass in the finite slab's straight-path projected silhouette.

    Arbitrary XY covariance is integrated as marginal X times conditional Y.
    Tilting the chief ray projects the slab into an envelope plus a segment;
    its vertical intervals are convex and are evaluated without pixel masks.
    The fixed covariance is the same thin-slab closure as the atomic density.
    """
    from scipy.integrate import quad
    from scipy.special import ndtr

    tilt_x, tilt_y = chief_slopes
    sigma_x, sigma_y = np.sqrt(np.diag(covariance))
    if tilt_x == 0.0 and tilt_y == 0.0 and covariance[0, 1] == 0.0 and sigma_x == sigma_y:
        return finite_sample_gaussian_overlap(sample, points[:, 0]*1e-3, points[:, 1]*1e-3,
                                              probe_sigma_nm=float(sigma_x))
    a, b, h = np.array((sample.size_x_nm, sample.size_y_nm, sample.thickness_nm))*0.5
    shape = sample_envelope_shape(sample)
    conditional_sigma = math.sqrt(float(np.linalg.det(covariance)/covariance[0, 0]))
    conditional_slope = float(covariance[0, 1]/covariance[0, 0])
    extent_x, extent_y = a+h*abs(tilt_x), b+h*abs(tilt_y)

    def limits(x):
        if abs(x) > extent_x:
            return 0.0, 0.0
        if tilt_x == 0.0:
            half_y = b if shape == "rectangle" else b*math.sqrt(max(0.0, 1.0-(x/a)**2))
            return -half_y-h*abs(tilt_y), half_y+h*abs(tilt_y)
        ends = sorted(((-a-x)/tilt_x, (a-x)/tilt_x))
        lo, hi = max(-h, ends[0]), min(h, ends[1])
        if lo > hi:
            return 0.0, 0.0
        if shape == "rectangle":
            values = -lo*tilt_y, -hi*tilt_y
            return -b+min(values), b+max(values)
        denominator = math.hypot(a*tilt_y, b*tilt_x)
        upper_u = -a*tilt_y*math.copysign(1.0, tilt_x)/denominator
        upper_z = min(max((a*upper_u-x)/tilt_x, lo), hi)
        lower_z = min(max((-a*upper_u-x)/tilt_x, lo), hi)
        def bound(z, sign):
            return -z*tilt_y + sign*b*math.sqrt(max(0.0, 1.0-((x+z*tilt_x)/a)**2))
        return (min(bound(z, -1) for z in (lo, hi, lower_z)),
                max(bound(z, 1) for z in (lo, hi, upper_z)))

    result = np.zeros(len(points))
    relative = points - [sample.centre_x_nm, sample.centre_y_nm]
    for index, (mean_x, mean_y) in enumerate(relative):
        _check_cancelled()
        if abs(mean_x) > extent_x+12*sigma_x or abs(mean_y) > extent_y+12*sigma_y:
            continue
        # An interior 12-sigma box of the unextended envelope is also wholly
        # inside its tilted projection. Omitted normal mass is below 1e-32.
        if ((abs(mean_x)+12*sigma_x <= a and abs(mean_y)+12*sigma_y <= b)
                and (shape == "rectangle" or ((abs(mean_x)+12*sigma_x)/a)**2
                     + ((abs(mean_y)+12*sigma_y)/b)**2 <= 1.0)):
            result[index] = 1.0
            continue
        lo, hi = max(-12., (-extent_x-mean_x)/sigma_x), min(12., (extent_x-mean_x)/sigma_x)
        if lo >= hi:
            continue
        def integrand(t):
            lower_y, upper_y = limits(mean_x+sigma_x*t)
            local_mean = mean_y+conditional_slope*sigma_x*t
            low, high = (lower_y-local_mean)/conditional_sigma, (upper_y-local_mean)/conditional_sigma
            mass = ndtr(-low)-ndtr(-high) if low >= 0 else ndtr(high)-ndtr(low)
            return math.exp(-t*t/2)/math.sqrt(2*math.pi)*max(0., mass)
        cuts = [(edge-mean_x)/sigma_x for edge in (-a, 0., a)]
        result[index] = quad(integrand, lo, hi, epsabs=2e-11, epsrel=2e-9, limit=160,
                             points=[value for value in cuts if lo < value < hi])[0]
    return np.clip(result, 0., 1.)


def _cache_put(key, result):
    global _MAP_CACHE_SIZE
    size = sum(array.nbytes for array in (result.optical_depth, result.direct_fraction,
                                          result.scattered_fraction, result.plural_event_fraction,
                                          result.material_overlap, result.direct_material_fraction, result.vacuum_fraction,
                                          result.probe_covariance_nm2))
    size += sum(sum(a.nbytes for a in (element.optical_depth, element.scattered_fraction,
                                     element.energy_ev, element.cross_section_weights))
                for element in result.elements)
    if size > _MAP_CACHE_BYTES:
        return result
    with _CACHE_LOCK:
        previous = _MAP_CACHE.pop(key, None)
        if previous is not None:
            _MAP_CACHE_SIZE -= previous[1]
        _MAP_CACHE[key] = (result, size)
        _MAP_CACHE_SIZE += size
        while _MAP_CACHE_SIZE > _MAP_CACHE_BYTES or len(_MAP_CACHE) > 8:
            _, (_, removed_size) = _MAP_CACHE.popitem(last=False)
            _MAP_CACHE_SIZE -= removed_size
    return result


def projected_optical_depth(state, simulation, scan_x_um, scan_y_um):
    """Return elastic probabilities conditional on current arriving at sample.

    Scan arrays are **absolute laboratory probe-centre positions** in um. The
    caller applies the scan baseline/incident-centroid transform exactly once.
    Returned probabilities are not multiplied by source current, detector
    acceptance, an existing Monte Carlo elastic probability, or inelastic loss.
    They must never be written back as an executed specimen-exit checkpoint.
    """
    from temsim.physics.optical_tuning import check_tuning_cancelled

    check_tuning_cancelled(state)
    token = _CANCEL_CHECK.set(lambda: check_tuning_cancelled(state))
    try:
        result = _projected_optical_depth(state, simulation, scan_x_um, scan_y_um)
        check_tuning_cancelled(state)
        return result
    finally:
        _CANCEL_CHECK.reset(token)


def _projected_optical_depth(state, simulation, scan_x_um, scan_y_um):
    x, y = np.asarray(scan_x_um, dtype=float), np.asarray(scan_y_um, dtype=float)
    if x.shape != y.shape or not x.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Projected scattering requires matching finite nonempty scan coordinates.")
    incident = incident_rays_from_simulation(state, simulation)
    rays = incident.rays
    weights = np.array([ray.weight for ray in rays])
    positions = np.array([ray.position_xy_nm for ray in rays])
    centre = weights @ positions
    centred = positions - centre
    covariance = np.einsum("n,ni,nj->ij", weights, centred, centred)
    eigenvalues = np.linalg.eigvalsh(covariance) if np.all(np.isfinite(covariance)) else np.array([0.])
    determinant = float(np.linalg.det(covariance))
    if eigenvalues[0] <= 0.0 or not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(
            "The executed incident rays do not define a finite two-dimensional probe footprint; "
            "run Ray Diagram with a resolved tip population. No artificial probe width is applied."
        )
    energies = np.array([ray.kinetic_energy_ev for ray in rays])
    energy_ev, energy_indices = np.unique(energies, return_inverse=True)
    energy_weights = np.bincount(energy_indices, weights=weights, minlength=len(energy_ev))
    chief_slopes = tuple(float(value) * 1e-3 for value in incident.chief_angle_mrad)
    # Projected sites live in the *laboratory z=constant plane*, not a plane
    # perpendicular to the chief ray. (x,y,z)->(x-z*tx,y-z*ty,z) has determinant
    # one, so a uniform slab still has n*t atoms per lab XY area after this
    # shear. The cross section is perpendicular to velocity: its lab-XY
    # interception area is sigma/|direction_z|. This factor supplies the
    # physical t/cos(theta) path length; site projection has not supplied it.
    path_factor = math.sqrt(1.0 + sum(value**2 for value in chief_slopes))
    sample = state.sample
    if bool(sample.inserted) and str(getattr(sample, "eds_support_material_key", "vacuum")) != "vacuum":
        raise ValueError(
            "Projected CIF scattering does not model a material support grid; "
            "use Material particle paths for this configuration."
        )
    thickness = float(sample.thickness_nm)
    sizes = float(sample.size_x_nm), float(sample.size_y_nm)
    centres = float(sample.centre_x_nm), float(sample.centre_y_nm)
    if (not all(math.isfinite(value) for value in (*sizes, *centres, thickness))
            or min(sizes) <= 0 or thickness < 0):
        raise ValueError("Projected specimen sizes must be finite and positive; thickness may be zero.")
    geometry = (*sizes, thickness, *centres, sample_envelope_shape(sample),
                sample_orientation_quaternion(sample))
    points = np.column_stack((x.ravel(), y.ravel())) * 1e3
    padding = GAUSSIAN_RADIUS_SIGMA * np.sqrt(np.diag(covariance))
    roi = tuple(np.r_[points.min(axis=0) - padding, points.max(axis=0) + padding])
    vacuum = not bool(sample.inserted) or thickness == 0.0
    contents = b"" if vacuum else input_io.read_bytes(active_cif_path(sample))
    identity = json_digest({"model": MODEL, "cif": sha256(contents).hexdigest(),
                            "geometry": geometry, "vacuum": vacuum, "probe": covariance.tolist(),
                            "centroid": centre.tolist(), "chief_slopes": chief_slopes,
                            "energies": sha256(energy_ev.tobytes() + energy_weights.tobytes()).hexdigest(),
                            "scan": sha256(x.tobytes() + y.tobytes()).hexdigest(), "shape": x.shape,
                            "survival": incident.surviving_fraction})
    with _CACHE_LOCK:
        hit = _MAP_CACHE.get(identity)
        if hit is not None:
            _MAP_CACHE.move_to_end(identity)
            return hit[0]
    if vacuum:
        atom_xy, atom_z, atom_weights, atom_count = np.empty((0, 2)), np.array([]), np.array([]), 0
    else:
        atom_xy, atom_z, atom_weights, atom_count = _projected_sites(contents, geometry, roi, chief_slopes)
    total = np.zeros(x.shape)
    pending = []
    for z in sorted(set(int(z) for z in atom_z)):
        cross_sections = np.array([screened_rutherford_total_cross_section_cm2(z, energy)
                                   * 1e14 for energy in energy_ev])  # cm^2 -> nm^2
        mean_cross_section = float(energy_weights @ cross_sections)
        mask = atom_z == z
        density = _gaussian_density(atom_xy[mask], atom_weights[mask], points, covariance).reshape(x.shape)
        tau = density * mean_cross_section * path_factor
        total += tau
        pending.append((z, tau, mean_cross_section, energy_weights * cross_sections / mean_cross_section))
    overlap = (np.zeros(x.shape) if vacuum else
               _material_overlap(sample, points, covariance, chief_slopes).reshape(x.shape))
    conditional_tau = np.divide(total, overlap, out=np.zeros_like(total), where=overlap > 0)
    direct_material = overlap*np.exp(-conditional_tau)
    scattered = overlap*(-np.expm1(-conditional_tau))
    vacuum_fraction = 1.0-overlap
    direct = vacuum_fraction+direct_material
    plural = np.maximum(0.0, scattered-conditional_tau*direct_material)
    elements = tuple(ProjectedElement(z, _readonly(tau),
                      _readonly(scattered * np.divide(tau, total, out=np.zeros_like(total), where=total > 0)),
                      cross_section, _readonly(energy_ev), _readonly(scattering_weights))
                     for z, tau, cross_section, scattering_weights in pending)
    metrics = freeze_json({
        "model": MODEL, "elastic_cross_section_model": RUTHERFORD_MODEL_NAME,
        "elastic_cross_section_reference": RUTHERFORD_REFERENCE_URL,
        "probe_model": "normalized Gaussian with executed sample-plane XY covariance; no fitted sharpening",
        "probe_sigma_principal_nm": np.sqrt(eigenvalues).tolist(),
        "probe_covariance_nm2": covariance.tolist(), "projected_atom_species_records": atom_count,
        "cif_sha256": sha256(contents).hexdigest(), "gaussian_roi_radius_sigma": GAUSSIAN_RADIUS_SIGMA,
        "roi_omission": "atom centres beyond eight marginal standard deviations from all scan centres omitted; total optical-depth error also depends on omitted occupancy and cross section",
        "angular_model": "at least one elastic event represented by one screened Rutherford angle; not crystalline Bragg diffraction",
        "limitations": ("thin-specimen fixed-footprint projection", "no channeling or coherent interference",
                        "no plural elastic angular redistribution", "no position-energy correlation in Gaussian closure",
                        "exp(-probe-averaged optical depth) is a thin-optical-depth closure, not average impact-parameter transmission",
                        "heavy elements above Z=30 and energies outside 100-300 keV are outside stated Rutherford accuracy"),
        "maximum_optical_depth": float(total.max()),
        "maximum_material_conditional_optical_depth": float(conditional_tau.max()),
        "maximum_plural_event_fraction": float(plural.max()),
        "finite_overlap_model": "same XY Gaussian in straight-chief-path projected finite slab silhouette",
        "rutherford_heavy_element_warning": any(item.atomic_number > 30 for item in elements),
        "energy_range_ev": (float(energy_ev.min()), float(energy_ev.max())),
        "probability_reference": "current arriving at specimen; source survival and physical detector routing applied by caller",
        "checkpoint": False,
    })
    result = ProjectedScattering(elements, _readonly(total), _readonly(direct), _readonly(scattered), _readonly(plural),
                                _readonly(overlap), _readonly(direct_material), _readonly(vacuum_fraction),
                                _readonly(covariance), tuple(float(value) for value in centre),
                                incident.chief_angle_mrad, float(energy_weights @ energy_ev),
                                incident.surviving_fraction, identity, metrics)
    return _cache_put(identity, result)


def screened_angular_quadrature(element, polar_edges_rad, *, azimuth_samples=32):
    """Full-sphere, unit-normalized elastic quadrature for one element.

    Bin probabilities integrate the existing screened Rutherford CDF exactly,
    including the incident energy mixture weighted by elastic cross section.
    The representative polar angle bisects each bin's probability; azimuth is
    uniform. Returned (polar_rad, azimuth_rad, weights) need actual downstream
    propagation; large/backward angles must not be inserted into a paraxial map.
    """
    edges = np.asarray(polar_edges_rad, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or edges[0] != 0.0 or edges[-1] != math.pi or np.any(np.diff(edges) <= 0)):
        raise ValueError("Polar quadrature edges must increase from zero to pi.")
    count = int(azimuth_samples)
    if count != azimuth_samples or count < 1 or count > 4096:
        raise ValueError("Azimuth quadrature count must be an integer from 1 to 4096.")
    delta = np.array([screened_rutherford_parameter(element.atomic_number, energy)
                      for energy in element.energy_ev])
    mix = np.asarray(element.cross_section_weights)

    def interval_mass(lower, upper):
        # Subtract the CDF algebraically. Direct subtraction of two values
        # near one can erase a real narrow detector-angle interval or create
        # a signed roundoff mass; this positive form retains its probability.
        lower, upper = np.asarray(lower)[:, None], np.asarray(upper)[:, None]
        return np.sum(mix[None, :] * (1 + delta)[None, :] * delta[None, :]
                      * (upper - lower)
                      / ((upper + delta[None, :]) * (lower + delta[None, :])), axis=1)

    u_edges = np.sin(edges * 0.5)**2
    masses = interval_mass(u_edges[:-1], u_edges[1:])
    target = 0.5 * masses
    lower, upper = u_edges[:-1].copy(), u_edges[1:].copy()
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        left = interval_mass(u_edges[:-1], middle) < target
        lower[left], upper[~left] = middle[left], middle[~left]
    theta = 2.0 * np.arcsin(np.sqrt(0.5 * (lower + upper)))
    phi = (np.arange(count) + 0.5) * (2.0 * math.pi / count)
    return (_readonly(np.repeat(theta, count)), _readonly(np.tile(phi, len(theta))),
            _readonly(np.repeat(masses / count, count)))
