"""Canonical finite-sample geometry shared by rendering and calculation.

The laboratory frame is right handed and the incident electron propagates
along +Z.  A sample orientation is represented by one unit quaternion in
``(w, x, y, z)`` order.  Zone-axis alignment, numeric tilt controls and the
interactive 3-D editor all update that same physical quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class SampleRegionSnapshot:
    name: str
    kind: str
    enabled: bool
    density: float
    centre_nm: tuple[float, float]
    size_nm: tuple[float, float]
    rotation_deg: float
    map_path: str | None = None


@dataclass(frozen=True, slots=True)
class SampleGeometrySnapshot:
    """Immutable geometry consumed by both the Sample page and STEM models."""

    mode: str
    inserted: bool
    centre_nm: tuple[float, float, float]
    size_nm: tuple[float, float, float]
    orientation_quaternion_wxyz: tuple[float, float, float, float]
    orientation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    zone_axis_uvw: tuple[int, int, int]
    in_plane_axis_uvw: tuple[int, int, int]
    cif_path: str | None
    atom_positions_nm: np.ndarray
    atomic_numbers: np.ndarray
    atom_bond_pairs: np.ndarray
    cell_vectors_nm: np.ndarray
    atoms_are_unit_cell_preview: bool
    atom_display_centre_nm: tuple[float, float, float] | None
    atom_display_size_nm: tuple[float, float, float] | None
    regions: tuple[SampleRegionSnapshot, ...]
    scan_fov_bounds_nm: tuple[float, float, float, float] | None
    calculation_roi_bounds_nm: tuple[float, float, float, float] | None
    current_probe_nm: tuple[float, float] | None
    warnings: tuple[str, ...]


def _finite_vector(name: str, values, length: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values.")
    return array


def normalise_quaternion_wxyz(values) -> tuple[float, float, float, float]:
    quaternion = _finite_vector("Sample orientation quaternion", values, 4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-15:
        raise ValueError("Sample orientation quaternion cannot be zero.")
    quaternion /= norm
    # q and -q encode the same rotation.  Canonicalising the sign makes state
    # comparisons, profile diffs and regression tests deterministic.
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def quaternion_to_matrix(values) -> np.ndarray:
    w, x, y, z = normalise_quaternion_wxyz(values)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Sample orientation matrix must be finite and 3 by 3.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("Sample orientation matrix must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8):
        raise ValueError("Sample orientation matrix must be right handed.")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            values = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            values = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            values = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    return normalise_quaternion_wxyz(values)


def quaternion_multiply(left, right) -> tuple[float, float, float, float]:
    aw, ax, ay, az = normalise_quaternion_wxyz(left)
    bw, bx, by, bz = normalise_quaternion_wxyz(right)
    return normalise_quaternion_wxyz(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quaternion_from_euler_xyz_deg(values) -> tuple[float, float, float, float]:
    angles = _finite_vector("Sample Euler rotation", values, 3)
    rx, ry, rz = np.radians(angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation = np.asarray(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )
    )
    return matrix_to_quaternion_wxyz(rotation)


def quaternion_to_euler_xyz_deg(values) -> tuple[float, float, float]:
    rotation = quaternion_to_matrix(values)
    sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1.0e-10:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        rz = 0.0
    return tuple(float(math.degrees(value)) for value in (rx, ry, rz))


def sample_orientation_quaternion(sample) -> tuple[float, float, float, float]:
    stored = getattr(
        sample,
        "specimen_orientation_quaternion_wxyz",
        IDENTITY_QUATERNION_WXYZ,
    )
    quaternion = normalise_quaternion_wxyz(stored)
    legacy = (
        float(getattr(sample, "specimen_rotation_x_deg", 0.0)),
        float(getattr(sample, "specimen_rotation_y_deg", 0.0)),
        float(getattr(sample, "specimen_rotation_z_deg", 0.0)),
    )
    # A state written before the quaternion field existed receives the new
    # identity default during construction.  Preserve its nonzero Euler
    # orientation exactly once at this compatibility boundary.
    if np.allclose(quaternion, IDENTITY_QUATERNION_WXYZ, atol=1.0e-12) and not np.allclose(
        legacy, 0.0, atol=1.0e-12
    ):
        return quaternion_from_euler_xyz_deg(legacy)
    return quaternion


def set_sample_orientation(sample, quaternion) -> None:
    canonical = normalise_quaternion_wxyz(quaternion)
    sample.specimen_orientation_quaternion_wxyz = canonical
    euler = quaternion_to_euler_xyz_deg(canonical)
    sample.specimen_rotation_x_deg = euler[0]
    sample.specimen_rotation_y_deg = euler[1]
    sample.specimen_rotation_z_deg = euler[2]


def quaternion_from_zone_axes(
    cell_vectors,
    zone_axis_uvw,
    in_plane_axis_uvw,
) -> tuple[float, float, float, float]:
    """Map direct-lattice ``[uvw]`` to +Z and an in-plane direction to +X."""

    cell = np.asarray(cell_vectors, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("Zone alignment requires a finite 3 by 3 direct cell.")
    zone = _finite_vector("Zone axis", zone_axis_uvw, 3) @ cell
    in_plane = _finite_vector("In-plane axis", in_plane_axis_uvw, 3) @ cell
    zone_norm = float(np.linalg.norm(zone))
    if zone_norm <= 1.0e-12:
        raise ValueError("Zone axis cannot be [0 0 0].")
    z_source = zone / zone_norm
    x_source = in_plane - float(in_plane @ z_source) * z_source
    x_norm = float(np.linalg.norm(x_source))
    if x_norm <= 1.0e-12:
        raise ValueError("In-plane axis must not be collinear with the zone axis.")
    x_source /= x_norm
    y_source = np.cross(z_source, x_source)
    y_source /= np.linalg.norm(y_source)
    source_basis = np.column_stack((x_source, y_source, z_source))
    # R @ source_basis == I: source x/y/z become laboratory +X/+Y/+Z.
    return matrix_to_quaternion_wxyz(source_basis.T)


def sample_scan_bounds_nm(scan_x_um, scan_y_um, sample) -> tuple[float, float, float, float] | None:
    if scan_x_um is None or scan_y_um is None:
        return None
    # Scan result coordinates are already absolute laboratory sample-plane
    # positions; the scan-origin offset is applied once in acquire_stem_scan.
    x = np.asarray(scan_x_um, dtype=float) * 1.0e3
    y = np.asarray(scan_y_um, dtype=float) * 1.0e3
    if x.shape != y.shape or x.size == 0 or not (
        np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    ):
        raise ValueError("Sample scan coordinates must be matching finite arrays.")
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def calculation_roi_bounds_nm(
    scan_bounds_nm,
    *,
    probe_padding_nm: float,
) -> tuple[float, float, float, float] | None:
    if scan_bounds_nm is None:
        return None
    padding = float(probe_padding_nm)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("Probe padding must be finite and non-negative.")
    x0, x1, y0, y1 = scan_bounds_nm
    return x0 - padding, x1 + padding, y0 - padding, y1 + padding


def _region_snapshots(sample) -> tuple[SampleRegionSnapshot, ...]:
    result = []
    for index, raw in enumerate(getattr(sample, "virtual_regions", ()) or ()):
        if not isinstance(raw, dict):
            raise ValueError("Each virtual-sample region must be a table.")
        kind = str(raw.get("kind", "rectangle")).strip().lower()
        if kind not in {"rectangle", "ellipse", "map"}:
            raise ValueError(f"Virtual region {index + 1}: unsupported kind {kind!r}.")
        density = float(raw.get("density", 1.0))
        if not math.isfinite(density) or not 0.0 <= density <= 1.0:
            raise ValueError(
                f"Virtual region {index + 1}: density must be in [0, 1]."
            )
        centre = (
            float(raw.get("centre_x_nm", 0.0)),
            float(raw.get("centre_y_nm", 0.0)),
        )
        size = (
            float(raw.get("size_x_nm", getattr(sample, "size_x_nm", 0.0))),
            float(raw.get("size_y_nm", getattr(sample, "size_y_nm", 0.0))),
        )
        if not all(math.isfinite(value) for value in (*centre, *size)) or min(size) <= 0.0:
            raise ValueError(
                f"Virtual region {index + 1}: centre and positive size must be finite."
            )
        map_path = str(raw.get("map_path", "")).strip() or None
        if kind == "map" and map_path is None:
            raise ValueError(f"Virtual region {index + 1}: map path is required.")
        result.append(
            SampleRegionSnapshot(
                name=str(raw.get("name", f"Region {index + 1}")),
                kind=kind,
                enabled=bool(raw.get("enabled", True)),
                density=density,
                centre_nm=centre,
                size_nm=size,
                rotation_deg=float(raw.get("rotation_deg", 0.0)),
                map_path=map_path,
            )
        )
    return tuple(result)


def read_cif_preview(
    cif_path: str,
    orientation_matrix,
    *,
    requested_bounds_nm: tuple[float, float, float, float],
    thickness_nm: float,
    specimen_size_xy_nm: tuple[float, float],
    specimen_centre_xy_nm: tuple[float, float],
    maximum_atoms: int = 2_500,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    tuple[float, float, float],
    bool,
    tuple[str, ...],
]:
    """Build a bounded, repeated CIF ball-and-stick display model.

    The requested X/Y range is the finite-sample/scan-ROI intersection in
    laboratory nanometres.  abTEM performs the same cell orthogonalisation as
    the IAM calculation, while ASE expands and crops the periodic structure.
    Very large requested volumes are represented by a centred display window
    whose atom count is bounded; this is a rendering limit only and never
    changes the multislice calculation ROI.
    """

    path = Path(cif_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"CIF file does not exist: {path}")
    if path.suffix.lower() not in {".cif", ".mcif"}:
        raise ValueError("Atomic specimen import requires a CIF or MCIF file.")
    try:
        from ase.io import read
        from ase.neighborlist import natural_cutoffs, neighbor_list

        unit = read(path)
    except Exception as exc:
        raise ValueError(f"Unable to read CIF file {path}: {exc}") from exc
    if len(unit) == 0:
        raise ValueError(f"CIF file contains no atoms: {path}")
    cell_angstrom = np.asarray(unit.cell.array, dtype=float)
    if cell_angstrom.shape != (3, 3) or abs(float(np.linalg.det(cell_angstrom))) <= 1.0e-12:
        raise ValueError("CIF structure must define a finite three-dimensional cell.")
    rotation = np.asarray(orientation_matrix, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
    ):
        raise ValueError("CIF display orientation must be a finite rotation matrix.")
    maximum_atoms = int(maximum_atoms)
    if maximum_atoms < 100:
        raise ValueError("CIF display atom limit must be at least 100.")
    x0, x1, y0, y1 = (float(value) for value in requested_bounds_nm)
    requested_size = np.asarray((x1 - x0, y1 - y0, float(thickness_nm)))
    if not np.all(np.isfinite(requested_size)) or np.any(requested_size <= 0.0):
        raise ValueError("CIF display range and thickness must be finite and positive.")
    requested_centre = np.asarray((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0))

    cell_volume_nm3 = abs(float(np.linalg.det(cell_angstrom))) * 1.0e-3
    density_atoms_nm3 = float(len(unit)) / cell_volume_nm3
    expected_atoms = density_atoms_nm3 * float(np.prod(requested_size))
    display_size = requested_size.copy()
    warnings: list[str] = []
    if expected_atoms > float(maximum_atoms):
        # Grow a window in units of the three lattice-vector lengths.  This
        # gives a useful multi-cell crystal view even when the requested
        # sample is macroscopic or strongly anisotropic.
        lattice_nm = np.linalg.norm(cell_angstrom, axis=1) * 0.1
        base = np.minimum(requested_size, np.maximum(lattice_nm, 1.0e-6))
        target_atoms = max(0.72 * float(maximum_atoms), float(len(unit)))
        if density_atoms_nm3 * float(np.prod(base)) >= target_atoms:
            scale = (
                target_atoms
                / max(density_atoms_nm3 * float(np.prod(base)), 1.0e-30)
            ) ** (1.0 / 3.0)
            display_size = np.minimum(requested_size, base * scale)
        else:
            lower = 1.0
            upper = max(float(np.max(requested_size / base)), 1.0)
            for _ in range(64):
                middle = 0.5 * (lower + upper)
                candidate = np.minimum(requested_size, base * middle)
                if density_atoms_nm3 * float(np.prod(candidate)) <= target_atoms:
                    lower = middle
                else:
                    upper = middle
            display_size = np.minimum(requested_size, base * lower)
        warnings.append(
            "Atomic rendering window was reduced from "
            f"{requested_size[0]:.6g} x {requested_size[1]:.6g} x "
            f"{requested_size[2]:.6g} nm to {display_size[0]:.6g} x "
            f"{display_size[1]:.6g} x {display_size[2]:.6g} nm to stay near "
            f"the {maximum_atoms:,}-atom display limit. The calculation ROI is unchanged."
        )

    from temsim.specimen.atomistic import build_cif_equilibrium_atoms

    repeated, periods_angstrom = build_cif_equilibrium_atoms(
        path,
        thickness_angstrom=float(display_size[2] * 10.0),
        field_of_view_angstrom=tuple(display_size[:2] * 10.0),
        rotation_matrix=rotation,
        specimen_size_xy_angstrom=tuple(
            np.asarray(specimen_size_xy_nm, dtype=float) * 10.0
        ),
        specimen_centre_xy_angstrom=tuple(
            np.asarray(specimen_centre_xy_nm, dtype=float) * 10.0
        ),
        calculation_roi_centre_xy_angstrom=tuple(
            requested_centre[:2] * 10.0
        ),
    )
    positions_nm = np.asarray(repeated.positions, dtype=float) * 0.1
    positions_nm -= 0.5 * display_size
    positions_nm[:, :2] += (
        requested_centre[:2]
        - np.asarray(specimen_centre_xy_nm, dtype=float)
    )
    numbers = np.asarray(repeated.numbers, dtype=int)
    cutoffs = natural_cutoffs(repeated, mult=1.15)
    first, second = neighbor_list(
        "ij",
        repeated,
        cutoffs,
        self_interaction=False,
    )
    unique = np.asarray(first, dtype=int) < np.asarray(second, dtype=int)
    bonds = np.column_stack(
        (np.asarray(first, dtype=int)[unique], np.asarray(second, dtype=int)[unique])
    )
    if bonds.size == 0:
        bonds = np.empty((0, 2), dtype=int)
    cell_nm = np.diag(np.asarray(periods_angstrom, dtype=float)) @ rotation.T * 0.1
    if len(repeated) > maximum_atoms:
        warnings.append(
            f"The boundary-complete display contains {len(repeated):,} atoms, "
            f"slightly above the requested {maximum_atoms:,}-atom soft limit."
        )
    for array in (positions_nm, numbers, bonds, cell_nm):
        array.setflags(write=False)
    return (
        positions_nm,
        numbers,
        bonds,
        cell_nm,
        tuple(float(value) for value in requested_centre),
        tuple(float(value) for value in display_size),
        False,
        tuple(warnings),
    )


def build_sample_geometry_snapshot(
    sample,
    *,
    scan_x_um=None,
    scan_y_um=None,
    current_probe_nm: tuple[float, float] | None = None,
    probe_padding_nm: float = 0.0,
    load_atoms: bool = True,
    maximum_display_atoms: int = 2_500,
) -> SampleGeometrySnapshot:
    mode = str(getattr(sample, "specimen_mode", "atomic")).strip().lower()
    if mode not in {"atomic", "virtual"}:
        raise ValueError("Sample mode must be atomic or virtual.")
    size = (
        float(getattr(sample, "size_x_nm", 0.0)),
        float(getattr(sample, "size_y_nm", 0.0)),
        float(getattr(sample, "thickness_nm", 0.0)),
    )
    if not all(math.isfinite(value) and value > 0.0 for value in size[:2]) or not (
        math.isfinite(size[2]) and size[2] >= 0.0
    ):
        raise ValueError(
            "Sample X/Y size must be positive and thickness must be non-negative."
        )
    centre = (
        float(getattr(sample, "centre_x_nm", 0.0)),
        float(getattr(sample, "centre_y_nm", 0.0)),
        0.0,
    )
    if not all(math.isfinite(value) for value in centre):
        raise ValueError("Sample centre must be finite.")
    quaternion = sample_orientation_quaternion(sample)
    orientation = quaternion_to_matrix(quaternion)
    zone = tuple(int(value) for value in getattr(sample, "zone_axis_uvw", (0, 0, 1)))
    in_plane = tuple(int(value) for value in getattr(sample, "in_plane_axis_uvw", (1, 0, 0)))
    if len(zone) != 3 or len(in_plane) != 3:
        raise ValueError("Zone and in-plane axes must each contain three integers.")
    scan_bounds = sample_scan_bounds_nm(scan_x_um, scan_y_um, sample)
    roi = calculation_roi_bounds_nm(
        scan_bounds,
        probe_padding_nm=probe_padding_nm,
    )
    warnings: list[str] = []
    cif = str(getattr(sample, "cif_path", "")).strip() or None
    atom_positions = np.empty((0, 3), dtype=float)
    atomic_numbers = np.empty(0, dtype=int)
    atom_bonds = np.empty((0, 2), dtype=int)
    cell_vectors = np.empty((0, 3), dtype=float)
    unit_preview = False
    atom_display_centre = None
    atom_display_size = None
    if mode == "atomic" and cif and load_atoms:
        sample_bounds = (
            centre[0] - 0.5 * size[0],
            centre[0] + 0.5 * size[0],
            centre[1] - 0.5 * size[1],
            centre[1] + 0.5 * size[1],
        )
        requested = roi or scan_bounds or sample_bounds
        display_bounds = (
            max(sample_bounds[0], requested[0]),
            min(sample_bounds[1], requested[1]),
            max(sample_bounds[2], requested[2]),
            min(sample_bounds[3], requested[3]),
        )
        if display_bounds[1] <= display_bounds[0] or display_bounds[3] <= display_bounds[2]:
            warnings.append(
                "The requested atomic display range is outside the finite sample; it contains vacuum only."
            )
        elif size[2] <= 0.0:
            warnings.append(
                "The zero-thickness sample has no repeated atomic volume to display."
            )
        else:
            try:
                (
                    atom_positions,
                    atomic_numbers,
                    atom_bonds,
                    cell_vectors,
                    atom_display_centre,
                    atom_display_size,
                    unit_preview,
                    cif_warnings,
                ) = read_cif_preview(
                    cif,
                    orientation,
                    requested_bounds_nm=display_bounds,
                    thickness_nm=size[2],
                    specimen_size_xy_nm=size[:2],
                    specimen_centre_xy_nm=centre[:2],
                    maximum_atoms=maximum_display_atoms,
                )
            except ValueError as exc:
                if "contains no atomic sites" not in str(exc):
                    raise
                warnings.append(
                    "The selected display window contains no CIF atom centres; "
                    "the surrounding finite sample geometry remains valid."
                )
            else:
                warnings.extend(cif_warnings)
    for array in (atom_positions, atomic_numbers, atom_bonds, cell_vectors):
        array.setflags(write=False)
    if mode == "atomic" and not cif:
        warnings.append(
            "No CIF is selected; the active TOML specimen preset supplies the calculation structure."
        )
    if not bool(getattr(sample, "inserted", True)):
        warnings.append(
            "Sample is retracted: this page retains geometry, but electron-sample interaction is disabled."
        )
    if roi is not None:
        x0, x1, y0, y1 = roi
        sx0 = centre[0] - 0.5 * size[0]
        sx1 = centre[0] + 0.5 * size[0]
        sy0 = centre[1] - 0.5 * size[1]
        sy1 = centre[1] + 0.5 * size[1]
        if x0 < sx0 or x1 > sx1 or y0 < sy0 or y1 > sy1:
            warnings.append(
                "The calculation ROI extends outside the finite sample; those probe positions are vacuum."
            )
    matrix_tuple = tuple(tuple(float(value) for value in row) for row in orientation)
    return SampleGeometrySnapshot(
        mode=mode,
        inserted=bool(getattr(sample, "inserted", True)),
        centre_nm=centre,
        size_nm=size,
        orientation_quaternion_wxyz=quaternion,
        orientation_matrix=matrix_tuple,
        zone_axis_uvw=zone,
        in_plane_axis_uvw=in_plane,
        cif_path=cif,
        atom_positions_nm=atom_positions,
        atomic_numbers=atomic_numbers,
        atom_bond_pairs=atom_bonds,
        cell_vectors_nm=cell_vectors,
        atoms_are_unit_cell_preview=unit_preview,
        atom_display_centre_nm=atom_display_centre,
        atom_display_size_nm=atom_display_size,
        regions=_region_snapshots(sample),
        scan_fov_bounds_nm=scan_bounds,
        calculation_roi_bounds_nm=roi,
        current_probe_nm=current_probe_nm,
        warnings=tuple(warnings),
    )
