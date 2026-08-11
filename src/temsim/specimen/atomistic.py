"""Atomistic independent-atom specimen potentials for multislice.

The specimen frame is right-handed with the incident electron travelling
along +Z.  Atomic coordinates, real-space sampling and slice boundaries are
in angstrom.  Every returned potential plane is already integrated through
its finite Z slice and therefore has units of volt-angstrom, matching
``temsim.physics.multislice.propagate_multislice``.

The optional abTEM/ASE backend uses the Lobato--Van Dyck neutral-atom
parameterization in the independent atom model (IAM).  IAM neglects bonding
charge redistribution.  Frozen phonons use independent isotropic Gaussian
displacements (the Einstein approximation); correlated phonons are outside
this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path

import numpy as np

from temsim.specimen.presets import SpecimenPreset

MAX_ATOMISTIC_POTENTIAL_BYTES = 4 * 1024**3
MAX_CIF_SUPERCELL_ATOMS = 5_000_000


@dataclass(frozen=True)
class AtomisticCapability:
    available: bool
    detail: str


@dataclass(frozen=True)
class AtomisticPotentialEnsemble:
    configurations_v_angstrom: tuple[np.ndarray, ...]
    slice_thicknesses_angstrom: np.ndarray
    mean_projected_potential_v_angstrom: np.ndarray
    atom_count: int
    configuration_count: int
    thermal_sigma_angstrom: float
    thermal_seed: int
    thermal_model: str
    thermal_sigma_reference: str
    parametrization: str
    projection: str
    builder_backend: str
    lateral_cell_commensurate: bool
    lateral_mismatch_angstrom: tuple[float, float]
    extent_angstrom_xy: tuple[float, float]
    sampling_angstrom_xy: tuple[float, float]
    grid_shape_yx: tuple[int, int]
    thickness_mismatch_angstrom: float
    potential_storage_bytes: int
    source_kind: str
    source_path: str | None
    rotation_deg_xyz: tuple[float, float, float]

    @property
    def slice_count(self) -> int:
        return int(self.slice_thicknesses_angstrom.size)

    @property
    def total_thickness_angstrom(self) -> float:
        return float(np.sum(self.slice_thicknesses_angstrom))


class AtomisticBackendUnavailable(RuntimeError):
    """Raised when an atomistic preset cannot be built in this installation."""


def atomistic_capability() -> AtomisticCapability:
    try:
        import abtem  # noqa: F401
        import ase  # noqa: F401

        return AtomisticCapability(
            True,
            f"abTEM {version('abtem')}; ASE {version('ase')}",
        )
    except Exception as exc:
        return AtomisticCapability(
            False,
            f"Atomistic backend unavailable: {exc}",
        )


def _require_backend():
    capability = atomistic_capability()
    if not capability.available:
        raise AtomisticBackendUnavailable(capability.detail)
    import abtem
    from ase import Atoms
    from ase.build import bulk, surface
    from ase.data import atomic_numbers

    return abtem, Atoms, bulk, surface, atomic_numbers, capability.detail


def _validate_request(
    preset: SpecimenPreset,
    *,
    thickness_angstrom: float,
    field_of_view_angstrom: float,
    pixels: int,
    target_slice_thickness_angstrom: float,
    frozen_phonon_configurations: int,
    thermal_sigma_angstrom: float,
    thermal_seed: int,
    cif_path: str = "",
) -> None:
    if preset.atomistic is None and not str(cif_path).strip():
        raise AtomisticBackendUnavailable(
            f"{preset.name} has no 3-D atomistic crystal definition."
        )
    values = (
        thickness_angstrom,
        field_of_view_angstrom,
        target_slice_thickness_angstrom,
        thermal_sigma_angstrom,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Atomistic specimen dimensions must be finite.")
    if thickness_angstrom <= 0.0:
        raise ValueError("Atomistic specimen thickness must be positive.")
    if field_of_view_angstrom <= 0.0 or int(pixels) < 32:
        raise ValueError("Atomistic wave grid is invalid.")
    if target_slice_thickness_angstrom <= 0.0:
        raise ValueError("Atomistic target slice thickness must be positive.")
    if not 1 <= int(frozen_phonon_configurations) <= 64:
        raise ValueError("Frozen-phonon configurations must be between 1 and 64.")
    if thermal_sigma_angstrom < 0.0:
        raise ValueError("Frozen-phonon RMS displacement cannot be negative.")
    if int(thermal_seed) < 0:
        raise ValueError("Frozen-phonon seed cannot be negative.")


def build_equilibrium_atoms(
    preset: SpecimenPreset,
    *,
    thickness_angstrom: float,
    field_of_view_angstrom: float,
    cif_path: str = "",
    rotation_deg_xyz=(0.0, 0.0, 0.0),
    rotation_matrix=None,
    specimen_size_xy_angstrom=None,
    specimen_centre_xy_angstrom=(0.0, 0.0),
    calculation_roi_centre_xy_angstrom=(0.0, 0.0),
):
    """Build an orthogonal, +Z-oriented, commensurate ASE supercell.

    A periodic crystal cannot in general have the exact user-requested box
    length.  Each axis is therefore realised as the nearest positive integer
    number of oriented unit cells.  The caller reports the small realised-size
    difference instead of straining the lattice or introducing a seam.
    """

    if str(cif_path).strip():
        return build_cif_equilibrium_atoms(
            cif_path,
            thickness_angstrom=thickness_angstrom,
            field_of_view_angstrom=field_of_view_angstrom,
            rotation_deg_xyz=rotation_deg_xyz,
            rotation_matrix=rotation_matrix,
            specimen_size_xy_angstrom=specimen_size_xy_angstrom,
            specimen_centre_xy_angstrom=specimen_centre_xy_angstrom,
            calculation_roi_centre_xy_angstrom=(
                calculation_roi_centre_xy_angstrom
            ),
        )

    (
        _abtem,
        Atoms,
        bulk,
        surface,
        atomic_numbers,
        _detail,
    ) = _require_backend()
    crystal = preset.atomistic
    if crystal is None:
        raise AtomisticBackendUnavailable(
            f"{preset.name} has no 3-D atomistic crystal definition."
        )
    try:
        expected_atomic_number = int(atomic_numbers[crystal.chemical_symbol])
    except KeyError as exc:
        raise ValueError(
            f"Unknown chemical symbol {crystal.chemical_symbol!r}."
        ) from exc
    if expected_atomic_number != crystal.atomic_number:
        raise ValueError(
            f"{preset.name}: symbol/atomic-number mismatch in atomistic TOML."
        )

    conventional = bulk(
        crystal.chemical_symbol,
        crystal.crystal_structure,
        a=crystal.lattice_constant_angstrom,
        cubic=True,
    )
    unit = surface(
        conventional,
        crystal.zone_axis,
        layers=1,
        periodic=True,
    )
    unit.wrap(eps=1.0e-12)
    cell = np.asarray(unit.cell.array, dtype=float)
    off_diagonal = cell - np.diag(np.diag(cell))
    if not np.allclose(off_diagonal, 0.0, atol=1.0e-10):
        raise ValueError(
            f"{preset.name}: oriented ASE cell is not orthogonal."
        )
    lengths = np.diag(cell)
    if np.any(lengths <= 0.0):
        raise ValueError(f"{preset.name}: oriented ASE cell is invalid.")

    repeats = tuple(
        max(1, int(round(requested / period)))
        for requested, period in zip(
            (field_of_view_angstrom, field_of_view_angstrom, thickness_angstrom),
            lengths,
        )
    )
    repeated = unit.repeat(repeats)
    repeated.wrap(eps=1.0e-12)
    positions = np.asarray(repeated.positions, dtype=float)
    if positions.size == 0:
        raise ValueError(f"{preset.name}: generated atomistic cell is empty.")
    atoms = Atoms(
        numbers=np.asarray(repeated.numbers),
        positions=positions,
        cell=repeated.cell,
        pbc=True,
    )
    atoms.wrap(eps=1.0e-12)
    return atoms, tuple(float(value) for value in lengths)


def _rotation_matrix_xyz(rotation_deg_xyz) -> np.ndarray:
    values = tuple(float(value) for value in rotation_deg_xyz)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("CIF specimen rotation must contain three finite angles.")
    rx, ry, rz = (math.radians(value) for value in values)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rotation_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rotation_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    # Extrinsic right-handed X, then Y, then Z rotations in the specimen
    # frame; the electron propagation direction remains laboratory +Z.
    return rotation_z @ rotation_y @ rotation_x


def build_cif_equilibrium_atoms(
    cif_path,
    *,
    thickness_angstrom: float,
    field_of_view_angstrom: float,
    rotation_deg_xyz=(0.0, 0.0, 0.0),
    rotation_matrix=None,
    specimen_size_xy_angstrom=None,
    specimen_centre_xy_angstrom=(0.0, 0.0),
    calculation_roi_centre_xy_angstrom=(0.0, 0.0),
):
    """Load a CIF and crop a finite sample inside the calculation ROI."""

    abtem, Atoms, *_ = _require_backend()
    from ase.io import read

    path = Path(cif_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"CIF file does not exist: {path}")
    if path.suffix.lower() not in {".cif", ".mcif"}:
        raise ValueError("Atomic specimen import requires a CIF or MCIF file.")
    try:
        unit = read(path)
    except Exception as exc:
        raise ValueError(f"Unable to read CIF file {path}: {exc}") from exc
    if len(unit) == 0:
        raise ValueError(f"CIF file contains no atoms: {path}")
    if np.any(np.asarray(unit.cell.lengths(), dtype=float) <= 0.0):
        raise ValueError("CIF structure must define a finite three-dimensional cell.")
    try:
        unit = abtem.orthogonalize_cell(
            unit,
            max_repetitions=5,
            allow_transform=True,
        )
    except Exception as exc:
        raise ValueError(f"Unable to orthogonalize CIF cell: {exc}") from exc
    unit.wrap(eps=1.0e-12)
    periods = np.asarray(unit.cell.lengths(), dtype=float)
    field_of_view = np.asarray(field_of_view_angstrom, dtype=float)
    if field_of_view.shape == ():
        field_of_view_xy = np.repeat(float(field_of_view), 2)
    elif field_of_view.shape == (2,):
        field_of_view_xy = field_of_view
    else:
        raise ValueError(
            "CIF calculation field of view must be one value or an X/Y pair."
        )
    calculation_target = np.asarray(
        (*field_of_view_xy, float(thickness_angstrom)),
        dtype=float,
    )
    specimen_xy = np.asarray(
        specimen_size_xy_angstrom
        if specimen_size_xy_angstrom is not None
        else (field_of_view_angstrom, field_of_view_angstrom),
        dtype=float,
    )
    specimen_centre = np.asarray(specimen_centre_xy_angstrom, dtype=float)
    roi_centre = np.asarray(
        calculation_roi_centre_xy_angstrom,
        dtype=float,
    )
    if (
        not np.all(np.isfinite(calculation_target))
        or np.any(calculation_target <= 0.0)
        or specimen_xy.shape != (2,)
        or np.any(~np.isfinite(specimen_xy))
        or np.any(specimen_xy <= 0.0)
        or specimen_centre.shape != (2,)
        or roi_centre.shape != (2,)
        or not np.all(np.isfinite(specimen_centre))
        or not np.all(np.isfinite(roi_centre))
    ):
        raise ValueError("CIF specimen box dimensions must be finite and positive.")
    rotation = (
        _rotation_matrix_xyz(rotation_deg_xyz)
        if rotation_matrix is None
        else np.asarray(rotation_matrix, dtype=float)
    )
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8)
    ):
        raise ValueError("CIF specimen rotation matrix must be right-handed and orthonormal.")
    physical_target = np.r_[specimen_xy, float(thickness_angstrom)]
    sample_min = specimen_centre - 0.5 * specimen_xy
    sample_max = specimen_centre + 0.5 * specimen_xy
    roi_min = roi_centre - 0.5 * calculation_target[:2]
    roi_max = roi_centre + 0.5 * calculation_target[:2]
    intersection_min = np.maximum(sample_min, roi_min)
    intersection_max = np.minimum(sample_max, roi_max)
    intersection_size = intersection_max - intersection_min
    if np.any(intersection_size <= 0.0):
        raise ValueError(
            "The calculation ROI does not intersect the finite CIF specimen."
        )
    # Generate only a lattice neighbourhood large enough to cover the
    # intersecting ROI after arbitrary rotation.  Its centre is snapped by an
    # integer lattice translation, preserving CIF phase without constructing
    # the rest of a macroscopic sample.
    local_target = np.r_[intersection_size, float(thickness_angstrom)]
    cover_length = float(np.linalg.norm(local_target)) + 2.0 * float(np.max(periods))
    repeat_estimates = np.ceil(cover_length / periods) + 1.0
    if (
        not np.all(np.isfinite(repeat_estimates))
        or np.any(repeat_estimates > MAX_CIF_SUPERCELL_ATOMS)
    ):
        raise ValueError(
            "Requested CIF specimen dimensions exceed the finite-supercell "
            "safety limit. Reduce specimen width or thickness."
        )
    repeats = tuple(
        max(1, int(value)) for value in repeat_estimates
    )
    estimated_atoms = int(len(unit)) * math.prod(repeats)
    if estimated_atoms > MAX_CIF_SUPERCELL_ATOMS:
        raise ValueError(
            "Rotated CIF supercell would contain approximately "
            f"{estimated_atoms:,} atoms before cropping, above the "
            f"{MAX_CIF_SUPERCELL_ATOMS:,}-atom safety limit. Reduce specimen "
            "width or thickness."
        )
    repeated = unit.repeat(repeats)
    repeated.wrap(eps=1.0e-12)
    repeated_centre_lattice = (
        np.asarray(repeats, dtype=int) // 2
    ) * periods
    intersection_centre_lab = 0.5 * (
        intersection_min + intersection_max
    ) - specimen_centre
    target_source = np.asarray(
        (intersection_centre_lab[0], intersection_centre_lab[1], 0.0)
    ) @ rotation
    lattice_shift = np.rint(target_source / periods) * periods
    positions = (
        np.asarray(repeated.positions, dtype=float)
        - repeated_centre_lattice
        + lattice_shift
    )
    rotated = positions @ rotation.T
    physical_half = 0.5 * physical_target
    tolerance = 1.0e-9
    physical_keep = np.all(np.abs(rotated) <= physical_half + tolerance, axis=1)
    lab_xy = rotated[:, :2] + specimen_centre
    local_xy = lab_xy - roi_centre
    roi_keep = np.all(
        np.abs(local_xy) <= 0.5 * calculation_target[:2] + tolerance,
        axis=1,
    )
    keep = physical_keep & roi_keep
    if not np.any(keep):
        raise ValueError(
            "The requested rotated CIF specimen box contains no atomic sites."
        )
    atoms = Atoms(
        numbers=np.asarray(repeated.numbers)[keep],
        positions=np.column_stack(
            (
                local_xy[keep] + 0.5 * calculation_target[:2],
                rotated[keep, 2] + 0.5 * calculation_target[2],
            )
        ),
        cell=np.diag(calculation_target),
        pbc=False,
    )
    return atoms, tuple(float(value) for value in periods)


def _build_one_potential(
    atoms,
    *,
    gpts_xy: tuple[int, int],
    target_slice_thickness_angstrom: float,
):
    abtem, *_ = _require_backend()
    abtem.config.set({"diagnostics.progress_bar": False})
    built = abtem.Potential(
        atoms,
        gpts=tuple(int(value) for value in gpts_xy),
        slice_thickness=float(target_slice_thickness_angstrom),
        parametrization="lobato",
        projection="finite",
        periodic=bool(np.all(np.asarray(atoms.pbc, dtype=bool))),
        device="cpu",
    ).build(lazy=False)
    # abTEM stores real-space axes as (X, Y); temsim and NumPy imaging use
    # (..., Y, X), so transpose exactly once at this boundary.
    array = np.asarray(built.array, dtype=np.float32).transpose(0, 2, 1)
    array = np.fft.fftshift(array, axes=(-2, -1))
    thicknesses = np.asarray(built.slice_thickness, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != thicknesses.size:
        raise RuntimeError("abTEM returned an invalid potential-slice array.")
    if not np.all(np.isfinite(array)):
        raise RuntimeError("abTEM potential contains NaN or infinity.")
    return array, thicknesses


def build_atomistic_potential_ensemble(
    preset: SpecimenPreset,
    *,
    thickness_angstrom: float,
    field_of_view_angstrom: float,
    pixels: int,
    target_slice_thickness_angstrom: float,
    frozen_phonon_enabled: bool,
    frozen_phonon_configurations: int,
    thermal_sigma_override_angstrom: float,
    thermal_seed: int,
    cif_path: str = "",
    rotation_deg_xyz=(0.0, 0.0, 0.0),
    rotation_matrix=None,
    specimen_size_xy_angstrom=None,
    specimen_centre_xy_angstrom=(0.0, 0.0),
    calculation_roi_centre_xy_angstrom=(0.0, 0.0),
    thermal_sigma_by_element_angstrom=None,
) -> AtomisticPotentialEnsemble:
    """Build static or frozen-phonon finite-projection potential slices."""

    crystal = preset.atomistic
    custom_cif = bool(str(cif_path).strip())
    preset_sigma = (
        float(crystal.thermal_sigma_angstrom)
        if crystal is not None and not custom_cif
        else 0.0
    )
    sigma = (
        float(thermal_sigma_override_angstrom)
        if float(thermal_sigma_override_angstrom) > 0.0
        else preset_sigma
    )
    configuration_count = (
        int(frozen_phonon_configurations) if frozen_phonon_enabled else 1
    )
    _validate_request(
        preset,
        thickness_angstrom=thickness_angstrom,
        field_of_view_angstrom=field_of_view_angstrom,
        pixels=pixels,
        target_slice_thickness_angstrom=target_slice_thickness_angstrom,
        frozen_phonon_configurations=configuration_count,
        thermal_sigma_angstrom=sigma,
        thermal_seed=thermal_seed,
        cif_path=str(cif_path),
    )
    element_sigma = dict(thermal_sigma_by_element_angstrom or {})
    if frozen_phonon_enabled and sigma <= 0.0 and not element_sigma:
        raise ValueError(
            f"{preset.name} needs a positive frozen-phonon RMS displacement "
            "or an explicit per-element RMS table."
        )

    equilibrium, _unit_periods = build_equilibrium_atoms(
        preset,
        thickness_angstrom=thickness_angstrom,
        field_of_view_angstrom=field_of_view_angstrom,
        cif_path=str(cif_path),
        rotation_deg_xyz=rotation_deg_xyz,
        rotation_matrix=rotation_matrix,
        specimen_size_xy_angstrom=specimen_size_xy_angstrom,
        specimen_centre_xy_angstrom=specimen_centre_xy_angstrom,
        calculation_roi_centre_xy_angstrom=(
            calculation_roi_centre_xy_angstrom
        ),
    )
    per_atom_sigma = None
    if frozen_phonon_enabled and element_sigma:
        converted = {}
        for symbol, value in element_sigma.items():
            sigma_value = float(value)
            if not math.isfinite(sigma_value) or sigma_value <= 0.0:
                raise ValueError(
                    f"Frozen-phonon RMS for {symbol} must be finite and positive."
                )
            converted[str(symbol)] = sigma_value
        symbols = equilibrium.get_chemical_symbols()
        missing = sorted(set(symbols) - set(converted))
        if missing:
            raise ValueError(
                "Custom CIF frozen phonons need explicit RMS values for: "
                + ", ".join(missing)
            )
        per_atom_sigma = np.asarray(
            [converted[symbol] for symbol in symbols],
            dtype=float,
        )
        sigma = float(np.sqrt(np.mean(per_atom_sigma**2)))
    configurations = []
    common_thicknesses = None
    realised_lengths = tuple(float(value) for value in equilibrium.cell.lengths())
    requested_sampling = field_of_view_angstrom / int(pixels)
    gpts_xy = tuple(
        max(16, int(round(length / requested_sampling)))
        for length in realised_lengths[:2]
    )
    estimated_slices = max(
        1,
        int(
            math.ceil(
                realised_lengths[2] / target_slice_thickness_angstrom
            )
        ),
    )
    estimated_storage = (
        configuration_count
        * estimated_slices
        * gpts_xy[0]
        * gpts_xy[1]
        * np.dtype(np.float32).itemsize
        + gpts_xy[0]
        * gpts_xy[1]
        * np.dtype(np.float64).itemsize
    )
    if estimated_storage > MAX_ATOMISTIC_POTENTIAL_BYTES:
        raise ValueError(
            "Atomistic potential ensemble would require approximately "
            f"{estimated_storage / 1024**3:.2f} GiB, above the 4 GiB "
            "specimen-potential safety limit. Reduce grid pixels, thickness "
            "or frozen-phonon configurations."
        )
    seed_sequence = np.random.SeedSequence(int(thermal_seed))
    child_seeds = seed_sequence.spawn(configuration_count)
    for index in range(configuration_count):
        atoms = equilibrium.copy()
        if frozen_phonon_enabled:
            rng = np.random.default_rng(child_seeds[index])
            atoms.positions += rng.normal(
                0.0,
                (
                    per_atom_sigma[:, None]
                    if per_atom_sigma is not None
                    else sigma
                ),
                size=atoms.positions.shape,
            )
            atoms.wrap(eps=1.0e-12)
        array, thicknesses = _build_one_potential(
            atoms,
            gpts_xy=gpts_xy,
            target_slice_thickness_angstrom=target_slice_thickness_angstrom,
        )
        if common_thicknesses is None:
            common_thicknesses = thicknesses
        elif not np.array_equal(common_thicknesses, thicknesses):
            raise RuntimeError(
                "Frozen-phonon configurations produced inconsistent slices."
            )
        configurations.append(array)

    if common_thicknesses is None:
        raise RuntimeError("No atomistic potential configurations were built.")
    grid_shape_yx = (gpts_xy[1], gpts_xy[0])
    projected_sum = np.zeros(grid_shape_yx, dtype=np.float64)
    for array in configurations:
        projected_sum += np.sum(array, axis=0, dtype=np.float64)
    projected_sum /= configuration_count

    mismatch = tuple(
        realised - field_of_view_angstrom
        for realised in realised_lengths[:2]
    )
    sampling_xy = tuple(
        realised / gpts
        for realised, gpts in zip(realised_lengths[:2], gpts_xy)
    )
    capability = atomistic_capability()
    potential_storage_bytes = int(
        sum(array.nbytes for array in configurations)
        + projected_sum.nbytes
        + common_thicknesses.nbytes
    )
    return AtomisticPotentialEnsemble(
        configurations_v_angstrom=tuple(configurations),
        slice_thicknesses_angstrom=common_thicknesses,
        mean_projected_potential_v_angstrom=projected_sum,
        atom_count=len(equilibrium),
        configuration_count=configuration_count,
        thermal_sigma_angstrom=sigma if frozen_phonon_enabled else 0.0,
        thermal_seed=int(thermal_seed),
        thermal_model=(
            "independent_isotropic_gaussian_einstein"
            if frozen_phonon_enabled
            else "static_equilibrium_atoms"
        ),
        thermal_sigma_reference=(
            "explicit per-element user table"
            if frozen_phonon_enabled and per_atom_sigma is not None
            else "user override"
            if frozen_phonon_enabled and thermal_sigma_override_angstrom > 0.0
            else (
                crystal.thermal_sigma_reference
                if frozen_phonon_enabled and crystal is not None
                else "not applicable"
            )
        ),
        parametrization="Lobato-Van Dyck 2014 neutral-atom IAM",
        projection="finite Z-slice projection",
        builder_backend=capability.detail,
        lateral_cell_commensurate=not custom_cif,
        lateral_mismatch_angstrom=mismatch,
        extent_angstrom_xy=realised_lengths[:2],
        sampling_angstrom_xy=sampling_xy,
        grid_shape_yx=grid_shape_yx,
        thickness_mismatch_angstrom=(
            realised_lengths[2] - thickness_angstrom
        ),
        potential_storage_bytes=potential_storage_bytes,
        source_kind="cif" if custom_cif else "toml_crystal",
        source_path=(
            str(Path(cif_path).expanduser().resolve()) if custom_cif else None
        ),
        rotation_deg_xyz=tuple(float(value) for value in rotation_deg_xyz),
    )
