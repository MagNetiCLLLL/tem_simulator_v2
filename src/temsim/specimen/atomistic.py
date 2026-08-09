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

import numpy as np

from temsim.specimen.presets import SpecimenPreset

MAX_ATOMISTIC_POTENTIAL_BYTES = 4 * 1024**3


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
) -> None:
    if preset.atomistic is None:
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
):
    """Build an orthogonal, +Z-oriented, commensurate ASE supercell.

    A periodic crystal cannot in general have the exact user-requested box
    length.  Each axis is therefore realised as the nearest positive integer
    number of oriented unit cells.  The caller reports the small realised-size
    difference instead of straining the lattice or introducing a seam.
    """

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
        periodic=True,
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
) -> AtomisticPotentialEnsemble:
    """Build static or frozen-phonon finite-projection potential slices."""

    crystal = preset.atomistic
    preset_sigma = (
        float(crystal.thermal_sigma_angstrom) if crystal is not None else 0.0
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
    )
    if frozen_phonon_enabled and sigma <= 0.0:
        raise ValueError(
            f"{preset.name} needs a positive frozen-phonon RMS displacement."
        )

    equilibrium, _unit_periods = build_equilibrium_atoms(
        preset,
        thickness_angstrom=thickness_angstrom,
        field_of_view_angstrom=field_of_view_angstrom,
    )
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
                sigma,
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
            "user override"
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
        lateral_cell_commensurate=True,
        lateral_mismatch_angstrom=mismatch,
        extent_angstrom_xy=realised_lengths[:2],
        sampling_angstrom_xy=sampling_xy,
        grid_shape_yx=grid_shape_yx,
        thickness_mismatch_angstrom=(
            realised_lengths[2] - thickness_angstrom
        ),
        potential_storage_bytes=potential_storage_bytes,
    )
