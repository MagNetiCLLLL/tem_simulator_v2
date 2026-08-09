"""Reusable CuPy execution plan for repeated multislice propagation.

The plan fixes one spatial grid, wavelength, slice geometry and anti-alias
bandwidth.  It caches the reciprocal-frequency grid, bandwidth mask and all
Fresnel propagators on one CUDA device, then reuses them for every leading
probe batch and frozen-phonon potential configuration.

The NumPy implementation in :mod:`temsim.physics.multislice` remains the
complex128 scientific reference.  This module is a separately validated
complex64 acceleration path rather than a replacement for that reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from temsim.physics.compute_backend import WAVE_BACKEND_CUPY, cupy_module
from temsim.physics.multislice import (
    MultisliceDiagnostics,
    PixelSizeAngstrom,
    _frequency_grid,
    _intensity_per_wave,
    _pixel_spacing_yx,
    _propagate,
    _propagator,
    _validate_wave_and_sampling,
)


def _unique_device_bytes(arrays) -> int:
    seen = set()
    total = 0
    for array in arrays:
        if array is None or id(array) in seen:
            continue
        seen.add(id(array))
        total += int(array.nbytes)
    return total


@dataclass
class CuPyMultislicePlan:
    """Cached device state for one fixed multislice discretisation."""

    cupy: object = field(repr=False, compare=False)
    potential_shape: tuple[int, ...]
    potential_ndim: int
    spatial_shape_yx: tuple[int, int]
    pixel_size_y_angstrom: float
    pixel_size_x_angstrom: float
    wavelength_angstrom: float
    interaction_constant_rad_per_v_angstrom: float
    slice_thicknesses_angstrom: np.ndarray
    bandwidth_fraction: float
    model: str
    potential_divisor: int
    maximum_isotropic_angle_mrad: float
    frequency_squared: object = field(repr=False, compare=False)
    bandwidth_mask: object = field(repr=False, compare=False)
    propagator_before_first_slice: object = field(
        repr=False,
        compare=False,
    )
    propagators_after_slices: tuple = field(repr=False, compare=False)
    cached_device_bytes: int
    cached_propagator_count: int
    build_elapsed_s: float
    use_count: int = 0

    @classmethod
    def build(
        cls,
        projected_potential_v_angstrom,
        *,
        pixel_size_angstrom: PixelSizeAngstrom,
        wavelength_angstrom: float,
        interaction_constant_rad_per_v_angstrom: float,
        total_thickness_angstrom: float | None = None,
        target_slice_thickness_angstrom: float = 2.0,
        slice_thicknesses_angstrom: np.ndarray | None = None,
        bandwidth_fraction: float = 2.0 / 3.0,
        cupy=None,
    ) -> "CuPyMultislicePlan":
        """Create and cache all grid- and thickness-dependent device arrays."""

        cp = cupy if cupy is not None else cupy_module()
        started = time.perf_counter()
        potential = cp.asarray(
            projected_potential_v_angstrom,
            dtype=cp.float32,
        )
        if potential.ndim not in (2, 3):
            raise ValueError(
                "Projected potential must have shape (Y, X) or (Z, Y, X)."
            )
        spatial_shape = tuple(int(value) for value in potential.shape[-2:])
        if min(spatial_shape) < 2:
            raise ValueError("Potential spatial axes must be non-trivial.")
        if not bool(cp.all(cp.isfinite(potential)).item()):
            raise ValueError("Projected potential contains NaN or infinity.")
        spacing_y, spacing_x = _pixel_spacing_yx(pixel_size_angstrom)
        wavelength = float(wavelength_angstrom)
        sigma = float(interaction_constant_rad_per_v_angstrom)
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("Electron wavelength must be finite and positive.")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Interaction constant must be finite and positive.")

        if potential.ndim == 2:
            if total_thickness_angstrom is None:
                raise ValueError(
                    "Total thickness is required for a two-dimensional total "
                    "projected potential."
                )
            total_thickness = float(total_thickness_angstrom)
            target_slice = float(target_slice_thickness_angstrom)
            if not math.isfinite(total_thickness) or total_thickness < 0.0:
                raise ValueError("Total specimen thickness cannot be negative.")
            if not math.isfinite(target_slice) or target_slice <= 0.0:
                raise ValueError("Target slice thickness must be positive.")
            if total_thickness == 0.0:
                slice_count = 0
                thicknesses = np.empty(0, dtype=np.float64)
            else:
                slice_count = max(
                    1,
                    int(math.ceil(total_thickness / target_slice)),
                )
                thicknesses = np.full(
                    slice_count,
                    total_thickness / slice_count,
                    dtype=np.float64,
                )
            model = "continuous_column_multislice"
            potential_divisor = max(slice_count, 1)
        else:
            slice_count = int(potential.shape[0])
            if slice_count == 0:
                raise ValueError(
                    "A three-dimensional potential needs at least one slice."
                )
            if slice_thicknesses_angstrom is None:
                target_slice = float(target_slice_thickness_angstrom)
                if not math.isfinite(target_slice) or target_slice <= 0.0:
                    raise ValueError("Slice thickness must be positive.")
                thicknesses = np.full(
                    slice_count,
                    target_slice,
                    dtype=np.float64,
                )
            else:
                thicknesses = np.asarray(
                    slice_thicknesses_angstrom,
                    dtype=np.float64,
                )
                if thicknesses.shape != (slice_count,):
                    raise ValueError(
                        "Slice thickness array must match the Z dimension."
                    )
            if not np.all(np.isfinite(thicknesses)) or np.any(
                thicknesses <= 0.0
            ):
                raise ValueError(
                    "Every physical slice thickness must be positive."
                )
            total_thickness = float(np.sum(thicknesses))
            model = "atomic_slice_multislice"
            potential_divisor = 1

        fraction = float(bandwidth_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("Bandwidth fraction must be in (0, 1].")
        _, _, frequency_squared = _frequency_grid(
            spatial_shape,
            (spacing_y, spacing_x),
            xp=cp,
            real_dtype=cp.float32,
        )
        nyquist = min(0.5 / spacing_y, 0.5 / spacing_x)
        cutoff = fraction * nyquist
        if fraction >= 1.0:
            bandwidth_mask = cp.ones(spatial_shape, dtype=cp.bool_)
        else:
            bandwidth_mask = frequency_squared <= cutoff**2
        maximum_angle_mrad = (
            math.asin(min(wavelength * cutoff, 1.0)) * 1.0e3
        )

        before_first = None
        after_slices = ()
        if slice_count:
            before_first = _propagator(
                frequency_squared,
                wavelength,
                0.5 * thicknesses[0],
                bandwidth_mask,
                xp=cp,
                complex_dtype=cp.complex64,
            )
            uniform = bool(
                np.allclose(
                    thicknesses,
                    thicknesses[0],
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
            if uniform:
                full = (
                    _propagator(
                        frequency_squared,
                        wavelength,
                        thicknesses[0],
                        bandwidth_mask,
                        xp=cp,
                        complex_dtype=cp.complex64,
                    )
                    if slice_count > 1
                    else None
                )
                after_slices = tuple(
                    full if index + 1 < slice_count else before_first
                    for index in range(slice_count)
                )
            else:
                propagators = []
                for index in range(slice_count):
                    if index + 1 < slice_count:
                        distance = 0.5 * (
                            thicknesses[index] + thicknesses[index + 1]
                        )
                    else:
                        distance = 0.5 * thicknesses[index]
                    propagators.append(
                        _propagator(
                            frequency_squared,
                            wavelength,
                            distance,
                            bandwidth_mask,
                            xp=cp,
                            complex_dtype=cp.complex64,
                        )
                    )
                after_slices = tuple(propagators)

        unique_propagators = tuple(
            dict.fromkeys(
                id(array)
                for array in (before_first,) + after_slices
                if array is not None
            )
        )
        cached_arrays = (
            frequency_squared,
            bandwidth_mask,
            before_first,
        ) + after_slices
        cp.cuda.get_current_stream().synchronize()
        return cls(
            cupy=cp,
            potential_shape=tuple(int(value) for value in potential.shape),
            potential_ndim=int(potential.ndim),
            spatial_shape_yx=spatial_shape,
            pixel_size_y_angstrom=spacing_y,
            pixel_size_x_angstrom=spacing_x,
            wavelength_angstrom=wavelength,
            interaction_constant_rad_per_v_angstrom=sigma,
            slice_thicknesses_angstrom=thicknesses,
            bandwidth_fraction=fraction,
            model=model,
            potential_divisor=potential_divisor,
            maximum_isotropic_angle_mrad=maximum_angle_mrad,
            frequency_squared=frequency_squared,
            bandwidth_mask=bandwidth_mask,
            propagator_before_first_slice=before_first,
            propagators_after_slices=after_slices,
            cached_device_bytes=_unique_device_bytes(cached_arrays),
            cached_propagator_count=len(unique_propagators),
            build_elapsed_s=time.perf_counter() - started,
        )

    @property
    def slice_count(self) -> int:
        return int(self.slice_thicknesses_angstrom.size)

    @property
    def total_thickness_angstrom(self) -> float:
        return float(np.sum(self.slice_thicknesses_angstrom))

    @property
    def maximum_slice_thickness_angstrom(self) -> float:
        if not self.slice_count:
            return 0.0
        return float(np.max(self.slice_thicknesses_angstrom))

    def _coerce_potential(
        self,
        projected_potential_v_angstrom,
        *,
        check_finite: bool,
    ):
        potential = self.cupy.asarray(
            projected_potential_v_angstrom,
            dtype=self.cupy.float32,
        )
        if tuple(int(value) for value in potential.shape) != self.potential_shape:
            raise ValueError(
                "Potential shape does not match the cached CUDA multislice plan."
            )
        if check_finite and not bool(
            self.cupy.all(self.cupy.isfinite(potential)).item()
        ):
            raise ValueError("Projected potential contains NaN or infinity.")
        return potential

    def validate_potential(self, projected_potential_v_angstrom):
        """Validate one resident potential once before repeated plan use."""

        return self._coerce_potential(
            projected_potential_v_angstrom,
            check_finite=True,
        )

    def maximum_phase_per_slice_rad(
        self,
        projected_potential_v_angstrom,
        *,
        potential_is_validated: bool = False,
    ) -> float:
        potential = self._coerce_potential(
            projected_potential_v_angstrom,
            check_finite=not potential_is_validated,
        )
        if not self.slice_count:
            return 0.0
        maximum = self.cupy.max(self.cupy.abs(potential))
        return float(
            (
                self.interaction_constant_rad_per_v_angstrom
                * maximum
                / self.potential_divisor
            ).item()
        )

    def propagate(
        self,
        incident_wave,
        projected_potential_v_angstrom,
        *,
        maximum_phase_per_slice_rad: float | None = None,
        fallback_reason: str | None = None,
        potential_is_validated: bool = False,
    ):
        """Propagate one leading probe batch using the cached device plan."""

        cp = self.cupy
        wave = _validate_wave_and_sampling(
            incident_wave,
            (self.pixel_size_y_angstrom, self.pixel_size_x_angstrom),
            xp=cp,
            complex_dtype=cp.complex64,
        )
        if tuple(int(value) for value in wave.shape[-2:]) != self.spatial_shape_yx:
            raise ValueError(
                "Wave spatial shape does not match the cached CUDA plan."
            )
        potential = self._coerce_potential(
            projected_potential_v_angstrom,
            check_finite=not potential_is_validated,
        )
        initial_intensity = _intensity_per_wave(wave, xp=cp)
        if bool(cp.any(initial_intensity <= 0.0).item()):
            raise ValueError("Every incident wave must contain positive intensity.")
        maximum_relative_change = cp.asarray(0.0, dtype=cp.float64)
        if maximum_phase_per_slice_rad is None:
            maximum_phase = self.maximum_phase_per_slice_rad(
                potential,
                potential_is_validated=True,
            )
        else:
            maximum_phase = float(maximum_phase_per_slice_rad)
            if not math.isfinite(maximum_phase) or maximum_phase < 0.0:
                raise ValueError("Maximum phase diagnostic must be finite and non-negative.")

        if self.potential_ndim == 2:
            slice_potential = potential / self.potential_divisor
        else:
            slice_potential = None

        if self.slice_count:
            wave = _propagate(
                wave,
                self.propagator_before_first_slice,
                xp=cp,
            )
            for index, propagator in enumerate(
                self.propagators_after_slices
            ):
                if self.potential_ndim == 2:
                    phase_potential = slice_potential
                else:
                    phase_potential = potential[index]
                transmission = cp.exp(
                    1j
                    * self.interaction_constant_rad_per_v_angstrom
                    * phase_potential
                ).astype(cp.complex64, copy=False)
                wave *= transmission
                wave = _propagate(wave, propagator, xp=cp)
                current_intensity = _intensity_per_wave(wave, xp=cp)
                relative_change = cp.max(
                    cp.abs(current_intensity - initial_intensity)
                    / initial_intensity
                )
                maximum_relative_change = cp.maximum(
                    maximum_relative_change,
                    relative_change,
                )

        final_intensity = _intensity_per_wave(wave, xp=cp)
        self.use_count += 1
        diagnostics = MultisliceDiagnostics(
            model=self.model,
            slice_count=self.slice_count,
            total_thickness_angstrom=self.total_thickness_angstrom,
            slice_thickness_angstrom=(
                self.maximum_slice_thickness_angstrom
            ),
            bandwidth_fraction=self.bandwidth_fraction,
            maximum_isotropic_angle_mrad=(
                self.maximum_isotropic_angle_mrad
            ),
            maximum_phase_per_slice_rad=maximum_phase,
            initial_integrated_intensity=float(
                cp.sum(initial_intensity).item()
            ),
            final_integrated_intensity=float(
                cp.sum(final_intensity).item()
            ),
            maximum_relative_intensity_change=float(
                maximum_relative_change.item()
            ),
            compute_backend=WAVE_BACKEND_CUPY,
            numeric_precision="complex64 / float32",
            fallback_reason=fallback_reason,
            pixel_size_y_angstrom=self.pixel_size_y_angstrom,
            pixel_size_x_angstrom=self.pixel_size_x_angstrom,
        )
        return wave, diagnostics
