import numpy as np
import pytest

from temsim.physics import compute_backend
from temsim.physics.cuda_multislice_plan import CuPyMultislicePlan
from temsim.physics.multislice import propagate_multislice


def _normalised_wave(shape):
    ny, nx = shape
    y = (np.arange(ny) - ny // 2) * 0.27
    x = (np.arange(nx) - nx // 2) * 0.19
    xx, yy = np.meshgrid(x, y, indexing="xy")
    wave = np.exp(
        -((xx / 1.8) ** 2 + (yy / 2.3) ** 2)
        + 1j * (0.21 * xx - 0.14 * yy)
    )
    return wave / np.linalg.norm(wave)


def _require_cupy():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    return compute_backend.cupy_module()


def test_cuda_plan_reuses_rectangular_continuous_column_state():
    cp = _require_cupy()
    shape = (28, 36)
    wave = _normalised_wave(shape)
    y = (np.arange(shape[0]) - shape[0] // 2) * 0.27
    x = (np.arange(shape[1]) - shape[1] // 2) * 0.19
    xx, yy = np.meshgrid(x, y, indexing="xy")
    potential = 28.0 * np.exp(
        -(xx**2 + yy**2) / (2.0 * 0.72**2)
    )
    parameters = {
        "pixel_size_angstrom": (0.27, 0.19),
        "wavelength_angstrom": 0.0197,
        "interaction_constant_rad_per_v_angstrom": 0.0065,
        "total_thickness_angstrom": 7.0,
        "target_slice_thickness_angstrom": 2.0,
        "bandwidth_fraction": 2.0 / 3.0,
    }
    reference, _ = propagate_multislice(wave, potential, **parameters)
    device_potential = cp.asarray(potential, dtype=cp.float32)
    plan = CuPyMultislicePlan.build(
        device_potential,
        **parameters,
        cupy=cp,
    )
    validated = plan.validate_potential(device_potential)
    maximum_phase = plan.maximum_phase_per_slice_rad(
        validated,
        potential_is_validated=True,
    )
    frequency_squared_id = id(plan.frequency_squared)
    propagator_ids = tuple(
        id(array) for array in plan.propagators_after_slices
    )

    first, diagnostics = plan.propagate(
        cp.asarray(wave),
        validated,
        maximum_phase_per_slice_rad=maximum_phase,
        potential_is_validated=True,
    )
    second, _ = plan.propagate(
        cp.asarray(wave),
        validated,
        maximum_phase_per_slice_rad=maximum_phase,
        potential_is_validated=True,
    )
    first_host = cp.asnumpy(first)
    second_host = cp.asnumpy(second)

    relative_error = np.linalg.norm(first_host - reference) / np.linalg.norm(
        reference
    )
    assert relative_error < 3.0e-5
    assert second_host == pytest.approx(first_host, rel=2.0e-6, abs=2.0e-7)
    assert plan.use_count == 2
    assert id(plan.frequency_squared) == frequency_squared_id
    assert (
        tuple(id(array) for array in plan.propagators_after_slices)
        == propagator_ids
    )
    assert plan.slice_count == 4
    assert plan.cached_propagator_count == 2
    assert plan.cached_device_bytes > 0
    assert diagnostics.maximum_phase_per_slice_rad == pytest.approx(
        0.0065 * np.max(potential) / 4.0,
        rel=2.0e-6,
    )


def test_cuda_plan_matches_nonuniform_explicit_atomic_slices():
    cp = _require_cupy()
    shape = (30, 24)
    wave = _normalised_wave(shape)
    y = (np.arange(shape[0]) - shape[0] // 2) * 0.23
    x = (np.arange(shape[1]) - shape[1] // 2) * 0.31
    xx, yy = np.meshgrid(x, y, indexing="xy")
    slices = np.stack(
        (
            12.0 * np.exp(-((xx + 0.3) ** 2 + yy**2) / 0.8**2),
            18.0 * np.exp(-(xx**2 + (yy - 0.2) ** 2) / 0.65**2),
            9.0 * np.exp(-((xx - 0.2) ** 2 + (yy + 0.4) ** 2) / 0.9**2),
        ),
        axis=0,
    )
    thicknesses = np.asarray([0.7, 1.4, 0.9])
    parameters = {
        "pixel_size_angstrom": (0.23, 0.31),
        "wavelength_angstrom": 0.0251,
        "interaction_constant_rad_per_v_angstrom": 0.0071,
        "total_thickness_angstrom": None,
        "slice_thicknesses_angstrom": thicknesses,
        "bandwidth_fraction": 0.75,
    }
    reference, reference_diagnostics = propagate_multislice(
        wave,
        slices,
        **parameters,
    )
    device_slices = cp.asarray(slices, dtype=cp.float32)
    plan = CuPyMultislicePlan.build(
        device_slices,
        **parameters,
        cupy=cp,
    )
    result, diagnostics = plan.propagate(
        cp.asarray(wave),
        device_slices,
    )
    result_host = cp.asnumpy(result)

    relative_error = np.linalg.norm(result_host - reference) / np.linalg.norm(
        reference
    )
    assert relative_error < 3.0e-5
    assert plan.model == "atomic_slice_multislice"
    assert plan.cached_propagator_count == 4
    assert diagnostics.total_thickness_angstrom == pytest.approx(3.0)
    assert diagnostics.slice_thickness_angstrom == pytest.approx(1.4)
    assert diagnostics.maximum_phase_per_slice_rad == pytest.approx(
        reference_diagnostics.maximum_phase_per_slice_rad,
        rel=2.0e-6,
    )


def test_cuda_plan_rejects_a_different_potential_shape():
    cp = _require_cupy()
    plan = CuPyMultislicePlan.build(
        cp.zeros((16, 20), dtype=cp.float32),
        pixel_size_angstrom=0.25,
        wavelength_angstrom=0.0197,
        interaction_constant_rad_per_v_angstrom=0.0065,
        total_thickness_angstrom=4.0,
        cupy=cp,
    )

    with pytest.raises(ValueError, match="does not match"):
        plan.validate_potential(cp.zeros((16, 21), dtype=cp.float32))
