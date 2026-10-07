"""Configured objective Cs in the actual posed column checkpoint path.

These checks retain the native 1.2 mm Cs and 300 keV field configuration.
An off-axis narrow packet makes the configured cubic ray kick measurable
without increasing Cs, changing lens strength, or allocating a full column.
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import e

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.column_wave import (
    _propagate_column, _propagate_column_segmented,
)
from temsim.physics.core import execute_propagation_plan
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from test_column_wave_transport import _sample_full_field_at_points
from test_tilted_column_wave import _objective, _prepared, _potential_at, _centroid
from test_wave_detector_readout import checkpoint


def spherical_fixture(*, angle_mrad=1., origin_m=(30e-6, 0.)):
    state = _objective(angle_mrad=angle_mrad, ideal=False, step_mm=.01)
    assert state.lenses[0].cs_mm == 1.2
    prepared = _prepared(state)
    plan = prepared[0]
    axis = (np.arange(64)-32)*5e-9
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(25e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    origin = np.array(origin_m)
    momentum = _momentum_velocity(state.beam_voltage_kv*1000.)[0]
    # The input wave and ray chief both have zero MECHANICAL slope.
    tilt = -e*_potential_at(plan, origin, float(plan.z_mm[0]))[:2]/momentum
    plane = PlaneWave(amplitude, np.eye(2)*5e-9, origin, np.zeros((2, 2)), tilt)
    initial = checkpoint()
    initial = replace(initial, plane_z_mm=float(plan.z_mm[0]),
        beam=replace(initial.beam, modes=(replace(initial.beam.modes[0], plane=plane),)))
    return state, prepared, initial


def zero_spherical(prepared):
    plan = replace(prepared[0], posed_spherical_kicks=tuple(
        replace(kick, strength_m3=0.) for kick in prepared[0].posed_spherical_kicks))
    return (plan, *prepared[1:])


def propagate(state, prepared, source, *, backend="CPU", combine=True):
    return _propagate_column(state, source, float(prepared[0].z_mm[-1]),
        _prepared=prepared, _combine_linear=combine,
        grid_numerics=WaveGridNumerics(compute_backend=backend,
            maximum_pixels=512, maximum_working_bytes=512*1024**2,
            maximum_device_working_bytes=512*1024**2))


def common_complex_field(mode, points):
    return _sample_full_field_at_points(mode.plane, float(wavelength_m(mode.energy_kev*1000.)),
                                       points)*np.sqrt(mode.weight_per_reference_electron)


def field_points(plane):
    xx, yy = np.meshgrid(np.linspace(-40e-9, 40e-9, 9), np.linspace(-40e-9, 40e-9, 7))
    return np.stack((xx.ravel(), yy.ravel()))+_centroid(plane)[:, None]


def action_count(record):
    count = sum(len(mode.get("posed_spherical_actions", ())) for mode in record.get("modes", ()))
    return count+(action_count(record["upstream"]) if "upstream" in record else 0)


def test_zero_posed_cs_is_identity_and_preserves_input_checkpoint():
    state, prepared, source = spherical_fixture()
    assert len(prepared[0].posed_spherical_kicks) == 1
    zero = zero_spherical(prepared)
    absent = (replace(zero[0], posed_spherical_kicks=()), *zero[1:])
    digest = source.digest
    # Identical step sequence isolates zero-operator identity from optional
    # grouping differences in otherwise equivalent linear propagators.
    actual = propagate(state, zero, source, combine=False)
    expected = propagate(state, absent, source, combine=False)
    assert actual.beam.total_weight == pytest.approx(expected.beam.total_weight, abs=2e-12)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(actual.beam.modes[0].plane, name),
            getattr(expected.beam.modes[0].plane, name), rtol=2e-11, atol=2e-12)
    assert source.digest == digest


def test_zero_tilt_cs_matches_existing_axial_column_operator():
    state, prepared, source = spherical_fixture(angle_mrad=0.)
    plan = prepared[0]
    node = int(np.argmax(abs(plan.cs_kick_m3)))
    assert plan.cs_kick_m3[node] > 0. and not plan.posed_spherical_kicks
    kick = FrozenLensAberrationKick(state.lenses[0].key, float(plan.z_mm[node])*1e-3,
                                   float(plan.cs_kick_m3[node]), CoordinateRegistration())
    posed = (replace(plan, cs_kick_m3=np.zeros_like(plan.cs_kick_m3),
                     posed_spherical_kicks=(kick,)), *prepared[1:])
    expected, actual = propagate(state, prepared, source), propagate(state, posed, source)
    points = field_points(expected.beam.modes[0].plane)
    reference = common_complex_field(expected.beam.modes[0], points)
    np.testing.assert_allclose(common_complex_field(actual.beam.modes[0], points), reference,
        rtol=2e-8, atol=2e-10*np.max(abs(reference)))
    assert actual.beam.total_weight == pytest.approx(expected.beam.total_weight, abs=2e-12)


def test_real_gpu_tilted_cs_matches_cpu_and_mechanical_ray_response():
    from test_wave_device import _cuda
    _cuda()
    state, prepared, source = spherical_fixture()
    zero = zero_spherical(prepared)
    with_cs = propagate(state, prepared, source)
    without_cs = propagate(state, zero, source)
    gpu = propagate(state, prepared, source, backend="Require GPU")
    assert gpu.record["modes"][0]["compute_backend"] == "cupy"
    assert len(gpu.record["modes"][0]["posed_spherical_actions"]) == 1
    assert with_cs.beam.total_weight == pytest.approx(source.beam.total_weight, abs=2e-12)
    assert gpu.beam.total_weight == pytest.approx(source.beam.total_weight, abs=2e-12)
    origin = source.beam.modes[0].plane.origin_m
    rays = [execute_propagation_plan(state, candidate[0], np.array((origin[0],)),
                np.zeros(1), np.array((origin[1],)), np.zeros(1)) for candidate in (prepared, zero)]
    ray_response = np.array((rays[0][1][-1, 0]-rays[1][1][-1, 0],
                             rays[0][3][-1, 0]-rays[1][3][-1, 0]))
    wave_response = (_centroid(with_cs.beam.modes[0].plane)-
                     _centroid(without_cs.beam.modes[0].plane))
    assert np.linalg.norm(ray_response) > 5e-11  # A measurable real-Cs effect.
    np.testing.assert_allclose(wave_response, ray_response, rtol=.03, atol=3e-12)
    points = field_points(with_cs.beam.modes[0].plane)
    reference = common_complex_field(with_cs.beam.modes[0], points)
    np.testing.assert_allclose(common_complex_field(gpu.beam.modes[0], points), reference,
        rtol=2e-8, atol=2e-10*np.max(abs(reference)))
    # The configured aberration actually changes phase, not only a record.
    assert np.linalg.norm(reference-common_complex_field(without_cs.beam.modes[0], points))/np.linalg.norm(reference) > .01


def test_tilted_cs_event_executes_once_across_segment_cache_boundary(tmp_path):
    state, prepared, source = spherical_fixture()
    plan = prepared[0]
    assert len(plan.posed_spherical_kicks) == 1
    event_z = plan.posed_spherical_kicks[0].z_mm
    node = int(np.argmin(abs(plan.z_mm-event_z)))
    assert node > 0 and abs(plan.z_mm[node]-event_z) < 1e-9
    target = float(plan.z_mm[-1])
    options = WaveGridNumerics(compute_backend="CPU", maximum_pixels=512,
                               maximum_working_bytes=512*1024**2)
    whole = propagate(state, prepared, source)
    store = ExecutedWaveStore(tmp_path, "tilted-spherical-objective", 1<<27)
    segmented, hit = _propagate_column_segmented(state, source, target, store=store,
        segment_steps=node, _prepared=prepared, grid_numerics=options)
    assert not hit
    assert action_count(whole.record) == action_count(segmented.record) == 1
    points = field_points(whole.beam.modes[0].plane)
    reference = common_complex_field(whole.beam.modes[0], points)
    np.testing.assert_allclose(common_complex_field(segmented.beam.modes[0], points), reference,
        rtol=3e-7, atol=2e-9*np.max(abs(reference)))
    assert segmented.beam.total_weight == pytest.approx(whole.beam.total_weight, abs=2e-12)
    cached, hit = _propagate_column_segmented(state, source, target, store=store,
        segment_steps=node, _prepared=prepared, grid_numerics=options)
    assert hit and cached.digest == segmented.digest
    assert action_count(cached.record) == 1


def test_captured_default_electric_tail_tilted_cs_cpu_gpu(record_property):
    """Actual immutable instrument solve, without removing its weak tail."""
    import json
    import time
    from temsim.optics.column import default_state
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    from test_wave_device import _cuda
    _cuda()
    field = capture_instrument_electric_field(default_state())
    state, prepared, source = spherical_fixture()
    before_digest = source.digest
    prepared = (replace(prepared[0], electric_field=field,
                        electric_field_identity=field.numerical_identity), *prepared[1:])
    assert prepared[0].electric_field is field
    assert field.numerical_identity is not None
    outputs, timings = [], []
    for backend in ("CPU", "Require GPU"):
        before = time.perf_counter()
        output = propagate(state, prepared, source, backend=backend)
        timings.append(time.perf_counter()-before)
        assert output.record["modes"][0]["compute_backend"] == (
            "cupy" if backend == "Require GPU" else "numpy")
        actions = output.record["modes"][0]["posed_spherical_actions"]
        assert len(actions) == 1
        assert actions[0]["electric_phase_remainder_bound_rad"] > 0.
        assert actions[0]["electric_field_bound_v_m"] > 0.
        assert actions[0]["electric_relative_momentum_variation_bound"] > 0.
        assert output.beam.total_weight == pytest.approx(source.beam.total_weight, abs=2e-12)
        outputs.append(output)
    cpu, gpu = outputs
    points = field_points(cpu.beam.modes[0].plane)
    reference = common_complex_field(cpu.beam.modes[0], points)
    actual = common_complex_field(gpu.beam.modes[0], points)
    relative_error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    assert relative_error < 2e-8
    z = np.array((prepared[0].z_mm[0], prepared[0].z_mm[-1]))*1e-3
    potentials = field.potential_rise_v_at_global_positions(np.column_stack((np.zeros(2), np.zeros(2), z)))
    expected_energy = source.beam.modes[0].energy_kev+(potentials[-1]-potentials[0])*1e-3
    assert gpu.beam.modes[0].energy_kev == pytest.approx(expected_energy, abs=1e-12)
    assert source.digest == before_digest and source.beam.modes[0].energy_kev == 300.
    zero = zero_spherical(prepared)
    without_cs = propagate(state, zero, source)
    origin = source.beam.modes[0].plane.origin_m
    rays = [execute_propagation_plan(state, plan[0], np.array((origin[0],)),
        np.zeros(1), np.array((origin[1],)), np.zeros(1)) for plan in (prepared, zero)]
    ray_delta = np.array((rays[0][1][-1, 0]-rays[1][1][-1, 0],
                          rays[0][3][-1, 0]-rays[1][3][-1, 0]))
    wave_delta = _centroid(cpu.beam.modes[0].plane)-_centroid(without_cs.beam.modes[0].plane)
    assert np.linalg.norm(ray_delta) > 5e-11
    np.testing.assert_allclose(wave_delta, ray_delta, rtol=.03, atol=3e-12)
    receipt = gpu.record["modes"][0]["posed_spherical_actions"][0]
    record_property("electric_metrics", json.dumps({
        "scope": "Actual captured default InstrumentElectricField in short objective/Cs propagation; input remains 300 keV and all physical parameters unchanged.",
        "field_type": type(field.base_field).__name__,
        "field_numerical_identity": field.numerical_identity,
        "backend": gpu.record["modes"][0]["compute_backend"],
        "cpu_gpu_relative_complex_l2": relative_error,
        "cpu_seconds": timings[0], "gpu_seconds": timings[1],
        "input_energy_kev": source.beam.modes[0].energy_kev,
        "output_energy_kev": gpu.beam.modes[0].energy_kev,
        "output_probability": gpu.beam.total_weight,
        "ray_cs_centroid_delta_m": ray_delta.tolist(),
        "wave_cs_centroid_delta_m": wave_delta.tolist(),
        "ray_wave_delta_error_m": float(np.linalg.norm(ray_delta-wave_delta)),
        "electric_bounds": {name: float(receipt[name]) for name in (
            "electric_potential_variation_bound_v", "electric_field_bound_v_m",
            "electric_relative_momentum_variation_bound", "electric_momentum_phase_bound_rad",
            "electric_bridge_phase_bound_rad", "electric_phase_remainder_bound_rad")},
    }))
