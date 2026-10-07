"""Short captured-objective wave/ray comparisons, not full TEM qualification.

The shared analytic lens remains locally paraxial. These checks use a narrow
beam and small installation errors, compare mechanical ray coordinates with
the correctly gauged wave, and execute the ordinary column checkpoint path.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e

from temsim.optics.column import default_state
from temsim.physics.column_wave import (
    _column_transports, _prepare_column, _propagate_column, _propagate_column_segmented,
)
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.core import execute_propagation_plan
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from test_wave_detector_readout import checkpoint


KEY = "objective_lens"


def _objective(*, angle_mrad=0., offset_mm=0., step_mm=.01, ideal=True):
    source = default_state()
    lens = next(item for item in source.lenses if item.key == KEY)
    assembly = replace(source._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": angle_mrad,
                           "offset_x_mm": offset_mm}) if part.key == KEY else part
        for part in source._resolved_assembly.parts))
    return SimpleNamespace(lenses=[lens], condenser_system=source.condenser_system,
        _resolved_assembly=assembly, stigmators=[], corrector_elements=[], deflectors=[],
        apertures=[], recording_planes=[], simulation_mode="ideal" if ideal else "analytical",
        beam_voltage_kv=300., step_mm=step_mm, history_step_mm=step_mm,
        acceleration_enabled=False, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=source.sample)


def _potential_at(plan, xy_m, z_mm):
    """Independent symmetric-gauge expression for the captured Gaussian Bz."""
    potential = np.zeros(3)
    for field in plan.mapped_fields:
        local = field.registration.positions_global_to_local_m(
            np.array((*xy_m, z_mm*1e-3)))
        bz = sum(amplitude*np.exp(-.5*((local[2]-centre)/sigma)**2)
                 for amplitude, centre, sigma in field.terms_t_m)
        potential += field.registration.vectors_local_to_global(
            np.array((-.5*local[1]*bz, .5*local[0]*bz, 0.)))
    if not plan.mapped_fields:
        bz = float(plan.magnetic_t[0])
        potential += np.array((-.5*xy_m[1]*bz, .5*xy_m[0]*bz, 0.))
    return potential


def _source(state, prepared, *, centred=False):
    plan = prepared[0]
    axis = (np.arange(64)-32)*5e-9
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(25e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    origin = np.zeros(2) if centred else np.array((20e-9, -15e-9))
    momentum, _ = _momentum_velocity(state.beam_voltage_kv*1000.)
    # q=-e; canonical p_perp/p = mechanical slope + q*A_perp/p.
    # This represents zero mechanical chief-ray slope at the entrance.
    tilt = -e*_potential_at(plan, origin, float(plan.z_mm[0]))[:2]/momentum
    plane = PlaneWave(amplitude, np.eye(2)*5e-9, origin,
                      np.zeros((2, 2)), tilt)
    old = checkpoint()
    mode = replace(old.beam.modes[0], plane=plane)
    return replace(old, plane_z_mm=float(plan.z_mm[0]),
                   beam=replace(old.beam, modes=(mode,)))


def _prepared(state):
    centre = state.lenses[0].z_mm
    return _prepare_column(state, centre-.2, centre+.2, state.step_mm)


def _centroid(wave):
    intensity = abs(wave.amplitude)**2
    return np.sum(wave.coordinates_m()*intensity, axis=(1, 2))/intensity.sum()


@pytest.mark.parametrize("angle_mrad", [-1., 1.])
def test_tilted_objective_wave_centroid_matches_shared_mechanical_ray(angle_mrad):
    state = _objective(angle_mrad=angle_mrad)
    prepared = _prepared(state)
    assert len(prepared[0].mapped_fields) == 1
    initial = _source(state, prepared)
    origin = initial.beam.modes[0].plane.origin_m
    ray = execute_propagation_plan(state, prepared[0], np.array([origin[0]]),
        np.zeros(1), np.array([origin[1]]), np.zeros(1))
    result = _propagate_column(state, initial, float(prepared[0].z_mm[-1]),
        _prepared=prepared, grid_numerics=WaveGridNumerics(compute_backend="CPU"))
    expected = np.array((ray[1][-1, 0], ray[3][-1, 0]))
    # 0.2 nm absolute tolerance resolves the tilt-induced displacement while
    # allowing midpoint integration and the stated paraxial truncation.
    np.testing.assert_allclose(_centroid(result.beam.modes[0].plane), expected,
                               rtol=2e-3, atol=2e-10)
    assert result.beam.total_weight == pytest.approx(initial.beam.total_weight, rel=1e-10)


def test_tilt_sign_reverses_centroid_and_zero_pose_limit_is_continuous():
    outputs = []
    for angle in (-1., 0., 1e-7, 1.):
        state = _objective(angle_mrad=angle)
        prepared = _prepared(state)
        source = _source(state, prepared, centred=True)
        output = _propagate_column(state, source, float(prepared[0].z_mm[-1]),
            _prepared=prepared, grid_numerics=WaveGridNumerics(compute_backend="CPU"))
        outputs.append(output.beam.modes[0].plane)
    np.testing.assert_allclose(_centroid(outputs[0]), -_centroid(outputs[3]), atol=2e-12)
    assert np.linalg.norm(_centroid(outputs[3])) > 1e-9
    np.testing.assert_allclose(_centroid(outputs[1]), _centroid(outputs[2]), atol=2e-12)
    np.testing.assert_allclose(outputs[1].basis_m, outputs[2].basis_m, rtol=2e-6, atol=1e-16)
    np.testing.assert_allclose(outputs[1].curvature_m1, outputs[2].curvature_m1,
                               rtol=2e-6, atol=1e-5)


def test_tilted_objective_wave_and_ray_first_order_focusing_use_same_coordinates():
    state = _objective(angle_mrad=1.)
    prepared = _prepared(state)
    plan = prepared[0]
    path = CanonicalPath(1e-3)
    for child in _column_transports(plan, state.beam_voltage_kv):
        path.append(child.matrix, child.offset, action_m=child.action_m)
    momentum, _ = _momentum_velocity(state.beam_voltage_kv*1000.)

    def canonical_from_mechanical(xy, z):
        # Independently differentiate the vector potential used to express
        # both results in the same LAB x,y,dx/dz,dy/dz coordinate convention.
        transform = np.eye(4)
        displacement = np.eye(2)*1e-9
        gradient = np.column_stack([
            (_potential_at(plan, xy+d, z)[:2]-_potential_at(plan, xy-d, z)[:2])/(2e-9)
            for d in displacement])
        transform[2:, :2] = -e*gradient/momentum
        return transform

    steps = np.array((1e-9, 1e-9, 1e-6, 1e-6))
    inputs = np.column_stack((np.zeros(4), np.diag(steps), -np.diag(steps)))
    ray = execute_propagation_plan(state, plan, inputs[0], inputs[2], inputs[1], inputs[3])
    final = np.stack((ray[1][-1], ray[3][-1], ray[2][-1], ray[4][-1]))
    ray_matrix = (final[:, 1:5]-final[:, 5:9])/(2*steps)
    before = canonical_from_mechanical(np.zeros(2), float(plan.z_mm[0]))
    after = canonical_from_mechanical(final[:2, 0], float(plan.z_mm[-1]))
    wave_matrix = np.linalg.solve(after, path.matrix@before)
    # Compare dimensionless matrices using a fixed 1 mm reference length.
    # This covers magnification, angle-to-position, focusing and rotation.
    scale = np.diag((1e-3, 1e-3, 1., 1.))
    np.testing.assert_allclose(np.linalg.solve(scale, wave_matrix@scale),
                               np.linalg.solve(scale, ray_matrix@scale),
                               rtol=5e-4, atol=2e-6)


def test_tilted_objective_segment_cache_reuses_and_pose_changes_identity(tmp_path):
    state = _objective(angle_mrad=.5)
    prepared = _prepared(state)
    source = _source(state, prepared)
    target = float(prepared[0].z_mm[-1])
    options = WaveGridNumerics(compute_backend="CPU")
    whole = _propagate_column(state, source, target, _prepared=prepared,
                              grid_numerics=options)
    store = ExecutedWaveStore(tmp_path, "tilted-objective", 1<<27)
    first, hit = _propagate_column_segmented(state, source, target, store=store,
        segment_steps=10, _prepared=prepared, grid_numerics=options)
    assert not hit
    cached, hit = _propagate_column_segmented(state, source, target, store=store,
        segment_steps=10, _prepared=prepared, grid_numerics=options)
    assert hit and cached.digest == first.digest
    np.testing.assert_allclose(_centroid(first.beam.modes[0].plane),
                               _centroid(whole.beam.modes[0].plane), atol=2e-12)
    assert first.beam.total_weight == pytest.approx(whole.beam.total_weight, rel=1e-10)
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from test_column_wave_transport import _sample_full_field_at_points
    yy, xx = np.meshgrid(np.linspace(-40e-9, 40e-9, 7),
                          np.linspace(-40e-9, 40e-9, 9), indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()))+_centroid(whole.beam.modes[0].plane)[:, None]
    wavelength = float(wavelength_m(whole.beam.modes[0].energy_kev*1000.))
    expected = _sample_full_field_at_points(whole.beam.modes[0].plane, wavelength, points)
    actual = _sample_full_field_at_points(first.beam.modes[0].plane, wavelength, points)
    # Compare the absolute complex wave at identical physical positions:
    # no phase fitting, normalization or assumption of identical lattices.
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9*np.max(abs(expected)))
    assert _prepared(_objective(angle_mrad=-.5))[0].signature != prepared[0].signature


def test_real_gpu_tilted_objective_matches_cpu_complex_checkpoint():
    from test_wave_device import _cuda
    _cuda()
    state = _objective(angle_mrad=.5)
    prepared = _prepared(state)
    source = _source(state, prepared)
    target = float(prepared[0].z_mm[-1])
    cpu = _propagate_column(state, source, target, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend="CPU"))
    gpu = _propagate_column(state, source, target, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend="Require GPU"))
    assert all(row["compute_backend"] == "cupy" for row in gpu.record["modes"])
    expected, actual = cpu.beam.modes[0], gpu.beam.modes[0]
    assert actual.weight_per_reference_electron == pytest.approx(
        expected.weight_per_reference_electron, abs=1e-12)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(actual.plane, name), getattr(expected.plane, name),
                                   rtol=2e-9, atol=2e-12)
    assert isinstance(actual.plane.amplitude, np.ndarray)
    assert not actual.plane.amplitude.flags.writeable


def test_real_column_higher_order_magnetic_residual_runs_on_cpu_and_gpu():
    """A stricter numerical budget resolves, rather than drops, native orders."""
    from test_wave_device import _cuda
    _cuda()
    state = _objective(angle_mrad=1.)
    prepared = _prepared(state)
    source = _source(state, prepared)
    target = float(prepared[0].z_mm[-1])
    outputs = []
    for backend in ("CPU", "Require GPU"):
        result = _propagate_column(state, source, target, _prepared=prepared,
            grid_numerics=WaveGridNumerics(compute_backend=backend,
                maximum_posed_lens_phase_error_rad_per_m=1e-7))
        records = result.record["modes"][0]["magnetic_residual_steps"]
        assert records  # A pure quadratic shortcut cannot satisfy this check.
        expected_backend = "cupy" if backend == "Require GPU" else "numpy"
        assert all(record["compute_backend"] == expected_backend for record in records)
        assert result.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-10)
        outputs.append(result.beam.modes[0].plane)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_allclose(getattr(outputs[0], name), getattr(outputs[1], name),
                                   rtol=2e-9, atol=2e-12)
    # No fitted global phase or normalization: retain the absolute complex
    # checkpoint needed for later coherent propagation and interference.
    np.testing.assert_allclose(outputs[0].amplitude, outputs[1].amplitude,
                               rtol=2e-9, atol=2e-12)


def test_magnetic_residual_rechecks_forward_support_after_refinement(monkeypatch):
    """The real doubled Nyquist support must be checked before any retry."""
    from temsim.physics import wave_grid, wave_magnetic_residual
    state = _objective(angle_mrad=1.)
    prepared = _prepared(state)
    source = _source(state, prepared)
    mode = source.beam.modes[0]
    # At 300 keV this lattice's represented transverse momentum has norm
    # about .07 p; doubling its resolution raises that above the .1 slope
    # ceiling. The physical-domain calculation is the production check.
    plane = replace(mode.plane, basis_m=np.eye(2)*2e-11)
    source = replace(source, beam=replace(source.beam, modes=(replace(mode, plane=plane),)))
    original_digest = source.digest
    operator_shapes, refined_shapes = [], []
    real_refine = wave_grid.refine_plane_wave

    def request_one_refinement(wave, *args, **kwargs):
        operator_shapes.append(wave.amplitude.shape)
        if len(operator_shapes) > 1:
            raise AssertionError("Unsupported refined support reached the magnetic operator")
        raise wave_grid.WaveSamplingError("Controlled sampling request", 2.)

    def record_real_refinement(wave, shape, **kwargs):
        refined = real_refine(wave, shape, **kwargs)
        refined_shapes.append(refined.amplitude.shape)
        return refined

    monkeypatch.setattr(wave_magnetic_residual, "apply_magnetic_residual", request_one_refinement)
    monkeypatch.setattr(wave_grid, "refine_plane_wave", record_real_refinement)
    with pytest.raises(ValueError, match="forward paraxial domain"):
        _propagate_column(state, source, float(prepared[0].z_mm[-1]), _prepared=prepared,
            _combine_linear=False,
            grid_numerics=WaveGridNumerics(compute_backend="CPU",
                maximum_posed_lens_phase_error_rad_per_m=1e-14))
    assert operator_shapes == [(64, 64)]
    assert refined_shapes == [(128, 128)]
    assert source.digest == original_digest


def test_nonzero_posed_spherical_aberration_is_retained_for_execution():
    state = _objective(angle_mrad=1., ideal=False)
    plan = _prepared(state)[0]
    assert len(plan.posed_spherical_kicks) == 1
    assert plan.posed_spherical_kicks[0].strength_m3 > 0
    # The ordinary axial Cs must not also apply to this placed lens.
    assert not np.any(plan.cs_kick_m3)


def test_radial_accelerator_hands_posed_lens_back_to_two_dimensional_solver():
    from temsim.physics.radial_column_wave import _round_column_prefix
    state = _objective(angle_mrad=1.)
    prepared = _prepared(state)
    source = _source(state, prepared)
    z, modes, record = _round_column_prefix(state, source, float(prepared[0].z_mm[-1]),
                                            maximum_step_mm=state.step_mm)
    assert z == source.plane_z_mm
    assert modes == ()
    assert "two-dimensional column operator" in record["reason"]


def test_tilted_lens_phase_budget_rejects_before_mutating_source():
    state = _objective(angle_mrad=1.)
    prepared = _prepared(state)
    source = _source(state, prepared)
    original_digest = source.digest
    original = source.beam.modes[0].plane.amplitude.copy()
    with pytest.raises(ValueError, match="(?i)(posed|tilted).*phase|phase.*(budget|error)"):
        _propagate_column(state, source, float(prepared[0].z_mm[-1]), _prepared=prepared,
            grid_numerics=WaveGridNumerics(compute_backend="CPU",
                maximum_posed_lens_phase_error_rad_per_m=1e-30))
    assert source.digest == original_digest
    np.testing.assert_array_equal(source.beam.modes[0].plane.amplitude, original)
