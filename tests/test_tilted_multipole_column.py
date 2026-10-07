"""Actual captured main-column stigmator: particle and CPU/GPU wave checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e

from temsim.optics.column import default_state
from temsim.physics.column_wave import _prepare_column, _propagate_column
from temsim.physics.core import execute_propagation_plan
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.posed_lens_wave import vector_potential_jet
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_grid import WaveGridNumerics
from test_tilted_column_wave import _centroid
from test_wave_detector_readout import checkpoint


def stigmator_fixture():
    source = default_state()
    component = next(item for item in source.stigmators if item.key == "objective_stigmator")
    component.strength_x_percent = 40.
    component.strength_y_percent = -25.
    assembly = replace(source._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": 1., "offset_x_mm": .05})
        if part.key == component.key else part for part in source._resolved_assembly.parts))
    state = SimpleNamespace(lenses=[], condenser_system=source.condenser_system,
        _resolved_assembly=assembly, stigmators=[component], corrector_elements=[],
        deflectors=[], apertures=[], recording_planes=[], simulation_mode="ideal",
        beam_voltage_kv=300., step_mm=.01, history_step_mm=.01,
        acceleration_enabled=False, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=source.sample)
    prepared = _prepare_column(state, component.z_mm-.2, component.z_mm+.2, .01)
    plan = prepared[0]
    assert len(plan.mapped_fields) == 1 and isinstance(plan.mapped_fields[0], FrozenPosedMultipole)
    assert plan.mapped_fields[0].component_key == component.key
    assert not np.any(plan.midpoint_sx_m2) and not np.any(plan.midpoint_sy_m2)
    assert not np.any(plan.midpoint_sxy_m2)
    axis = (np.arange(64)-32)*5e-9
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(25e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    origin = np.array((20e-9, -15e-9))
    p = float(_momentum_velocity(300000.)[0])
    a = vector_potential_jet(plan.mapped_fields, (*origin, plan.z_mm[0]*1e-3))[0]
    plane = PlaneWave(amplitude, np.eye(2)*5e-9, origin, np.zeros((2, 2)), -e*a[:2]/p)
    initial = checkpoint()
    initial = replace(initial, plane_z_mm=float(plan.z_mm[0]),
        beam=replace(initial.beam, modes=(replace(initial.beam.modes[0], plane=plane),)))
    return state, prepared, initial


def test_actual_tilted_stigmator_gpu_and_cpu_wave_match_particle_centroid():
    from test_wave_device import _cuda
    _cuda()
    state, prepared, initial = stigmator_fixture()
    plan = prepared[0]
    origin = initial.beam.modes[0].plane.origin_m
    ray = execute_propagation_plan(state, plan, np.array((origin[0],)), np.zeros(1),
                                  np.array((origin[1],)), np.zeros(1))
    expected = np.array((ray[1][-1, 0], ray[3][-1, 0]))
    assert np.isfinite(expected).all()
    assert np.linalg.norm(expected-origin) > 1e-10  # Visible response, not a zero-strength fixture.
    outputs = []
    for backend in ("CPU", "Require GPU"):
        output = _propagate_column(state, initial, float(plan.z_mm[-1]), _prepared=prepared,
            grid_numerics=WaveGridNumerics(compute_backend=backend,
                maximum_posed_lens_phase_error_rad_per_m=1e-7))
        assert output.record["modes"][0]["compute_backend"] == ("cupy" if backend == "Require GPU" else "numpy")
        assert output.beam.total_weight == pytest.approx(initial.beam.total_weight, rel=1e-10)
        np.testing.assert_allclose(_centroid(output.beam.modes[0].plane), expected,
                                   rtol=0., atol=2e-12)
        outputs.append(output.beam.modes[0].plane)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        # Identical physical input and operator: absolute complex amplitude,
        # without adjusting global phase or applying an intensity fit.
        np.testing.assert_allclose(getattr(outputs[0], name), getattr(outputs[1], name),
                                   rtol=2e-9, atol=2e-12)
