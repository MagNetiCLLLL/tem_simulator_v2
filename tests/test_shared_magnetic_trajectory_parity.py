"""Same-B near-axis comparisons; not whole-column E/trajectory qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import prepare_magnetic_scene
from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
from temsim.optics.model import DeflectorPair, Stigmator
from temsim.physics.core import propagate


def _state():
    return SimpleNamespace(
        lenses=(), stigmators=(Stigmator("Stigmator", "s", 29.,
            strength_x_percent=20., strength_y_percent=-13.),),
        corrector_elements=(),
        deflectors=(DeflectorPair("Deflector", "d", 25., 38.,
            upper_x_mrad=1., upper_y_mrad=-.4, lower_x_mrad=-.3,
            lower_y_mrad=.2, thickness_mm=4.),),
        beam_voltage_kv=300., simulation_mode="custom", simulation_time_s=0.,
        step_mm=.1, history_step_mm=.1, acceleration_enabled=False,
        acceleration_backend="CPU", sample=SimpleNamespace(z_mm=0.),
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
    )


@pytest.mark.parametrize("energy_ev", (150_000., 300_000.))
def test_shared_magnetic_field_approaches_same_near_axis_paths_at_declared_planes(energy_ev):
    state = _state()
    scene = prepare_magnetic_scene(state, z_limits_mm=(0., 60.))
    initial = (2e-6, -1e-6, 0.)
    settings = TestElectronSettings(kinetic_energy_ev=energy_ev, position_m=initial,
        max_path_length_m=.055, step_m=.0002, relative_tolerance=1e-7,
        position_tolerance_m=1e-13)
    coarse = trace_test_electron(scene, settings, use_compiled=False)
    fine = trace_test_electron(scene, replace(settings, step_m=.0001), use_compiled=False)
    assert coarse.completed and fine.completed
    assert np.all(np.diff(fine.positions_m[:, 2]) > 0.)
    # Include the coil interior before/after its centre: matching only a final
    # signed integral would miss the former thin-kick/finite-coil divergence.
    planes_mm = (24., 26., 35., 50.)
    events = tuple(event for coil in state.deflectors for event in (
        (coil.upper_z_mm, coil.upper_x_mrad*1e-3, coil.upper_y_mrad*1e-3),
        (coil.lower_z_mm, coil.lower_x_mrad*1e-3, coil.lower_y_mrad*1e-3)))
    result = propagate(state, 0., 50., np.array([initial[0]]), np.zeros(1),
        np.array([initial[1]]), np.zeros(1), events=events,
        energy_offset_ev=np.array([energy_ev-300_000.]),
        include_spherical_aberration=False, include_hexapole=False,
        save_z_mm=planes_mm, checkpoint_z_mm=planes_mm, return_checkpoints=True)
    checkpoints = result[-1]
    for index, z_mm in enumerate(checkpoints.z_mm):
        z = z_mm*1e-3
        sampled = np.array([np.interp(z, fine.positions_m[:, 2], fine.positions_m[:, axis])
                            for axis in (0, 1)])
        previous = np.array([np.interp(z, coarse.positions_m[:, 2], coarse.positions_m[:, axis])
                             for axis in (0, 1)])
        main = np.array([checkpoints.x_m[index, 0], checkpoints.y_m[index, 0]])
        # Declared transverse tolerance at <=2 mrad: covers small-angle
        # approximation and interpolation, not arbitrary large-angle equality.
        np.testing.assert_allclose(main, sampled, atol=2e-9, rtol=2e-5)
        np.testing.assert_allclose(previous, sampled, atol=5e-10, rtol=1e-5)
    np.testing.assert_allclose(fine.kinetic_energy_ev, energy_ev, rtol=1e-12)
