"""Complete physical tip transport stays equivalent with compiled stepping."""
import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tracing import trace_feg_to_exit
from temsim.physics import analytic_particle_step


@pytest.mark.skipif(analytic_particle_step._compiled_step is None, reason="Numba optional")
def test_compiled_tip_to_exit_preserves_particles_stops_current_and_arrivals():
    gun = default_state().electron_gun
    assert gun.emitter.surface_model is None
    gun.compiled_particle_steps = False
    reference = trace_feg_to_exit(gun, 49)
    gun.compiled_particle_steps = True
    actual = trace_feg_to_exit(gun, 49)
    for key in ("alive", "ray_id", "weight", "energy_offset_ev"):
        np.testing.assert_array_equal(getattr(actual.exit_bundle, key), getattr(reference.exit_bundle, key))
    for key in ("x_m", "y_m", "tx_rad", "ty_rad"):
        np.testing.assert_allclose(getattr(actual.exit_bundle, key), getattr(reference.exit_bundle, key),
                                   rtol=1e-10, atol=1e-13)
    assert actual.blocked_key == reference.blocked_key
    np.testing.assert_array_equal(actual.blocked_z_mm, reference.blocked_z_mm)
    for key in ("emitted_current_a", "dpa_transmitted_current_a", "c1_transmitted_current_a"):
        assert getattr(actual, key) == getattr(reference, key)
    for actual_plane, reference_plane in zip(actual.plane_arrivals, reference.plane_arrivals, strict=True):
        np.testing.assert_array_equal(actual_plane.reached, reference_plane.reached)
        np.testing.assert_array_equal(actual_plane.transmitted, reference_plane.transmitted)
        np.testing.assert_allclose(actual_plane.time_s, reference_plane.time_s, rtol=1e-10, atol=1e-20)
        for key in ("x_m", "y_m"):
            np.testing.assert_allclose(getattr(actual_plane, key), getattr(reference_plane, key),
                                       rtol=1e-10, atol=1e-13)
