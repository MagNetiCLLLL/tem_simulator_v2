import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics.simulation import run


def test_cold_feg_energy_samples_are_positive_and_match_requested_spread():
    state = default_state()
    emitter = state.electron_gun.emitter

    for ray_count in (49, 1_000, 15_000):
        emitted = emitter.emit(ray_count)
        kinetic_energy = (
            emitter.emission_energy_ev + emitted.energy_offset_ev
        )

        assert np.all(kinetic_energy >= emitter.minimum_kinetic_energy_ev)
        assert np.mean(kinetic_energy) == pytest.approx(
            emitter.emission_energy_ev, abs=1.0e-10
        )
        assert 2.354820045 * np.std(kinetic_energy) == pytest.approx(
            emitter.energy_spread_fwhm_ev, abs=1.0e-10
        )
        assert np.max(np.abs(emitted.energy_offset_ev)) <= (
            emitter.energy_half_range_ev + 1.0e-12
        )


def test_preview_cold_feg_trace_has_no_clamped_energy_outlier():
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection(
            "FEG", "C3 + Probe Corrector", "Energy Filter"
        ),
    )
    state.electron_gun.emitter.ray_count = 49
    trace = state.electron_gun.trace_to_exit()

    exit_radius_um = np.hypot(
        trace.exit_bundle.x_m, trace.exit_bundle.y_m
    ) * 1.0e6
    assert np.all(trace.exit_bundle.alive)
    assert np.max(exit_radius_um) < 10.0


def test_gun_paths_use_one_strict_z_grid_and_preserve_equal_time_history():
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection(
            "FEG", "C3 + Probe Corrector", "Energy Filter"
        ),
    )
    state.electron_gun.emitter.ray_count = 25
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.sample.diffraction_enabled = False

    simulation = run(state)
    trace = simulation.gun_trace
    exit_z = float(state.electron_gun.exit_plane_z_mm)

    assert np.all(np.diff(trace.z_mm) > 0.0)
    assert np.count_nonzero(np.isclose(trace.z_mm, exit_z)) == 1
    assert np.all(np.diff(simulation.incident.z) > 0.0)
    assert trace.x_m[-1] == pytest.approx(trace.exit_bundle.x_m)
    assert trace.y_m[-1] == pytest.approx(trace.exit_bundle.y_m)

    history = trace.equal_time_history
    assert history is not None
    assert np.all(np.diff(history.time_s) > 0.0)
    assert history.z_mm.shape == history.x_m.shape
    assert history.z_mm.shape[1] == 25

    exit_plane = next(
        plane
        for plane in trace.plane_arrivals
        if np.isclose(plane.z_mm, exit_z)
    )
    assert np.all(exit_plane.reached)
    assert np.all(exit_plane.transmitted)
    assert np.all(np.isfinite(exit_plane.time_s))
    assert np.ptp(exit_plane.time_s) > 0.0

    front = trace.equal_time_front_at_plane(exit_plane.key)
    assert front.z_mm.shape == (25,)
    assert front.x_m.shape == (25,)
    # Simultaneously emitted electrons do not reach the exit simultaneously;
    # the equal-time front therefore has a real axial curvature/spread.
    assert np.ptp(front.z_mm) > 0.0
