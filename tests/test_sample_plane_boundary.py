import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.core import _endpoint_exact_axial_grid, propagate
from temsim.physics.simulation import run


@pytest.mark.parametrize(
    ("stop_z_mm", "step_mm"),
    (
        (454.2, 2.5),  # Previously overshot the requested plane by 0.8 mm.
        (451.2, 1.0),  # Previously stopped 0.2 mm before the plane.
    ),
)
def test_propagate_hits_non_divisible_endpoint_exactly(stop_z_mm, step_mm):
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = step_mm
    state.history_step_mm = step_mm
    start_z_mm = 450.0
    zeros = np.zeros(3, dtype=float)

    integration_grid, integration_steps_mm = _endpoint_exact_axial_grid(
        start_z_mm, stop_z_mm, step_mm
    )

    z_mm, *_ = propagate(
        state,
        start_z_mm,
        stop_z_mm,
        zeros,
        np.array([-1.0e-3, 0.0, 1.0e-3]),
        zeros,
        zeros,
    )

    assert z_mm[0] == start_z_mm
    assert z_mm[-1] == stop_z_mm
    intervals = np.diff(integration_grid)
    assert np.all(intervals > 0.0)
    assert np.max(intervals) <= step_mm
    assert integration_steps_mm == pytest.approx(
        intervals, rel=0.0, abs=0.0
    )
    if intervals.size > 1:
        assert intervals[:-1] == pytest.approx(
            np.full(intervals.size - 1, step_mm), rel=0.0, abs=1.0e-12
        )


def test_numba_and_cpu_match_on_endpoint_exact_grid():
    if not compute_backend.numba_cpu_capability().available:
        pytest.skip("Numba CPU backend unavailable")

    initial = (
        np.array([-20.0e-6, 0.0, 20.0e-6]),
        np.array([-1.0e-3, 0.0, 1.0e-3]),
        np.array([10.0e-6, 0.0, -10.0e-6]),
        np.array([0.5e-3, 0.0, -0.5e-3]),
    )

    def trace(backend):
        state = default_state()
        state.acceleration_enabled = backend != "CPU"
        state.acceleration_backend = backend
        state.step_mm = 2.5
        state.history_step_mm = 2.5
        return propagate(state, 450.0, 454.2, *initial)

    cpu = trace("CPU")
    numba = trace("Numba CPU")

    assert cpu[0][-1] == 454.2
    assert numba[0][-1] == 454.2
    for cpu_values, numba_values in zip(cpu, numba):
        assert numba_values == pytest.approx(
            cpu_values, rel=2.0e-6, abs=1.0e-9
        )


def test_preview_incident_and_outgoing_bundles_meet_only_at_sample_plane():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    state.step_mm = 2.5
    state.history_step_mm = 2.5
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.diffraction_enabled = False
    state.sample.wave_enabled = False
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = 9
    else:
        state.electron_gun.ray_count = 9

    simulation = run(state, resolved_layout=layout)
    incident = simulation.incident
    outgoing = simulation.branches["000"]
    sample_z_mm = float(state.sample.z_mm)

    assert incident.z[-1] == sample_z_mm
    assert outgoing.z[0] == sample_z_mm
    assert np.all(incident.z <= sample_z_mm)
    assert np.all(outgoing.z >= sample_z_mm)
    assert np.array_equal(incident.x[-1], outgoing.x[0])
    assert np.array_equal(incident.y[-1], outgoing.y[0])


def test_retracted_sample_keeps_probe_plane_but_removes_scattering_branches():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.inserted = False
    state.sample.diffraction_enabled = True
    state.sample.diffuse_broadening_mrad = 25.0
    state.electron_gun.emitter.ray_count = 9

    simulation = run(state, resolved_layout=layout)

    assert tuple(simulation.branches) == ("000",)
    assert simulation.metrics["sample_inserted"] is False
    assert simulation.metrics["sample_scattering_applied"] is False
    assert simulation.incident.z[-1] == pytest.approx(state.sample.z_mm)
    assert simulation.branches["000"].z[0] == pytest.approx(
        state.sample.z_mm
    )


def test_real_sample_never_creates_user_invented_diffraction_branches():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.inserted = True
    state.sample.specimen_mode = "atomic"
    state.sample.diffraction_enabled = True
    # These persisted legacy values must be dormant in Real sample mode.
    state.sample.g_inv_nm = 1.0e5
    state.sample.excitation_error_inv_nm = 3.0
    state.sample.rocking_width_inv_nm = 0.01
    state.sample.diffuse_broadening_mrad = 100.0
    state.electron_gun.emitter.ray_count = 9

    simulation = run(state, resolved_layout=layout)

    assert "000" in simulation.branches
    assert not any(
        "+g" in name or "-g" in name or "diffuse" in name
        for name in simulation.branches
    )
    assert {
        branch.interaction_kind
        for branch in simulation.branches.values()
    } == {
        "real_zero_loss",
        "real_plasmon",
        "real_ionisation",
        "real_plural_inelastic",
    }
    assert simulation.branches["000"].interaction_kind == "real_zero_loss"
    assert sum(
        branch.weight for branch in simulation.branches.values()
    ) + simulation.metrics["sample_absorbed_probability"] == pytest.approx(1.0)
    assert simulation.metrics["sample_scattering_applied"] is True
    assert simulation.metrics["sample_scattering_model"] == (
        "real_material_inelastic_poisson_plus_elastic_wave"
    )
    assert simulation.metrics["ray_interaction_types"] == (
        "real_zero_loss",
        "real_plasmon",
        "real_ionisation",
        "real_plural_inelastic",
    )
    plasmon = simulation.branches["real_plasmon"]
    zero_loss = simulation.branches["000"]
    assert plasmon.energy_offset_ev - zero_loss.energy_offset_ev == pytest.approx(
        np.full(plasmon.energy_offset_ev.shape, -16.7)
    )
    assert np.hypot(
        plasmon.interaction_kick_x_rad,
        plasmon.interaction_kick_y_rad,
    ) * 1.0e3 == pytest.approx(
        np.full(
            plasmon.energy_offset_ev.shape,
            simulation.real_interactions.channels[1].characteristic_angle_mrad,
        )
    )


def test_sample_inserted_state_round_trips_without_owning_sample_geometry():
    state = default_state()
    state.sample.inserted = False

    payload = state.to_dict()
    restored = type(state).from_dict(payload)

    assert payload["sample"]["inserted"] is False
    assert restored.sample.inserted is False
