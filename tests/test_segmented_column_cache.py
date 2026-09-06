import numpy as np

import temsim.physics.simulation as simulation_module
from temsim.optics.column import default_state
from temsim.physics.core import fields, propagate
from temsim.physics.simulation import (
    INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES,
    INCIDENT_CHECKPOINT_SPACING_MM,
    _column_checkpoint_planes,
    run,
)


def _small_vacuum_state():
    state = default_state()
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 10.0
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = 9
    else:
        state.electron_gun.ray_count = 9
    return state


def _assert_incident_equal(actual, expected):
    np.testing.assert_array_equal(actual.z, expected.z)
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.tx, expected.tx)
    np.testing.assert_array_equal(actual.ty, expected.ty)
    np.testing.assert_array_equal(actual.alive, expected.alive)
    np.testing.assert_allclose(
        actual.blocked_z, expected.blocked_z, atol=1e-9, equal_nan=True
    )
    assert actual.blocked_key == expected.blocked_key


def test_changed_c2_reuses_prefix_and_matches_a_fresh_full_trace():
    state = _small_vacuum_state()
    state.step_mm = 0.7
    state.history_step_mm = 5.0
    previous = run(state)
    c2 = next(lens for lens in state.lenses if lens.key == "condenser_lens_2")
    c2.percent += 0.25
    cold_state = type(state).from_dict(state.to_dict())

    incremental = run(state, existing_simulation=previous)
    cold = run(cold_state)

    _assert_incident_equal(incremental.incident, cold.incident)
    cache = incremental.metrics["column_segment_cache"]
    assert cache["mode"] == "checkpoint"
    assert cache["hit"] is True
    assert cache["reused_prefix_rows"] > 0
    assert cache["recomputed_history_rows"] < len(incremental.incident.z)
    assert cache["reused_integration_nodes"] > 0
    assert cache["recomputed_integration_nodes"] < len(
        incremental.incident_plan.z_mm
    )
    assert cache["resume_z_mm"] < state.condenser_system[
        c2.key
    ].field_support_mm()[0]


def test_segment_resume_never_mutates_the_previous_result():
    state = _small_vacuum_state()
    previous = run(state)
    old_incident = {
        name: np.asarray(getattr(previous.incident, name)).copy()
        for name in ("z", "x", "tx", "y", "ty", "alive", "blocked_z")
    }
    old_checkpoints = {
        name: np.asarray(getattr(previous.incident_checkpoints, name)).copy()
        for name in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad")
    }
    c2 = next(lens for lens in state.lenses if lens.key == "condenser_lens_2")
    c2.percent += 0.25

    run(state, existing_simulation=previous)

    for name, values in old_incident.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(previous.incident, name)), values
        )
    for name, values in old_checkpoints.items():
        cached = np.asarray(getattr(previous.incident_checkpoints, name))
        np.testing.assert_array_equal(cached, values)
        assert cached.flags.writeable is False


def test_gui_cache_passes_raster_checkpoint_to_the_validating_solver():
    from temsim.calculation_cache import calculation_signatures, matching_products
    from temsim.gui.calculation_controller import CalculationController
    from temsim.simulation_pipeline import CalculationResult
    from temsim.simulation_modes import switch_mode

    state = _small_vacuum_state()
    switch_mode(state, "ideal")
    state.ac_deflector.wobble_enabled = False
    state = CalculationController._calculation_snapshot(state, "High accuracy", 9, 5.0)
    previous = run(state)
    previous_result = CalculationResult(
        simulation=previous, energy_filter=None,
        state_snapshot=state, signatures=calculation_signatures(state),
        calculated_products=frozenset({"column"}),
    )
    controller = CalculationController(persistent_cache_enabled=False)
    controller._cache_result(previous_result)
    changed = type(state).from_dict(state.to_dict())
    changed.ac_deflector.scan_enabled = True
    changed.ac_deflector.scan_pixels_x = changed.ac_deflector.scan_lines = 2
    assert not matching_products(previous_result.signatures, calculation_signatures(changed))
    workers = []
    controller.pool.start = workers.append
    controller.submit(changed, "High accuracy", 9, 5.0)
    assert workers[0].existing_result is previous_result
    incremental = run(workers[0].state, existing_simulation=workers[0].existing_result.simulation)
    cold = run(type(workers[0].state).from_dict(workers[0].state.to_dict()))
    _assert_incident_equal(incremental.incident, cold.incident)
    assert incremental.metrics["column_segment_cache"]["mode"] == "checkpoint"
    assert incremental.metrics["column_segment_cache"]["reused_integration_nodes"] > 0


def test_unchanged_optical_plan_reuses_the_complete_incident_trace():
    state = _small_vacuum_state()
    previous = run(state)
    state.sample.eds_detector_efficiency = 0.625

    updated = run(state, existing_simulation=previous)

    assert updated.metrics["column_segment_cache"]["mode"] == "full_incident"
    _assert_incident_equal(updated.incident, previous.incident)
    for name in ("z", "x", "tx", "y", "ty"):
        assert not np.shares_memory(
            np.asarray(getattr(updated.incident, name)),
            np.asarray(getattr(previous.incident, name)),
        )


def test_changed_aperture_reuses_coordinates_but_recomputes_all_stops():
    state = _small_vacuum_state()
    previous = run(state)
    aperture = next(
        item for item in state.apertures
        if item.key == "condenser_aperture_2"
    )
    aperture.radius_mm *= 0.8
    cold_state = type(state).from_dict(state.to_dict())

    incremental = run(state, existing_simulation=previous)
    cold = run(cold_state)

    assert incremental.metrics["column_segment_cache"]["mode"] == (
        "full_incident"
    )
    _assert_incident_equal(incremental.incident, cold.incident)


def test_solver_fields_are_exactly_zero_outside_declared_support():
    state = _small_vacuum_state()
    target = next(
        lens for lens in state.lenses if lens.key == "condenser_lens_2"
    )
    for lens in state.lenses:
        lens.enabled = lens is target
    for stigmator in state.stigmators:
        stigmator.enabled = False
    for component in state.corrector_elements:
        component.enabled = False
    provider = state.condenser_system[target.key]
    lower, upper = provider.field_support_mm()
    z_mm = np.asarray((lower - 1e-6, target.z_mm, upper + 1e-6))

    solver_field = fields(z_mm, state)[0]

    assert solver_field[0] == 0.0
    assert solver_field[2] == 0.0
    assert solver_field[1] == provider.magnetic_field_t(z_mm)[1]


def test_checkpoint_capture_does_not_change_a_non_divisible_rk4_grid():
    state = _small_vacuum_state()
    state.step_mm = 0.7
    state.history_step_mm = 2.1
    initial_x = np.asarray((-2.0e-6, 0.0, 2.0e-6))
    initial_y = initial_x[::-1].copy()
    initial_tx = np.asarray((-1.0e-3, 0.0, 1.0e-3))
    initial_ty = initial_tx[::-1].copy()

    baseline = propagate(
        state, 450.0, 800.0,
        initial_x, initial_tx, initial_y, initial_ty,
    )
    captured = propagate(
        state, 450.0, 800.0,
        initial_x, initial_tx, initial_y, initial_ty,
        checkpoint_z_mm=(475.0, 500.0, 525.0, 550.0, 575.0),
        return_checkpoints=True,
    )

    for actual, expected in zip(captured[:5], baseline):
        np.testing.assert_array_equal(actual, expected)
    assert captured[5].z_mm.size == 5


def test_checkpoint_density_respects_the_memory_budget():
    ray_count = 100_000
    planes = _column_checkpoint_planes(450.0, 1_600.0, ray_count)
    used_bytes = len(planes) * ray_count * 4 * np.dtype(np.float64).itemsize

    assert used_bytes <= INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES
    assert len(planes) > 0


def test_default_checkpoint_cache_retains_more_column_history():
    ray_count = 15_000
    planes = _column_checkpoint_planes(450.0, 1_600.0, ray_count)
    used_bytes = len(planes) * ray_count * 4 * np.dtype(np.float64).itemsize

    assert INCIDENT_CHECKPOINT_SPACING_MM == 5.0
    assert INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES == 512 * 1024 * 1024
    assert len(planes) == 229
    assert used_bytes <= INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES


def test_checkpoint_cache_disables_capture_if_one_plane_exceeds_budget(
    monkeypatch,
):
    monkeypatch.setattr(
        simulation_module,
        "INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES",
        31,
    )

    assert _column_checkpoint_planes(450.0, 1_600.0, 1) == ()
