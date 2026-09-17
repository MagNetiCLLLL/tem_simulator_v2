"""Registry decisions never narrow the existing cache contract."""
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.parameter_registry import dependency_plan, runtime_definition
from temsim.runtime_parameters import runtime_targets


@pytest.mark.parametrize("change", ["lens", "source", "aperture", "display", "overlap", "vacuum", "unknown"])
def test_dependency_explanations_are_conservative_and_never_solve(change, monkeypatch):
    state = default_state()
    before = capture_instrument_snapshot(state)
    if change == "lens":
        state.lenses[0].percent += 1
    elif change == "source":
        state.electron_gun.accelerator.high_tension_kv -= 1
    elif change == "aperture":
        state.condenser_aperture_2.radius_mm *= .5
    elif change == "display":
        state.virtual_observation_z_mm = getattr(state, "virtual_observation_z_mm", 0.) + 1
    elif change == "overlap":
        lens = next(l for l in state.lenses if l.key == "diffraction_lens")
        lens.z_mm = state.sample.z_mm + .01
        lens.percent += 1
    elif change == "vacuum":
        state.vacuum_map.enabled = True
    else:
        state.new_physical_input_for_test = 9.
    after = capture_instrument_snapshot(state)
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.physics import core, simulation
    def forbidden(*args, **kwargs):
        pytest.fail("Dependency planning must not execute physics")
    monkeypatch.setattr(FieldEmissionGun, "trace_to_exit", forbidden)
    monkeypatch.setattr(core, "propagate", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    plan = dependency_plan(before, after)
    assert all(not row["reusable"] or row["legacy_reusable"] for row in plan["stages"].values())
    if change in {"unknown", "vacuum"}:
        assert plan["unknown_paths"]
        assert not any(row["reusable"] for row in plan["stages"].values())
    elif change != "display":
        assert not plan["stages"]["incident"]["reusable"]
    if change == "overlap":
        assert any(row["key"] == "diffraction_lens" for row in plan["incident_field_support"]["rows"])


def test_one_definition_drives_runtime_tooltip_and_sweep_boundary():
    from temsim.design_experiments import validate_runtime_sweep_path
    state = default_state()
    target = runtime_targets(state)["condenser_lens_1"]
    definition = runtime_definition(target, "percent")
    assert definition.category == "operating" and definition.unit == "%"
    assert definition.sweep_eligible
    assert "Stored but inactive" in definition.tooltip(enabled=False)
    assert validate_runtime_sweep_path(state.to_dict(), "lenses[condenser_lens_1].percent") == target.obj.percent
    with pytest.raises(ValueError, match="not an allowed"):
        validate_runtime_sweep_path(state.to_dict(), "lenses[condenser_lens_1].a_mm")


@pytest.mark.parametrize('owner', ['state', 'lens', 'sample', 'tip'])
def test_unknown_input_invalidates_actual_stage_cache_keys(owner):
    from temsim.calculation_cache import calculation_signatures
    from temsim.parameter_registry import unmapped_public_inputs
    state = default_state()
    before = calculation_signatures(state)
    assert not unmapped_public_inputs(state)
    target = {'state':state, 'lens':state.lenses[-1], 'sample':state.sample, 'tip':state.electron_gun.emitter}[owner]
    target.extension_physical_control = {'value':9.125}
    after = calculation_signatures(state)
    assert before.keys() == after.keys()
    assert all(before[key] != after[key] for key in before)
    target.extension_physical_control['value'] += .001
    edited = calculation_signatures(state)
    assert all(after[key] != edited[key] for key in after)


def test_lens_metadata_exceptions_do_not_hide_unrelated_controls():
    from temsim.parameter_registry import unmapped_public_inputs
    state = default_state()
    state.field_calibration_status = 1.25
    assert 'instrument.field_calibration_status' in unmapped_public_inputs(state)


@pytest.mark.parametrize('quality', ['Preview', 'High accuracy'])
def test_unknown_inputs_survive_background_and_synchronous_preparation(quality):
    from threading import Event
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.calculation_controller import CalculationController
    state = default_state()
    state.new_physical_control = [1., 2.]
    state.lenses[-1].extra_model_coefficient = .125
    request = CapturedCalculationRequest.capture(state, quality, 9, 1.)
    state.new_physical_control[0] = 99.
    try:
        prepared = request.prepare(Event())
        assert prepared.snapshot.new_physical_control == [1.,2.]
        assert prepared.snapshot.lenses[-1].extra_model_coefficient == .125
        direct = CalculationController._calculation_snapshot(state, quality, 9, 1.)
        assert direct.new_physical_control == [99.,2.]
    finally:
        request._input_assets.close()
