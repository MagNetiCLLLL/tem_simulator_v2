import pytest

from temsim.hardware_tuning import (
    TASK_BY_KEY, TUNING_TASKS, resolve_task,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import editable_parameters


@pytest.fixture(scope="module")
def instrument():
    return default_state()


def test_registry_resolves_existing_live_controls_without_new_state(instrument):
    before = instrument.to_dict()
    assert len(TASK_BY_KEY) == len(TUNING_TASKS)
    for task in TUNING_TASKS:
        for group in resolve_task(instrument, task):
            if group.target is None:
                assert group.notice and not group.fields
                continue
            actual = {parameter.name for parameter in editable_parameters(group.target)}
            assert group.fields
            assert {field.name for field in group.fields} <= actual
            assert len(group.fields) == len(group.binding.fields)
    assert instrument.to_dict() == before


@pytest.mark.parametrize("first,second", [
    ("gun_shift", "gun_tilt"), ("condenser_shift", "condenser_tilt"),
    ("beam_shift", "beam_tilt"), ("image_shift", "image_tilt"),
    ("image_shift", "diffraction_shift"),
    ("condenser_current", "condenser_convergence"),
    ("magnification", "camera_length"),
])
def test_related_tasks_share_the_same_hardware_objects(instrument, first, second):
    a, b = resolve_task(instrument, first), resolve_task(instrument, second)
    assert len(a) == len(b)
    assert all(left.target.obj is right.target.obj for left, right in zip(a, b))
    assert all(left.fields == right.fields for left, right in zip(a, b))


def test_disabled_installed_component_can_be_reenabled(instrument, monkeypatch):
    state = instrument
    monkeypatch.setattr(state.beam_deflector, "enabled", False)
    group, = resolve_task(state, "beam_shift")
    assert group.target.obj is state.beam_deflector
    assert group.fields[0].name == "enabled"
    assert "Disabled" in group.notice


def test_uninstalled_corrector_and_third_condenser_are_not_editable(instrument, monkeypatch):
    state = instrument
    monkeypatch.setattr(state, "probe_corrector_installed", False)
    monkeypatch.setattr(state, "layout_c3_hardware", "two_condenser")
    assert all(group.target is None for group in resolve_task(state, "probe_corrector"))
    groups = resolve_task(state, "condenser_current")
    assert groups[0].target is not None and groups[1].target is not None
    assert groups[2].target is None and "not installed" in groups[2].notice


def test_ideal_mode_locks_only_hexapole_terms_without_switching_mode(instrument, monkeypatch):
    state = instrument
    monkeypatch.setattr(state, "simulation_mode", "ideal")
    groups = resolve_task(state, "probe_corrector")
    locked = [group for group in groups if group.target is None]
    assert len(locked) == 4
    assert all(group.binding.component_key.endswith("hexapole") for group in locked)
    assert all("Ideal Optics" in group.notice for group in locked)
    assert any(group.target is not None and group.binding.component_key.endswith("quadrupole") for group in groups)
    assert state.simulation_mode == "ideal"


def test_threefold_tasks_do_not_borrow_twofold_or_corrector_controls(instrument):
    for key in ("condenser_threefold", "objective_threefold"):
        assert resolve_task(instrument, key) == ()
        assert "Unsupported" in TASK_BY_KEY[key].description


def test_scan_static_bindings_exclude_clock_and_derived_calibration(instrument):
    for key in ("scan_static", "descan_static"):
        group, = resolve_task(instrument, key)
        assert group.target is not None
        names = {field.name for field in group.fields}
        assert {"kick_x_mrad", "kick_y_mrad", "upper_coil_gain"} <= names
        assert not names.intersection({"scan_enabled", "scan_frame_period_s", "scan_lines",
                                       "scan_pixels_x", "calibration_record_json",
                                       "lower_coil_gain", "scan_reference", "descan_target_key"})


def test_layout_owned_activation_is_read_only(instrument):
    c3 = resolve_task(instrument, "condenser_current")[-1]
    mini = resolve_task(instrument, "stem_focus")[0]
    assert [field.name for field in c3.fields] == ["percent"]
    assert [field.name for field in mini.fields] == ["percent"]
    assert "configuration" in c3.notice
    assert "layout" in mini.notice


def test_wobble_exposes_only_existing_drive_and_reports_active_raster(instrument, monkeypatch):
    group, = resolve_task(instrument, "beam_wobble")
    assert group.target.obj is instrument.ac_deflector
    assert {field.name for field in group.fields} == {
        "enabled", "wobble_enabled", "wobble_amplitude_x_mrad", "wobble_amplitude_y_mrad",
        "wobble_period_s", "wobble_phase_deg",
    }
    monkeypatch.setattr(instrument.ac_deflector, "scan_enabled", True)
    group, = resolve_task(instrument, "beam_wobble")
    assert "Raster scanning is active" in group.notice


def test_gun_selector_follows_active_source_components(instrument, monkeypatch):
    state = instrument
    from temsim.optics.electron_gun.thermionic import ThermionicGun
    monkeypatch.setattr(state, "electron_gun", ThermionicGun())
    group, = resolve_task(state, "gun_shift")
    assert group.target.obj is state.electron_gun.deflector
    assert group.target.key == "thermionic_deflector"
    group, = resolve_task(state, "gun_stigmation")
    assert group.target.obj is state.electron_gun.stigmator
    assert group.target.key == "thermionic_stigmator"
    assert all(group.target is not None for group in resolve_task(state, "gun_extraction"))


def test_registry_user_visible_text_uses_functional_names():
    strings = []
    for task in TUNING_TASKS:
        strings.extend((task.label, task.category, task.description))
        for binding in task.bindings:
            strings.append(binding.label)
            strings.extend(field.label for field in binding.fields)
    combined = " ".join(strings).lower()
    assert not any(word in combined for word in ("iliad", "ultra", "spectra", "ceta", "s-corr", "ceos"))
