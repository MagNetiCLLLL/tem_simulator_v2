"""Existing physical component scalars are editable without changing time or geometry."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.interactive_calculation import (
    CalculationRange, available_controls, apply_live_tuning_values, _assign,
)


@pytest.fixture
def component_state():
    from temsim.optics.condenser_stigmator import create_condenser_stigmator
    from temsim.optics.beam_deflector import create_beam_deflector
    from temsim.optics.probe_corrector import (
        create_dph2_deflector, create_hp2_hexapole, create_qph2_quadrupole,
    )
    from temsim.optics.ac_deflector import create_ac_deflector
    from temsim.optics.descan_deflector import create_descan_deflector
    return SimpleNamespace(
        lenses=[], apertures=[], recording_planes=[], electron_gun=SimpleNamespace(),
        sample=SimpleNamespace(z_mm=1500.), stigmators=[create_condenser_stigmator()],
        deflectors=[create_beam_deflector()],
        corrector_elements=[create_dph2_deflector(), create_hp2_hexapole(),
                            create_qph2_quadrupole(), create_ac_deflector(), create_descan_deflector()],
        simulation_time_s=.375, _resolved_assembly=object(),
    )


def axis(state, component, field, *, limits=(-1e10, 1e10)):
    control = next(c for c in available_controls(state) if c.key == component.key and c.field == field)
    return CalculationRange(control, *limits)


def test_existing_components_expose_static_scalars_without_scan_or_geometry(component_state):
    controls = available_controls(component_state)
    by_group = {group: {c.field for c in controls if c.group == group}
                for group in ("stigmator", "deflector", "corrector")}
    assert by_group["stigmator"] == {"strength_x_percent", "strength_y_percent"}
    assert by_group["deflector"] == {"upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad",
                                      "kick_x_mrad", "kick_y_mrad"}
    assert by_group["corrector"] == {"strength_m2", "strength_m3", "orientation_rad"}
    assert all(c.stage == "optical" for c in controls)
    assert not any(c.field.startswith(("scan_", "wobble_")) or c.field in {"z_mm", "enabled"} for c in controls)
    assert len({c.identity for c in controls}) == len(controls)
    # A physical element may be referenced by more than one installed list.
    component_state.deflectors.append(component_state.corrector_elements[0])
    controls = available_controls(component_state)
    assert len({c.identity for c in controls}) == len(controls)


def test_scalar_edits_change_existing_field_and_kick_models(component_state):
    state = component_state
    stig = state.stigmators[0]
    stig.field_model = "normal_skew"
    single, hexa, quad = state.corrector_elements[:3]
    geometry = state._resolved_assembly
    positions = (stig.z_mm, single.z_mm, hexa.z_mm, quad.z_mm)
    edits = ((axis(state, stig, "strength_x_percent"), 10.),
             (axis(state, stig, "strength_y_percent"), 20.),
             (axis(state, single, "kick_x_mrad"), .25),
             (axis(state, hexa, "strength_m3"), 2e5),
             (axis(state, hexa, "orientation_rad"), np.pi / 6),
             (axis(state, quad, "strength_m2"), 25.))
    apply_live_tuning_values(state, edits)
    normal, _negative, skew = stig.quadrupole_tensor_m2(stig.z_mm)
    assert normal != 0. and skew != 0.
    assert single.kick_events()[0][1] == pytest.approx(.25e-3)
    assert hexa.strength_m3 == 2e5 and hexa.orientation_rad == np.pi / 6
    assert quad.strength_m2 == 25.
    assert state._resolved_assembly is geometry
    assert (stig.z_mm, single.z_mm, hexa.z_mm, quad.z_mm) == positions


@pytest.mark.parametrize("index,field,limit_field", [
    (0, "kick_x_mrad", "maximum_kick_mrad"),
    (1, "strength_m3", "maximum_strength_m3"),
    (2, "strength_m2", "maximum_strength_m2"),
])
def test_component_limits_reject_complete_batch_before_any_live_change(component_state, index, field, limit_field):
    state = component_state
    first = state.stigmators[0]
    target = state.corrector_elements[index]
    before = getattr(target, field)
    invalid = 2. * getattr(target, limit_field)
    requested = axis(state, target, field)
    with pytest.raises(ValueError):
        apply_live_tuning_values(state, ((axis(state, first, "strength_x_percent"), 30.),
                                        (requested, invalid)))
    assert first.strength_x_percent == 0.
    assert getattr(target, field) == before
    with pytest.raises(ValueError):
        _assign(state, requested.control, invalid)
    assert getattr(target, field) == before


def test_multi_channel_component_is_validated_as_one_candidate(component_state):
    @dataclass
    class CoupledDeflector:
        key: str = "coupled"
        name: str = "Coupled deflector"
        kick_x_mrad: float = 0.
        kick_y_mrad: float = 0.
        enabled: bool = True

        def validate(self):
            if np.hypot(self.kick_x_mrad, self.kick_y_mrad) > 1.:
                raise ValueError("Combined coil limit")

    target = CoupledDeflector()
    component_state.deflectors.append(target)
    with pytest.raises(ValueError, match="Combined coil"):
        apply_live_tuning_values(component_state, (
            (axis(component_state, target, "kick_x_mrad"), .8),
            (axis(component_state, target, "kick_y_mrad"), .8),
        ))
    assert target.kick_x_mrad == target.kick_y_mrad == 0.


def test_static_scan_coil_offsets_preserve_time_dependent_controls(component_state):
    state = component_state
    ac, descan = state.corrector_elements[-2:]
    ac.wobble_enabled = False  # Existing model requires mutually exclusive scan/wobble.
    ac.scan_enabled = descan.scan_enabled = True
    saved = [(item.scan_enabled, item.scan_frame_period_s, item.scan_amplitude_x_mrad,
              item.scan_amplitude_y_mrad, item.scan_pixels_x, item.scan_lines)
             for item in (ac, descan)]
    apply_live_tuning_values(state, ((axis(state, ac, "kick_x_mrad"), .15),
                                    (axis(state, descan, "kick_y_mrad"), -.15)))
    assert state.simulation_time_s == .375
    assert not ac.wobble_enabled
    assert saved == [(item.scan_enabled, item.scan_frame_period_s, item.scan_amplitude_x_mrad,
                      item.scan_amplitude_y_mrad, item.scan_pixels_x, item.scan_lines)
                     for item in (ac, descan)]


@pytest.mark.parametrize("change", ["disabled", "uninstalled", "field", "nonfinite"])
def test_new_live_controls_keep_scope_and_finiteness_checks(component_state, change):
    target = component_state.corrector_elements[0]
    requested = axis(component_state, target, "kick_x_mrad")
    value = .25
    if change == "disabled":
        target.enabled = False
    elif change == "uninstalled":
        target.installed = False
    elif change == "field":
        requested = replace(requested, control=replace(requested.control, field="maximum_kick_mrad"))
    else:
        value = float("nan")
    with pytest.raises(ValueError):
        apply_live_tuning_values(component_state, ((requested, value),))
    assert target.kick_x_mrad == 0.
