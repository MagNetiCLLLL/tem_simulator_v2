"""Permanent insertion, shared controls, TOML placement and hard-stop regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_keys import (
    ENERGY_FILTER_ENTRANCE_APERTURE, FIXED_APERTURE_KEYS,
    GUN_EXTRACTOR_APERTURE, PROJECTION_CHAMBER_DPA_APERTURE as DPA,
    THERMIONIC_ANODE_APERTURE,
)
from temsim.gui.instrument_tree import InstrumentTree
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.physics.aperture_clipping import clip_segment
from temsim.runtime_parameters import editable_parameters, runtime_targets, validate_runtime_assignment
from temsim.simulation_pipeline import aperture_stop_records


def _state(gun="FEG", recording="Energy Filter", blanker="None"):
    state = default_state()
    assembly = AssemblyCatalog().apply(state, AssemblySelection(
        gun=gun, column="C3 + Probe Corrector", recording=recording,
        beam_blanker=blanker,
    ))
    return state, assembly


@pytest.mark.parametrize("gun,key", [("FEG", GUN_EXTRACTOR_APERTURE),
                                     ("Thermionic", THERMIONIC_ANODE_APERTURE)])
def test_fixed_stops_are_aperture_leaves_with_non_retractable_controls(qtbot, gun, key):
    state, assembly = _state(gun)
    targets = runtime_targets(state)
    tree = InstrumentTree()
    panel = ParameterPanel()
    qtbot.addWidget(tree)
    qtbot.addWidget(panel)
    tree.load_optical(assembly, targets, category="aperture")
    for aperture_key in (key, DPA, ENERGY_FILTER_ENTRANCE_APERTURE):
        assert tree.select_key(aperture_key)
        assert "always inserted" in tree.currentItem().text(0)
        target = targets[aperture_key]
        assert "enabled" not in {p.name for p in editable_parameters(target)}
        with pytest.raises(ValueError, match="always inserted"):
            validate_runtime_assignment(target, "enabled", False)
        target.obj.enabled = False  # Legacy model/snapshot API cannot retract it.
        assert target.obj.enabled is True
        panel.set_context(target.label, target, None, (), None)
        assert set(panel._quick_widgets) == {"diameter_mm", "offset_x_mm", "offset_y_mm"}
        panel._quick_widgets["diameter_mm"].setValue(80.0)
        assert target.obj.radius_mm == pytest.approx(0.04)
    assert tree.select_key("objective_aperture")
    target = targets["objective_aperture"]
    panel.set_context(target.label, target, None, (), None)
    assert "enabled" in panel._quick_widgets
    assert not target.obj.enabled


def test_dpa_edits_survive_layout_and_snapshot_and_clip_once():
    from temsim.calculation_cache import calculation_signatures
    state, _ = _state()
    stop = runtime_targets(state)[DPA].obj
    stop.diameter_mm = 0.08
    stop.offset_x_mm = 0.1
    stop.offset_y_mm = -0.02
    for _ in range(2):
        layout = apply_physical_layout_to_state(state)
    assert sum(a.key == DPA for a in state.apertures) == 1
    assert next(c for c in layout if c.key == DPA).effective_aperture_radius_mm == pytest.approx(0.04)
    snapshot = state.to_dict()
    signatures = calculation_signatures(state)
    saved_stop = next(a for a in snapshot["apertures"] if a["key"] == DPA)
    saved_stop["enabled"] = False
    restored = type(state).from_dict(snapshot)
    apply_physical_layout_to_state(restored, assembly=state._resolved_assembly)
    assert calculation_signatures(restored) == signatures
    stop = runtime_targets(restored)[DPA].obj
    assert stop.enabled
    assert stop.radius_mm == pytest.approx(0.04)
    assert stop.offset_y_mm == pytest.approx(-0.02)
    z = np.array([stop.z_mm - 0.1, stop.z_mm, stop.z_mm + 0.1])
    x = np.tile([0.1, 0.139, 0.141], (3, 1)) * 1e-3  # ray arrays in metres
    y = np.full_like(x, -0.02e-3)
    alive, blocked_z, blocked_key = clip_segment(restored, z, x, y)
    np.testing.assert_array_equal(alive, [True, True, False])
    assert blocked_z[2] == stop.z_mm
    assert blocked_key == ["", "", DPA]
    assert len([r for r in aperture_stop_records(restored) if r["key"] == DPA]) == 1


def test_dpa_toml_position_and_opening_are_not_reset_by_layout_refresh():
    state, assembly = _state()
    part = assembly.part(DPA)
    data = dict(part.data)
    for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
                  "optical_reference_local_z_mm"):
        data[field] += 2.0
    data["mechanical_bore_diameter_mm"] = 0.3
    changed = replace(part, start_z_mm=part.start_z_mm + 2,
                      center_z_mm=part.center_z_mm + 2,
                      end_z_mm=part.end_z_mm + 2, data=data)
    assembly = replace(assembly, parts=tuple(changed if p.key == DPA else p for p in assembly.parts))
    for _ in range(2):
        apply_physical_layout_to_state(state, assembly=assembly)
    stop = runtime_targets(state)[DPA].obj
    assert stop.z_mm == pytest.approx(changed.center_z_mm)
    assert stop.radius_mm == pytest.approx(0.15)


def test_absent_energy_filter_does_not_clip_or_appear_in_aperture_tree(qtbot):
    # The current GUI catalog normalizes to Energy Filter. Exercise the
    # supported low-level absent-branch topology without changing that UI policy.
    from temsim.column.layout import LayoutConfiguration
    from temsim.column.module_assembly import resolve_module_assembly
    state = default_state()
    state.energy_filter.enabled = False
    state.energy_filter_mode = "no_energy_filter"
    state.energy_filter_installed = False
    assembly = resolve_module_assembly(LayoutConfiguration(energy_filter_selected=False))
    apply_physical_layout_to_state(state, assembly=assembly)
    stop = state.energy_filter_entrance_aperture
    assert not stop.installed
    assert stop.enabled  # Insertion policy is not the installation gate.
    tree = InstrumentTree()
    qtbot.addWidget(tree)
    tree.load_optical(assembly, runtime_targets(state), category="aperture")
    assert not tree.select_key(ENERGY_FILTER_ENTRANCE_APERTURE)
    z = np.array([stop.z_mm - 1, stop.z_mm + 1])
    isolated = SimpleNamespace(apertures=[stop])
    alive, _, _ = clip_segment(isolated, z, np.ones((2, 1)), np.ones((2, 1)))
    assert alive.all()
    from temsim.physics.multiplane_wave import intermediate_apertures
    assert ENERGY_FILTER_ENTRANCE_APERTURE not in {
        a.key for a in intermediate_apertures(state, stop.z_mm + 1.0)
    }


def test_nanopulser_fixed_stop_is_a_toml_editable_aperture(qtbot):
    state, assembly = _state(blanker="NanoPulser")
    tree = InstrumentTree()
    qtbot.addWidget(tree)
    tree.load_optical(assembly, runtime_targets(state), category="aperture")
    assert tree.select_key("nanopulser_aperture")
    assert state.nanopulser.aperture.enabled
    assert "nanopulser_aperture" in FIXED_APERTURE_KEYS
    state.nanopulser.installed = False
    stop = state.nanopulser.aperture
    isolated = SimpleNamespace(apertures=[stop])
    alive, _, _ = clip_segment(
        isolated, np.array([stop.z_mm - 1, stop.z_mm + 1]),
        np.ones((2, 1)), np.ones((2, 1)),
    )
    assert alive.all()


def test_closed_dpa_drawing_matches_runtime_and_profile_roundtrip(tmp_path):
    from temsim.diagnostics import physical_layout_records
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    state, assembly = _state()
    stop = runtime_targets(state)[DPA].obj
    stop.radius_mm = 0.0
    stop.offset_y_mm = 0.01
    layout = apply_physical_layout_to_state(state)
    record = next(r for r in physical_layout_records(
        SimpleNamespace(layout=layout, assembly=assembly)
    ) if r.key == DPA)
    assert record.bore_diameter_mm == 0.0
    path = tmp_path / "fixed_apertures.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    assert "enabled" not in values[DPA]
    restored, _ = _state()
    apply_profile_values(restored, values)
    apply_physical_layout_to_state(restored)
    assert runtime_targets(restored)[DPA].obj.radius_mm == 0.0
    assert runtime_targets(restored)[DPA].obj.offset_y_mm == pytest.approx(0.01)


def test_dpa_is_shared_by_geometric_and_coherent_detector_transport():
    from temsim.physics.multiplane_wave import intermediate_apertures
    from temsim.physics.record_plane import runtime_recording_stops
    from temsim.physics.recording_clipping import clip_recording_planes
    state, _ = _state()
    stop = runtime_targets(state)[DPA].obj
    assert sum(a.key == DPA for a in intermediate_apertures(state, state.camera.z_mm)) == 1
    assert sum(a.key == DPA for a in runtime_recording_stops(state)) == 1
    z = np.array([stop.z_mm - 1.0, stop.z_mm + 1.0])
    # Coarse history still interpolates to the actual stop before detectors.
    x = np.tile([0.0, 1.1 * stop.radius_mm * 1e-3], (2, 1))
    alive, blocked_z, blocked_key = clip_recording_planes(
        state, z, x, np.zeros_like(x), np.ones(2, dtype=bool),
        np.full(2, np.nan), ["", ""],
    )
    np.testing.assert_array_equal(alive, [True, False])
    assert blocked_key == ["", DPA]
    assert blocked_z[1] == stop.z_mm
