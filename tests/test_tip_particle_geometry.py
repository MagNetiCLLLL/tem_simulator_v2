"""Curved classical launch and editor regressions; no coherent wave execution."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_patch import patch_dimensions, sample_cap_frame
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence, emit_surface
from temsim.optics.electron_gun.field_emission import FieldEmissionGun


def test_patch_geometry_has_one_physical_radius_and_angle():
    geometry = load_tip_surface_reference().geometry
    d = patch_dimensions(geometry, 30.)
    assert d["projected_diameter_nm"] == pytest.approx(geometry.apex_radius_nm)
    assert d["half_angle_rad"] == pytest.approx(math.pi / 6)
    assert d["apex_to_edge_arc_nm"] == pytest.approx(geometry.apex_radius_nm * math.pi / 6)
    assert d["surface_area_nm2"] == pytest.approx(2 * math.pi * geometry.apex_radius_nm * d["cap_depth_nm"])
    assert d["apex_curvature_nm_inv"] == pytest.approx(1 / geometry.apex_radius_nm)


@pytest.mark.parametrize("angle", [1e-7, 10., 60.])
def test_uniform_cap_and_local_frame_remain_resolved_for_narrow_patches(angle):
    geometry = load_tip_surface_reference().geometry
    u = np.linspace(0, 1, 501)
    p, normal, tangent1, tangent2 = sample_cap_frame(geometry, angle, u, np.full_like(u, .17))
    d = patch_dimensions(geometry, angle)
    assert p[-1, 2] < 0
    assert np.hypot(*p[-1, :2]) * 1e9 == pytest.approx(d["projected_diameter_nm"] / 2)
    np.testing.assert_allclose(-p[:, 2] * 1e9 / d["cap_depth_nm"], u, atol=1e-14)
    assert d["surface_area_nm2"] > 0
    frame = np.stack((tangent1, tangent2, normal), axis=2)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", frame, frame),
                               np.broadcast_to(np.eye(3), frame.shape), atol=1e-14)
    np.testing.assert_allclose(np.cross(tangent1, tangent2), normal, atol=1e-14)


@pytest.mark.parametrize("angle", [0., -1., 85., float("nan"), float("inf"), True])
def test_invalid_patch_is_not_silently_clamped(angle):
    with pytest.raises(ValueError):
        patch_dimensions(load_tip_surface_reference().geometry, angle)


def test_particle_directions_follow_edited_cap_normals_without_wave_phase():
    model = load_tip_surface_reference()
    model = replace(model, emission=replace(model.emission, tangential_mean_energy_ev=0.))
    p, directions, energy, weights = emit_surface(model, 193)
    radius = model.geometry.apex_radius_nm * 1e-9
    normal = (p + [0., 0., radius]) / radius
    np.testing.assert_allclose(directions, normal, atol=1e-14)
    enlarged = replace(model, geometry=replace(model.geometry, apex_radius_nm=200.))
    other = emit_surface(enlarged, 193)
    np.testing.assert_array_equal(other[0], p * 2)
    np.testing.assert_array_equal(other[1], directions)
    np.testing.assert_array_equal(other[2], energy)
    np.testing.assert_array_equal(other[3], weights)
    wider = replace(model, emission=replace(model.emission, cap_half_angle_deg=20.))
    assert np.mean(emit_surface(wider, 193)[1][:, 2]) < np.mean(directions[:, 2])


def test_editor_explicit_particle_selection_and_geometry_are_draft_only(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    previous = gun.emitter.surface_model
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    assert dialog.surface_coherent.isChecked()  # no silent profile conversion
    dialog.particle_button.click()
    assert not dialog.surface_coherent.isChecked()
    assert not dialog.wave_options.isChecked()
    assert not dialog.near_field_button.isEnabled()
    for key, value in {"apex_radius_nm": "150", "cone_half_angle_deg": "8", "shank_length_um": "800"}.items():
        dialog.geometry_inputs[key].setText(value)
    dialog.surface_inputs["cap_half_angle_deg"].setText("15")
    assert "0.2618 rad" in dialog.patch_summary.text()
    assert dialog.geometry_preview._model.geometry.apex_radius_nm == 150
    dialog.accept()
    edited = dialog.value()["surface_model"]
    assert edited.coherence is None
    assert edited.geometry.apex_radius_nm == 150
    assert edited.geometry.cone_half_angle_deg == 8
    assert edited.geometry.shank_length_um == 800
    assert edited.emission.cap_half_angle_deg == 15
    assert gun.emitter.surface_model is previous


def test_invalid_geometry_keeps_live_source_and_does_not_apply(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.particle_button.click()
    dialog.geometry_inputs["apex_radius_nm"].setText("-1")
    dialog.accept()
    assert dialog._value is None
    assert "Apex radius" in dialog.error.text()
    assert gun.emitter.surface_model is None
    assert dialog.geometry_preview._model is None


def test_edited_geometry_round_trips_and_invalidates_only_consumed_field_inputs(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.grounded_tip_field import field_request
    state = default_state()
    gun = state.electron_gun
    reference = load_tip_surface_reference()
    gun.emitter.surface_model = reference
    first_key = gun._cache_key(9)
    first_field = field_request(gun)
    gun.emitter.surface_model = replace(reference, geometry=replace(reference.geometry, apex_radius_nm=125.))
    assert gun._cache_key(9) != first_key
    assert field_request(gun) != first_field
    before_angle = field_request(gun)
    gun.emitter.surface_model = replace(gun.emitter.surface_model,
        emission=replace(reference.emission, cap_half_angle_deg=12.))
    assert field_request(gun) == before_angle  # emission patch does not move metal
    path = tmp_path / "particle-tip.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    other = default_state()
    _, values = read_profile(path)
    assert apply_profile_values(other, values) == []
    assert other.electron_gun.emitter.surface_model == gun.emitter.surface_model
    captured = capture_instrument_snapshot(other)
    other.electron_gun.emitter.surface_model = reference
    assert capture_instrument_snapshot(other).digest != captured.digest


def test_geometry_preview_renders_without_propagation(qtbot, tmp_path):
    from temsim.gui.tip_geometry_preview import TipGeometryPreview
    widget = TipGeometryPreview()
    qtbot.addWidget(widget)
    widget.resize(460, 180)
    widget.set_model(load_tip_surface_reference())
    widget.show()
    assert widget.grab().save(str(tmp_path / "tip-geometry.png"))
