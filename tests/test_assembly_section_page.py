"""Small CPU-only checks of the independent saved-mesh section viewer."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from temsim.assembly_model_3d import AssemblyModel3D
from temsim.gui.assembly_section_page import AssemblySectionPage
from temsim.part_model_3d import revolve_section


def _model(key="coil", offset_y=0.0):
    mesh = revolve_section(((10, 2), (10, 4), (14, 4), (14, 2)),
                           key=key, angular_segments=16, material_class="copper")
    vertices = mesh.vertices.copy()
    vertices[:, 1] += offset_y
    return AssemblyModel3D((replace(mesh, vertices=vertices),))


def _assembly(key="coil"):
    return SimpleNamespace(parts=(SimpleNamespace(key=key, name=key, center_z_mm=12.0,
                                                 data={}),), vacuum_liner_segments=())


class _Cache:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.error = None

    def model_for(self, assembly, runtime_values=None):
        self.calls += 1
        if self.error:
            raise ValueError(self.error)
        return self.model


def _show(qtbot, cache=None):
    cache = cache or _Cache(_model())
    page = AssemblySectionPage(geometry_cache=cache)
    qtbot.addWidget(page)
    page.resize(1000, 550)
    page.set_assembly(_assembly())
    page.show()
    qtbot.waitUntil(lambda: page.section_builds == 1)
    return page, cache


def test_section_build_is_lazy_and_rotation_reuses_mesh_and_recent_angles(qtbot):
    cache = _Cache(_model())
    page = AssemblySectionPage(geometry_cache=cache)
    qtbot.addWidget(page)
    page.set_assembly(_assembly())
    page.set_projection_angle(45)
    assert cache.calls == page.section_builds == 0
    page.show()
    qtbot.waitUntil(lambda: page.section_builds == 1)
    assert cache.calls == 1
    assert page._section.angle_deg == 45

    page.set_projection_angle(90)
    qtbot.waitUntil(lambda: page.section_builds == 2)
    assert cache.calls == 1
    page.set_projection_angle(45)
    page._flush()
    assert page.section_builds == 2
    assert page._section.angle_deg == 45
    assert cache.calls == 1
    page.hide()
    page.set_projection_angle(180)
    page._flush()
    assert page.section_builds == 2


def test_follow_angle_has_no_signal_echo_and_can_be_unlocked(qtbot):
    page, _cache = _show(qtbot)
    received = []
    page.projection_angle_changed.connect(received.append)
    page.set_projection_angle(35)
    assert page.angle_spin.value() == 35
    assert page.angle_slider.value() == 350
    assert received == []
    page.angle_spin.setValue(47.5)
    assert received == [47.5]
    page.follow_ray_diagram.setChecked(False)
    page.set_projection_angle(90)
    assert page._angle_deg == 47.5
    page.angle_slider.setValue(1200)
    assert page._angle_deg == 120
    assert received == [47.5]
    page.follow_ray_diagram.setChecked(True)
    assert page._angle_deg == 90
    assert page.yz_button.isChecked()
    assert received == [47.5]


def test_angle_preserves_zoom_and_mesh_change_replaces_section(qtbot):
    page, cache = _show(qtbot)
    page.view.setRange(xRange=(11, 13), yRange=(-3, 3), padding=0)
    expected = np.array(page.view.viewRange())
    page.set_projection_angle(90)
    page._flush()
    np.testing.assert_allclose(page.view.viewRange(), expected)
    original_section = page._section
    cache.model = _model(offset_y=10)
    page.set_assembly(_assembly())
    page._flush()
    assert page._section is not original_section
    assert page.mesh_builds == 2
    assert all(mesh.segments_mm[:, :, 1].min() >= 6 for mesh in page._section.meshes)
    np.testing.assert_allclose(page.view.viewRange(), expected)


def test_new_geometry_error_clears_old_contours_and_recovers(qtbot):
    page, cache = _show(qtbot)
    assert page._curves
    cache.error = "bad geometry"
    page.set_assembly(_assembly())
    page._flush()
    assert page._model is None and page._section is None
    assert page._curves == []
    assert page.component_combo.count() == 0
    assert "bad geometry" in page.status.text()
    cache.error = None
    page.set_assembly(_assembly())
    page._flush()
    assert page._curves
    assert "bad geometry" not in page.status.text()


def test_contours_are_true_plane_intersections_and_nonmaterial_is_excluded(qtbot):
    material = _model(offset_y=10).meshes[0]
    reference = replace(_model("reference").meshes[0], wireframe=True)
    cache = _Cache(AssemblyModel3D((material, reference)))
    page, _cache = _show(qtbot, cache)
    assert page._curves == []  # X-Z plane misses this offset part entirely.
    page.set_projection_angle(90)
    page._flush()
    assert {mesh.key for mesh, _curve in page._curves} == {"coil"}
    assert all(curve.opts["connect"] == "pairs" for _mesh, curve in page._curves)


def test_selection_and_approximate_envelope_toggle_do_not_reslice(qtbot):
    exact = _model().meshes[0]
    approximate = replace(exact, key="envelope", is_exact=False)
    page, cache = _show(qtbot, _Cache(AssemblyModel3D((exact, approximate))))
    signals = []
    page.edit_part_requested.connect(lambda key, z: signals.append((key, z)))
    page.focus_component("coil")
    assert page.edit_part.isEnabled()
    page.edit_part.click()
    assert signals == [("coil", 12.0)]
    assert page.fit_current_selection()
    page.envelopes.setChecked(False)
    assert {mesh.key for mesh, _curve in page._curves} == {"coil"}
    assert page.section_builds == cache.calls == 1


def test_same_geometry_refresh_and_angle_cache_are_bounded(qtbot):
    page, cache = _show(qtbot)
    first = page._section
    page.set_assembly(_assembly(), {"coil": {"current": 3}})
    page._flush()
    assert page._section is first
    assert page.section_builds == 1
    for angle in range(10, 110, 10):
        page.set_projection_angle(angle)
        page._flush()
    assert len(page._sections) == 8
    assert cache.calls == 2
