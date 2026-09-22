"""The active launch geometry is visible without changing solids or transport."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_curvature import MODEL
from temsim.part_model_3d import part_model_from_document, part_dimension_specs
from temsim.part_model_document import PartModelDocument
from temsim.paths import CONFIG_ROOT
from temsim.tip_emission_view import emission_display_meshes, emission_dimensions, emission_view_values


def context(k=0., model=MODEL):
    doc = PartModelDocument(CONFIG_ROOT / "sources/FEG_tip.toml").document
    gun = FieldEmissionGun()
    gun.emitter.surface_model = gun.emitter.coherence = None
    gun.emitter.curvature_nm_inv = k
    gun.emitter.curvature_model = model
    runtime = {"tip_surface_model": None, "tip_analytic_emission": emission_view_values(gun.emitter)}
    return doc, gun, runtime


@pytest.mark.parametrize("k", [0., 1e-12, .01, .02])
def test_visible_surface_matches_launch_coordinates_and_preserves_reference_solid(k):
    doc, gun, runtime = context(k)
    part = doc["parts"][0]
    before = gun.to_dict()
    bundle = gun.emit(193)
    surface, normals = emission_display_meshes(part, runtime)
    dims = emission_dimensions(part, runtime)
    v_nm = surface.vertices*1e6
    r2 = np.sum(v_nm[:, :2]**2, axis=1)
    # Sphere centred one radius upstream, including its stable flat limit.
    np.testing.assert_allclose(v_nm[:, 2], -k*(r2+v_nm[:, 2]**2)/2, atol=1e-15)
    assert v_nm[:, 2].max() == 0.
    assert -v_nm[:, 2].min() == pytest.approx(dims["emission_depth_nm"])
    assert dims["emission_radius_nm"] == (1/k if k else float("inf"))
    assert dims["emission_support_diameter_nm"] == pytest.approx(2*np.sqrt(r2.max()))
    if k:
        xyz = bundle.surface_position_m*1e9
        assert xyz[:, 2].min() >= v_nm[:, 2].min()
    assert np.all(np.linalg.norm(np.cross(
        surface.vertices[surface.faces[:, 1]]-surface.vertices[surface.faces[:, 0]],
        surface.vertices[surface.faces[:, 2]]-surface.vertices[surface.faces[:, 0]]), axis=1) > 0)
    assert normals.wireframe
    assert gun.to_dict() == before
    saved = part_model_from_document(doc, "feg_tip", runtime_values={"feg_tip": {"tip_surface_model": None}})
    active = part_model_from_document(doc, "feg_tip", runtime_values={"feg_tip": runtime})
    assert len(active.meshes) == len(saved.meshes) == 1  # no open surface in exported/copied solids
    np.testing.assert_array_equal(saved.meshes[0].vertices, active.meshes[0].vertices)
    assert part["tip_radius_nm"] == 100.


def test_current_emission_rejects_old_angle_only_model_and_keeps_nonemitting_copies():
    with pytest.raises(ValueError, match="Unsupported"):
        context(.02, "axisymmetric_cap_v1")
    doc, _, runtime = context(.02)
    part = doc["parts"][0]
    surface, normals = emission_display_meshes(part, runtime)
    assert surface.vertices[:, 2].min() < 0
    assert emission_dimensions(part, runtime)["emission_depth_nm"] > 0
    assert normals.edges[1]["vertices"][1, 0] > normals.edges[1]["vertices"][0, 0]
    assert emission_display_meshes({**part, "mechanical_only": True}, runtime) == ()
    assert emission_display_meshes(part, {}) == ()


def test_zero_spatial_support_has_no_invented_surface():
    doc, gun, runtime = context()
    gun.emitter.virtual_source_fwhm_nm = 0.
    runtime["tip_analytic_emission"] = emission_view_values(gun.emitter)
    assert emission_display_meshes(doc["parts"][0], runtime) == ()
    assert emission_dimensions(doc["parts"][0], runtime)["emission_support_diameter_nm"] == 0.


@pytest.fixture
def active_page(qtbot):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.gui.main_window import MainWindow
    from temsim.gui.part_model_editor import PartModelEditorPage
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.surface_model = None
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.resize(1600, 1000)
    def refresh(k, fwhm=5., flush=True):
        state.electron_gun.emitter.curvature_nm_inv = k
        state.electron_gun.emitter.virtual_source_fwhm_nm = fwhm
        values = {"feg_tip": {"curvature_nm_inv": k, "virtual_source_fwhm_nm": fwhm}}
        MainWindow._add_tip_render_values(SimpleNamespace(state=state), values)
        page.set_runtime_values(values)
        if flush:
            page._flush_runtime_refresh()
        return values
    values = refresh(0.)
    page.set_project_context(CONFIG_ROOT/"instruments", assembly, None, values)
    page.set_simulation_context("ideal", {}, {p.key: p.data for p in assembly.parts})
    assert page.open_path(CONFIG_ROOT/"sources/FEG_tip.toml", selected_key="feg_tip")
    page.show()
    qtbot.wait(30)
    return page, state, refresh


def test_live_curvature_rebuilds_mesh_preserves_camera_and_highlights_parameters(active_page, qtbot):
    page, state, refresh = active_page
    assert not page.tip_display.isHidden()
    assert not page.session.dirty
    flat = next(m for m in page._mesh_records if m["region"] == "emitting_cap")
    assert np.all(flat["vertices"][:, 2] == 0)
    page.view.set_isometric_view()
    camera = (page.view._rotation.copy(), page.view._center.copy(), page.view._radius)
    document = deepcopy(page.session.document)
    refresh(.01)
    curved = next(m for m in page._mesh_records if m["region"] == "emitting_cap")
    assert not np.array_equal(flat["vertices"], curved["vertices"])
    np.testing.assert_array_equal(page.view._rotation, camera[0])
    np.testing.assert_array_equal(page.view._center, camera[1])
    assert page.view._radius == camera[2]
    assert page.session.document == document and not page.session.dirty
    rows = {tuple(page.dimensions.item(r, 1).data(Qt.ItemDataRole.UserRole)): r
            for r in range(page.dimensions.rowCount())}
    ref = rows[("parts", "feg_tip", "tip_radius_nm")]
    assert page.dimensions.item(ref, 1).text() == "100.0"
    assert page.dimensions.item(ref, 3).text() == "Reference"
    radius = rows[("derived", "feg_tip", "emission_radius_nm")]
    assert page.dimensions.item(radius, 1).text() == "100.0"
    assert page.dimensions.item(radius, 3).text() == "Derived"
    picked = None
    for face in curved["faces"][::37]:
        point = page.view.project_points([curved["vertices"][face].mean(axis=0)])[0]
        picked = page.view.pick_topology_at(QPointF(*point[:2]), mode="face")
        if picked and picked["id"] == "emitting_cap":
            break
    assert picked and picked["id"] == "emitting_cap"
    page.view.set_topology_selection([picked], emit=True)
    for name in ("curvature_nm_inv", "virtual_source_fwhm_nm"):
        r = rows[("runtime", "feg_tip", name)]
        assert page.dimensions.item(r, 1).data(Qt.ItemDataRole.UserRole+1)
    refresh(.02)
    assert page.dimensions.item(radius, 1).text() == "50.0"
    refresh(0.)
    assert page.dimensions.item(radius, 1).text() == "∞"
    assert not state.electron_gun.emitter.curvature_nm_inv


def test_reference_display_draft_and_hidden_refresh_do_not_replace_emission(active_page, qtbot):
    page, state, refresh = active_page
    page.tip_display.setCurrentIndex(page.tip_display.findData("reference"))
    assert all(m["region"] == "body" for m in page._mesh_records)
    page._fit_tip_emission()
    assert page.tip_display.currentData() == "emission"
    page.session.set_dimension(("parts", "feg_tip", "tip_radius_nm"), 150.)
    page._load_parameters()
    page._render(preserve_view=True)
    page.hide()
    refresh(.02, 8., flush=False)
    assert page._runtime_refresh_pending
    page.show()
    qtbot.wait(30)
    assert not page._runtime_refresh_pending
    surface = next(m for m in page._mesh_records if m["region"] == "emitting_cap")
    assert surface["vertices"][:, 2].min() < 0
    assert page.session.dirty and page.session.part("feg_tip")["tip_radius_nm"] == 150.
    assert state.electron_gun.emitter.curvature_nm_inv == .02
    assert all(m.region != "emitting_cap" for m in part_model_from_document(
        page.session.document, "feg_tip", runtime_values=page._model_runtime_values()).meshes)
