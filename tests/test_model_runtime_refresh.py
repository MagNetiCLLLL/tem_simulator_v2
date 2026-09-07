"""GUI dependency decisions only; no field solves or high-accuracy tracing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector_ImageCorrector.toml"
APERTURE = "condenser_aperture_2"
LENS = "condenser_lens_1"


@pytest.fixture
def editor(qtbot, monkeypatch):
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.session = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    page._project_root = INSTRUMENT_CONFIG_ROOT.resolve()
    page._project_paths = (MODULE,)
    page._selected_key = APERTURE
    page._runtime_values = {APERTURE: {"radius_mm": .05}, LENS: {"percent": 50.}}
    calls = []
    monkeypatch.setattr(page, "_load_parameters", lambda: calls.append("dimensions"))
    monkeypatch.setattr(page, "_load_all_parameters", lambda: calls.append("annotations"))
    monkeypatch.setattr(page, "_render", lambda **_kwargs: calls.append("meshes"))
    page.show()
    calls.clear()
    return page, calls


@pytest.mark.parametrize("visible", [True, False])
def test_unrelated_lens_changes_do_not_rebuild_selected_aperture(editor, visible):
    page, calls = editor
    page.setVisible(visible)
    values = deepcopy(page._runtime_values)
    values[LENS]["percent"] = 60.
    page.set_runtime_values(values)
    page.show()
    assert calls == []


def test_module_scope_updates_unselected_aperture(editor):
    page, calls = editor
    page._selected_key = LENS
    page.scope.setCurrentIndex(page.scope.findData("module"))
    calls.clear()
    values = deepcopy(page._runtime_values)
    values[APERTURE]["radius_mm"] = .1
    page.set_runtime_values(values)
    assert calls == ["dimensions", "meshes"]
    calls.clear()
    page.set_runtime_values(deepcopy(values))
    assert calls == []


def test_hidden_updates_coalesce_and_use_latest_values(editor):
    page, calls = editor
    page.hide()
    values = deepcopy(page._runtime_values)
    for radius in (.06, .07, .08):
        values[APERTURE]["radius_mm"] = radius
        page.set_runtime_values(values)
    assert calls == []
    assert page._runtime_refresh_pending
    assert page._runtime_values[APERTURE]["radius_mm"] == .08
    values[APERTURE]["radius_mm"] = 99.  # Caller mutation cannot alter the stored snapshot.
    page.show()
    assert calls == ["dimensions", "meshes"]
    assert page._runtime_values[APERTURE]["radius_mm"] == .08
    page.hide()
    page.show()
    assert calls == ["dimensions", "meshes"]


@pytest.mark.parametrize("scope, expected", [("part", False), ("context", True), ("module", True)])
def test_only_displayed_dependencies_rebuild(editor, scope, expected):
    page, calls = editor
    page._selected_key = LENS
    # The renderer includes siblings only in context, descendants in all scopes.
    page.session.part(LENS)["parent_key"] = "test_parent"
    page.session.part(APERTURE)["parent_key"] = "test_parent"
    page.scope.setCurrentIndex(page.scope.findData(scope))
    calls.clear()
    values = deepcopy(page._runtime_values)
    values[APERTURE]["offset_x_mm"] = .01
    page.set_runtime_values(values)
    assert ("meshes" in calls) == expected


def test_selected_lens_annotations_update_without_mesh_rebuild(editor):
    page, calls = editor
    page._selected_key = LENS
    values = deepcopy(page._runtime_values)
    values[LENS]["percent"] = 60.
    page.set_runtime_values(values)
    assert calls == ["annotations"]


def test_explicit_base_and_explicit_holes_do_not_use_live_opening(editor):
    page, calls = editor
    part = page.session.part(APERTURE)
    for override in ({"aperture_hole_diameters_mm": [.05, .1]},
                     {"model_3d": {"base": {"kind": "box"}}}):
        part.update(override)
        calls.clear()
        values = deepcopy(page._runtime_values)
        values[APERTURE]["radius_mm"] += .01
        page.set_runtime_values(values)
        assert "meshes" not in calls


def test_inactive_module_runtime_changes_and_activation(editor):
    page, calls = editor
    page.hide()
    page.set_project_context(INSTRUMENT_CONFIG_ROOT,
        SimpleNamespace(selected_module_paths=()), None, page._runtime_values)
    assert calls == []
    assert page._runtime_refresh_pending
    page.show()
    assert "meshes" in calls  # Remove the old instrument's working opening.
    calls.clear()
    values = {APERTURE: {"radius_mm": .2}}
    page.set_runtime_values(values)
    assert calls == []


def test_descendant_aperture_refreshes_selected_parent(editor):
    page, calls = editor
    page._selected_key = LENS
    page.session.part(APERTURE)["parent_key"] = LENS
    values = deepcopy(page._runtime_values)
    values[APERTURE]["radius_mm"] = .1
    page.set_runtime_values(values)
    assert calls == ["dimensions", "meshes"]
