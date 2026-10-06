"""Control coordinates stay selectable without becoming mechanical bodies."""

from copy import deepcopy
from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsRectItem

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_representation import (
    IMAGE_CONTROL_CHANNEL_KEYS, PROBE_CONTROL_CHANNEL_KEYS,
    SHARED_DEFLECTOR_HOSTS, shared_deflector_field_owner,
)
from temsim.diagnostics import physical_layout_records
from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.optics.column import default_state


CHANNEL_KEYS = (PROBE_CONTROL_CHANNEL_KEYS | IMAGE_CONTROL_CHANNEL_KEYS
                | SHARED_DEFLECTOR_HOSTS.keys() | {"probe_dp12_scan_deflector"})


def _present_channels(resolved):
    return CHANNEL_KEYS & {part.key for part in resolved.assembly.parts}


@pytest.fixture(scope="module", params=(
    "C3 + Probe Corrector", "C3 + Image Corrector",
    "C3 + Probe Corrector + Image Corrector",
))
def resolved(request):
    state = default_state()
    catalog = AssemblyCatalog()
    defaults = catalog.default_selection()
    assembly = catalog.apply(state, AssemblySelection(defaults.gun, request.param, defaults.recording))
    layout = apply_physical_layout_to_state(state)
    return SimpleNamespace(assembly=assembly, state_snapshot=state, layout=layout)


def test_display_records_distinguish_channels_without_mutating_transport(resolved):
    before = deepcopy(resolved.state_snapshot.to_dict())
    parts_before = [dict(part.data) for part in resolved.assembly.parts]
    records = {record.key: record for record in physical_layout_records(resolved)}
    for key in _present_channels(resolved):
        record = records[key]
        assert record.layout_role
        assert record.start_z_mm == record.center_z_mm == record.end_z_mm
        assert record.outer_diameter_mm == record.mechanical_bore_diameter_mm == 0
        assert record.optical_references_mm
        assert "no independent material body" in record.representation_note.lower()
        assert record.center_z_mm == pytest.approx(resolved.assembly.part(key).center_z_mm)
        if key in IMAGE_CONTROL_CHANNEL_KEYS:
            assert "in image corrector" in record.representation_note
            assert "unverified" in record.representation_note
            assert "physical_host_key" not in resolved.assembly.part(key).data
    for key, host in SHARED_DEFLECTOR_HOSTS.items():
        assert records[key].center_z_mm == pytest.approx(records[host].center_z_mm)
        assert "same physical coils" in records[key].representation_note
        assert "unverified" not in records[key].representation_note
    for key in ("probe_hp1_hexapole", "probe_hp2_hexapole", "probe_tl21_lens", "probe_tl22_lens",
                "image_hp1_hexapole", "image_hp2_hexapole", "image_ol_post_lens",
                "image_tl11_lens", "image_tl12_lens", "image_tl21_lens", "image_tl22_lens",
                "image_adapter_lens", "ac_deflector", "beam_deflector",
                *SHARED_DEFLECTOR_HOSTS.values()):
        if key not in records:
            continue
        assert not records[key].layout_role
        assert records[key].end_z_mm > records[key].start_z_mm
        assert records[key].outer_diameter_mm > 0
    assert records["ac_deflector"].center_z_mm > records["beam_deflector"].center_z_mm
    assert resolved.state_snapshot.to_dict() == before
    assert [dict(part.data) for part in resolved.assembly.parts] == parts_before


def test_2d_channels_have_only_markers_and_selection_never_adds_a_solid(qtbot, resolved):
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.resize(1350, 750)
    page.display_result(resolved)
    channels = _present_channels(resolved)
    assert set(page._channel_marker_items) == channels
    # All independently drawn rectangles remain owned by real components.
    physical_rectangles = set()
    for item in page.plot.scene().items():
        key = page._selectable_item_keys.get(id(item))
        if key and isinstance(item, QGraphicsRectItem):
            assert key not in CHANNEL_KEYS
            physical_rectangles.add(key)
    expected_hexapoles = {key for key in ("probe_hp1_hexapole", "probe_hp2_hexapole",
                                        "image_hp1_hexapole", "image_hp2_hexapole")
                         if key in page._component_label_items}
    assert expected_hexapoles <= physical_rectangles
    for key in channels:
        line = page._channel_marker_items[key]
        assert line.opts["pen"].style() == Qt.PenStyle.DashLine
        assert "channel" in page._component_label_items[key].toPlainText() or "virtual" in page._component_label_items[key].toPlainText()
        page.focus_component(resolved.assembly.part(key))
        assert isinstance(page._highlight, pg.InfiniteLine)
        assert "no separate body" in page.summary.text()
        assert "OD " not in page.summary.text()
    if "probe_hp1_hexapole" in page._component_label_items:
        assert page._component_label_items["probe_hp1_hexapole"].toPlainText() == "HP1 Hexapole"
    page.display_result(resolved)
    assert set(page._channel_marker_items) == channels


def test_legacy_nonzero_channel_envelopes_do_not_reappear_without_new_metadata(resolved):
    parts = []
    for part in resolved.assembly.parts:
        data = dict(part.data)
        for field in ("layout_role", "layout_owner", "physical_host_key", "physical_host_status"):
            data.pop(field, None)
        parts.append(SimpleNamespace(**{
            name: getattr(part, name) for name in ("key", "name", "branch", "start_z_mm", "center_z_mm", "end_z_mm")
        }, data=data))
    legacy = SimpleNamespace(assembly=SimpleNamespace(parts=parts), layout=resolved.layout)
    records = {record.key: record for record in physical_layout_records(legacy)}
    channels = _present_channels(resolved)
    assert all(records[key].layout_role for key in channels)
    assert all(records[key].outer_diameter_mm == 0 for key in channels)
    assert all("in image corrector" in records[key].representation_note
               for key in channels & IMAGE_CONTROL_CHANNEL_KEYS)


@pytest.mark.parametrize("key,host", SHARED_DEFLECTOR_HOSTS.items())
def test_shared_channel_structure_is_read_only_in_both_editors(qtbot, key, host):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.module_manifest import read_document
    from temsim.part_model_3d import part_dimension_specs
    from temsim.part_model_document import PartModelDocument

    editor = ManifestEditor()
    target = ManifestTarget("column/C3_ProbeCorrector.toml", key)
    fields = editor.fields(target)
    owned = [field for field in fields if shared_deflector_field_owner(key, field.path[2])]
    assert owned and all(not field.editable for field in owned)
    assert next(field for field in fields if field.label == "name").editable
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context(key, None, target, fields, None)
    assert host in panel.manifest_draft_notice.text()
    for row, field in enumerate(fields):
        if shared_deflector_field_owner(key, field.path[2]):
            value = panel.manifest_table.item(row, 1)
            assert not value.flags() & Qt.ItemFlag.ItemIsEditable
            assert host in value.toolTip()
    path = editor.root / target.module_path
    specs = part_dimension_specs(read_document(path), key)
    owned_specs = [item for item in specs if shared_deflector_field_owner(key, item.path[2])]
    assert owned_specs and all(not item.editable for item in owned_specs)
    session = PartModelDocument(path)
    before = deepcopy(session.document)
    with pytest.raises(ValueError, match=host):
        session.set_dimension(("parts", key, "length_mm"), 100)
    with pytest.raises(ValueError, match=host):
        session.place_component(key, 100)
    with pytest.raises(ValueError, match=host):
        session.copy_component_from(session.document, key, key + "_copy", 100)
    assert session.document == before
    with pytest.raises(ValueError, match=host):
        editor.save(target, {("parts", key, "local_center_z_mm"): 100}, None)
