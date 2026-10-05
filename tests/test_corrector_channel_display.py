"""Control coordinates stay selectable without becoming mechanical bodies."""

from copy import deepcopy
from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsRectItem

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_representation import PROBE_CONTROL_CHANNEL_KEYS
from temsim.diagnostics import physical_layout_records
from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.optics.column import default_state


CHANNEL_KEYS = PROBE_CONTROL_CHANNEL_KEYS | {"probe_dp12_scan_deflector"}


@pytest.fixture(scope="module")
def resolved():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    return SimpleNamespace(assembly=assembly, state_snapshot=state, layout=layout)


def test_display_records_distinguish_channels_without_mutating_transport(resolved):
    before = deepcopy(resolved.state_snapshot.to_dict())
    parts_before = [dict(part.data) for part in resolved.assembly.parts]
    records = {record.key: record for record in physical_layout_records(resolved)}
    for key in CHANNEL_KEYS:
        record = records[key]
        assert record.layout_role
        assert record.start_z_mm == record.center_z_mm == record.end_z_mm
        assert record.outer_diameter_mm == record.mechanical_bore_diameter_mm == 0
        assert record.optical_references_mm
        assert "no independent material body" in record.representation_note.lower()
        assert record.center_z_mm == pytest.approx(resolved.assembly.part(key).center_z_mm)
    for key in ("probe_hp1_hexapole", "probe_hp2_hexapole", "probe_tl21_lens", "probe_tl22_lens"):
        assert not records[key].layout_role
        assert records[key].end_z_mm > records[key].start_z_mm
        assert records[key].outer_diameter_mm > 0
    assert resolved.state_snapshot.to_dict() == before
    assert [dict(part.data) for part in resolved.assembly.parts] == parts_before


def test_2d_channels_have_only_markers_and_selection_never_adds_a_solid(qtbot, resolved):
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.resize(1350, 750)
    page.display_result(resolved)
    assert set(page._channel_marker_items) == CHANNEL_KEYS
    # All independently drawn rectangles remain owned by real components.
    physical_rectangles = set()
    for item in page.plot.scene().items():
        key = page._selectable_item_keys.get(id(item))
        if key and isinstance(item, QGraphicsRectItem):
            assert key not in CHANNEL_KEYS
            physical_rectangles.add(key)
    assert {"probe_hp1_hexapole", "probe_hp2_hexapole"} <= physical_rectangles
    for key in CHANNEL_KEYS:
        line = page._channel_marker_items[key]
        assert line.opts["pen"].style() == Qt.PenStyle.DashLine
        assert "channel" in page._component_label_items[key].toPlainText() or "virtual" in page._component_label_items[key].toPlainText()
        page.focus_component(resolved.assembly.part(key))
        assert isinstance(page._highlight, pg.InfiniteLine)
        assert "no separate body" in page.summary.text()
        assert "OD " not in page.summary.text()
    assert page._component_label_items["probe_hp1_hexapole"].toPlainText() == "HP1 Hexapole"
    page.display_result(resolved)
    assert set(page._channel_marker_items) == CHANNEL_KEYS


def test_legacy_nonzero_channel_envelopes_do_not_reappear_without_new_metadata(resolved):
    parts = []
    for part in resolved.assembly.parts:
        data = dict(part.data)
        for field in ("layout_role", "layout_owner", "physical_host_status"):
            data.pop(field, None)
        parts.append(SimpleNamespace(**{
            name: getattr(part, name) for name in ("key", "name", "branch", "start_z_mm", "center_z_mm", "end_z_mm")
        }, data=data))
    legacy = SimpleNamespace(assembly=SimpleNamespace(parts=parts), layout=resolved.layout)
    records = {record.key: record for record in physical_layout_records(legacy)}
    assert all(records[key].layout_role for key in CHANNEL_KEYS)
    assert all(records[key].outer_diameter_mm == 0 for key in CHANNEL_KEYS)
