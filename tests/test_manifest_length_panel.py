"""Length edits update staged table coordinates without writing a manifest."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit

from temsim.gui.parameter_panel import ParameterPanel
from temsim.manifest_editor import ManifestField, ManifestTarget, parse_toml_value


PART_KEY = "example_part"


def _part_fields(start=137.5, center=252.5, end=367.5, length=230.0):
    return tuple(
        ManifestField(("parts", PART_KEY, name), name, value)
        for name, value in (
            ("length_mm", length),
            ("local_start_z_mm", start),
            ("local_center_z_mm", center),
            ("local_end_z_mm", end),
            ("optical_reference_local_z_mm", 244.0),
            ("mechanical_outer_diameter_mm", 60.0),
        )
    )


@pytest.fixture
def panel_factory(qtbot):
    def create(fields=None, *, target=None):
        panel = ParameterPanel()
        qtbot.addWidget(panel)
        panel.set_context(
            "Example part",
            None,
            target or ManifestTarget("example.toml", PART_KEY),
            _part_fields() if fields is None else fields,
            None,
        )
        panel.tabs.setCurrentIndex(1)
        panel.resize(760, 620)
        panel.show()
        return panel

    return create


def _item(panel, name):
    return next(
        panel.manifest_table.item(row, 1)
        for row in range(panel.manifest_table.rowCount())
        if panel.manifest_table.item(row, 0).text() == name
    )


def _values(panel):
    return {
        panel.manifest_table.item(row, 0).text(): parse_toml_value(
            panel.manifest_table.item(row, 1).text()
        )
        for row in range(panel.manifest_table.rowCount())
    }


def _begin_edit(qtbot, panel, name, text):
    table = panel.manifest_table
    item = _item(panel, name)
    table.setCurrentItem(item)
    table.scrollToItem(item)
    table.editItem(item)
    QApplication.processEvents()
    editor = next(editor for editor in table.findChildren(QLineEdit) if editor.isVisible())
    editor.selectAll()
    qtbot.keyClicks(editor, text)
    return editor


def _edit(qtbot, panel, name, text):
    editor = _begin_edit(qtbot, panel, name, text)
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    QApplication.processEvents()


def test_length_commit_updates_visible_endpoints_and_save_payload(qtbot, panel_factory):
    panel = panel_factory()
    saved = []
    panel.manifest_save_requested.connect(lambda target, updates: saved.append((target, updates)))
    before = _values(panel)

    editor = _begin_edit(qtbot, panel, "length_mm", "225")
    assert _values(panel) == before  # Delegate input has not been committed yet.
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    QApplication.processEvents()

    expected = {**before, "length_mm": 225, "local_start_z_mm": 140.0, "local_end_z_mm": 365.0}
    assert _values(panel) == expected
    assert saved == []
    qtbot.mouseClick(panel.save_manifest_button, Qt.MouseButton.LeftButton)
    assert saved == [
        (
            ManifestTarget("example.toml", PART_KEY),
            {
                ("parts", PART_KEY, "length_mm"): 225,
                ("parts", PART_KEY, "local_start_z_mm"): 140.0,
                ("parts", PART_KEY, "local_end_z_mm"): 365.0,
            },
        )
    ]


def test_repeated_lengths_use_current_coordinates_and_can_restore_original(qtbot, panel_factory):
    panel = panel_factory()
    before = _values(panel)
    saved = []
    panel.manifest_save_requested.connect(lambda _target, updates: saved.append(updates))

    for length, start, end in ((225, 140, 365), (220, 142.5, 362.5), (230, 137.5, 367.5)):
        _edit(qtbot, panel, "length_mm", str(length))
        values = _values(panel)
        assert values["local_start_z_mm"] == start
        assert values["local_center_z_mm"] == 252.5
        assert values["local_end_z_mm"] == end

    assert _values(panel) == before
    qtbot.mouseClick(panel.save_manifest_button, Qt.MouseButton.LeftButton)
    assert saved == [{}]


def test_asymmetric_part_preserves_centre_fraction_and_live_coordinate_edits(qtbot, panel_factory):
    panel = panel_factory(_part_fields(start=10.0, center=20.0, end=50.0, length=40.0))
    _edit(qtbot, panel, "length_mm", "20")
    assert _values(panel)["local_start_z_mm"] == 15.0
    assert _values(panel)["local_end_z_mm"] == 35.0

    _edit(qtbot, panel, "local_center_z_mm", "25")
    _edit(qtbot, panel, "length_mm", "30")
    assert _values(panel)["local_start_z_mm"] == 10.0
    assert _values(panel)["local_center_z_mm"] == 25
    assert _values(panel)["local_end_z_mm"] == 40.0


def test_zero_span_uses_symmetric_endpoints(qtbot, panel_factory):
    panel = panel_factory(_part_fields(start=20.0, center=20.0, end=20.0, length=0.0))
    _edit(qtbot, panel, "length_mm", "10")
    assert _values(panel)["local_start_z_mm"] == 15.0
    assert _values(panel)["local_center_z_mm"] == 20.0
    assert _values(panel)["local_end_z_mm"] == 25.0


@pytest.mark.parametrize("text", ["-1", "nan", "inf", '"invalid"', "invalid", "true"])
def test_invalid_lengths_leave_other_cells_untouched(qtbot, panel_factory, text):
    panel = panel_factory()
    before = {name: _item(panel, name).text() for name in _values(panel) if name != "length_mm"}
    saved = []
    errors = []
    panel.manifest_save_requested.connect(lambda *_args: saved.append(True))
    panel.error.connect(errors.append)

    _edit(qtbot, panel, "length_mm", text)

    assert _item(panel, "length_mm").text() == text
    assert {name: _item(panel, name).text() for name in before} == before
    assert saved == []
    assert errors == []
    _edit(qtbot, panel, "length_mm", "225")
    assert _values(panel)["local_start_z_mm"] == 140.0
    assert _values(panel)["local_end_z_mm"] == 365.0


def test_invalid_coordinate_prevents_partial_length_linkage(qtbot, panel_factory):
    panel = panel_factory()
    _edit(qtbot, panel, "local_center_z_mm", "invalid")
    _edit(qtbot, panel, "length_mm", "225")
    assert _item(panel, "local_center_z_mm").text() == "invalid"
    assert _item(panel, "local_start_z_mm").text() == "137.5"
    assert _item(panel, "local_end_z_mm").text() == "367.5"


@pytest.mark.parametrize("name", ["length_mm", "local_center_z_mm"])
@pytest.mark.parametrize("text", ["[252.5]", "{ value = 252.5 }", "1" + "0" * 400])
def test_valid_toml_with_invalid_numeric_type_stays_staged_for_save(
    qtbot, panel_factory, name, text
):
    panel = panel_factory()
    errors = []
    saved = []
    panel.error.connect(errors.append)
    panel.manifest_save_requested.connect(lambda _target, updates: saved.append(updates))

    _edit(qtbot, panel, name, text)
    if name != "length_mm":
        _edit(qtbot, panel, "length_mm", "225")

    assert _item(panel, name).text() == text
    assert _item(panel, "local_start_z_mm").text() == "137.5"
    assert _item(panel, "local_end_z_mm").text() == "367.5"
    assert errors == []
    assert saved == []

    # Numeric/domain validation remains in the existing save receiver; valid
    # TOML is passed through without silently replacing the invalid value.
    qtbot.mouseClick(panel.save_manifest_button, Qt.MouseButton.LeftButton)
    expected = {("parts", PART_KEY, name): parse_toml_value(text)}
    if name != "length_mm":
        expected[("parts", PART_KEY, "length_mm")] = 225
    assert saved == [expected]
    assert errors == []


def test_invalid_toml_length_uses_existing_save_error_signal(qtbot, panel_factory):
    panel = panel_factory()
    errors = []
    saved = []
    panel.error.connect(errors.append)
    panel.manifest_save_requested.connect(lambda *_args: saved.append(True))
    _edit(qtbot, panel, "length_mm", "invalid")

    qtbot.mouseClick(panel.save_manifest_button, Qt.MouseButton.LeftButton)

    assert len(errors) == 1
    assert "Invalid TOML value" in errors[0]
    assert saved == []


@pytest.mark.parametrize("name, text", [("local_center_z_mm", "250"), ("local_start_z_mm", "135"), ("local_end_z_mm", "370"), ("mechanical_outer_diameter_mm", "55")])
def test_non_length_edits_do_not_update_other_fields(qtbot, panel_factory, name, text):
    panel = panel_factory()
    before = _values(panel)
    _edit(qtbot, panel, name, text)
    assert _values(panel) == {**before, name: parse_toml_value(text)}


def test_module_geometry_length_is_not_a_part_length(qtbot, panel_factory):
    fields = tuple(
        ManifestField(("geometry", field.label), field.label, field.value)
        for field in _part_fields()
    )
    panel = panel_factory(fields, target=ManifestTarget("example.toml"))
    before = _values(panel)
    _edit(qtbot, panel, "length_mm", "225")
    assert _values(panel) == {**before, "length_mm": 225}
    assert _item(panel, "length_mm").toolTip() == ""


def test_context_and_direct_reload_do_not_link_staged_fields(panel_factory):
    # Deliberately inconsistent staged fields reveal any unintended resize.
    fields = _part_fields(length=225.0)
    panel = panel_factory(fields)
    assert _values(panel)["local_start_z_mm"] == 137.5
    assert _values(panel)["local_end_z_mm"] == 367.5
    _item(panel, "length_mm").setText("220")
    panel._load_manifest()
    assert _values(panel)["local_start_z_mm"] == 137.5
    assert _values(panel)["local_end_z_mm"] == 367.5
    assert panel.manifest_table.signalsBlocked() is False


def test_length_and_endpoints_explain_linkage(panel_factory):
    panel = panel_factory()
    for name in ("length_mm", "local_start_z_mm", "local_end_z_mm"):
        tooltip = _item(panel, name).toolTip()
        assert "local_center_z_mm" in tooltip
        assert "Gaps" in tooltip
    assert "Validate and save TOML" in _item(panel, "length_mm").toolTip()
