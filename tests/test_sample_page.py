import json

import pytest
from PySide6.QtWidgets import QLabel

from temsim.gui.sample_panel import SamplePage
from temsim.optics.column import default_state
from temsim.specimen.atomistic import atomistic_capability


def test_sample_page_binds_modes_envelope_and_safe_offscreen_view(qtbot):
    state = default_state()
    page = SamplePage()
    qtbot.addWidget(page)

    page.set_state(state)
    page.mode.setCurrentIndex(page.mode.findData("virtual"))
    page.scalar_controls["size_x_nm"].setValue(250.0)
    page.inserted.setChecked(False)

    assert state.sample.specimen_mode == "virtual"
    assert state.sample.size_x_nm == 250.0
    assert state.sample.inserted is False
    assert page.real_group.isHidden() is True
    assert page.virtual_group.isHidden() is False
    assert page.scene.opengl_available is False
    assert "offscreen" in page.scene.opengl_detail


def test_sample_page_applies_extensible_interaction_table(qtbot):
    state = default_state()
    state.sample.specimen_mode = "virtual"
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.interaction_table.setRowCount(0)
    page._append_table_row(
        page.interaction_table,
        (
            True,
            "absorbed",
            "absorption",
            0.2,
            json.dumps({}),
        ),
    )

    page._apply_interactions()

    assert state.sample.virtual_interactions == [
        {
            "enabled": True,
            "name": "absorbed",
            "kind": "absorption",
            "probability": 0.2,
        }
    ]


def test_sample_page_contains_only_structure_and_labels_ball_elements(
    qtbot,
    tmp_path,
):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "nacl.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64))
    state = default_state()
    state.sample.cif_path = str(path)
    state.sample.size_x_nm = 1.2
    state.sample.size_y_nm = 1.2
    state.sample.thickness_nm = 1.2
    page = SamplePage()
    qtbot.addWidget(page)

    page.set_state(state)

    assert not hasattr(page, "image_panels")
    assert page._snapshot.atomic_numbers.size > 2
    assert page._snapshot.atom_bond_pairs.shape[0] > 0
    legend_text = "\n".join(
        label.text() for label in page.element_legend.findChildren(QLabel)
    )
    assert "Na — Sodium" in legend_text
    assert "Cl — Chlorine" in legend_text
