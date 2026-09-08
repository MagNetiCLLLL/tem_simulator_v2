"""Sample labels do not confuse a rendering crop with a physical specimen."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from temsim.gui.sample_scene_labels import sample_scene_labels
from temsim.optics.column import default_state
from temsim.specimen.geometry import build_sample_geometry_snapshot


def test_ten_nm_sample_and_two_nm_local_region_have_separate_dimensions():
    sample = default_state().sample
    sample.size_x_nm = sample.size_y_nm = sample.thickness_nm = 10.0
    snapshot = build_sample_geometry_snapshot(
        sample, load_atoms=False,
        calculation_roi_bounds_nm_override=(-1.0, 1.0, -1.0, 1.0),
    )
    snapshot = replace(snapshot, atomic_numbers=np.full(100, 14))
    full, local, atoms = sample_scene_labels(snapshot, completed_region=True)

    assert full == "Full sample — outline | Diameter 10 nm | Thickness 10 nm"
    assert local == "Local calculation region | 2 nm × 2 nm × 10 nm"
    assert atoms == "Spheres | 100 atoms in the local region"


def test_display_subset_does_not_claim_to_be_the_full_calculation_region():
    sample = default_state().sample
    snapshot = build_sample_geometry_snapshot(sample, load_atoms=False)
    snapshot = replace(snapshot, atomic_numbers=np.full(1800, 14),
                       atom_display_capped=True, atom_display_size_nm=(3.3, 3.3, 3.3))
    full, local, atoms = sample_scene_labels(snapshot, completed_region=False)

    assert "Diameter 10 nm" in full
    assert "Local structure preview" in local
    assert "calculation" not in local
    assert "1,800 atoms shown | Display subset: 3.3 nm × 3.3 nm × 3.3 nm" in atoms


def test_empty_material_region_is_not_replaced_with_atoms():
    sample = default_state().sample
    sample.size_x_nm = sample.size_y_nm = 10.0
    snapshot = build_sample_geometry_snapshot(
        sample, load_atoms=False,
        calculation_roi_bounds_nm_override=(20.0, 30.0, 20.0, 30.0),
    )
    _full, local, atoms = sample_scene_labels(snapshot, completed_region=True)
    assert "No material intersection" in local
    assert atoms == "Spheres | No explicit atoms"


def test_page_retains_completed_region_labels_without_using_new_draft_dimensions(qtbot):
    from temsim.gui.sample_panel import SamplePage

    state = default_state()
    saved = type(state).from_dict(state.to_dict())
    saved.sample.size_x_nm = saved.sample.size_y_nm = saved.sample.thickness_nm = 10.0
    result = SimpleNamespace(
        state_snapshot=saved,
        wave_imaging=SimpleNamespace(metrics={
            "specimen_wave_window_bounds_nm": (-1.0, 1.0, -1.0, 1.0),
        }),
    )
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.display_result(result)
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)

    assert "Diameter 10 nm" in page.full_sample_label.text()
    assert "2 nm × 2 nm × 10 nm" in page.local_region_label.text()
    assert "TEM" in page.scene_status.text()
    before = page._snapshot.size_nm
    page.fit_local_region_button.click()
    page.fit_full_sample_button.click()
    assert page._snapshot.size_nm == before
    assert state.sample.size_x_nm == 10.0


def test_page_missing_cif_preserves_full_and_local_outlines(qtbot, tmp_path):
    from temsim.gui.sample_panel import SamplePage

    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(tmp_path / "missing-cif.cif")
    state.sample.size_x_nm = state.sample.size_y_nm = state.sample.thickness_nm = 10.0
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.display_result(SimpleNamespace(
        state_snapshot=state,
        wave_imaging=SimpleNamespace(metrics={
            "specimen_wave_window_bounds_nm": (-1.0, 1.0, -1.0, 1.0),
        }),
    ))
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)

    assert page._snapshot.size_nm == (10.0, 10.0, 10.0)
    assert page._snapshot.local_material_bounds_nm == (-1.0, 1.0, -1.0, 1.0, -5.0, 5.0)
    assert not page._snapshot.atomic_numbers.size
    assert page.atom_display_label.text() == "Spheres unavailable | Geometry retained"
    assert "CIF file does not exist" in page.atom_display_label.toolTip()
    assert page.fit_full_sample_button.isEnabled()
    assert page.fit_local_region_button.isEnabled()
    assert len(page.scene.view.listDataItems()) >= 2
