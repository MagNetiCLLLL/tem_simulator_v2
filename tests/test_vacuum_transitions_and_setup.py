from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.vacuum import Medium, ResolvedMedium, resolve_regions
from temsim.physics.residual_medium import MediumTransport, medium_coefficients


@pytest.fixture(scope="module")
def state():
    from temsim.optics.column import default_state
    value = default_state()
    value.vacuum_map.enabled = True  # These fixtures exercise enabled transport.
    return value


def test_gap_follows_component_and_does_not_move_neighbours(state):
    s = copy(state)
    s.vacuum_map = deepcopy(state.vacuum_map)
    region = s.vacuum_map.regions[2]
    region.end_anchor = "condenser_lens_2.center"
    region.end_offset_mm = 3
    rows = resolve_regions(s)
    gap = next(r for r in rows if r.end_medium is not None)
    assert gap.start_z_mm == state._resolved_assembly.part("condenser_lens_2").center_z_mm+3
    assert gap.end_z_mm == state.sample.z_mm-5
    assert gap.medium == region.medium
    assert gap.end_medium == s.vacuum_map.regions[3].medium
    s.vacuum_map.enabled = False
    assert not resolve_regions(s)
    assert resolve_regions(s, include_disabled=True) == rows
    s.vacuum_map.regions[2].end_anchor = "sample"
    s.vacuum_map.regions[2].end_offset_mm = 0
    with pytest.raises(ValueError, match="overlap"):
        resolve_regions(s, include_disabled=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_linear_gap_integrates_clipped_oblique_path_and_mixed_gases(reverse):
    a = Medium(formula="N2", pressure_mbar=1e-6, temperature_k=290)
    b = Medium(formula="He", pressure_mbar=4e-6, temperature_k=310)
    region = ResolvedMedium("gap", "gap", 0, 2, a, end_medium=b)
    start = np.array([[0., 0., -.001], [0., 0., .0005]])
    end = np.array([[.004, 0., .003], [.001, 0., .0015]])
    if reverse:
        start, end = end, start
    expected = (medium_coefficients(a, np.array([300000.]))[0]+medium_coefficients(b, np.array([300000.]))[0])*.5
    results = []
    for steps in (1, 17):
        run = MediumTransport([region], 2, 10)
        for i in range(steps):
            run.advance(start+(end-start)*i/steps, start+(end-start)*(i+1)/steps,
                        np.tile([0., 0., 1.], (2, 1)), 300000.)
        np.testing.assert_allclose(run.path_m["gap"], np.sqrt(2)*np.array([.002, .001]))
        np.testing.assert_allclose(run.tau["gap"], expected*run.path_m["gap"], rtol=1e-12)
        results.append(run.tau["gap"])
    np.testing.assert_allclose(*results, rtol=1e-12)


def test_linear_removal_position_has_exact_integrated_hazard():
    a = Medium(phase="vacuum")
    b = Medium(pressure_mbar=1e-8, removal_cross_section_m2=1e-10, removal_reference="mathematical fixture")
    region = ResolvedMedium("gap", "gap", 0, 1, a, end_medium=b)
    run = MediumTransport([region], 1, 3)
    loss = b.number_density_m3()*b.removal_cross_section_m2
    run.removal_clock[:] = .5*loss*.001*.3**2
    run.advance(np.zeros((1, 3)), np.array([[0., 0., .001]]), np.array([[0., 0., 1.]]), 300000.)
    assert not run.alive[0]
    assert run.blocked_z[0] == pytest.approx(.3)
    assert run.path_m["gap"][0] == pytest.approx(.0003)


def test_setup_changes_readouts_not_scan_or_detectors(qtbot, state, tmp_path):
    from temsim.gui.calculate_setup import CalculateSetupDialog
    from temsim.gui.calculation_controller import CalculationController
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    s = capture_instrument_snapshot(state).restore()
    s.ac_deflector.wobble_enabled = False
    s.ac_deflector.scan_enabled = True
    dialog = CalculateSetupDialog(s)
    qtbot.addWidget(dialog)
    before = [(d.inserted, d.inner_diameter_mm) for d in s.stem_detectors]
    for key in ("stem", "eds", "vacuum"):
        dialog.controls[key][0].setChecked(False)
    assert dialog.apply()
    assert not s.sample.stem_image_enabled and not s.sample.eds_enabled and not s.vacuum_map.enabled
    assert s.ac_deflector.scan_enabled
    assert before == [(d.inserted, d.inner_diameter_mm) for d in s.stem_detectors]
    requested = CalculationController._requested_design_stage_keys(s)
    assert "stem" not in requested and "eds" not in requested
    assert "column" in requested and "scan_geometry" in requested
    assert not dialog.controls["tem"][0].isEnabled()
    assert not dialog.controls["stem_wave"][0].isEnabled()
    path = tmp_path/"setup.toml"
    save_profile(path, s, AssemblyCatalog().default_selection())
    restored = capture_instrument_snapshot(state).restore()
    _, values = read_profile(path)
    apply_profile_values(restored, values)
    assert not restored.sample.stem_image_enabled
    assert not restored.sample.eds_enabled and not restored.vacuum_map.enabled


def test_setup_stem_off_skips_frame_without_retracting_detectors(state, monkeypatch):
    from temsim.simulation_pipeline import calculate_stem_scan_frame
    s = copy(state)
    s.sample = replace(state.sample, stem_image_enabled=False)
    s.deflectors = deepcopy(state.deflectors)
    s.ac_deflector.wobble_enabled = False
    s.ac_deflector.scan_enabled = True
    monkeypatch.setattr("temsim.simulation_pipeline.acquire_stem_scan", lambda *a, **k: pytest.fail("disabled readout ran"))
    assert calculate_stem_scan_frame(s, object()) is None


def test_map_tab_order_horizontal_and_disabled_editing(qtbot, state):
    from temsim.gui.visualization import VisualizationWorkspace
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    page = workspace.vacuum_map
    page.set_state(state)
    labels = [workspace.tabs.tabText(i) for i in range(workspace.tabs.count())]
    assert labels[labels.index("Sample")+1] == "Vacuum map"
    assert "Vacuum map" not in [workspace.physical_layout.tabs.tabText(i) for i in range(workspace.physical_layout.tabs.count())]
    page.resize(1250, 700)
    page.show()
    qtbot.wait(30)
    page.diagram.grab()
    boxes = [b for b, key in page.diagram.boxes if key != "specimen_cell"]
    assert len(boxes) == 6
    assert len({b.top() for b in boxes}) == 1
    assert [b.left() for b in boxes] == sorted(b.left() for b in boxes)


def test_surface_tuning_origin_weights_and_cache(state):
    from temsim.optics.electron_gun.tip_surface import surface_bundle
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    from temsim import module_manifest
    state.electron_gun.emitter.surface_model = model_from_part(
        module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    model = state.electron_gun.emitter.surface_model
    # Support probes carry no current; compare the same 48 physical samples.
    # Grouped surface sampling need not have a stable prefix at a new count.
    a = surface_bundle(model, 48)
    b = surface_bundle(model, 49, support_probes=1)
    np.testing.assert_array_equal(a.surface_position_m, b.surface_position_m[:-1])
    np.testing.assert_array_equal(a.surface_direction, b.surface_direction[:-1])
    np.testing.assert_array_equal(b.surface_position_m[-1], [0, 0, 0])
    np.testing.assert_array_equal(b.surface_direction[-1], [0, 0, 1])
    assert b.weight[-1] == 0 and b.weight.sum() == pytest.approx(1.)
    medium = surface_bundle(model, 193, support_probes=33)
    assert np.count_nonzero(medium.weight == 0) == 33
    assert np.all(medium.surface_position_m[-33:, 2] <= 0)
    gun = deepcopy(state.electron_gun)
    key = gun._cache_key(49)
    gun.emitter._tuning_surface_probes = 1
    assert gun._cache_key(49) != key
