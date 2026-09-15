"""Applied cell geometry, pressure and transport share one source of truth."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cell_geometry import CellPhysicalContext, CELL_SAMPLE_KEY
from temsim.vacuum import VacuumMap, Medium, resolve_cell_layers, resolve_regions
from temsim.physics.residual_medium import MediumTransport, medium_coefficients


@pytest.fixture
def state():
    from temsim.optics.column import default_state
    state = default_state()
    state.vacuum_map.cell.inserted = True
    state.vacuum_map.cell.diameter_mm = .002
    state.vacuum_map.cell.length_mm = 100e-6
    state.sample.thickness_nm = 10
    state.vacuum_map.cell.upstream_window.thickness_nm = 8
    state.vacuum_map.cell.downstream_window.thickness_nm = .3354
    return state


def test_geometry_and_transport_faces_match_even_while_opted_out(state):
    state.vacuum_map.cell.upstream_window.diameter_mm = .004
    state.vacuum_map.cell.offset_x_mm = .0002
    state.vacuum_map.cell.offset_y_mm = -.0003
    context = CellPhysicalContext.capture(state)
    assert not resolve_regions(state)
    state.vacuum_map.enabled = True
    transport = {r.key: r for r in resolve_regions(state)}
    for r in context.layers:
        assert r == transport[r.key]
    assert next(r.radius_mm for r in context.layers if r.key == 'cell_window_upstream') == .002
    meshes = {m.key: m for m in context.meshes()}
    assert CELL_SAMPLE_KEY in meshes
    assert meshes['specimen_cell'].wireframe and meshes[CELL_SAMPLE_KEY].wireframe
    assert not meshes['cell_window_upstream'].wireframe
    for r in context.layers:
        vertices = meshes[r.key].vertices
        assert vertices[:, 2].min() == r.start_z_mm
        assert vertices[:, 2].max() == r.end_z_mm
        radial = np.hypot(vertices[:, 0]-r.center_x_mm, vertices[:, 1]-r.center_y_mm)
        assert radial.max() == pytest.approx(r.radius_mm, rel=1e-12)
        assert not vertices.flags.writeable
    before = context.signature()
    state.vacuum_map.cell.upstream_window.thickness_nm = 12
    assert context.signature() == before  # Presentation owns a detached capture.
    assert CellPhysicalContext.capture(state).signature() != before


@pytest.mark.parametrize('start_pressure,end_pressure', [(1., 9.), (9., 1.), (0., 10.)])
def test_linear_cell_pressure_integrates_once(state, start_pressure, end_pressure):
    cell = state.vacuum_map.cell
    cell.medium = Medium(formula='He', pressure_mbar=start_pressure)
    cell.pressure_gradient_enabled, cell.end_pressure_mbar = True, end_pressure
    cell.upstream_window.thickness_nm = cell.downstream_window.thickness_nm = 0
    r = resolve_cell_layers(state)[0]
    assert r.end_medium.pressure_mbar == end_pressure
    run = MediumTransport([r], 1, 4)
    start, end = np.array([[0., 0., r.start_z_mm*.001]]), np.array([[0., 0., r.end_z_mm*.001]])
    run.advance(start, end, np.array([[0., 0., 1.]]), 300000.)
    # Independent ideal-gas linear-density integral = midpoint pressure * path.
    midpoint = replace(cell.medium, pressure_mbar=(start_pressure+end_pressure)/2)
    expected = medium_coefficients(midpoint, 300000.)[0] * np.linalg.norm(end-start)
    assert run.tau[r.key][0] == pytest.approx(expected, rel=1e-8)
    assert run.path_m[r.key][0] == pytest.approx(cell.length_mm*.001, rel=1e-8)


def test_wider_window_intercepts_outside_fluid_without_double_ambient(state):
    cell = state.vacuum_map.cell
    cell.upstream_window.diameter_mm = .004
    cell.medium.phase = 'vacuum'
    # Explicit weak synthetic material isolates geometry, not a material preset.
    cell.upstream_window.medium.density_kg_m3 = 1e-6
    cell.downstream_window.medium.density_kg_m3 = 1e-6
    rows = list(resolve_cell_layers(state))
    from temsim.vacuum import ResolvedMedium
    z0, z1 = min(r.start_z_mm for r in rows)-1e-5, max(r.end_z_mm for r in rows)+1e-5
    rows.append(ResolvedMedium('ambient', 'ambient', z0, z1, Medium(pressure_mbar=1e-9)))
    run = MediumTransport(rows, 1, 44)
    start, end = np.array([[1.5e-6, 0., z0*.001]]), np.array([[1.5e-6, 0., z1*.001]])
    run.advance(start, end, np.array([[0., 0., 1.]]), 300000.)
    path = run.path_m
    assert path['cell_window_upstream'][0] == pytest.approx(8e-9, rel=1e-7)
    assert path['cell_window_downstream'][0] == path['specimen_cell'][0] == 0
    assert sum(p[0] for p in path.values()) == pytest.approx(np.linalg.norm(end-start), rel=1e-10)


def test_new_inputs_roundtrip_and_invalidate_active_cache(state, tmp_path):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.calculation_cache import calculation_signatures
    state.vacuum_map.enabled = True
    before = calculation_signatures(state)
    cell = state.vacuum_map.cell
    cell.pressure_gradient_enabled, cell.end_pressure_mbar = True, 30.
    cell.downstream_window.diameter_mm = .003
    assert calculation_signatures(state)['incident'] != before['incident']
    path = tmp_path/'map.toml'
    state.vacuum_map.save(path)
    assert VacuumMap.load(path) == state.vacuum_map
    assert capture_instrument_snapshot(state).restore().vacuum_map == state.vacuum_map
    historical = state.vacuum_map.to_dict()
    del historical['cell']['end_pressure_mbar'], historical['cell']['pressure_gradient_enabled']
    del historical['cell']['upstream_window']['diameter_mm'], historical['cell']['downstream_window']['diameter_mm']
    old = VacuumMap.from_dict(historical)
    assert not old.cell.pressure_gradient_enabled
    assert old.cell.upstream_window.radius_mm(old.cell) == old.cell.diameter_mm/2


def test_invalid_window_and_liquid_gradient_rejected(state):
    state.vacuum_map.cell.upstream_window.diameter_mm = .001
    with pytest.raises(ValueError, match='cover the cell aperture'):
        state.vacuum_map.validate()
    state.vacuum_map.cell.upstream_window.diameter_mm = 0
    state.vacuum_map.cell.medium.phase = 'liquid'
    state.vacuum_map.cell.pressure_gradient_enabled = True
    with pytest.raises(ValueError, match='requires a gas'):
        state.vacuum_map.validate()


@pytest.mark.parametrize('phase,gap_mm', [('gas', .1), ('liquid', .0001)])
def test_executed_cell_medium_scattering_and_pressure_response(state, phase, gap_mm):
    """Real short particle segment with active objective field; not a new source."""
    from temsim.physics.core import propagate
    cell = state.vacuum_map.cell
    state.vacuum_map.enabled = True
    state.sample.inserted = False  # Isolate the cell, not an overlapping solid.
    cell.length_mm, cell.diameter_mm = gap_mm, .01
    cell.upstream_window.thickness_nm = cell.downstream_window.thickness_nm = 0
    cell.medium = Medium(phase=phase, formula='N2' if phase == 'gas' else 'H2O',
                         pressure_mbar=1000, density_kg_m3=1000)
    cell.pressure_gradient_enabled = phase == 'gas'
    cell.end_pressure_mbar = 2000
    z = state.sample.z_mm
    zeros = np.zeros(512)
    reports = []
    _, _, tx, _, ty = propagate(state, z-gap_mm/2, z+gap_mm/2,
        zeros, zeros, zeros, zeros, particle_medium=True, medium_output=reports)
    report = reports[0]
    assert np.sum(report.events['specimen_cell']) > 10
    assert np.any(np.hypot(tx[-1], ty[-1]) > 1e-6)
    assert np.median(report.path_m['specimen_cell']) == pytest.approx(gap_mm*.001, rel=.02)
    midpoint = replace(cell.medium, pressure_mbar=1500 if phase == 'gas' else 1000)
    expected = medium_coefficients(midpoint, np.array([300000.]))[0][0]*gap_mm*.001
    assert np.median(report.tau['specimen_cell']) == pytest.approx(expected, rel=.02)
    if phase == 'gas':
        cell.medium.pressure_mbar = cell.end_pressure_mbar = 0
        clear = []
        propagate(state, z-gap_mm/2, z+gap_mm/2, zeros, zeros, zeros, zeros,
                  particle_medium=True, medium_output=clear)
        assert not np.any(clear[0].events['specimen_cell'])


def test_live_physical_overlay_and_3d_reuse_without_calculation(state, qtbot, monkeypatch):
    from temsim.gui.diagnostic_tabs import PhysicalLayoutView
    from temsim.gui import assembly_model_page
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.resize(1300, 800)
    page.show()
    page.assembly_3d.set_assembly(state._resolved_assembly)
    page.plot.setRange(xRange=(1000, 1800), yRange=(-30, 30), padding=0)
    previous = deepcopy(page.plot.getViewBox().viewRange())
    page.set_cell_state(state)
    assert page.plot.getViewBox().viewRange() == previous
    assert page.cell_part('cell_window_downstream').data['material'] == 'Si3N4'
    assert page.cell_overlay.fit()
    page.tabs.setCurrentWidget(page.assembly_3d)
    qtbot.waitUntil(lambda: page.assembly_3d.mesh_builds > 0, timeout=15000)
    view = page.assembly_3d
    assert not view._model.errors
    assert 'cell_window_upstream' in view._tree_items
    assert view.cell_only.isEnabled()
    view.cell_only.setChecked(True)
    assert {m.key for m in view.view._meshes} == {'specimen_cell', 'cell_window_upstream', 'cell_window_downstream', CELL_SAMPLE_KEY}
    assert page._fit_cell_view()
    assert view.view._radius < .01
    view.focus_component('cell_window_upstream')
    assert view.edit_part.text() == 'Edit cell / windows'
    with qtbot.waitSignal(view.edit_part_requested) as event:
        view.edit_part.click()
    assert event.args[0] == 'cell_window_upstream'
    monkeypatch.setattr(assembly_model_page, 'assembly_model_from_assembly', lambda *a, **k: pytest.fail('Unchanged column rebuilt'))
    state.vacuum_map.cell.upstream_window.thickness_nm = 18
    page.set_cell_state(state)
    qtbot.waitUntil(lambda: view.mesh_builds == 2)
    state.vacuum_map.cell.inserted = False
    page.set_cell_state(state)
    qtbot.waitUntil(lambda: view.mesh_builds == 3)
    assert not any(m.key.startswith('cell_') or m.key == 'specimen_cell' for m in view._model.meshes)
    assert not page.fit_cell.isEnabled()
    assert not view.cell_only.isEnabled()
    assert view.view._meshes  # Retracting the cell restores the column display.
