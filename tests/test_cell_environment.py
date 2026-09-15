"""Bounded cell geometry/transport checks; no full coherent-source execution."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.vacuum import CellWindow, Medium, ResolvedMedium, VacuumMap, resolve_regions
from temsim.physics.residual_medium import MediumTransport, medium_coefficients, medium_grid_nodes


@pytest.fixture
def state():
    from temsim.optics.column import default_state
    state = default_state()
    state.vacuum_map.enabled = True
    state.vacuum_map.cell.inserted = True
    state.vacuum_map.cell.length_mm = 100e-6
    state.vacuum_map.cell.upstream_window.thickness_nm = 5
    state.vacuum_map.cell.downstream_window.thickness_nm = 10
    state.sample.thickness_nm = 10
    return state


def test_window_faces_gap_and_authoritative_sample(state):
    original = deepcopy(state.sample)
    state.vacuum_map.cell.offset_z_mm = 20e-6
    rows = {r.key: r for r in resolve_regions(state)}
    centre = state.sample.z_mm
    assert (rows['specimen_cell'].start_z_mm-centre)*1e6 == pytest.approx(-30, abs=1e-6)
    assert (rows['specimen_cell'].end_z_mm-centre)*1e6 == pytest.approx(70, abs=1e-6)
    assert (rows['cell_window_upstream'].start_z_mm-centre)*1e6 == pytest.approx(-35, abs=1e-6)
    assert rows['cell_window_upstream'].end_z_mm == rows['specimen_cell'].start_z_mm
    assert rows['cell_window_downstream'].start_z_mm == rows['specimen_cell'].end_z_mm
    assert (rows['cell_window_downstream'].end_z_mm-centre)*1e6 == pytest.approx(80, abs=1e-6)
    assert state.sample == original


def test_window_profile_snapshot_and_map_roundtrip(state, tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.profile_io import apply_profile_values, read_profile, save_profile
    state.vacuum_map.cell.medium = Medium(phase='liquid', formula='H2O', pressure_mbar=2500,
        density_kg_m3=997, mixture_mole_fractions={'H2O': .9, 'C2H6O': .1})
    state.vacuum_map.cell.downstream_window = CellWindow(.3354, 'Graphene',
        Medium(phase='solid', formula='C', density_kg_m3=2260), 'Synthetic test input')
    state.vacuum_map.save(tmp_path / 'cell.toml')
    assert VacuumMap.load(tmp_path / 'cell.toml') == state.vacuum_map
    assert capture_instrument_snapshot(state).restore().vacuum_map == state.vacuum_map
    save_profile(tmp_path/'profile.toml', state, AssemblyCatalog().default_selection())
    _, values = read_profile(tmp_path/'profile.toml')
    restored = type(state).from_dict(state.to_dict())
    apply_profile_values(restored, values)
    assert restored.vacuum_map == state.vacuum_map


def test_historical_windowless_map_not_silently_changed(state):
    data = state.vacuum_map.to_dict()
    for key in ('upstream_window', 'downstream_window'):
        del data['cell'][key]
    restored = VacuumMap.from_dict(data)
    assert restored.cell.length_mm == data['cell']['length_mm']
    assert restored.cell.upstream_window.thickness_nm == 0
    assert restored.cell.downstream_window.thickness_nm == 0
    state.vacuum_map = restored
    assert len(resolve_regions(state)) == 7


def test_windows_disabled_with_vacuum_opt_out(state):
    state.vacuum_map.enabled = False
    assert not resolve_regions(state)
    assert not medium_grid_nodes(state, state.sample.z_mm-.001, state.sample.z_mm+.001)
    assert len(resolve_regions(state, include_disabled=True)) == 9


@pytest.mark.parametrize('thickness', [-1, float('nan'), float('inf'), True])
def test_window_thickness_validation(thickness):
    with pytest.raises(ValueError, match='thickness'):
        CellWindow(thickness_nm=thickness).validate()


@pytest.mark.parametrize('mixture', [{'Ar': .9}, {'Xx': 1}, {'Ar': -1, 'N2': 2},
                                   {'Ar': float('nan')}, {'Ar': True}, ['H2O'],
                                   {'': .5, 'Ar': .5}, {'Ar0': .5, 'Ar': .5}])
def test_mixture_validation(mixture):
    with pytest.raises(ValueError):
        Medium(mixture_mole_fractions=mixture).validate()


def test_gas_mixture_uses_partial_densities():
    mixture = Medium(pressure_mbar=8, mixture_mole_fractions={'Ar': .75, 'N2': .25}).validate()
    energy = np.array([100000., 300000.])
    expected = sum(medium_coefficients(Medium(formula=f, pressure_mbar=p), energy)[0]
                   for f, p in [('Ar', 6), ('N2', 2)])
    np.testing.assert_allclose(medium_coefficients(mixture, energy)[0], expected, rtol=2e-15)


def test_liquid_pressure_does_not_invent_equation_of_state():
    medium = Medium(phase='liquid', formula='H2O', density_kg_m3=997,
                    mixture_mole_fractions={'H2O': .5, 'C2H6O': .5}).validate()
    from scipy.constants import atomic_mass
    assert medium.number_density_m3() == pytest.approx(997/(32.042*atomic_mass), rel=1e-4)
    number = medium.number_density_m3()
    medium.pressure_mbar = 10000
    assert medium.number_density_m3() == number


def test_sample_window_overlap_rejected_not_resized(state):
    state.sample.thickness_nm = 105
    with pytest.raises(ValueError, match='Sample intersects'):
        resolve_regions(state)
    assert state.sample.thickness_nm == 105
    # Exact contact is allowed; no negative gas layer or duplicate solid path.
    state.sample.thickness_nm = 100
    assert len(resolve_regions(state)) == 9
    state.vacuum_map.cell.offset_z_mm = 50e-6
    with pytest.raises(ValueError, match='Sample intersects'):
        resolve_regions(state)


def test_displaced_sample_does_not_falsely_overlap_window(state):
    state.sample.thickness_nm = 105
    state.sample.size_x_nm = state.sample.size_y_nm = 10
    state.sample.centre_x_nm = 1e8
    assert len(resolve_regions(state)) == 9


@pytest.mark.parametrize('scale', [1e-3, 1, 1e3])
def test_elliptical_sample_overlap_matches_physical_envelope(state, scale):
    from temsim.vacuum import sample_overlaps_window
    sample, cell = state.sample, state.vacuum_map.cell
    sample.envelope_shape = 'disk'
    sample.size_x_nm, sample.size_y_nm = 20*scale, 200*scale
    sample.thickness_nm = 20
    cell.diameter_mm = 20e-6*scale
    cell.offset_y_mm = 105e-6*scale
    sample.centre_x_nm = sample.centre_y_nm = 0
    assert sample_overlaps_window(sample, cell, sample.z_mm, sample.z_mm+.001)
    cell.offset_y_mm = 115e-6*scale
    assert not sample_overlaps_window(sample, cell, sample.z_mm, sample.z_mm+.001)
    cell.offset_y_mm = 0
    cell.offset_x_mm = 25e-6*scale
    assert not sample_overlaps_window(sample, cell, sample.z_mm, sample.z_mm+.001)


def test_window_optical_depth_matches_analytic_and_refined_paths():
    medium = Medium(phase='solid', formula='Si3N4', density_kg_m3=3100)
    thickness = 5e-9
    region = ResolvedMedium('cell_window_upstream', 'window', 0, thickness*1000, medium, 1)
    expected = float(medium_coefficients(medium, np.array([300000.]))[0][0])*thickness
    results = []
    for steps in (16, 32, 64):
        run = MediumTransport([region], 8, 5)
        points = np.zeros((steps+1, 8, 3))
        points[:, :, 2] = np.linspace(0, thickness, steps+1)[:, None]
        direction = np.tile([0., 0., 1.], (8, 1))
        for a, b in zip(points, points[1:]):
            run.advance(a, b, direction, 300000.)
        np.testing.assert_allclose(run.tau[region.key], expected, rtol=2e-14, atol=1e-16)
        results.append(run.tau[region.key])
    np.testing.assert_allclose(results[0], results[-1], rtol=2e-14, atol=1e-16)


def test_outer_windows_must_fit_chamber(state):
    state.vacuum_map.cell.offset_z_mm = 4.99994
    state.vacuum_map.cell.downstream_window.thickness_nm = 100
    with pytest.raises(ValueError, match='fit inside'):
        resolve_regions(state)


@pytest.mark.parametrize('field', ['gap', 'window'])
def test_sub_float_resolution_geometry_rejected_explicitly(state, field):
    if field == 'gap':
        state.vacuum_map.cell.length_mm = 1e-30
    else:
        state.vacuum_map.cell.upstream_window.thickness_nm = 1e-30
    with pytest.raises(ValueError, match='numerical resolution'):
        resolve_regions(state)


def test_ambient_replaced_by_union_of_windows_and_fluid():
    # Synthetic weak media allow a single analytic step through every layer.
    gas = Medium(pressure_mbar=1e-10)
    solid = Medium(phase='solid', formula='Si3N4', density_kg_m3=1e-6)
    regions = [ResolvedMedium('specimen', 'ambient', 0, 4, gas),
        ResolvedMedium('cell_window_upstream', 'up', 1, 1.1, solid, .5),
        ResolvedMedium('specimen_cell', 'fluid', 1.1, 2.9, gas, .5),
        ResolvedMedium('cell_window_downstream', 'down', 2.9, 3, solid, .5)]
    run = MediumTransport(regions, 3, 914)
    run.solid_specimen = SimpleNamespace(z_mm=2, thickness_nm=1e6, size_x_nm=2e6,
        size_y_nm=2e6, centre_x_nm=0, centre_y_nm=0, envelope_shape='disk')
    start = np.array([[0., 0, 0], [.001, 0, 0], [0, 0, 0]])
    end = start+[0, 0, .004]
    end[2, 0] = .0001
    direction = end-start
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    run.advance(start, end, direction, 300000)
    elongation = np.sqrt(1+.025**2)
    np.testing.assert_allclose(run.path_m['specimen'], [.002, .003, .002*elongation])
    np.testing.assert_allclose(run.path_m['specimen_cell'], [.0008, 0, .0008*elongation])
    for key in ('cell_window_upstream', 'cell_window_downstream'):
        np.testing.assert_allclose(run.path_m[key], [.0001, 0, .0001*elongation])
    np.testing.assert_allclose(sum(run.path_m.values()), [.003, .003, .003*elongation])


def test_vacuum_interior_displaces_ambient_but_retains_windows():
    weak = Medium(phase='solid', formula='C', density_kg_m3=1e-6)
    rows = [ResolvedMedium('specimen', 'ambient', 0, 4, Medium(pressure_mbar=1e-10)),
        ResolvedMedium('cell_window_upstream', 'up', 1, 1.1, weak, 1),
        ResolvedMedium('specimen_cell', 'vacuum', 1.1, 2.9, Medium(phase='vacuum'), 1),
        ResolvedMedium('cell_window_downstream', 'down', 2.9, 3, weak, 1)]
    run = MediumTransport(rows, 1, 4)
    run.advance(np.array([[0., 0, 0]]), np.array([[0., 0, .004]]), np.array([[0., 0, 1]]), 300000.)
    assert run.path_m['specimen'][0] == pytest.approx(.002)
    assert run.path_m['cell_window_downstream'][0] == pytest.approx(.0001)
    assert run.tau['cell_window_downstream'][0] > 0


def test_graphene_window_exact_grid_and_cache_identity(state):
    from temsim.calculation_cache import calculation_signatures
    from temsim.physics.core import build_propagation_plan
    before = calculation_signatures(state)
    state.vacuum_map.cell.upstream_window.thickness_nm = .3354
    after = calculation_signatures(state)
    assert after['incident'] != before['incident']
    z = state.sample.z_mm
    plan = build_propagation_plan(state, z-.001, z+.001, particle_medium=True)
    row = next(r for r in resolve_regions(state) if r.key == 'cell_window_upstream')
    assert row.start_z_mm in plan.z_mm and row.end_z_mm in plan.z_mm
    inside = plan.z_mm[(plan.z_mm >= row.start_z_mm) & (plan.z_mm <= row.end_z_mm)]
    assert len(inside) >= 17
    assert np.all(np.diff(inside) > 0)


def test_executed_particle_transport_consumes_both_windows(state):
    from temsim.physics.core import propagate
    state.vacuum_map.cell.medium.phase = 'vacuum'
    z = state.sample.z_mm
    zeros = np.zeros(400)
    reports = []
    # Bounded optical segment; does not replace the full upstream source chain.
    _, _, tx, _, ty = propagate(state, z-.0002, z+.0002, zeros, zeros, zeros, zeros,
                                particle_medium=True, medium_output=reports)
    assert len(reports) == 1
    result = reports[0]
    for key, thickness in [('cell_window_upstream', 5e-9), ('cell_window_downstream', 1e-8)]:
        assert np.count_nonzero(result.path_m[key]) > 300
        assert np.sum(result.events[key]) > 0
        assert np.median(result.path_m[key]) == pytest.approx(thickness, rel=.02)
    assert np.any(np.hypot(tx[-1], ty[-1]) > 1e-6)
