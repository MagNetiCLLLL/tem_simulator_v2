from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.vacuum import Medium, VacuumMap, ResolvedMedium, resolve_regions, bind_gun_environment
from temsim.physics.residual_medium import (MediumTransport, atomic_cross_section, medium_coefficients,
                                            segment_fraction, specimen_interval, medium_grid_nodes)


@pytest.fixture(scope="module")
def instrument():
    from temsim.optics.column import default_state
    state = default_state()
    state.vacuum_map.enabled = True  # Explicit opt-in for medium transport tests.
    return state


def test_default_pressures_and_physical_boundaries(instrument):
    regions = resolve_regions(instrument)
    assert [r.medium.pressure_mbar for r in regions] == [3e-11, 2e-8, 1e-7, 1e-7, 4e-8, 8e-7]
    assert regions[-2].end_z_mm == regions[-1].start_z_mm
    assert regions[-1].start_z_mm == instrument._resolved_assembly.part("projection_chamber_dpa_aperture").center_z_mm
    assert regions[1].end_z_mm == instrument.electron_gun.exit_plane_z_mm
    assert regions[3].start_z_mm == instrument.sample.z_mm-5


def test_map_persistence_and_historical_state(instrument, tmp_path):
    config = deepcopy(instrument.vacuum_map)
    config.cell.inserted = True
    config.cell.medium = Medium(phase="liquid", formula="H2O", density_kg_m3=997)
    config.save(tmp_path/"vacuum.toml")
    assert VacuumMap.load(tmp_path/"vacuum.toml").to_dict() == config.to_dict()
    from temsim.optics.model import State
    data = instrument.to_dict()
    assert State.from_dict(data).vacuum_map.to_dict() == instrument.vacuum_map.to_dict()
    del data["vacuum_map"]
    assert not State.from_dict(data).vacuum_map.enabled
    from temsim.instrument_snapshot import capture_instrument_snapshot
    snapshot = capture_instrument_snapshot(instrument)
    assert snapshot is not None
    assert snapshot.restore().vacuum_map.to_dict() == instrument.vacuum_map.to_dict()


@pytest.mark.parametrize("field,value", [("pressure_mbar", -1), ("temperature_k", 0),
    ("pressure_mbar", float("nan")), ("density_kg_m3", -1), ("formula", "NotAnElement")])
def test_reject_invalid_medium(field, value):
    medium = Medium()
    setattr(medium, field, value)
    with pytest.raises(ValueError):
        medium.validate()


def test_density_and_removal_are_independent():
    from scipy.constants import Boltzmann, Avogadro
    assert Medium(pressure_mbar=1, temperature_k=300).number_density_m3() == pytest.approx(100/(Boltzmann*300))
    assert Medium(phase="liquid", formula="H2O", density_kg_m3=1000).number_density_m3() == pytest.approx(1000/.018015*Avogadro, rel=2e-4)
    with pytest.raises(ValueError, match="reference"):
        Medium(removal_cross_section_m2=1e-20).validate()


def test_full_sphere_cross_section_integral():
    from scipy.integrate import quad
    sigma, screening = atomic_cross_section(300000., 7)
    # Stable transformed integration resolves the sharply forward-peaked DCS.
    a = float(screening)
    result = quad(lambda t: 1/(1+t)**2, 0, 1/a, epsabs=1e-11)[0]
    assert result/a == pytest.approx(1/(a*(a+1)), rel=1e-9)
    assert 0 < sigma < 1e-18
    assert atomic_cross_section(100000., 7)[0] > sigma


def test_cell_segment_clipping_and_oblique_path():
    region = ResolvedMedium("cell", "cell", 0., 2., Medium(), 1.)
    start = np.array([[0, 0, -.001], [-.002, 0, .001], [.002, 0, -.001]])
    end = np.array([[0, 0, .003], [.002, 0, .001], [.002, 0, .003]])
    lo, hi = segment_fraction(region, start, end)
    np.testing.assert_allclose((hi-lo)*np.linalg.norm(end-start, axis=1), [.002, .002, 0])


def test_cell_replaces_ambient_and_solid_is_excluded():
    ambient = ResolvedMedium("specimen", "ambient", 0, 4, Medium(pressure_mbar=1e-10))
    cell = ResolvedMedium("specimen_cell", "cell", 1, 3, Medium(pressure_mbar=1e-10), .5)
    run = MediumTransport([ambient, cell], 2, 5)
    run.solid_specimen = SimpleNamespace(z_mm=2, thickness_nm=1e6, size_x_nm=2e6,
                                        size_y_nm=2e6, centre_x_nm=0, centre_y_nm=0, envelope_shape="disk")
    start = np.array([[0., 0., 0.], [.001, 0., 0.]])
    end = start+np.array([0, 0, .004])
    run.advance(start, end, np.tile([0., 0., 1.], (2, 1)), 300000.)
    np.testing.assert_allclose(run.path_m["specimen_cell"], [.001, 0])
    np.testing.assert_allclose(run.path_m["specimen"], [.002, .003])


def test_seeded_scattering_matches_path_attenuation_without_double_loss():
    n = 12000
    gas = Medium(pressure_mbar=1.)
    mu = float(medium_coefficients(gas, np.array([300000.]))[0][0])
    step = .01/mu
    region = ResolvedMedium("gas", "gas", 0, 1e8, gas)
    outputs = []
    for _ in range(2):
        run = MediumTransport([region], n, 42)
        direction = np.tile([0., 0., 1.], (n, 1))
        point = np.zeros((n, 3))
        for j in range(100):
            end = point+np.array([0, 0, step])
            direction = run.advance(point, end, direction, 300000.)
            point = end
        assert np.all(run.alive)  # Elastic scattering redistributes, never absorbs.
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1, atol=2e-15)
        assert np.mean(run.events["gas"] == 0) == pytest.approx(np.exp(-1), abs=.016)
        assert np.mean(run.events["gas"]) == pytest.approx(1., abs=.035)
        np.testing.assert_allclose(run.tau["gas"], 1., atol=1e-13)
        outputs.append(direction)
    np.testing.assert_array_equal(*outputs)


def test_removal_stops_at_sampled_path_coordinate():
    gas = Medium(pressure_mbar=1e-6, removal_cross_section_m2=1e-12, removal_reference="synthetic test fixture")
    n = 20000
    mu = gas.number_density_m3()*gas.removal_cross_section_m2
    distance = 1/mu
    run = MediumTransport([ResolvedMedium("gas", "gas", 0, 1e9, gas)], n, 4)
    start = np.zeros((n, 3))
    end = np.tile([0., 0., distance], (n, 1))
    run.advance(start, end, np.tile([0., 0., 1.], (n, 1)), 300000.)
    assert np.mean(run.alive) == pytest.approx(np.exp(-1), abs=.015)
    assert np.all(run.blocked_z[~run.alive] < distance*1000)
    assert np.all(run.path_m["gas"][~run.alive] < distance)


def test_map_edits_change_source_and_calculation_identity(instrument):
    from temsim.calculation_cache import calculation_signatures
    state = type(instrument).from_dict(instrument.to_dict())
    bind_gun_environment(state)
    key = state.electron_gun._cache_key(9)
    before = calculation_signatures(state)
    state.vacuum_map.regions[0].medium.pressure_mbar *= 10
    bind_gun_environment(state)
    assert state.electron_gun._cache_key(9) != key
    assert calculation_signatures(state)["incident"] != before["incident"]
    state.vacuum_map.cell.inserted = True
    state.vacuum_map.cell.medium.phase = "liquid"
    state.vacuum_map.cell.medium.formula = "H2O"
    nodes = medium_grid_nodes(state, state.sample.z_mm-.01, state.sample.z_mm+.01)
    assert len(nodes) >= 16


def test_optical_propagation_consumes_medium_and_respects_stop(instrument):
    from temsim.physics.core import propagate
    state = type(instrument).from_dict(instrument.to_dict())
    state.step_mm = .02
    # A short empty region between gun exit and the first column aperture.
    for lens in state.lenses:
        lens.enabled = False
    state.vacuum_map.regions[2].medium = Medium(pressure_mbar=1e-6,
        removal_cross_section_m2=1e-12, removal_reference="synthetic fixture")
    media = []
    n = 300
    z, x, tx, y, ty = propagate(state, 460, 461, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                               particle_medium=True, medium_output=media)
    assert media and np.count_nonzero(media[0].alive) < n
    assert np.all(np.isfinite(x))
    assert np.all(media[0].path_m["column"] <= .001+1e-12)
    # Optical transfer fixtures remain deterministic and have no random loss.
    vacuum = []
    propagate(state, 460, 461, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), medium_output=vacuum)
    assert not vacuum


def test_scattered_particles_propagate_to_downstream_position(instrument):
    from temsim.physics.core import propagate
    state = type(instrument).from_dict(instrument.to_dict())
    for lens in state.lenses:
        lens.enabled = False
    for s in state.stigmators:
        s.enabled = False
    state.step_mm = .05
    state.vacuum_map.regions[2].medium.pressure_mbar = 10.
    media = []
    count = 600
    zero = np.zeros(count)
    _, x, tx, y, ty = propagate(state, 460., 470., zero, zero, zero, zero,
                                particle_medium=True, medium_output=media)
    assert np.sum(media[0].events["column"]) > 20
    assert np.count_nonzero(np.hypot(x[-1], y[-1]) > 1e-9) > 15
    assert np.count_nonzero(np.hypot(tx[-1], ty[-1]) > 1e-5) > 15


def test_operating_profile_contains_full_map(instrument, tmp_path):
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.model import State
    path = tmp_path/"profile.toml"
    save_profile(path, instrument, AssemblyCatalog().default_selection())
    selection, values = read_profile(path)
    restored = State.from_dict(instrument.to_dict())
    apply_profile_values(restored, values)
    assert restored.vacuum_map.to_dict() == instrument.vacuum_map.to_dict()


def test_all_catalog_selections_keep_contiguous_map(instrument):
    from itertools import product
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    catalog = AssemblyCatalog()
    state = type(instrument).from_dict(instrument.to_dict())
    checked = 0
    for gun, column, recording, blanker in product(catalog.guns, catalog.columns,
                                                   catalog.recording_systems, catalog.beam_blankers):
        selection = AssemblySelection(gun.name, column.name, recording.name, blanker.name)
        catalog.apply(state, selection)
        rows = resolve_regions(state)
        assert all(a.end_z_mm == pytest.approx(b.start_z_mm) for a, b in zip(rows, rows[1:]))
        assert rows[1].end_z_mm == state.electron_gun.exit_plane_z_mm
        checked += 1
    assert checked >= 30


def test_medium_path_stops_at_camera_not_end_of_trace(instrument):
    from temsim.physics.core import propagate
    state = type(instrument).from_dict(instrument.to_dict())
    state.camera.inserted = True
    state.step_mm = .02
    media = []
    zero = np.zeros(3)
    zcam = state.camera.z_mm
    angles = np.array([0., .001, 0.])
    z, x, _, _, _ = propagate(state, zcam-.3, zcam+.3, zero, angles, zero, zero,
                              particle_medium=True, medium_output=media)
    np.testing.assert_allclose(media[0].path_m["projection"], .0003*np.sqrt(1+angles**2), atol=2e-12)
    np.testing.assert_allclose(media[0].blocked_z, zcam, atol=1e-9)
    assert not np.any(media[0].alive)
    camera_row = np.flatnonzero(z == zcam)[0]
    assert x[-1, 1]-x[camera_row, 1] > 1e-8  # Diagnostic optical continuation, no live gas path.
