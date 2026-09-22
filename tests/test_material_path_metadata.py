"""Material participation belongs to every physical history, not display paths."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_specimen_time_of_flight import _field_free_state, _downstream_fixture
from temsim.specimen import elastic_transport as elastic
from temsim.specimen import downstream_transport as downstream


def test_all_5000_histories_retain_material_path_despite_display_limit(monkeypatch):
    state = _field_free_state()
    state.sample.thickness_nm = 5.
    state.sample.size_x_nm = state.sample.size_y_nm = 100.
    monkeypatch.setattr(elastic, "elastic_scattering_rates_nm_inverse", lambda *_: ((14, 1e-30),))
    rays = tuple(elastic.IncidentElectronRay(index, (0. if index % 2 else 1000., 0.),
        (0., 0., 1.), 200_000., 1/5000) for index in range(5000))
    result = elastic.simulate_elastic_point_transport(state, incident_rays=rays)
    terminal = result.terminal_electrons
    assert len(result.trajectories) == 256
    assert terminal.material_path_nm.shape == (5000,)
    assert terminal.material_path_nm.dtype == np.float64
    assert not terminal.material_path_nm.flags.writeable
    np.testing.assert_array_equal(terminal.material_path_nm[::2], 0.)
    np.testing.assert_allclose(terminal.material_path_nm[1::2], 5., atol=1e-4)
    np.testing.assert_array_equal(terminal.source_ray_index, np.arange(5000))


def test_missed_material_does_not_acquire_nominal_energy_loss_or_absorption(monkeypatch):
    state, simulation, transport = _downstream_fixture(monkeypatch)
    terminal = replace(transport.terminal_electrons, material_path_nm=np.array((0., 10., 0.)))
    transport = replace(transport, terminal_electrons=terminal)
    distribution = SimpleNamespace(channels=(
        SimpleNamespace(key="real_zero_loss", probability=.7, characteristic_angle_mrad=0., energy_loss_ev=0.),
        SimpleNamespace(key="real_plasmon", probability=.2, characteristic_angle_mrad=2., energy_loss_ev=20.)),
        tracked_probability=.9, absorbed_probability=.1)
    result = downstream.build_geometric_specimen_exit(state, simulation, transport, distribution)
    misses = [b for b in result.branches if b.interaction_kind == "vacuum"]
    assert len(misses) == 1 and misses[0].weight == pytest.approx(2/3)
    np.testing.assert_array_equal(misses[0].energy_offset_ev, 0.)
    assert np.all(np.isfinite(misses[0].flight_time_s[-1]))
    losses = [b for b in result.branches if np.any(b.energy_offset_ev < 0.)]
    assert sum(b.weight for b in losses) == pytest.approx(.2/3)
    assert result.metrics["inelastic_absorbed_source_probability"] == pytest.approx(.1/3)
    assert result.metrics["source_probability_conserved"]
    assert result.metrics["inelastic_geometry_scope"] == "executed_material_paths"


def test_current_terminal_geometry_requires_executed_material_paths(monkeypatch):
    state, simulation, transport = _downstream_fixture(monkeypatch)
    result = downstream.build_geometric_specimen_exit(state, simulation, transport)
    assert result.metrics["inelastic_geometry_scope"] == "executed_material_paths"
    assert result.metrics["inelastic_material_path_known"]
    missing = replace(transport, terminal_electrons=replace(
        transport.terminal_electrons, material_path_nm=None))
    with pytest.raises(ValueError, match="requires executed material_path_nm"):
        downstream.build_geometric_specimen_exit(state, simulation, missing)
    with pytest.raises(ValueError, match="material paths"):
        replace(transport.terminal_electrons, material_path_nm=np.array((0., -1., 0.)))
