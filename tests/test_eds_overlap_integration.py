"""EDS quadrature must retain rare-overlap mass and original electron history."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import (
    _vacancy_emission_origin_mm, simulate_eds_point, simulate_eds_tracks,
)
from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import simulate_elastic_point_transport
from temsim.specimen.interaction_engine import _eds_event_ledger
from temsim.specimen.interaction_types import IncidentElectronRay, IncidentRayBundle


@pytest.fixture(scope="module")
def geometry():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    return EDSDetectorArrayGeometry.from_part_data(assembly.part(EDS_DETECTOR_SYSTEM).data)


@pytest.fixture
def weighted_case(monkeypatch):
    state = default_state()
    state.sample.size_x_nm = state.sample.size_y_nm = 10.
    state.sample.thickness_nm = 5.
    state.sample.eds_support_material_key = "vacuum"
    rays = tuple(IncidentElectronRay(i, (x, 0.), (0., 0., 1.), 300_000., .5)
                 for i, x in enumerate((-50., 50.)))
    bundle = IncidentRayBundle(rays, 4, 2, .5, (0., 0.), (0., 0.),
                               (0., 0.), (300_000., 300_000.))
    original = simulate_elastic_point_transport(state, incident_rays=rays)
    assert not original.eds_tracks
    quadrature_rays = (
        IncidentElectronRay(4, (-2., 0.), (0., 0., 1.), 300_000., 1.e-6),
        IncidentElectronRay(5, (2., 0.), (0., 0., 1.), 300_000., 3.e-6),
    )
    # A controlled integration rule isolates the mass/origin interface. KDE
    # accuracy and eligibility are tested independently against analytic beams.
    plan = SimpleNamespace(
        rays=quadrature_rays, parent_source_ray_indices=(0, 1),
        metrics={"eds_overlap_sampling_status": "active",
                 "eds_overlap_sampling_point_count": 2,
                 "eds_overlap_probability_mass": 4.e-6},
    )
    monkeypatch.setattr("temsim.specimen.overlap_sampling.build_overlap_sampling_plan",
                        lambda *args: plan)
    return state, bundle, original


def test_rare_mass_applied_once_without_modifying_downstream(weighted_case, geometry):
    state, bundle, original = weighted_case
    terminal_before = original.terminal_electrons.weight.copy()
    progress = []
    result = simulate_eds_point(state, geometry, incident_bundle=bundle,
                                elastic_transport=original, incident_electrons=1.e8,
                                progress_callback=lambda done, total, label: progress.append(done / total))
    assert progress == sorted(progress)
    assert result.elastic_transport is original
    np.testing.assert_array_equal(original.terminal_electrons.weight, terminal_before)
    assert not original.eds_tracks
    quad = result.material_quadrature
    assert quad is not None and not hasattr(quad, "terminal_electrons")
    assert quad.parent_source_ray_indices == (0, 1)
    assert quad.quadrature_source_ray_indices == (4, 5)
    assert sum(track.electron_weight for track in quad.eds_tracks) == pytest.approx(4.e-6)
    assert {track.source_ray_index for track in quad.eds_tracks} == {4, 5}
    assert result.metrics["electrons_reaching_sample_plane"] == 5.e7
    assert result.metrics["incident_electrons"] == 5.e7
    assert result.metrics["eds_overlap_material_weight_fraction"] == pytest.approx(4.e-6)
    # Independent algebraic check: normalised paths with overlap-reduced dose
    # must have the same shell-vacancy expectation as absolute paths/full dose.
    reference = simulate_eds_tracks(
        tuple(replace(track, electron_weight=track.electron_weight / 4.e-6)
              for track in quad.eds_tracks),
        geometry, incident_electrons=5.e7 * 4.e-6,
    )
    assert len(result.vacancies) > 0
    np.testing.assert_allclose(
        [row.expected_vacancies for row in result.vacancies],
        [row.expected_vacancies for row in reference.vacancies], rtol=2.e-12,
    )


def test_each_quadrature_photon_has_its_own_origin_and_no_column_identity(weighted_case, geometry):
    state, bundle, original = weighted_case
    result = simulate_eds_point(state, geometry, incident_bundle=bundle,
                                elastic_transport=original, incident_electrons=1.e8)
    quad = result.material_quadrature
    for vacancy in result.vacancies:
        origin, status = _vacancy_emission_origin_mm(
            state, vacancy, quad, fallback_xy_nm=(999., 999.),
        )
        assert "material_flight" in status
        expected_x = -2. if vacancy.source_ray_index == 4 else 2.
        assert origin[0] * 1.e6 == pytest.approx(expected_x, abs=1.e-5)
        assert origin[1] * 1.e6 == pytest.approx(0., abs=1.e-5)
        assert abs(origin[2] - state.sample.z_mm) <= 2.501e-6
    events = _eds_event_ledger(result, original)
    assert events
    assert all(event.parent_electron_index is None for event in events)
    vacancies = [event for event in events if str(event.process) == "core_ionisation"]
    assert all("overlap quadrature" in event.provenance for event in vacancies)
    assert sum(event.expected_occurrences_per_source_electron for event in vacancies) == pytest.approx(
        sum(row.expected_vacancies for row in result.vacancies) / 1.e8
    )


def test_empty_plan_preserves_exact_original_eds(monkeypatch, weighted_case, geometry):
    state, bundle, original = weighted_case
    monkeypatch.setattr("temsim.specimen.overlap_sampling.build_overlap_sampling_plan",
                        lambda *args: SimpleNamespace(rays=(), metrics={"eds_overlap_sampling_status": "disabled"}))
    result = simulate_eds_point(state, geometry, incident_bundle=bundle,
                                elastic_transport=original, incident_electrons=1.e8)
    assert result.material_quadrature is None
    assert result.elastic_transport is original
    assert result.total_expected_counts == 0.
    assert not np.any(result.expected_counts)


def test_auxiliary_misses_do_not_become_material_signal(monkeypatch, weighted_case, geometry):
    state, bundle, original = weighted_case
    monkeypatch.setattr("temsim.specimen.overlap_sampling.build_overlap_sampling_plan", lambda *args:
        SimpleNamespace(rays=(IncidentElectronRay(4, (100., 0.), (0., 0., 1.), 300_000., 1.e-6),),
                        parent_source_ray_indices=(0,),
                        metrics={"eds_overlap_sampling_status": "active"}))
    result = simulate_eds_point(state, geometry, incident_bundle=bundle,
                                elastic_transport=original, incident_electrons=1.e8)
    assert result.material_quadrature is not None
    assert result.material_quadrature.quadrature_source_ray_indices == (4,)
    assert result.material_quadrature.parent_source_ray_indices == (0,)
    assert result.metrics["eds_overlap_material_weight_fraction"] == 0.
    assert result.total_expected_counts == 0.
    assert not result.vacancies
