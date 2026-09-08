"""Point acquisition coordinates must not depend on solver invocation order."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import incident_rays_from_simulation
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import SpecimenInteractionRequest


@pytest.fixture(scope="module")
def detector_geometry():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    return EDSDetectorArrayGeometry.from_part_data(
        assembly.part(EDS_DETECTOR_SYSTEM).data
    )


@pytest.fixture
def point_fixture():
    state = default_state()
    state.sample.size_x_nm = state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 5.0
    state.sample.envelope_shape = "disk"
    state.sample.scan_origin_x_nm = 2.0
    state.sample.scan_origin_y_nm = 1.0
    state.sample.eds_poisson_enabled = False
    # A tiny synthetic bundle, not a column or multislice calculation: the
    # unshifted beam misses the specimen, while the requested point lies in it.
    count = 9
    branch = SimpleNamespace(
        alive=np.ones(count, dtype=bool),
        x=(20.0 + np.linspace(-0.5, 0.5, count))[None, :] * 1.0e-9,
        y=np.zeros((1, count)),
        tx=np.zeros((1, count)),
        ty=np.zeros((1, count)),
        energy_offset_ev=np.zeros(count),
        ray_weight=np.full(count, 1.0 / count),
    )
    return state, SimpleNamespace(incident=branch)


@pytest.mark.parametrize(
    ("x_nm", "y_nm", "target"),
    ((None, None, (2.0, 1.0)), (-1.0, -2.0, (-1.0, -2.0)),
     (-1.0, None, (-1.0, 1.0))),
)
def test_eds_matches_elastic_then_eds_at_default_and_explicit_points(
    detector_geometry, point_fixture, x_nm, y_nm, target,
):
    state, simulation = point_fixture
    request = SpecimenInteractionRequest.eds_point(x_nm=x_nm, y_nm=y_nm)
    direct = run_specimen_interactions(
        state, simulation, request, detector_geometry=detector_geometry,
    )
    elastic = run_specimen_interactions(
        state, simulation,
        SpecimenInteractionRequest.elastic_point(x_nm=x_nm, y_nm=y_nm),
    )
    reused = run_specimen_interactions(
        state, simulation, request, detector_geometry=detector_geometry,
        existing_result=elastic,
    )

    assert direct.eds_spectrum.total_expected_counts > 0.0
    np.testing.assert_array_equal(
        reused.eds_spectrum.expected_counts, direct.eds_spectrum.expected_counts
    )
    assert reused.elastic_transport is elastic.elastic_transport
    assert reused.elastic_transport.eds_tracks == direct.elastic_transport.eds_tracks
    assert "elastic_transport" in reused.metrics["reused_observables"]
    assert (reused.request.point_x_nm, reused.request.point_y_nm) == target
    assert elastic.incident_bundle.original_centroid_nm == pytest.approx((20.0, 0.0))
    assert elastic.incident_bundle.target_centroid_nm == pytest.approx(target)


def test_explicit_scan_origin_reuses_default_point_result(
    detector_geometry, point_fixture,
):
    state, simulation = point_fixture
    first = run_specimen_interactions(
        state, simulation, SpecimenInteractionRequest.eds_point(),
        detector_geometry=detector_geometry,
    )
    # No detector is supplied: a recomputation would fail, so this verifies
    # that omitted and explicit coordinates share a canonical cache identity.
    second = run_specimen_interactions(
        state, simulation,
        SpecimenInteractionRequest.eds_point(x_nm=2.0, y_nm=1.0),
        existing_result=first,
    )
    assert second.eds_spectrum is first.eds_spectrum
    assert second.metrics["calculated_observables_this_call"] == ()


def test_changed_default_scan_origin_recomputes_material_paths(
    detector_geometry, point_fixture,
):
    state, simulation = point_fixture
    first = run_specimen_interactions(
        state, simulation, SpecimenInteractionRequest.eds_point(),
        detector_geometry=detector_geometry,
    )
    state.sample.scan_origin_x_nm = 20.0
    second = run_specimen_interactions(
        state, simulation, SpecimenInteractionRequest.eds_point(),
        detector_geometry=detector_geometry, existing_result=first,
    )
    assert first.request.point_x_nm == 2.0
    assert first.eds_spectrum.total_expected_counts > 0.0
    assert second.request.point_x_nm == 20.0
    assert not second.elastic_transport.eds_tracks
    assert second.eds_spectrum.total_expected_counts == 0.0
    assert second.elastic_transport is not first.elastic_transport


def test_legacy_unresolved_point_is_not_reinterpreted_as_current_scan_origin(
    detector_geometry, point_fixture,
):
    state, simulation = point_fixture
    missed = run_specimen_interactions(
        state, simulation,
        SpecimenInteractionRequest.elastic_point(x_nm=20.0, y_nm=0.0),
    )
    assert not missed.elastic_transport.eds_tracks
    legacy = replace(missed, request=SpecimenInteractionRequest.elastic_point())
    corrected = run_specimen_interactions(
        state, simulation, SpecimenInteractionRequest.eds_point(),
        detector_geometry=detector_geometry, existing_result=legacy,
    )
    assert corrected.eds_spectrum.total_expected_counts > 0.0
    assert corrected.elastic_transport is not legacy.elastic_transport
    assert "elastic_transport" not in corrected.metrics["reused_observables"]


def test_raw_incident_boundary_none_still_preserves_physical_centroid(point_fixture):
    state, simulation = point_fixture
    branch = simulation.incident
    branch.z = np.array([state.sample.z_mm - 1.0, state.sample.z_mm])
    branch.blocked_z = np.full(branch.alive.size, np.nan)
    for name in ("x", "y", "tx", "ty"):
        setattr(branch, name, np.repeat(getattr(branch, name), 2, axis=0))

    boundary = incident_rays_from_simulation(
        state, simulation, boundary_z_mm=state.sample.z_mm - 0.5,
    )
    assert boundary.original_centroid_nm == pytest.approx((20.0, 0.0))
    assert boundary.target_centroid_nm == pytest.approx((20.0, 0.0))
    assert boundary.boundary_z_mm == state.sample.z_mm - 0.5
    assert np.mean([ray.position_xy_nm[0] for ray in boundary.rays]) == pytest.approx(20.0)
