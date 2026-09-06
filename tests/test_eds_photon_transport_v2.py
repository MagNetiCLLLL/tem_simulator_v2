import math
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_photon_transport import (
    AnnularPlanarLayerOccluder,
    EDSPhotonRay,
    FiniteSpecimenOccluder,
    PlanarEDSDetectorSegment,
    trace_eds_photon,
)
from temsim.detector.eds_signal import (
    ElectronTrackSegment,
    elemental_material,
    simulate_eds_tracks,
)
from temsim.optics.column import default_state


class _Material:
    key = "known-test-material"
    density_g_cm3 = 2.0

    @staticmethod
    def mass_attenuation_cm2_g(_energy_ev):
        return 3.0


@pytest.fixture
def geometry():
    return EDSDetectorArrayGeometry(
        system_key="eds",
        segment_count=1,
        azimuth_centers_deg=(0.0,),
        takeoff_angle_deg=32.0,
        minimum_unshadowed_solid_angle_sr=1.0,
        analytical_holder_solid_angle_sr=1.0,
        windowless=True,
    )


def test_sourced_sensor_reports_exact_hit_and_finite_material_path(geometry):
    photon = EDSPhotonRay(
        photon_id="p0",
        origin_mm=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        energy_ev=1740.0,
        statistical_weight=2.0,
        source_key="sample",
        transition="Si Ka",
    )
    detector = PlanarEDSDetectorSegment(
        segment_index=0,
        center_mm=(10.0, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        active_shape="circle",
        active_size_mm=(4.0, 4.0),
        efficiency=0.5,
        provenance="sourced test drawing",
    )
    specimen = FiniteSpecimenOccluder(
        key="sample",
        centre_global_mm=(0.0, 0.0, 0.0),
        size_local_mm=(1.0, 1.0, 1.0),
        rotation_local_to_global=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        envelope_shape="rectangle",
        material=_Material(),
    )

    result = trace_eds_photon(
        photon,
        geometry,
        detector_surfaces=(detector,),
        occluders=(specimen,),
    )

    assert result.detector_segment == 0
    assert result.detector_hit_mm == pytest.approx((10.0, 0.0, 0.0))
    assert result.material_intervals[0].path_length_mm == pytest.approx(0.5)
    assert result.transmission == pytest.approx(math.exp(-0.3))
    assert result.detected_weight == pytest.approx(math.exp(-0.3))


def test_holder_hard_shadow_stops_a_photon_before_detector(geometry):
    photon = EDSPhotonRay(
        photon_id="p1",
        origin_mm=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        energy_ev=8040.0,
        statistical_weight=1.0,
        source_key="support",
        transition="Cu Ka",
    )
    detector = PlanarEDSDetectorSegment(
        segment_index=0,
        center_mm=(0.0, 0.0, 10.0),
        normal=(0.0, 0.0, -1.0),
        u_axis=(1.0, 0.0, 0.0),
        active_shape="circle",
        active_size_mm=(4.0, 4.0),
        provenance="sourced test drawing",
    )
    holder = AnnularPlanarLayerOccluder(
        key="holder",
        z_start_mm=2.0,
        z_end_mm=3.0,
        outer_radius_mm=2.0,
        hard_shadow=True,
    )

    result = trace_eds_photon(
        photon,
        geometry,
        detector_surfaces=(detector,),
        occluders=(holder,),
    )

    assert result.transmission == 0.0
    assert result.detected_weight == 0.0
    assert result.terminal_status == "blocked_by:holder"


def _silicon_track():
    return ElectronTrackSegment(
        "sample",
        elemental_material(14, density_g_cm3=2.33),
        path_length_nm=10.0,
        electron_energy_ev=200_000.0,
        emitting_layer_thickness_nm=10.0,
    )


def test_main_spectrum_aggregate_quadrature_preserves_unshadowed_weight(
    geometry,
):
    state = default_state()
    result = simulate_eds_tracks(
        (_silicon_track(),),
        geometry,
        incident_electrons=1.0e6,
        detector_efficiency=0.4,
        state=state,
        photon_include_specimen=False,
        photon_include_support=False,
    )

    expected = sum(
        line.expected_emitted_photons for line in result.lines
    ) * geometry.analytical_holder_solid_angle_sr / (4.0 * math.pi) * 0.4
    assert result.total_expected_counts == pytest.approx(expected)
    assert result.metrics["photon_transport_applied_to_main_spectrum"]
    assert result.metrics["photon_transport_model"] == (
        "EDS photon transport v2"
    )
    assert result.photon_transport is not None
    assert result.photon_transport.metrics["total_quadrature_weight"] == (
        pytest.approx(expected)
    )
    assert result.metrics["detector_efficiency_model"] == (
        "global_absolute_qe"
    )
    assert result.metrics["global_detector_efficiency_application_count"] == 1
    assert result.metrics["surface_relative_efficiency_application_count"] == 0
    assert result.metrics["detector_efficiency_application_count"] == 1


def test_support_shadow_changes_main_spectrum_before_poisson(
    geometry, monkeypatch
):
    state = default_state()
    sample_z = float(state.sample.z_mm)
    support_shadow = AnnularPlanarLayerOccluder(
        key="support:test-shadow",
        z_start_mm=sample_z - 2.0,
        z_end_mm=sample_z - 1.0,
        outer_radius_mm=100.0,
        hard_shadow=True,
        provenance="explicit regression-test support shadow",
    )
    captured = []

    class _DeterministicPoisson:
        def poisson(self, expected):
            captured.append(np.asarray(expected, dtype=float).copy())
            return np.rint(expected).astype(np.int64)

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda _seed: _DeterministicPoisson(),
    )
    result = simulate_eds_tracks(
        (_silicon_track(),),
        geometry,
        incident_electrons=1.0e6,
        state=state,
        photon_holder_occluders=(support_shadow,),
        photon_include_specimen=False,
        photon_include_support=False,
        poisson_enabled=True,
        poisson_seed=7,
    )

    assert result.total_expected_counts == 0.0
    assert captured and np.array_equal(captured[0], result.expected_counts)
    assert result.metrics["poisson_sampling_stage"] == (
        "after_final_expected_spectrum"
    )
    assert result.photon_transport.metrics[
        "blocked_by_component_counts"
    ]["support:test-shadow"] > 0


def test_changed_objective_pole_shape_changes_main_spectrum_counts():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    detector_geometry = EDSDetectorArrayGeometry.from_part_data(
        assembly.part(EDS_DETECTOR_SYSTEM).data
    )
    clear = simulate_eds_tracks(
        (_silicon_track(),),
        detector_geometry,
        incident_electrons=1.0e6,
        state=state,
        photon_include_specimen=False,
        photon_include_support=False,
    )
    changed_parts = tuple(
        replace(
            part,
            data={
                **dict(part.data),
                "mechanical_tip_diameter_mm": 12.0,
            },
        )
        if part.key == "objective_upper_pole"
        else part
        for part in assembly.parts
    )
    state._resolved_assembly = replace(assembly, parts=changed_parts)
    shadowed = simulate_eds_tracks(
        (_silicon_track(),),
        detector_geometry,
        incident_electrons=1.0e6,
        state=state,
        photon_include_specimen=False,
        photon_include_support=False,
    )

    assert clear.total_expected_counts > 0.0
    assert shadowed.total_expected_counts < clear.total_expected_counts
    assert shadowed.photon_transport.metrics[
        "blocked_by_component_counts"
    ]["objective_upper_pole"] > 0


def test_main_spectrum_uses_sourced_detector_face_hits(geometry):
    state = default_state()
    detector = PlanarEDSDetectorSegment(
        segment_index=0,
        center_mm=(10.0, 0.0, float(state.sample.z_mm)),
        normal=(-1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        active_shape="rectangle",
        active_size_mm=(2.0, 2.0),
        efficiency=0.5,
        provenance="sourced regression-test detector face",
    )
    result = simulate_eds_tracks(
        (_silicon_track(),),
        geometry,
        incident_electrons=1.0e6,
        detector_efficiency=0.4,
        state=state,
        photon_detector_surfaces=(detector,),
        photon_include_specimen=False,
        photon_include_support=False,
        photon_quadrature_order=2,
    )

    assert result.total_expected_counts > 0.0
    assert result.metrics["photon_transport_geometry_complete"]
    assert result.metrics["photon_transport_mode"] == (
        "sourced_detector_faces"
    )
    assert result.photon_transport.paths
    assert all(
        path.detector_hit_mm is not None
        for path in result.photon_transport.paths
    )


def test_sourced_face_response_multiplies_global_qe_exactly_once(geometry):
    state = default_state()
    unity_surface = PlanarEDSDetectorSegment(
        segment_index=0,
        center_mm=(10.0, 0.0, float(state.sample.z_mm)),
        normal=(-1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        active_shape="rectangle",
        active_size_mm=(2.0, 2.0),
        efficiency=1.0,
        provenance="sourced regression-test detector face",
    )
    relative_surface = replace(unity_surface, efficiency=0.5)

    reference = simulate_eds_tracks(
        (_silicon_track(),),
        geometry,
        incident_electrons=1.0e6,
        detector_efficiency=1.0,
        state=state,
        photon_detector_surfaces=(unity_surface,),
        photon_include_specimen=False,
        photon_include_support=False,
        photon_quadrature_order=2,
    )
    scaled = simulate_eds_tracks(
        (_silicon_track(),),
        geometry,
        incident_electrons=1.0e6,
        detector_efficiency=0.4,
        state=state,
        photon_detector_surfaces=(relative_surface,),
        photon_include_specimen=False,
        photon_include_support=False,
        photon_quadrature_order=2,
    )

    assert scaled.total_expected_counts == pytest.approx(
        reference.total_expected_counts * 0.4 * 0.5,
        rel=2.0e-13,
    )
    assert scaled.metrics["detector_efficiency_model"] == (
        "global_absolute_qe_x_surface_relative_response"
    )
    assert scaled.metrics["global_detector_efficiency_application_count"] == 1
    assert scaled.metrics["surface_relative_efficiency_application_count"] == 1
    assert scaled.metrics["detector_efficiency_application_count"] == 2
    assert scaled.photon_transport.metrics[
        "surface_relative_response_applied"
    ]
    assert scaled.photon_transport.metrics[
        "surface_relative_response_values"
    ] == pytest.approx((0.5,))
