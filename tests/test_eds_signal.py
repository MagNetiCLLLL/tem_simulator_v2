import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_atomic import (
    bote_element_data,
    bote_ionisation_cross_section_cm2,
    load_bote_salvat_coefficients,
)
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import (
    EDSMaterial,
    ElectronTrackSegment,
    elemental_material,
    material_from_sample,
    point_track_segments,
    simulate_eds_point,
    simulate_eds_tracks,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import (
    runtime_targets,
    validate_runtime_assignment,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _sample_plane_simulation(ray_count=8):
    x = np.zeros((1, ray_count), dtype=float)
    y = np.zeros((1, ray_count), dtype=float)
    tx = np.linspace(-1.0e-3, 1.0e-3, ray_count)[None, :]
    ty = np.linspace(0.5e-3, -0.5e-3, ray_count)[None, :]
    return SimpleNamespace(
        incident=SimpleNamespace(
            alive=np.ones(ray_count, dtype=bool),
            x=x,
            y=y,
            tx=tx,
            ty=ty,
            energy_offset_ev=np.linspace(-0.2, 0.2, ray_count),
            ray_weight=np.full(ray_count, 1.0 / ray_count),
        )
    )


@pytest.fixture(scope="module")
def installed_geometry():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    return EDSDetectorArrayGeometry.from_part_data(
        assembly.part(EDS_DETECTOR_SYSTEM).data
    )


def test_bote_salvat_table_covers_k_l_m_for_z_1_to_99():
    data = load_bote_salvat_coefficients()

    assert set(data) == set(range(1, 100))
    assert bote_element_data(14).edge_ev == pytest.approx(
        (1828.5, 151.529, 108.647, 107.944)
    )
    assert bote_element_data(79).subshell_names == (
        "K",
        "L1",
        "L2",
        "L3",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    )


def test_eds_material_follows_the_mode_owned_structure_source():
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "dormant-missing.cif"

    material = material_from_sample(state)

    assert material is not None
    assert material.key == "specimen:si_110"

    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = ""
    assert material_from_sample(state) is None

    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "vacuum"
    assert material_from_sample(state) is None


def test_straight_eds_path_respects_circular_sample_edge():
    state = default_state()
    centre_tracks = point_track_segments(state, x_nm=0.0, y_nm=0.0)
    corner_tracks = point_track_segments(
        state,
        x_nm=1_400_000.0,
        y_nm=1_400_000.0,
    )

    assert [track.source_key for track in centre_tracks] == ["sample"]
    assert corner_tracks == ()


def test_bote_salvat_threshold_units_and_reference_values():
    edge = bote_element_data(14).edge_ev[0]

    assert bote_ionisation_cross_section_cm2(14, "K", edge) == 0.0
    assert bote_ionisation_cross_section_cm2(
        14, "K", math.nextafter(edge, math.inf)
    ) >= 0.0
    assert bote_ionisation_cross_section_cm2(
        14, "K", 200_000.0
    ) == pytest.approx(1.7452830450842684e-21, rel=1.0e-13)
    assert bote_ionisation_cross_section_cm2(
        29, "L3", 200_000.0
    ) == pytest.approx(8.696508772323364e-21, rel=1.0e-13)
    with pytest.raises(ValueError, match="Z=1..99"):
        bote_ionisation_cross_section_cm2(100, "K", 200_000.0)


def test_characteristic_si_counts_conserve_line_histogram(installed_geometry):
    silicon = elemental_material(14, density_g_cm3=2.33)
    track = ElectronTrackSegment(
        "sample",
        silicon,
        path_length_nm=100.0,
        electron_energy_ev=200_000.0,
        emitting_layer_thickness_nm=100.0,
    )

    result = simulate_eds_tracks(
        (track,),
        installed_geometry,
        incident_electrons=1.0e6,
    )

    assert result.metrics["system_name"] == "EDS"
    assert result.metrics["elastic_trajectory_generation"] is False
    assert result.metrics["bremsstrahlung_included"] is False
    assert result.vacancies
    assert result.metrics["duplicate_shell_ionisation_passes"] == 0
    assert result.metrics["vacancy_contribution_count"] == len(
        result.vacancies
    )
    assert all(
        vacancy.relaxation_yield_sum == pytest.approx(1.0, abs=2.0e-12)
        for vacancy in result.vacancies
    )
    assert (
        result.metrics["total_expected_radiative_relaxations"]
        + result.metrics["total_expected_auger_relaxations"]
        + result.metrics["total_expected_unresolved_relaxations"]
    ) == pytest.approx(result.metrics["total_expected_vacancies"])
    si_l1 = next(
        vacancy for vacancy in result.vacancies if vacancy.subshell == "L1"
    )
    assert si_l1.unresolved_relaxation_yield > 0.0
    assert result.total_expected_counts > 0.0
    assert np.sum(result.expected_counts) == pytest.approx(
        result.total_expected_counts
    )
    ka1 = next(line for line in result.lines if line.transition == "K-L3")
    assert ka1.vacancy_id in {
        vacancy.vacancy_id for vacancy in result.vacancies
    }
    assert ka1.energy_ev == pytest.approx(1740.0)
    assert ka1.expected_detected_counts > 0.0
    assert len(ka1.expected_counts_per_segment) == 6
    assert sum(ka1.expected_counts_per_segment) == pytest.approx(
        ka1.expected_detected_counts
    )


def test_shell_cross_sections_are_evaluated_once_then_reused(
    installed_geometry, monkeypatch
):
    import temsim.detector.eds_signal as eds_signal

    original = eds_signal.bote_ionisation_cross_section_cm2
    calls = []

    def counted_cross_section(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        eds_signal,
        "bote_ionisation_cross_section_cm2",
        counted_cross_section,
    )
    result = simulate_eds_tracks(
        (
            ElectronTrackSegment(
                "sample",
                elemental_material(14, density_g_cm3=2.33),
                path_length_nm=20.0,
                electron_energy_ev=200_000.0,
                source_ray_index=4,
            ),
        ),
        installed_geometry,
        incident_electrons=1000.0,
    )

    assert len(calls) == result.metrics[
        "shell_cross_section_evaluation_count"
    ]
    assert result.metrics["duplicate_shell_ionisation_passes"] == 0
    assert all(vacancy.source_ray_index == 4 for vacancy in result.vacancies)
    assert {line.vacancy_id for line in result.lines} <= {
        vacancy.vacancy_id for vacancy in result.vacancies
    }


def test_counts_scale_with_track_weight_efficiency_and_solid_angle(
    installed_geometry,
):
    material = elemental_material(29)
    base_track = ElectronTrackSegment(
        "support:bar",
        material,
        path_length_nm=10.0,
        electron_energy_ev=300_000.0,
        emitting_layer_thickness_nm=100.0,
    )
    half_track = ElectronTrackSegment(
        "support:bar",
        material,
        path_length_nm=10.0,
        electron_energy_ev=300_000.0,
        electron_weight=0.5,
        emitting_layer_thickness_nm=100.0,
    )

    base = simulate_eds_tracks(
        (base_track,),
        installed_geometry,
        incident_electrons=1.0e5,
        detector_efficiency=1.0,
    )
    half = simulate_eds_tracks(
        (half_track,),
        installed_geometry,
        incident_electrons=1.0e5,
        detector_efficiency=0.5,
    )
    unshadowed = simulate_eds_tracks(
        (base_track,),
        installed_geometry,
        incident_electrons=1.0e5,
        use_analytical_holder_solid_angle=False,
    )

    assert half.total_expected_counts == pytest.approx(
        0.25 * base.total_expected_counts
    )
    assert unshadowed.total_expected_counts / base.total_expected_counts == (
        pytest.approx(4.45 / 4.04)
    )


def test_caller_supplied_elastically_scattered_track_retains_provenance(
    installed_geometry,
):
    track = ElectronTrackSegment(
        "holder",
        elemental_material(79),
        path_length_nm=20.0,
        electron_energy_ev=120_000.0,
        history="elastic_scattered",
    )

    result = simulate_eds_tracks(
        (track,),
        installed_geometry,
        incident_electrons=10_000.0,
    )

    assert result.lines
    assert {line.electron_history for line in result.lines} == {
        "elastic_scattered"
    }
    assert result.metrics["electron_transport_model"] == (
        "caller_supplied_weighted_track_segments"
    )


def test_support_grid_adds_copper_only_when_track_intersects_material():
    state = default_state()
    state.sample.eds_support_material_key = "copper"
    state.sample.eds_support_mesh_key = "square_200"

    opening = point_track_segments(state, x_nm=0.0, y_nm=0.0)
    bar = point_track_segments(state, x_nm=60_000.0, y_nm=0.0)

    assert [segment.source_key for segment in opening] == ["sample"]
    assert [segment.source_key for segment in bar] == [
        "sample",
        "support:bar",
    ]
    assert bar[-1].material.mass_fractions == ((29, 1.0),)
    assert bar[-1].path_length_nm == pytest.approx(25_000.0)


def test_point_spectrum_uses_generic_name_and_reproducible_poisson(
    installed_geometry,
):
    state = default_state()
    state.sample.eds_poisson_enabled = True
    state.sample.eds_poisson_seed = 123
    state.sample.eds_energy_resolution_fwhm_ev = 125.0

    first = simulate_eds_point(
        state,
        installed_geometry,
        simulation=_sample_plane_simulation(),
        incident_electrons=1.0e6,
    )
    second = simulate_eds_point(
        state,
        installed_geometry,
        simulation=_sample_plane_simulation(),
        incident_electrons=1.0e6,
    )

    assert first.metrics["system_name"] == "EDS"
    assert first.metrics["elastic_trajectory_generation"] is True
    assert first.elastic_transport is not None
    assert first.metrics["incident_electron_reference"] == (
        "explicit source-electron argument"
    )
    assert np.array_equal(first.sampled_counts, second.sampled_counts)
    assert first.expected_counts.shape == first.sampled_counts.shape


def test_point_spectrum_retains_explicit_straight_transport_reference(
    installed_geometry,
):
    state = default_state()
    state.sample.eds_transport_mode = "straight_primary"

    result = simulate_eds_point(
        state, installed_geometry, incident_electrons=1.0e4
    )

    assert result.elastic_transport is None
    assert result.metrics["requested_transport_mode"] == "straight_primary"
    assert result.metrics["elastic_trajectory_generation"] is False
    assert {line.electron_history for line in result.lines} == {
        "straight_primary"
    }


def test_mixture_number_density_uses_mass_fraction_not_atomic_fraction():
    mixture = EDSMaterial(
        key="test",
        name="test",
        density_g_cm3=2.0,
        mass_fractions=((6, 0.25), (14, 0.75)),
        provenance="unit test",
    )

    ratio = (
        mixture.atom_number_density_cm3(6)
        / mixture.atom_number_density_cm3(14)
    )
    expected = (0.25 / 12.01) / (0.75 / 28.09)
    assert ratio == pytest.approx(expected)


def test_runtime_rejects_unknown_support_and_nonphysical_detector_values():
    state = default_state()
    target = runtime_targets(state)["sample"]

    with pytest.raises(ValueError, match="support catalog"):
        validate_runtime_assignment(
            target, "eds_support_material_key", "platinum"
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_runtime_assignment(
            target, "eds_detector_efficiency", 1.1
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_runtime_assignment(
            target, "eds_energy_resolution_fwhm_ev", -1.0
        )
    with pytest.raises(ValueError, match="transport_mode"):
        validate_runtime_assignment(target, "eds_transport_mode", "fake")
