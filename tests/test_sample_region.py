import math
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.specimen.sample_region as sample_region_module
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.optics.column import default_state
from temsim.specimen.downstream_transport import GeometricSpecimenExit
from temsim.specimen.sample_region import (
    SampleRegionElectronPath,
    SampleRegionResult,
    _angular_acceptance,
    _electron_paths,
    _isotropic_directions,
    bind_sample_region_downstream,
    reproject_sample_region_downstream,
    simulate_sample_region,
)


@pytest.fixture
def detector_geometry():
    return EDSDetectorArrayGeometry(
        system_key="eds",
        segment_count=6,
        azimuth_centers_deg=(0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
        takeoff_angle_deg=32.06,
        minimum_unshadowed_solid_angle_sr=4.45,
        analytical_holder_solid_angle_sr=4.04,
        windowless=True,
    )


def test_isotropic_photon_sampler_is_uniform_in_solid_angle():
    directions = _isotropic_directions(np.random.default_rng(71), 200_000)

    assert np.linalg.norm(directions, axis=1) == pytest.approx(
        np.ones(len(directions)), abs=2.0e-15
    )
    assert float(np.mean(directions[:, 2])) == pytest.approx(0.0, abs=0.004)
    assert float(np.mean(directions[:, 2] ** 2)) == pytest.approx(
        1.0 / 3.0, abs=0.004
    )


def test_angular_acceptance_matches_configured_aggregate_solid_angle(
    detector_geometry,
):
    directions = _isotropic_directions(np.random.default_rng(13), 250_000)
    accepted = np.asarray(
        [
            _angular_acceptance(
                direction,
                detector_geometry,
                detector_geometry.analytical_holder_solid_angle_sr,
            )[0]
            for direction in directions
        ],
        dtype=bool,
    )

    expected = detector_geometry.analytical_holder_solid_angle_sr / (
        4.0 * math.pi
    )
    assert float(np.mean(accepted)) == pytest.approx(expected, abs=0.0025)


def test_secondary_marker_can_be_explicitly_zero_weight_and_local_only():
    marker = SampleRegionElectronPath(
        positions_mm=np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0e-4, 1.0))),
        kind="secondary_candidate",
        weight=0.0,
        kinetic_energy_ev=None,
        provenance="qualitative test marker",
        downstream_eligible=False,
    )

    assert marker.weight == 0.0
    assert marker.kinetic_energy_ev is None
    assert not marker.downstream_eligible


def test_boundary_input_path_ends_at_the_centred_specimen_entrance_face():
    state = default_state()
    sample_z_mm = float(state.sample.z_mm)
    entry_z_mm = sample_z_mm - 0.05
    slope_x = 0.01
    sample_x_m = 12.0e-9
    entry_x_m = sample_x_m + (entry_z_mm - sample_z_mm) * 1.0e-3 * slope_x
    simulation = SimpleNamespace(
        incident=SimpleNamespace(
            z=np.asarray((entry_z_mm, sample_z_mm)),
            alive=np.asarray((True,)),
            blocked_z=np.asarray((np.nan,)),
            x=np.asarray(((entry_x_m,), (sample_x_m,))),
            y=np.zeros((2, 1)),
            tx=np.full((2, 1), slope_x),
            ty=np.zeros((2, 1)),
            energy_offset_ev=np.zeros(1),
            ray_weight=np.ones(1),
        )
    )
    elastic = SimpleNamespace(trajectories=(), material_flights=())

    paths, _entry_bundle, _sample_bundle = _electron_paths(
        state,
        simulation,
        elastic,
        entry_z_mm=entry_z_mm,
        secondary_count=0,
        rng=np.random.default_rng(1),
    )

    endpoint = paths[0].positions_mm[-1]
    expected_top_z_mm = sample_z_mm - 0.5 * state.sample.thickness_nm * 1.0e-6
    assert endpoint[2] == pytest.approx(expected_top_z_mm)
    assert endpoint[0] == pytest.approx(
        -0.5 * state.sample.thickness_nm * slope_x * 1.0e-6
    )


def test_downstream_reprojection_shares_local_paths_and_rebinds_observables(
    monkeypatch,
):
    state = default_state()
    electron_paths = ("cached-electron",)
    photon_paths = ("cached-photon",)
    old = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=electron_paths,
        photon_paths=photon_paths,
        downstream_branches=("old-downstream",),
        spectrum="old-spectrum",
        interactions="old-interactions",
        metrics={
            "kept_local_metric": 7,
            "downstream_forward_weight": 0.25,
            "model": "old-model",
        },
    )
    elastic = object()
    spectrum = SimpleNamespace(elastic_transport=elastic)
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=spectrum,
        inelastic_distribution="current-inelastic",
        metrics={
            "dependency_signatures": {
                "sample_downstream": "downstream-current",
            }
        },
    )
    calls = []
    new_downstream_branch = SimpleNamespace(weight=0.75)

    def fake_downstream(
        passed_state,
        passed_simulation,
        passed_elastic,
        passed_inelastic,
        *,
        save_z_mm,
        dependency_signature,
    ):
        calls.append((
            passed_state,
            passed_simulation,
            passed_elastic,
            passed_inelastic,
            save_z_mm,
            dependency_signature,
        ))
        return GeometricSpecimenExit(
            (new_downstream_branch,),
            {
                "model": "new-model",
                "downstream_forward_weight": 0.75,
                "tracked_downstream_source_probability": 0.75,
                "inelastic_absorbed_source_probability": 0.05,
            },
            dependency_signature=dependency_signature,
        )

    monkeypatch.setattr(
        sample_region_module,
        "build_geometric_specimen_exit",
        fake_downstream,
    )
    simulation = object()

    updated = reproject_sample_region_downstream(
        state,
        simulation,
        old,
        interactions,
        wave_imaging=object(),
        dependency_signatures={
            "sample_region": "local-current",
            "sample_downstream": "downstream-current",
        },
    )

    assert calls == [(
        state,
        simulation,
        elastic,
        "current-inelastic",
        (2.0,),
        "downstream-current",
    )]
    assert updated is not old
    assert updated.electron_paths is electron_paths
    assert updated.photon_paths is photon_paths
    assert updated.downstream_branches == (new_downstream_branch,)
    assert updated.spectrum is spectrum
    assert updated.interactions is interactions
    assert updated.specimen_exit.dependency_signature == "downstream-current"
    assert updated.metrics["kept_local_metric"] == 7
    assert updated.metrics["model"] == "new-model"
    assert updated.metrics["downstream_forward_weight"] == pytest.approx(0.75)
    assert updated.metrics["sample_region_signature"] == "local-current"
    assert updated.metrics["sample_downstream_signature"] == (
        "downstream-current"
    )
    assert updated.metrics["channeling_model"].startswith("coherent wave")
    assert old.downstream_branches == ("old-downstream",)
    assert old.metrics["downstream_forward_weight"] == pytest.approx(0.25)


def test_bind_sample_region_rejects_checkpoint_from_other_interactions():
    elastic = object()
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=SimpleNamespace(elastic_transport=elastic),
        metrics={
            "dependency_signatures": {
                "sample_downstream": "different-state",
            }
        },
    )
    checkpoint = GeometricSpecimenExit(
        (),
        {
            "tracked_downstream_source_probability": 0.0,
            "inelastic_absorbed_source_probability": 0.0,
        },
        dependency_signature="current-state",
    )
    sample_region = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=(),
        photon_paths=(),
        downstream_branches=(),
        spectrum=interactions.eds_spectrum,
        interactions=interactions,
        metrics={"sample_region_signature": "local"},
    )

    with pytest.raises(ValueError, match="same current dependency signature"):
        bind_sample_region_downstream(
            sample_region,
            checkpoint,
            interactions,
            expected_signature="current-state",
        )


def test_manual_sample_region_reuses_high_accuracy_specimen_exit(
    monkeypatch,
    detector_geometry,
):
    state = default_state()
    state.sample.inserted = True
    state.sample.eds_enabled = True
    elastic = SimpleNamespace(metrics={})
    shared_photon_transport = SimpleNamespace(
        metrics={"model": "shared main-spectrum photon transport"},
        geometry_complete=False,
        expected_detected_weight_per_segment=(0.0,) * 6,
    )
    spectrum = SimpleNamespace(
        elastic_transport=elastic,
        photon_transport=shared_photon_transport,
    )
    downstream_signature = "shared-downstream"
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=spectrum,
        inelastic_distribution=object(),
        metrics={
            "dependency_signatures": {
                "sample_region": "local-region",
                "sample_downstream": downstream_signature,
            },
            "existing_result_reused": True,
            "calculated_observables_this_call": (),
        },
    )
    shared_downstream_branch = SimpleNamespace(weight=0.8)
    checkpoint = GeometricSpecimenExit(
        (shared_downstream_branch,),
        {
            "downstream_forward_weight": 0.8,
            "tracked_downstream_source_probability": 0.8,
            "inelastic_absorbed_source_probability": 0.05,
        },
        dependency_signature=downstream_signature,
    )
    calculation_result = SimpleNamespace(
        simulation=object(),
        wave_imaging=None,
        specimen_exit=checkpoint,
    )
    bundle = SimpleNamespace(reaching_ray_count=1)
    local_electron = SimpleNamespace(kind="incident")
    monkeypatch.setattr(
        sample_region_module,
        "run_specimen_interactions",
        lambda *_args, **_kwargs: interactions,
    )
    monkeypatch.setattr(
        sample_region_module,
        "_electron_paths",
        lambda *_args, **_kwargs: ((local_electron,), bundle, bundle),
    )
    monkeypatch.setattr(
        sample_region_module,
        "_photon_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shared main-spectrum photons were resampled")
        ),
    )
    monkeypatch.setattr(
        sample_region_module,
        "build_geometric_specimen_exit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compatible high-accuracy checkpoint was rebuilt")
        ),
    )

    result = simulate_sample_region(
        state,
        calculation_result,
        detector_geometry,
        upstream_distance_um=50.0,
        downstream_distance_um=50.0,
        photon_path_count=0,
        secondary_path_count=0,
        seed=17,
        existing_interactions=interactions,
    )

    assert result.specimen_exit is checkpoint
    assert result.photon_transport is shared_photon_transport
    assert result.metrics["eds_photon_transport_shared_with_spectrum"]
    assert result.downstream_branches is checkpoint.branches
    assert result.metrics["sample_downstream_signature"] == (
        downstream_signature
    )
