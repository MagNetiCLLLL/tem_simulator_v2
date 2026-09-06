from types import SimpleNamespace

import numpy as np
import pytest

import temsim.detector.stem_signal as stem_signal
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def _branch(name, kind, x_mm, *, blocked_z, blocked_key, weight):
    x_m = float(x_mm) * 1.0e-3
    return SimpleNamespace(
        name=name,
        interaction_kind=kind,
        weight=float(weight),
        ray_weight=np.asarray((1.0,)),
        z=np.asarray((1.0, 2.0)),
        x=np.asarray(((0.0,), (x_m,))),
        y=np.asarray(((0.0,), (0.0,))),
        blocked_z=np.asarray((float(blocked_z),)),
        blocked_key=[blocked_key],
    )


def test_virtual_stem_counts_only_post_column_physical_detector_hits(monkeypatch):
    plane = SimpleNamespace(
        key="detector",
        name="Detector",
        inserted=True,
        z_mm=2.0,
        hit_mask=lambda x, y: np.hypot(x, y) <= 0.5,
    )
    state = SimpleNamespace(
        beam_voltage_kv=300.0,
        sample=SimpleNamespace(
            inserted=True,
            thickness_nm=10.0,
            specimen_mode="virtual",
            specimen_preset_key="si_110",
            diffraction_enabled=True,
            stem_poisson_enabled=False,
        ),
        recording_planes=(plane,),
        ac_deflector=SimpleNamespace(
            scan_frame_period_s=1.0,
            scan_pixel_size_nm=1.0,
            scan_field_of_view_x_nm=1.0,
            scan_field_of_view_y_nm=1.0,
        ),
        descan_deflector=SimpleNamespace(enabled=False, scan_enabled=False),
        electron_gun=SimpleNamespace(emitted_current_a=1.0e-12, ray_count=1),
    )
    direct = _branch(
        "000",
        "transmitted",
        0.0,
        blocked_z=2.0,
        blocked_key="detector",
        weight=0.5,
    )
    scattered = _branch(
        "virtual_+g",
        "diffraction_spots",
        2.0,
        blocked_z=np.nan,
        blocked_key=None,
        weight=0.5,
    )
    simulation = SimpleNamespace(
        branches={"000": direct, "virtual_+g": scattered},
    )
    physical = SimpleNamespace(
        key="detector",
        detector=SimpleNamespace(name="Detector"),
    )
    distribution = SimpleNamespace(
        scattered_probability=0.5,
        absorbed_probability=0.0,
        components=(),
    )
    probe = SimpleNamespace(probe_sigma_nm=0.0)
    monkeypatch.setattr(
        stem_signal,
        "build_virtual_angular_distribution",
        lambda *_args, **_kwargs: distribution,
    )
    monkeypatch.setattr(
        stem_signal,
        "virtual_density_at_scan",
        lambda *_args, **_kwargs: np.ones((1, 1)),
    )
    monkeypatch.setattr(
        stem_signal,
        "measure_sample_current",
        lambda *_args, **_kwargs: SimpleNamespace(fraction=1.0),
    )
    monkeypatch.setattr(
        stem_signal,
        "paired_kick_response",
        lambda *_args, **_kwargs: np.zeros((2, 2)),
    )
    monkeypatch.setattr(
        stem_signal,
        "probe_state_from_simulation",
        lambda *_args, **_kwargs: probe,
    )

    result = stem_signal._virtual_stem_scan(
        simulation,
        state,
        (physical,),
        {"detector": object()},
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1, 2)),
        np.zeros(2),
        np.zeros((1, 1)),
        np.zeros(2),
    )

    assert result.fractions["detector"][0, 0] == pytest.approx(0.5)
    assert result.uncollected_fraction[0, 0] == pytest.approx(0.5)
    assert result.metrics["post_sample_lens_transport_applied"] is True
    assert result.metrics["detector_signal_requires_physical_intersection"] is True

    direct.blocked_z[0] = 1.5
    direct.blocked_key[0] = "column_wall"
    blocked = stem_signal._virtual_stem_scan(
        simulation,
        state,
        (physical,),
        {"detector": object()},
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1, 2)),
        np.zeros(2),
        np.zeros((1, 1)),
        np.zeros(2),
    )

    assert blocked.fractions["detector"][0, 0] == 0.0
    assert blocked.uncollected_fraction[0, 0] == 1.0


def test_geometric_stem_consumes_shared_specimen_exit_without_retracing(
    monkeypatch,
):
    detector = SimpleNamespace(
        key="detector",
        name="Detector",
        inserted=True,
        readout_enabled=True,
    )
    plane = SimpleNamespace(
        key="detector",
        inserted=True,
        z_mm=2.0,
        hit_mask=lambda x, y: np.hypot(x, y) <= 0.5,
    )
    ac = SimpleNamespace(
        enabled=True,
        scan_enabled=True,
        scan_pixels_x=1,
        scan_lines=1,
        scan_command_matrix_mrad=np.eye(2),
        scan_frame_period_s=1.0,
        scan_pixel_size_nm=1.0,
        scan_field_of_view_x_nm=1.0,
        scan_field_of_view_y_nm=1.0,
        scan_kick_mrad=lambda _time_s: (0.0, 0.0),
    )
    descan = SimpleNamespace(
        enabled=False,
        scan_enabled=False,
        scan_kick_mrad=lambda _time_s: (0.0, 0.0),
    )
    state = SimpleNamespace(
        simulation_time_s=0.0,
        sample=SimpleNamespace(
            inserted=True,
            z_mm=1.0,
            specimen_mode="atomic",
            stem_wave_enabled=False,
            scan_origin_x_nm=0.0,
            scan_origin_y_nm=0.0,
        ),
        ac_deflector=ac,
        descan_deflector=descan,
        stem_detectors=(detector,),
        recording_planes=(plane,),
    )
    branch = SimpleNamespace(
        z=np.asarray((1.0, 2.0)),
        x=np.zeros((2, 1)),
        y=np.zeros((2, 1)),
        blocked_z=np.asarray((np.nan,)),
        blocked_key=[""],
        ray_weight=np.asarray((1.0,)),
        weight=0.8,
    )
    downstream_signature = "current-downstream-signature"
    shared_exit = GeometricSpecimenExit(
        (branch,),
        {
            "tracked_downstream_source_probability": 0.8,
            "inelastic_absorbed_source_probability": 0.1,
        },
        dependency_signature=downstream_signature,
    )
    elastic = object()
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=SimpleNamespace(elastic_transport=elastic),
        inelastic_distribution=object(),
    )
    simulation = SimpleNamespace(incident=object(), branches={})
    physical = SimpleNamespace(key="detector", detector=detector)

    monkeypatch.setattr(
        stem_signal, "sample_illumination_absent", lambda *_args: False
    )
    monkeypatch.setattr(
        stem_signal,
        "calculation_signatures",
        lambda _state: {"sample_downstream": downstream_signature},
    )
    monkeypatch.setattr(
        stem_signal,
        "calibrate_scan_system",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        stem_signal,
        "raster_sample_grid",
        lambda *_args, **_kwargs: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        ),
    )
    monkeypatch.setattr(
        stem_signal,
        "paired_kick_response",
        lambda *_args, **_kwargs: np.zeros((2, 2)),
    )
    monkeypatch.setattr(
        stem_signal,
        "physical_angular_detectors",
        lambda *_args: ((physical,), {"detector": object()}),
    )
    monkeypatch.setattr(
        stem_signal,
        "_detector_center_shifts_mrad",
        lambda *_args, **_kwargs: np.zeros((1, 1, 1, 2)),
    )
    monkeypatch.setattr(
        stem_signal,
        "build_geometric_specimen_exit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shared specimen exit was retraced")
        ),
    )
    monkeypatch.setattr(
        stem_signal,
        "_current_values",
        lambda *_args: (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(stem_signal, "collection_angle", lambda *_args: None)
    monkeypatch.setattr(
        stem_signal,
        "_stem_result",
        lambda _state, _simulation, _scan_x, _scan_y, _images, _signals,
        metrics, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        stem_signal,
        "_readout_view",
        lambda result, **_kwargs: result,
    )

    metrics = stem_signal.acquire_stem_scan(
        simulation,
        state,
        specimen_interactions=interactions,
        geometric_specimen_exit=shared_exit,
    )

    assert metrics["shared_specimen_exit_transport_used"] is True
    assert metrics["finite_specimen_exit"] == shared_exit.metrics

    rebuild_calls = []

    def rebuild_exit(*_args, **kwargs):
        rebuild_calls.append(kwargs)
        return GeometricSpecimenExit(
            (branch,),
            {
                "tracked_downstream_source_probability": 0.7,
                "inelastic_absorbed_source_probability": 0.2,
            },
            dependency_signature=kwargs["dependency_signature"],
        )

    monkeypatch.setattr(
        stem_signal,
        "build_geometric_specimen_exit",
        rebuild_exit,
    )
    stale_exit = GeometricSpecimenExit(
        (branch,),
        {
            "tracked_downstream_source_probability": 0.6,
            "inelastic_absorbed_source_probability": 0.3,
        },
        dependency_signature="stale-downstream-signature",
    )

    rebuilt_metrics = stem_signal.acquire_stem_scan(
        simulation,
        state,
        specimen_interactions=interactions,
        geometric_specimen_exit=stale_exit,
    )

    assert len(rebuild_calls) == 1
    assert rebuild_calls[0]["dependency_signature"] == downstream_signature
    assert rebuilt_metrics["shared_specimen_exit_transport_used"] is False
    assert rebuilt_metrics["rejected_specimen_exit_reason"]
    assert rebuilt_metrics["finite_specimen_exit"][
        "inelastic_absorbed_source_probability"
    ] == pytest.approx(0.2)

    stale_assertion_metrics = stem_signal.acquire_stem_scan(
        simulation,
        state,
        specimen_interactions=interactions,
        geometric_specimen_exit=stale_exit,
        geometric_specimen_exit_signature="stale-downstream-signature",
    )

    assert len(rebuild_calls) == 2
    assert rebuild_calls[-1]["dependency_signature"] == downstream_signature
    assert stale_assertion_metrics[
        "shared_specimen_exit_transport_used"
    ] is False
    assert stale_assertion_metrics["rejected_specimen_exit_reason"]

    with pytest.raises(ValueError, match="share one elastic"):
        stem_signal.acquire_stem_scan(
            simulation,
            state,
            specimen_interactions=SimpleNamespace(
                elastic_transport=object(),
                eds_spectrum=SimpleNamespace(elastic_transport=object()),
                inelastic_distribution=object(),
            ),
            geometric_specimen_exit=shared_exit,
        )
