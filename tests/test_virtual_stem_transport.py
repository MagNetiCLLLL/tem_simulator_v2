from types import SimpleNamespace

import numpy as np
import pytest

import temsim.detector.stem_signal as stem_signal


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
