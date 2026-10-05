"""Wave recording labels follow captured conjugacy, not the requested mode."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import camera_wave
from temsim.physics.first_order import TransverseTransfer


@pytest.mark.parametrize("projector_mode", ("image", "diffraction"))
@pytest.mark.parametrize("a,b,kind,observable", (
    (1., 1e-5, "image", "image"),
    (0., .02, "diffraction", "diffraction_pattern"),
    (1., .02, "mixed", "mixed"),
    (5e-4, 1e-5, "degenerate", "degenerate"),
))
def test_wave_recording_classification_uses_canonical_map(
    monkeypatch, projector_mode, a, b, kind, observable,
):
    from temsim.optics import direct_alignment

    # These analytically specified maps are symplectic in each transverse
    # axis. All B blocks are nonsingular, so either existing camera method
    # remains executable even when the selected mode disagrees with the map.
    c, d = ((-1. / b, 0.) if a == 0. else (0., 1. / a))
    transfer = TransverseTransfer(0., 10., a*np.eye(2), b*np.eye(2),
                                  c*np.eye(2), d*np.eye(2),
                                  input_basis="specimen_canonical_momentum")
    monkeypatch.setattr(direct_alignment, "diffraction_transfer", lambda *_: transfer)
    monkeypatch.setattr(camera_wave, "_camera_affine_offset_m", lambda *_: np.zeros(2))
    plane = SimpleNamespace(
        key="camera", name="Camera", z_mm=10., inserted=True, pixels=8,
        width_mm=1., outer_width_mm=1.,
        point_spread_model="none", point_spread_sigma_x_mm=0.,
        point_spread_sigma_y_mm=0., point_spread_rotation_deg=0.,
        point_spread_status="provisional_model_parameter",
        point_spread_source="Analytical classification fixture",
        hit_mask=lambda x, y: np.ones(np.broadcast_shapes(x.shape, y.shape), dtype=bool),
    )
    plane.validate = lambda: plane
    state = SimpleNamespace(projector_mode=projector_mode, camera=plane,
                            sample=SimpleNamespace(z_mm=0.), apertures=())
    axis = np.linspace(-8., 8., 8)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    wave = np.exp(-(x*x+y*y)/16.).astype(complex)

    result = camera_wave.project_wave_to_camera(state, wave, axis, axis, .0197)

    assert result.metrics["camera_projector_mode"] == projector_mode
    assert result.metrics["recording_plane_kind"] == kind
    assert result.metrics["recording_plane_observable"] == observable
    assert result.metrics["recording_plane_image_residual_m_per_rad"] == pytest.approx(b)
    assert result.metrics["recording_plane_diffraction_residual"] == pytest.approx(a)
    assert np.all(np.isfinite(result.intensity))


@pytest.mark.parametrize("classification,legacy_observable,expected", (
    ("image", "diffraction_pattern", "Image"),
    ("diffraction", "image", "Diffraction pattern"),
    ("mixed", "diffraction_pattern", "Mixed-plane intensity"),
    ("degenerate", "image", "Degenerate-plane intensity"),
    (None, "diffraction_pattern", "Intensity"),
    (None, "image", "Intensity"),
    (None, None, "Intensity"),
))
def test_wave_view_titles_use_conjugacy_and_legacy_results_remain_neutral(
    qtbot, classification, legacy_observable, expected,
):
    from temsim.gui.visualization import WaveImagingView

    view = WaveImagingView()
    qtbot.addWidget(view)
    metrics = {"recording_plane_name": "Camera", "surviving_rays": 1}
    if classification is not None:
        metrics["recording_plane_kind"] = classification
    if legacy_observable is not None:
        metrics["recording_plane_observable"] = legacy_observable
    axis = np.linspace(-1., 1., 4)
    result = SimpleNamespace(
        x_angstrom=axis, y_angstrom=axis,
        image_intensity=np.ones((4, 4)), diffraction_intensity=np.ones((4, 4)),
        spatial_frequency_inv_angstrom=axis, metrics=metrics,
        preset_name="Vacuum", preset_key="vacuum",
    )

    view._display_wave_result(result)

    assert view.image.getView().titleLabel.text == f"Camera: {expected} (display-normalised)"
    assert f"Camera {expected.lower()}" in view.summary.text()
