import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.physics.first_order import (
    DetectorFrameCalibration,
    TransverseTransfer,
    linear_map_properties,
    relative_image_diffraction_orientation,
    trace_transverse_transfer,
)


def _rotation(degrees):
    angle = np.deg2rad(float(degrees))
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])


def _field_free_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.step_mm = 0.5
    state.history_step_mm = 0.5
    state.acceleration_enabled = False
    for component in (
        *state.lenses,
        *state.stigmators,
        *state.corrector_elements,
    ):
        if hasattr(component, "enabled"):
            component.enabled = False
    return state


def test_field_free_first_order_transfer_matches_exact_paraxial_drift():
    state = _field_free_state()
    source_z_mm = float(state.sample.z_mm)

    transfer = trace_transverse_transfer(
        state, source_z_mm, source_z_mm + 100.0
    )

    expected = np.array([
        [1.0, 0.0, 0.1, 0.0],
        [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert transfer.matrix == pytest.approx(expected, abs=2.0e-8)
    assert transfer.j_img.flags.writeable is False
    assert transfer.j_diff_m_per_rad.flags.writeable is False


def test_linear_map_properties_keep_rotation_reflection_and_anisotropy():
    preserved = _rotation(27.0) @ np.diag((4.0, 2.0))
    mirrored = _rotation(-18.0) @ np.diag((-3.0, 1.5))

    preserved_properties = linear_map_properties(preserved)
    mirrored_properties = linear_map_properties(mirrored)

    assert preserved_properties.orientation_deg == pytest.approx(27.0)
    assert preserved_properties.mirrored is False
    assert preserved_properties.isotropic_scale == pytest.approx(np.sqrt(8.0))
    assert preserved_properties.anisotropy_ratio == pytest.approx(2.0)
    assert mirrored_properties.orientation_deg == pytest.approx(-18.0)
    assert mirrored_properties.mirrored is True
    assert mirrored_properties.anisotropy_ratio == pytest.approx(2.0)


def test_relative_image_diffraction_map_preserves_full_signed_2d_relation():
    zeros = np.zeros((2, 2))
    identity = np.eye(2)
    image = TransverseTransfer(
        0.0,
        1.0,
        2.0 * _rotation(20.0),
        zeros,
        zeros,
        identity,
    )
    diffraction = TransverseTransfer(
        0.0,
        1.0,
        zeros,
        3.0 * _rotation(-15.0),
        zeros,
        identity,
    )
    calibrated_frame = DetectorFrameCalibration(
        key="test_camera",
        uncertainty_deg=0.2,
        status="measured_calibration",
        source="synthetic unit-test calibration",
    )

    relation = relative_image_diffraction_orientation(
        image,
        diffraction,
        1.9687e-12,
        image_detector=calibrated_frame,
        diffraction_detector=calibrated_frame,
    )

    assert relation.normalized_direction_map == pytest.approx(_rotation(35.0))
    assert relation.properties.orientation_deg == pytest.approx(35.0)
    assert relation.properties.mirrored is False
    assert relation.properties.anisotropy_ratio == pytest.approx(1.0)
    assert relation.detector_uncertainty_deg == pytest.approx(np.hypot(0.2, 0.2))
    assert relation.calibration_status == "calibrated_detector_axes"


def test_uncalibrated_detector_frame_blocks_absolute_uncertainty_claim():
    zeros = np.zeros((2, 2))
    identity = np.eye(2)
    image = TransverseTransfer(0.0, 1.0, identity, zeros, zeros, identity)
    diffraction = TransverseTransfer(
        0.0, 1.0, zeros, identity, zeros, identity
    )

    relation = relative_image_diffraction_orientation(
        image,
        diffraction,
        2.0e-12,
        image_detector=DetectorFrameCalibration(key="camera"),
        diffraction_detector=DetectorFrameCalibration(key="camera"),
    )

    assert relation.calibration_status == "uncalibrated_detector_axes"
    assert relation.detector_uncertainty_deg is None
