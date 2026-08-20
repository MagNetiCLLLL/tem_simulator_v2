from copy import deepcopy

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state


def _document():
    return deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "NoEnergyFilter.toml"
    ))


def _camera_part(document):
    return next(part for part in document["parts"] if part["key"] == "camera")


def test_selected_recording_manifest_owns_camera_axis_calibration():
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection(
        "FEG", "C3 + Probe Corrector", "Energy Filter"
    ))
    camera_part = state._module_manifest_parts["camera"].data

    assert state.camera.detector_axis_rotation_deg == pytest.approx(
        camera_part["detector_axis_rotation_deg"]
    )
    assert state.camera.detector_flip_x is camera_part["detector_flip_x"]
    assert state.camera.detector_flip_y is camera_part["detector_flip_y"]
    assert state.camera.detector_orientation_uncertainty_deg == pytest.approx(
        camera_part["detector_orientation_uncertainty_deg"]
    )
    assert state.camera.detector_orientation_status == (
        "uncalibrated_identity"
    )
    assert state.camera.detector_orientation_source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("detector_flip_x", 1, "must be Boolean"),
        ("detector_axis_rotation_deg", True, "rotation must be finite"),
        (
            "detector_orientation_uncertainty_deg",
            181.0,
            "uncertainty must be between",
        ),
        (
            "detector_orientation_status",
            "assumed_true",
            "status must be one of",
        ),
        ("detector_orientation_source", "", "source must not be empty"),
    ),
)
def test_manifest_rejects_invalid_camera_axis_calibration(field, value, message):
    document = _document()
    _camera_part(document)[field] = value

    with pytest.raises(ValueError, match=message):
        module_manifest.validate_document(document)
