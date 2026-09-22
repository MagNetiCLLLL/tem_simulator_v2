"""Current detector topology and its independent operating controls."""

from temsim.component_keys import (
    BRIGHT_FIELD_DETECTOR, CAMERA, DARK_FIELD_DETECTOR,
    FLUORESCENT_SCREEN, HAADF_DETECTOR, STEM_DETECTOR_KEYS,
)
from temsim.detector.camera import CameraDetectorComponent, camera_detector_from_dict, create_camera_detector
from temsim.detector.fluorescent_screen import (
    FluorescentScreenComponent, create_fluorescent_screen, fluorescent_screen_from_dict,
)
from temsim.detector.stem_detector import StemDetectorComponent, create_stem_detectors, stem_detector_from_dict

_RECORDING_ORDER = (HAADF_DETECTOR, FLUORESCENT_SCREEN, DARK_FIELD_DETECTOR, BRIGHT_FIELD_DETECTOR, CAMERA)


def default_recording_planes(state):
    anchor_z = float(state.selected_area_aperture.z_mm)
    planes = {detector.key: detector for detector in create_stem_detectors(anchor_z)}
    planes[FLUORESCENT_SCREEN] = create_fluorescent_screen(anchor_z)
    planes[CAMERA] = create_camera_detector(anchor_z)
    return [planes[key] for key in _RECORDING_ORDER]


def ensure_recording_system(state):
    if not getattr(state, "recording_planes", None):
        state.recording_planes = default_recording_planes(state)
    planes = {}
    for plane in state.recording_planes:
        key = plane.key
        if key not in _RECORDING_ORDER or key in planes:
            raise ValueError(f"Unknown or duplicate recording-plane key: {key!r}")
        expected = (StemDetectorComponent if key in STEM_DETECTOR_KEYS else
                    CameraDetectorComponent if key == CAMERA else FluorescentScreenComponent)
        if not isinstance(plane, expected):
            raise ValueError(f"Recording plane {key!r} requires its current physical component")
        planes[key] = plane
    if set(planes) != set(_RECORDING_ORDER):
        raise ValueError("The current recording system requires all five detector records")
    anchor_z = float(state.selected_area_aperture.z_mm)
    for plane in planes.values():
        plane.resolve_against(anchor_z).validate()
    state.recording_planes = [planes[key] for key in _RECORDING_ORDER]
    state.camera = planes[CAMERA]
    return state


def serialise_recording_system(state):
    ensure_recording_system(state)
    payload = []
    for plane in state.recording_planes:
        values = {"key": plane.key, "inserted": bool(plane.inserted), "colour": plane.colour}
        if isinstance(plane, StemDetectorComponent):
            values.update(
                readout_enabled=bool(plane.readout_enabled),
                centre_offset_x_mm=float(plane.centre_offset_x_mm),
                centre_offset_y_mm=float(plane.centre_offset_y_mm),
            )
        if plane.key == CAMERA:
            values["pixels"] = int(plane.pixels)
        payload.append(values)
    return payload


def restore_recording_system(state, data):
    if data is None:
        return ensure_recording_system(state)
    if not isinstance(data, (list, tuple)):
        raise ValueError("Recording-system controls must be a list of records")
    loaded = {}
    anchor_z = float(state.selected_area_aperture.z_mm)
    for values in data:
        if not isinstance(values, dict):
            raise ValueError("Each recording-plane record must be a mapping")
        key = values.get("key")
        if key not in _RECORDING_ORDER or key in loaded:
            raise ValueError(f"Unknown or duplicate recording-plane key: {key!r}")
        loader = (stem_detector_from_dict if key in STEM_DETECTOR_KEYS else
                  camera_detector_from_dict if key == CAMERA else fluorescent_screen_from_dict)
        loaded[key] = loader(values, anchor_z)
    if set(loaded) != set(_RECORDING_ORDER):
        raise ValueError("The current recording system requires all five detector records")
    # All records are validated before replacing the current detector state.
    state.recording_planes = [loaded[key] for key in _RECORDING_ORDER]
    state.camera = loaded[CAMERA]
    return state
