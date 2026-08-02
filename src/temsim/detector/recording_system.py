from dataclasses import dataclass, asdict

from temsim.component_keys import (
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    DARK_FIELD_DETECTOR,
    FLUORESCENT_SCREEN,
    HAADF_DETECTOR,
    STEM_DETECTOR_KEYS,
    canonical_recording_plane_key,
)
from temsim.detector.camera import (
    CameraDetectorComponent,
    camera_detector_from_dict,
    create_camera_detector,
)
from temsim.detector.fluorescent_screen import (
    FluorescentScreenComponent,
    create_fluorescent_screen,
    fluorescent_screen_from_dict,
)
from temsim.detector.stem_detector import (
    StemDetectorComponent,
    create_stem_detectors,
    stem_detector_from_dict,
)


@dataclass

class RecordingPlane:

    key: str

    name: str

    z_mm: float

    geometry: str

    outer_width_mm: float

    inner_diameter_mm: float = 0.0

    inserted: bool = True

    colour: str = "#455a64"


    @property

    def outer_diameter_mm(self):

        return self.outer_width_mm


def energy_filter_recording_plane_insertions(recording_planes, energy_filter_enabled):
    """Return recording-plane insertion states compatible with the filter branch."""
    return {
        plane.key: False
        if energy_filter_enabled
        and plane.key in {
            FLUORESCENT_SCREEN,
            BRIGHT_FIELD_DETECTOR,
            CAMERA,
        }
        else bool(plane.inserted)
        for plane in recording_planes
    }
def default_recording_planes(state):

    anchor_z = float(state.selected_area_aperture.z_mm)
    stem = {
        detector.key: detector
        for detector in create_stem_detectors(anchor_z)
    }
    screen = create_fluorescent_screen(anchor_z)
    camera = create_camera_detector(anchor_z)

    return [

        stem[HAADF_DETECTOR],

        screen,

        stem[DARK_FIELD_DETECTOR],

        stem[BRIGHT_FIELD_DETECTOR],

        camera,

    ]



def ensure_recording_system(state):

    if not getattr(state, "recording_planes", None):

        state.recording_planes = default_recording_planes(state)
    else:
        anchor_z = float(state.selected_area_aperture.z_mm)
        defaults = {
            plane.key: plane
            for plane in default_recording_planes(state)
        }
        canonical = {}
        canonical_source_keys = {}
        for plane in state.recording_planes:
            source_key = str(plane.key)
            key = canonical_recording_plane_key(source_key)
            plane.key = key
            if key in canonical:
                prefer_component = (
                    isinstance(plane, StemDetectorComponent)
                    and not isinstance(
                        canonical[key], StemDetectorComponent
                    )
                ) or (
                    isinstance(plane, FluorescentScreenComponent)
                    and not isinstance(
                        canonical[key], FluorescentScreenComponent
                    )
                ) or (
                    isinstance(plane, CameraDetectorComponent)
                    and not isinstance(
                        canonical[key], CameraDetectorComponent
                    )
                )
                prefer_canonical_key = (
                    source_key == key
                    and canonical_source_keys[key] != key
                )
                if prefer_component or prefer_canonical_key:
                    canonical[key] = plane
                    canonical_source_keys[key] = source_key
                continue
            canonical[key] = plane
            canonical_source_keys[key] = source_key
        for key in STEM_DETECTOR_KEYS:
            plane = canonical.get(key)
            if plane is None:
                canonical[key] = defaults[key]
            elif not isinstance(plane, StemDetectorComponent):
                canonical[key] = stem_detector_from_dict(
                    asdict(plane), anchor_z
                )
            else:
                plane.resolve_against(anchor_z).validate()
        screen = canonical.get(FLUORESCENT_SCREEN)
        if screen is None:
            canonical[FLUORESCENT_SCREEN] = defaults[FLUORESCENT_SCREEN]
        elif not isinstance(screen, FluorescentScreenComponent):
            canonical[FLUORESCENT_SCREEN] = (
                fluorescent_screen_from_dict(asdict(screen), anchor_z)
            )
        else:
            screen.resolve_against(anchor_z).validate()
        camera = canonical.get(CAMERA)
        if camera is None:
            canonical[CAMERA] = defaults[CAMERA]
        elif not isinstance(camera, CameraDetectorComponent):
            canonical[CAMERA] = camera_detector_from_dict(
                asdict(camera), anchor_z
            )
        else:
            camera.resolve_against(anchor_z).validate()
        state.recording_planes = [
            canonical[key]
            for key in (
                HAADF_DETECTOR,
                FLUORESCENT_SCREEN,
                DARK_FIELD_DETECTOR,
                BRIGHT_FIELD_DETECTOR,
                CAMERA,
            )
        ]
    if getattr(state, "wobble_observation_plane_key", "") in {
        "df_s",
        "adf",
    }:
        state.wobble_observation_plane_key = DARK_FIELD_DETECTOR

    state.camera = next(
        p for p in state.recording_planes
        if p.key == CAMERA
    )

    return state



def serialise_recording_system(state):

    ensure_recording_system(state)

    payload = []
    for plane in state.recording_planes:
        values = {
            "key": plane.key,
            "inserted": bool(plane.inserted),
        }
        if plane.key == CAMERA:
            values["pixels"] = int(plane.pixels)
        payload.append(values)
    return payload



def restore_recording_system(state, data):

    if data:

        allowed={"key","name","z_mm","geometry","outer_width_mm","inner_diameter_mm","inserted","colour"}
        anchor_z = float(state.selected_area_aperture.z_mm)
        loaded = {}
        source_keys = {}
        for item in data:
            source_key = str(item.get("key", ""))
            key = canonical_recording_plane_key(source_key)
            values = dict(item)
            values["key"] = key
            if key in STEM_DETECTOR_KEYS:
                plane = stem_detector_from_dict(values, anchor_z)
            elif key == FLUORESCENT_SCREEN:
                plane = fluorescent_screen_from_dict(values, anchor_z)
            elif key == CAMERA:
                plane = camera_detector_from_dict(values, anchor_z)
            else:
                plane = RecordingPlane(**{
                    k: v for k, v in values.items() if k in allowed
                })
            if (
                key not in loaded
                or (
                    source_key == key
                    and source_keys[key] != key
                )
            ):
                loaded[key] = plane
                source_keys[key] = source_key
        state.recording_planes = list(loaded.values())

    return ensure_recording_system(state)
