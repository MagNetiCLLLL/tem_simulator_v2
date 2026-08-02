from dataclasses import dataclass



@dataclass

class DetectorPlane:

    name: str

    z_mm: float

    colour: str

    segmented: bool = False

    enabled: bool = True



def ensure_detector_planes(state):

    if not getattr(state, "detector_planes", None):

        camera_z = float(state.camera.z_mm)

        state.detector_planes = [

            DetectorPlane("Fluorescent Screen", camera_z - 250.0, "#43a047"),

            DetectorPlane("Segmented STEM detector plane (BF / ADF / HAADF)", camera_z - 120.0, "#00897b", segmented=True),

            DetectorPlane("Camera", camera_z, "#5e35b1"),

        ]

    return state

