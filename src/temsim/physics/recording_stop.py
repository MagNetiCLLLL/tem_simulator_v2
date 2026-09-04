MINIMUM_TEM_STOP_Z_MM = 2800.0
RECORDING_STOP_MARGIN_MM = 0.5


def active_tem_recording_plane(state):
    """Return the first inserted TEM screen/Camera reached by the beam.

    Recording stops are physical interceptors.  An inserted fluorescent screen
    upstream of the Camera therefore owns the observable; the Camera becomes
    active only after that screen is retracted.
    """

    candidates = []
    for attribute in ("fluorescent_screen", "camera"):
        try:
            component = getattr(state, attribute, None)
        except (AttributeError, StopIteration):
            component = None
        if component is None or not bool(getattr(component, "inserted", False)):
            continue
        z_mm = float(component.z_mm)
        if z_mm > float(state.sample.z_mm):
            candidates.append(component)
    if not candidates:
        raise ValueError(
            "Insert the Fluorescent Screen or Camera to record a TEM result."
        )
    target = min(candidates, key=lambda component: float(component.z_mm))
    upstream_detectors = tuple(
        detector
        for detector in getattr(state, "stem_detectors", ())
        if bool(getattr(detector, "inserted", False))
        and float(state.sample.z_mm) < float(detector.z_mm) < float(target.z_mm)
    )
    if upstream_detectors:
        names = ", ".join(str(detector.name) for detector in upstream_detectors)
        raise ValueError(
            f"Retract upstream STEM detector(s) before TEM recording: {names}."
        )
    return target


def active_tem_recording_plane_z(state) -> float:
    """Return the axial position of the first inserted TEM recording stop."""

    return float(active_tem_recording_plane(state).z_mm)


def tem_projection_reference_plane(state):
    """Return the plane used to solve the projector optics.

    An inserted upstream screen is the physical target.  When every TEM
    recording device is retracted, keep the TOML Camera plane as the optical
    reference so projector alignment remains available without pretending an
    exposure was recorded.
    """

    try:
        return active_tem_recording_plane(state)
    except ValueError:
        if any(
            bool(getattr(component, "inserted", False))
            for component in (
                getattr(state, "fluorescent_screen", None),
                getattr(state, "camera", None),
            )
            if component is not None
        ):
            raise
        camera = getattr(state, "camera", None)
        if camera is None:
            raise ValueError("TEM projection reference plane is not configured.")
        z_mm = float(camera.z_mm)
        if not (z_mm == z_mm and abs(z_mm) < float("inf")):
            raise ValueError("TEM projection reference plane Z must be finite.")
        if z_mm <= float(state.sample.z_mm):
            raise ValueError(
                "TEM projection reference plane must be downstream of the sample."
            )
        return camera


def tem_projection_reference_plane_z(state) -> float:
    """Return the active or fallback TEM projector-reference coordinate."""

    return float(tem_projection_reference_plane(state).z_mm)


def tem_camera_plane_z(state) -> float:
    """Return the physical Camera interaction plane used by TEM imaging.

    Camera insertion controls whether an exposure is recorded, but it must not
    move the projector calibration reference or change a requested
    magnification.  Geometry remains owned by the selected recording-system
    TOML through ``state.camera.z_mm``.
    """

    camera = getattr(state, "camera", None)
    if camera is None:
        raise ValueError("TEM Camera plane is not configured.")
    z_mm = float(camera.z_mm)
    if not (z_mm == z_mm and abs(z_mm) < float("inf")):
        raise ValueError("TEM Camera plane Z must be finite.")
    if z_mm <= float(state.sample.z_mm):
        raise ValueError("TEM Camera plane must be downstream of the sample.")
    return z_mm

def determine_tem_stop_z(state):
    # Propagate through the full simulated column. Individual rays are stopped by
    # apertures, recording planes and the column wall, never by a global shortcut.
    interaction_positions = []
    for collection_name in (
        "recording_planes",
        "apertures",
        "lenses",
        "deflectors",
        "stigmators",
        "corrector_elements",
    ):
        for component in getattr(state, collection_name, ()):
            z_mm = getattr(component, "z_mm", None)
            if z_mm is not None:
                interaction_positions.append(float(z_mm))
            if not hasattr(component, "kick_events"):
                continue
            try:
                events = component.kick_events(
                    time_s=float(
                        getattr(state, "simulation_time_s", 0.0)
                    )
                )
            except TypeError:
                events = component.kick_events()
            interaction_positions.extend(
                float(event[0]) for event in events
            )
    furthest_interaction_z_mm = max(
        MINIMUM_TEM_STOP_Z_MM,
        *interaction_positions,
    )
    # This is a physical observation coordinate, not an integration-grid
    # sentinel.  Tying it to ``state.step_mm`` moved the image/diffraction
    # conjugate plane whenever preview accuracy changed.
    return furthest_interaction_z_mm + RECORDING_STOP_MARGIN_MM
