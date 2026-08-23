MINIMUM_TEM_STOP_Z_MM = 2800.0
RECORDING_STOP_MARGIN_MM = 0.5


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
