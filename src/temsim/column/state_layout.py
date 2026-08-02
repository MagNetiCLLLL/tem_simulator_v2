"""State adapters for the standalone mechanical column layout."""

from dataclasses import replace

from temsim.column.layout import (
    C3Hardware,
    CorrectorAssembly,
    LayoutConfiguration,
    ObjectiveLayout,
    build_optics_layout,
)
from temsim.column.effective_axis import (
    topology_signature,
)
from temsim.configuration import corrector_mode_for_hardware
from temsim.optics.selected_area_aperture import (
    IMAGE_CORRECTED_INSTALLATION,
    STANDALONE_INSTALLATION,
)
from temsim.optics.diffraction_stigmator import (
    IMAGE_CORRECTED_INSTALLATION as DIFFRACTION_IMAGE_INSTALLATION,
    STANDALONE_INSTALLATION as DIFFRACTION_STANDALONE_INSTALLATION,
)
from temsim.optics.diffraction_lens import (
    IMAGE_CORRECTED_INSTALLATION as DIFFRACTION_LENS_IMAGE_INSTALLATION,
    STANDALONE_INSTALLATION as DIFFRACTION_LENS_STANDALONE_INSTALLATION,
)
from temsim.component_keys import (
    AC_DEFLECTOR,
    ADAPTER_LENS,
    BEAM_DEFLECTOR,
    BRIGHT_FIELD_DETECTOR,
    CAMERA,
    C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    CONDENSER_DEFLECTOR,
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    CONDENSER_STIGMATOR,
    DESCAN_DEFLECTOR,
    DARK_FIELD_DETECTOR,
    DIFFRACTION_LENS,
    DIFFRACTION_STIGMATOR,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    FLUORESCENT_SCREEN,
    INTERMEDIATE_LENS,
    HAADF_DETECTOR,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    GUN_EXTRACTOR_APERTURE,
    MINI_CONDENSER,
    OBJECTIVE_LENS,
    OBJECTIVE_APERTURE,
    OBJECTIVE_STIGMATOR,
    IMAGE_DIFFRACTION_DEFLECTOR,
    IMAGE_CORRECTOR_ELEMENT_KEYS,
    IMAGE_CORRECTOR_LENS_KEYS,
    IMAGE_CORRECTOR_OL_POST_LENS,
    PROBE_DPH2_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR,
    PROBE_DP11_DEFLECTOR,
    PROBE_DP12_SCAN_DEFLECTOR,
    PROBE_DP21_DEFLECTOR,
    PROBE_DP22_DEFLECTOR,
    PROBE_HP2_HEXAPOLE,
    PROBE_HP1_HEXAPOLE,
    PROBE_HPOL_HEXAPOLE,
    PROBE_HPC_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE,
    PROBE_QPH1_QUADRUPOLE,
    PROBE_QPOL_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
    SELECTED_AREA_APERTURE,
)


_CORRECTORS = {
    "no_corrector": CorrectorAssembly.NO_CORRECTOR,
    "probe_corrector": CorrectorAssembly.PROBE_CORRECTOR,
    "image_corrector": CorrectorAssembly.IMAGE_CORRECTOR,
    "double_corrector": CorrectorAssembly.DOUBLE_CORRECTOR,
    # Read-only migration alias for states written before schema V10.
    "dual_corrector": CorrectorAssembly.DOUBLE_CORRECTOR,
}

_LENS_KEYS = {
    CONDENSER_LENS_1: CONDENSER_LENS_1,
    CONDENSER_LENS_2: CONDENSER_LENS_2,
    CONDENSER_LENS_3: CONDENSER_LENS_3,
    ADAPTER_LENS: ADAPTER_LENS,
    PROBE_TL22_LENS: PROBE_TL22_LENS,
    PROBE_TL21_LENS: PROBE_TL21_LENS,
    PROBE_TL12_LENS: PROBE_TL12_LENS,
    MINI_CONDENSER: MINI_CONDENSER,
    OBJECTIVE_LENS: OBJECTIVE_LENS,
    DIFFRACTION_LENS: DIFFRACTION_LENS,
    INTERMEDIATE_LENS: INTERMEDIATE_LENS,
    PROJECTOR_LENS_1: PROJECTOR_LENS_1,
    PROJECTOR_LENS_2: PROJECTOR_LENS_2,
    **{key: key for key in IMAGE_CORRECTOR_LENS_KEYS},
}

_APERTURE_KEYS = {
    CONDENSER_APERTURE_2: CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3: CONDENSER_APERTURE_3,
    OBJECTIVE_APERTURE: OBJECTIVE_APERTURE,
    SELECTED_AREA_APERTURE: SELECTED_AREA_APERTURE,
    ENERGY_FILTER_ENTRANCE_APERTURE: ENERGY_FILTER_ENTRANCE_APERTURE,
}

_STIGMATOR_KEYS = {
    CONDENSER_STIGMATOR: CONDENSER_STIGMATOR,
    "objective_stigmator": OBJECTIVE_STIGMATOR,
    DIFFRACTION_STIGMATOR: DIFFRACTION_STIGMATOR,
}

_DEFLECTOR_KEYS = {
    CONDENSER_DEFLECTOR: CONDENSER_DEFLECTOR,
    BEAM_DEFLECTOR: BEAM_DEFLECTOR,
    PROBE_DP12_SCAN_DEFLECTOR: PROBE_DP12_SCAN_DEFLECTOR,
    IMAGE_DIFFRACTION_DEFLECTOR: IMAGE_DIFFRACTION_DEFLECTOR,
}

_RECORDING_PLANE_KEYS = {
    HAADF_DETECTOR: HAADF_DETECTOR,
    FLUORESCENT_SCREEN: FLUORESCENT_SCREEN,
    DARK_FIELD_DETECTOR: DARK_FIELD_DETECTOR,
    BRIGHT_FIELD_DETECTOR: BRIGHT_FIELD_DETECTOR,
    CAMERA: CAMERA,
}

_CORRECTOR_ELEMENT_KEYS = {
    PROBE_DPH2_DEFLECTOR: PROBE_DPH2_DEFLECTOR,
    PROBE_QPH2_QUADRUPOLE: PROBE_QPH2_QUADRUPOLE,
    PROBE_HP2_HEXAPOLE: PROBE_HP2_HEXAPOLE,
    PROBE_HPC_HEXAPOLE: PROBE_HPC_HEXAPOLE,
    PROBE_DP22_DEFLECTOR: PROBE_DP22_DEFLECTOR,
    PROBE_QPC_QUADRUPOLE: PROBE_QPC_QUADRUPOLE,
    PROBE_DP21_DEFLECTOR: PROBE_DP21_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR: PROBE_DPH1_DEFLECTOR,
    PROBE_QPH1_QUADRUPOLE: PROBE_QPH1_QUADRUPOLE,
    PROBE_HP1_HEXAPOLE: PROBE_HP1_HEXAPOLE,
    PROBE_HPOL_HEXAPOLE: PROBE_HPOL_HEXAPOLE,
    PROBE_QPOL_QUADRUPOLE: PROBE_QPOL_QUADRUPOLE,
    PROBE_DP11_DEFLECTOR: PROBE_DP11_DEFLECTOR,
    AC_DEFLECTOR: AC_DEFLECTOR,
    DESCAN_DEFLECTOR: DESCAN_DEFLECTOR,
    **{key: key for key in IMAGE_CORRECTOR_ELEMENT_KEYS},
}


def resolve_selected_area_downstream_anchors(
    state,
    *,
    image_corrected,
):
    """Resolve the Selected Area Aperture and its complete rigid child group."""

    selected_area_installation = (
        IMAGE_CORRECTED_INSTALLATION
        if image_corrected
        else STANDALONE_INSTALLATION
    )
    state.selected_area_aperture.select_installation(
        selected_area_installation
    )
    selected_area_geometry = (
        state.selected_area_aperture.geometry_for(
            selected_area_installation
        )
    )
    state.diffraction_stigmator.select_installation(
        DIFFRACTION_IMAGE_INSTALLATION
        if image_corrected
        else DIFFRACTION_STANDALONE_INSTALLATION
    )
    state.diffraction_stigmator.resolve_against(
        selected_area_geometry
    )
    state.diffraction_lens.select_installation(
        DIFFRACTION_LENS_IMAGE_INSTALLATION
        if image_corrected
        else DIFFRACTION_LENS_STANDALONE_INSTALLATION
    )
    state.diffraction_lens.resolve_against(
        selected_area_geometry
    )
    state.intermediate_lens.resolve_against(
        selected_area_geometry
    )
    state.projector_lens_p1.resolve_against(
        selected_area_geometry
    )
    state.projector_lens_p2.resolve_against(
        selected_area_geometry
    )
    anchor_z_mm = float(state.selected_area_aperture.z_mm)
    for detector in state.stem_detectors:
        detector.resolve_against(anchor_z_mm)
    state.fluorescent_screen.resolve_against(anchor_z_mm)
    state.camera.resolve_against(anchor_z_mm)
    state.energy_filter_entrance_aperture.resolve_against(
        anchor_z_mm
    ).validate()
    return selected_area_geometry


def layout_configuration_from_state(
    state, *, preserve_operating_parameters=True
):
    """Translate persisted topology selections into layout configuration."""
    hardware = C3Hardware(getattr(state, "layout_c3_hardware", "three_condenser"))
    corrector = _CORRECTORS.get(
        corrector_mode_for_hardware(
            getattr(state, "corrector_mode", "probe_corrector"),
            hardware.value,
        ),
        CorrectorAssembly.PROBE_CORRECTOR,
    )
    c3_excited = bool(getattr(
        state, "layout_c3_excited",
        getattr(state, "column_mode", "three_lens") == "three_lens",
    ))
    if hardware is C3Hardware.TWO_CONDENSER:
        c3_excited = False

    from temsim.optics.beam_deflector import (
        resolve_beam_deflector_after_active_aperture,
    )
    resolve_beam_deflector_after_active_aperture(state)
    # These components now own one physical coordinate each.  Discard stale
    # effective-axis cache entries from older dual-coordinate state files.
    for binding_key in (
        f"aperture:{CONDENSER_APERTURE_2}",
        f"aperture:{CONDENSER_APERTURE_3}",
        f"deflector:{BEAM_DEFLECTOR}",
    ):
        state.layout_reference_positions.pop(binding_key, None)
        state.layout_reference_enabled.pop(binding_key, None)
    if corrector in (
        CorrectorAssembly.PROBE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    ):
        from temsim.optics.probe_corrector import (
            anchor_probe_corrector_to_beam_deflector,
            synchronise_probe_corrector_physical_axis,
        )
        anchor_probe_corrector_to_beam_deflector(state)
        synchronise_probe_corrector_physical_axis(state)
    probe_installed = corrector in (
        CorrectorAssembly.PROBE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    )
    mini_installation = (
        "integrated" if probe_installed else "standalone"
    )
    state.mini_condenser.select_installation(mini_installation)
    if corrector in (
        CorrectorAssembly.IMAGE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    ):
        from temsim.optics.image_corrector import (
            resolve_image_corrector_mechanical_anchors,
        )
        resolve_image_corrector_mechanical_anchors(state)
    else:
        from temsim.optics.selected_area_aperture import (
            resolve_standalone_selected_area_aperture_anchor,
        )
        resolve_standalone_selected_area_aperture_anchor(state)
    energy_filter = getattr(state, "energy_filter", None)
    energy_filter_selected = bool(
        getattr(energy_filter, "enabled", False)
        or getattr(state, "energy_filter_mode", "no_energy_filter") == "energy_filter"
    )
    resolve_selected_area_downstream_anchors(
        state,
        image_corrected=corrector in (
            CorrectorAssembly.IMAGE_CORRECTOR,
            CorrectorAssembly.DOUBLE_CORRECTOR,
        ),
    )
    state.energy_filter_entrance_aperture.installed = (
        energy_filter_selected
    )
    configuration = LayoutConfiguration(
        corrector=corrector,
        electron_gun_type=state.electron_gun.type_key,
        c3_hardware=hardware,
        c3_excited=c3_excited,
        monochromator_installed=bool(
            state.electron_gun.type_key == "cold_feg"
            and getattr(state, "monochromator_installed", False)
        ),
        source_relative_column_offset_mm=(
            float(getattr(
                state, "monochromator_column_offset_mm", 0.0
            ))
        ),
        sample_center_from_source_mm=float(state.sample.z_mm),
        objective=ObjectiveLayout(
            inner_face_gap_mm=state.objective_lens.inner_face_gap_mm,
            sample_axial_offset_mm=getattr(
                state.objective_lens, "sample_axial_offset_mm", 0.0
            ),
            specimen_thickness_mm=state.sample.thickness_nm * 1.0e-6,
        ),
        energy_filter_selected=energy_filter_selected,
        gun_components=state.electron_gun.components,
        condenser_components=tuple(state.condenser_system),
        condenser_aperture_2_component=state.condenser_aperture_2,
        condenser_aperture_3_component=state.condenser_aperture_3,
        condenser_deflector_component=state.condenser_deflector,
        beam_deflector_component=state.beam_deflector,
        ac_deflector_component=state.ac_deflector,
        mini_condenser_component=state.mini_condenser,
        condenser_stigmator_component=state.condenser_stigmator,
        diffraction_stigmator_component=state.diffraction_stigmator,
        diffraction_lens_component=state.diffraction_lens,
        intermediate_lens_component=state.intermediate_lens,
        projector_lens_p1_component=state.projector_lens_p1,
        projector_lens_p2_component=state.projector_lens_p2,
        stem_detector_components=tuple(state.stem_detectors),
        fluorescent_screen_component=state.fluorescent_screen,
        camera_component=state.camera,
        energy_filter_entrance_aperture_component=(
            state.energy_filter_entrance_aperture
        ),
        objective_lens_component=state.objective_lens,
        objective_aperture_component=state.objective_aperture,
        selected_area_aperture_component=(
            state.selected_area_aperture
        ),
        objective_stigmator_component=state.objective_stigmator,
        image_diffraction_deflector_component=(
            state.image_diffraction_deflector
        ),
        descan_deflector_component=state.descan_deflector,
        adapter_lens_component=state.adapter_lens,
        dph2_deflector_component=state.dph2_deflector,
        dp22_deflector_component=state.dp22_deflector,
        qph2_quadrupole_component=state.qph2_quadrupole,
        hp2_hexapole_component=state.hp2_hexapole,
        hpc_hexapole_component=state.hpc_hexapole,
        probe_corrector_tail_components=(
            state.qpc_quadrupole,
            state.dp21_deflector,
            state.tl21_lens,
            state.dph1_deflector,
            state.qph1_quadrupole,
            state.hp1_hexapole,
            state.hpol_hexapole,
            state.qpol_quadrupole,
            state.dp11_deflector,
            state.tl12_lens,
            state.dp12_scan_deflector,
        ),
        tl22_lens_component=state.tl22_lens,
        image_corrector_components=state.image_corrector_system.components,
    )
    # The selected TOML assembly is the sole mechanical and optical-reference
    # authority. Runtime operating controls are preserved while manifest-owned
    # geometry and field calibration are reloaded here.
    from temsim.column.module_assembly import apply_column_manifest_geometry
    apply_column_manifest_geometry(
        state,
        configuration,
        preserve_operating_parameters=preserve_operating_parameters,
    )
    configuration = replace(
        configuration,
        sample_center_from_source_mm=float(state.sample.z_mm),
        objective=ObjectiveLayout(
            inner_face_gap_mm=state.objective_lens.inner_face_gap_mm,
            sample_axial_offset_mm=state.objective_lens.sample_axial_offset_mm,
            specimen_thickness_mm=state.sample.thickness_nm * 1.0e-6,
        ),
        resolved_assembly=state._resolved_assembly,
    )
    state.objective_aperture.validate()
    state.objective_lens._back_focal_plane_z_mm = (
        state.objective_lens.back_focal_plane_z_mm(
            state.beam_voltage_kv, state.sample
        )
    )
    state.objective_lens._image_plane_z_mm = (
        state.objective_lens.image_plane_z_mm(
            state.beam_voltage_kv, state.sample
        )
    )
    state.objective_image_plane_z_mm = (
        state.objective_lens._image_plane_z_mm
    )
    return configuration


def _by_key(items):
    return {item.key: item for item in items}


def _set_center(item, z_mm):
    item.z_mm = float(z_mm)


def _set_pair(pair, component, scale):
    start, end = component.rendered_z_range_mm
    pair.upper_z_mm = float(start * scale)
    pair.lower_z_mm = float(end * scale)


def _set_topology_installation(state, item, binding_key, installed):
    reference = getattr(state, "layout_reference_enabled", None)
    if not isinstance(reference, dict):
        reference = {}
        state.layout_reference_enabled = reference
    was_installed = bool(getattr(item, "_layout_installed", True))
    preference = bool(getattr(
        item,
        "_layout_enabled_preference",
        getattr(item, "enabled", True),
    ))
    if installed:
        if binding_key not in reference:
            reference[binding_key] = (
                bool(getattr(item, "enabled", True))
                if was_installed
                else preference
            )
        elif was_installed:
            # While installed, the user's latest enabled state becomes the
            # value restored after a later topology-driven removal.
            reference[binding_key] = bool(
                getattr(item, "enabled", True)
            )
        item._layout_enabled_preference = bool(reference[binding_key])
    else:
        if was_installed:
            preference = bool(getattr(item, "enabled", True))
            item._layout_enabled_preference = preference
            if binding_key in reference:
                reference[binding_key] = preference
        elif binding_key in reference:
            item._layout_enabled_preference = bool(reference[binding_key])
    item._layout_installed = bool(installed)
    item.enabled = (
        bool(reference[binding_key]) if installed else False
    )


def apply_physical_layout_to_state(
    state, *, preserve_operating_parameters=True
):
    """Resolve selected hardware topology onto the effective ray-tracing axis."""
    from temsim.column.module_assembly import (
        apply_column_manifest_geometry,
        apply_module_state_offsets,
        clear_module_state_offsets,
    )

    clear_module_state_offsets(state)
    configuration = layout_configuration_from_state(
        state,
        preserve_operating_parameters=preserve_operating_parameters,
    )
    state.corrector_mode = configuration.corrector.value
    state.probe_corrector_installed = configuration.corrector in (
        CorrectorAssembly.PROBE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    )
    state.image_corrector_installed = configuration.corrector in (
        CorrectorAssembly.IMAGE_CORRECTOR,
        CorrectorAssembly.DOUBLE_CORRECTOR,
    )
    state.layout_c3_hardware = configuration.c3_hardware.value
    state.layout_c3_excited = configuration.c3_excited

    layout = build_optics_layout(configuration)
    lenses = _by_key(state.lenses)
    apertures = _by_key(state.apertures)

    if CONDENSER_LENS_3 in lenses:
        lenses[CONDENSER_LENS_3].enabled = (
            configuration.c3_hardware is C3Hardware.THREE_CONDENSER
            and configuration.c3_excited
        )
    c3_installed = (
        configuration.c3_hardware is C3Hardware.THREE_CONDENSER
    )
    if CONDENSER_APERTURE_3 in apertures:
        _set_topology_installation(
            state,
            apertures[CONDENSER_APERTURE_3],
            f"aperture:{CONDENSER_APERTURE_3}",
            c3_installed,
        )

    probe_on = state.probe_corrector_installed
    if ADAPTER_LENS in lenses:
        _set_topology_installation(
            state,
            lenses[ADAPTER_LENS],
            f"lens:{ADAPTER_LENS}",
            probe_on,
        )
    if PROBE_TL22_LENS in lenses:
        _set_topology_installation(
            state,
            lenses[PROBE_TL22_LENS],
            f"lens:{PROBE_TL22_LENS}",
            probe_on,
        )
    for key in (PROBE_TL21_LENS, PROBE_TL12_LENS):
        if key in lenses:
            _set_topology_installation(
                state,
                lenses[key],
                f"lens:{key}",
                probe_on,
            )

    image_on = state.image_corrector_installed
    for key in IMAGE_CORRECTOR_LENS_KEYS:
        if key in lenses:
            _set_topology_installation(
                state,
                lenses[key],
                f"lens:{key}",
                image_on,
            )

    stigmators = _by_key(state.stigmators)
    deflectors = _by_key(state.deflectors)
    corrector_elements = _by_key(
        getattr(state, "corrector_elements", [])
    )
    for key in IMAGE_CORRECTOR_ELEMENT_KEYS:
        if key in corrector_elements:
            _set_topology_installation(
                state,
                corrector_elements[key],
                f"corrector:{key}",
                image_on,
            )
    if PROBE_DPH2_DEFLECTOR in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[PROBE_DPH2_DEFLECTOR],
            f"corrector:{PROBE_DPH2_DEFLECTOR}",
            probe_on,
        )
    if PROBE_DP22_DEFLECTOR in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[PROBE_DP22_DEFLECTOR],
            f"corrector:{PROBE_DP22_DEFLECTOR}",
            probe_on,
        )
    if PROBE_QPH2_QUADRUPOLE in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[PROBE_QPH2_QUADRUPOLE],
            f"corrector:{PROBE_QPH2_QUADRUPOLE}",
            probe_on,
        )
    if PROBE_HP2_HEXAPOLE in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[PROBE_HP2_HEXAPOLE],
            f"corrector:{PROBE_HP2_HEXAPOLE}",
            probe_on,
        )
    if PROBE_HPC_HEXAPOLE in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[PROBE_HPC_HEXAPOLE],
            f"corrector:{PROBE_HPC_HEXAPOLE}",
            probe_on,
        )
    for key in (
        PROBE_QPC_QUADRUPOLE,
        PROBE_DP21_DEFLECTOR,
        PROBE_DPH1_DEFLECTOR,
        PROBE_QPH1_QUADRUPOLE,
        PROBE_HP1_HEXAPOLE,
        PROBE_HPOL_HEXAPOLE,
        PROBE_QPOL_QUADRUPOLE,
        PROBE_DP11_DEFLECTOR,
    ):
        if key in corrector_elements:
            _set_topology_installation(
                state,
                corrector_elements[key],
                f"corrector:{key}",
                probe_on,
            )
    if PROBE_DP12_SCAN_DEFLECTOR in deflectors:
        _set_topology_installation(
            state,
            deflectors[PROBE_DP12_SCAN_DEFLECTOR],
            f"deflector:{PROBE_DP12_SCAN_DEFLECTOR}",
            probe_on,
        )
    if CONDENSER_DEFLECTOR in deflectors:
        _set_topology_installation(
            state,
            deflectors[CONDENSER_DEFLECTOR],
            f"deflector:{CONDENSER_DEFLECTOR}",
            c3_installed,
        )
    if OBJECTIVE_STIGMATOR in stigmators:
        _set_topology_installation(
            state,
            stigmators[OBJECTIVE_STIGMATOR],
            f"stigmator:{OBJECTIVE_STIGMATOR}",
            True,
        )
    if DIFFRACTION_STIGMATOR in stigmators:
        _set_topology_installation(
            state,
            stigmators[DIFFRACTION_STIGMATOR],
            f"stigmator:{DIFFRACTION_STIGMATOR}",
            True,
        )
    if DIFFRACTION_LENS in lenses:
        _set_topology_installation(
            state,
            lenses[DIFFRACTION_LENS],
            f"lens:{DIFFRACTION_LENS}",
            True,
        )
    if INTERMEDIATE_LENS in lenses:
        _set_topology_installation(
            state,
            lenses[INTERMEDIATE_LENS],
            f"lens:{INTERMEDIATE_LENS}",
            True,
        )
    if PROJECTOR_LENS_1 in lenses:
        _set_topology_installation(
            state,
            lenses[PROJECTOR_LENS_1],
            f"lens:{PROJECTOR_LENS_1}",
            True,
        )
    if PROJECTOR_LENS_2 in lenses:
        _set_topology_installation(
            state,
            lenses[PROJECTOR_LENS_2],
            f"lens:{PROJECTOR_LENS_2}",
            True,
        )
    if IMAGE_DIFFRACTION_DEFLECTOR in deflectors:
        _set_topology_installation(
            state,
            deflectors[IMAGE_DIFFRACTION_DEFLECTOR],
            f"deflector:{IMAGE_DIFFRACTION_DEFLECTOR}",
            True,
        )
    if DESCAN_DEFLECTOR in corrector_elements:
        _set_topology_installation(
            state,
            corrector_elements[DESCAN_DEFLECTOR],
            f"corrector:{DESCAN_DEFLECTOR}",
            True,
        )

    if (
        CONDENSER_LENS_3 in lenses
        and configuration.c3_hardware is C3Hardware.TWO_CONDENSER
    ):
        lenses[CONDENSER_LENS_3].enabled = False

    applied_source_offset = float(
        getattr(state, "monochromator_axis_offset_mm", 0.0)
    )
    mini_condenser = lenses.get(MINI_CONDENSER)
    if mini_condenser is not None:
        mini_condenser.enabled = True
        mini_condenser.validate()
    condenser_stigmator = stigmators.get(CONDENSER_STIGMATOR)
    if condenser_stigmator is not None:
        condenser_stigmator.validate()
        state.layout_reference_positions[
            f"stigmator:{CONDENSER_STIGMATOR}"
        ] = condenser_stigmator.optical_reference_z_mm
    objective_lens = lenses.get(OBJECTIVE_LENS)
    if objective_lens is not None:
        objective_lens.validate()
        state.layout_reference_positions[
            f"lens:{OBJECTIVE_LENS}"
        ] = (
            objective_lens.virtual_lens_reference_z_mm
            - applied_source_offset
        )
    objective_aperture = apertures.get(OBJECTIVE_APERTURE)
    if objective_aperture is not None:
        objective_aperture.validate()
        state.layout_reference_positions[
            f"aperture:{OBJECTIVE_APERTURE}"
        ] = objective_aperture.z_mm - applied_source_offset
    selected_area_aperture = apertures.get(SELECTED_AREA_APERTURE)
    if selected_area_aperture is not None:
        selected_area_aperture.validate()
    objective_stigmator = stigmators.get(OBJECTIVE_STIGMATOR)
    if objective_stigmator is not None:
        objective_stigmator.validate()
        state.layout_reference_positions.pop(
            f"stigmator:{OBJECTIVE_STIGMATOR}", None
        )
    diffraction_stigmator = stigmators.get(DIFFRACTION_STIGMATOR)
    if diffraction_stigmator is not None:
        diffraction_stigmator.validate()
        state.layout_reference_positions[
            f"stigmator:{DIFFRACTION_STIGMATOR}"
        ] = diffraction_stigmator.z_mm
    diffraction_lens = lenses.get(DIFFRACTION_LENS)
    if diffraction_lens is not None:
        diffraction_lens.validate()
    intermediate_lens = lenses.get(INTERMEDIATE_LENS)
    if intermediate_lens is not None:
        intermediate_lens.validate()
    projector_lens_p1 = lenses.get(PROJECTOR_LENS_1)
    if projector_lens_p1 is not None:
        projector_lens_p1.validate()
    projector_lens_p2 = lenses.get(PROJECTOR_LENS_2)
    if projector_lens_p2 is not None:
        projector_lens_p2.validate()
    for component in state.image_corrector_system.components:
        component.validate()
        prefix = (
            "lens"
            if component.key in IMAGE_CORRECTOR_LENS_KEYS
            else "corrector"
        )
        state.layout_reference_positions.pop(
            f"{prefix}:{component.key}", None
        )
    image_diffraction_deflector = deflectors.get(
        IMAGE_DIFFRACTION_DEFLECTOR
    )
    if image_diffraction_deflector is not None:
        image_diffraction_deflector.validate()
        state.layout_reference_positions.pop(
            f"deflector:{IMAGE_DIFFRACTION_DEFLECTOR}", None
        )
    descan_deflector = corrector_elements.get(DESCAN_DEFLECTOR)
    if descan_deflector is not None:
        descan_deflector.validate()
        state.layout_reference_positions[
            f"corrector:{DESCAN_DEFLECTOR}"
        ] = descan_deflector.optical_reference_z_mm

    state.electron_gun.validate()
    state.condenser_system.validate().apply_optical_positions()
    state.condenser_aperture_2.validate().apply_optical_position()
    state.condenser_aperture_3.validate().apply_optical_position()
    state.condenser_deflector.validate().apply_optical_positions()
    state.beam_deflector.validate().apply_optical_positions()
    state.probe_corrector_system.validate().apply_optical_positions()
    # Image-corrector optical interactions use the same OL-Post-anchored
    # coordinates as their mechanical components.
    state.image_corrector_system.validate()
    apply_module_state_offsets(
        state,
        configuration,
        layout,
        1.0,
    )
    state.objective_back_focal_plane_z_mm = (
        state.objective_lens.back_focal_plane_z_mm(
            state.beam_voltage_kv, state.sample
        )
    )
    state.objective_image_plane_z_mm = (
        state.objective_lens.image_plane_z_mm(
            state.beam_voltage_kv, state.sample
        )
    )
    layout_order = {
        component.key: index
        for index, component in enumerate(layout)
    }
    for collection_name in (
        "lenses",
        "apertures",
        "deflectors",
        "stigmators",
        "corrector_elements",
        "recording_planes",
    ):
        collection = getattr(state, collection_name, None)
        if isinstance(collection, list):
            collection.sort(
                key=lambda component: layout_order.get(
                    component.key,
                    len(layout_order),
                )
            )
    state._physical_layout_source_to_sample_mm = layout.source_to_sample_mm
    state._resolved_optics_layout = layout
    state._resolved_layout_configuration = configuration
    return layout


def state_topology_signature(state):
    """Return the canonical signature used by GUI view invalidation."""

    hardware = C3Hardware(getattr(
        state,
        "layout_c3_hardware",
        "three_condenser",
    ))
    corrector = _CORRECTORS.get(
        corrector_mode_for_hardware(
            getattr(state, "corrector_mode", "probe_corrector"),
            hardware.value,
        ),
        CorrectorAssembly.PROBE_CORRECTOR,
    )
    c3_excited = bool(getattr(
        state,
        "layout_c3_excited",
        getattr(state, "column_mode", "three_lens") == "three_lens",
    ))
    if hardware is C3Hardware.TWO_CONDENSER:
        c3_excited = False
    energy_filter = getattr(state, "energy_filter", None)
    configuration = LayoutConfiguration(
        corrector=corrector,
        electron_gun_type=state.electron_gun.type_key,
        c3_hardware=hardware,
        c3_excited=c3_excited,
        monochromator_installed=bool(
            state.electron_gun.type_key == "cold_feg"
            and getattr(state, "monochromator_installed", False)
        ),
        objective=ObjectiveLayout(
            inner_face_gap_mm=state.objective_lens.inner_face_gap_mm,
            sample_axial_offset_mm=getattr(
                state.objective_lens,
                "sample_axial_offset_mm",
                0.0,
            ),
            specimen_thickness_mm=state.sample.thickness_nm * 1.0e-6,
        ),
        energy_filter_selected=bool(
            getattr(energy_filter, "enabled", False)
            or getattr(
                state,
                "energy_filter_mode",
                "no_energy_filter",
            ) == "energy_filter"
        ),
    )
    return topology_signature(configuration)
