"""Resolve a mechanical column topology onto the effective ray-tracing axis.

The mechanical layout owns component order, envelopes and relative clearances.
The effective axis keeps the existing calibrated probe-corrector coordinates as
its reference and applies topology-derived displacements to every bound field,
stop and recording plane.  This avoids treating provisional mechanical
dimensions as absolute electron-optical distances while still making a changed
physical topology change the simulated drift distances.
"""

from dataclasses import dataclass

from temsim.column.layout import (
    C3Hardware,
    CorrectorAssembly,
    LayoutConfiguration,
    build_optics_layout,
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
_REFERENCE_SAMPLE_Z_MM = 935.0
# Compatibility calibration for the retired effective-axis adapter. Keep it
# independent of the live TOML assembly so importing this legacy module never
# reads or snapshots current mechanical geometry.
_LEGACY_REFERENCE_SOURCE_TO_SAMPLE_MM = 1599.2
MECHANICAL_TO_EFFECTIVE_SCALE = (
    _REFERENCE_SAMPLE_Z_MM / _LEGACY_REFERENCE_SOURCE_TO_SAMPLE_MM
)


_COMPONENT_BINDINGS = {
    CONDENSER_LENS_1: (f"lens:{CONDENSER_LENS_1}",),
    CONDENSER_LENS_2: (f"lens:{CONDENSER_LENS_2}",),
    CONDENSER_DEFLECTOR: (f"deflector:{CONDENSER_DEFLECTOR}",),
    CONDENSER_LENS_3: (f"lens:{CONDENSER_LENS_3}",),
    ADAPTER_LENS: (f"lens:{ADAPTER_LENS}",),
    PROBE_DPH2_DEFLECTOR: (
        f"corrector:{PROBE_DPH2_DEFLECTOR}",
    ),
    PROBE_QPH2_QUADRUPOLE: (
        f"corrector:{PROBE_QPH2_QUADRUPOLE}",
    ),
    PROBE_HP2_HEXAPOLE: (f"corrector:{PROBE_HP2_HEXAPOLE}",),
    PROBE_HPC_HEXAPOLE: (f"corrector:{PROBE_HPC_HEXAPOLE}",),
    PROBE_TL22_LENS: (f"lens:{PROBE_TL22_LENS}",),
    PROBE_DP22_DEFLECTOR: (
        f"corrector:{PROBE_DP22_DEFLECTOR}",
    ),
    PROBE_QPC_QUADRUPOLE: (f"corrector:{PROBE_QPC_QUADRUPOLE}",),
    PROBE_DP21_DEFLECTOR: (f"corrector:{PROBE_DP21_DEFLECTOR}",),
    PROBE_TL21_LENS: (f"lens:{PROBE_TL21_LENS}",),
    PROBE_DPH1_DEFLECTOR: (f"corrector:{PROBE_DPH1_DEFLECTOR}",),
    PROBE_QPH1_QUADRUPOLE: (
        f"corrector:{PROBE_QPH1_QUADRUPOLE}",
    ),
    PROBE_HP1_HEXAPOLE: (f"corrector:{PROBE_HP1_HEXAPOLE}",),
    PROBE_HPOL_HEXAPOLE: (f"corrector:{PROBE_HPOL_HEXAPOLE}",),
    PROBE_QPOL_QUADRUPOLE: (
        f"corrector:{PROBE_QPOL_QUADRUPOLE}",
    ),
    PROBE_DP11_DEFLECTOR: (f"corrector:{PROBE_DP11_DEFLECTOR}",),
    PROBE_TL12_LENS: (f"lens:{PROBE_TL12_LENS}",),
    DESCAN_DEFLECTOR: (f"corrector:{DESCAN_DEFLECTOR}",),
    **{
        key: (f"lens:{key}",)
        for key in IMAGE_CORRECTOR_LENS_KEYS
    },
    **{
        key: (f"corrector:{key}",)
        for key in IMAGE_CORRECTOR_ELEMENT_KEYS
    },
    DIFFRACTION_LENS: (f"lens:{DIFFRACTION_LENS}",),
    INTERMEDIATE_LENS: (f"lens:{INTERMEDIATE_LENS}",),
    PROJECTOR_LENS_1: (f"lens:{PROJECTOR_LENS_1}",),
    PROJECTOR_LENS_2: (f"lens:{PROJECTOR_LENS_2}",),
    HAADF_DETECTOR: (f"recording:{HAADF_DETECTOR}",),
    FLUORESCENT_SCREEN: (f"recording:{FLUORESCENT_SCREEN}",),
    DARK_FIELD_DETECTOR: (f"recording:{DARK_FIELD_DETECTOR}",),
    BRIGHT_FIELD_DETECTOR: (f"recording:{BRIGHT_FIELD_DETECTOR}",),
    CAMERA: (f"recording:{CAMERA}",),
    ENERGY_FILTER_ENTRANCE_APERTURE: (
        f"aperture:{ENERGY_FILTER_ENTRANCE_APERTURE}",
    ),
}

@dataclass(frozen=True)
class EffectiveAxisResolution:
    topology_signature: tuple
    mechanical_to_effective_scale: float
    sample_shift_mm: float
    positions_mm: dict


class _Binding:
    def __init__(self, getter, setter):
        self.get = getter
        self.set = setter


def topology_signature(configuration):
    objective = configuration.objective
    return (
        configuration.electron_gun_type,
        configuration.corrector.value,
        configuration.c3_hardware.value,
        bool(configuration.c3_excited),
        bool(configuration.monochromator_installed),
        objective.pole_piece_type,
        float(objective.inner_face_gap_mm),
        float(objective.sample_axial_offset_mm),
        bool(configuration.energy_filter_selected),
    )


def _attribute_binding(item, attribute):
    return _Binding(
        lambda item=item, attribute=attribute: float(getattr(item, attribute)),
        lambda value, item=item, attribute=attribute: setattr(
            item, attribute, float(value)
        ),
    )


def _pair_binding(pair):
    if hasattr(pair, "optical_center_from_tip_mm"):
        return _attribute_binding(pair, "optical_center_from_tip_mm")
    if hasattr(pair, "optical_center_z_mm"):
        return _attribute_binding(pair, "optical_center_z_mm")

    def get():
        return (float(pair.upper_z_mm) + float(pair.lower_z_mm)) / 2.0

    def set_center(value):
        half_span = abs(
            float(pair.lower_z_mm) - float(pair.upper_z_mm)
        ) / 2.0
        pair.upper_z_mm = float(value) - half_span
        pair.lower_z_mm = float(value) + half_span

    return _Binding(get, set_center)


def _bindings(state):
    result = {}
    ac_anchored_keys = {
        AC_DEFLECTOR,
        BEAM_DEFLECTOR,
        CONDENSER_DEFLECTOR,
        MINI_CONDENSER,
        CONDENSER_STIGMATOR,
        OBJECTIVE_LENS,
        OBJECTIVE_APERTURE,
        OBJECTIVE_STIGMATOR,
        IMAGE_DIFFRACTION_DEFLECTOR,
        DESCAN_DEFLECTOR,
    }
    probe_corrector_keys = {
        ADAPTER_LENS,
        PROBE_DPH2_DEFLECTOR,
        PROBE_QPH2_QUADRUPOLE,
        PROBE_HP2_HEXAPOLE,
        PROBE_TL22_LENS,
        PROBE_DP22_DEFLECTOR,
        PROBE_HPC_HEXAPOLE,
        PROBE_QPC_QUADRUPOLE,
        PROBE_DP21_DEFLECTOR,
        PROBE_TL21_LENS,
        PROBE_DPH1_DEFLECTOR,
        PROBE_QPH1_QUADRUPOLE,
        PROBE_HP1_HEXAPOLE,
        PROBE_HPOL_HEXAPOLE,
        PROBE_QPOL_QUADRUPOLE,
        PROBE_DP11_DEFLECTOR,
        PROBE_TL12_LENS,
        PROBE_DP12_SCAN_DEFLECTOR,
    }
    image_corrector_keys = {
        *IMAGE_CORRECTOR_LENS_KEYS,
        *IMAGE_CORRECTOR_ELEMENT_KEYS,
    }
    collections = (
        ("lens", state.lenses),
        ("aperture", state.apertures),
        ("stigmator", state.stigmators),
        ("corrector", getattr(state, "corrector_elements", [])),
        ("recording", getattr(state, "recording_planes", [])),
    )
    for prefix, items in collections:
        for item in items:
            if (
                item.key in ac_anchored_keys
                or item.key in probe_corrector_keys
                or item.key in image_corrector_keys
            ):
                # These components use the condenser-stigmator package,
                # condenser-aperture/ADL, or Image Corrector OL-Post anchor
                # directly; an effective-axis binding would create a second
                # coordinate definition.
                continue
            if (
                prefix == "aperture"
                and item.key in {
                    CONDENSER_APERTURE_2,
                    CONDENSER_APERTURE_3,
                }
            ):
                # These aperture planes now use their mechanical coordinate
                # directly; the effective-axis resolver must not create a
                # second position.
                continue
            if (
                prefix == "aperture"
                and item.key == SELECTED_AREA_APERTURE
            ):
                # This component owns two topology-specific calibrated image
                # planes and is resolved explicitly below.
                continue
            if (
                prefix == "stigmator"
                and item.key == DIFFRACTION_STIGMATOR
            ):
                # Two saved installation calibrations are resolved below.
                continue
            if prefix == "lens" and item.key == DIFFRACTION_LENS:
                # Two saved installation calibrations are resolved below.
                continue
            if prefix == "lens" and item.key == INTERMEDIATE_LENS:
                # This lens follows the Selected Area Aperture.
                continue
            if prefix == "lens" and item.key == PROJECTOR_LENS_1:
                # This lens follows the Selected Area Aperture.
                continue
            if prefix == "lens" and item.key == PROJECTOR_LENS_2:
                # This lens follows the Selected Area Aperture.
                continue
            if (
                prefix == "recording"
                and item.key in {
                    HAADF_DETECTOR,
                    DARK_FIELD_DETECTOR,
                    BRIGHT_FIELD_DETECTOR,
                    FLUORESCENT_SCREEN,
                    CAMERA,
                }
            ):
                # These recording planes follow the Selected Area Aperture.
                continue
            if (
                prefix == "aperture"
                and item.key == ENERGY_FILTER_ENTRANCE_APERTURE
            ):
                # This stop follows the Selected Area Aperture.
                continue
            result[f"{prefix}:{item.key}"] = _attribute_binding(
                item, "z_mm"
            )
    for pair in state.deflectors:
        if (
            pair.key in ac_anchored_keys
            or pair.key in probe_corrector_keys
            or pair.key in image_corrector_keys
        ):
            # These paired deflectors already derive both kick planes from
            # their canonical mechanical anchor. Applying an effective-axis
            # binding here would translate their optical planes a second time.
            continue
        result[f"deflector:{pair.key}"] = _pair_binding(pair)
    return result


def _target_components(layout):
    targets = {}
    for component in layout:
        for binding_key in _COMPONENT_BINDINGS.get(component.key, ()):
            targets[binding_key] = component
    return targets


def _source_distance(layout, component):
    return float(layout.source_to_sample_mm - component.local_s_center_mm)


def _reference_configuration(configuration):
    return LayoutConfiguration(
        corrector=CorrectorAssembly.PROBE_CORRECTOR,
        electron_gun_type=configuration.electron_gun_type,
        c3_hardware=C3Hardware.THREE_CONDENSER,
        c3_excited=True,
        monochromator_installed=configuration.monochromator_installed,
        sample_center_from_source_mm=(
            configuration.sample_center_from_source_mm
        ),
        objective=configuration.objective,
        energy_filter_selected=configuration.energy_filter_selected,
        # Mechanical gun edits are drawing data, not a source of optical-axis
        # displacement. Use the same gun geometry on both sides of the
        # topology comparison.
        gun_components=configuration.gun_components,
        condenser_components=configuration.condenser_components,
        condenser_aperture_2_component=(
            configuration.condenser_aperture_2_component
        ),
        condenser_aperture_3_component=(
            configuration.condenser_aperture_3_component
        ),
        condenser_deflector_component=(
            configuration.condenser_deflector_component
        ),
        beam_deflector_component=configuration.beam_deflector_component,
        ac_deflector_component=configuration.ac_deflector_component,
        mini_condenser_component=configuration.mini_condenser_component,
        condenser_stigmator_component=(
            configuration.condenser_stigmator_component
        ),
        diffraction_stigmator_component=(
            configuration.diffraction_stigmator_component
        ),
        diffraction_lens_component=(
            configuration.diffraction_lens_component
        ),
        intermediate_lens_component=(
            configuration.intermediate_lens_component
        ),
        projector_lens_p1_component=(
            configuration.projector_lens_p1_component
        ),
        projector_lens_p2_component=(
            configuration.projector_lens_p2_component
        ),
        stem_detector_components=configuration.stem_detector_components,
        fluorescent_screen_component=(
            configuration.fluorescent_screen_component
        ),
        camera_component=configuration.camera_component,
        energy_filter_entrance_aperture_component=(
            configuration.energy_filter_entrance_aperture_component
        ),
        objective_lens_component=configuration.objective_lens_component,
        objective_aperture_component=(
            configuration.objective_aperture_component
        ),
        selected_area_aperture_component=(
            configuration.selected_area_aperture_component
        ),
        objective_stigmator_component=(
            configuration.objective_stigmator_component
        ),
        image_diffraction_deflector_component=(
            configuration.image_diffraction_deflector_component
        ),
        descan_deflector_component=(
            configuration.descan_deflector_component
        ),
        adapter_lens_component=configuration.adapter_lens_component,
        dph2_deflector_component=configuration.dph2_deflector_component,
        dp22_deflector_component=configuration.dp22_deflector_component,
        qph2_quadrupole_component=(
            configuration.qph2_quadrupole_component
        ),
        hp2_hexapole_component=configuration.hp2_hexapole_component,
        hpc_hexapole_component=configuration.hpc_hexapole_component,
        probe_corrector_tail_components=(
            configuration.probe_corrector_tail_components
        ),
        tl22_lens_component=configuration.tl22_lens_component,
        image_corrector_components=configuration.image_corrector_components,
    )


def apply_effective_axis(state, configuration, layout):
    """Move every bound physical interaction onto the resolved active topology."""

    source_axis_offset = float(
        getattr(state, "monochromator_column_offset_mm", 0.0)
    )
    previously_applied_source_offset = float(
        getattr(state, "monochromator_axis_offset_mm", 0.0)
    )
    reference_layout = build_optics_layout(_reference_configuration(configuration))
    current_targets = _target_components(layout)
    reference_targets = _target_components(reference_layout)
    bindings = _bindings(state)

    reference_positions = getattr(state, "layout_reference_positions", None)
    if not isinstance(reference_positions, dict):
        reference_positions = {}
        state.layout_reference_positions = reference_positions
    for key, binding in bindings.items():
        reference_positions.setdefault(
            key,
            binding.get() - previously_applied_source_offset,
        )

    sample_delta = (
        float(layout.source_to_sample_mm)
        - float(reference_layout.source_to_sample_mm)
    ) * MECHANICAL_TO_EFFECTIVE_SCALE
    p2_binding = f"lens:{PROJECTOR_LENS_2}"
    current_p2 = current_targets.get(p2_binding)
    reference_p2 = reference_targets.get(p2_binding)
    downstream_delta = sample_delta
    if current_p2 is not None and reference_p2 is not None:
        downstream_delta = (
            _source_distance(layout, current_p2)
            - _source_distance(reference_layout, reference_p2)
        ) * MECHANICAL_TO_EFFECTIVE_SCALE

    resolved = {}
    for key, binding in bindings.items():
        current_component = current_targets.get(key)
        reference_component = reference_targets.get(key)
        if current_component is not None and reference_component is not None:
            mechanical_delta = (
                _source_distance(layout, current_component)
                - _source_distance(reference_layout, reference_component)
            )
            delta = mechanical_delta * MECHANICAL_TO_EFFECTIVE_SCALE
        else:
            # Optional/disabled components still follow the specimen station.
            # When they return to the active topology their component-specific
            # placement overrides this fallback.
            delta = (
                downstream_delta
                if key.startswith(("recording:", "energy_filter:"))
                else sample_delta
            )
        value = (
            float(reference_positions[key])
            + float(delta)
            + source_axis_offset
        )
        binding.set(value)
        resolved[key] = value

    selected_area_aperture = getattr(
        state, "selected_area_aperture", None
    )
    if selected_area_aperture is not None:
        geometry = selected_area_aperture.geometry_for(
            selected_area_aperture.active_installation
        )
        value = float(geometry.optical_reference_z_mm)
        object.__setattr__(selected_area_aperture, "z_mm", value)
        key = f"aperture:{SELECTED_AREA_APERTURE}"
        reference_positions[key] = value
        resolved[key] = value

    diffraction_stigmator = getattr(
        state, "diffraction_stigmator", None
    )
    if (
        diffraction_stigmator is not None
        and selected_area_aperture is not None
    ):
        value = (
            float(selected_area_aperture.z_mm)
            + float(
                diffraction_stigmator
                .mechanical_center_downstream_of_anchor_mm
            )
        )
        object.__setattr__(diffraction_stigmator, "z_mm", value)
        key = f"stigmator:{DIFFRACTION_STIGMATOR}"
        reference_positions[key] = value - source_axis_offset
        resolved[key] = value

    diffraction_lens = getattr(state, "diffraction_lens", None)
    if (
        diffraction_lens is not None
        and selected_area_aperture is not None
    ):
        value = (
            float(selected_area_aperture.z_mm)
            + float(
                diffraction_lens
                .mechanical_center_downstream_of_anchor_mm
            )
        )
        object.__setattr__(diffraction_lens, "z_mm", value)
        key = f"lens:{DIFFRACTION_LENS}"
        reference_positions[key] = value - source_axis_offset
        resolved[key] = value

    intermediate_lens = getattr(state, "intermediate_lens", None)
    if (
        intermediate_lens is not None
        and selected_area_aperture is not None
    ):
        value = (
            float(selected_area_aperture.z_mm)
            + float(
                intermediate_lens
                .mechanical_center_downstream_of_anchor_mm
            )
        )
        object.__setattr__(intermediate_lens, "z_mm", value)
        key = f"lens:{INTERMEDIATE_LENS}"
        reference_positions[key] = value - source_axis_offset
        resolved[key] = value

    projector_lens_p1 = getattr(state, "projector_lens_p1", None)
    if (
        projector_lens_p1 is not None
        and selected_area_aperture is not None
    ):
        value = (
            float(selected_area_aperture.z_mm)
            + float(
                projector_lens_p1
                .mechanical_center_downstream_of_anchor_mm
            )
        )
        object.__setattr__(projector_lens_p1, "z_mm", value)
        key = f"lens:{PROJECTOR_LENS_1}"
        reference_positions[key] = value - source_axis_offset
        resolved[key] = value

    projector_lens_p2 = getattr(state, "projector_lens_p2", None)
    if (
        projector_lens_p2 is not None
        and selected_area_aperture is not None
    ):
        value = (
            float(selected_area_aperture.z_mm)
            + float(
                projector_lens_p2
                .mechanical_center_downstream_of_anchor_mm
            )
        )
        object.__setattr__(projector_lens_p2, "z_mm", value)
        key = f"lens:{PROJECTOR_LENS_2}"
        reference_positions[key] = value - source_axis_offset
        resolved[key] = value

    if selected_area_aperture is not None:
        for detector in getattr(state, "stem_detectors", ()):
            detector.resolve_against(selected_area_aperture.z_mm)
            key = f"recording:{detector.key}"
            reference_positions[key] = (
                float(detector.z_mm) - source_axis_offset
            )
            resolved[key] = float(detector.z_mm)
        fluorescent_screen = getattr(state, "fluorescent_screen", None)
        if fluorescent_screen is not None:
            fluorescent_screen.resolve_against(
                selected_area_aperture.z_mm
            )
            key = f"recording:{FLUORESCENT_SCREEN}"
            reference_positions[key] = (
                float(fluorescent_screen.z_mm) - source_axis_offset
            )
            resolved[key] = float(fluorescent_screen.z_mm)
        camera = getattr(state, "camera", None)
        if camera is not None:
            camera.resolve_against(selected_area_aperture.z_mm)
            key = f"recording:{CAMERA}"
            reference_positions[key] = (
                float(camera.z_mm) - source_axis_offset
            )
            resolved[key] = float(camera.z_mm)
            entrance_aperture = getattr(
                state, "energy_filter_entrance_aperture", None
            )
            if entrance_aperture is not None:
                entrance_aperture.resolve_against(
                    selected_area_aperture.z_mm
                )
                key = (
                    f"aperture:"
                    f"{ENERGY_FILTER_ENTRANCE_APERTURE}"
                )
                reference_positions[key] = float(
                    entrance_aperture.z_mm
                ) - source_axis_offset
                resolved[key] = float(entrance_aperture.z_mm)

    state.corrector_crossover_targets_mm = (
        [
            float(state.adapter_lens.z_mm),
            float(state.tl22_lens.z_mm),
            float(state.tl12_lens.z_mm),
        ]
        if configuration.corrector
        in (CorrectorAssembly.PROBE_CORRECTOR, CorrectorAssembly.DOUBLE_CORRECTOR)
        else []
    )
    resolution = EffectiveAxisResolution(
        topology_signature(configuration),
        MECHANICAL_TO_EFFECTIVE_SCALE,
        sample_delta,
        resolved,
    )
    state._effective_axis_resolution = resolution
    state.monochromator_axis_offset_mm = source_axis_offset
    return resolution
