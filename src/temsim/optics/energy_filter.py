from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    ENERGY_FILTER_BIAS_TUBE,
    ENERGY_FILTER_CAMERA_DEFLECTOR,
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
    ENERGY_FILTER_ENTRANCE_APERTURE,
    ENERGY_FILTER_MULTIPOLE_KEYS,
    ENERGY_FILTER_SHUTTER,
    ENERGY_FILTER_SLIT,
    ENERGY_FILTER_TAPERED_PRISM,
    ENERGY_FILTER_ZEBRA,
)
from temsim.optics.energy_filter_m12 import (
    MAXIMUM_MATCH_VOLTAGE_KV,
    MINIMUM_MATCH_VOLTAGE_KV,
    create_iliad_multipoles,
    energy_filter_multipole_from_dict,
    energy_filter_m12_from_dict,
    magnetic_rigidity_t_m,
    rigidity_scale,
    serialise_energy_filter_m12,
)
from temsim.optics.energy_filter_detector import (
    EnergyFilterBiasTube,
    EnergyFilterCameraDeflector,
    EnergyFilterShutter,
    ZebraEELSDetector,
    detector_component_from_dict,
    serialise_detector_component,
)
from temsim.optics.energy_filter_slit import (
    create_energy_selection_slit,
    energy_selection_slit_from_dict,
    serialise_energy_selection_slit,
)


_ENERGY_FILTER_MODULE_PATH = (
    "project_and_recording_system/EnergyFilter.toml"
)
_DEFAULT_ENERGY_FILTER_INTERFACE = module_manifest.part_data(
    _ENERGY_FILTER_MODULE_PATH,
    "energy_filter",
)


def _manifest_part(key):
    return module_manifest.part_data(_ENERGY_FILTER_MODULE_PATH, key)


def _manifest_geometry_float(key, field):
    return float(_manifest_part(key)[field])


def _manifest_geometry_text(field):
    return str(_DEFAULT_ENERGY_FILTER_INTERFACE[field])


def _multipole_path_mm(index):
    return _manifest_geometry_float(
        ENERGY_FILTER_MULTIPOLE_KEYS[int(index) - 1],
        "path_center_mm",
    )


@dataclass
class EnergyFilterSystem:
    enabled: bool=False
    operating_mode: str="eels"
    optical_integration_enabled: bool=True
    multi_eels_enabled: bool=False
    multi_eels_region_count: int=1
    calibration_status: str="reference_calibration_non_oem_300kv"
    prism_radius_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "prism_radius_mm"
    )
    bend_angle_deg: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "bend_angle_deg"
    )
    prism_radial_field_index: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "prism_radial_field_index"
    )
    slit_width_ev: float=10.0
    selected_loss_ev: float=0.0
    entrance_multipole_s_mm: float = _multipole_path_mm(3)
    prism_entrance_s_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "path_entrance_mm"
    )
    prism_fringe_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "fringe_length_mm"
    )
    pole_gap_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "pole_gap_mm"
    )
    sector_radial_aperture_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_TAPERED_PRISM, "radial_clear_half_width_mm"
    )
    ray_step_mm: float=0.25
    maximum_trace_rays: int=256
    exit_multipole_d_mm: float = _multipole_path_mm(4)
    multipole_01_s_mm: float = _multipole_path_mm(1)
    multipole_02_s_mm: float = _multipole_path_mm(2)
    multipole_03_s_mm: float = _multipole_path_mm(3)
    multipole_04_d_mm: float = _multipole_path_mm(4)
    multipole_05_d_mm: float = _multipole_path_mm(5)
    multipole_06_d_mm: float = _multipole_path_mm(6)
    multipole_07_d_mm: float = _multipole_path_mm(7)
    multipole_08_d_mm: float = _multipole_path_mm(8)
    multipole_09_d_mm: float = _multipole_path_mm(9)
    multipole_10_d_mm: float = _multipole_path_mm(10)
    slit_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_SLIT, "path_center_mm"
    )
    dynamic_focus_quadrupole_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE, "path_center_mm"
    )
    bias_tube_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_BIAS_TUBE, "path_center_mm"
    )
    fast_shutter_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_SHUTTER, "path_center_mm"
    )
    camera_deflector_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_CAMERA_DEFLECTOR, "path_center_mm"
    )
    output_detector_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_EFTEM_OUTPUT_PLANE, "path_center_mm"
    )
    output_detector_width_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_EFTEM_OUTPUT_PLANE, "active_width_mm"
    )
    output_detector_inserted: bool=False
    zebra_detector_d_mm: float = _manifest_geometry_float(
        ENERGY_FILTER_ZEBRA, "path_center_mm"
    )
    eels_plane_offset_mm: float = (
        _manifest_geometry_float(ENERGY_FILTER_ZEBRA, "path_center_mm")
        - _manifest_geometry_float(
            ENERGY_FILTER_EFTEM_OUTPUT_PLANE, "path_center_mm"
        )
    )
    m12_housing_geometry_status: str = _manifest_geometry_text(
        "m12_housing_geometry_status"
    )
    m12_housing_geometry_source: str = _manifest_geometry_text(
        "m12_housing_geometry_source"
    )
    alignment_x_mrad: float=0.0
    alignment_y_mrad: float=0.0
    voltage_reference_kv: object=300.0
    matched_voltage_kv: object=None
    sector_reference_field_t: object=None
    sector_field_t: object=None
    entrance_m12: object=None
    exit_m12: object=None
    multipoles: object=None
    m12_frames_placed: bool=False
    energy_slit: object=None
    bias_tube: object=None
    fast_shutter: object=None
    camera_deflector: object=None
    zebra_detector: object=None

    @property
    def sector_soft_edges_enabled(self):
        """The completed sector always has finite C6 soft edges."""

        return True

    def bind_entrance_aperture(self, component):
        self._entrance_aperture_component = component
        if hasattr(self, "_pending_entrance_z_mm"):
            self.entrance_z_mm = self._pending_entrance_z_mm
            del self._pending_entrance_z_mm
        if hasattr(self, "_pending_entrance_aperture_mm"):
            self.entrance_aperture_mm = (
                self._pending_entrance_aperture_mm
            )
            del self._pending_entrance_aperture_mm
        return component

    @property
    def entrance_z_mm(self):
        component = getattr(
            self, "_entrance_aperture_component", None
        )
        if component is None:
            pending = getattr(self, "_pending_entrance_z_mm", None)
            if pending is not None:
                return float(pending)
            from temsim.optics.energy_filter_entrance_aperture import (
                ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION,
            )

            return float(
                ENERGY_FILTER_ENTRANCE_APERTURE_DEFINITION
                .create_component()
                .z_mm
            )
        return float(component.z_mm)

    @entrance_z_mm.setter
    def entrance_z_mm(self, value):
        component = getattr(
            self, "_entrance_aperture_component", None
        )
        if component is None:
            self._pending_entrance_z_mm = float(value)
            return
        selected_area_z_mm = (
            float(component.z_mm)
            - float(
                component
                .layout_center_downstream_of_anchor_mm
            )
        )
        component.set_optical_reference_z_mm(
            selected_area_z_mm,
            value,
        )

    @property
    def entrance_aperture_mm(self):
        component = getattr(
            self, "_entrance_aperture_component", None
        )
        if component is None:
            return float(
                getattr(
                    self, "_pending_entrance_aperture_mm", 5.0
                )
            )
        return 2.0 * float(component.radius_mm)

    @entrance_aperture_mm.setter
    def entrance_aperture_mm(self, value):
        diameter_mm = max(float(value), 0.0)
        component = getattr(
            self, "_entrance_aperture_component", None
        )
        if component is None:
            self._pending_entrance_aperture_mm = diameter_mm
            return
        component.radius_mm = min(
            diameter_mm / 2.0,
            float(component.maximum_radius_mm),
        )


@dataclass(frozen=True)
class EnergyFilterVoltageMatchResult:
    target_voltage_kv: float
    rigidity_scale: float
    sector_field_t: float
    entrance_m12_scale: float
    exit_m12_scale: float
    multipole_scales: tuple = ()
    slit_dispersion_um_per_ev: object = None
    diagnostic_message: str = ""


def _initialise_voltage_reference(ef, voltage_kv):
    voltage = float(voltage_kv)
    if ef.voltage_reference_kv is None:
        ef.voltage_reference_kv = voltage
    reference_voltage = float(ef.voltage_reference_kv)
    if ef.sector_reference_field_t is None:
        radius_m = float(ef.prism_radius_mm) * 1.0e-3
        if radius_m <= 0.0:
            raise ValueError("Sector magnet radius must be positive.")
        from temsim.optics.energy_filter_sector import (
            sector_plateau_field_t,
        )

        ef.sector_reference_field_t = sector_plateau_field_t(
            reference_voltage,
            radius_m,
            math.radians(float(ef.bend_angle_deg)),
            float(ef.prism_fringe_mm) * 1.0e-3,
            float(ef.prism_radial_field_index),
        )
    if ef.sector_field_t is None:
        ef.sector_field_t = float(ef.sector_reference_field_t)
    if ef.matched_voltage_kv is None:
        ef.matched_voltage_kv = reference_voltage
    if not isinstance(ef.multipoles, list) or len(ef.multipoles) != 10:
        ef.multipoles = create_iliad_multipoles(reference_voltage)
    # The legacy aliases remain readable for old profiles and external code;
    # they refer to the historical pre-prism and first post-prism locations.
    ef.entrance_m12 = ef.multipoles[2]
    ef.exit_m12 = ef.multipoles[3]
    if not bool(ef.m12_frames_placed):
        from temsim.optics.energy_filter_sector import (
            place_m12_in_sector_frames,
        )

        place_m12_in_sector_frames(ef)
    if ef.energy_slit is None:
        ef.energy_slit = create_energy_selection_slit(
            centre_loss_ev=float(ef.selected_loss_ev),
            width_ev=float(ef.slit_width_ev),
            dispersion_um_per_ev=0.7490714,
            distance_from_sector_exit_m=(
                float(ef.slit_d_mm) * 1.0e-3
            ),
        )
    if ef.bias_tube is None:
        ef.bias_tube = EnergyFilterBiasTube().validate()
    if ef.fast_shutter is None:
        ef.fast_shutter = EnergyFilterShutter().validate()
    if ef.camera_deflector is None:
        ef.camera_deflector = EnergyFilterCameraDeflector().validate()
    if ef.zebra_detector is None:
        ef.zebra_detector = ZebraEELSDetector().validate()
    configure_energy_filter_operating_mode(ef, ef.operating_mode)
    return ef


def ensure_energy_filter(state):
    created=not bool(getattr(state,'energy_filter',None))
    if created: state.energy_filter=EnergyFilterSystem()
    defaults=EnergyFilterSystem();ef=state.energy_filter
    for name in defaults.__dataclass_fields__:
        if not hasattr(ef,name):setattr(ef,name,getattr(defaults,name))
    if not hasattr(state,'energy_filter_mode'):state.energy_filter_mode='no_energy_filter'
    if not hasattr(state,'show_field_diagram'):state.show_field_diagram=True
    selected_area = getattr(state, "selected_area_aperture", None)
    aperture=next((
        item for item in getattr(state,'apertures',[])
        if item.key==ENERGY_FILTER_ENTRANCE_APERTURE
    ),None)
    if selected_area is not None and aperture is not None:
        aperture.resolve_against(selected_area.z_mm).validate()
        aperture.installed=bool(
            ef.enabled
            or getattr(state,'energy_filter_mode','no_energy_filter')
            == 'energy_filter'
        )
        ef.bind_entrance_aperture(aperture)
    _initialise_voltage_reference(
        ef,
        float(state.beam_voltage_kv),
    )
    return state


def match_energy_filter_to_voltage(state, voltage_kv=None):
    """Explicitly load the matched sector and M12 fields for one HT."""

    ensure_energy_filter(state)
    ef = state.energy_filter
    target_voltage = float(
        state.beam_voltage_kv
        if voltage_kv is None
        else voltage_kv
    )
    if not (
        MINIMUM_MATCH_VOLTAGE_KV
        <= target_voltage
        <= MAXIMUM_MATCH_VOLTAGE_KV
    ):
        raise ValueError(
            "Energy Filter voltage matching is supported from "
            f"{MINIMUM_MATCH_VOLTAGE_KV:g} to "
            f"{MAXIMUM_MATCH_VOLTAGE_KV:g} kV."
        )
    from temsim.optics.energy_filter_sector import (
        sector_plateau_field_t,
    )

    ef.sector_reference_field_t = sector_plateau_field_t(
        float(ef.voltage_reference_kv),
        float(ef.prism_radius_mm) * 1.0e-3,
        math.radians(float(ef.bend_angle_deg)),
        float(ef.prism_fringe_mm) * 1.0e-3,
        float(ef.prism_radial_field_index),
    )
    scale = rigidity_scale(
        target_voltage,
        float(ef.voltage_reference_kv),
    )
    ef.sector_field_t = float(ef.sector_reference_field_t) * scale
    multipole_scales = tuple(
        element.apply_voltage_match(target_voltage)
        for element in ef.multipoles
    )
    entrance_scale = multipole_scales[2]
    exit_scale = multipole_scales[3]
    ef.matched_voltage_kv = target_voltage
    from temsim.optics.energy_filter_metrics import (
        measure_slit_plane_metrics,
    )

    metrics = None
    diagnostic_message = ""
    try:
        metrics = measure_slit_plane_metrics(state)
    except RuntimeError as exc:
        diagnostic_message = str(exc)
    if metrics is not None:
        ef._last_slit_metrics = metrics
        ef.energy_slit.calibrated_dispersion_um_per_ev = (
            metrics.dispersion_um_per_ev
        )
        ef.energy_slit.zero_loss_offset_m = (
            metrics.reference_dispersive_um * 1.0e-6
        )
        configure_energy_slit_from_software(ef)
    return EnergyFilterVoltageMatchResult(
        target_voltage_kv=target_voltage,
        rigidity_scale=scale,
        sector_field_t=float(ef.sector_field_t),
        entrance_m12_scale=entrance_scale,
        exit_m12_scale=exit_scale,
        multipole_scales=multipole_scales,
        slit_dispersion_um_per_ev=(
            metrics.dispersion_um_per_ev
            if metrics is not None
            else None
        ),
        diagnostic_message=diagnostic_message,
    )


def energy_filter_voltage_match_status(state):
    ensure_energy_filter(state)
    current = float(state.beam_voltage_kv)
    matched = float(state.energy_filter.matched_voltage_kv)
    if math.isclose(current, matched, abs_tol=1.0e-12, rel_tol=0.0):
        return f"Energy Filter matched to {matched:g} kV"
    recommended = rigidity_scale(
        current,
        float(state.energy_filter.voltage_reference_kv),
    )
    return (
        f"HT is {current:g} kV; Energy Filter remains matched to "
        f"{matched:g} kV. Recommended rigidity scale "
        f"{recommended:.6g}; use Match filter to current HT."
    )


def serialise_energy_filter(energy_filter):
    if energy_filter is None:
        return None
    reference_voltage = (
        float(energy_filter.voltage_reference_kv)
        if energy_filter.voltage_reference_kv is not None
        else 300.0
    )
    _initialise_voltage_reference(
        energy_filter,
        reference_voltage,
    )
    data = asdict(energy_filter)
    for field in module_manifest.ENERGY_FILTER_GEOMETRY_FIELDS:
        data.pop(field, None)
    for field in module_manifest.ENERGY_FILTER_MECHANICAL_METADATA_FIELDS:
        data.pop(field, None)
    for field in (
        "multipoles", "entrance_m12", "exit_m12", "energy_slit",
        "bias_tube", "fast_shutter", "camera_deflector", "zebra_detector",
    ):
        data.pop(field, None)
    data["multipoles"] = [
        serialise_energy_filter_m12(element)
        for element in energy_filter.multipoles
    ]
    # Retain the legacy names as migration snapshots.
    data["entrance_m12"] = serialise_energy_filter_m12(
        energy_filter.entrance_m12
    )
    data["exit_m12"] = serialise_energy_filter_m12(
        energy_filter.exit_m12
    )
    data["energy_slit"] = serialise_energy_selection_slit(
        energy_filter.energy_slit
    )
    data["bias_tube"] = serialise_detector_component(
        energy_filter.bias_tube
    )
    data["fast_shutter"] = serialise_detector_component(
        energy_filter.fast_shutter
    )
    data["camera_deflector"] = serialise_detector_component(
        energy_filter.camera_deflector
    )
    data["zebra_detector"] = serialise_detector_component(
        energy_filter.zebra_detector
    )
    return data


def energy_filter_from_dict(values, source_voltage_kv):
    values = dict(values or {})
    known = EnergyFilterSystem.__dataclass_fields__
    scalar_values = {
        key: value
        for key, value in values.items()
        if key in known and key not in {
            "multipoles",
            "entrance_m12",
            "exit_m12",
            "energy_slit",
            "bias_tube",
            "fast_shutter",
            "camera_deflector",
            "zebra_detector",
            *module_manifest.ENERGY_FILTER_GEOMETRY_FIELDS,
            *module_manifest.ENERGY_FILTER_MECHANICAL_METADATA_FIELDS,
        }
    }
    energy_filter = EnergyFilterSystem(**scalar_values)
    reference_voltage = float(
        energy_filter.voltage_reference_kv
        if energy_filter.voltage_reference_kv is not None
        else source_voltage_kv
    )
    saved_multipoles = values.get("multipoles")
    if isinstance(saved_multipoles, list) and len(saved_multipoles) == 10:
        energy_filter.multipoles = [
            energy_filter_multipole_from_dict(
                item, index, reference_voltage
            )
            for index, item in enumerate(saved_multipoles, start=1)
        ]
    else:
        energy_filter.multipoles = create_iliad_multipoles(reference_voltage)
        # Migrate the former two-carrier state into the corresponding slots.
        for target, role, saved in (
            (
                energy_filter.multipoles[2],
                "entrance",
                values.get("entrance_m12"),
            ),
            (
                energy_filter.multipoles[3],
                "exit",
                values.get("exit_m12"),
            ),
        ):
            if not isinstance(saved, dict):
                continue
            legacy = energy_filter_m12_from_dict(
                saved, role, reference_voltage
            )
            legacy_values = np.concatenate((
                legacy.multipole_field.normal_coefficients,
                legacy.multipole_field.skew_coefficients,
                legacy.calibration.reference_normal_coefficients,
                legacy.calibration.reference_skew_coefficients,
            ))
            if np.any(np.abs(legacy_values) > 0.0):
                target.field_backend = legacy.field_backend
                target.calibration = legacy.calibration
                target.enabled = legacy.enabled
    energy_filter.energy_slit = energy_selection_slit_from_dict(
        values.get("energy_slit"),
        centre_loss_ev=float(energy_filter.selected_loss_ev),
        width_ev=float(energy_filter.slit_width_ev),
        dispersion_um_per_ev=0.7490714,
        distance_from_sector_exit_m=(
            float(energy_filter.slit_d_mm) * 1.0e-3
        ),
    )
    energy_filter.bias_tube = detector_component_from_dict(
        EnergyFilterBiasTube, values.get("bias_tube")
    )
    energy_filter.fast_shutter = detector_component_from_dict(
        EnergyFilterShutter, values.get("fast_shutter")
    )
    energy_filter.camera_deflector = detector_component_from_dict(
        EnergyFilterCameraDeflector, values.get("camera_deflector")
    )
    energy_filter.zebra_detector = detector_component_from_dict(
        ZebraEELSDetector, values.get("zebra_detector")
    )
    energy_filter.m12_frames_placed = False
    return _initialise_voltage_reference(
        energy_filter,
        source_voltage_kv,
    )

def configure_energy_slit_from_software(energy_filter):
    energy_filter.energy_slit.distance_from_sector_exit_m = (
        float(energy_filter.slit_d_mm) * 1.0e-3
    )
    return energy_filter.energy_slit.configure_energy_window(
        energy_filter.selected_loss_ev,
        energy_filter.slit_width_ev,
    )


ENERGY_FILTER_OPERATING_MODES = ("eels", "eftem")


def configure_energy_filter_operating_mode(energy_filter, mode):
    """Apply mutually consistent EELS or EFTEM detector/slit states."""

    mode = str(mode).strip().lower()
    if mode not in ENERGY_FILTER_OPERATING_MODES:
        raise ValueError("Energy Filter mode must be EELS or EFTEM.")
    energy_filter.operating_mode = mode
    if energy_filter.energy_slit is not None:
        energy_filter.energy_slit.inserted = mode == "eftem"
    energy_filter.output_detector_inserted = mode == "eftem"
    if energy_filter.zebra_detector is not None:
        energy_filter.zebra_detector.inserted = mode == "eels"
    if energy_filter.bias_tube is not None:
        energy_filter.bias_tube.enabled = (
            mode == "eels" and bool(energy_filter.multi_eels_enabled)
        )
    if energy_filter.fast_shutter is not None:
        energy_filter.fast_shutter.enabled = mode == "eels"
    if energy_filter.camera_deflector is not None:
        energy_filter.camera_deflector.enabled = mode == "eels"
    region_count = int(energy_filter.multi_eels_region_count)
    if not 1 <= region_count <= 5:
        raise ValueError("MultiEELS region count must be 1 through 5.")
    if not energy_filter.multi_eels_enabled:
        energy_filter.multi_eels_region_count = 1
    if energy_filter.camera_deflector is not None:
        energy_filter.camera_deflector.active_strip = min(
            int(energy_filter.camera_deflector.active_strip),
            int(energy_filter.multi_eels_region_count),
        )
    return energy_filter
