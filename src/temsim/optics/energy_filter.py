from dataclasses import asdict, dataclass
import math

from temsim import module_manifest
from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE
from temsim.optics.energy_filter_m12 import (
    MAXIMUM_MATCH_VOLTAGE_KV,
    MINIMUM_MATCH_VOLTAGE_KV,
    create_entrance_m12,
    create_exit_m12,
    energy_filter_m12_from_dict,
    magnetic_rigidity_t_m,
    rigidity_scale,
    serialise_energy_filter_m12,
)
from temsim.optics.energy_filter_slit import (
    create_energy_selection_slit,
    energy_selection_slit_from_dict,
    serialise_energy_selection_slit,
)

@dataclass
class EnergyFilterSystem:
    enabled: bool=False
    prism_radius_mm: float=150.0
    bend_angle_deg: float=90.0
    slit_width_ev: float=10.0
    selected_loss_ev: float=0.0
    entrance_multipole_s_mm: float=73.0
    prism_entrance_s_mm: float=92.0
    prism_fringe_mm: float=20.0
    pole_gap_mm: float=30.0
    sector_radial_aperture_mm: float=30.0
    ray_step_mm: float=0.25
    exit_multipole_d_mm: float=32.0
    slit_d_mm: float=205.0
    output_detector_d_mm: float=470.0
    output_detector_width_mm: float=57.344
    output_detector_inserted: bool=False
    eels_plane_offset_mm: float=5.0
    alignment_x_mrad: float=0.0
    alignment_y_mrad: float=0.0
    voltage_reference_kv: object=None
    matched_voltage_kv: object=None
    sector_reference_field_t: object=None
    sector_field_t: object=None
    entrance_m12: object=None
    exit_m12: object=None
    m12_frames_placed: bool=False
    energy_slit: object=None

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
            return float(
                getattr(self, "_pending_entrance_z_mm", 2169.0)
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
        )
    if ef.sector_field_t is None:
        ef.sector_field_t = float(ef.sector_reference_field_t)
    if ef.matched_voltage_kv is None:
        ef.matched_voltage_kv = reference_voltage
    if ef.entrance_m12 is None:
        ef.entrance_m12 = create_entrance_m12(reference_voltage)
    if ef.exit_m12 is None:
        ef.exit_m12 = create_exit_m12(reference_voltage)
    if not bool(ef.m12_frames_placed):
        from temsim.optics.energy_filter_sector import (
            place_m12_in_sector_frames,
        )

        place_m12_in_sector_frames(ef)
    if ef.energy_slit is None:
        ef.energy_slit = create_energy_selection_slit(
            centre_loss_ev=float(ef.selected_loss_ev),
            width_ev=float(ef.slit_width_ev),
            dispersion_um_per_ev=3.6,
            distance_from_sector_exit_m=(
                float(ef.slit_d_mm) * 1.0e-3
            ),
        )
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
    )
    scale = rigidity_scale(
        target_voltage,
        float(ef.voltage_reference_kv),
    )
    ef.sector_field_t = float(ef.sector_reference_field_t) * scale
    entrance_scale = ef.entrance_m12.apply_voltage_match(
        target_voltage
    )
    exit_scale = ef.exit_m12.apply_voltage_match(target_voltage)
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
        configure_energy_slit_from_software(ef)
    return EnergyFilterVoltageMatchResult(
        target_voltage_kv=target_voltage,
        rigidity_scale=scale,
        sector_field_t=float(ef.sector_field_t),
        entrance_m12_scale=entrance_scale,
        exit_m12_scale=exit_scale,
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
    data["entrance_m12"] = serialise_energy_filter_m12(
        energy_filter.entrance_m12
    )
    data["exit_m12"] = serialise_energy_filter_m12(
        energy_filter.exit_m12
    )
    data["energy_slit"] = serialise_energy_selection_slit(
        energy_filter.energy_slit
    )
    return data


def energy_filter_from_dict(values, source_voltage_kv):
    values = dict(values or {})
    known = EnergyFilterSystem.__dataclass_fields__
    scalar_values = {
        key: value
        for key, value in values.items()
        if key in known and key not in {
            "entrance_m12",
            "exit_m12",
            "energy_slit",
            *module_manifest.ENERGY_FILTER_GEOMETRY_FIELDS,
        }
    }
    energy_filter = EnergyFilterSystem(**scalar_values)
    reference_voltage = float(
        energy_filter.voltage_reference_kv
        if energy_filter.voltage_reference_kv is not None
        else source_voltage_kv
    )
    energy_filter.entrance_m12 = energy_filter_m12_from_dict(
        values.get("entrance_m12"),
        "entrance",
        reference_voltage,
    )
    energy_filter.exit_m12 = energy_filter_m12_from_dict(
        values.get("exit_m12"),
        "exit",
        reference_voltage,
    )
    energy_filter.energy_slit = energy_selection_slit_from_dict(
        values.get("energy_slit"),
        centre_loss_ev=float(energy_filter.selected_loss_ev),
        width_ev=float(energy_filter.slit_width_ev),
        dispersion_um_per_ev=3.6,
        distance_from_sector_exit_m=(
            float(energy_filter.slit_d_mm) * 1.0e-3
        ),
    )
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
