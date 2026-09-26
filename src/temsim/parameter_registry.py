"""Small adapter-backed parameter registry; existing validators remain owners.

Reuse explanations can broaden existing invalidation, never narrow it. Unknown
graph changes invalidate the plan conservatively. No field or particle solve is
performed while classifying parameters or comparing dependency signatures.
"""
from dataclasses import dataclass
import re

from temsim.component_keys import (
    APERTURE_KEYS, PROBE_CORRECTOR_KEYS, IMAGE_CORRECTOR_KEYS,
    CAMERA, FLUORESCENT_SCREEN, STEM_DETECTOR_KEYS,
)
from temsim.parameter_semantics import parameter_unit


# Existing dynamic aliases/readbacks already have scoped cache owners. Anything
# else added to a public model object is conservatively an unclassified input.
_KNOWN_DYNAMIC = {
    "State": {"corrector_elements", "recording_planes", "energy_filter_installed",
              "image_corrector_installed", "probe_corrector_installed", "show_field_diagram",
              "objective_back_focal_plane_z_mm", "objective_image_plane_z_mm", "simulation_time_s"},
    "EnergyFilterSystem": {"dynamic_focus_quadrupole_bore_mm", "dynamic_focus_quadrupole_geometry_source",
        "dynamic_focus_quadrupole_geometry_status", "dynamic_focus_quadrupole_length_mm",
        "dynamic_focus_quadrupole_model_status", "dynamic_focus_quadrupole_outer_mm",
        "output_plane_geometry_source", "output_plane_geometry_status"},
    "Aperture": {"maximum_radius_mm"},
}
for _lens_type in ("CondenserLensState", "AdapterLensComponent", "Tl22LensComponent",
        "Tl21LensComponent", "Tl12LensComponent", "MiniCondenserComponent",
        "ObjectiveLensComponent", "DiffractionLensComponent", "IntermediateLensComponent",
        "ProjectorLensP1Component", "ProjectorLensP2Component"):
    _KNOWN_DYNAMIC[_lens_type] = {"field_polarity_source", "field_polarity_status",
                                "field_calibration_source", "field_calibration_status"}


def unmapped_public_inputs(state):
    """Exact extension identity used by real caches, not just UI explanations."""
    supplied = getattr(state, "_captured_unmapped_inputs", None)
    if supplied is not None:
        return supplied
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from temsim.instrument_snapshot import _RUNTIME_NAMES, _STATE_PRODUCT_NAMES, encode_instrument
    from temsim.immutable_json import json_digest
    visited, extensions = set(), {}
    assets = None

    def extension_identity(item):
        nonlocal assets
        if assets is None:
            from temsim.input_assets import INPUT_ASSETS
            assets = INPUT_ASSETS.capture()
        # Already-admitted immutable arrays contribute exact content references,
        # without undoing lightweight capture by creating a full inline hex copy.
        return json_digest(encode_instrument(item, asset_store=assets))

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{path}[{key}]")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif is_dataclass(value) and not isinstance(value, type):
            declared = {field.name for field in fields(value)}
            attributes = {name: getattr(value, name) for name in declared}
            if hasattr(value, "__dict__"):
                attributes.update(vars(value))
            attributes = {name: item for name, item in attributes.items()
                if not name.startswith("_") and name not in _RUNTIME_NAMES
                and not (type(value).__name__ == "State" and name in _STATE_PRODUCT_NAMES)}
            known = _KNOWN_DYNAMIC.get(type(value).__name__, set())
            for name, item in sorted(attributes.items()):
                if name not in declared and name not in known:
                    extensions[f"{path}.{name}"] = extension_identity(item)
                visit(item, f"{path}.{name}")
    try:
        visit(state, "instrument")
    finally:
        if assets is not None:
            assets.close()
    return extensions


@dataclass(frozen=True)
class ParameterDefinition:
    component: str
    name: str
    category: str
    unit: str
    label: str
    description: str
    sweep_eligible: bool = False
    adapter: str = "existing runtime validator"

    @property
    def identity(self):
        return f"{self.component}.{self.name}"

    def tooltip(self, *, enabled=True, surface_source=False):
        status = "Active in current model"
        if self.category == "metadata":
            status = "Metadata only; no active transport effect"
        elif self.category == "presentation":
            status = "Display geometry only"
        elif self.category == "inverse_target":
            status = "Derived readout; explicit Direct Alignment required"
        elif not enabled or (surface_source and self.component == "feg_tip" and self.name != "ray_count"):
            status = "Stored but inactive"
        elif self.category == "execution":
            status = "Numerical control; not a physical source setting"
        elif self.category == "calibration":
            status = "Saved calibration; consumed in held mode"
        elif self.component in _HEXAPOLE_KEYS:
            status = "Active in analytical transport when enabled; inactive in ideal mode"
        return (f"{self.label} | {self.category} | {self.unit or 'dimensionless'}\n{status}\n"
                f"{self.description}\nSweep: {'eligible through existing validator' if self.sweep_eligible else 'not in the validated sweep pilot'}")


_TIP_RESPONSES = {
    "emission_current_na": "Prescribed emitted current at the tip. With fixed emission shape and fields, current scales particle contributions, not their trajectories; self-consistent space charge is not solved.",
    "virtual_source_fwhm_nm": "Width of the prescribed spatial distribution at the physical tip, despite the historical field name. It is not an independently placed virtual or downstream source. Wider emission need not produce a wider beam at every later plane.",
    "angular_rms_mrad": "Scale of the local angular distribution before its cutoff, not necessarily the RMS of the truncated sample. Hold the cutoff and tip geometry fixed when comparing angular responses.",
    "angular_cutoff_mrad": "Maximum local angular support. Increasing it can admit larger launch angles; transmitted current depends on subsequent focusing and clipping.",
    "energy_spread_fwhm_ev": "RMS-equivalent FWHM of the prescribed positive launch-energy distribution, not necessarily its measured spectral FWHM. Hold mean energy and energy bounds fixed; this does not set an independent exit energy.",
    "emission_energy_ev": "Mean kinetic energy at emission, before extraction and acceleration. Static electric transport must retain K - e*potential for each electron; exit energy also includes the potential rise.",
    "minimum_kinetic_energy_ev": "Lower bound of the positive launch-energy distribution. Changing it can reshape that distribution; it is not a downstream energy cutoff.",
    "energy_half_range_ev": "Bound on the prescribed launch-energy offsets. Changing this bound can reshape the spectrum; it is not an energy-filter slit.",
    "young_decay_width_ev": "One-sided launch-energy tail shape coefficient. The resulting distribution is rescaled to the requested mean and RMS-equivalent width; this parameter alone does not set its final width.",
    "boersch_sigma_ev": "Prescribed Gaussian contribution to the launch-energy shape. This is not a self-consistent electron-electron interaction calculation; the combined shape is rescaled to the requested width.",
    "curvature_nm_inv": "Curvature of the continuous analytic tip model. It changes local surface normals and, in the current model, launch positions; zero is the planar limit. It does not solve a new electrode boundary or change the prescribed total current.",
    "ray_count": "Numerical samples of the same tip distribution. Increasing the count refines sampling and must not multiply emitted current. Gun and column step sizes require separate convergence checks.",
}
_TIP_METADATA = {
    "tip_radius_nm": "Historical tip-radius metadata. The analytic launch uses curvature_nm_inv; the solved surface model owns a separate physical geometry table edited through Physical Layout.",
    "tip_cone_half_angle_deg": "Historical cone metadata. Edit the installed tip geometry in Physical Layout for the connected surface-field model; this scalar does not reshape analytic launch trajectories.",
    "emitter_material": "Stored material identity. Emission is prescribed; no material-dependent tunnelling current is inferred.",
    "work_function_ev": "Stored work function. The current classical emission law prescribes flux and energy; it does not calculate a tunnelling current from this value.",
    "vacuum_pa": "Historical source vacuum metadata. Residual-gas transport is controlled by the explicit vacuum map and its opt-in switch, not this value.",
}

_CORRECTOR_KEYS = frozenset((*PROBE_CORRECTOR_KEYS, *IMAGE_CORRECTOR_KEYS))
_HEXAPOLE_KEYS = frozenset(k for k in _CORRECTOR_KEYS if k.endswith("_hexapole"))
_QUADRUPOLE_KEYS = frozenset(k for k in _CORRECTOR_KEYS if k.endswith("_quadrupole"))
_FIELD_LENGTH = ("Axial Gaussian FWHM of the effective field, not the material body length. "
                 "At fixed peak coefficient, a larger width increases the integrated field; "
                 "the response at a later plane also depends on intervening optics.")
_STIGMATOR_RESPONSE = ("X/Y are two trace-free quadrupole bases in column coordinates, not separate X/Y round lenses. "
    "In the independent model, positive strength focuses along that basis and defocuses its perpendicular axis. "
    "Reversing a pure component exchanges these axes; with both channels active the result depends on their vector sum. "
    "The principal-axis angle is half atan2(skew, normal). The normal_skew model uses independent quadrupole channels. "
    "Hold energy, beam and other optics fixed; reduced spot ellipticity is not guaranteed at every plane.")
_SCAN_RESPONSES = {
    "calibration_mode": "Automatic mode re-solves first-order drive ratios and requested specimen FOV as optics change. Held mode keeps the saved calibration; changed optics can alter actual pitch and leave residual pattern motion. Use Calibrate and hold to explicitly replace it.",
    "calibration_record_json": "Saved drive ratios, command matrix, reference FOV and captured input identity. Consumed only in held mode; this is not a source state. Use Calibrate and hold to generate a consistent record.",
    "scan_reference": "Physical specimen centre or entrance used for scan-position and angle calibration. Changing this choice in held mode requires explicit recalibration; it does not relocate the specimen.",
    "descan_target_key": "Explicit installed component key for the physical plane where opposite descan commands compensate scan displacement to first order; no inferred fallback. Held mode measures the existing drives at a changed target without moving a detector or refitting. A target at/after the energy-filter entrance requires its transported response.",
    "pivot_offset_x": "Dimensionless addition to the X diagonal of the lower/upper scan-coil ratio. It changes the specimen pivot and residual angle, not an X position in mm. Held calibration exposes the resulting pattern motion; its sign at a detector depends on the intervening optics.",
    "pivot_offset_y": "Dimensionless addition to the Y diagonal of the lower/upper scan-coil ratio. It changes the specimen pivot and residual angle, not a Y position in mm. Held calibration exposes the resulting pattern motion; its sign at a detector depends on the intervening optics.",
    "upper_coil_gain": "Signed dimensionless multiplier from common pair command to the upper kick; the lower kick uses the full coupled 2x2 ratio. At fixed held commands it scales both kicks. Automatic FOV calibration may compensate this change, so actual FOV need not grow.",
    "lower_coil_gain": "Representative scalar derived from the upper gain and the mean diagonal of the lower/upper ratio. The actual lower kick uses the full 2x2 matrix, including cross-axis coupling and any scan pivot offset. Editing this scalar adjusts the upper pair gain; it is not a second independent coil control.",
    "scan_pixel_size_nm": "Requested square pixel pitch at the specimen reference. Footprint FOV is N times pitch; the first-to-last pixel-centre span is (N-1) times pitch. Held mode rescales commands for a changed request without correcting changes in optics.",
    "scan_pixels_x": "Number of raster columns. At fixed requested pitch, increases footprint FOV and changes pixel sampling. At fixed frame period, ideal dwell is period/(columns times lines); no flyback dead time is modelled.",
    "scan_lines": "Number of raster rows. At fixed requested pitch, increases footprint FOV and changes line sampling. At fixed frame period, ideal dwell is period/(columns times lines); no flyback dead time is modelled.",
    "scan_frame_period_s": "Period of the shared scan/descan raster clock. Changing it changes dwell and scan speed, not the geometric path at the same normalized raster phase. The present instantaneous-kick model has no electronic lag or hysteresis.",
    "scan_enabled": "Enables this pair's raster drive. Both pairs share one raster clock; descan receives the opposite common command through its own physical coil ratios. Static offsets remain separate.",
    "enabled": "Enables the pair's static and dynamic transverse kicks. Installed physical geometry remains separate from this field switch.",
    "kick_x_mrad": "Static X command added to the dynamic raster/wobble command before the pair's coil matrices. It is an angular command, not a specimen displacement; the resulting position depends on optics.",
    "kick_y_mrad": "Static Y command added to the dynamic raster/wobble command before the pair's coil matrices. It is an angular command, not a specimen displacement; the resulting position depends on optics.",
    "scan_amplitude_x_mrad": "Initial uncalibrated X diagonal seed, not an editable scan control. Active scan drives use the calibrated 2x2 command matrix from requested pitch and counts.",
    "scan_amplitude_y_mrad": "Initial uncalibrated Y diagonal seed, not an editable scan control. Active scan drives use the calibrated 2x2 command matrix from requested pitch and counts.",
    "effective_thickness_mm": "Stored effective coil thickness for validation and layout annotations. The present transport applies thin kicks at the separately defined upper/lower planes; this value does not change their separation or create a distributed coil field.",
    "optical_plane_separation_mm": "Axial separation of the two physical descan kick planes. Changing it changes their optical response and pivot; a held calibration keeps the old drive ratios.",
    "wobble_enabled": "Enables periodic beam-tilt commands for alignment. Scan and wobble are mutually exclusive in this component; wobble does not change the tip source.",
    "wobble_amplitude_x_mrad": "Signed X amplitude of the sinusoidal common-pair wobble command. The coil ratios and optics determine position and angle at an observation plane.",
    "wobble_amplitude_y_mrad": "Signed Y amplitude of the sinusoidal common-pair wobble command. Both axes share one time factor; this is a line wobble, not an independently phased ellipse.",
    "wobble_period_s": "Period of the sinusoidal wobble command. Hold normalized phase fixed to compare amplitudes; no frequency-dependent electronics are modelled.",
    "wobble_phase_deg": "Time-phase offset of the shared sinusoidal wobble, not rotation of the physical X/Y coil axes.",
}

_DETECTOR_KEYS = frozenset((CAMERA, FLUORESCENT_SCREEN, *STEM_DETECTOR_KEYS))
_DETECTOR_RESPONSES = {
    "inserted": "Physical insertion at the installed collection surface. Intercepted electrons stop even when electronic readout is disabled. Retracting a detector changes downstream transmission; inspecting its virtual diagnostic does not insert it.",
    "readout_enabled": "Selects the electronic output only. An inserted detector still intercepts electrons and removes their current from downstream detectors when this readout is off.",
    "geometry": "Active collection shape used by the hit mask, constrained by the existing component validator. Collection depends on the incident distribution and the full transport to this physical plane, not only nominal scattering angles.",
    "outer_width_mm": "Physical active width for a square sensor, or outer diameter for a circular detector. With fixed incident particles, centre and inner opening, enlarging nested active areas cannot reduce interception. It can reduce current at later detectors.",
    "inner_diameter_mm": "Diameter of the non-sensitive central opening of an annular detector. At fixed outer diameter and incident particles, enlarging this opening cannot increase this detector's interception; passing electrons can reach later planes.",
    "centre_offset_x_mm": "X displacement of the physical active-area centre in column coordinates. It changes collection and downstream interception. Translating both an isolated detector and its incident distribution equally must preserve collection.",
    "centre_offset_y_mm": "Y displacement of the physical active-area centre in column coordinates. A position sweep has no universal monotonic collection trend for an arbitrary incident distribution.",
    "pixels": "Native square-camera pixels per side. Fixed physical width sets native pixel pitch = width / pixels. The supported camera image path uses an explicit calculation grid; the ray PSF diagnostic has its own display sampling. More pixels do not create incident electrons or change the physical active width.",
    "point_spread_model": "Forward detector-plane point spread, separate from optical aberrations and physical electron interception. The virtual ray response is a diagnostic; integrated STEM collection is not given an electronic efficiency by this setting.",
    "point_spread_sigma_x_mm": "One-standard-deviation PSF width along its first principal axis, in the detector plane. Increasing spread redistributes signal; the finite output and active-area boundaries can lose response. Hold incident weights and sampling fixed; peak-normalized brightness is not collected current.",
    "point_spread_sigma_y_mm": "One-standard-deviation PSF width along its second principal axis. This is detector response blur, not probe size, material thickness or electron-optical defocus.",
    "point_spread_rotation_deg": "Counter-clockwise orientation of the first PSF principal axis from detector +X. Rotates the response covariance; an isotropic PSF is rotation-invariant. It does not rotate the column or deflect electrons.",
    "point_spread_status": "Provenance of the stored PSF parameters; it does not independently change the response law or certify a measured calibration.",
    "point_spread_source": "Source description of the stored PSF parameters, not an additional response control.",
    "external_envelope": "Stored external-envelope description, not the active-area hit mask or an independent material boundary.",
}
_DOSE_RESPONSES = {
    "stem_poisson_enabled": "Sample optional independent Poisson counts around the expected integrated STEM electrons. Expected electrons = collected current times dwell time / elementary charge. This does not change physical transmission or deterministic current.",
    "stem_poisson_seed": "Non-negative random seed for repeatable Poisson readout. Changing it changes a noise realization, not incident dose, optics or expected electrons.",
    "stem_fourdstem_response_mode": "Ideal or adjustable response applied to supported pixelated acquisitions. Response settings do not replace tip-origin transport or detector interception. Ideal mode leaves the adjustable response values inactive.",
    "stem_fourdstem_quantum_efficiency": "Mean detected fraction in the adjustable pixel-response model, between zero and one. At fixed incident dose and other controls it scales signal expectation. This is not DQE and does not turn undetected electrons into transmitted electrons.",
    "stem_fourdstem_charge_spread_sigma_px": "Gaussian width in output detector pixels, before dark signal and independent pixel Poisson sampling in this simplified model. Zero padding permits edge loss. It is not a calibrated model of charge-sharing noise correlations.",
    "stem_fourdstem_dark_electrons_per_pixel": "Mean dark electrons per detector pixel per exposure, added after signal spreading. This is a per-exposure amount, not a dark-current rate; changing dwell time does not automatically rescale it.",
    "stem_fourdstem_read_noise_electrons_rms": "RMS additive Gaussian electronic noise in electron-equivalent units, after Poisson sampling. The current response clips negative values at zero, which biases low-signal means upward; it is not an unbiased signed amplifier model.",
    "stem_fourdstem_saturation_electrons": "Upper electron-equivalent clipping level before gain and offset. Zero disables saturation in this setting. Above the limit, output ceases to scale linearly with incident dose; clipped counts are not transmitted current.",
    "stem_fourdstem_gain_counts_per_electron": "Positive counts per detected electron after saturation. Changes recorded units, not dose, efficiency or physical interception. Compare counts after subtracting the output offset.",
    "stem_fourdstem_offset_counts": "Additive electronic output offset after gain. It can be negative and is not an electron population or incident-current contribution.",
    "stem_fourdstem_poisson_enabled": "Optional independent Poisson sampling of the blurred signal plus dark expectation in adjustable mode. Does not reproduce event-by-event charge-sharing correlations.",
    "stem_fourdstem_seed": "Non-negative seed for repeatable pixel-response noise in adjustable mode; does not change the incident electron expectation or transport.",
}


def parameter_definition(component, name, *, lens=False):
    category, label, detail, sweep = None, name.replace("_", " "), "", False
    if component == "simulation":
        if name == "virtual_observation_z_mm":
            category, label, detail = "presentation", "Observation plane", "Read-only virtual observation; does not move a physical optical element."
        elif name in {"step_mm", "history_step_mm", "acceleration_backend"}:
            category, detail = "execution", "Numerical resolution/backend evidence remains part of the consumed stage identities."
    elif component == "electron_gun" and name in {"trace_step_mm", "drift_step_mm", "history_step_mm"}:
        category, detail = "execution", (
            "Gun integration or retained-history spacing. Refine independently of tip sampling and column integration; "
            "history spacing affects stored diagnostics, not an independent source. Extraction, acceleration and stops remain executed.")
    elif component == "feg_tip" and name in _TIP_METADATA:
        category, detail = "metadata", _TIP_METADATA[name]
    elif component == "feg_tip" and name in _TIP_RESPONSES:
        category = "execution" if name == "ray_count" else "operating"
        label, detail = "Tip " + label, _TIP_RESPONSES[name]
        # Explanation coverage is not new sweep authorization.
        sweep = name in {"emission_current_na", "virtual_source_fwhm_nm", "angular_rms_mrad",
                         "angular_cutoff_mrad", "energy_spread_fwhm_ev"}
    elif component == "feg_extractor" and name in {"voltage_kv", "transition_start_mm", "transition_end_mm"}:
        category = "operating" if name == "voltage_kv" else "structural"
        detail = ("Extractor electrode potential rise from the tip. "
                  "Extraction changes acceleration and focusing; prescribed emission current is held independently. "
                  "At fixed final potential, increasing extraction voltage is not an additional final-energy gain. "
                  "The coupled electrode field uses physical metal boundaries instead of analytic transition endpoints.")
    elif component == "feg_electrostatic_lens" and name in {"voltage_kv", "voltage_reference", "potential_scale", "soft_edge_mm"}:
        category = "structural" if name == "soft_edge_mm" else "operating"
        detail = ("Changes electrostatic gun focusing, with no universal monotonic downstream spot-size response. "
                  "Voltage reference selects tip, extractor or ground for the actual electrode potential. "
                  "The coupled field does not multiply this voltage by an analytic focusing coefficient.")
    elif component == "feg_accelerator" and name == "high_tension_kv":
        category, label, detail, sweep = "operating", "Gun accelerating voltage", (
            "Changes the connected gun field and every dependent electron transport stage. "
            "At a field-free exit, the electron energy gain follows the potential rise from its actual launch position. "
            "Do not set an independent accelerated source or silently refit lens strengths."), True
    elif lens:
        if name in {"percent", "polarity", "enabled", "cs_mm", "cc_mm"}:
            category, label, detail = "operating", "Lens " + label, "Changes the existing lens control. Dependencies follow complete field support, including overlap and shared circuits."
            sweep = name == "percent"
            if name in {"percent", "polarity"}:
                detail += (" Percent is a model excitation scale, not measured coil current. For an isolated ideal round lens "
                           "at fixed energy and field shape, focusing power scales with squared field amplitude; polarity reverses "
                           "rotation while retaining radial focusing. Beam size at an arbitrary plane is not globally monotonic. "
                           "Material-field models require their own response checks.")
        elif name in {"z_mm", "a_mm", "b0_t", "gaussian"}:
            category, detail = "structural", "Geometry/calibration changes require the existing structural editor; component centres alone do not bound a field."
        elif name == "colour":
            category, detail = "presentation", "Display colour only; existing stage signatures still conservatively own reuse decisions."
    elif component in APERTURE_KEYS and name in {"radius_mm", "diameter_mm", "offset_x_mm", "offset_y_mm", "enabled"}:
        category, label, detail = "operating", "Aperture " + label, (
            "Physical transmission/masking at the installed plane, independent of display and detector readout selections. "
            "For a fixed incident particle population and fixed centre, enlarging nested openings cannot reduce transmitted "
            "current. This does not imply monotonic normalized brightness or RMS beam size.")
    elif component in _DETECTOR_KEYS and name in _DETECTOR_RESPONSES:
        category, detail = "operating", _DETECTOR_RESPONSES[name]
        if name in {"external_envelope", "point_spread_source", "point_spread_status"}:
            category = "metadata"
        elif name.startswith("point_spread_") or name in {"outer_width_mm", "inner_diameter_mm"}:
            category = "structural"
    elif component == "sample" and name in _DOSE_RESPONSES:
        category, detail = "readout", _DOSE_RESPONSES[name]
    elif component == "sample" and name == "thickness_nm":
        category, label, detail, sweep = "operating", "Specimen thickness", "Specimen interactions and surface positions remain physical dependencies.", True
    elif component == "inverse" and name in {"alpha95", "diameter95"}:
        category, detail = "inverse_target", "A requested measured quantity; cannot be assigned as a downstream source or raw lens parameter."
    elif component in {"condenser_stigmator", "objective_stigmator", "diffraction_stigmator"}:
        if name == "field_model":
            category, detail = "metadata", "Current normal_skew quadrupole model identifier; independent physical channels."
        elif name in {"strength_x_percent", "strength_y_percent", "enabled"}:
            category, label = "operating", "Stigmator " + label
            detail = _STIGMATOR_RESPONSE + " Percent scales max_strength_m2, not calibrated coil current."
        elif name == "max_strength_m2":
            category, detail = "operating", "Peak quadrupole coefficient scale at 100 percent. " + _STIGMATOR_RESPONSE
        elif name == "length_mm":
            category, detail = "operating", _FIELD_LENGTH
        elif name in {"channel_x_angle_deg", "channel_y_angle_deg", "z_mm"}:
            category, detail = "structural", "TOML-owned mechanical basis and Gaussian effective axial support; not an OEM field map."
    elif component in _HEXAPOLE_KEYS | _QUADRUPOLE_KEYS:
        if name == "effective_length_mm":
            category, detail = "operating", _FIELD_LENGTH
        elif name in {"strength_m3", "orientation_rad", "enabled"} and component in _HEXAPOLE_KEYS:
            category, label = "operating", "Hexapole " + label
            detail = ("Signed effective nonlinear coefficient: normal/skew = strength times cos/sin(3 times orientation). "
                "The local transverse force is quadratic in displacement; a centred hexapole has no on-axis first-order power. "
                "Rotation by 120 degrees repeats the pattern; 60 degrees reverses it. "
                "Correction depends on the paired fields, relay optics and incident beam; more strength need not improve the probe or image. "
                "Hold energy and relay optics fixed when comparing. The coefficient is not a magnetic gradient in T/m². "
                "This switch controls this field only, not the relay lenses or apertures. Ideal mode omits nonlinear action.")
        elif name in {"strength_m2", "enabled"} and component in _QUADRUPOLE_KEYS:
            category, label = "operating", "Quadrupole " + label
            detail = ("Signed effective coefficient in theta_x' = -q*x, theta_y' = +q*y. Positive q focuses X and defocuses Y "
                "in the column frame; reversing q exchanges these axes. It is not a round-lens power or calibrated coil current. "
                "The later beam shape depends on the incoming beam and other optics. This switch controls only this field.")
    elif component in {"ac_deflector", "descan_deflector"}:
        if name in _SCAN_RESPONSES:
            category, label = "operating", "Scan / descan " + label
            detail = _SCAN_RESPONSES[name]
            if name.startswith("scan_amplitude_"):
                category = "metadata"
            elif name == "lower_coil_gain":
                category = "readout"
            elif name == "calibration_record_json":
                category = "calibration"
            elif name == "effective_thickness_mm":
                category = "metadata"
    if category is None:
        return None
    return ParameterDefinition(str(component), str(name), category, parameter_unit((name,)), label, detail, sweep)


def runtime_definition(target, name):
    return parameter_definition(target.key, name, lens=hasattr(target.obj, "percent") and hasattr(target.obj, "b0_t"))


def registered_sweep_eligibility(parsed):
    """Return a pilot decision or None to retain the established wider allowlist."""
    if len(parsed) == 2 and parsed[0][0] == "lenses" and parsed[0][1] is not None:
        definition = parameter_definition(parsed[0][1], parsed[1][0], lens=True)
    elif parsed == (("sample", None), ("thickness_nm", None)):
        definition = parameter_definition("sample", "thickness_nm")
    else:
        return None
    return None if definition is None else definition.sweep_eligible


def dependency_plan(before, after):
    """Read-only plan backed by existing signatures and the full captured diff."""
    from temsim.calculation_cache import calculation_signatures, explain_product_reuse, incident_field_dependencies
    from temsim.immutable_json import freeze_json
    from temsim.instrument_snapshot import decode_instrument
    from temsim.working_point import snapshot_changes
    left, right = decode_instrument(before.graph), decode_instrument(after.graph)
    old, new = calculation_signatures(left), calculation_signatures(right)
    product_reuse = explain_product_reuse(old, new)
    changes, unknown = [], []
    nodes = after.graph["nodes"]
    root_id = after.graph["root"]["ref"]
    lens_refs = {v["ref"] for v in nodes[root_id]["attributes"]["lenses"]["list"]}
    for path, old_value, new_value in snapshot_changes(before, after):
        match = re.fullmatch(r"/graph/nodes/(\d+)/attributes/([^/]+)", path)
        definition = None
        if match:
            node_id, name = int(match[1]), match[2]
            if node_id < len(nodes):
                key = "simulation" if node_id == root_id else nodes[node_id].get("attributes", {}).get("key")
                definition = parameter_definition(key, name, lens=node_id in lens_refs)
        if definition is None:
            unknown.append(path)
        changes.append(dict(path=path, before=old_value, after=new_value,
            parameter_id=None if definition is None else definition.identity,
            category="unmapped" if definition is None else definition.category,
            reason="Unmapped input; conservative invalidation" if definition is None else definition.label+" changed"))
    support = incident_field_dependencies(right)
    reasons = tuple(dict.fromkeys(row["reason"] for row in changes))
    stages = {}
    for key, row in product_reuse.items():
        reusable = row["reusable"] and not unknown
        stages[key] = dict(reusable=reusable, signature_matches=row["reusable"], previous_signature=old.get(key),
            signature=new[key], reasons=("Exact existing stage signature; captured changes are mapped",) if reusable else
            (reasons or (row["reason"],)))
    return freeze_json(dict(schema="parameter-dependency-plan-v2", before=before.digest, after=after.digest,
        changes=changes, stages=stages, unknown_paths=unknown, incident_field_support=support,
        policy="Read-only conservative adapter; no existing invalidation is narrowed; no solver is executed"))
