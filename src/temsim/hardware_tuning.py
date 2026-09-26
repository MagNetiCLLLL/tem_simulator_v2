"""Manual tuning tasks bound to the existing, unique hardware state.

This registry groups operating controls; it does not define calibrated logical
commands, solve an alignment or create a second instrument state. Gun selectors
resolve against the installed gun, so one task also serves a thermionic source.
"""

from __future__ import annotations

from dataclasses import dataclass

from temsim.component_keys import PROBE_CORRECTOR_KEYS
from temsim.runtime_parameters import RuntimeTarget, editable_parameters, runtime_targets


@dataclass(frozen=True)
class TunableField:
    name: str
    label: str
    unit: str = ""
    choices: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class TuningBinding:
    component_key: str
    label: str
    fields: tuple[TunableField, ...]


@dataclass(frozen=True)
class TuningTask:
    key: str
    label: str
    category: str
    description: str
    bindings: tuple[TuningBinding, ...]


@dataclass(frozen=True)
class ResolvedTuningBinding:
    binding: TuningBinding
    target: RuntimeTarget | None
    fields: tuple[TunableField, ...]
    notice: str


ENABLED = TunableField("enabled", "Enabled")
EXCITATION_VALUE = (TunableField("percent", "Excitation", "%"),)
EXCITATION = (ENABLED,) + EXCITATION_VALUE
PAIRED_KICKS = (ENABLED,) + tuple(
    TunableField(f"{level}_{axis}_mrad", f"{level.title()} {axis.upper()}", "mrad")
    for level in ("upper", "lower") for axis in ("x", "y")
)
GUN_FIELDS = (ENABLED,) + tuple(
    TunableField(f"{level}_field_{axis}_mt", f"{level.title()} magnetic field {axis.upper()}", "mT")
    for level in ("upper", "lower") for axis in ("x", "y")
)
TWO_FOLD = (ENABLED, TunableField("strength_x_percent", "Normal channel", "%"),
            TunableField("strength_y_percent", "Skew channel", "%"))
KICKS = (ENABLED, TunableField("kick_x_mrad", "Static X command", "mrad"),
         TunableField("kick_y_mrad", "Static Y command", "mrad"))
QUADRUPOLE = (ENABLED, TunableField("strength_m2", "Effective quadrupole coefficient", "m⁻²"))
HEXAPOLE = (ENABLED, TunableField("strength_m3", "Effective hexapole coefficient", "m⁻³"),
            TunableField("orientation_rad", "Orientation", "rad"))

# These are selectors, not persisted component identities or new components.
GUN_SELECTORS = {"gun.deflector": "deflector", "gun.stigmator": "stigmator",
                 "gun.extractor": "extractor", "gun.lens": "electrostatic_lens"}

_GUN_PAIR = TuningBinding("gun.deflector", "Gun deflector pair", GUN_FIELDS)
_CONDENSER_PAIR = TuningBinding("condenser_deflector", "Condenser deflector pair", PAIRED_KICKS)
_BEAM_PAIR = TuningBinding("beam_deflector", "Beam deflector pair", PAIRED_KICKS)
_IMAGE_PAIR = TuningBinding("image_diffraction_deflector", "Image / diffraction deflector pair", PAIRED_KICKS)
_CONDENSERS = tuple(TuningBinding(f"condenser_lens_{i}", f"Condenser lens {i}",
                                 EXCITATION_VALUE if i == 3 else EXCITATION)
                    for i in range(1, 4))
_OBJECTIVE = TuningBinding("objective_lens", "Objective lens", EXCITATION)
_MINI_CONDENSER = TuningBinding("mini_condenser", "Mini condenser lens", EXCITATION_VALUE)
_PROJECTORS = tuple(TuningBinding(key, label, EXCITATION) for key, label in (
    ("diffraction_lens", "Diffraction lens"), ("intermediate_lens", "Intermediate lens"),
    ("projector_lens_1", "Projector lens 1"), ("projector_lens_2", "Projector lens 2")))
_MANUAL_PAIR = (
    "The shift and tilt tasks share this physical pair. These are independent manual "
    "hardware drives; a pure shift or tilt at a chosen plane requires a calibrated "
    "combination and is not imposed by this page."
)
_CONDENSER_NOTE = (
    "Adjust the existing condenser excitations together. Current, beam size and angular "
    "spread are determined by forward propagation and apertures; this page does not "
    "assign a beam-current or convergence-angle result."
)
_PROBE_BINDINGS = tuple(TuningBinding(key, label, fields) for key, label, fields in (
    ("adapter_lens", "Corrector entrance adapter lens", EXCITATION),
    ("probe_dph2_deflector", "Upper corrector steering deflector", KICKS),
    ("probe_qph2_quadrupole", "Upper corrector quadrupole", QUADRUPOLE),
    ("probe_hp2_hexapole", "Upper main hexapole", HEXAPOLE),
    ("probe_tl22_lens", "Upper transfer lens", EXCITATION),
    ("probe_dp22_deflector", "Upper transfer deflector", KICKS),
    ("probe_hpc_hexapole", "Intermediate trim hexapole", HEXAPOLE),
    ("probe_qpc_quadrupole", "Intermediate trim quadrupole", QUADRUPOLE),
    ("probe_dp21_deflector", "Lower transfer deflector", KICKS),
    ("probe_tl21_lens", "Lower transfer lens", EXCITATION),
    ("probe_dph1_deflector", "Lower corrector steering deflector", KICKS),
    ("probe_qph1_quadrupole", "Lower corrector quadrupole", QUADRUPOLE),
    ("probe_hp1_hexapole", "Lower main hexapole", HEXAPOLE),
    ("probe_hpol_hexapole", "Exit trim hexapole", HEXAPOLE),
    ("probe_qpol_quadrupole", "Exit trim quadrupole", QUADRUPOLE),
    ("probe_dp11_deflector", "Exit steering deflector", KICKS),
    ("probe_tl12_lens", "Exit relay lens", EXCITATION),
))

TUNING_TASKS = (
    TuningTask("gun_shift", "Gun shift", "Electron gun", _MANUAL_PAIR, (_GUN_PAIR,)),
    TuningTask("gun_tilt", "Gun tilt", "Electron gun", _MANUAL_PAIR, (_GUN_PAIR,)),
    TuningTask("gun_stigmation", "Gun stigmation", "Electron gun",
               "Adjust the finite gun quadrupole's magnetic gradient and physical orientation.",
               (TuningBinding("gun.stigmator", "Gun stigmator", (ENABLED,
                 TunableField("gradient_t_per_m", "Magnetic gradient", "T/m"),
                 TunableField("rotation_deg", "Orientation", "°"))),)),
    TuningTask("gun_extraction", "Extraction and gun focusing", "Electron gun",
               "Existing gun electrode voltages act within the tip-to-column chain. "
               "They do not define an independent accelerated source. Voltage reference "
               "is the reference already declared by the gun model.",
               (TuningBinding("gun.extractor", "Extraction electrode", (
                   TunableField("voltage_kv", "Extraction voltage", "kV"),)),
                TuningBinding("gun.lens", "Electrostatic gun lens", (
                   TunableField("voltage_kv", "Lens voltage", "kV"),)))),
    TuningTask("condenser_current", "Beam current / spot size", "Illumination", _CONDENSER_NOTE, _CONDENSERS),
    TuningTask("condenser_convergence", "Convergence / illumination", "Illumination", _CONDENSER_NOTE, _CONDENSERS),
    TuningTask("condenser_shift", "Condenser shift", "Illumination", _MANUAL_PAIR, (_CONDENSER_PAIR,)),
    TuningTask("condenser_tilt", "Condenser tilt", "Illumination", _MANUAL_PAIR, (_CONDENSER_PAIR,)),
    TuningTask("beam_shift", "Beam shift", "Illumination", _MANUAL_PAIR, (_BEAM_PAIR,)),
    TuningTask("beam_tilt", "Beam tilt", "Illumination", _MANUAL_PAIR, (_BEAM_PAIR,)),
    TuningTask("tem_focus", "Image focus", "Focus and projection",
               "Manually change objective excitation for image formation. The actual image "
               "plane follows the current optics; no calibrated defocus command is supplied.", (_OBJECTIVE,)),
    TuningTask("stem_focus", "Probe focus", "Focus and projection",
               "The mini condenser and objective both influence incident probe formation. "
               "These are manual excitations, not a mode-calibrated focus or compensation control.",
               (_MINI_CONDENSER, _OBJECTIVE)),
    TuningTask("diffraction_focus", "Diffraction focus", "Focus and projection",
               "Adjust downstream lens excitations for the current diffraction plane. "
               "Focus and scale remain coupled until their response has been calibrated.", _PROJECTORS),
    TuningTask("magnification", "Image magnification", "Focus and projection",
               "The downstream lens group jointly changes image scale and focus. "
               "Read magnification from a new calculation in the appropriate image mode.", _PROJECTORS),
    TuningTask("camera_length", "Diffraction camera length", "Focus and projection",
               "The downstream lens group jointly changes angular scale and focus. "
               "Read camera length from a new calculation in the appropriate diffraction mode.", _PROJECTORS),
    TuningTask("image_shift", "Image shift", "Image and diffraction", _MANUAL_PAIR, (_IMAGE_PAIR,)),
    TuningTask("image_tilt", "Image tilt", "Image and diffraction", _MANUAL_PAIR, (_IMAGE_PAIR,)),
    TuningTask("diffraction_shift", "Diffraction shift", "Image and diffraction",
               _MANUAL_PAIR + " The observed response depends on the selected projection mode.", (_IMAGE_PAIR,)),
    *(TuningTask(f"{prefix}_stigmation", f"{label} two-fold stigmation", "Stigmation",
                  "Normal and skew are independent quadrupole channels in the declared "
                  "hardware basis. They are not separate horizontal and vertical lenses.",
                  (TuningBinding(f"{prefix}_stigmator", f"{label} stigmator", TWO_FOLD),))
      for prefix, label in (("condenser", "Condenser"), ("objective", "Objective"), ("diffraction", "Diffraction"))),
    *(TuningTask(f"{prefix}_threefold", f"{label} three-fold stigmation", "Stigmation",
                  "Unsupported: no independent three-fold stigmator is declared at this "
                  "location. Corrector hexapoles remain separate hardware controls; a "
                  "hexapole drive is not a calibrated specimen-plane aberration coefficient.", ())
      for prefix, label in (("condenser", "Condenser"), ("objective", "Objective"))),
    TuningTask("probe_corrector", "Hexapole probe corrector", "Corrector",
               "Manual controls of the installed composite corrector. Coefficients are "
               "those of the effective field model, not measured coil currents or individual "
               "wave-aberration targets. No automatic correction is performed.", _PROBE_BINDINGS),
    TuningTask("scan_static", "Scan deflector static controls", "Scanning",
               "Adjust the static drive and pair coupling of the physical scan deflector. "
               "Raster timing, dynamic drive and calibration remain in Scanning Parameters.",
               (TuningBinding("ac_deflector", "Scan deflector pair", KICKS + (
                   TunableField("upper_coil_gain", "Upper coil command gain"),
                   TunableField("pivot_offset_x", "X lower-ratio offset"),
                   TunableField("pivot_offset_y", "Y lower-ratio offset"))),)),
    TuningTask("descan_static", "Descan deflector static controls", "Scanning",
               "Adjust the static drive and upper gain of the physical descan pair. "
               "The lower drive is derived from the existing calibration; raster timing "
               "and the observation plane remain in Scanning Parameters.",
               (TuningBinding("descan_deflector", "Descan deflector pair", KICKS + (
                   TunableField("upper_coil_gain", "Upper coil command gain"),)),)),
    TuningTask("beam_wobble", "Beam wobble", "Scanning",
               "The existing scan deflector adds a time-dependent sinusoidal kick to its "
               "static drive. Set amplitudes, period and shared temporal phase. Wobble and "
               "raster scanning are mutually exclusive; this page does not change scan timing.",
               (TuningBinding("ac_deflector", "Scan deflector wobble drive", (
                   ENABLED, TunableField("wobble_enabled", "Wobble enabled"),
                   TunableField("wobble_amplitude_x_mrad", "X amplitude", "mrad"),
                   TunableField("wobble_amplitude_y_mrad", "Y amplitude", "mrad"),
                   TunableField("wobble_period_s", "Period", "s"),
                   TunableField("wobble_phase_deg", "Temporal phase", "°"))),)),
)
TASK_BY_KEY = {task.key: task for task in TUNING_TASKS}


def _target_key(state, component_key):
    if component_key in GUN_SELECTORS:
        component = getattr(state.electron_gun, GUN_SELECTORS[component_key], None)
        return getattr(component, "key", "")
    return component_key


def resolve_task(state, task: TuningTask | str) -> tuple[ResolvedTuningBinding, ...]:
    """Resolve read-only bindings without reconciling geometry or running physics."""
    if isinstance(task, str):
        task = TASK_BY_KEY[task]
    targets = runtime_targets(state)
    groups = []
    for binding in task.bindings:
        key = _target_key(state, binding.component_key)
        target = targets.get(key)
        notice = ""
        if key in PROBE_CORRECTOR_KEYS and not bool(getattr(state, "probe_corrector_installed", False)):
            target = None
            notice = "Unavailable: the probe corrector is not installed."
        elif key == "condenser_lens_3" and getattr(state, "layout_c3_hardware", "three_condenser") != "three_condenser":
            target = None
            notice = "Unavailable: condenser lens 3 is not installed."
        elif target is None:
            notice = "Unavailable: this hardware is not present in the current instrument."
        elif not bool(getattr(target.obj, "_layout_installed", True)):
            target = None
            notice = "Unavailable: this hardware is not installed in the current column layout."
        elif getattr(target.obj, "KIND", None) == "virtual_layout":
            target = None
            notice = "Unavailable: this is a layout marker with no physical drive."
        elif getattr(state, "simulation_mode", "custom") == "ideal" and hasattr(target.obj, "strength_m3"):
            target = None
            notice = "Inactive in Ideal Optics: hexapole field terms are excluded by this model."
        if target is None:
            groups.append(ResolvedTuningBinding(binding, None, (), notice))
            continue
        allowed = {field.name for field in editable_parameters(target)}
        fields = tuple(field for field in binding.fields if field.name in allowed)
        if not fields:
            groups.append(ResolvedTuningBinding(binding, None, (),
                          "Unavailable: no supported operating controls are exposed by this component."))
            continue
        if not bool(getattr(target.obj, "enabled", True)):
            notice = "Disabled: stored drives have no effect until this component is enabled."
        if key == "condenser_lens_3":
            activity = "enabled" if bool(getattr(target.obj, "enabled", True)) else "disabled"
            notice = f"Excitation is {activity}; activation is controlled by the column configuration."
        elif key == "mini_condenser":
            notice = "This lens is enabled by the column layout."
        elif task.key == "beam_wobble" and bool(getattr(target.obj, "scan_enabled", False)):
            notice = "Raster scanning is active. Stop the raster in Scanning Parameters before enabling wobble."
        reference = getattr(target.obj, "voltage_reference", None)
        if reference is not None:
            notice = f"Lens voltage reference: {reference}."
        if len(fields) != len(binding.fields):
            notice = (notice + " Some controls are not supported by this component model.").strip()
        groups.append(ResolvedTuningBinding(binding, target, fields, notice))
    return tuple(groups)
