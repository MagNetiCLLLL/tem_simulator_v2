"""Traceable intrinsic and system-level electron-optical aberrations.

The simulator does not claim OEM calibration.  Explicit component values are
kept authoritative; otherwise a focal-length-scaled round-lens estimate is
reported as provisional instead of silently presenting a missing value as
zero.  System coefficients use the conventional wave-aberration expansion at
the probe/specimen or Objective image reference plane.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
import math

import numpy as np

from temsim.optics.lens_focal_length import focal_length_mm
from temsim.optics.aberration_basis import ABERRATION_SCHEMA, WAVE_TERMS, UNIMPLEMENTED_TERMS


DEFAULT_CS_TO_FOCAL_LENGTH_RATIO = 3.0 / 7.0
DEFAULT_CC_TO_FOCAL_LENGTH_RATIO = 1.0


@dataclass(frozen=True)
class IntrinsicLensAberrationProfile:
    """Intrinsic round-lens coefficients and their provenance, in mm."""

    cs_mm: float | None
    cc_mm: float | None
    model: str
    status: str
    source: str


@dataclass(frozen=True)
class EffectiveAberrationSet:
    """Effective wave-aberration coefficients at one reference plane.

    Length coefficients are millimetres and azimuths are degrees in the
    simulator's right-handed X-Y frame.  ``C1`` is defocus, ``A1`` two-fold
    astigmatism, ``B2`` axial coma, ``A2`` three-fold astigmatism, ``C3``
    spherical aberration, ``S3`` star aberration, ``A3`` four-fold
    astigmatism, ``C5`` fifth-order spherical aberration, ``A5`` six-fold
    astigmatism (C56), and ``Cc`` the
    first-order chromatic coefficient.
    """

    reference_plane: str
    correction_state: str
    c1_mm: float = 0.0
    a1_mm: float = 0.0
    a1_azimuth_deg: float = 0.0
    b2_mm: float = 0.0
    b2_azimuth_deg: float = 0.0
    a2_mm: float = 0.0
    a2_azimuth_deg: float = 0.0
    c3_mm: float = 0.0
    s3_mm: float = 0.0
    s3_azimuth_deg: float = 0.0
    a3_mm: float = 0.0
    a3_azimuth_deg: float = 0.0
    c5_mm: float = 0.0
    a5_mm: float = 0.0
    a5_azimuth_deg: float = 0.0
    cc_mm: float = 0.0
    status: str = "principle_model"
    source: str = "simulator state; non-OEM calibration"

    def validate(self) -> "EffectiveAberrationSet":
        for name, value in self.__dict__.items():
            if name.endswith(("_mm", "_deg")) and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.cc_mm < 0.0 and self.correction_state != "field-derived":
            raise ValueError("Cc cannot be negative in this first-order model.")
        return self


SYSTEM_COEFFICIENT_ROWS = (
    ("C1", "c1_mm", None),
    ("A1", "a1_mm", "a1_azimuth_deg"),
    ("B2", "b2_mm", "b2_azimuth_deg"),
    ("A2", "a2_mm", "a2_azimuth_deg"),
    ("C3", "c3_mm", None),
    ("S3", "s3_mm", "s3_azimuth_deg"),
    ("A3", "a3_mm", "a3_azimuth_deg"),
    ("C5", "c5_mm", None),
    ("A5", "a5_mm", "a5_azimuth_deg"),
    ("Cc", "cc_mm", None),
)


def _finite_explicit(value, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def intrinsic_lens_aberration_profile(lens, voltage_kv) -> IntrinsicLensAberrationProfile:
    """Resolve a round lens without confusing unavailable and zero values."""

    name = str(getattr(lens, "name", "Round lens"))
    explicit_cs = _finite_explicit(getattr(lens, "cs_mm", None), f"{name} Cs")
    explicit_cc = _finite_explicit(getattr(lens, "cc_mm", None), f"{name} Cc")
    if explicit_cc is not None and explicit_cc < 0.0:
        raise ValueError(f"{name} Cc cannot be negative.")

    focal = None
    if explicit_cs is None or explicit_cc is None:
        try:
            candidate = float(focal_length_mm(lens, voltage_kv))
        except Exception:
            candidate = math.nan
        if math.isfinite(candidate) and candidate > 0.0:
            focal = candidate

    cs_mm = (
        explicit_cs
        if explicit_cs is not None
        else None if focal is None else DEFAULT_CS_TO_FOCAL_LENGTH_RATIO * focal
    )
    cc_mm = (
        explicit_cc
        if explicit_cc is not None
        else None if focal is None else DEFAULT_CC_TO_FOCAL_LENGTH_RATIO * focal
    )
    explicit = explicit_cs is not None and explicit_cc is not None
    mixed = (explicit_cs is None) != (explicit_cc is None)
    unavailable = cs_mm is None or cc_mm is None
    if unavailable:
        model = "partially unavailable"
        status = "unavailable"
    elif explicit:
        model = "explicit component coefficients"
        status = "configured"
    elif mixed:
        model = "mixed explicit and focal-length estimate"
        status = "provisional principle model"
    else:
        model = "focal-length-scaled round-lens estimate"
        status = "provisional principle model"
    sources = []
    sources.append("explicit Cs" if explicit_cs is not None else "Cs=(3/7)f estimate")
    sources.append("explicit Cc" if explicit_cc is not None else "Cc=f estimate")
    return IntrinsicLensAberrationProfile(
        cs_mm=cs_mm,
        cc_mm=cc_mm,
        model=model,
        status=status,
        source="; ".join(sources) + "; non-OEM calibration",
    )


def spherical_aberration_mm(lens, voltage_kv):
    return intrinsic_lens_aberration_profile(lens, voltage_kv).cs_mm


def chromatic_aberration_mm(lens, voltage_kv):
    return intrinsic_lens_aberration_profile(lens, voltage_kv).cc_mm


def _objective_defocus_mm(state) -> float:
    objective = getattr(state, "objective_lens", None)
    sample = getattr(state, "sample", None)
    user_defocus = float(getattr(sample, "wave_defocus_nm", 0.0)) * 1.0e-6
    if objective is None:
        return user_defocus
    try:
        current = float(objective.focal_length_for_voltage_mm(state.beam_voltage_kv))
        nominal = float(objective.nominal_focal_length_mm)
    except Exception:
        return user_defocus
    return user_defocus + current - nominal if math.isfinite(current) else user_defocus


def configured_probe_defocus_mm(state) -> float:
    """Return the additional coherent-probe C1 requested by the user.

    The condenser and Objective excitations are deliberately excluded here:
    their first-order focus is already present in the sample-plane ray bundle
    and is converted to a wave-aberration term by the STEM probe builder.
    ``wave_defocus_nm`` is therefore an additional specimen-referenced offset;
    an explicit probe C1 override replaces that offset.
    """

    sample = getattr(state, "sample", None)
    value = float(getattr(sample, "wave_defocus_nm", 0.0)) * 1.0e-6
    overrides = getattr(state, "probe_aberrations", {}) or {}
    if "c1_mm" in overrides:
        value = float(overrides["c1_mm"])
    if not math.isfinite(value):
        raise ValueError("Configured probe defocus must be finite.")
    return value


def _configured_system_set(state, system: str) -> EffectiveAberrationSet:
    system = str(system).lower()
    if system not in {"probe", "image"}:
        raise ValueError("Aberration system must be 'probe' or 'image'.")
    objective = getattr(state, "objective_lens", None)
    profile = (
        intrinsic_lens_aberration_profile(objective, state.beam_voltage_kv)
        if objective is not None
        else IntrinsicLensAberrationProfile(None, None, "unavailable", "unavailable", "no Objective")
    )
    values = {
        "reference_plane": "sample/probe" if system == "probe" else "objective image",
        "correction_state": "uncorrected",
        "c1_mm": (
            configured_probe_defocus_mm(state)
            if system == "probe"
            else _objective_defocus_mm(state)
        ),
        "c3_mm": float(profile.cs_mm or 0.0),
        "cc_mm": float(profile.cc_mm or 0.0),
    }
    overrides = getattr(state, f"{system}_aberrations", {}) or {}
    allowed = EffectiveAberrationSet.__dataclass_fields__
    for key, value in dict(overrides).items():
        if key in allowed and key not in {"reference_plane", "correction_state"}:
            values[key] = value
    return EffectiveAberrationSet(**values).validate()


def _corrector_trace_ratio(state, system: str) -> tuple[float, float, float, str]:
    """Return a compact-ring C3 ratio and transverse-position RMS errors in m.

    This probe-independent diagnostic does not measure the other coefficients
    or establish the quality of the actual finite-source sample probe.
    """

    from temsim.physics.core import propagate

    # Corrector calibration is sensitive to coarse axial integration.  A
    # shallow diagnostic copy preserves the user's state while enforcing the
    # same converged maximum step used by the corrector regression tests.
    state = copy.copy(state)
    state.step_mm = min(float(getattr(state, "step_mm", 0.1)), 0.1)
    state.history_step_mm = state.step_mm
    components = {
        getattr(item, "key", ""): item
        for item in (*getattr(state, "lenses", ()), *getattr(state, "corrector_elements", ()))
    }
    phi = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    zero = np.zeros(phi.size)
    system = str(system).lower()
    if system == "image":
        end = components.get("image_sad_plane")
        active = bool(getattr(state, "image_corrector_installed", False) and end is not None)
        if not active:
            return 1.0, 0.0, 0.0, "image corrector not active"
        z0, z1 = float(state.sample.z_mm), float(end.z_mm)
        alpha = 1.0e-3
        inputs = (zero, alpha * np.cos(phi), zero, alpha * np.sin(phi))
    else:
        hp2 = components.get("probe_hp2_hexapole")
        active = bool(getattr(state, "probe_corrector_installed", False) and hp2 is not None)
        if not active:
            return 1.0, 0.0, 0.0, "probe corrector not active"
        z0 = float(hp2.z_mm) - 5.0 * float(hp2.effective_length_mm) / 2.355
        z1 = float(state.sample.z_mm)
        radius = 1.0e-4
        inputs = (radius * np.cos(phi), zero, radius * np.sin(phi), zero)

    baseline = propagate(state, z0, z1, *inputs, include_spherical_aberration=False, include_hexapole=False)
    uncorrected = propagate(state, z0, z1, *inputs, include_spherical_aberration=True, include_hexapole=False)
    corrected = propagate(state, z0, z1, *inputs, include_spherical_aberration=True, include_hexapole=True)

    def error(result):
        return (result[1][-1] - baseline[1][-1]) + 1j * (result[3][-1] - baseline[3][-1])

    positive = error(uncorrected)
    residual = error(corrected)
    denominator = float(np.vdot(positive, positive).real)
    if denominator <= 1.0e-30:
        return 1.0, 0.0, 0.0, "C3 trace was below numerical resolution"
    ratio = float(np.vdot(positive, residual).real / denominator)
    rms_before = float(np.sqrt(np.mean(np.abs(positive) ** 2)))
    rms_after = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    return ratio, rms_before, rms_after, "distributed round-lens Cs plus explicit hexapole fields"


def _public_optical_value(value):
    """Freeze component parameters without retaining private runtime caches."""

    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _public_optical_value(item))
            for key, item in value.items() if not str(key).startswith("_")
        ))
    if isinstance(value, (list, tuple)):
        return tuple(_public_optical_value(item) for item in value)
    if hasattr(value, "__dict__"):
        return (type(value).__qualname__, _public_optical_value(vars(value)))
    return value


def _aberration_cache_signature(state, system: str) -> str:
    # Include all public field parameters, not just current settings. This
    # covers hexapole orientation_rad, quadrupoles, field widths and profiles,
    # polarity, explicit Cs/Cc, and sample/reference-plane motion. Mechanics
    # included here can cause harmless cache misses; stale physics cannot.
    snapshot = {
        name: getattr(state, name, None)
        for name in (
            "beam_voltage_kv", "step_mm", "acceleration_enabled",
            "acceleration_backend", "chromatic_aberration_enabled",
            "equivalent_image_lenses_enabled", "illumination_mode",
            "projector_mode", "objective_coupled", "objective_image_plane_z_mm",
            "objective_back_focal_plane_z_mm", "lenses", "corrector_elements",
            "stigmators", "deflectors", "electron_gun", "component_placements",
            "lens_field_map_descriptors", "_lens_field_map_bindings",
            "simulation_mode",
        )
    }
    snapshot.update(
        system=system,
        sample_z_mm=float(state.sample.z_mm),
        sample_wave_defocus_nm=float(getattr(state.sample, "wave_defocus_nm", 0.0)),
        overrides=getattr(state, f"{system}_aberrations", {}),
        installed=bool(getattr(state, f"{system}_corrector_installed", False)),
    )
    from temsim.physics.lens_field_provider import lens_geometry_binding
    snapshot["field_geometry_bindings"] = {
        key: lens_geometry_binding(state, key).geometry_fingerprint
        for key in getattr(state, "lens_field_map_descriptors", {})
    }
    snapshot["bound_map_contents"] = {
        key: value.content_fingerprint
        for key, value in getattr(state, "_lens_field_map_bindings", {}).items()
    }
    snapshot["aberration_schema"] = ABERRATION_SCHEMA
    return hashlib.sha256(repr(_public_optical_value(snapshot)).encode("utf-8")).hexdigest()


def effective_aberration_comparison(state, system: str) -> tuple[EffectiveAberrationSet, EffectiveAberrationSet, dict]:
    """Build coefficient models, with explicit measurement/provenance limits."""

    system = str(system).lower()
    if system not in {"probe", "image"}:
        raise ValueError("Aberration system must be 'probe' or 'image'.")
    from temsim.simulation_modes import is_ideal
    if is_ideal(state):
        overrides = getattr(state, f"{system}_aberrations", {}) or {}
        defocus = (configured_probe_defocus_mm(state) if system == "probe" else
                   float(overrides.get("c1_mm", _objective_defocus_mm(state))))
        ideal = EffectiveAberrationSet(
            "specimen" if system == "probe" else "objective image", "ideal",
            c1_mm=defocus, status="ideal_model", source="Ideal Optics; configured defocus retained",
        ).validate()
        return ideal, ideal, {
            "c3_residual_ratio": 0.0, "ray_error_rms_before": 0.0, "ray_error_rms_after": 0.0,
            "source": ideal.source, "diagnostic_scope": "Lens aberrations disabled by model; not a measurement or fit",
            "coefficient_status": {term: "configured defocus" if term == "C1" else "disabled by Ideal Optics"
                                   for term, _, _ in SYSTEM_COEFFICIENT_ROWS},
            "inferred_coefficients": (), "unmeasured_coefficients": (),
            "unimplemented_terms": UNIMPLEMENTED_TERMS,
        }
    if hasattr(state, "sync_objective"):
        state.sync_objective()
    signature = _aberration_cache_signature(state, system)
    cache = getattr(state, "_effective_aberration_cache", None)
    if cache is None:
        cache = {}
        setattr(state, "_effective_aberration_cache", cache)
    if system in cache and cache[system][0] == signature:
        return cache[system][1]
    mode = str((getattr(state, f"{system}_aberrations", {}) or {}).get("mode", "manual"))
    if mode == "field_derived":
        from temsim.optics.field_aberrations import derive_field_aberrations
        result = derive_field_aberrations(state, system)
        # Resolving a saved map can populate its immutable binding registry.
        # Store the resolved identity, otherwise the next consumer repeats the fit.
        cache[system] = (_aberration_cache_signature(state, system), result)
        return result
    if mode != "manual":
        raise ValueError("Aberration mode must be manual or field_derived")
    before = _configured_system_set(state, system)
    ratio, rms_before, rms_after, source = _corrector_trace_ratio(state, system)
    after = replace(
        before,
        correction_state="corrected" if source.startswith("distributed") else "uncorrected",
        c3_mm=before.c3_mm * ratio,
        source=source,
    )
    overrides = getattr(state, f"{system}_aberrations", {}) or {}
    inferred = ("C3",) if source.startswith("distributed") else ()
    coefficient_status = {}
    unmeasured = []
    for term, value_name, _angle_name in SYSTEM_COEFFICIENT_ROWS:
        if term in inferred:
            coefficient_status[term] = "compact-ring C3 estimate"
        elif value_name in overrides:
            coefficient_status[term] = "configured; not fitted"
        elif term == "C1":
            coefficient_status[term] = (
                "configured offset; ray focus reported separately"
                if system == "probe" else "lens/configured offset"
            )
        elif term in {"C3", "Cc"}:
            coefficient_status[term] = "lens coefficient; not fitted"
        else:
            coefficient_status[term] = "unmeasured; wave-model default zero"
            unmeasured.append(term)
    result = before, after, {
        "c3_residual_ratio": ratio,
        "ray_error_rms_before": rms_before,
        "ray_error_rms_after": rms_after,
        "source": source,
        "ray_error_rms_unit": "m",
        "coefficient_status": coefficient_status,
        "inferred_coefficients": inferred,
        "unmeasured_coefficients": tuple(unmeasured),
        "unimplemented_terms": UNIMPLEMENTED_TERMS,
        "diagnostic_scope": (
            "Compact reference ring with nonlinear fields off/on; C3 only. "
            "Other residual coefficients are not fitted from the actual probe."
        ),
    }
    # Keep one entry per system while adjusting channels; numerical searches
    # must not retain an unbounded dictionary of historical optical settings.
    cache[system] = (signature, result)
    return result


def active_effective_aberrations(state, system: str) -> EffectiveAberrationSet:
    return effective_aberration_comparison(state, system)[1]


def prepare_field_aberration_diagnostics(state, previous_state=None) -> None:
    """Populate worker-owned fits once; unchanged snapshots can reuse them."""
    from temsim.physics.lens_field_provider import active_mapped_providers
    from temsim.simulation_modes import is_ideal
    if is_ideal(state):
        return
    systems = tuple(system for system in ("probe", "image")
                    if getattr(state, f"{system}_aberrations", {}).get("mode") == "field_derived")
    if not systems:
        return
    if hasattr(state, "sync_objective"):
        state.sync_objective()
    active_mapped_providers(state)
    # Never share a mutable cache dictionary with the previous result.
    cache = dict(getattr(state, "_effective_aberration_cache", {}))
    state._effective_aberration_cache = cache
    previous = getattr(previous_state, "_effective_aberration_cache", {})
    for system in systems:
        signature = _aberration_cache_signature(state, system)
        entry = previous.get(system)
        if (cache.get(system, (None,))[0] != signature
                and entry is not None and entry[0] == signature):
            cache[system] = entry
        effective_aberration_comparison(state, system)


def aberration_phase_rad(
    frequency_x_inv_angstrom,
    frequency_y_inv_angstrom,
    wavelength_angstrom: float,
    coefficients: EffectiveAberrationSet,
):
    """Evaluate the declared wave basis; transfer uses exp(-i chi).

    C5 and A5 are supported; C41, C43, C45, C52 and C54 are not implemented.
    """

    coefficients.validate()
    fx = np.asarray(frequency_x_inv_angstrom, dtype=float)
    fy = np.asarray(frequency_y_inv_angstrom, dtype=float)
    if fx.shape != fy.shape:
        raise ValueError("Aberration frequency grids must have matching shapes.")
    wavelength_m = float(wavelength_angstrom) * 1.0e-10
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("Electron wavelength must be finite and positive.")
    theta_x = wavelength_m * fx * 1.0e10
    theta_y = wavelength_m * fy * 1.0e10
    theta = np.hypot(theta_x, theta_y)
    phi = np.arctan2(theta_y, theta_x)
    wave_m = np.zeros_like(theta)
    for term in WAVE_TERMS:
        azimuth = math.radians(getattr(coefficients, term.azimuth_field)) if term.azimuth_field else 0.0
        wave_m += (getattr(coefficients, term.field) * 1e-3 / term.power * theta**term.power
                   * np.cos(term.harmonic * (phi - azimuth)))
    return (2.0 * math.pi / wavelength_m) * wave_m


def chromatic_defocus_mm(cc_mm: float, energy_offset_ev: float, beam_voltage_kv: float) -> float:
    """First-order chromatic defocus, using Δf = Cc ΔE/E0."""

    cc = float(cc_mm)
    energy = float(beam_voltage_kv) * 1000.0
    if not math.isfinite(cc) or cc < 0.0 or not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("Cc and beam energy must define a finite physical scale.")
    return cc * float(energy_offset_ev) / energy
