"""Rigid main-column placement shared by CAD, rays and wave field queries.

Offsets use mm, right-handed Euler angles use mrad, in Rz @ Ry @ Rx order.
Each part rotates about its unposed mechanical centre. Parent transforms are
then applied outside the child's transform. Shape and excitation stay separate.
"""
import math
from numbers import Real

import numpy as np


PHYSICAL_POSE_FIELDS = (
    "offset_x_mm", "offset_y_mm", "offset_z_mm",
    "rotation_x_mrad", "rotation_y_mrad", "rotation_z_mrad",
)
_PROFILES = frozenset({"magnetic_lens_assembly", "magnetic_lens_housing"})


def column_pose_kind(part):
    """Return ``hardware``/``field`` for supported post-gun column entries.

    A control channel without a verified material host moves its local field
    coordinates only; it does not acquire a fictitious solid. Known channels
    on shared hardware have no second editable placement. Gun, receiver and
    energy-filter branches are deliberately outside this model.
    """
    from temsim.component_keys import PROBE_CORRECTOR_KEYS, IMAGE_CORRECTOR_KEYS
    from temsim.component_representation import non_material_role, SHARED_DEFLECTOR_HOSTS
    key = str(part.get("key", ""))
    if (key.startswith(("feg_", "gun_", "thermionic_", "energy_filter_"))
            or part.get("branch") in {"gun", "source", "detection", "energy_filter"}
            or part.get("mechanical_part_role") == "custom_mechanical_copy"
            or key in SHARED_DEFLECTOR_HOSTS
            or key in {"image_sad_plane", "probe_dp12_scan_deflector"}
            or non_material_role(part) == "virtual_reference"):
        return None
    profile = part.get("mechanical_profile")
    if (profile == "magnetic_lens_assembly"
            or (profile == "magnetic_lens_housing" and not part.get("parent_key"))):
        return "hardware"
    independent = {
        "condenser_deflector", "beam_deflector", "dc_deflector", "ac_deflector",
        "image_diffraction_deflector", "condenser_stigmator", "objective_stigmator",
        "diffraction_stigmator", "condenser_aperture_2", "condenser_aperture_3",
        "objective_aperture", "selected_area_aperture", "projection_chamber_dpa_aperture",
        *PROBE_CORRECTOR_KEYS, *IMAGE_CORRECTOR_KEYS,
    }
    if key not in independent:
        return None
    return "field" if (non_material_role(part) == "control_channel"
                       or profile in {"quadrupole_field_channel", "deflector_field_channel"}) else "hardware"


def supports_physical_column_pose(part):
    return column_pose_kind(part) is not None


def supports_physical_lens_pose(part):
    """Compatibility entry point for the shared main-column placement law."""
    return supports_physical_column_pose(part)


def physical_pose_description(part):
    if column_pose_kind(part) == "field":
        subject = ("Local field-coordinate placement. This channel has no verified separate material host; "
                   "the placement moves its field without creating a material body. ")
    elif "aperture" in str(part.get("key", "")):
        subject = ("Physical aperture placement, shared by the opening and its material plate. "
                   "The operating hole offset remains in the aperture's local frame. ")
    else:
        subject = "Physical component placement. Its field and physically owned child hardware move together. "
    return subject + ("Rotations are right-handed X, then Y, then Z about the component centre, in mrad. "
                      "Excitation and material dimensions are unchanged by rigid placement.")


def inherits_parent_lens_pose(part):
    """Distinguish rigid lens ownership from navigation or packing parents.

    Specimen and steering devices can sit in the objective's envelope without
    belonging to that lens's rigid magnetic assembly. Their children must not
    inherit the objective pose through that independently mounted device.
    """
    from temsim.component_representation import non_material_role
    if supports_physical_lens_pose(part) and part.get("mechanical_profile") in _PROFILES:
        return True
    if supports_physical_column_pose(part):
        # Apertures and non-round optical devices are separately mounted even
        # when parent_key names an enclosing lens for packing/navigation.
        return False
    key = str(part.get("key", ""))
    return not (
        key in {"sample", "sample_stage"}
        or key.endswith(("_deflector", "_stigmator"))
        or part.get("mechanical_profile") == "transverse_goniometer"
        or non_material_role(part)
        or any(part.get(name) for name in (
            "stigmator_structure", "deflector_structure", "scan_structure", "field_basis"))
    )


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def physical_pose_values(part):
    """Read canonical values, accepting historical lens CAD rigid placement.

    Canonical scalar values override the corresponding legacy component; they
    are never added twice. Scaling and Boolean CAD features remain shape edits.
    """
    if not supports_physical_lens_pose(part):
        return dict.fromkeys(PHYSICAL_POSE_FIELDS, 0.0)
    transform = part.get("model_3d", {}).get("transform", {})
    offset = transform.get("offset_mm", (0.0, 0.0, 0.0))
    angles = transform.get("rotation_deg", (0.0, 0.0, 0.0))
    if len(offset) != 3 or len(angles) != 3:
        raise ValueError("Component placement requires three offsets and rotations")
    legacy = tuple(offset) + tuple(_finite(v, "rotation_deg") * math.pi / 180.0 * 1000.0 for v in angles)
    return {name: _finite(part.get(name, value), name)
            for name, value in zip(PHYSICAL_POSE_FIELDS, legacy)}


def validate_physical_lens_pose(part):
    if not supports_physical_lens_pose(part):
        # Apertures and detectors have their own independent offset controls.
        if any(name in part for name in PHYSICAL_POSE_FIELDS[2:]):
            raise ValueError("Physical pose is supported only for post-gun main-column components; edit the physical host for a shared channel")
        return
    physical_pose_values(part)


def migrate_legacy_lens_pose(part):
    """Canonicalise an editable document without losing existing placement."""
    if not supports_physical_lens_pose(part):
        return
    part.update(physical_pose_values(part))
    transform = part.get("model_3d", {}).get("transform", {})
    transform.pop("offset_mm", None)
    transform.pop("rotation_deg", None)


def rotation_matrix_mrad(angles):
    x, y, z = np.asarray(angles, dtype=float) * 1.0e-3
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    return np.array(((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx),
                     (sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx),
                     (-sy, cy*sx, cy*cx)))


def effective_part_transform_mm(part, by_key):
    """Return (R,t) with posed row-vector points = points @ R.T + t.

    ``part`` and ``by_key`` are module-coordinate TOML dictionaries. A child
    inherits its physical parent's placement even if the child has no controls.
    Specimen/steering/channel entries stop inheritance: their parent_key can
    express shared space or navigation rather than a rigid mechanical mount.
    """
    rotation, translation = np.eye(3), np.zeros(3)
    seen = set()
    current = part
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    host = SHARED_DEFLECTOR_HOSTS.get(str(part.get("key", "")))
    if host:
        current = by_key.get(host)
        if current is None:
            raise ValueError(f"Shared field channel requires its captured physical host: {host}")
    while current is not None:
        key = current.get("key", "")
        if key in seen:
            raise ValueError(f"Cyclic lens pose ownership: {key}")
        seen.add(key)
        if supports_physical_lens_pose(current):
            values = physical_pose_values(current)
            local_r = rotation_matrix_mrad([values[name] for name in PHYSICAL_POSE_FIELDS[3:]])
            centre = np.array((0.0, 0.0, float(current.get("local_center_z_mm", 0.0))))
            local_t = centre - local_r @ centre + np.array([values[name] for name in PHYSICAL_POSE_FIELDS[:3]])
            translation = local_r @ translation + local_t
            rotation = local_r @ rotation
        if not inherits_parent_lens_pose(current):
            break
        parent = current.get("parent_key")
        current = by_key.get(parent) if parent else None
    return rotation, translation


def lens_pose_registration(state, lens_key):
    """Baseline global-SI to placed global-SI transform, captured from assembly."""
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    lens_key = SHARED_DEFLECTOR_HOSTS.get(str(lens_key), lens_key)
    from temsim.physics.lens_field_provider import CoordinateRegistration
    assembly = getattr(state, "_resolved_assembly", None)
    from temsim.column.module_assembly import ResolvedAssembly
    # Real assembled inputs are immutable and replaced after a geometry edit.
    # Cache their registration for per-step clipping/field queries on CPU. Do
    # not memoise mutable ad-hoc fixtures or retain this memo in a snapshot.
    cache = None
    if isinstance(assembly, ResolvedAssembly):
        captured = getattr(state, "_lens_pose_registration_cache", None)
        if captured is None or captured[0] is not assembly:
            captured = (assembly, {})
            state._lens_pose_registration_cache = captured
        cache = captured[1]
        if lens_key in cache:
            return cache[lens_key]
    parts = getattr(assembly, "parts", ())
    part = next((p for p in parts if p.key == lens_key), None)
    if part is None:
        result = CoordinateRegistration()
        if cache is not None:
            cache[lens_key] = result
        return result
    # Omitted optical-parent envelopes can still own active physical children.
    # Match the renderer's captured source context, then let active rows win.
    by_key = {p.key: p.data for module in getattr(assembly, "modules", ())
              if module.key == part.module_key for p in module.parts}
    by_key.update({p.key: p.data for p in parts if p.module_key == part.module_key})
    rotation, translation = effective_part_transform_mm(part.data, by_key)
    origin = np.array((0.0, 0.0, part.center_z_mm-float(part.data.get("local_center_z_mm", part.center_z_mm))))
    global_t = origin - rotation @ origin + translation
    result = CoordinateRegistration(tuple(global_t * 1e-3), tuple(map(tuple, rotation)))
    if cache is not None:
        cache[lens_key] = result
    return result


def has_lens_pose(state, lens_key):
    registration = lens_pose_registration(state, lens_key)
    return (any(v != 0.0 for v in registration.origin_global_m)
            or registration.rotation_local_to_global != ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))


def without_lens_rigid_cad(part):
    """Copy CAD settings with physical rigid placement removed from mesh edits."""
    if not supports_physical_lens_pose(part) or "model_3d" not in part:
        return part
    model = dict(part["model_3d"])
    transform = dict(model.get("transform", {}))
    transform.pop("offset_mm", None)
    transform.pop("rotation_deg", None)
    model["transform"] = transform
    return {**part, "model_3d": model}


def has_nonrigid_lens_cad(part):
    """Distinguish unsupported shape edits from a supported rigid placement."""
    model = without_lens_rigid_cad(part).get("model_3d", {})
    return bool(model and (
        model.get("base", {}).get("kind", "existing") != "existing"
        or model.get("features")
        or tuple(model.get("transform", {}).get("scale_xy", (1., 1.))) != (1., 1.)
        or any(model.get("transform", {}).get("offset_mm", ()))
        or any(model.get("transform", {}).get("rotation_deg", ()))
    ))
