"""Multiple operating channels on one physical deflector pair.

The host owns geometry. Channel records retain their independent operating
settings, but neither their legacy coordinates nor their envelopes are hardware.
"""
from __future__ import annotations

from temsim.component_representation import SHARED_DEFLECTOR_HOSTS


_GEOMETRY_FIELDS = (
    "local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
    "optical_reference_local_z_mm", "interaction_centers_local_z_mm",
    "mechanical_outer_diameter_mm", "mechanical_clear_bore_diameter_mm",
    "effective_thickness_mm", "mechanical_inter_coil_gap_mm",
    "vacuum_inner_diameter_mm",
)


def resolve_shared_deflector_parts(document):
    """Resolve channel geometry from its host, including older module files."""
    parts = document.get("parts", ())
    by_key = {str(part["key"]): part for part in parts}
    if not SHARED_DEFLECTOR_HOSTS.keys() & by_key.keys():
        return document
    result = dict(document)
    resolved = []
    for original in parts:
        key = str(original["key"])
        host_key = SHARED_DEFLECTOR_HOSTS.get(key)
        if host_key is None:
            resolved.append(original)
            continue
        if original.get("physical_host_key", host_key) != host_key:
            raise ValueError(f"{key}: physical host must be {host_key}")
        if host_key not in by_key:
            raise ValueError(f"{key}: missing physical deflector host {host_key}")
        host = by_key[host_key]
        part = dict(original)
        for field in _GEOMETRY_FIELDS:
            part[field] = host[field]
        part["mechanical_coil_length_mm"] = host.get(
            "mechanical_coil_length_mm", host["effective_thickness_mm"])
        part.update(layout_role="control_channel", layout_owner=host_key,
                    physical_host_key=host_key, physical_host_status="shared_hardware")
        # A channel is associated with its host, not independently nested in
        # a lens bore at its former position.
        for field in ("parent_key", "mechanical_overlap_group",
                      "mechanical_overlap_role", "mechanical_overlap_reason"):
            part.pop(field, None)
        resolved.append(part)
    result["parts"] = resolved
    return result


_HOST_GEOMETRY_NAMES = frozenset({
    "z_mm", "upper_z_mm", "lower_z_mm", "optical_reference_z_mm",
    "optical_reference_from_tip_mm", "mechanical_center_from_tip_mm",
    "mechanical_center_below_sample_mm", "mechanical_length_mm",
    "mechanical_outer_diameter_mm", "mechanical_clear_bore_diameter_mm",
    "effective_thickness_mm", "mechanical_coil_length_mm",
    "mechanical_inter_coil_gap_mm", "optical_plane_separation_mm",
})


class SharedDeflectorChannel:
    """Expose live host geometry without serialising an extra hardware record."""

    def __getattribute__(self, name):
        if name in _HOST_GEOMETRY_NAMES:
            values = object.__getattribute__(self, "__dict__")
            host = values.get("_physical_host")
            if host is not None:
                if name in {"upper_z_mm", "lower_z_mm"}:
                    return float(getattr(host, name))
                if name == "optical_plane_separation_mm":
                    return float(host.lower_z_mm - host.upper_z_mm)
                if name in {"effective_thickness_mm", "mechanical_coil_length_mm"}:
                    return float(host.thickness_mm)
                if name == "mechanical_inter_coil_gap_mm":
                    return float(host.inter_coil_gap_mm)
                if name in {"mechanical_length_mm", "mechanical_outer_diameter_mm",
                            "mechanical_clear_bore_diameter_mm"}:
                    return float(getattr(host, name))
                center = 0.5 * (float(host.upper_z_mm) + float(host.lower_z_mm))
                if name == "mechanical_center_below_sample_mm":
                    return center - float(values["_physical_host_sample_z_mm"])
                return center
        return object.__getattribute__(self, name)


def bind_shared_deflector_channel(channel, host, sample_z_mm):
    expected = SHARED_DEFLECTOR_HOSTS[str(channel.key)]
    if str(host.key) != expected:
        raise ValueError(f"{channel.key}: physical host must be {expected}")
    object.__setattr__(channel, "_physical_host", host)
    object.__setattr__(channel, "_physical_host_key", expected)
    object.__setattr__(channel, "_physical_host_sample_z_mm", float(sample_z_mm))
    return channel


def bind_shared_deflector_channels(state):
    """Bind installed channels; a minimal state without channels is unchanged."""
    hosts = {str(item.key): item for item in getattr(state, "deflectors", ())}
    for channel in getattr(state, "corrector_elements", ()):
        host_key = SHARED_DEFLECTOR_HOSTS.get(str(channel.key))
        if host_key is None:
            continue
        host = hosts.get(host_key)
        if host is None:
            if bool(getattr(channel, "enabled", False)):
                raise ValueError(f"{channel.key}: missing physical deflector host {host_key}")
            continue
        sample_z = float(getattr(getattr(state, "sample", None), "z_mm", 0.))
        bind_shared_deflector_channel(channel, host, sample_z)


def shared_channel_enabled(component):
    """Respect a shared host's switch without overwriting channel preferences."""
    if not bool(getattr(component, "enabled", False)):
        return False
    if str(getattr(component, "key", "")) not in SHARED_DEFLECTOR_HOSTS:
        return True
    host = getattr(component, "_physical_host", None)
    return host is None or bool(getattr(host, "enabled", False))
