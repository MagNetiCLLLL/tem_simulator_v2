"""Audited AutoScript 1.18 passive getter catalog and tolerant snapshots.

The catalog is generated from the vendor wheel without importing it. Only
catalogued properties and the explicit read/enumeration calls below are used.
No setter, auto-function, insertion or restoration is part of this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
import json
import math
from threading import Event
from typing import Callable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@lru_cache(maxsize=1)
def catalog() -> dict:
    return json.loads(files(__package__).joinpath("autoscript_1_18.json").read_text("utf-8"))


class Cancelled(RuntimeError):
    pass


def check_cancel(cancel: Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Stopped before the next operation; no active SDK call was interrupted.")


def resolve(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def serialise(value, depth=0):
    """Keep native quantities; never guess units or use repr as numeric data."""
    if depth > 12:
        raise ValueError("Readback structure is too deeply nested.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return serialise(value.item(), depth + 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"nonfinite": str(value)}
        return value
    if isinstance(value, (list, tuple)):
        return [serialise(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(k): serialise(v, depth + 1) for k, v in value.items()}
    fields = catalog()["structures"].get(type(value).__name__)
    if fields is None:
        raise TypeError(f"Uncatalogued readback structure: {type(value).__name__}")
    result = {"sdk_type": type(value).__name__}
    for field in fields:
        try:
            result[field["name"]] = serialise(getattr(value, field["name"]), depth + 1)
        except Exception as exc:
            result[field["name"]] = {"read_error": f"{type(exc).__name__}: {exc}"}
    return result


def reading(path: str, getter: Callable, description="", sdk_type="") -> tuple[dict, object]:
    row = {"path": path, "started_utc": utc_now(), "description": description,
           "sdk_type": sdk_type, "status": "unavailable"}
    raw = None
    try:
        raw = getter()
        row.update(status="ok", value=serialise(raw))
        if _has_value_issues(row["value"]):
            row.update(status="partial", error="Some fields are unavailable or non-finite; inspect the saved value.")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["finished_utc"] = utc_now()
    return row, raw


def _has_value_issues(value):
    if isinstance(value, dict):
        return "read_error" in value or "nonfinite" in value or any(_has_value_issues(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_value_issues(v) for v in value)
    return False


class SnapshotCollector:
    def __init__(self, client, progress: Callable[[str], None] = lambda text: None):
        self.client = client
        self.progress = progress

    def collect(self, cancel: Event | None = None) -> dict:
        result = {"started_utc": utc_now(), "sdk_catalog_version": catalog()["sdk_version"],
                  "exclusions": catalog()["exclusions"], "readings": []}
        rows = result["readings"]

        def read(path, getter, description="", sdk_type=""):
            check_cancel(cancel)
            self.progress(f"Reading {path}")
            row, raw = reading(path, getter, description, sdk_type)
            rows.append(row)
            return raw if row["status"] in ("ok", "partial") else None

        for prop in catalog()["properties"]:
            read(prop["path"], lambda p=prop["path"]: resolve(self.client, p),
                 prop["description"], prop["type"])

        def dynamic(obj, prefix, kind):
            for field in catalog()["dynamic_properties"][kind]:
                name = field["name"]
                nested = {("BiPrism", "voltage"): "BiPrismVoltage",
                          ("BiPrism", "rotation"): "BiPrismRotation",
                          ("XLensAlignment", "preset_raw"): "PresetRaw"}.get((kind, name))
                if nested:
                    try:
                        child = getattr(obj, name)
                    except Exception as exc:
                        read(f"{prefix}.{name}", lambda exc=exc: _raise(exc))
                    else:
                        dynamic(child, f"{prefix}.{name}", nested)
                else:
                    read(f"{prefix}.{name}", lambda n=name: getattr(obj, n), field["description"], field.get("type", ""))

        enumerations = [
            ("optics.lenses", "get_available", "get_lens", "Lens"),
            ("optics.aperture_mechanisms", "get_available", "get_mechanism", "ApertureMechanism"),
            ("optics.aperture_mechanisms.bi_prisms", "get_available_bi_prisms", "get_bi_prism", "BiPrism"),
            ("detectors", "camera_detectors", "get_camera_detector", "CameraDetector"),
            ("detectors", "scanning_detectors", "get_scanning_detector", "ScanningDetector"),
            ("detectors", "eds_detectors", "get_eds_detector", "EdsDetector"),
            ("detectors", "eels_detectors", "get_eels_detector", "EelsDetector"),
        ]
        for base, enum_name, factory, kind in enumerations:
            enum_path = f"{base}.{enum_name}"
            # Detector identities have already been read as catalog properties.
            existing = next((r for r in rows if r["path"] == enum_path), None)
            names = (existing.get("value") if existing else
                     read(enum_path, lambda b=base, e=enum_name: getattr(resolve(self.client, b), e)()))
            if not isinstance(names, list):
                continue
            for name in dict.fromkeys(n for n in names if isinstance(n, str)):
                prefix = f"{base}.{factory}[{json.dumps(name)}]"
                check_cancel(cancel)
                try:
                    obj = getattr(resolve(self.client, base), factory)(name)
                except Exception as exc:
                    read(prefix, lambda exc=exc: _raise(exc))
                else:
                    dynamic(obj, prefix, kind)
                    if kind == "CameraDetector":
                        read(prefix + ".get_supported_image_sizes",
                             lambda obj=obj: obj.get_supported_image_sizes(), "Square pixels / sensor readout area")
                    if kind in ("ScanningDetector", "EdsDetector"):
                        segments = []
                        def get_segments(obj=obj):
                            segments.extend(obj.get_segments())
                            return {"count": len(segments)}
                        read(prefix + ".get_segments", get_segments,
                             "All available segments; identity and enabled state are read below.")
                        segment_kind = "DetectorSegment" if kind == "ScanningDetector" else "EdsDetectorSegment"
                        for index, segment in enumerate(segments):
                            dynamic(segment, f"{prefix}.segments[{index}]", segment_kind)

        deflectors = read("optics.deflectors.get_available_deflectors",
                          lambda: self.client.optics.deflectors.get_available_deflectors())
        if isinstance(deflectors, list):
            for name in deflectors:
                read(f"optics.deflectors.get_deflector_value[{json.dumps(name)}]",
                     lambda n=name: self.client.optics.deflectors.get_deflector_value(n),
                     "Vendor readback. Units depend on the deflector; see API catalog.")

        parameters = read("optics.alignments.get_available_parameters",
                          lambda: self.client.optics.alignments.get_available_parameters())
        if isinstance(parameters, list):
            for index, parameter in enumerate(parameters):
                identity = serialise(parameter)
                prefix = f"optics.alignments.parameter[{index}:{json.dumps(identity, sort_keys=True)}]"
                value = read(prefix, lambda p=parameter: self.client.optics.alignments.read_parameter(p))
                if value is None:
                    read(prefix + ".list", lambda p=parameter: self.client.optics.alignments.read_parameter_list(p))

        try:
            x_alignment = self.client.optics.lenses.alignments.x
        except Exception as exc:
            read("optics.lenses.alignments.x", lambda exc=exc: _raise(exc))
        else:
            dynamic(x_alignment, "optics.lenses.alignments.x", "XLensAlignment")

        read("specimen.stage.get_holder_type", lambda: self.client.specimen.stage.get_holder_type())
        for stage, axes in (("stage", "XYZAB"), ("piezo_stage", "XYZ")):
            for axis in axes:
                read(f"specimen.{stage}.get_axis_limits[{axis}]",
                     lambda s=stage, a=axis: getattr(self.client.specimen, s).get_axis_limits(a),
                     "m" if axis in "XYZ" else "rad")
        result["finished_utc"] = utc_now()
        result["coverage"] = {"attempted": len(rows),
                              "readable": sum(r["status"] == "ok" for r in rows),
                              "unavailable": sum(r["status"] != "ok" for r in rows)}
        return result


def _raise(exc):
    raise exc


def compare_snapshots(before: dict, after: dict) -> dict:
    """Endpoint comparison, not a guarantee of stability during the exposure."""
    left = {r["path"]: r for r in before["readings"]}
    right = {r["path"]: r for r in after["readings"]}
    changed, unknown = [], []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key, {}), right.get(key, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            unknown.append(key)
        elif a["value"] != b["value"]:
            changed.append(key)
    return {"changed": changed, "unknown": unknown,
            "meaning": "Sequential before/after readbacks, not an atomic or continuous measurement."}
