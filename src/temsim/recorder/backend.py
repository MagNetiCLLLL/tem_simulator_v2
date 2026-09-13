"""Collect existing microscope streams without supplying acquisition settings."""
from __future__ import annotations

from importlib.metadata import version
import re
from threading import Event

from .records import CaptureRequest, RecordBundle
from .frame_metadata import collect_frame_metadata, frame_timing
from .snapshot import (
    Cancelled, SnapshotCollector, catalog, check_cancel, compare_snapshots,
    reading, resolve, utc_now,
)


def load_client():
    try:
        installed = version("autoscript-tem-microscope-client")
        if installed != "1.18.0":
            raise RuntimeError(f"AutoScript client {installed} is not audited; install vendor version 1.18.0.")
        from autoscript_tem_microscope_client import TemMicroscopeClient
    except ImportError as exc:
        raise RuntimeError("Install the licensed AutoScript TEM 1.18.0 vendor wheels in this Python environment. "
                           "The recorder does not install software or connect automatically.") from exc
    return TemMicroscopeClient()


class InstrumentRecorder:
    def __init__(self, client_factory=load_client):
        self._client_factory = client_factory
        self.client = None
        self.identity = {}

    def connect(self, host: str, port=7521):
        if self.client is not None:
            raise RuntimeError("Disconnect before opening a new microscope connection.")
        if not host.strip() or not 1 <= port <= 65535:
            raise ValueError("Enter the microscope host and a valid TCP port.")
        client = self._client_factory()
        # Retain the handle if cleanup fails, so the UI cannot falsely claim disconnection.
        self.client = client
        try:
            client.connect(host.strip(), port)
            server = client.service.autoscript.server
            if server.is_offline is not False:
                raise RuntimeError("Offline or unverified AutoScript server: real-instrument recording is disabled.")
            server_version = str(server.version)
            if not re.match(r"^1\.18(?:\.|$)", server_version):
                raise RuntimeError(f"Server {server_version} is outside the audited AutoScript 1.18 API.")
            name = str(client.service.system.name).strip()
            serial = str(client.service.system.serial_number).strip()
            if not name or not serial or name.lower() == "none" or serial.lower() == "none":
                raise RuntimeError("The server did not provide a real instrument name and serial number.")
            self.identity = {"name": name, "serial_number": serial, "server_version": server_version,
                             "client_version": str(client.service.autoscript.client.version),
                             "host": host.strip(), "port": port, "verified_utc": utc_now(),
                             "catalog_wheel_sha256": catalog()["wheel_sha256"], "offline": False}
            return self.inspect()
        except Exception:
            self.identity = {}
            self.disconnect()
            raise

    def disconnect(self):
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        self.identity = {}

    def require_client(self):
        if self.client is None or not self.identity:
            raise RuntimeError("Connect and verify a real microscope first.")
        return self.client

    def inspect(self):
        client = self.require_client()
        devices = {}
        errors = {}
        for kind in ("camera", "scanning"):
            try:
                names = getattr(client.detectors, f"{kind}_detectors")
                if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
                    raise ValueError("Unexpected detector identity list.")
                devices[kind] = names
            except Exception as exc:
                devices[kind] = []
                errors[kind] = str(exc)
        return {"identity": self.identity.copy(), "detectors": devices, "errors": errors,
                "mode": client.optics.optical_mode, "projector_mode": client.optics.projector_mode}

    def snapshot(self, cancel=None, progress=lambda text: None):
        return SnapshotCollector(self.require_client(), progress).collect(cancel)

    def _preflight(self, request):
        client = self.require_client()
        # Recheck identity: a reconnect or endpoint replacement must not relabel images.
        if client.service.autoscript.server.is_offline is not False:
            raise RuntimeError("The endpoint is no longer a verified real microscope.")
        if str(client.service.system.serial_number).strip() != self.identity["serial_number"]:
            raise RuntimeError("Instrument identity changed; reconnect before acquisition.")
        optical, projector = client.optics.optical_mode, client.optics.projector_mode
        if request.route == "ceta" and optical != "Tem":
            raise RuntimeError("Ceta capture requires TEM optical mode. Set it on the microscope; no automatic switch.")
        if request.route == "stem_all" and optical != "Stem":
            raise RuntimeError("STEM capture requires STEM optical mode. Set it on the microscope; no automatic switch.")
        kind = "scanning" if request.route == "stem_all" else "camera"
        available = getattr(client.detectors, f"{kind}_detectors")
        if not isinstance(available, list) or not all(isinstance(n, str) and n for n in available):
            raise RuntimeError("Cannot verify the current detector list.")
        allowed = {"Flucam"} if request.route == "flucam" else {"BM-Ceta", "EF-Ceta"}
        names = list(dict.fromkeys(available)) if kind == "scanning" else [n for n in available if n in allowed]
        if not names:
            raise RuntimeError("The requested camera or STEM detectors are not available.")
        aliases = {}
        for name in names:
            detector = getattr(client.detectors, f"get_{kind}_detector")(name)
            aliases[name] = [name]
            for field in ("name", "display_name"):
                row, value = reading(f"{name}.{field}", lambda f=field: getattr(detector, f))
                if row["status"] == "ok" and isinstance(value, str) and value:
                    aliases[name].append(value)
        return {"optical_mode": optical, "projector_mode": projector, "detectors": names,
                "available_detectors": list(available),
                "detector_aliases": aliases, "route": request.route,
                "checked_utc": utc_now(), "serial_number": self.identity["serial_number"]}

    def _streams(self, route):
        attr = "continuous_stem_acquisitions" if route == "stem_all" else "continuous_camera_acquisitions"
        streams = getattr(self.require_client().acquisition, attr)
        if not isinstance(streams, list) or not streams:
            raise RuntimeError("No current signal is exposed by AutoScript. Start the intended live acquisition "
                               "on the microscope and try again. Vendor GUI streams may not be exposed; "
                               "no fallback exposure or default acquisition settings will be used.")
        return streams

    @staticmethod
    def _channel(image, context):
        """An existing stream is not owned by this request; require image detector identity."""
        try:
            reported = image.metadata.binary_result.detector
        except Exception:
            reported = None
        reported = reported if isinstance(reported, str) and reported.strip() else None
        candidates = [name for name, aliases in context["detector_aliases"].items() if reported in aliases]
        return {"detector": candidates[0] if len(candidates) == 1 else None,
                "reported_detector": reported,
                "identity_source": "vendor binary-result metadata" if len(candidates) == 1 else "unresolved"}

    def _guard(self):
        paths = ["service.system.serial_number", "service.autoscript.server.is_offline",
                 "optics.optical_mode", "optics.projector_mode", "optics.objective_lens_mode",
                 "source.acceleration_voltage.value", "optics.magnification.value",
                 "optics.camera_length.value", "specimen.stage.position", "specimen.stage.is_moving",
                 "specimen.piezo_stage.position", "specimen.piezo_stage.is_moving"]
        paths += [p["path"] for p in catalog()["properties"] if
                  p["path"].startswith(("optics.deflectors", "optics.stigmators", "optics.focusing."))]
        rows = [reading(p, lambda p=p: resolve(self.client, p))[0] for p in paths]
        enum, lenses = reading("optics.lenses.get_available", lambda: self.client.optics.lenses.get_available())
        rows.append(enum)
        if enum["status"] == "ok" and isinstance(lenses, list):
            for name in lenses:
                rows.append(reading(f"optics.lenses[{name}].value_raw",
                                    lambda n=name: self.client.optics.lenses.get_lens(n).value_raw)[0])
        return {"readings": rows}

    def capture(self, request: CaptureRequest, cancel: Event | None = None,
                progress=lambda text: None):
        request.validate()
        check_cancel(cancel)
        initial_context = self._preflight(request)
        self._streams(request.route)  # Fail before making a misleading image/state record.
        bundle = RecordBundle(request, self.identity.copy(), initial_context)
        previews, read_errors, ignored = [], [], []
        collection_start = utc_now()
        bundle.manifest["collection_started_utc"] = collection_start
        try:
            before = self.snapshot(cancel, lambda s: progress("Before: " + s))
            bundle.json_artifact("parameters_before.json", before)
            before_rows = {row["path"]: row for row in before["readings"]}
            bundle.manifest["machine_context_before"] = {
                key: before_rows[key] for key in (
                    "optics.optical_mode", "optics.projector_mode", "optics.magnification.value",
                    "optics.camera_length.value", "source.acceleration_voltage.value",
                    "optics.spot_size_index", "optics.probe_mode", "optics.illumination_mode",
                ) if key in before_rows}
            guard_before = self._guard()
            bundle.json_artifact("guard_before.json", guard_before)
            context = self._preflight(request)
            if any(context[key] != initial_context[key] for key in
                   ("optical_mode", "projector_mode", "detectors")):
                raise RuntimeError("Mode or detector list changed during readbacks; collect again after settings settle.")
            bundle.manifest["preflight"] = context
            streams = self._streams(request.route)
            bundle.manifest["stream_count"] = len(streams)
            bundle.update()
            seen = set()
            frame_index = 0
            for stream_index, stream in enumerate(streams):
                targets = context["detectors"] if request.route == "stem_all" else [None]
                for target in targets:
                    if cancel and cancel.is_set():
                        break
                    if target in seen:
                        continue
                    progress(f"Reading current signal {target or 'camera'} (stream {stream_index + 1})...")
                    if cancel and cancel.is_set():
                        break
                    try:
                        if target is None:
                            image = stream.wait_for_next_frame()
                        else:
                            image = stream.wait_for_next_frame(detector_type=target)
                    except Exception as exc:
                        read_errors.append(f"Stream {stream_index + 1}, {target or 'camera'}: {exc}")
                        continue
                    returned_utc = utc_now()
                    channel = self._channel(image, context)
                    # Known other camera images are not silently relabelled as the requested camera.
                    if (channel["detector"] is None and request.route != "stem_all"
                            and channel["reported_detector"] in context["available_detectors"]):
                        ignored.append({"stream": stream_index + 1,
                                        "reported_detector": channel["reported_detector"]})
                        continue
                    name = channel["detector"]
                    if name in seen:
                        ignored.append({"stream": stream_index + 1, "duplicate_detector": name})
                        continue
                    metadata = collect_frame_metadata(image)
                    channel.update(stream_index=stream_index + 1, retrieved_utc=returned_utc,
                                   timing=frame_timing(metadata, collection_start, returned_utc),
                                   acquisition_id=next((r.get("value") for r in metadata["readings"]
                                       if r["path"] == "acquisition.acquisition_id" and r["status"] == "ok"), None))
                    frame_index += 1
                    prefix = f"channel_{frame_index:03d}"
                    try:
                        pixels = bundle.save_image(image, channel, prefix)
                        previews.append({"label": name or "Unidentified signal", "pixels": pixels})
                        if name:
                            seen.add(name)
                        bundle.json_artifact(f"{prefix}/image_metadata.json", metadata)
                    except Exception as exc:
                        read_errors.append(f"{prefix}: {type(exc).__name__}: {exc}")
                if cancel and cancel.is_set():
                    break
            bundle.manifest["collection_finished_utc"] = utc_now()
            bundle.manifest["ignored_frames"] = ignored
            guard_after = self._guard()
            bundle.json_artifact("guard_after.json", guard_after)
            bundle.manifest["guard_comparison"] = compare_snapshots(guard_before, guard_after)
            progress("Saving system readbacks; the microscope acquisition is unchanged.")
            after = self.snapshot(None, lambda s: progress("After: " + s))
            bundle.json_artifact("parameters_after.json", after)
            bundle.manifest["parameter_comparison"] = compare_snapshots(before, after)
            bundle.manifest["stop_requested"] = bool(cancel and cancel.is_set())
            missing = [name for name in context["detectors"] if name not in seen] if request.route == "stem_all" else []
            images = bundle.manifest["images"]
            unresolved = sum(item["detector"] is None for item in images)
            complete = bool(seen) and not missing and not unresolved
            bundle.manifest["channel_coverage"] = {
                "requested": context["detectors"], "saved": len(images), "missing": missing,
                "unresolved": unresolved, "read_errors": read_errors, "complete": complete}
            coverage = before["coverage"]["unavailable"] + after["coverage"]["unavailable"]
            changed = bundle.manifest["guard_comparison"]["changed"]
            timing = [item["timing"]["status"] for item in images]
            acquisition_ids = [item["acquisition_id"] for item in images]
            bundle.manifest["quality"] = {
                "parameter_coverage": "partial" if coverage else "all catalogued reads succeeded",
                "unavailable_reads": coverage, "changed_guard_fields": changed,
                "state_stability": "changed" if changed else "endpoint comparison only",
                "image_state_association": "sequential readbacks; not an atomic exposure snapshot",
                "frame_timing": timing,
                "simultaneous_channels": "not established; frames are read separately",
                "vendor_acquisition_ids": acquisition_ids,
                "calibration_status": "Unvalidated raw instrument observation",
            }
            bundle.manifest["warnings"].extend(read_errors)
            if not seen:
                bundle.manifest["warnings"].append(
                    "No signal could be identified as the requested detector. No default exposure was started.")
            if missing:
                bundle.manifest["warnings"].append("No current frame received for: " + ", ".join(missing))
            if not complete:
                status = "partial_channels" if images else "failed"
            elif "outside_collection_window" in timing:
                status = "frame_state_mismatch"
            elif changed or "unverified" in timing:
                status = "frame_state_unverified"
            else:
                status = "recorded_with_warnings" if coverage or read_errors or bundle.manifest["warnings"] else "recorded"
            bundle.finish(status)
        except Exception as exc:
            status = "cancelled_before_signal" if isinstance(exc, Cancelled) else "failed"
            if any(name.endswith("image.npy") for name in bundle.manifest["artifacts"]):
                status = "image_saved_incomplete_metadata"
            bundle.finish(status, f"{type(exc).__name__}: {exc}")
        return {"path": str(bundle.path), "manifest": bundle.manifest, "previews": previews}
