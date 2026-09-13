"""Lossless image/parameter bundles with unique, non-overwriting identities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
import tifffile

from .snapshot import catalog, utc_now


@dataclass(frozen=True)
class CaptureRequest:
    output: str
    route: str = "flucam"

    def validate(self):
        if self.route not in ("flucam", "ceta", "stem_all"):
            raise ValueError("Select Flucam, Ceta or all STEM detectors.")
        if not self.output.strip():
            raise ValueError("Output directory is required.")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-_")[:48] or "record"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RecordBundle:
    def __init__(self, request: CaptureRequest, identity: dict, context: dict):
        request.validate()
        self.root = Path(request.output).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError("Choose an existing output directory.")
        stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
        record_id = uuid.uuid4().hex
        self.path = self.root / (f"{stamp}_{request.route}_"
                                 f"{slug(context['optical_mode'])}_{slug(context['projector_mode'])}_"
                                 f"{record_id[:12]}")
        self.path.mkdir(exist_ok=False)
        self.manifest = {
            "schema": "temsim.instrument-record/3", "record_id": record_id,
            "created_utc": utc_now(), "status": "pending", "request": asdict(request),
            "instrument": identity, "artifacts": {}, "warnings": [], "images": [],
            "acquisition_context": context,
            "acquisition_settings_source": "Existing microscope stream; no acquisition settings supplied or changed.",
            "capture_kind": "existing_stream_frame",
            "sample_identity": None,
            "physical_interpretation": {
                "sample_identity": "Unspecified. No specimen type or particle assumption.",
                "readbacks": "Native SDK values; optical units are not calibrated currents or fields.",
                "coherence": "Intensity images and readbacks do not measure complex wave phase.",
                "simulator": "No automatic simulator parameter fitting or cache modification.",
                "geometry": "Assembly dimensions may be joint-fit variables across records; not fixed by readbacks.",
            },
        }
        self.update()
        self.json_artifact("api_catalog.json", catalog())

    def update(self):
        # Only our manifest is replaced; the unique directory never reuses a capture.
        target = self.path / "manifest.json"
        temp = self.path / "manifest.json.tmp"
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)

    def register(self, name):
        path = self.path / name
        self.manifest["artifacts"][name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        self.update()

    def json_artifact(self, name, data):
        with (self.path / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.register(name)

    def save_image(self, image, channel: dict, prefix=""):
        data = np.array(image.data, copy=True)
        if data.ndim not in (2, 3) or data.size == 0 or data.dtype.kind not in "buif":
            raise ValueError("SDK returned an empty or unsupported intensity image.")
        folder = self.path / prefix
        folder.mkdir(exist_ok=True)
        name = lambda file: f"{prefix}/{file}" if prefix else file
        with (folder / "image.npy").open("xb") as stream:
            np.save(stream, data, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.register(name("image.npy"))  # Durable raw pixels precede optional exports/readbacks.
        self.manifest["images"].append({**channel, "path": name("image.npy"),
            "dtype": str(data.dtype), "shape": list(data.shape),
            "orientation": "Native SDK array order; no transpose or normalisation."})
        try:
            with (folder / "image.tiff").open("xb") as stream:
                tifffile.imwrite(stream, data, metadata=None)
                stream.flush()
                os.fsync(stream.fileno())
            self.register(name("image.tiff"))
        except Exception as exc:
            self.manifest["warnings"].append(f"TIFF export failed; NPY retained: {exc}")
        try:
            xml = image.metadata.metadata_as_xml
            if not isinstance(xml, str) or not xml.strip():
                raise ValueError("SDK image metadata XML is empty.")
            with (folder / "vendor_metadata.xml").open("x", encoding="utf-8") as stream:
                stream.write(xml)
                stream.flush()
                os.fsync(stream.fileno())
            self.register(name("vendor_metadata.xml"))
        except Exception as exc:
            self.manifest["warnings"].append(f"Vendor metadata unavailable: {exc}")
        self.update()
        return data

    def finish(self, status, error=None):
        self.manifest["status"] = status
        self.manifest["finished_utc"] = utc_now()
        if error:
            self.manifest["error"] = error
        self.update()
        row = {"record_id": self.manifest["record_id"], "directory": self.path.name,
               "status": status, "created_utc": self.manifest["created_utc"],
               **{key: self.manifest["request"][key]
                  for key in ("route",)},
               "images": self.manifest["images"],
               "acquisition_context": self.manifest["acquisition_context"]}
        row["machine_context_before"] = self.manifest.get("machine_context_before", {})
        try:
            # Independent manifests remain authoritative if a shared index is unavailable.
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(self.root / "records.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                if os.write(fd, payload) != len(payload):
                    raise OSError("Incomplete index append")
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            self.manifest["warnings"].append(f"Index append failed: {exc}; use manifest.json.")
            self.update()
