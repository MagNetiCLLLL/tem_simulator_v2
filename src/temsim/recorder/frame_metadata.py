"""Audited image readbacks, kept separate from the current system snapshot."""
from __future__ import annotations

from datetime import datetime

from .snapshot import reading, resolve


def collect_frame_metadata(image):
    paths = (
        "acquisition.acquisition_id", "acquisition.acquisition_start_date_time",
        "acquisition.acquisition_date_time", "binary_result.detector",
        "binary_result.image_size", "binary_result.pixel_size",
        "binary_result.pixel_size_x_unit", "binary_result.pixel_size_y_unit",
        "scan_settings.dwell_time", "scan_settings.scan_size", "scan_settings.scan_area",
        "scan_settings.frame_time", "scan_settings.scan_rotation",
    )
    rows = [reading(path, lambda p=path: resolve(image.metadata, p))[0] for path in paths]
    enum, detectors = reading("imaging_detectors", lambda: image.metadata.imaging_detectors)
    # Individual fields remain readable even if the whole structure is not in the catalog.
    rows.append(enum)
    if isinstance(detectors, list):
        for index, detector in enumerate(detectors):
            for field in ("detector_name", "detector_type", "exposure_time", "exposure_time_2",
                          "binning", "readout_area", "post_magnification"):
                rows.append(reading(f"imaging_detectors[{index}].{field}",
                                    lambda d=detector, f=field: getattr(d, f))[0])
    return {"source": "Original SDK image metadata, not current system settings or user inputs",
            "units": "Native vendor metadata; consult original XML. No inferred unit conversions.",
            "readings": rows}


def frame_timing(metadata, window_start, window_end):
    """Never substitute collection time for exposure time or guess a camera timezone."""
    values = {r["path"]: r.get("value") for r in metadata["readings"] if r["status"] == "ok"}
    start = values.get("acquisition.acquisition_start_date_time")
    end = values.get("acquisition.acquisition_date_time")
    result = {"vendor_start": start, "vendor_time": end,
              "recorder_start_utc": window_start, "recorder_end_utc": window_end,
              "status": "unverified", "clock_synchronisation": "not verified"}
    try:
        times = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (start, end)]
        if any(t.tzinfo is None or t.utcoffset() is None for t in times):
            return result
        lower, upper = (datetime.fromisoformat(v) for v in (window_start, window_end))
        if not lower <= times[0] <= times[1] <= upper:
            result["status"] = "outside_collection_window"
        else:
            result["status"] = "timestamps_within_collection_window"
    except (AttributeError, TypeError, ValueError):
        pass
    return result
