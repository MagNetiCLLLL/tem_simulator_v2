# Instrument Recorder

Open **Microscope > Instrument Recorder...** in the simulator, or run
`python -m temsim.recorder`. The independent window contains three collection
buttons and one last-signal preview:

- **Collect screen camera**: current screen camera signal, in any optical mode.
- **Collect pixelated camera**: current pixelated-camera signal in TEM optical mode, including diffraction.
- **Collect all STEM detectors**: current signals from the available STEM detectors
  in STEM optical mode. A channel selector appears above the single preview.

There are no specimen labels, particle IDs, manual image-size, exposure or dwell
inputs. No specimen type or composition is assumed. Successful and partial
records receive unique automatic names; a failed collection retains the previous
preview with its original record ID.

## Workflow

1. Use **Connection > Connect...** to connect to a real AutoScript TEM 1.18 server.
2. Set the instrument optics and run the desired acquisition on the microscope.
3. Click the corresponding collect button. Image data, original image metadata
   and system readbacks are saved automatically.
4. Change the microscope settings and repeat. Keep the instrument settings stable
   while collection is running.

The default output is **`instrument_records/` inside the project folder**.
**Records > Choose output folder...** changes it; the choice and window geometry
are remembered. **Records > Last record details...** shows the saved paths,
record identity and diagnostic information without adding controls to the main
page. Files are eligible for Git; collection does not commit or push.

## Current-signal API boundary

The locally supplied AutoScript TEM **client 1.18.0** wheel is the API authority.
Install the licensed client and its dependencies in the application's Python
environment. Opening the recorder does not instantiate a client or connect.

The recorder reads `continuous_camera_acquisitions` or
`continuous_stem_acquisitions` and calls `wait_for_next_frame()` on existing
handles; STEM reads include a detector filter. This call has a documented
10-second timeout. Reading can consume a buffered frame; coordinate access with
other clients that use the same acquisition.

**This does not guarantee access to every image window in Velox or other vendor
software.** An acquisition must be exposed through these AutoScript handles.
When it is not exposed, the recorder reports that no current signal is available.
It does not silently start an exposure with default settings.

No start/stop acquisition command, microscope mode change, detector insertion,
lens adjustment, scan reconfiguration, hidden pixel count, exposure, or dwell is
supplied. There is no generic "acquire using the current vendor GUI settings"
method in the audited interface. Three-button collection uses the existing
signal path, not a replacement for that missing API.

STEM channels are read separately from existing streams. Their individual vendor
acquisition IDs and timestamps are retained; they are **not asserted to be one
simultaneous scan**. Unavailable channels are reported, not synthesised or silently
omitted. More than one available pixelated-camera stream can yield more than one identified
pixelated-camera signal. Detector identity comes from image metadata matched to live device
names, not camera defaults or the order in which frames arrive.

## Image-to-system association

Each collection has a unique record ID, actual optical/projector mode, detector
identities, UTC read intervals, per-file hashes, and before/after system snapshots.

Image metadata remains separate from current system readbacks:

- Original SDK XML is preserved unchanged, including available acquisition
  settings and image calibration.
- Audited image fields, including dwell, scan dimensions, exposure, binning,
  pixel size and acquisition timestamps, are additionally saved in
  `image_metadata.json`. Missing fields retain explicit errors or nulls.
- Values retain native vendor conventions. No guessed unit conversions,
  sample metadata, or acquisition defaults overwrite original values.
- Complete image timing is checked against the local collection interval only
  when both vendor timestamps have explicit timezone offsets. Unknown timestamps,
  missing timezones and out-of-window frames remain flagged.
- Host/instrument clock synchronisation is not verified. Matching timestamps
  and unchanged endpoint readings do not prove an atomic exposure-state snapshot.
- A state change, unverified frame time, or stale frame is not presented as a
  verified image/current-state pair.

A versioned getter catalog attempts public passive system properties and
enumerates available lenses, deflectors, stigmators, apertures, alignments,
detectors and related controls. Unavailable reads keep their path and error.
"All readable" means the documented API subset exposed by this microscope,
not inaccessible OEM internal state.

Nominal magnification and image-measured relative magnification are distinct.
Corrected image rotation is not automatically interpreted as lens/Larmor
rotation. Assembly dimensions and lens locations may later be jointly fitted
across recorded states. This recorder does no image matching, specimen analysis,
magnification measurement, inverse modelling or simulation-cache modification.

Original EMD import/association is not implemented. Existing EMD files cannot be
retrospectively assigned the current instrument state as their historical state.

## Files

New records use schema `temsim.instrument-record/3`. Earlier v1/v2 files remain
unchanged; their original exposure requests and specimen labels are historical
metadata only and are not used by this collector.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Record ID, actual modes, image list, detector identities, retrieval timing, coverage and quality flags |
| `channel_NNN/image.npy` | Lossless original pixel array, native shape and dtype, including non-square images |
| `channel_NNN/image.tiff` | Unnormalised TIFF export, when available |
| `channel_NNN/vendor_metadata.xml` | Original SDK image metadata |
| `channel_NNN/image_metadata.json` | Audited image-specific metadata and read failures |
| `parameters_before.json`, `parameters_after.json` | Full attempted system snapshots |
| `guard_before.json`, `guard_after.json` | Optical-state endpoint comparisons |
| `api_catalog.json` | Getter/structure descriptions, source SDK version and wheel hash |

The output root also receives `records.jsonl`. The per-record manifest remains
authoritative if appending to that index fails. Use its image paths and SHA-256
hashes to resolve the image-to-record association.

Statuses distinguish `recorded_with_warnings`, `partial_channels`,
`frame_state_unverified`, `frame_state_mismatch`, and
`image_saved_incomplete_metadata`. Returned images are saved before subsequent
system reads; a later failure does not erase them. No automatic new exposure
attempts are used to replace a missing signal.

**Records > Cancel collection** stops between SDK calls. It does not interrupt a
blocking read or stop the microscope acquisition. Already returned frames and
post-readbacks are retained. Closing waits for work and confirmed disconnection.

## Validation

Focused offline fake-stream tests cover all three routes, absence of acquisition
parameter defaults, lossless pixels and metadata, identity filtering, old/unknown
frame timestamps, partial signals, cancellation, connection lifecycle and the
minimal GUI. SDK structures can be checked locally without a microscope client.

Real microscope stream visibility, concurrent-reader buffer behaviour,
instrument/host clock alignment, network failures and acquisition timing remain
unvalidated. No hardware measurements or calibration results are implied by
offline tests.
