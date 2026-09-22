"""Executed particle sections, stored as checked JSON and numeric arrays.

This is a cache of tip-origin transport, never a configurable downstream source.
Restoration admits a fixed list of data classes; it never imports a class named
by a file and never deserializes pickle. Current inputs are checked again by the
section solver before any saved upstream calculation is reused.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import math
import json
from pathlib import Path
import stat

import numpy as np

from temsim.immutable_json import json_digest, thaw_json

SECTION_PACKAGE_SCHEMA = "optical-particle-section-package-v2"
_GRAPH_ARRAY = "section_record_graph_utf8"
_SNAPSHOT_RESULT_FIELDS = frozenset({"state_snapshot", "layout", "assembly"})
_RESULT_FIELDS = frozenset({
    "simulation", "energy_filter", "state_snapshot", "layout", "assembly",
    "specimen_interactions", "wave_imaging", "scan_geometry", "scan_ray_paths",
    "stem_scan", "specimen_exit", "sample_region", "lens_crossovers", "aperture_stops",
    "model_signature", "signatures", "calculated_products", "reused_products", "cache_hit",
    "performance", "external_inputs", "calculation_manifest", "working_point_parent_id",
    "particle_signals",
})


def normalise_section_request(request, *, quality=None):
    if request is None:
        return None
    if quality is not None and quality not in {"Preview", "Medium", "High accuracy"}:
        raise ValueError("Unknown particle-section calculation quality")
    if not isinstance(request, Mapping) or set(request) != {"target_z_mm", "component_keys"}:
        raise ValueError("Section tuning requires a target plane and selected components")
    target = float(request["target_z_mm"])
    if not math.isfinite(target):
        raise ValueError("The section plane must be finite")
    raw_keys = request["component_keys"]
    if not isinstance(raw_keys, (list, tuple)) or any(not isinstance(k, str) or not k for k in raw_keys):
        raise ValueError("Select components to tune in this section")
    keys = tuple(sorted(set(raw_keys)))
    return {"target_z_mm": target, "component_keys": keys}


def section_result_quality(result):
    metrics = result.simulation.metrics
    quality = metrics.get("tuning_quality")
    if quality not in {"Preview", "Medium", "High accuracy"}:
        quality = "High accuracy" if not metrics.get("optical_tuning", False) else "Preview"
    return quality


def section_archive_identity(result):
    """Stable request identity for archive deduplication, not transport admission."""
    checkpoint = getattr(result.simulation, "section_checkpoint", None)
    if checkpoint is None:
        raise ValueError("This result has no executed section checkpoint to archive")
    metrics = result.simulation.metrics
    return json_digest({"schema": SECTION_PACKAGE_SCHEMA,
        "signatures": dict(getattr(result, "signatures", None) or {}),
        "model": getattr(result, "model_signature", None),
        "quality": section_result_quality(result),
        "target_z_mm": float(metrics["section_target_z_mm"]),
        "component_keys": sorted(metrics.get("section_component_keys", ())),
        "gun": checkpoint.gun_dependency_signature})


def _section_plane_name(state, target):
    from temsim.component_names import (
        APERTURE_NAMES, LENS_NAMES, RECORDING_PLANE_NAMES, STIGMATOR_NAMES, DEFLECTOR_NAMES,
    )
    choices = [(float(state.electron_gun.exit_plane_z_mm), "Gun exit"),
               (float(state.sample.z_mm), "Specimen reference plane")]
    energy_filter = getattr(state, "energy_filter", None)
    if energy_filter is not None and bool(energy_filter.enabled):
        choices.append((float(energy_filter.entrance_z_mm), "Energy-filter entrance"))
    for collection, names in (("lenses", LENS_NAMES), ("apertures", APERTURE_NAMES),
            ("recording_planes", RECORDING_PLANE_NAMES), ("stigmators", STIGMATOR_NAMES),
            ("deflectors", DEFLECTOR_NAMES)):
        for component in getattr(state, collection, ()):
            if component.key in names and hasattr(component, "z_mm"):
                choices.append((float(component.z_mm), names[component.key]))
    return next((name for z, name in choices if math.isclose(z, target, rel_tol=0., abs_tol=5e-7)),
                "Custom plane")


def section_archive_summary(result, *, path=None, saved_at_utc=None):
    """Small UI metadata, explicitly separate from successful disk persistence."""
    metrics = result.simulation.metrics
    target = float(metrics["section_target_z_mm"])
    checkpoint = getattr(result.simulation, "section_checkpoint", None)
    if checkpoint is None:
        raise ValueError("This result has no executed section checkpoint to archive")
    resume = metrics.get("section_resume_z_mm")
    details = []
    reasons = {
        "no_prior_checkpoint": "No previously executed section was available.",
        "upstream_inputs_or_model_changed": "Saved upstream inputs, numerical settings or solver version did not match.",
        "invalid_saved_state": "The previous restart state failed validation and was not reused.",
        "no_compatible_column_prefix": "No compatible executed column prefix was available.",
        "vacuum_restart_required": "Vacuum transport required a new execution.",
        "compatible_executed_prefix": "A compatible executed prefix was reused.",
    }
    reason = metrics.get("section_reuse_reason")
    if reason:
        details.append(reasons.get(reason, str(reason)))
    if isinstance(resume, (int, float)) and math.isfinite(resume):
        details.append(("Upstream state reused through" if metrics.get("section_reused_prefix")
                        else "Column transport calculated from") + f" Z = {resume:.9g} mm.")
    if metrics.get("section_gun_reused") is False:
        details.append("Gun was calculated from tip emission; no compatible saved gun was reused.")
    if metrics.get("section_vacuum_restart") == "recomputed":
        details.append("Vacuum participation required recalculation.")
    material = metrics.get("material_section_resume", ())
    for row in material:
        if isinstance(row, Mapping) and isinstance(row.get("resume_z_mm"), (int, float)):
            details.append(f"Specimen-exit path resumed from Z = {row['resume_z_mm']:.9g} mm.")
    resumable = metrics.get("section_resumable_through_z_mm", target)
    from temsim.detector.eds_signal import EDSSpectrum
    material_cache = getattr(result.simulation, "material_section_cache", None)
    spectrum = getattr(material_cache, "eds_spectrum", None)
    eds_completed = (isinstance(spectrum, EDSSpectrum)
                     and spectrum.elastic_transport is material_cache.elastic_transport)
    return {"identity": section_archive_identity(result), "target_z_mm": target,
            "resumable_through_z_mm": float(resumable),
            "energy_filter_completed": getattr(result, "energy_filter", None) is not None,
            "eds_completed": eds_completed,
            "plane_name": _section_plane_name(result.state_snapshot, target),
            "quality": section_result_quality(result),
            "particle_count": int(len(checkpoint.gun_trace.exit_bundle.x_m)),
            "saved_at_utc": saved_at_utc,
            "path": None if path is None else str(Path(path).resolve()),
            "resume_details": " ".join(dict.fromkeys(details)),
            "resume_policy": "Matching upstream state can resume here; changed source, fields, material or numerical settings are checked and recalculated where required."}


def section_file_fingerprint(path):
    """Cheap identity of the exact file checked by an archive worker."""
    record = Path(path).stat()
    if not stat.S_ISREG(record.st_mode):
        raise ValueError("Particle-section archive is not a regular file")
    return (record.st_dev, record.st_ino, record.st_size,
            record.st_mtime_ns, record.st_ctime_ns)


def checked_section_archive_info(result, path, *, maximum_unpacked_bytes,
                                 expected_package_digest=None, verify_payload=False):
    """Bind a result to a stable archive without reconstructing its large arrays.

    Fresh atomic saves need only their checked manifest digest. An existing file
    not already verified in this process also receives streaming content checks.
    This function runs in the file worker, never in a GUI reuse check.
    """
    from hashlib import sha256
    from zipfile import ZipFile
    from temsim.working_point_archive import WorkingPointArchiveIndex
    before = section_file_fingerprint(path)
    index = WorkingPointArchiveIndex.read(path, maximum_unpacked_bytes=maximum_unpacked_bytes)
    metadata = thaw_json(index.metadata)
    expected = section_archive_summary(result)
    stored = metadata.get("archive_summary", {})
    if (metadata.get("package_kind") != SECTION_PACKAGE_SCHEMA
            or any(stored.get(key) != expected[key] for key in (
                "identity", "target_z_mm", "resumable_through_z_mm", "quality",
                "particle_count", "energy_filter_completed", "eds_completed"))
            or metadata.get("signatures") != dict(result.signatures)
            or metadata.get("model_signature") != result.model_signature
            or index.plane_z_mm != expected["target_z_mm"]
            or (expected_package_digest is not None and index.digest != expected_package_digest)):
        raise ValueError("Existing section archive belongs to a different calculation")
    if verify_payload:
        # Current section writers store C-order arrays. Hash bounded byte chunks
        # rather than allocating or decoding a second particle result for reuse.
        with ZipFile(path) as archive:
            for record in index.array_records.values():
                digest = sha256()
                with archive.open(record["entry"]) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError("Unsupported numeric NPY header version")
                    _, fortran_order, _ = reader(stream, max_header_size=16384)
                    if fortran_order:
                        raise ValueError("Current particle archives require C-order numeric arrays")
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("Particle-section numeric content checksum mismatch")
    if section_file_fingerprint(path) != before:
        raise ValueError("Section archive changed during verification; retry the operation")
    return dict(stored, path=str(Path(path).resolve()), _file_fingerprint=before,
                _package_digest=index.digest)


def archive_section_result(result, directory, *, maximum_unpacked_bytes=8*1024**3):
    """Write one automatic archive per exact request without deleting any archive."""
    summary = section_archive_summary(result)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    quality = summary["quality"].lower().replace(" ", "-")
    path = directory / f"section-{quality}-{summary['identity']}.temsection"
    if path.exists():
        info = checked_section_archive_info(result, path,
            maximum_unpacked_bytes=maximum_unpacked_bytes, verify_payload=True)
        return dict(info, reused=True)
    package = save_section_result(result, path, maximum_unpacked_bytes=maximum_unpacked_bytes)
    info = checked_section_archive_info(result, path, maximum_unpacked_bytes=maximum_unpacked_bytes,
                                        expected_package_digest=package.digest)
    return dict(info, reused=False)


def section_request_signatures(signatures, request):
    request = normalise_section_request(request)
    result = dict(signatures)
    result["section"] = json_digest({"schema": SECTION_PACKAGE_SCHEMA, **request})
    result["request"] = json_digest({"base_request": result["request"], "section": result["section"]})
    return result


def _data_classes():
    from temsim.physics.particle_sections import ParticleSectionCheckpoint, ParticleSectionSegment, MaterialSectionCache
    from temsim.physics.simulation import Simulation, Branch
    from temsim.physics.core import AxialPropagationPlan, PropagationCheckpoints
    from temsim.optics.electron_gun.base import (
        GunTraceResult, GunExitBundle, GunPlaneArrival, GunEqualTimeHistory,
    )
    from temsim.physics.lens_field_provider import (
        FrozenMappedField, MagneticFieldMap, CoordinateRegistration, FieldMapProvenance,
    )
    from temsim.specimen.elastic_transport import (
        ElasticTransportResult, ElasticTerminalBundle, ElasticTrajectory,
        ElasticScatterEvent, ElasticMaterialFlight,
    )
    from temsim.specimen.inelastic import RealInteractionDistribution, RealInteractionChannel
    from temsim.specimen.downstream_transport import GeometricSpecimenExit
    from temsim.detector.eds_signal import (
        ElectronTrackSegment, EDSMaterial, EDSVacancySignal, EDSLineSignal,
        EDSMaterialQuadrature, EDSSpectrum,
    )
    from temsim.detector.eds_photon_transport import (
        EDSPhotonRay, PhotonMaterialInterval, EDSPhotonPathResult,
        EDSPhotonTransportResult,
    )
    from temsim.detector.eds_response import EDSResponseRates
    from temsim.specimen.interaction_types import (
        IncidentElectronRay, IncidentRayBundle, InteractionEvent, SpecimenModelCoupling,
        ConservationCheck, SpecimenInteractionRequest, SpecimenInteractionResult,
    )
    from temsim.specimen.scene import SpecimenScene, SceneMaterialRegion
    from temsim.specimen.support import SupportGrid, SupportMaterial, SupportMesh
    from temsim.detector.particle_readout import ParticleDetectorReadout
    from temsim.optics.energy_filter_raytrace import EnergyFilterResult, FilterPlaneArrival
    from temsim.detector.stem_signal import BeamCurrentSignal, DetectorSignal, CollectionAngle
    from temsim.detector.eels_forward import EELSForwardResult, SpectrometerTransmission
    from temsim.physics.scan_geometry import ScanGeometryResult, ScanRayPathResult, DescanCalibrationResult
    from temsim.detector.stem_signal import StemScanResult
    from temsim.physics.probe_state import ProbeState, ProbeEnergyBin
    from temsim.specimen.sample_region import SampleRegionResult, SampleRegionElectronPath, SampleRegionPhotonPath
    from temsim.calculation_manifest import CalculationManifest, SolverIdentity, ExternalInputIdentity
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.physics.first_order import TransverseTransfer, LinearMapProperties, DetectorFrameCalibration
    from temsim.diagnostics import OpticalTransferRecord
    return {cls.__name__: cls for cls in (
        ParticleSectionCheckpoint, ParticleSectionSegment, Simulation, Branch,
        AxialPropagationPlan, PropagationCheckpoints, GunTraceResult, GunExitBundle,
        GunPlaneArrival, GunEqualTimeHistory, FrozenMappedField, MagneticFieldMap,
        CoordinateRegistration, FieldMapProvenance, MaterialSectionCache,
        ElasticTransportResult, ElasticTerminalBundle, ElasticTrajectory,
        ElasticScatterEvent, ElasticMaterialFlight, RealInteractionDistribution,
        RealInteractionChannel, GeometricSpecimenExit, ElectronTrackSegment, EDSMaterial,
        EDSVacancySignal, EDSLineSignal, EDSMaterialQuadrature, EDSSpectrum,
        EDSPhotonRay, PhotonMaterialInterval, EDSPhotonPathResult, EDSPhotonTransportResult,
        EDSResponseRates,
        IncidentElectronRay, IncidentRayBundle, InteractionEvent, SpecimenModelCoupling,
        ConservationCheck, SpecimenInteractionRequest, SpecimenInteractionResult,
        SpecimenScene, SceneMaterialRegion, SupportGrid, SupportMaterial, SupportMesh,
        ParticleDetectorReadout, EnergyFilterResult, FilterPlaneArrival,
        BeamCurrentSignal, DetectorSignal, CollectionAngle, EELSForwardResult,
        SpectrometerTransmission,
        ScanGeometryResult, ScanRayPathResult, DescanCalibrationResult, StemScanResult,
        ProbeState, ProbeEnergyBin, SampleRegionResult, SampleRegionElectronPath, SampleRegionPhotonPath,
        CalculationManifest, SolverIdentity, ExternalInputIdentity, InstrumentSnapshot,
        TransverseTransfer, LinearMapProperties, DetectorFrameCalibration, OpticalTransferRecord,
    )}


def _enum_types():
    from temsim.column.layout import C3Hardware, CorrectorAssembly, Branch
    from temsim.specimen.interaction_types import SpecimenObservable, InteractionProcess
    return {cls.__module__ + ":" + cls.__name__: cls for cls in
            (C3Hardware, CorrectorAssembly, Branch, SpecimenObservable, InteractionProcess)}


def _pack(value, arrays, memo, types):
    if isinstance(value, Enum):
        name = type(value).__module__ + ":" + type(value).__name__
        if _enum_types().get(name) is not type(value):
            raise ValueError("Unsupported section enumeration")
        return {"enum": name, "value": value.value}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or value.dtype.fields is not None:
            raise ValueError("Section checkpoints require plain numeric arrays")
        key = memo.get(id(value))
        if key is None:
            key = f"section_array_{len(arrays):06d}"
            memo[id(value)] = key
            arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays, memo, types)
    if is_dataclass(value):
        name = type(value).__name__
        if types.get(name) is not type(value):
            raise ValueError(f"Unsupported section data type: {name}")
        if id(value) in memo:
            key = memo[id(value)]
            if key is None:
                raise ValueError("Cyclic section records are not supported")
            return {"ref": key}
        key = f"record_{len(memo)}"
        memo[id(value)] = None
        record = {"record_id": key, "type": name, "fields": {
            f.name: _pack(getattr(value, f.name), arrays, memo, types)
            for f in fields(value) if f.init
        }}
        memo[id(value)] = key
        return record
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Section metadata requires string keys")
        return {"mapping": {k: _pack(v, arrays, memo, types) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"tuple" if isinstance(value, tuple) else "list": [_pack(v, arrays, memo, types) for v in value]}
    if isinstance(value, frozenset):
        return {"frozenset": [_pack(v, arrays, memo, types) for v in sorted(value)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": value.hex()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported section metadata: {type(value).__name__}")


def _same_executed_record(left, right):
    """Compare complete decoded records before restoring a shared object link.

    Numeric arrays normally share the archive's array entry already. The exact
    fallback comparison also handles separately stored but equal entries; it
    must not accept a different trajectory merely because summary counts agree.
    """
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (left.shape == right.shape and left.dtype == right.dtype
                and np.array_equal(np.ascontiguousarray(left).view(np.uint8),
                                   np.ascontiguousarray(right).view(np.uint8)))
    if is_dataclass(left):
        return all(_same_executed_record(getattr(left, field.name), getattr(right, field.name))
                   for field in fields(left) if field.init)
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _same_executed_record(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _same_executed_record(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def _unpack(value, arrays, types, depth=0, _records=None, _definitions=None):
    if depth > 60:
        raise ValueError("Section metadata is too deeply nested")
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError("Invalid section metadata")
    if _records is None:
        _records = {}
        _definitions = {}
        pending = [(value, 0)]
        while pending:
            item, level = pending.pop()
            if level > 60:
                raise ValueError("Section metadata is too deeply nested")
            if isinstance(item, dict):
                if "record_id" in item:
                    key = item["record_id"]
                    if (not isinstance(key, str) or not key.startswith("record_")
                            or not key[7:].isdigit() or key in _definitions):
                        raise ValueError("Invalid or duplicate section record identity")
                    _definitions[key] = item
                pending.extend((child, level + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, level + 1) for child in item)
    def decode(v):
        return _unpack(v, arrays, types, depth + 1, _records, _definitions)
    if set(value) == {"enum", "value"}:
        cls = _enum_types().get(value["enum"])
        if cls is None:
            raise ValueError("Unknown section enumeration")
        return cls(value["value"])
    if set(value) == {"ref"}:
        key = value["ref"]
        if not isinstance(key, str) or key not in _definitions:
            raise ValueError("Invalid section record reference")
        return decode(_definitions[key])
    if set(value) == {"array"}:
        return arrays[value["array"]]
    if set(value) == {"float"} and value["float"] in {"nan", "inf", "-inf"}:
        return float.fromhex(value["float"])
    if set(value) == {"mapping"}:
        return {k: decode(v) for k, v in value["mapping"].items()}
    if set(value) == {"tuple"}:
        return tuple(decode(v) for v in value["tuple"])
    if set(value) == {"list"}:
        return [decode(v) for v in value["list"]]
    if set(value) == {"frozenset"}:
        return frozenset(decode(v) for v in value["frozenset"])
    if set(value) == {"record_id", "type", "fields"} and value["type"] in types:
        key = value["record_id"]
        if key in _records:
            if _records[key] is None:
                raise ValueError("Cyclic section record reference")
            return _records[key]
        _records[key] = None
        cls = types[value["type"]]
        allowed = {f.name for f in fields(cls) if f.init}
        if not isinstance(value["fields"], dict) or set(value["fields"]) != allowed:
            raise ValueError("Unknown or missing current section data fields")
        record_fields = {k: decode(v) for k, v in value["fields"].items()}
        result = cls(**record_fields)
        if cls is types.get("EDSSpectrum") and result.response_rates is not None:
            from temsim.detector.eds_response import _validate_alignment
            _validate_alignment(result, result.response_rates)
        if cls is types.get("EDSPhotonRay"):
            # Construction validates a ray and normalises new directions. An
            # archived direction is already normalised; a second division can
            # change its final bit. Validate, then retain the executed values.
            direction = tuple(record_fields["direction"])
            if not math.isclose(float(np.linalg.norm(direction)), 1.0,
                                rel_tol=0.0, abs_tol=8.0*np.finfo(float).eps):
                raise ValueError("Archived EDS photon direction is not a unit vector")
            object.__setattr__(result, "direction", direction)
        if cls is types.get("MaterialSectionCache"):
            spectrum = getattr(result, "eds_spectrum", None)
            if spectrum is not None:
                if (not isinstance(spectrum, types["EDSSpectrum"])
                        or not _same_executed_record(
                            spectrum.elastic_transport, result.elastic_transport)):
                    raise ValueError("Archived EDS and specimen transport must share one executed elastic result")
                if spectrum.elastic_transport is not result.elastic_transport:
                    raise ValueError("Archived EDS and specimen transport must share one executed elastic result")
        _records[key] = result
        return result
    raise ValueError("Unknown section data type or record")


def _pack_record_graph(value, arrays):
    tree = _pack(value, arrays, {}, _data_classes())
    return np.frombuffer(json.dumps(tree, allow_nan=False,
        separators=(",", ":")).encode("utf-8"), dtype=np.uint8)


def _unpack_record_graph(graph, arrays, *, maximum_unpacked_bytes):
    from temsim.working_point_archive import _unique_pairs
    if graph.dtype != np.dtype(np.uint8) or graph.ndim != 1 or graph.nbytes > maximum_unpacked_bytes:
        raise ValueError("Invalid or oversized section record graph")
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value in section record graph")
    try:
        tree = json.loads(graph.tobytes(), object_pairs_hook=_unique_pairs, parse_constant=nonfinite)
        return _unpack(tree, arrays, _data_classes())
    except (UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("Invalid or too deeply nested section record graph") from exc


def _result_records(result):
    from temsim.simulation_pipeline import CalculationResult
    if type(result) is not CalculationResult or {f.name for f in fields(result) if f.init} != _RESULT_FIELDS:
        raise ValueError("The current calculation-result fields require an explicit archive contract")
    if result.wave_imaging is not None:
        raise ValueError("Coherent wave results require a wave archive, not a classical particle section")
    return {name: getattr(result, name) for name in sorted(_RESULT_FIELDS - _SNAPSHOT_RESULT_FIELDS)}


def save_section_result(result, path, *, overwrite=False, maximum_unpacked_bytes=8*1024**3):
    """Save a completed section and its executed upstream state atomically."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.physics.particle_sections import validate_section_checkpoint
    from zipfile import ZIP_STORED
    simulation = result.simulation
    checkpoint = getattr(simulation, "section_checkpoint", None)
    if checkpoint is None:
        raise ValueError("Calculate a section before saving its upstream state")
    validate_section_checkpoint(checkpoint)
    target = float(simulation.metrics["section_target_z_mm"])
    request = normalise_section_request({"target_z_mm": target,
        "component_keys": tuple(simulation.metrics.get("section_component_keys", ()))})
    summary = section_archive_summary(result, path=path,
        saved_at_utc=datetime.now(timezone.utc).isoformat())
    from temsim.calculation_manifest import assert_external_input_inventory_unchanged
    external_inputs = getattr(result, "external_inputs", None)
    if external_inputs is not None:
        assert_external_input_inventory_unchanged(result.state_snapshot, external_inputs)
    arrays = {}
    # Record every executed result field. Snapshot-owned input geometry is
    # restored independently; no readout or interaction is recomputed at load.
    graph = _pack_record_graph(_result_records(result), arrays)
    # The complete typed graph is a checksummed numeric product. The manifest
    # remains a small index even for millions of material/EDS event records.
    arrays[_GRAPH_ARRAY] = graph
    snapshot = capture_instrument_snapshot(result.state_snapshot)
    if external_inputs is not None:
        assert_external_input_inventory_unchanged(result.state_snapshot, external_inputs)
    package = WorkingPointCheckpoint(snapshot, arrays, target,
        json_digest({"schema": SECTION_PACKAGE_SCHEMA, "snapshot": snapshot.digest, "request": request}),
        {"package_kind": SECTION_PACKAGE_SCHEMA, "record_graph": _GRAPH_ARRAY, "request": request,
         "quality": section_result_quality(result), "archive_summary": summary,
         "model_signature": result.model_signature, "signatures": dict(result.signatures),
         "particle_tuning": bool(simulation.metrics.get("particle_tuning", False)),
         "particle_section": bool(simulation.metrics.get("particle_section", False)),
         "physics_scope": ("Classical particle transport with inserted-specimen scattering"
             if simulation.metrics.get("particle_section", simulation.metrics.get("particle_tuning", False))
                and not simulation.metrics.get("optical_tuning", False)
             else "Historical optical preview without specimen scattering"),
         "source_representation": "executed-tip-origin-particle-section"})
    package.write_package(path, overwrite=overwrite, compression=ZIP_STORED,
                          maximum_unpacked_bytes=maximum_unpacked_bytes)
    return package


def load_section_result(path, *, maximum_unpacked_bytes=8*1024**3):
    """Validate a checked package and report malformed input as a user error."""
    from zipfile import BadZipFile
    try:
        return _load_section_result(path, maximum_unpacked_bytes=maximum_unpacked_bytes)
    except (BadZipFile, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete or invalid particle-section checkpoint") from exc


def _load_section_result(path, *, maximum_unpacked_bytes):
    """Restore a seed without changing any live instrument parameter."""
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.simulation_pipeline import CalculationResult
    from temsim.physics.simulation import Simulation, _validate_gun_flight_times
    from temsim.physics.particle_sections import ParticleSectionCheckpoint, validate_section_checkpoint
    from temsim.physics.optical_tuning import prepare_tuning_snapshot
    from temsim.working_point_archive import WorkingPointArchiveIndex
    before = section_file_fingerprint(path)
    index = WorkingPointArchiveIndex.read(path, maximum_unpacked_bytes=maximum_unpacked_bytes)
    if index.metadata.get("package_kind") != SECTION_PACKAGE_SCHEMA:
        raise ValueError("Unsupported particle-section schema; calculate and save a current section")
    package = index.load()
    metadata = thaw_json(package.metadata)
    if metadata.get("package_kind") != SECTION_PACKAGE_SCHEMA:
        raise ValueError("This file is not an executed particle-section checkpoint")
    request = normalise_section_request(metadata["request"], quality=metadata["quality"])
    # Restore only the detached saved input graph; its solver and input-content
    # checks prevent an old implementation from becoming an active source.
    state = package.snapshot.restore()
    physical = bool(metadata["particle_section"])
    if metadata["quality"] in {"Preview", "Medium"}:
        prepare_tuning_snapshot(state, metadata["quality"], particle_signals=physical)
    else:
        state._tuning_quality = "High accuracy"
    try:
        payload = _unpack_record_graph(package.arrays[metadata["record_graph"]], package.arrays,
                                      maximum_unpacked_bytes=maximum_unpacked_bytes)
        if set(payload) != _RESULT_FIELDS - _SNAPSHOT_RESULT_FIELDS or payload["wave_imaging"] is not None:
            raise ValueError("Incomplete or unsupported current particle result fields")
        simulation = payload["simulation"]
        checkpoint = simulation.section_checkpoint
        if not isinstance(simulation, Simulation) or not isinstance(checkpoint, ParticleSectionCheckpoint):
            raise ValueError("The file has no executed section state")
        if (float(simulation.metrics["section_target_z_mm"]) != request["target_z_mm"]
                or tuple(sorted(simulation.metrics["section_component_keys"])) != request["component_keys"]
                or not _validate_gun_flight_times(checkpoint.gun_trace)):
            raise ValueError("Section checkpoint metadata does not match its executed state")
        simulation.section_checkpoint = checkpoint
        validate_section_checkpoint(checkpoint)
        material = simulation.material_section_cache
        if material is not None:
            from temsim.physics.particle_sections import MaterialSectionCache, MATERIAL_SECTION_SCHEMA
            if not isinstance(material, MaterialSectionCache) or material.schema != MATERIAL_SECTION_SCHEMA:
                raise ValueError("Unsupported executed specimen checkpoint")
            simulation.material_section_cache = material
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete particle-section checkpoint") from exc
    from temsim.column.state_layout import apply_physical_layout_to_state
    layout = apply_physical_layout_to_state(state)
    if payload["model_signature"] != metadata["model_signature"] or payload["signatures"] != metadata["signatures"]:
        raise ValueError("Stored result and archive dependency signatures differ")
    result = CalculationResult(**payload, state_snapshot=state, layout=layout, assembly=state._resolved_assembly)
    summary = section_archive_summary(result, path=path,
        saved_at_utc=metadata["archive_summary"]["saved_at_utc"])
    if metadata["archive_summary"]["identity"] != summary["identity"]:
        raise ValueError("Section archive summary does not identify its executed result")
    if section_file_fingerprint(path) != before:
        raise ValueError("Section archive changed during loading; retry the operation")
    result.section_archive_info = dict(summary, _file_fingerprint=before,
                                      _package_digest=index.digest)
    result.loaded_section_only = True
    return result
