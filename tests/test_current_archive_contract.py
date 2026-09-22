"""Current-only schemas and compact, exact executed-record storage."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from test_particle_section_eds_archive import material_cache, snapshot
from temsim.particle_section_io import (
    _data_classes, _pack, _unpack, _pack_record_graph, _unpack_record_graph,
    _same_executed_record, _RESULT_FIELDS, _SNAPSHOT_RESULT_FIELDS, _result_records,
    SECTION_PACKAGE_SCHEMA, load_section_result,
)
from temsim.working_point import WorkingPointCheckpoint
from temsim.working_point_archive import WorkingPointArchiveIndex


def _difference_path(left, right, path="root"):
    if _same_executed_record(left, right):
        return None
    if type(left) is not type(right):
        return path, type(left), type(right)
    if is_dataclass(left):
        for field in fields(left):
            if field.init:
                mismatch = _difference_path(getattr(left, field.name), getattr(right, field.name), path+"."+field.name)
                if mismatch:
                    return mismatch
    elif isinstance(left, Mapping):
        for key in left:
            mismatch = _difference_path(left[key], right[key], path+"."+key)
            if mismatch:
                return mismatch
    elif isinstance(left, (tuple, list)):
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            mismatch = _difference_path(a, b, path+f"[{index}]")
            if mismatch:
                return mismatch
    return path, repr(left)[:200], repr(right)[:200]


def test_compact_graph_roundtrip_keeps_shared_executed_records(material_cache, snapshot, tmp_path):
    arrays = {}
    graph = _pack_record_graph({"material": material_cache, "again": material_cache,
                               "spectrum": material_cache.eds_spectrum}, arrays)
    arrays["records"] = graph
    original = WorkingPointCheckpoint(snapshot, arrays, 1234., "fixture", {"record_graph": "records"})
    path = tmp_path / "compact.temwp"
    original.write_package(path)
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["metadata"] == {"record_graph": "records"}
    assert "EDSLineSignal" not in json.dumps(manifest)
    loaded = WorkingPointCheckpoint.read_package(path)
    result = _unpack_record_graph(loaded.arrays["records"], loaded.arrays, maximum_unpacked_bytes=8*1024**3)
    assert result["again"] is result["material"]
    assert result["spectrum"] is result["material"].eds_spectrum
    assert result["spectrum"].elastic_transport is result["material"].elastic_transport
    assert _same_executed_record(result["material"], material_cache)
    assert graph.dtype == np.uint8 and not graph.flags.writeable
    single = _pack_record_graph(material_cache, {})
    many = _pack_record_graph(tuple(material_cache for _ in range(100)), {})
    assert many.nbytes < single.nbytes + 3000


@pytest.mark.parametrize("mutation", ["missing", "cycle", "duplicate"])
def test_record_graph_rejects_bad_references(material_cache, mutation):
    arrays = {}
    tree = _pack(material_cache, arrays, {}, _data_classes())
    if mutation == "missing":
        tree["fields"]["elastic_transport"] = {"ref": "record_9999999"}
    elif mutation == "cycle":
        tree["fields"]["elastic_transport"] = {"ref": tree["record_id"]}
    else:
        tree = {"tuple": [tree, deepcopy(tree)]}
    with pytest.raises(ValueError, match="reference|identity"):
        _unpack(tree, arrays, _data_classes())


def test_record_graph_checks_dtype_shape_and_byte_budget(material_cache):
    arrays = {}
    graph = _pack_record_graph(material_cache, arrays)
    for bad, maximum in ((graph.astype(float), graph.nbytes*9),
                         (graph.reshape(1, -1), graph.nbytes), (graph, graph.nbytes-1)):
        with pytest.raises(ValueError, match="record graph"):
            _unpack_record_graph(bad, arrays, maximum_unpacked_bytes=maximum)


def test_writer_manifest_limit_is_checked_before_file_creation(snapshot, tmp_path, monkeypatch):
    from temsim import working_point_archive as module
    point = WorkingPointCheckpoint(snapshot, {}, 1., "fixture", {"large": "x"*1024})
    path = tmp_path / "preserved.temwp"
    path.write_bytes(b"old file unchanged")
    monkeypatch.setattr(module, "MAXIMUM_MANIFEST_BYTES", 512)
    with pytest.raises(ValueError, match="manifest"):
        point.write_package(path, overwrite=True)
    assert path.read_bytes() == b"old file unchanged"
    assert not tuple(tmp_path.glob(".working-point-*.tmp"))


def test_writer_and_reader_use_same_exact_unpacked_budget(snapshot, tmp_path):
    point = WorkingPointCheckpoint(snapshot, {"fixture": np.arange(32.)}, 1., "fixture", {})
    path = tmp_path / "exact.temwp"
    point.write_package(path)
    index = WorkingPointArchiveIndex.read(path)
    with ZipFile(path) as archive:
        assert index.unpacked_size_bytes == sum(row.file_size for row in archive.infolist())
    point.write_package(path, overwrite=True, maximum_unpacked_bytes=index.unpacked_size_bytes)
    assert WorkingPointCheckpoint.read_package(path, maximum_unpacked_bytes=index.unpacked_size_bytes).digest == point.digest
    with pytest.raises(ValueError, match="budget"):
        point.write_package(path, overwrite=True, maximum_unpacked_bytes=index.unpacked_size_bytes-1)
    with pytest.raises(ValueError, match="budget"):
        WorkingPointCheckpoint.read_package(path, maximum_unpacked_bytes=index.unpacked_size_bytes-1)


def test_index_checks_npy_header_without_allocating_numeric_arrays(snapshot, tmp_path, monkeypatch):
    point = WorkingPointCheckpoint(snapshot, {"fixture": np.zeros(3)}, 1., "fixture", {})
    path = tmp_path / "original.temwp"
    point.write_package(path)
    bad = tmp_path / "bad.temwp"
    with ZipFile(path) as source, ZipFile(bad, "w") as output:
        for row in source.infolist():
            data = source.read(row.filename)
            if row.filename.endswith(".npy"):
                buffer = BytesIO()
                np.lib.format.write_array_header_1_0(buffer, dict(descr="<f8", fortran_order=False, shape=(999,)))
                data = buffer.getvalue() + bytes(24)
            output.writestr(row.filename, data)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Index must not allocate arrays"))
    with pytest.raises(ValueError, match="header|payload"):
        WorkingPointArchiveIndex.read(bad)


def test_unsupported_section_schema_is_rejected_before_numeric_loading(snapshot, tmp_path, monkeypatch):
    point = WorkingPointCheckpoint(snapshot, {"fixture": np.zeros(3)}, 1., "fixture",
                                  {"package_kind": "optical-particle-section-package-v1"})
    path = tmp_path / "old.temsection"
    point.write_package(path)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Old schema must not load numeric arrays"))
    with pytest.raises(ValueError, match="Unsupported particle-section schema"):
        load_section_result(path)
    assert SECTION_PACKAGE_SCHEMA.endswith("v2")


def test_result_contract_covers_every_current_field_and_rejects_wave():
    from temsim.simulation_pipeline import CalculationResult
    assert {field.name for field in fields(CalculationResult) if field.init} == _RESULT_FIELDS
    result = CalculationResult(None, None)
    assert set(_result_records(result)) == _RESULT_FIELDS - _SNAPSHOT_RESULT_FIELDS
    with pytest.raises(ValueError, match="Coherent wave"):
        _result_records(replace(result, wave_imaging=object()))


def test_complete_classical_result_records_roundtrip_without_recomputing(material_cache, tmp_path):
    """Synthetic executed-result contracts; no transport or detector solver."""
    from temsim.optics.column import default_state
    from temsim.calculation_manifest import capture_calculation_manifest
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.simulation_pipeline import CalculationResult
    from temsim.specimen.interaction_types import (
        SpecimenInteractionRequest, SpecimenInteractionResult, SpecimenObservable,
        IncidentElectronRay, IncidentRayBundle, ConservationCheck,
    )
    from temsim.specimen.scene import SpecimenScene
    from temsim.specimen.sample_region import SampleRegionResult, SampleRegionElectronPath, SampleRegionPhotonPath
    from temsim.detector.particle_readout import ParticleDetectorReadout
    from temsim.detector.stem_signal import StemScanResult, DetectorSignal, CollectionAngle, BeamCurrentSignal
    from temsim.physics.scan_geometry import ScanGeometryResult, ScanRayPathResult
    from temsim.optics.energy_filter_raytrace import EnergyFilterResult, FilterPlaneArrival

    state = default_state()
    observables = frozenset({SpecimenObservable.ELASTIC_TRANSPORT, SpecimenObservable.CHARACTERISTIC_X_RAY})
    incident = IncidentRayBundle((IncidentElectronRay(2, (0., 0.), (0., 0., 1.), 200000., 1.),),
        3, 1, .75, (0., 0.), (0., 0.), (0., 0.), (200000., 200000.))
    interaction = SpecimenInteractionResult(SpecimenInteractionRequest(observables), observables,
        scene=SpecimenScene.from_state(state), incident_bundle=incident,
        elastic_transport=material_cache.elastic_transport, eds_spectrum=material_cache.eds_spectrum,
        conservation=(ConservationCheck("fixture", "population", "1", 1., (("transmitted", 1.),)),))
    position = np.array([[0., 0., 1.], [0., 0., 2.]])
    sample_region = SampleRegionResult(1., 2.,
        (SampleRegionElectronPath(position, "fixture", 1., 200000., "synthetic", True, 2, 42, .5),),
        (SampleRegionPhotonPath(position, (0., 0., 1.), 1740., 1., .5, "sample", "Ka1", True, 0, "synthetic"),),
        (), material_cache.eds_spectrum, interaction, {"fixture": True})
    vector = np.array([0., 1.])
    scan = ScanGeometryResult(vector, vector, vector, {"sample": (vector, vector)},
        {"sample": "Specimen"}, 2, 2, True, False, 0., None)
    scan_paths = ScanRayPathResult({"sample": (vector, vector)}, np.zeros(2), np.zeros(2), 1., 2, 2)
    collection = CollectionAngle(0., 10., (0., 0.), (10., 10.), False)
    detector = DetectorSignal("bf", "BF detector", .5, 1., .5, 2., collection)
    stem = StemScanResult(vector, vector, {"bf": np.full((2, 2), .5)}, {"bf": detector}, {"fixture": True})
    timed = FilterPlaneArrival("camera", vector, vector, vector, vector, vector, np.ones(2, dtype=bool))
    filtered = EnergyFilterResult([vector], [vector], [vector], ["green"], vector, 2, 1, True,
        BeamCurrentSignal("entrance", "Filter entrance", 1., 2., 1., 2.), "fixture", ("camera",),
        1., 1., .5, .5, .5, .5, source_ray_id=np.array([42, 43]), source_fraction=np.array([.5, .5]),
        source_branch=("primary", "primary"), source_path_index=np.array([0, 1]),
        entrance_time_s=vector, timed_planes=(timed,))
    manifest = capture_calculation_manifest(state)
    original = CalculationResult(None, filtered, state_snapshot=state, specimen_interactions=interaction,
        scan_geometry=scan, scan_ray_paths=scan_paths, stem_scan=stem, sample_region=sample_region,
        lens_crossovers=({"z_mm": 1.25},), aperture_stops=({"key": "fixture", "z_mm": 1.5},),
        model_signature="fixture", signatures={"request": "fixture"}, calculated_products=frozenset({"eds"}),
        performance={"scope": "synthetic", "wall_time_s": 1.25}, external_inputs=manifest.external_inputs,
        calculation_manifest=manifest, working_point_parent_id="parent",
        particle_signals=(ParticleDetectorReadout("bf", "BF detector", "AVAILABLE", .5, 1., .5, 2., 3.),))
    arrays = {}
    arrays["records"] = _pack_record_graph(_result_records(original), arrays)
    package = WorkingPointCheckpoint(capture_instrument_snapshot(state), arrays, 2., "fixture", {})
    path = tmp_path / "all-fields.temwp"
    package.write_package(path)
    loaded = WorkingPointCheckpoint.read_package(path)
    payload = _unpack_record_graph(loaded.arrays["records"], loaded.arrays, maximum_unpacked_bytes=8*1024**3)
    original_fields = _result_records(original)
    assert _difference_path(payload, original_fields) is None
    assert payload["sample_region"].interactions is payload["specimen_interactions"]
    assert payload["sample_region"].spectrum is payload["specimen_interactions"].eds_spectrum
    assert payload["particle_signals"][0].expected_electrons == 3.


@pytest.mark.parametrize("version", [None, True, 77, 79, "78", 78.0])
def test_state_rejects_noncurrent_or_ambiguous_versions(version):
    from temsim.optics.model import State
    with pytest.raises(ValueError, match="schema"):
        State.from_dict({"schema_version": version})


def test_current_state_roundtrip_preserves_tip_and_rejects_obsolete_inputs():
    from temsim.optics.column import default_state
    from temsim.optics.model import State, STATE_SCHEMA_VERSION
    state = default_state()
    payload = state.to_dict()
    assert payload["schema_version"] == STATE_SCHEMA_VERSION == 78
    restored = State.from_dict(payload)
    assert restored.electron_gun.to_dict() == state.electron_gun.to_dict()
    for mutation in (lambda d: d["sample"].update(atomic_structure_source="preset"),
                     lambda d: d["lenses"][0].update(key="c1"),
                     lambda d: d["stigmators"][0].pop("field_model")):
        malformed = deepcopy(payload)
        mutation(malformed)
        with pytest.raises(ValueError, match="Legacy|legacy|Retired|stigmator"):
            State.from_dict(malformed)


@pytest.mark.parametrize("module", ["diffraction_lens", "intermediate_lens", "projector_lens_p1", "projector_lens_p2"])
def test_current_lens_loader_uses_manifest_geometry_without_legacy_position_parameters(module):
    from importlib import import_module
    implementation = import_module("temsim.optics." + module)
    create = getattr(implementation, "create_" + module)
    restore = getattr(implementation, module + "_from_dict")
    original = create()
    percent = .5 * float(original.max_percent)
    restored = restore({"key": original.key, "percent": percent})
    assert restored.percent == percent
    assert restored.z_mm == original.z_mm
    assert restored.optical_reference_downstream_of_anchor_mm == original.optical_reference_downstream_of_anchor_mm
    assert restored.mechanical_center_downstream_of_anchor_mm == original.mechanical_center_downstream_of_anchor_mm
    with pytest.raises(TypeError, match="unexpected keyword"):
        restore({"key": original.key}, legacy_reference_z_mm=1.)
    with pytest.raises(ValueError, match="canonical key"):
        restore({"key": "diff"})
