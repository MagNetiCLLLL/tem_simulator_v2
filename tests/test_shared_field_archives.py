"""Exact magnetic-plan storage fixtures; no gun or material physics is executed."""
from dataclasses import fields, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.artifact_store import ArtifactIntegrityError, ArtifactStore, INCIDENT_SEED_CODEC
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.optics.electron_gun.base import GunExitBundle, GunTraceResult
from temsim.particle_section_io import (
    _data_classes, _pack, _unpack, _unpack_record_graph,
    load_section_result, save_section_result,
)
from temsim.physics.core import AxialPropagationPlan, PropagationCheckpoints
from temsim.physics.particle_sections import ParticleSectionCheckpoint, ParticleSectionSegment, SECTION_SCHEMA
from temsim.physics.simulation import Branch, Simulation
from temsim.simulation_pipeline import CalculationResult
from temsim.working_point import WorkingPointCheckpoint
from temsim.working_point_archive import WorkingPointArchiveIndex


@pytest.fixture(scope="module")
def storage_result():
    """Small synthetic arrays satisfy the executed-record storage contracts only."""
    state = default_state()
    manifest = capture_calculation_manifest(state)
    z = np.array((1., 2., 3.))
    arguments = {}
    for field in fields(AxialPropagationPlan):
        if field.name.startswith("midpoint_"):
            arguments[field.name] = np.zeros(2)
        elif field.name.endswith(("_t", "_m2", "_m3", "_m1", "_rad")):
            arguments[field.name] = np.zeros(3)
    arguments.update(z_mm=z, step_m=np.array((.001, .001)),
        save_index=np.array((0, 1, 2), dtype=np.int64),
        checkpoint_index=np.array((0, 2), dtype=np.int64),
        solver_signature="synthetic shared-field storage fixture", signature="synthetic plan",
        dipole_bx_t=np.array((1., 1., 1., -2., -2., -2.)) * 1e-6,
        dipole_by_t=np.array((3., 3., 3., -4., -4., -4.)) * 1e-6,
        reference_momentum_kg_m_s=3.366415614509291e-22, mapped_fields=())
    plan = AxialPropagationPlan(**arguments)
    xy = np.array(((1., -1.), (2., -2.), (3., -3.))) * 1e-9
    slopes = xy * 1e3
    clock = np.array(((1., 1.1), (2., 2.1), (3., 3.1))) * 1e-9
    kinetic = np.array(((300000., 300001.), (300012., 300013.), (300025., 300026.)))
    alive, blocked = np.ones(2, dtype=bool), np.full(2, np.nan)
    ids, weights, energies = np.array((4, 7), dtype=np.int64), np.array((.4, .6)), np.zeros(2)
    exit_bundle = GunExitBundle(xy[0], xy[0], slopes[0], slopes[0], energies,
                                weights, ids, alive, flight_time_s=clock[0])
    gun = GunTraceResult(np.array((0., 1.)), xy[:2], xy[:2], slopes[:2], slopes[:2],
        exit_bundle, blocked, ("", ""), 1e-9, 1e-9, 1e-9,
        flight_time_s=clock[:2], source_record={"scope": "storage-only fixture"})
    incident = Branch("Fixture", (0., 1., 0.), z, xy, xy, slopes, slopes,
        alive, blocked, ["", ""], 1., energies, ray_weight=weights,
        source_ray_id=ids, source_azimuth_rad=np.zeros(2), flight_time_s=clock,
        kinetic_energy_ev=kinetic)
    checkpoints = PropagationCheckpoints(z[[0, 2]], xy[[0, 2]], slopes[[0, 2]],
        xy[[0, 2]], slopes[[0, 2]], clock[[0, 2]], kinetic_energy_ev=kinetic[[0, 2]])
    checkpoint = ParticleSectionCheckpoint(SECTION_SCHEMA, "synthetic gun identity", gun,
        (ParticleSectionSegment("incident", incident, plan, checkpoints, "synthetic executed initial state"),))
    simulation = Simulation(incident, {}, {
        "section_target_z_mm": 3., "section_component_keys": (),
        "section_resumable_through_z_mm": 3., "tuning_quality": "High accuracy",
        "particle_section": True, "scope": "storage-only fixture"},
        gun_trace=gun, incident_plan=plan, incident_checkpoints=checkpoints,
        section_checkpoint=checkpoint)
    return CalculationResult(simulation, None, state_snapshot=state,
        model_signature="synthetic fixture", signatures=dict(manifest.calculation_signatures),
        calculation_manifest=manifest)


def assert_magnetic_plan_equal(left, right):
    for name in ("dipole_bx_t", "dipole_by_t"):
        original, restored = getattr(left, name), getattr(right, name)
        assert np.any(original != 0.)
        assert restored.dtype == np.dtype(np.float64)
        assert restored.shape == (3 * (len(right.z_mm) - 1),)
        assert restored.tobytes() == original.tobytes()
        assert not restored.flags.writeable
    assert right.reference_momentum_kg_m_s == left.reference_momentum_kg_m_s


def seed_store(tmp_path, storage_result):
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    manifest = storage_result.calculation_manifest
    store.put_incident_simulation_seed(manifest, storage_result.simulation)
    bundle = store.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=INCIDENT_SEED_CODEC)
    return store, manifest, bundle


def test_incident_seed_roundtrip_keeps_all_three_stage_fields_and_reference_momentum(storage_result, tmp_path):
    store, manifest, bundle = seed_store(tmp_path, storage_result)
    assert "electromagnetic-energy" in INCIDENT_SEED_CODEC
    assert bundle.metadata["plan_reference_momentum_kg_m_s"] == storage_result.simulation.incident_plan.reference_momentum_kg_m_s
    restored = store.get_incident_simulation_seed(manifest)
    assert_magnetic_plan_equal(storage_result.simulation.incident_plan, restored.incident_plan)
    assert restored.incident_checkpoints.kinetic_energy_ev.tobytes() == storage_result.simulation.incident_checkpoints.kinetic_energy_ev.tobytes()
    assert restored.incident.kinetic_energy_ev.tobytes() == storage_result.simulation.incident.kinetic_energy_ev.tobytes()


@pytest.mark.parametrize("missing", ["plan.dipole_bx_t", "plan.dipole_by_t", "plan_reference_momentum_kg_m_s"])
def test_incomplete_current_incident_seed_cannot_invent_zero_magnetic_fields(storage_result, tmp_path, missing):
    store, manifest, bundle = seed_store(tmp_path, storage_result)
    arrays, metadata = dict(bundle.arrays), thaw_json(bundle.metadata)
    del (arrays if missing.startswith("plan.") else metadata)[missing]
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        codec=INCIDENT_SEED_CODEC, arrays=arrays, metadata=metadata)
    with pytest.raises(ArtifactIntegrityError, match="historical read-only.*cannot resume"):
        store.get_incident_simulation_seed(manifest)


@pytest.mark.parametrize("damage", ["shape", "float32", "nonfinite", "zero_momentum", "string_momentum"])
def test_current_incident_magnetic_payload_is_semantically_validated(storage_result, tmp_path, damage):
    store, manifest, bundle = seed_store(tmp_path, storage_result)
    arrays, metadata = dict(bundle.arrays), thaw_json(bundle.metadata)
    if damage == "zero_momentum":
        metadata["plan_reference_momentum_kg_m_s"] = 0.
    elif damage == "string_momentum":
        metadata["plan_reference_momentum_kg_m_s"] = "3e-22"
    else:
        value = arrays["plan.dipole_bx_t"].copy()
        if damage == "shape":
            value = value[:-1]
        elif damage == "float32":
            value = value.astype(np.float32)
        else:
            value[0] = np.nan
        arrays["plan.dipole_bx_t"] = value
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        codec=INCIDENT_SEED_CODEC, arrays=arrays, metadata=metadata)
    with pytest.raises(ArtifactIntegrityError, match="Incident seed metadata"):
        store.get_incident_simulation_seed(manifest)


@pytest.mark.parametrize("old_codec", ["incident-simulation-seed-v2-quadrupole-tensor",
                                        "incident-simulation-seed-v3-flight-time",
                                        "incident-simulation-seed-v4-shared-magnetic-field"])
def test_historical_thin_kick_bundle_remains_readable_but_is_not_an_active_seed(storage_result, tmp_path, old_codec):
    _, manifest, current = seed_store(tmp_path / "current", storage_result)
    old_store = ArtifactStore(tmp_path / "historical", quota_bytes=100_000_000)
    arrays, metadata = dict(current.arrays), thaw_json(current.metadata)
    for name in ("plan.dipole_bx_t", "plan.dipole_by_t"):
        arrays.pop(name)
    metadata.pop("plan_reference_momentum_kg_m_s")
    old_store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        codec=old_codec, arrays=arrays, metadata=metadata)
    assert old_store.get_incident_simulation_seed(manifest) is None
    historical = old_store.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec=old_codec)
    np.testing.assert_array_equal(historical.arrays["incident.x"], arrays["incident.x"])
    assert not historical.arrays["incident.x"].flags.writeable
    assert "plan.dipole_bx_t" not in historical.arrays
    assert "working_point" in historical.metadata


def test_current_section_roundtrip_retains_magnetic_plan_without_numeric_execution(storage_result, tmp_path):
    path = tmp_path / "current.temsection"
    save_section_result(storage_result, path)
    restored = load_section_result(path)
    assert_magnetic_plan_equal(storage_result.simulation.incident_plan, restored.simulation.incident_plan)
    section_plan = restored.simulation.section_checkpoint.segments[0].plan
    assert section_plan is restored.simulation.incident_plan
    np.testing.assert_array_equal(restored.simulation.incident.kinetic_energy_ev,
                                  storage_result.simulation.incident.kinetic_energy_ev)


def test_zero_length_gun_exit_plan_keeps_empty_interval_fields_in_both_codecs(storage_result, tmp_path):
    """Saving exactly at an executed gun exit is a valid one-node section."""
    simulation = storage_result.simulation
    original_plan = simulation.incident_plan
    values = {}
    for field in fields(original_plan):
        value = getattr(original_plan, field.name)
        if isinstance(value, np.ndarray):
            interval_field = (field.name == "step_m" or field.name.startswith("midpoint_")
                              or field.name in {"dipole_bx_t", "dipole_by_t"})
            values[field.name] = value[:0 if interval_field else 1]
    plan = replace(original_plan, **values)
    branch = replace(simulation.incident, **{
        name: getattr(simulation.incident, name)[:1]
        for name in ("z", "x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev")})
    original_checkpoint = simulation.incident_checkpoints
    checkpoint = replace(original_checkpoint, **{
        field.name: getattr(original_checkpoint, field.name)[:1]
        for field in fields(original_checkpoint)})
    section = replace(simulation.section_checkpoint, segments=(replace(
        simulation.section_checkpoint.segments[0], branch=branch, plan=plan,
        checkpoints=checkpoint),))
    result = replace(storage_result, simulation=replace(simulation,
        incident=branch, incident_plan=plan, incident_checkpoints=checkpoint,
        section_checkpoint=section, metrics=dict(simulation.metrics,
            section_target_z_mm=1., section_resumable_through_z_mm=1.)))

    store, manifest, _ = seed_store(tmp_path / "seed", result)
    from_seed = store.get_incident_simulation_seed(manifest)
    path = tmp_path / "gun-exit.temsection"
    save_section_result(result, path)
    from_section = load_section_result(path).simulation
    for restored in (from_seed, from_section):
        np.testing.assert_array_equal(restored.incident_plan.z_mm, [1.])
        np.testing.assert_array_equal(restored.incident.x, branch.x)
        for name in ("dipole_bx_t", "dipole_by_t"):
            value = getattr(restored.incident_plan, name)
            assert value.shape == (0,) and value.dtype == np.dtype(np.float64)
            assert not value.flags.writeable
        assert restored.incident_plan.reference_momentum_kg_m_s == plan.reference_momentum_kg_m_s
        np.testing.assert_array_equal(restored.incident_checkpoints.kinetic_energy_ev,
                                      checkpoint.kinetic_energy_ev)


@pytest.mark.parametrize("missing", ["dipole_bx_t", "dipole_by_t", "reference_momentum_kg_m_s"])
def test_graph_decoder_requires_shared_magnetic_fields_and_leaves_historical_bytes_untouched(storage_result, missing):
    arrays = {}
    tree = _pack(storage_result.simulation.incident_plan, arrays, {}, _data_classes())
    del tree["fields"][missing]
    before = json.dumps(tree, sort_keys=True)
    with pytest.raises(ValueError, match="historical read-only.*continuation requires recalculation"):
        _unpack(tree, arrays, _data_classes())
    assert json.dumps(tree, sort_keys=True) == before


def test_old_section_package_opens_in_historical_reader_but_cannot_restore_or_decode_a_plan(storage_result, tmp_path):
    original = save_section_result(storage_result, tmp_path / "current.temsection")
    graph_name = original.metadata["record_graph"]
    tree = json.loads(original.arrays[graph_name].tobytes())
    pending = [tree]
    removed = 0
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get("type") == "AxialPropagationPlan":
                for name in ("dipole_bx_t", "dipole_by_t", "reference_momentum_kg_m_s"):
                    item["fields"].pop(name)
                removed += 1
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    assert removed == 1
    arrays = dict(original.arrays)
    arrays[graph_name] = np.frombuffer(json.dumps(tree).encode(), dtype=np.uint8)
    historical = WorkingPointCheckpoint(replace(original.snapshot, implementation="0" * 64),
        arrays, original.plane_z_mm, original.stage_signature, original.metadata)
    path = tmp_path / "historical.temsection"
    historical.write_package(path)
    index = WorkingPointArchiveIndex.read(path)
    loaded = index.load()  # This is the existing read-only Working points path.
    assert loaded.digest == historical.digest
    np.testing.assert_array_equal(loaded.arrays[graph_name], arrays[graph_name])
    assert all(not value.flags.writeable for value in loaded.arrays.values())
    with pytest.raises(ValueError, match="Solver implementation changed; historical viewing only"):
        load_section_result(path)
    with pytest.raises(ValueError, match="historical read-only.*continuation requires recalculation"):
        _unpack_record_graph(loaded.arrays[graph_name], loaded.arrays,
                             maximum_unpacked_bytes=8 * 1024**3)


@pytest.mark.parametrize("value", [np.zeros(5), np.zeros(6, dtype=np.float32), np.full(6, np.inf)])
def test_section_writer_rejects_invalid_magnetic_plan_coefficients(storage_result, tmp_path, value):
    plan = replace(storage_result.simulation.incident_plan, dipole_bx_t=value)
    altered = replace(storage_result, simulation=replace(storage_result.simulation, incident_plan=plan))
    with pytest.raises(ValueError, match="finite float64 values at all three stages"):
        save_section_result(altered, tmp_path / "invalid.temsection")


@pytest.mark.parametrize("missing", ["checkpoint.kinetic_energy_ev", "incident.kinetic_energy_ev"])
def test_current_seed_missing_executed_energy_cannot_resume(storage_result, tmp_path, missing):
    store, manifest, bundle = seed_store(tmp_path, storage_result)
    arrays = dict(bundle.arrays)
    del arrays[missing]
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        codec=INCIDENT_SEED_CODEC, arrays=arrays, metadata=bundle.metadata)
    with pytest.raises(ArtifactIntegrityError, match="lacks executed energy"):
        store.get_incident_simulation_seed(manifest)


@pytest.mark.parametrize("invalid", [np.zeros((2, 2)), np.full((2, 2), np.inf),
                                      np.ones((2, 2), dtype=np.float32), np.ones((1, 2))])
def test_seed_rejects_malformed_checkpoint_energy(storage_result, tmp_path, invalid):
    store, manifest, bundle = seed_store(tmp_path, storage_result)
    arrays = dict(bundle.arrays, **{"checkpoint.kinetic_energy_ev": invalid})
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        codec=INCIDENT_SEED_CODEC, arrays=arrays, metadata=bundle.metadata)
    with pytest.raises(ArtifactIntegrityError, match="Incident seed metadata"):
        store.get_incident_simulation_seed(manifest)


def test_plan_codec_retains_electric_identity_without_serializing_runtime_field(storage_result):
    identity = "a" * 64
    plan = replace(storage_result.simulation.incident_plan,
        electric_field=SimpleNamespace(numerical_identity=identity, huge_runtime_array=np.ones(8000)),
        electric_field_identity=identity, electric_reference_invariant_ev=123456.)
    arrays = {}
    tree = _pack(plan, arrays, {}, _data_classes())
    restored = _unpack(tree, arrays, _data_classes())
    assert restored.electric_field is None
    assert restored.electric_field_identity == identity
    assert restored.electric_reference_invariant_ev == 123456.
    assert all(value.size < 8000 for value in arrays.values())


def test_plan_codec_rejects_missing_electric_identity_contract(storage_result):
    arrays = {}
    tree = _pack(storage_result.simulation.incident_plan, arrays, {}, _data_classes())
    del tree["fields"]["electric_field_identity"]
    with pytest.raises(ValueError, match="electric-field identity; historical viewing only"):
        _unpack(tree, arrays, _data_classes())
