from dataclasses import replace
import json
import multiprocessing
from threading import Barrier, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.artifact_store import (
    ArtifactIntegrityError,
    ArtifactStore,
)
from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_manifest import (
    SOLVER_IMPLEMENTATION_SCHEMA,
    assert_external_inputs_unchanged,
    capture_calculation_manifest,
    changed_external_inputs,
    resolved_assembly_geometry_fingerprint,
    resolved_assembly_geometry_fingerprints,
)
from temsim.optics.column import default_state
from temsim.gui.calculation_controller import CalculationController
from temsim.immutable_json import canonical_json_bytes, json_digest
from temsim.physics.core import PropagationCheckpoints
from temsim.physics.simulation import run as run_ray_simulation


def _hold_artifact_store_lock(root, entered, release):
    store = ArtifactStore(root, quota_bytes=100_000_000)
    with store._locked():
        entered.set()
        if not release.wait(15.0):
            raise RuntimeError("Timed out waiting to release artifact lock")


def _enter_artifact_store_lock(root, started, entered):
    started.set()
    store = ArtifactStore(root, quota_bytes=100_000_000)
    with store._locked():
        entered.set()


def _assembled_state():
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    return state, selection


def _manifest(state, selection, **kwargs):
    return capture_calculation_manifest(
        state,
        selection=selection,
        ray_count=25,
        step_mm=0.1,
        created_at_utc=kwargs.pop("created_at_utc", "2026-09-05T10:00:00+00:00"),
        **kwargs,
    )


def _checkpoints(offset=0.0):
    z = np.asarray([100.0, 105.0, 110.0])
    shape = (z.size, 4)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return PropagationCheckpoints(
        z_mm=z,
        x_m=(values + offset) * 1.0e-9,
        tx_rad=(values + offset) * 1.0e-6,
        y_m=(values + offset + 1.0) * 1.0e-9,
        ty_rad=(values + offset + 1.0) * 1.0e-6,
    )


def test_manifest_is_repeatable_and_timestamp_is_not_scientific_identity():
    state, selection = _assembled_state()
    first = _manifest(state, selection)
    second = _manifest(
        state,
        selection,
        created_at_utc="2026-09-05T11:00:00+00:00",
    )

    assert first.digest == second.digest
    assert first.geometry_fingerprint == second.geometry_fingerprint
    assert first.solver.state_schema_version == state.schema_version
    assert first.external_inputs
    with pytest.raises(TypeError):
        first.state_payload["beam_voltage_kv"] = 80.0


def test_manifest_export_is_detached_json_without_changing_cache_identity():
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    original_identity = canonical_json_bytes(manifest.identity_payload)
    original_digest = manifest.digest
    original_signatures = dict(manifest.calculation_signatures)
    assert manifest.external_inputs

    document = manifest.to_dict()
    decoded = json.loads(json.dumps(document, allow_nan=False))
    assert decoded == document
    assert decoded["external_inputs"][0]["sha256"] == manifest.external_inputs[0].sha256
    assert decoded["solver"]["package_version"] == manifest.solver.package_version
    exported_identity = {
        key: value for key, value in decoded.items()
        if key not in {"created_at_utc", "solver", "digest"}
    }
    assert json_digest(exported_identity) == original_digest

    # Exported records may be edited or written without owning any live or
    # immutable manifest data, including nested external-input dataclasses.
    document["external_inputs"][0]["sha256"] = "changed"
    document["state_payload"]["beam_voltage_kv"] = 80.0
    document["solver"]["package_version"] = "changed"
    document["calculation_signatures"]["incident"] = "changed"
    assert canonical_json_bytes(manifest.identity_payload) == original_identity
    assert manifest.digest == original_digest
    assert dict(manifest.calculation_signatures) == original_signatures


def test_geometry_fingerprint_ignores_percent_but_tracks_position_and_poles():
    state, _selection = _assembled_state()
    baseline = resolved_assembly_geometry_fingerprint(state)
    state.lenses[0].percent += 1.0
    assert resolved_assembly_geometry_fingerprint(state) == baseline

    state.lenses[0].z_mm += 0.25
    assert resolved_assembly_geometry_fingerprint(state) != baseline

    assembly = state._resolved_assembly
    part = next(row for row in assembly.parts if "pole" in row.key)
    changed_data = dict(part.data)
    dimension = next(
        name
        for name in changed_data
        if "diameter_mm" in name or name == "pole_gap_mm"
    )
    changed_data[dimension] = float(changed_data[dimension]) + 0.1
    changed_part = replace(part, data=changed_data)
    changed_assembly = replace(
        assembly,
        parts=tuple(
            changed_part if row is part else row for row in assembly.parts
        ),
    )
    assert (
        resolved_assembly_geometry_fingerprint(changed_assembly)
        != resolved_assembly_geometry_fingerprint(assembly)
    )


def test_geometry_fingerprint_normalises_only_binary_roundoff():
    state, _selection = _assembled_state()
    state.projector_lens_p1.z_mm = 3025.9
    baseline = resolved_assembly_geometry_fingerprint(state)

    state.projector_lens_p1.z_mm = np.nextafter(3025.9, 0.0)
    assert state.projector_lens_p1.z_mm == 3025.8999999999996
    assert resolved_assembly_geometry_fingerprint(state) == baseline

    state.projector_lens_p1.z_mm = 3025.900000001
    assert resolved_assembly_geometry_fingerprint(state) != baseline


def test_geometry_fingerprints_expose_stage_scopes():
    state = default_state()
    fingerprints = resolved_assembly_geometry_fingerprints(state)

    assert set(fingerprints) == {"upstream", "post_sample", "complete"}
    assert all(len(value) == 64 for value in fingerprints.values())


def test_manifest_detects_external_cif_mutation(tmp_path):
    state, selection = _assembled_state()
    cif = tmp_path / "sample.cif"
    cif.write_text("data_first\n", encoding="utf-8")
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(cif)
    manifest = _manifest(state, selection)

    cif.write_text("data_second\n", encoding="utf-8")

    changed = changed_external_inputs(manifest)
    assert [row.role for row in changed] == ["specimen:cif"]
    with pytest.raises(RuntimeError, match="specimen:cif"):
        assert_external_inputs_unchanged(manifest)


def test_field_map_descriptor_and_current_source_bytes_enter_identity(tmp_path):
    state, selection = _assembled_state()
    source = tmp_path / "objective-field.npz"
    source.write_bytes(b"first-field-map")
    state.lens_field_map_descriptors = {
        "objective_lens": {
            "source_path": str(source),
            "source_sha256": "descriptor-sha-from-import",
            "geometry_fingerprint": "bound-geometry",
        }
    }
    first = _manifest(state, selection)
    source.write_bytes(b"replacement-field-map")
    second = _manifest(state, selection)

    assert (
        first.calculation_signatures["incident"]
        != second.calculation_signatures["incident"]
    )
    field_inputs = [
        row for row in first.external_inputs
        if row.role == "lens_field_map:objective_lens"
    ]
    assert len(field_inputs) == 1
    assert field_inputs[0].sha256 != [
        row for row in second.external_inputs
        if row.role == "lens_field_map:objective_lens"
    ][0].sha256


def test_checkpoint_store_survives_restart_and_returns_read_only_arrays(tmp_path):
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    first_store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)
    identity = first_store.put_propagation_checkpoints(
        manifest, _checkpoints()
    )

    second_store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)
    restored = second_store.get_propagation_checkpoints(manifest)

    assert len(identity) == 64
    assert restored is not None
    assert np.array_equal(restored.x_m, _checkpoints().x_m)
    assert not restored.x_m.flags.writeable


@pytest.mark.parametrize("with_vector_map", [False, True])
def test_complete_incident_seed_restarts_ray_solver(tmp_path, with_vector_map):
    state, selection = _assembled_state()
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.ac_deflector.scan_enabled = False
    if with_vector_map:
        from temsim.physics.lens_field_provider import (
            bind_imported_lens_field_map, lens_geometry_binding, load_magnetic_field_map,
        )
        lens = next(lens for lens in state.lenses if lens.key == "objective_lens")
        binding = lens_geometry_binding(state, lens.key, lens)
        path = tmp_path / "restart_field.npz"
        shape = (3, 3, 3)
        np.savez(path, x_m=np.linspace(-.1,.1,3), y_m=np.linspace(-.1,.1,3),
            z_m=lens.z_mm*.001 + np.linspace(-.001,.001,3),
            bx_t=np.full(shape, 1e-6), by_t=np.zeros(shape), bz_t=np.zeros(shape),
            metadata_json=json.dumps({"map_type": "cartesian_xyz"}))
        field_map = load_magnetic_field_map(path, geometry_binding=binding,
            provenance_kind="fem", reference_excitation_percent=100.0,
            source_note="Synthetic restart fixture, not measured/FEM data")
        bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    manifest = capture_calculation_manifest(
        state,
        selection=selection,
        ray_count=9,
        step_mm=5.0,
    )
    first = run_ray_simulation(state)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=100_000_000)

    store.put_incident_simulation_seed(manifest, first)
    restored = ArtifactStore(
        tmp_path / "cache", quota_bytes=100_000_000
    ).get_incident_simulation_seed(manifest)
    second = run_ray_simulation(state, existing_simulation=restored)

    assert restored is not None
    assert restored.incident_plan.signature == first.incident_plan.signature
    if with_vector_map:
        before_map = first.incident_plan.mapped_fields[0]
        restored_map = restored.incident_plan.mapped_fields[0]
        assert restored_map.fingerprint == before_map.fingerprint
        assert restored_map.scale == before_map.scale
        for before, after in zip(before_map.field_map.components_t,
                                 restored_map.field_map.components_t):
            np.testing.assert_array_equal(before, after)
            assert not after.flags.writeable
    assert second.metrics["column_segment_cache"]["mode"] == "full_incident"
    assert second.metrics["column_segment_cache"]["recomputed_history_rows"] == 0

    state.projector_lens_p1.z_mm += 0.5
    downstream_manifest = capture_calculation_manifest(
        state,
        selection=selection,
        ray_count=9,
        step_mm=5.0,
    )
    assert (
        downstream_manifest.calculation_signatures["incident"]
        == manifest.calculation_signatures["incident"]
    )
    assert store.get_incident_simulation_seed(downstream_manifest) is not None

    assembly = state._resolved_assembly
    pole = assembly.part("condenser_lens_1_lower_pole")
    changed_pole = replace(
        pole,
        data={
            **dict(pole.data),
            "pole_piece_bore_diameter_mm": float(
                pole.data.get("pole_piece_bore_diameter_mm", 5.76)
            ) + 0.25,
        },
    )
    state._resolved_assembly = replace(
        assembly,
        parts=tuple(
            changed_pole if row is pole else row for row in assembly.parts
        ),
    )
    upstream_manifest = capture_calculation_manifest(
        state,
        selection=selection,
        ray_count=9,
        step_mm=5.0,
    )
    assert (
        upstream_manifest.calculation_signatures["incident"]
        != manifest.calculation_signatures["incident"]
    )
    assert store.get_incident_simulation_seed(upstream_manifest) is None


def test_calculation_controller_exposes_manifest_and_persistence_hooks(tmp_path):
    state, selection = _assembled_state()
    manifest = CalculationController.build_calculation_manifest(
        state, 25, 0.1, selection=selection
    )
    result = SimpleNamespace(
        signatures=dict(manifest.calculation_signatures),
        simulation=SimpleNamespace(incident_checkpoints=_checkpoints()),
    )
    store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)

    identity = CalculationController.persist_incident_checkpoints(
        store, manifest, result
    )
    restored = CalculationController.load_persisted_incident_checkpoints(
        ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000),
        manifest,
    )

    assert identity is not None
    assert restored is not None
    assert np.array_equal(restored.ty_rad, _checkpoints().ty_rad)


def test_checkpoint_store_uses_scoped_geometry_and_rejects_tampering(tmp_path):
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)
    store.put_propagation_checkpoints(manifest, _checkpoints())

    state.projector_lens_p1.z_mm += 0.5
    downstream_changed = _manifest(state, selection)
    assert (
        downstream_changed.geometry_fingerprint
        != manifest.geometry_fingerprint
    )
    assert (
        downstream_changed.calculation_signatures["incident"]
        == manifest.calculation_signatures["incident"]
    )
    assert store.get_propagation_checkpoints(downstream_changed) is not None

    assembly = state._resolved_assembly
    pole = assembly.part("condenser_lens_1_lower_pole")
    changed_pole = replace(
        pole,
        data={
            **dict(pole.data),
            "pole_piece_bore_diameter_mm": float(
                pole.data.get("pole_piece_bore_diameter_mm", 5.76)
            ) + 0.25,
        },
    )
    state._resolved_assembly = replace(
        assembly,
        parts=tuple(
            changed_pole if row is pole else row for row in assembly.parts
        ),
    )
    upstream_changed = _manifest(state, selection)
    assert (
        upstream_changed.calculation_signatures["incident"]
        != manifest.calculation_signatures["incident"]
    )
    assert store.get_propagation_checkpoints(upstream_changed) is None

    array_path = next((tmp_path / "cache" / "objects").glob("*/*/array-*.npy"))
    data = bytearray(array_path.read_bytes())
    data[-1] ^= 0x01
    array_path.write_bytes(data)
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        store.get_propagation_checkpoints(downstream_changed)


def test_store_rejects_object_arrays_and_prunes_oldest_reference(tmp_path):
    state, selection = _assembled_state()
    first_manifest = _manifest(state, selection)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)
    with pytest.raises(TypeError, match="object"):
        store.put_array_bundle(
            first_manifest,
            product_key="incident",
            dependency_signature=first_manifest.calculation_signatures["incident"],
            arrays={"unsafe": np.asarray([object()], dtype=object)},
        )

    first_values = np.arange(2048, dtype=np.float64)
    store.put_array_bundle(
        first_manifest,
        product_key="incident",
        dependency_signature=first_manifest.calculation_signatures["incident"],
        arrays={"values": first_values},
    )
    first_size = store._store_size()

    state.lenses[0].percent += 2.0
    second_manifest = _manifest(state, selection)
    store.quota_bytes = first_size + 512
    store.put_array_bundle(
        second_manifest,
        product_key="incident",
        dependency_signature=second_manifest.calculation_signatures["incident"],
        arrays={"values": first_values + 1.0},
    )

    assert store.get_array_bundle(
        first_manifest,
        product_key="incident",
        dependency_signature=first_manifest.calculation_signatures["incident"],
    ) is None
    assert store.get_array_bundle(
        second_manifest,
        product_key="incident",
        dependency_signature=second_manifest.calculation_signatures["incident"],
    ) is not None


def test_store_mutex_is_reentrant_and_released_after_exception(tmp_path):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)

    with store._locked():
        with store._locked():
            pass
    with pytest.raises(RuntimeError, match="deliberate"):
        with store._locked():
            raise RuntimeError("deliberate lock-body failure")

    second = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    with second._locked():
        pass


def test_store_mutex_serialises_independent_processes(tmp_path):
    root = str(tmp_path / "cache")
    ArtifactStore(root, quota_bytes=100_000_000)
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    second_started = context.Event()
    second_entered = context.Event()
    first = context.Process(
        target=_hold_artifact_store_lock,
        args=(root, first_entered, release_first),
    )
    second = context.Process(
        target=_enter_artifact_store_lock,
        args=(root, second_started, second_entered),
    )
    try:
        first.start()
        assert first_entered.wait(10.0)
        second.start()
        assert second_started.wait(10.0)
        assert not second_entered.wait(0.4)
        release_first.set()
        assert second_entered.wait(10.0)
    finally:
        release_first.set()
        first.join(15.0)
        second.join(15.0)
        if first.is_alive():
            first.terminate()
            first.join(5.0)
        if second.is_alive():
            second.terminate()
            second.join(5.0)

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_two_store_instances_can_publish_and_read_concurrently(tmp_path):
    first_state, selection = _assembled_state()
    second_state, _ = _assembled_state()
    second_state.lenses[0].percent += 2.0
    manifests = (
        _manifest(first_state, selection),
        _manifest(second_state, selection),
    )
    stores = (
        ArtifactStore(tmp_path / "cache", quota_bytes=100_000_000),
        ArtifactStore(tmp_path / "cache", quota_bytes=100_000_000),
    )
    barrier = Barrier(3)
    failures = []

    def publish(index):
        try:
            manifest = manifests[index]
            values = np.arange(4096, dtype=np.float64) + float(index)
            barrier.wait()
            for _ in range(3):
                stores[index].put_array_bundle(
                    manifest,
                    product_key="incident",
                    dependency_signature=(
                        manifest.calculation_signatures["incident"]
                    ),
                    arrays={"values": values},
                )
                restored = stores[index].get_array_bundle(
                    manifest,
                    product_key="incident",
                    dependency_signature=(
                        manifest.calculation_signatures["incident"]
                    ),
                )
                assert restored is not None
                assert np.array_equal(restored.arrays["values"], values)
        except BaseException as exc:
            failures.append(exc)

    workers = [Thread(target=publish, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(15.0)

    assert all(not worker.is_alive() for worker in workers)
    assert failures == []


def test_prune_reclaims_crash_orphan_before_quota_decision(tmp_path):
    state, selection = _assembled_state()
    first_manifest = _manifest(state, selection)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=100_000_000)
    values = np.arange(4096, dtype=np.float64)
    first_identity = store.put_array_bundle(
        first_manifest,
        product_key="incident",
        dependency_signature=first_manifest.calculation_signatures["incident"],
        arrays={"values": values},
    )
    first_size = store._store_size()
    first_object = next(store.objects_root.glob("*/*"))
    store._reference_path(first_identity).unlink()

    state.lenses[0].percent += 2.0
    second_manifest = _manifest(state, selection)
    store.quota_bytes = first_size + 256
    store.put_array_bundle(
        second_manifest,
        product_key="incident",
        dependency_signature=second_manifest.calculation_signatures["incident"],
        arrays={"values": values + 1.0},
    )

    assert 33_000 <= first_size <= 35_000
    assert not first_object.exists()
    assert store.get_array_bundle(
        second_manifest,
        product_key="incident",
        dependency_signature=second_manifest.calculation_signatures["incident"],
    ) is not None


def test_store_initialisation_reclaims_unreferenced_objects(tmp_path):
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    root = tmp_path / "cache"
    store = ArtifactStore(root, quota_bytes=100_000_000)
    identity = store.put_array_bundle(
        manifest,
        product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        arrays={"values": np.arange(4096, dtype=np.float64)},
    )
    object_path = next(store.objects_root.glob("*/*"))
    store._reference_path(identity).unlink()
    pending_object = store.objects_root / ".pending-crashed-writer"
    pending_object.mkdir()
    (pending_object / "partial.npy").write_bytes(b"partial")
    pending_reference = (
        store.references_root / ".partial-reference.json.crashed.tmp"
    )
    pending_reference.write_bytes(b"partial")

    ArtifactStore(root, quota_bytes=100_000_000)

    assert not object_path.exists()
    assert not pending_object.exists()
    assert not pending_reference.exists()


@pytest.mark.parametrize("legacy_schema", [
    "temsim-solver-2026-09-segmented-v1", "temsim-solver-2026-09-segmented-v2",
    "temsim-solver-2026-09-vector-fields-v3",
    "temsim-solver-2026-09-six-stage-physics-v1",
    "temsim-solver-2026-09-magnetic-circuits-v2",
    "temsim-solver-2026-09-simulation-modes-v1",
])
def test_solver_implementation_bump_does_not_reuse_legacy_seed(tmp_path, legacy_schema):
    state, selection = _assembled_state()
    current = _manifest(state, selection)
    legacy = replace(
        current,
        solver=replace(
            current.solver,
            implementation_schema=legacy_schema,
        ),
    )
    store = ArtifactStore(tmp_path / "cache", quota_bytes=10_000_000)
    store.put_array_bundle(
        legacy,
        product_key="incident",
        dependency_signature=legacy.calculation_signatures["incident"],
        arrays={"values": np.arange(32, dtype=np.float64)},
    )

    assert SOLVER_IMPLEMENTATION_SCHEMA == "temsim-solver-2026-09-static-bh-v1"
    assert legacy.solver.digest != current.solver.digest
    assert store.get_array_bundle(
        current,
        product_key="incident",
        dependency_signature=current.calculation_signatures["incident"],
    ) is None
