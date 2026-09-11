"""Shared-field coherent transport evidence; domain limits remain explicit."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.effective_source import EffectiveGunSource, bind_effective_source
from temsim.physics.canonical_phase import canonical_basis_matrix, validate_canonical_map
from temsim.physics.gun_wave_transport import canonical_magnus_step, _generator, build_gun_wave_checkpoint
from temsim.physics.ray_integrator import canonical_rk4_step
from temsim.simulation_modes import switch_mode


def test_quadratic_generator_matches_shared_canonical_particle_equations():
    u = np.array((1e-8, -2e-8, 1e-5, -2e-5))
    g, kx, ky, h = 220., 1234., -987., 1e-7
    source = np.array((u[0], u[2]+g*u[1], u[1], u[3]-g*u[0]))
    out = canonical_rk4_step(*source, h, g, g, g, kx, kx, kx, ky, ky, ky, 0., 0., 0., 0., 0., 0.)
    expected = np.array((out[0], out[2], out[1]-g*out[2], out[3]+g*out[0]))
    matrix = canonical_magnus_step(h, (g,)*3, (kx,)*3, (ky,)*3)
    np.testing.assert_allclose(matrix@u, expected, rtol=1e-12, atol=1e-20)
    assert validate_canonical_map(matrix) < 1e-12


def test_variable_field_magnus_converges_against_fine_independent_rk4():
    u0 = np.array((2e-8, -1e-8, 3e-6, -2e-6))
    length = .0001
    def coefficients(z):
        return 200+3000*z/length, 1000*(1+z/length), -300*(1-z/length)
    def trace(n, use_magnus):
        u, h = u0.copy(), length/n
        for i in range(n):
            z = i*h
            start, mid, end = coefficients(z), coefficients(z+h/2), coefficients(z+h)
            if use_magnus:
                u = canonical_magnus_step(h, *zip(start, mid, end))@u
            else:
                # Independent direct matrix ODE RK4, without exponential maps.
                a, b, c = (_generator(*v) for v in (start, mid, end))
                k1 = a@u
                k2 = b@(u+h*k1/2)
                k3 = b@(u+h*k2/2)
                k4 = c@(u+h*k3)
                u += h*(k1+2*k2+2*k3+k4)/6
        return u
    reference = trace(2048, False)
    errors = [np.linalg.norm(trace(n, True)-reference) for n in (4, 8, 16)]
    assert errors[1] < errors[0]/10
    assert errors[2] < errors[1]/10


def _state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    switch_mode(state, "ideal")
    state.step_mm = .2
    state.electron_gun.effective_source = bind_effective_source(state.electron_gun,
        EffectiveGunSource(1e-9, energy_fwhm_ev=0, grid_pixels=256))
    state.electron_gun.source_representation = "effective_gaussian_schell"
    return state


@pytest.fixture(scope="module")
def assembled_checkpoint():
    state = _state()
    before = capture_instrument_snapshot(state).digest
    progress = []
    checkpoint = build_gun_wave_checkpoint(state, progress_callback=lambda done, total, label: progress.append((done, total)))
    assert progress[0][0] == 0
    assert progress[-1][0] == progress[-1][1]
    assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))
    assert checkpoint.snapshot.digest == before
    assert capture_instrument_snapshot(state).digest == before
    return checkpoint


def test_actual_assembled_ideal_gun_field_chain_keeps_source_and_snapshot(assembled_checkpoint):
    checkpoint = assembled_checkpoint
    state = checkpoint.snapshot.restore()
    assert checkpoint.plane_z_mm == state.sample.upper_surface_z_mm
    assert checkpoint.execution["output_boundary"] == "finite-specimen-upper-face"
    assert checkpoint.plane_z_mm < state.sample.z_mm
    assert checkpoint.execution["source_id"] == checkpoint.emission.digest
    assert checkpoint.execution["validation_status"] == "NOT_VALIDATED"
    assert 0 < checkpoint.beam.total_weight <= 1+1e-12
    assert checkpoint.beam.reference_plane == "gun_exit_source_electron"


def test_gun_wave_uses_existing_persistent_store_exactly_and_never_recomputes(assembled_checkpoint, tmp_path, monkeypatch):
    from temsim.artifact_store import ArtifactStore
    from temsim.physics.gun_wave_cache import (gun_wave_manifest, save_gun_wave_checkpoint,
        load_gun_wave_checkpoint, cached_gun_wave_checkpoint)
    cp = assembled_checkpoint
    state = cp.snapshot.restore()
    store = ArtifactStore(tmp_path, quota_bytes=64*1024**2)
    manifest = gun_wave_manifest(state)
    save_gun_wave_checkpoint(store, manifest, cp)
    def forbidden(*args, **kwargs):
        raise AssertionError("An exact persisted source checkpoint must not repeat propagation")
    monkeypatch.setattr("temsim.physics.gun_wave_cache._execute_gun_wave_plan", forbidden)
    restored, reused = cached_gun_wave_checkpoint(state, store)
    assert reused and restored.digest == cp.digest
    for old, new in zip(cp.beam.modes, restored.beam.modes):
        np.testing.assert_array_equal(old.plane.amplitude, new.plane.amplitude)
        assert not new.plane.amplitude.flags.writeable
        with pytest.raises(ValueError):
            new.plane.amplitude.setflags(write=True)
    state.virtual_observation_z_mm += 1.
    display_only, reused = cached_gun_wave_checkpoint(state, store)
    assert reused and display_only.digest == cp.digest
    # Keep the original full snapshot; a display change does not relabel data.
    assert display_only.snapshot.digest == cp.snapshot.digest
    assert capture_instrument_snapshot(state).digest != cp.snapshot.digest
    state.lenses[1].percent += .00001
    assert load_gun_wave_checkpoint(store, gun_wave_manifest(state)) is None


def test_specimen_change_reuses_exact_upper_face_operator_and_retains_old_point(assembled_checkpoint, tmp_path, monkeypatch):
    from temsim.artifact_store import ArtifactStore
    from temsim.physics.gun_wave_cache import gun_wave_manifest, save_gun_wave_checkpoint, cached_gun_wave_checkpoint
    parent = assembled_checkpoint
    state = parent.snapshot.restore()
    store = ArtifactStore(tmp_path, quota_bytes=64*1024**2)
    save_gun_wave_checkpoint(store, gun_wave_manifest(state), parent)
    original_digest = parent.digest
    def forbidden(*args, **kwargs):
        raise AssertionError("An unchanged upper-face operator must not repeat coherent propagation")
    monkeypatch.setattr("temsim.physics.gun_wave_cache._execute_gun_wave_plan", forbidden)
    state.sample.inserted = not state.sample.inserted
    state.sample.centre_x_nm += 5.
    requested = capture_instrument_snapshot(state)
    child, reused = cached_gun_wave_checkpoint(state, store)
    assert reused
    assert child.snapshot.digest == requested.digest
    assert child.digest != parent.digest
    assert child.execution["reuse"]["parent_checkpoint_id"] == original_digest
    assert child.execution["executed_snapshot_id"] == parent.snapshot.digest
    assert child.execution["stage_signature"] == parent.execution["stage_signature"]
    assert parent.digest == original_digest
    for old, new in zip(parent.beam.modes, child.beam.modes):
        np.testing.assert_array_equal(old.plane.amplitude, new.plane.amplitude)
        assert old.weight_per_reference_electron == new.weight_per_reference_electron
    # Independent cold execution at the changed physical work point: no
    # additional fitting, preset or numeric tolerance is used for exact reuse.
    cold = build_gun_wave_checkpoint(state)
    assert cold.snapshot.digest == requested.digest
    for actual, expected in zip(child.beam.modes, cold.beam.modes):
        np.testing.assert_array_equal(actual.plane.amplitude, expected.plane.amplitude)
        np.testing.assert_array_equal(actual.plane.basis_m, expected.plane.basis_m)
        np.testing.assert_array_equal(actual.plane.origin_m, expected.plane.origin_m)
        np.testing.assert_array_equal(actual.plane.curvature_m1, expected.plane.curvature_m1)
        assert actual.weight_per_reference_electron == expected.weight_per_reference_electron


def test_replaced_cif_reuses_only_incident_operator_not_the_old_specimen(assembled_checkpoint, tmp_path, monkeypatch):
    from temsim.artifact_store import ArtifactStore
    from temsim.physics.gun_wave_cache import gun_wave_manifest, save_gun_wave_checkpoint, cached_gun_wave_checkpoint
    from temsim.specimen.source import active_cif_path
    state = assembled_checkpoint.snapshot.restore()
    path = tmp_path / "selected.cif"
    path.write_bytes(Path(active_cif_path(state.sample)).read_bytes())
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=64*1024**2)
    save_gun_wave_checkpoint(store, gun_wave_manifest(assembled_checkpoint.snapshot.restore()), assembled_checkpoint)
    def forbidden(*args, **kwargs):
        raise AssertionError("CIF content does not enter the finite-upper-face vacuum operator")
    monkeypatch.setattr("temsim.physics.gun_wave_cache._execute_gun_wave_plan", forbidden)
    parent, reused = cached_gun_wave_checkpoint(state, store)
    assert reused
    save_gun_wave_checkpoint(store, gun_wave_manifest(state), parent)
    # Real library structures, replaced at the SAME path. This does not test
    # material scattering; it tests dependency ownership at its entrance.
    path.write_bytes((Path(__file__).resolve().parents[1] / "configs/reference_samples/Au.cif").read_bytes())
    child, reused = cached_gun_wave_checkpoint(state, store)
    assert reused and child.snapshot.digest == capture_instrument_snapshot(state).digest
    assert child.execution["reuse"]["parent_checkpoint_id"] == parent.digest
    before = next(row for row in parent.snapshot.external_inputs if row["role"] == "specimen:cif")
    after = next(row for row in child.snapshot.external_inputs if row["role"] == "specimen:cif")
    assert before["path"] == after["path"] and before["sha256"] != after["sha256"]
    with pytest.raises(ValueError, match="Changed specimen:cif"):
        parent.snapshot.restore()
    for old, new in zip(parent.beam.modes, child.beam.modes):
        np.testing.assert_array_equal(old.plane.amplitude, new.plane.amplitude)


@pytest.mark.parametrize("edit", ["objective", "plane", "thickness", "aperture", "step"])
def test_incident_operator_cache_misses_changed_physical_dependencies(assembled_checkpoint, edit):
    from temsim.physics.gun_wave_cache import gun_wave_manifest, PRODUCT
    state = assembled_checkpoint.snapshot.restore()
    before = gun_wave_manifest(state).calculation_signatures[PRODUCT]
    if edit == "objective":
        state.objective_lens.percent += .00001
    elif edit == "plane":
        state.sample.z_mm += .00001
    elif edit == "thickness":
        state.sample.thickness_nm += .01
    elif edit == "aperture":
        active = next(a for a in state.apertures if a.enabled
                      and state.electron_gun.exit_plane_z_mm < a.z_mm < state.sample.z_mm)
        active.offset_x_mm += 1e-8
    else:
        state.step_mm *= .9
    assert gun_wave_manifest(state).calculation_signatures[PRODUCT] != before


def test_effective_source_particles_reach_actual_column_without_mutating_old_gun():
    from temsim.physics.simulation import run
    state = _state()
    state.electron_gun.emitter.ray_count = 49
    source = state.electron_gun.effective_source
    from temsim.instrument_snapshot import encode_instrument
    legacy = encode_instrument(state.electron_gun.emitter)
    result = run(state, optical_only=True)
    assert result.gun_trace.source_record["model_id"] == source.model_id
    assert result.incident.x.shape[1] == 49
    assert np.all(np.isfinite(result.incident.x))
    assert encode_instrument(state.electron_gun.emitter) == legacy


def test_gun_wave_portable_working_point_keeps_full_phase_and_readonly_diagnostics(assembled_checkpoint, tmp_path, monkeypatch):
    from temsim.working_point import WorkingPointCheckpoint
    cp = assembled_checkpoint
    point = WorkingPointCheckpoint.from_gun_wave(cp)
    assert point.stage_signature == cp.execution["stage_signature"]
    path = tmp_path / "gun.temwp"
    point.write_package(path)
    loaded = WorkingPointCheckpoint.read_package(path)
    assert loaded.digest == point.digest
    assert loaded.gun_wave_checkpoint().digest == cp.digest
    for mode in cp.beam.modes:
        assert mode.plane.curvature_m1 is not None
    def forbidden(*args):
        raise AssertionError("Read-only diagnostics must not restore/evaluate a gun or live instrument")
    monkeypatch.setattr(type(cp.snapshot), "restore", forbidden)
    reader = loaded.observables
    assert reader.get("wave_source_fraction").value == cp.beam.total_weight
    assert reader.get("wave_radius95").status == "AVAILABLE"
    assert reader.get("alpha95").status == "NOT_COMPUTED"
    assert reader.get("canonical_alpha95").status in {"AVAILABLE", "OUT_OF_VALIDATED_RANGE"}


def test_source_transport_cancel_never_publishes_checkpoint():
    state = _state()
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(InterruptedError, match="cancelled"):
        build_gun_wave_checkpoint(state, cancelled=lambda: True)
    assert capture_instrument_snapshot(state).digest == before


def test_incident_plane_is_a_physical_face_and_excludes_actions_inside_material():
    from types import SimpleNamespace
    from temsim.physics.gun_wave_transport import specimen_entrance_z_mm, upstream_component_events
    state = _state()
    state.sample.thickness_nm = 10.
    plane = specimen_entrance_z_mm(state)
    assert plane == state.sample.z_mm-5e-6
    element = SimpleNamespace(key="bounded-kick-fixture", enabled=True,
        kick_events=lambda **kw: [(plane-1e-6, .001, 0.), (plane+1e-6, .002, 0.)])
    # Isolated event-boundary test, not a synthetic specimen illumination.
    state.deflectors = [element]
    state.corrector_elements = []
    events, _ = upstream_component_events(state)
    assert events == [(plane-1e-6, .001, 0.)]
    state.sample.inserted = False
    assert specimen_entrance_z_mm(state) == plane
    state.sample.thickness_nm = -1.
    with pytest.raises(ValueError, match="non-negative thickness"):
        specimen_entrance_z_mm(state)
