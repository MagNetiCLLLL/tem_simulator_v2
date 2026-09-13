"""Executed energy resume/identity checks, NOT full-gun convergence evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
from temsim.physics.radial_gun_wave import RadialGunNumerics
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from test_wave_detector_readout import checkpoint


def test_actual_completed_energy_resumes_without_reexecuting_it_or_losing_phase(tmp_path, monkeypatch):
    import temsim.physics.surface_gun_wave as module
    gun = default_state().electron_gun
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    store = ExecutedWaveStore(tmp_path, "executed-coarse-gun-cache-regression-not-acceptance", 1<<30)
    options = dict(surface=SurfaceWaveNumerics(element_order=2, radial_nodes=97, axial_nodes=65,
        outer_radius_factor=4.), radial=RadialGunNumerics(radial_modes=8, potential_quadrature=32,
        relative_axial_step=.1, field_step_mm=.5), _energy_cache=store)
    first = []
    def interrupt(*result):
        first.append(result)
        raise InterruptedError("After the first complete energy was committed")
    with threadpool_limits(1), pytest.raises(InterruptedError, match="first complete"):
        module.build_surface_gun_checkpoint(gun, **options, _mode_completed=interrupt)
    assert len(first) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("pending-*"))
    original = module.prepare_round_gun
    computed, restored = [], []
    def counted(gun, height, energy, *args, **kwargs):
        computed.append(energy)
        return original(gun, height, energy, *args, **kwargs)
    monkeypatch.setattr(module, "prepare_round_gun", counted)
    with threadpool_limits(1):
        actual, near = module.build_surface_gun_checkpoint(gun, **options,
            _mode_completed=lambda *value: restored.append(value))
    assert len(restored) == len(actual.beam.modes) == 3
    assert computed == [mode.energy_ev for mode in near.modes[1:]]
    np.testing.assert_array_equal(restored[0][0].plane.amplitude, first[0][0].plane.amplitude)
    np.testing.assert_array_equal(restored[0][1].amplitude, first[0][1].amplitude)
    assert restored[0][0].axial_reference == first[0][0].axial_reference
    for name in first[0][4]:
        np.testing.assert_array_equal(restored[0][4][name], first[0][4][name])
        assert not restored[0][4][name].flags.writeable
    computed.clear()
    with threadpool_limits(1):
        again, again_near = module.build_surface_gun_checkpoint(gun, **options)
    assert not computed
    assert again.digest == actual.digest and again_near.digest == near.digest
    histories = actual.record["executed_energy_histories"]
    assert len(histories) == 3
    for row in histories:
        cached = store.get(row["cache_key"])
        assert cached.record["executed_identity_digest"] == row["executed_identity_digest"]
        assert cached.beam.modes[0].mode_id == row["mode_id"]
        assert "covariant_derivatives" in store.auxiliary_arrays(row["cache_key"])
    # Force fresh physics while retaining a new complete history. No old
    # energy may be reused just because mandatory storage remains enabled.
    with threadpool_limits(1), pytest.raises(InterruptedError, match="first complete"):
        module.build_surface_gun_checkpoint(gun, **options, _reuse_energy_cache=False,
                                            _mode_completed=interrupt)
    assert computed == [near.modes[0].energy_ev]
    assert len(list(tmp_path.glob("*.json"))) == 3
    # A changed upstream voltage must miss; stop BEFORE expensive propagation.
    gun.extractor.voltage_kv += .1
    def refuse(*args, **kwargs):
        raise InterruptedError("Changed physical input correctly requires propagation")
    monkeypatch.setattr(module, "prepare_round_gun", refuse)
    with threadpool_limits(1), pytest.raises(InterruptedError, match="Changed physical input"):
        module.build_surface_gun_checkpoint(gun, **options)
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_auxiliary_arrays_are_atomic_dependency_bound_and_checksum_checked(tmp_path):
    store = ExecutedWaveStore(tmp_path, "array-codec-fixture", 1<<20)
    key = store.key("whole-complex-state")
    wave = checkpoint()
    writer = store.writer(key, wave.beam.reference_plane)
    writer.append(wave.beam.modes[0])
    original = np.arange(12).reshape(3, 4)*(1+2j)
    writer.append_auxiliary("near_amplitude", original)
    assert store.auxiliary_arrays(key) is None
    saved = writer.finish(wave.plane_z_mm, wave.reference_current_a, {})
    writer.abort()
    np.testing.assert_array_equal(store.auxiliary_arrays(key)["near_amplitude"], original)
    manifest = json.loads((saved.beam.modes.root/"manifest.json").read_text())
    path = saved.beam.modes.root/manifest["auxiliary"]["near_amplitude"]["file"]
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError, match="checksum"):
        store.auxiliary_arrays(key)
    with pytest.raises(ValueError, match="dependency"):
        ExecutedWaveStore(tmp_path, "changed-implementation", 1<<20).auxiliary_arrays(key)


@pytest.mark.parametrize("name", ["../external", "", "A", "a/b", "x"*81])
def test_auxiliary_names_cannot_escape_writer(tmp_path, name):
    store = ExecutedWaveStore(tmp_path, "codec-fixture", 1<<20)
    writer = store.writer(store.key("bad"), "test")
    try:
        with pytest.raises(ValueError, match="array name"):
            writer.append_auxiliary(name, np.ones(2))
    finally:
        writer.abort()
    assert not list(tmp_path.iterdir())


def test_cancelled_complete_mode_does_not_publish_partial_state(tmp_path):
    from temsim.physics.surface_mode_cache import preserve_mode, mode_key, HISTORY_FIELDS
    from temsim.physics.surface_wave import SurfaceWaveMode
    store = ExecutedWaveStore(tmp_path, "cancel-fixture", 1<<20)
    wave = checkpoint()
    identity = {"mode_index": 0, "energy_ev": .3, "mixture_weight": 1.}
    def verify():
        raise InterruptedError("inputs changed before commit")
    with pytest.raises(InterruptedError, match="before commit"):
        preserve_mode(store, identity, wave.beam.modes[0], SurfaceWaveMode(.3, 1., np.ones((2, 2)), {}),
            {}, {}, {key: np.ones((2, 2)) for key in HISTORY_FIELDS},
            z_mm=450., current_a=1e-7, verify=verify)
    assert store.get(mode_key(store, identity)) is None
    assert not list(tmp_path.iterdir())
