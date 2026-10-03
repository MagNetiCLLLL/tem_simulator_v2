"""Conditional material instrument and double propagation; independent fixtures."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.inelastic_wave import _collide_slice, _trajectory_rng, _propagate_inelastic_specimen
from temsim.physics.wave_execution import InelasticWaveNumerics
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint, _momentum_velocity
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from test_segmented_tip_wave import quiet_state


def distribution(zero=.5, scatter=.4, absorbed=.1):
    return SimpleNamespace(absorbed_probability=absorbed, channels=(
        SimpleNamespace(key="real_zero_loss", probability=zero, energy_loss_ev=0., characteristic_angle_mrad=0., mean_events=0., approximation="fixture"),
        SimpleNamespace(key="real_plasmon", probability=scatter, energy_loss_ev=100., characteristic_angle_mrad=.02, mean_events=1., approximation="fixture")))


def mode():
    # Same 20 pm sampling and 40 pm width, with enough vacuum padding to
    # resolve the analytical carrier without a truncated Gaussian boundary.
    n = 64
    axis = (np.arange(n)-n//2)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    a = np.exp(-(xx*xx+yy*yy)/(4*(4e-11)**2))*(1+.2j)
    a /= np.linalg.norm(a)
    return WaveMode(PlaneWave(a, np.eye(2)*2e-11, np.array((1e-11, -2e-11)), np.eye(2)*20., np.array((1e-5, 2e-5))),
                    .4, TIP_REFERENCE, "material-fixture", 300., AxialWaveReference(1e-8, 1e-22))


def test_inelastic_jump_keeps_full_phase_changes_energy_and_canonical_carrier_units():
    old = mode()
    new, event = _collide_slice(old, np.ones(old.plane.amplitude.shape, bool), distribution(0., 1., 0.), _trajectory_rng(0, "phase"), 1.)
    assert new.energy_kev == pytest.approx(299.9)
    assert new.weight_per_reference_electron == old.weight_per_reference_electron
    assert new.axial_reference is old.axial_reference
    xy = old.plane.coordinates_m()
    expected = old.plane.full_amplitude(float(wavelength_m(300000)))*np.exp(2j*np.pi*np.einsum("i,iyx->yx", event["kick_rad"], xy)/float(wavelength_m(299900)))
    np.testing.assert_allclose(new.plane.full_amplitude(float(wavelength_m(299900))), expected, atol=5e-14)
    assert new.scattering_history[-1]["kind"] == "real_plasmon"


def test_sampled_instrument_matches_born_probabilities_in_finite_material():
    original = mode()
    inside = original.plane.coordinates_m()[0] > original.plane.origin_m[0]
    material_probability = float(np.sum(abs(original.plane.amplitude[inside])**2))
    expected = np.array([1-material_probability*.5, material_probability*.4, material_probability*.1])
    names = ("real_zero_loss", "real_plasmon", "effective_absorption")
    counts, samples = np.zeros(3), 2500
    rng = _trajectory_rng(937, "population")
    for _ in range(samples):
        result, event = _collide_slice(original, inside, distribution(), rng, 1.)
        counts[names.index(event["kind"])] += 1
        assert result.weight_per_reference_electron+event["absorbed_weight"] == pytest.approx(.4)
    sigma = np.sqrt(expected*(1-expected)/samples)
    assert np.all(abs(counts/samples-expected) < 5*sigma)


def test_counter_rng_is_independent_of_preceding_slices():
    first = _trajectory_rng(3, "tip", 0, 4, 7).random(10)
    _trajectory_rng(3, "tip", 0, 4, 6).random(1000)
    np.testing.assert_array_equal(first, _trajectory_rng(3, "tip", 0, 4, 7).random(10))
    assert not np.array_equal(first, _trajectory_rng(4, "tip", 0, 4, 7).random(10))


def test_outgoing_wave_propagates_remaining_material_at_new_energy_and_resumes(tmp_path, monkeypatch):
    state = quiet_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(Path(__file__).parent/"fixtures"/"cif"/"Si.cif")
    state.sample.inserted = True
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 30.
    state.sample.wave_frozen_phonon_enabled = False
    axis = (np.arange(128)-64)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(.2e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    reference = AxialWaveReference(1e-8, 1e-22)
    beam_mode = WaveMode(PlaneWave(amplitude, np.eye(2)*2e-11, np.zeros(2)), .4, TIP_REFERENCE, "specimen-fixture", 300., reference)
    checkpoint = TipGunCheckpoint(BeamState((beam_mode,), TIP_REFERENCE), state.sample.z_mm-.2e-6, 1e-9, {"fixture": "independent atomistic operator"})
    energies = []
    def material(energy_state):
        energies.append(energy_state.beam_voltage_kv)
        d = distribution(0., 1., 0.)
        d.channels[1].characteristic_angle_mrad = 0.
        return d
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", material)
    store = ExecutedWaveStore(tmp_path, "actual-specimen-operator-fixture", 1<<29)
    kw = dict(numerics=InelasticWaveNumerics(trajectories_per_mode=2), store=store,
        maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=0.,
        cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True)
    result = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert energies == pytest.approx([300., 299.9, 300., 299.9])
    assert len(result.beam.modes) == 2
    p1, v1 = _momentum_velocity(300000.); p2, v2 = _momentum_velocity(299900.)
    for value in result.beam.modes:
        assert value.energy_kev == pytest.approx(299.8)
        assert len(value.scattering_history) == 2
        assert value.axial_reference.flight_time_s-reference.flight_time_s == pytest.approx(.2e-9/v1+.2e-9/v2, rel=3e-6, abs=1e-23)
        assert value.axial_reference.longitudinal_action_j_s-reference.longitudinal_action_j_s == pytest.approx(.2e-9*(p1+p2), rel=3e-6, abs=1e-38)
    reused = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert reused.digest == result.digest
    assert len(energies) == 4
    # Interrupted execution must reuse a committed conditional slice and its
    # complete energy/history/RNG state, not restart from an invented pupil.
    interrupted_store = ExecutedWaveStore(tmp_path/"interrupted", "actual-specimen-operator-fixture", 1<<29)
    stop = [False]
    put = interrupted_store.put
    def cancel_after_commit(key, checkpoint):
        value = put(key, checkpoint)
        stop[0] = True
        return value
    monkeypatch.setattr(interrupted_store, "put", cancel_after_commit)
    kw["store"], kw["cancelled"] = interrupted_store, lambda: stop[0]
    with pytest.raises(InterruptedError):
        _propagate_inelastic_specimen(state, checkpoint, **kw)
    # Distributed transport converts eV/keV; allow < 5 ulps at 300 keV.
    assert energies[4:] == pytest.approx([300.], rel=0, abs=4*np.spacing(300.))
    stop[0] = False
    monkeypatch.setattr(interrupted_store, "put", put)
    resumed = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert energies[5:] == pytest.approx([299.9, 300., 299.9])
    for a, b in zip(resumed.beam.modes, result.beam.modes):
        np.testing.assert_allclose(a.plane.amplitude, b.plane.amplitude, rtol=0, atol=0)
        assert a.energy_kev == b.energy_kev
        assert a.scattering_history == b.scattering_history
    assert not list((tmp_path/"interrupted").glob("pending-*"))


def material_fixture(configurations=1, sources=1):
    """Small analytic periodic potential with actual complex/column operators."""
    state = quiet_state()
    state.sample.thickness_nm = .4
    n, spacing = 64, 2e-11
    axis = (np.arange(n)-n//2)*spacing
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(.08e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    modes = tuple(WaveMode(PlaneWave(amplitude*np.exp(.2j*i), np.eye(2)*spacing, np.zeros(2)),
        .4/(i+1), TIP_REFERENCE, f"analytic-source-{i}", 300.-i,
        AxialWaveReference(1e-8+i*1e-10, 1e-22)) for i in range(sources))
    checkpoint = TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), state.sample.z_mm-.2e-6,
        1e-9, {"fixture": "analytic material; not tip-chain qualification"})
    fine = (np.arange(2*n)-n)*spacing/2
    potential = .2*np.cos(2*np.pi*np.arange(2*n)/(2*n))[None, :]*np.ones((2*n, 1))
    configs = tuple(np.stack((potential*(i+1), potential*(i+2))) for i in range(configurations))
    prepared = SimpleNamespace(potential_configurations_v_angstrom=configs, metrics={"fixture": "periodic potential"})
    scene = SimpleNamespace(sample_contains_xy=lambda x, y: np.ones_like(x, dtype=bool))
    return state, checkpoint, (scene, prepared, fine, fine, np.array((2., 2.)))


def execute_fixture(tmp_path, monkeypatch, *, trajectories=4, configs=1, sources=1,
                    collision=None, progress=None, cancelled=lambda: False, store=None):
    from temsim.physics.inelastic_wave import _propagate_inelastic_attempt
    state, checkpoint, material = material_fixture(configs, sources)
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution",
        lambda _: distribution(1., 0., 0.) if collision is None else collision())
    store = store or ExecutedWaveStore(tmp_path, "analytic-material", 1<<28)
    return _propagate_inelastic_attempt(state, checkpoint,
        numerics=InelasticWaveNumerics(trajectories_per_mode=trajectories), store=store,
        maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=0.,
        cancelled=cancelled, progress_callback=progress, verify=lambda: None, use_cache=True,
        key=store.key("analytic", trajectories, configs, sources), material=material,
        refinement_factor=1, refinement_history=[])


def test_identity_histories_execute_once_with_full_complex_state_and_probability(tmp_path, monkeypatch):
    import temsim.physics.column_wave as column
    calls, actual = [], column._propagate_column
    def counted(*args, **kw):
        calls.append(args[1].plane_z_mm)
        return actual(*args, **kw)
    monkeypatch.setattr(column, "_propagate_column", counted)
    reference = execute_fixture(tmp_path/"one", monkeypatch, trajectories=1)
    calls.clear()
    grouped = execute_fixture(tmp_path/"many", monkeypatch, trajectories=32)
    assert len(calls) == 2  # two physical slices, not 32 duplicate executions
    assert len(grouped.beam.modes) == 1
    a, b = reference.beam.modes[0], grouped.beam.modes[0]
    for field in ("amplitude", "basis_m", "origin_m"):
        np.testing.assert_array_equal(getattr(a.plane, field), getattr(b.plane, field))
    assert a.axial_reference == b.axial_reference
    assert a.energy_kev == b.energy_kev
    assert a.scattering_history == b.scattering_history
    assert a.weight_per_reference_electron == b.weight_per_reference_electron
    row = grouped.record["modes"][0]
    assert row["represented_trajectories"] == 32
    assert row["input_weight"] == .4
    assert row["probability_residual"] == pytest.approx(0, abs=1e-13)
    for one, many in zip(reference.record["modes"][0]["steps"], row["steps"]):
        assert one["column"] == many["column"]


@pytest.mark.parametrize("collision", [lambda: distribution(.9999, .0001, 0.),
    lambda: distribution(1., 1e-30, 0.), lambda: distribution(1., 0., 1e-30)])
def test_any_nonzero_branch_probability_keeps_all_requested_histories(tmp_path, monkeypatch, collision):
    result = execute_fixture(tmp_path, monkeypatch, collision=collision)
    assert len(result.beam.modes) == 4
    assert all(row["represented_trajectories"] == 1 for row in result.record["modes"])
    assert result.beam.total_weight == pytest.approx(.4, abs=2e-12)


def test_grouping_never_merges_distinct_source_or_phonon_modes(tmp_path, monkeypatch):
    result = execute_fixture(tmp_path, monkeypatch, configs=2, sources=2)
    assert len(result.beam.modes) == 4
    assert len({mode.mode_id for mode in result.beam.modes}) == 4
    assert sorted({mode.energy_kev for mode in result.beam.modes}) == pytest.approx([299., 300.])
    assert result.beam.total_weight == pytest.approx(.6, abs=2e-12)
    assert all(row["represented_trajectories"] == 4 for row in result.record["modes"])


def test_identity_at_entrance_does_not_hide_later_random_material_slices(tmp_path, monkeypatch):
    calls = [0]
    def changed_distribution():
        calls[0] += 1
        return distribution(1., 0., 0.) if calls[0] % 2 else distribution(.9999, .0001, 0.)
    result = execute_fixture(tmp_path, monkeypatch, collision=changed_distribution)
    assert calls[0] == 8
    assert len(result.beam.modes) == 4
    assert all(not row["identity_history"] for row in result.record["modes"])


def test_grouped_probability_records_preserve_aperture_and_wall_losses():
    from temsim.physics.inelastic_wave import _scale_step_weights
    from temsim.immutable_json import freeze_json
    record = freeze_json([{"event": {"absorbed_weight": 0.}, "column": [{
        "input_weight": .125, "output_weight": .05, "losses": [
            {"component": "column_wall", "z_mm": 7., "lost_weight": .025},
            {"component": "aperture", "z_mm": 8., "input_weight": .1,
             "output_weight": .05, "lost_weight": .05}]}]}])
    scaled = _scale_step_weights(record, 8)
    row = scaled[0]["column"][0]
    assert row["input_weight"] == 1.
    assert row["output_weight"] == .4
    assert sum(loss["lost_weight"] for loss in row["losses"]) == pytest.approx(.6)
    assert [loss["z_mm"] for loss in row["losses"]] == [7., 8.]
    assert record[0]["column"][0]["input_weight"] == .125


def test_progress_precedes_expensive_slice_and_updates_before_trajectory_end(tmp_path, monkeypatch):
    import temsim.physics.specimen_wave_transport as transport
    messages, seen, phase = [], [], transport._material_phase
    def inspected(*args, **kw):
        seen.append(tuple(messages))
        return phase(*args, **kw)
    monkeypatch.setattr(transport, "_material_phase", inspected)
    execute_fixture(tmp_path, monkeypatch, progress=lambda d, t, text: messages.append((d, t, text)))
    assert seen and all(history for history in seen)
    assert "slice 1/2" in seen[0][-1][2]
    assert "slice 2/2" in seen[-1][-1][2]
    assert messages[-1][0] == messages[-1][1]


def test_slice_progress_cancellation_prevents_expensive_work(tmp_path, monkeypatch):
    cancel = [False]
    def progress(*args):
        cancel[0] = True
    def unexpected(*args, **kw):
        pytest.fail("A cancelled slice started expensive material work")
    monkeypatch.setattr("temsim.physics.specimen_wave_transport._material_phase", unexpected)
    with pytest.raises(InterruptedError):
        execute_fixture(tmp_path, monkeypatch, progress=progress, cancelled=lambda: cancel[0])
    assert not list(tmp_path.glob("*.json"))  # no partial checkpoint publication


def test_identity_proof_survives_cancelled_slice_continuation(tmp_path, monkeypatch):
    store = ExecutedWaveStore(tmp_path, "analytic-material", 1<<28)
    stop = [False]
    put = store.put
    def cancel_after_put(*args, **kw):
        checkpoint = put(*args, **kw)
        stop[0] = True
        return checkpoint
    monkeypatch.setattr(store, "put", cancel_after_put)
    with pytest.raises(InterruptedError):
        execute_fixture(tmp_path, monkeypatch, store=store, cancelled=lambda: stop[0])
    stop[0] = False
    monkeypatch.setattr(store, "put", put)
    resumed = execute_fixture(tmp_path, monkeypatch, store=store)
    assert len(resumed.beam.modes) == 1
    assert resumed.record["modes"][0]["represented_trajectories"] == 4
    fresh = execute_fixture(tmp_path/"fresh", monkeypatch)
    np.testing.assert_array_equal(resumed.beam.modes[0].plane.amplitude, fresh.beam.modes[0].plane.amplitude)
    assert resumed.record["modes"] == fresh.record["modes"]
