"""Numerical retry and cache contracts; no full tip-source imaging claim."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import specimen_wave_transport as transport
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.wave_reference import AxialWaveReference
from test_segmented_tip_wave import quiet_state


def entrance(state):
    n = 64
    x = (np.arange(n)-n//2)*2e-11
    xx, yy = np.meshgrid(x, x)
    amplitude = np.exp(-(xx**2+yy**2)/(2*(.07e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*2e-11, np.zeros(2)), .4,
        TIP_REFERENCE, "independent-specimen-fixture", 300., AxialWaveReference(1e-8, 1e-22))
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE),
        state.sample.z_mm-state.sample.thickness_nm*.5e-6, 1e-9,
        {"fixture": "independent specimen operator, not executed tip acceptance"})


def install_potential_fixture(monkeypatch):
    """Analytic smooth-potential builder, rebuilt on each requested grid."""
    import temsim.physics.wave_imaging as imaging
    calls = []
    def prepare(state, preset, **kw):
        n = state.sample.wave_grid_pixels
        fov = kw["field_of_view_angstrom_override"]
        axis = (np.arange(n)-n//2)*fov/n
        # Uniform phase has an exact bandwidth. Real finite material occupancy
        # and physical column propagation still run in the production adapter.
        potential = np.ones((2, n, n))*0.01
        calls.append((n, fov, kw["calculation_roi_centre_nm"],
                      state.sample.wave_frozen_phonon_seed))
        return SimpleNamespace(x_angstrom=axis, y_angstrom=axis,
            potential_configurations_v_angstrom=(potential,), slice_thicknesses_angstrom=np.array([2., 2.]),
            metrics={"fixture": "constant phase", "prepared_specimen_cache_hit": len(calls) > 1})
    monkeypatch.setattr(imaging, "prepare_specimen_potentials", prepare)
    return calls


def state_and_wave():
    state = quiet_state()
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 64
    state.sample.wave_field_of_view_angstrom = 30.
    return state, entrance(state)


def test_retry_rebuilds_potential_preserves_physical_domain_source_and_phase(monkeypatch):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    original_source = source.digest
    original = vars(state.sample).copy()
    operation = transport._slice_phase
    def needs_finer(mode, potential, x, y, sigma, fraction):
        if len(x) < 128:
            raise WaveSamplingError("controlled initial lattice failure")
        return operation(mode, potential, x, y, sigma, fraction)
    monkeypatch.setattr(transport, "_slice_phase", needs_finer)
    result = transport._propagate_specimen(state, source, grid_numerics=WaveGridNumerics(maximum_pixels=256, specimen_phase_method="sampled"))
    assert [c[0] for c in calls] == [64, 128]
    assert calls[0][1:] == calls[1][1:]
    assert vars(state.sample) == original
    assert source.digest == original_source
    assert result.record["material_grid_refinement"]["factor"] == 2
    assert len(result.record["material_grid_refinement"]["attempts"]) == 1
    assert 0 < result.beam.total_weight <= source.beam.total_weight
    assert result.beam.modes[0].axial_reference.flight_time_s > source.beam.modes[0].axial_reference.flight_time_s
    assert np.max(abs(result.beam.modes[0].plane.amplitude.imag)) > 0


def test_retry_is_opt_out_and_never_catches_physical_errors(monkeypatch):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    def unresolved(*args):
        raise WaveSamplingError("not yet resolved")
    with pytest.raises(WaveSamplingError):
        transport._with_material_refinement(state, source, unresolved,
            grid_numerics=WaveGridNumerics(automatic_refinement=False, specimen_phase_method="sampled"), cancelled=lambda: False, progress_callback=None)
    assert len(calls) == 1
    def physical_failure(*args):
        raise ValueError("unsupported physical interaction")
    with pytest.raises(ValueError, match="unsupported physical interaction"):
        transport._with_material_refinement(state, source, physical_failure,
            grid_numerics=WaveGridNumerics(specimen_phase_method="sampled"), cancelled=lambda: False, progress_callback=None)
    assert len(calls) == 2


def test_refinement_budget_checked_before_next_potential_allocation(monkeypatch):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    def unresolved(*args):
        raise WaveSamplingError("not resolved")
    with pytest.raises(ValueError, match="budget exceeded"):
        transport._with_material_refinement(state, source, unresolved,
            grid_numerics=WaveGridNumerics(maximum_pixels=64, specimen_phase_method="sampled"), cancelled=lambda: False, progress_callback=None)
    assert len(calls) == 1


def test_cancelled_refinement_does_not_build_another_potential(monkeypatch):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    stop = [False]
    def unresolved(*args):
        stop[0] = True
        raise WaveSamplingError("not resolved")
    with pytest.raises(InterruptedError):
        transport._with_material_refinement(state, source, unresolved,
            grid_numerics=WaveGridNumerics(specimen_phase_method="sampled"), cancelled=lambda: stop[0], progress_callback=None)
    assert len(calls) == 1


def test_inelastic_retry_does_not_reuse_slices_from_a_coarser_potential(monkeypatch, tmp_path):
    from temsim.physics.inelastic_wave import _propagate_inelastic_specimen
    from temsim.physics.wave_execution import InelasticWaveNumerics
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    operation = transport._slice_phase
    coarse_phases = [0]
    def fail_after_a_coarse_slice_was_saved(mode, potential, x, y, sigma, fraction):
        if len(x) == 64:
            coarse_phases[0] += 1
            if coarse_phases[0] == 3:
                raise WaveSamplingError("second slice requires a new potential lattice")
        return operation(mode, potential, x, y, sigma, fraction)
    monkeypatch.setattr(transport, "_slice_phase", fail_after_a_coarse_slice_was_saved)
    store = ExecutedWaveStore(tmp_path, "independent-material-retry-fixture", 1<<28)
    saved_keys = {64: set(), 128: set()}
    put = store.put
    def record_put(key, checkpoint):
        saved_keys[checkpoint.beam.modes[0].plane.amplitude.shape[0]].add(key)
        return put(key, checkpoint)
    monkeypatch.setattr(store, "put", record_put)
    kw = dict(numerics=InelasticWaveNumerics(trajectories_per_mode=2), store=store,
        maximum_step_mm=.5, grid_numerics=WaveGridNumerics(maximum_pixels=256, specimen_phase_method="sampled"), tip_time_s=0.,
        cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True)
    result = _propagate_inelastic_specimen(state, source, **kw)
    assert [c[0] for c in calls] == [64, 128]
    assert len(saved_keys[64]) == 1
    assert len(saved_keys[128]) == 4  # both full trajectories recomputed on fine material
    assert saved_keys[64].isdisjoint(saved_keys[128])
    assert result.record["material_grid_refinement"]["factor"] == 2
    assert len(result.beam.modes) == 2
    assert all(abs(row["probability_residual"]) < 1e-13 for row in result.record["modes"])
    assert not list(tmp_path.glob("pending-*"))
    # Cached final result must not re-prepare a potential or rerun collisions.
    again = _propagate_inelastic_specimen(state, source, **kw)
    assert again.digest == result.digest
    assert len(calls) == 2
    # Same seed/counters and same material arrays produce the same fine result
    # in a second execution even when the potential's cache-hit flag differs.
    coarse_phases[0] = 0
    repeated = _propagate_inelastic_specimen(state, source, **{**kw, "use_cache": False})
    for a, b in zip(result.beam.modes, repeated.beam.modes):
        np.testing.assert_array_equal(a.plane.amplitude, b.plane.amplitude)
        assert a.energy_kev == b.energy_kev
        assert a.scattering_history == b.scattering_history
