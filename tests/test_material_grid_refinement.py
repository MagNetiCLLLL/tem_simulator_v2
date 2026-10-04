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
        return imaging.PreparedSpecimen(x_angstrom=axis, y_angstrom=axis,
            potential_configurations_v_angstrom=(potential,), slice_thicknesses_angstrom=np.array([2., 2.]),
            mean_projected_potential_v_angstrom=potential.sum(axis=0),
            metrics={"fixture": "constant phase", "prepared_specimen_cache_hit": len(calls) > 1})
    monkeypatch.setattr(imaging, "prepare_specimen_potentials", prepare)
    return calls


def state_and_wave():
    state = quiet_state()
    from specimen_inputs import imported_sample
    imported_sample(state.sample, zone=(0, 0, 1), in_plane=(1, 0, 0))
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 64
    state.sample.wave_field_of_view_angstrom = 30.
    # Imported CIFs no longer invent material MFPs. This declared weak
    # absorption keeps the lossy retry fixture nontrivial and its independent
    # inelastic histories distinct from the exact identity-channel shortcut.
    state.sample.real_inelastic_enabled = True
    state.sample.real_absorption_mean_free_path_nm = 1000.
    return state, entrance(state)


@pytest.mark.parametrize("minimum,factor,expected", ((6616, 2, 6720), (136, 2, 140), (65, 1, 66), (128, 2, 128)))
def test_material_fft_rounding_only_adds_even_samples(minimum, factor, expected):
    # Production-size planning is scalar only; no large potential is built.
    pixels = transport._material_fft_pixels(minimum, factor)
    assert pixels == expected
    assert pixels >= minimum and (pixels//factor) % 2 == 0
    assert pixels % factor == 0


def test_fft_rounding_keeps_full_material_domain_origin_and_finer_sampling(monkeypatch):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    state.sample.wave_grid_pixels = 68
    before = vars(state.sample).copy()
    centre, fov = transport._covering_domain(source.beam)
    numerics = WaveGridNumerics(maximum_pixels=256)
    _, prepared, x, y, _ = transport._prepare_material_grid(state, source, grid_numerics=numerics)
    record = prepared.metrics["coherent_material_grid"]
    assert record["requested_potential_pixels"] == 136
    assert record["executed_potential_pixels"] > 136
    assert record["executed_sampling_m"] <= record["requested_sampling_m"]
    assert record["field_of_view_m"] == fov
    assert calls[0][1] == pytest.approx(fov*1e10)
    wave_axes = transport._material_wave_axes(x, y, numerics)
    for axis, coarse, origin in zip((x, y), wave_axes, centre, strict=True):
        assert axis[len(axis)//2] == origin == coarse[len(coarse)//2]
        assert len(axis)*(axis[1]-axis[0]) == pytest.approx(fov)
        assert len(coarse)*(coarse[1]-coarse[0]) == pytest.approx(fov)
    xx, yy = source.beam.modes[0].plane.coordinates_m()
    assert xx.min() >= x[0]-(x[1]-x[0])/2
    assert xx.max() <= x[-1]+(x[1]-x[0])/2
    assert yy.min() >= y[0]-(y[1]-y[0])/2
    assert yy.max() <= y[-1]+(y[1]-y[0])/2
    assert vars(state.sample) == before


@pytest.mark.parametrize("limit", ("pixels", "bytes"))
def test_fft_rounding_is_budgeted_before_any_potential_allocation(monkeypatch, limit):
    calls = install_potential_fixture(monkeypatch)
    state, source = state_and_wave()
    state.sample.wave_grid_pixels = 68
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    from temsim.physics.wave_grid import WaveGridBudgetError
    numerics = (WaveGridNumerics(maximum_pixels=136) if limit == "pixels" else
        WaveGridNumerics(maximum_pixels=256,
            maximum_working_bytes=resident_wave_bytes(source.beam)+256*136**2))
    # The requested lattice fits exactly; the larger execution lattice does
    # not, so neither atom generation nor potential construction may start.
    numerics.check((136, 136), retained_bytes=resident_wave_bytes(source.beam))
    with pytest.raises(WaveGridBudgetError, match="budget exceeded"):
        transport._prepare_material_grid(state, source, grid_numerics=numerics)
    assert not calls


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


def test_real_cif_material_budget_changes_no_complex_field_or_source():
    """Actual small Si IAM + coherent transport; no potential/solver mock."""
    from copy import deepcopy
    from dataclasses import asdict
    from specimen_inputs import imported_sample
    from temsim.specimen.atomistic import atomistic_capability
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    state, source = state_and_wave()
    imported_sample(state.sample, zone=(0, 0, 1), in_plane=(1, 0, 0))
    state.sample.envelope_shape = "rectangle"
    state.sample.size_x_nm = state.sample.size_y_nm = 2.
    state.sample.wave_atomistic_enabled = state.sample.wave_multislice_enabled = True
    original = deepcopy(asdict(state.sample))
    original_source = source.digest
    base = WaveGridNumerics(maximum_pixels=512, maximum_working_bytes=64*1024**2)
    small = transport._propagate_specimen(state, source, grid_numerics=base)
    larger = transport._propagate_specimen(state, source,
        grid_numerics=replace(base, maximum_working_bytes=96*1024**2))
    assert small.record["potential"]["atomistic_applied"]
    assert small.record["potential"]["atom_count"] > 0
    assert small.plane_z_mm == larger.plane_z_mm
    assert small.reference_current_a == larger.reference_current_a == source.reference_current_a
    assert 0 < small.beam.total_weight <= source.beam.total_weight
    for a, b in zip(small.beam.modes, larger.beam.modes, strict=True):
        assert a.mode_id == b.mode_id
        assert a.energy_kev == b.energy_kev
        assert a.axial_reference == b.axial_reference
        assert a.weight_per_reference_electron == b.weight_per_reference_electron
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(a.plane, name), getattr(b.plane, name))
        assert np.max(abs(a.plane.amplitude.imag)) > 0
    assert source.digest == original_source
    assert asdict(state.sample) == original
