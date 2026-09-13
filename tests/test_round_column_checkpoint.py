"""Representation/clock bookkeeping fixtures, not source qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics
from temsim.physics.radial_column_wave import RadialWave
from temsim.physics.round_column_checkpoint import execute_round_prefix
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference


def fixture():
    plane = PlaneWave(np.full((32, 32), 1/32+0j), np.eye(2)*1e-9, np.zeros(2))
    mode = WaveMode(plane, .4, TIP_REFERENCE, "tip_energy_0", 300.,
                   AxialWaveReference(1e-9, 1e-25), ({"event": "existing_history"},))
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 450., 100e-9,
                                  {"schema": "explicit_test_fixture_not_physical_source"})
    radius = np.geomspace(1e-20, 5e-6, 65536, endpoint=False)
    wave = RadialWave(radius, .6e7*np.exp(-(radius*1e7)**2/2+1j*.7)/np.sqrt(np.pi), .25)
    return checkpoint, wave


def test_prefix_advances_executed_clock_and_keeps_absolute_current_and_phase(monkeypatch):
    import temsim.physics.round_column_checkpoint as module
    checkpoint, radial = fixture()
    monkeypatch.setattr(module, "round_column_prefix", lambda *a, **kw:
        (500., ((checkpoint.beam.modes[0], radial),), {"executed_steps": 1}))
    result = execute_round_prefix(object(), checkpoint, 510.,
        numerics=RadialColumnNumerics(initial_cartesian_pixels=32))
    mode = result.beam.modes[0]
    assert result.plane_z_mm == 500.
    assert result.reference_current_a == checkpoint.reference_current_a
    assert mode.mode_id == checkpoint.beam.modes[0].mode_id
    assert mode.scattering_history == checkpoint.beam.modes[0].scattering_history
    assert mode.weight_per_reference_electron == pytest.approx(.4*.36, rel=1e-8)
    assert result.transmitted_current_a == pytest.approx(14.4e-9, rel=1e-8)
    gamma = 1+300000*e/(m_e*c*c)
    velocity = c*np.sqrt(1-1/gamma**2)
    momentum = gamma*m_e*velocity
    assert mode.axial_reference.flight_time_s == pytest.approx(1e-9+.05/velocity)
    assert mode.axial_reference.longitudinal_action_j_s == pytest.approx(1e-25+.05*momentum)
    x, y = mode.plane.coordinates_m()
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    wavelength = wavelength_m(300000.)
    expected = np.sqrt(.4)*.6e7*np.exp(-(x*x+y*y)*.5e14+1j*.7+
        1j*np.pi/wavelength*.25*(x*x+y*y))/np.sqrt(np.pi)
    assert np.linalg.norm(mode.weighted_density_amplitude()-expected)*mode.plane.basis_m[0, 0] < 1e-7
    assert result.record["upstream_digest"] == checkpoint.digest


def test_no_round_prefix_keeps_same_executed_plane_and_clock(monkeypatch):
    import temsim.physics.round_column_checkpoint as module
    checkpoint, _ = fixture()
    monkeypatch.setattr(module, "round_column_prefix", lambda *a, **kw: (450., (), {}))
    assert execute_round_prefix(object(), checkpoint, 510.) is checkpoint


@pytest.mark.parametrize("case", ["empty_advance", "wrong_plane", "wrong_id"])
def test_inconsistent_stage_cannot_publish(monkeypatch, case):
    import temsim.physics.round_column_checkpoint as module
    checkpoint, radial = fixture()
    mode = checkpoint.beam.modes[0]
    output = (500., ((mode, radial),), {})
    if case == "empty_advance": output = (500., (), {})
    if case == "wrong_plane": output = (600., ((mode, radial),), {})
    if case == "wrong_id": output = (500., ((replace(mode, mode_id="other"), radial),), {})
    monkeypatch.setattr(module, "round_column_prefix", lambda *a, **kw: output)
    with pytest.raises(ValueError): execute_round_prefix(object(), checkpoint, 510.)


def test_tip_request_validates_radial_numerics():
    from temsim.physics.tip_wave_pipeline import TipWaveRequest
    with pytest.raises(ValueError, match="radial samples"):
        TipWaveRequest(radial_column=RadialColumnNumerics(maximum_samples=32)).validate()


def test_extinguished_wave_traverses_real_round_plan_without_dividing_by_zero():
    from temsim.optics.column import default_state
    state = default_state()
    original, _ = fixture()
    mode = original.beam.modes[0]
    mode = replace(mode, weight_per_reference_electron=0.,
                   plane=replace(mode.plane, amplitude=np.zeros_like(mode.plane.amplitude)))
    payload = {"mode_id": mode.mode_id, "width_nm": 100., "curvature_m1": 0.,
               "coefficients_real": [0.]*8, "coefficients_imag": [0.]*8}
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), state.electron_gun.exit_plane_z_mm,
        original.reference_current_a, {"scope": "ZERO FIELD FIXTURE", "radial_output_modes": (payload,)})
    result = execute_round_prefix(state, checkpoint, checkpoint.plane_z_mm+1.,
        numerics=RadialColumnNumerics(initial_samples=1024, initial_cartesian_pixels=32))
    assert result.plane_z_mm == checkpoint.plane_z_mm+1.
    assert result.transmitted_current_a == 0.
    assert not np.any(result.beam.modes[0].plane.amplitude)
    assert result.beam.modes[0].axial_reference.flight_time_s > mode.axial_reference.flight_time_s
    assert result.record["round_execution"]["steps"] > 0


@pytest.mark.parametrize("segmented", [False, True])
def test_pipeline_routes_and_caches_round_prefix_without_repeating_the_gun(monkeypatch, tmp_path, segmented):
    """Explicit routing fixture; the returned field is not a physical source."""
    from temsim.optics.column import default_state
    from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
    from temsim.physics.wave_execution import WaveExecutionOptions
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.round_column_checkpoint as adapter
    state = default_state()
    checkpoint, _ = fixture()
    checkpoint = replace(checkpoint, plane_z_mm=state.electron_gun.exit_plane_z_mm,
        record={**checkpoint.record, "radial_output_modes": ("test_route_only",)})
    calls = {"gun": 0, "radial": 0, "cartesian": 0}
    def gun(*a, **kw):
        calls["gun"] += 1
        return checkpoint
    def radial(state, incoming, stop, **kw):
        calls["radial"] += 1
        # A disk-backed checkpoint has its own verified storage identity.
        # Its physical payload must remain identical, not its wrapper digest.
        np.testing.assert_array_equal(incoming.beam.modes[0].plane.amplitude,
                                      checkpoint.beam.modes[0].plane.amplitude)
        assert incoming.beam.modes[0].axial_reference == checkpoint.beam.modes[0].axial_reference
        return replace(incoming, plane_z_mm=incoming.plane_z_mm+.5,
            record={"schema": "test_radial_route", "upstream_digest": incoming.digest,
                    "numerics": kw["numerics"].complex_tolerance})
    def cartesian(state, incoming, stop, **kw):
        calls["cartesian"] += 1
        assert incoming.plane_z_mm == checkpoint.plane_z_mm+.5
        assert incoming.record["schema"] == "test_radial_route"
        return replace(incoming, plane_z_mm=stop)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", gun)
    monkeypatch.setattr(adapter, "execute_round_prefix", radial)
    monkeypatch.setattr(pipeline, "_propagate_column", cartesian)
    monkeypatch.setattr(pipeline, "_propagate_column_segmented", lambda *a, **kw: (cartesian(*a, **kw), False))
    request = TipWaveRequest(stop="specimen_entrance", execution=WaveExecutionOptions(
        segmented=segmented, cache_directory=str(tmp_path/"executed")))
    pipeline._STAGES.clear()
    first = simulate_tip_wave(state, request)
    again = simulate_tip_wave(state, request)
    assert first.checkpoint.digest == again.checkpoint.digest
    assert calls == {"gun": 1, "radial": 1, "cartesian": 2}
    changed = replace(request, radial_column=replace(request.radial_column, complex_tolerance=5e-7))
    simulate_tip_wave(state, changed)
    assert calls == {"gun": 1, "radial": 2, "cartesian": 3}
