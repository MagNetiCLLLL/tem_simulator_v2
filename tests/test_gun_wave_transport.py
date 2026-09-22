"""Canonical operator mathematics and explicit rejection of retired exit sources.

No test bypasses source admission or executes an independently defined exit wave.
"""
import numpy as np
import pytest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.effective_source import EffectiveGunSource
from test_effective_gun_source import historical_parameters as bind_effective_source
from temsim.physics.canonical_phase import validate_canonical_map
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

@pytest.mark.parametrize("entry", ["build", "manifest", "cache", "restore", "particle"])
def test_retired_exit_source_is_rejected_before_transport_or_cache_publication(entry, tmp_path, monkeypatch):
    from temsim.artifact_store import ArtifactStore
    from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel
    from temsim.physics import gun_wave_transport, gun_wave_cache, simulation
    state = _state()
    before = capture_instrument_snapshot(state)
    forbidden = lambda *a, **kw: pytest.fail("Retired exit source must not execute transport")
    monkeypatch.setattr(gun_wave_transport, "_execute_gun_wave_plan", forbidden)
    monkeypatch.setattr(gun_wave_cache, "_execute_gun_wave_plan", forbidden)
    store = ArtifactStore(tmp_path, quota_bytes=64*1024**2)
    with pytest.raises(UnsupportedSourceModel, match="exit sources are not permitted"):
        if entry == "build":
            build_gun_wave_checkpoint(state)
        elif entry == "manifest":
            gun_wave_cache.gun_wave_manifest(state)
        elif entry == "cache":
            gun_wave_cache.cached_gun_wave_checkpoint(state, store)
        elif entry == "restore":
            before.restore()
        else:
            simulation.run(state)
    assert capture_instrument_snapshot(state).digest == before.digest
    assert not tuple(tmp_path.rglob("*.npz"))
