"""Bounded electric-energy handoff fixtures, not full microscope qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import core, particle_sections
from temsim.physics.particle_energy import sample_kinetic_energy, validate_kinetic_energy_array, energy_survival_mask
from temsim.specimen.elastic_transport import incident_rays_from_simulation
from test_shared_column_magnetic_transport import state_with_coil


class UniformElectricPotential:
    """Explicit mathematical field phi=10^6 z, SI metres and volts."""
    numerical_identity = "e" * 64
    z = np.array((0., .01))

    @property
    def base_field(self):
        return self

    def interpolate(self, positions):
        points = np.asarray(positions)
        electric = np.zeros_like(points)
        electric[:, 2] = -1e6
        return points[:, 2]*1e6, electric

    def potential_rise_v_at_global_positions(self, positions):
        return self.interpolate(positions)[0]


@pytest.mark.parametrize("through_archive", [False, True])
def test_energy_section_resume_uses_exact_checkpoint_energy_without_double_acceleration(monkeypatch, through_archive):
    state, _ = state_with_coil(step=.2)
    state.vacuum_map = SimpleNamespace(enabled=False)
    field = UniformElectricPotential()
    original_build = core.build_propagation_plan

    def build(*args, **kwargs):
        return replace(original_build(*args, **kwargs), electric_field=field,
            electric_field_identity=field.numerical_identity,
            electric_reference_invariant_ev=300000.)

    monkeypatch.setattr(particle_sections, "build_propagation_plan", build)
    for module, function in (("aperture_clipping", "clip_segment"),
                             ("recording_clipping", "clip_recording_planes"),
                             ("column_wall", "clip_column_wall")):
        monkeypatch.setattr(f"temsim.physics.{module}.{function}",
                            lambda state, z, x, y, alive, blocked, labels: (alive, blocked, labels))
    x, zero = np.array((1e-8, -2e-8)), np.zeros(2)
    initial = (x, zero, zero, zero, np.full(2, 1e-9))
    energy = np.array((123456., 234567.))
    weights = np.array((.4, .6))
    ancestry = (np.array((1, 3)), zero)
    survival = (np.ones(2, bool), np.full(2, np.nan), ["", ""])

    def trace(stop, previous=None):
        return particle_sections._trace_segment(state, "incident", 0., stop,
            initial, energy-300000., weights, ancestry, survival, (), previous,
            np.inf, initial_kicks=True, initial_kinetic_energy_ev=energy)

    first, _, _ = trace(4.)
    if through_archive:
        from temsim.particle_section_io import _pack, _unpack, _data_classes
        arrays = {}
        first = _unpack(_pack(first, arrays, {}, _data_classes()), arrays, _data_classes())
        assert first.plan.electric_field is None  # Runtime E is recaptured from the instrument.
        assert first.plan.electric_field_identity == field.numerical_identity
    resumed, where, reused = trace(8., first)
    whole, _, _ = trace(8.)
    assert reused and where == 4.
    np.testing.assert_allclose(first.checkpoints.kinetic_energy_ev[-1], energy+4000., atol=1e-9, rtol=0)
    np.testing.assert_allclose(resumed.checkpoints.kinetic_energy_ev[-1], energy+8000., atol=1e-9, rtol=0)
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_allclose(getattr(resumed.checkpoints, name)[-1],
            getattr(whole.checkpoints, name)[-1], rtol=3e-12, atol=2e-18)
    np.testing.assert_array_equal(resumed.branch.kinetic_energy_ev[-1],
                                  resumed.checkpoints.kinetic_energy_ev[-1])


def _incident():
    zeros = np.zeros((2, 2))
    return SimpleNamespace(z=np.array((0., 2.)), x=zeros, y=zeros, tx=zeros, ty=zeros,
        alive=np.ones(2, bool), blocked_z=np.full(2, np.nan),
        energy_offset_ev=np.array((999., 999.)), ray_weight=np.array((.4, .6)),
        kinetic_energy_ev=np.array(((100., 200.), (300., 400.))))


def test_material_entrance_reads_executed_energy_at_requested_plane():
    branch = _incident()
    state = SimpleNamespace(beam_voltage_kv=300., sample=SimpleNamespace(z_mm=2.))
    result = incident_rays_from_simulation(state, SimpleNamespace(incident=branch), boundary_z_mm=1.)
    np.testing.assert_array_equal([ray.kinetic_energy_ev for ray in result.rays], [200., 300.])
    branch.kinetic_energy_ev[-1, 0] = np.nan
    with pytest.raises(ValueError, match="kinetic energy|kinetic energies"):
        incident_rays_from_simulation(state, SimpleNamespace(incident=branch))


def test_readout_energy_keeps_unknown_and_exact_endpoints():
    branch = _incident()
    np.testing.assert_array_equal(sample_kinetic_energy(branch, 2.), [300., 400.])
    np.testing.assert_array_equal(sample_kinetic_energy(branch, 1.), [200., 300.])
    assert np.isnan(sample_kinetic_energy(branch, 3.)).all()
    branch.kinetic_energy_ev[0, 0] = np.nan
    assert np.isnan(sample_kinetic_energy(branch, 1.)[0])


def test_energy_validation_permits_stopped_unknown_without_admitting_alive_unknown():
    branch = _incident()
    branch.alive[0] = False
    branch.blocked_z[0] = 1.
    branch.kinetic_energy_ev[-1, 0] = np.nan
    kwargs = dict(allow_unknown=True, required=energy_survival_mask(branch, branch.z))
    assert validate_kinetic_energy_array(branch.kinetic_energy_ev, (2, 2), **kwargs)
    branch.kinetic_energy_ev[-1, 1] = np.nan
    with pytest.raises(ValueError, match="surviving particles"):
        validate_kinetic_energy_array(branch.kinetic_energy_ev, (2, 2), **kwargs)


def test_recording_with_electric_field_subtracts_reference_orbit_per_unique_scan_time(monkeypatch):
    from temsim.physics import record_plane
    from test_shared_recording_field_transport import OscillatingPair
    state, _ = state_with_coil()
    state.simulation_time_s = .125
    state.deflectors = [OscillatingPair("Fixture drive", "driven", 5., 15.,
        .3, 0., 0., 0., thickness_mm=2.)]
    planes = (SimpleNamespace(z_mm=6.), SimpleNamespace(z_mm=9.))
    times = np.array([[.125, .25, .125]])
    background = np.array((.003, -.007))
    transfers = tuple(SimpleNamespace(position_offset_m=(base+state.simulation_time_s**2, .002))
                      for base in background)
    calls = []

    def build(*args, **kwargs):
        return SimpleNamespace(electric_field=SimpleNamespace())

    def execute(working, *_args, **_kwargs):
        calls.append(working.simulation_time_s)
        return (SimpleNamespace(z_mm=np.array((6., 9.)),
            x_m=(background+working.simulation_time_s**2)[:, None],
            y_m=np.full((2, 1), .002)),)

    monkeypatch.setattr(core, "build_propagation_plan", build)
    monkeypatch.setattr(core, "execute_propagation_plan", execute)
    offsets = record_plane._scan_deflection_offsets(state, 0., planes, times, .2, transfers)
    assert calls == [.125, .25]
    for values in offsets:
        np.testing.assert_array_equal(values[0, [0, 2]], 0.)
        np.testing.assert_allclose(values[0, 1], [.25**2-.125**2, 0.], atol=1e-17, rtol=0)
