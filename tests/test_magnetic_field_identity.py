"""Captured identities and finite-coil field bounds, not OEM validation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from temsim.magnetic_field_scene import MagneticSceneField, prepare_magnetic_scene
from temsim.optics.condenser_lens import AxialFieldTerm
from temsim.optics.model import DeflectorPair, Stigmator
from temsim.optics.round_lens import RoundLensComponent
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FieldMapProvenance, FrozenMappedField, GeometryAwareAnalyticFieldProvider,
    MagneticFieldMap, MappedLensFieldProvider, lens_geometry_binding,
)


def bare_state(**kwargs):
    values = dict(lenses=(), stigmators=(), corrector_elements=(), deflectors=(),
                  electron_gun=None, simulation_mode="custom", beam_voltage_kv=300.,
                  simulation_time_s=.125, lens_field_map_descriptors={})
    values.update(kwargs)
    return SimpleNamespace(**values)


def round_lens():
    return RoundLensComponent(name="Round lens", key="round", z_mm=25., b0_t=.2,
        a_mm=2., percent=30., max_percent=100., colour="blue",
        gaussian=[AxialFieldTerm(1., 0., 1.)], enabled=True, cs_mm=None, cc_mm=None,
        polarity=1, normalise_profile_peak=False, mechanical_center_from_tip_mm=25.,
        mechanical_length_mm=50., mechanical_outer_diameter_mm=15., bore_diameter_mm=3.,
        pole_gap_mm=2., optical_reference_from_tip_mm=25.)


def analytic_scene(monkeypatch, state):
    def provider(_state, lens):
        return GeometryAwareAnalyticFieldProvider(lens.key, lens,
            lens_geometry_binding(state, lens.key, lens), "Analytic test fixture")
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", provider)
    return prepare_magnetic_scene(state)


def test_analytic_identity_is_reproducible_and_ignores_presentation_and_electron_inputs(monkeypatch):
    lens = round_lens()
    state = bare_state(lenses=(lens,))
    first = analytic_scene(monkeypatch, state)
    assert first.physical_identity and first.numerical_identity
    assert first.support_metadata[0].model == "near_axis_first_order"
    lens.name, lens.colour = "Renamed", "red"
    state.projection_angle_deg = 72.
    state.virtual_electron_initial_energy_ev = 12.
    state.virtual_electron_polar_angle_mrad = 1.5
    again = analytic_scene(monkeypatch, deepcopy(state))
    assert (again.physical_identity, again.numerical_identity) == (first.physical_identity, first.numerical_identity)
    assert "higher radial orders" in first.support_metadata[0].limitation
    with pytest.raises(FrozenInstanceError):
        first.support_metadata[0].model = "changed"


@pytest.mark.parametrize("attribute,value", [("percent", 31.), ("b0_t", .21), ("polarity", -1),
                                              ("a_mm", 2.1), ("z_mm", 25.1), ("bore_diameter_mm", 2.9)])
def test_analytic_consumed_geometry_and_excitation_change_identities(monkeypatch, attribute, value):
    lens = round_lens()
    state = bare_state(lenses=(lens,))
    before = analytic_scene(monkeypatch, state)
    setattr(lens, attribute, value)
    after = analytic_scene(monkeypatch, state)
    assert after.physical_identity != before.physical_identity
    assert after.numerical_identity != before.numerical_identity


def test_supported_numeric_domain_change_preserves_physical_identity(monkeypatch):
    scene = analytic_scene(monkeypatch, bare_state(lenses=(round_lens(),)))
    bounds = scene.diagnostic_bounds_m.copy()
    bounds[1, 2] += .01
    extended = scene.with_diagnostic_bounds(bounds)
    assert extended.physical_identity == scene.physical_identity
    assert extended.numerical_identity != scene.numerical_identity
    assert not extended.diagnostic_bounds_m.flags.writeable
    old = MagneticSceneField(scene.bounds_m, scene.transverse_radius_m, scene.source_keys,
                             scene.notes, scene.seed_regions_m, scene._sources)
    assert old.physical_identity is None and old.numerical_identity is None
    assert old.with_diagnostic_bounds(bounds).numerical_identity is None


def test_unknown_callback_does_not_get_a_total_field_identity(monkeypatch):
    lens = round_lens()
    # Instance-specific callbacks also have no admitted, reproducible law.
    original = lens.magnetic_field_t
    lens.magnetic_field_t = lambda z: original(z) * 1.01
    scene = analytic_scene(monkeypatch, bare_state(lenses=(lens,),
        stigmators=(Stigmator("Stigmator", "s", 25., strength_x_percent=5.),)))
    assert scene.physical_identity is None and scene.numerical_identity is None
    assert scene.support_metadata[0].model == "unknown_provider"
    assert scene.support_metadata[1].physical_identity is not None
    assert "unknown" in " ".join(scene.notes)


def mapped_scene(monkeypatch, *, field=.4, scale=30., origin=(0., 0., 0.), spacing=1.,
                 rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), mutate_provider=None):
    lens = round_lens()
    lens.percent = scale
    state = bare_state(lenses=(lens,))
    binding = lens_geometry_binding(state, lens.key, lens)
    axes = (np.array((0., .001, .002)), np.array((.02, .025, .03)))
    if spacing != 1.:
        axes = (axes[0], np.array((.02, .02 + .005 * spacing, .03)))
    field_map = MagneticFieldMap("axisymmetric_rz", axes,
        (np.zeros((3, 3)), np.full((3, 3), field)), CoordinateRegistration(origin, rotation),
        binding.geometry_fingerprint, 100., 1,
        FieldMapProvenance("fem", "fixture", "0" * 64, "Synthetic reference only"))
    provider = MappedLensFieldProvider(lens.key, field_map, lens, binding)
    if mutate_provider is not None:
        mutate_provider(provider)
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda *_: provider)
    return prepare_magnetic_scene(state)


def test_mapped_identity_binds_actual_arrays_registration_grid_and_frozen_scale(monkeypatch):
    first = mapped_scene(monkeypatch)
    again = mapped_scene(monkeypatch)
    assert first.physical_identity == again.physical_identity
    assert first.numerical_identity == again.numerical_identity
    for kwargs in ({"field": .41}, {"origin": (.0001, 0., 0.)}, {"spacing": .8}):
        changed = mapped_scene(monkeypatch, **kwargs)
        # Identical claimed geometry/source SHA does not hide different numerical data.
        assert changed.numerical_identity != first.numerical_identity
    assert mapped_scene(monkeypatch, scale=31.).physical_identity != first.physical_identity
    assert first.support_metadata[0].model == "registered_mapped_field"
    assert "unknown" in first.support_metadata[0].limitation


@pytest.mark.parametrize("registration", [
    {"origin": (.0001, 0., 0.)},
    {"rotation": ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))},
])
def test_mapped_registration_changes_physical_identity(monkeypatch, registration):
    baseline = mapped_scene(monkeypatch)
    changed = mapped_scene(monkeypatch, **registration)
    assert baseline.physical_identity and changed.physical_identity
    assert baseline.physical_identity != changed.physical_identity
    assert baseline.numerical_identity != changed.numerical_identity


@pytest.mark.parametrize("cls,method", [
    (MappedLensFieldProvider, "excitation_scale"),
    (FrozenMappedField, "field_at_global_positions_t"),
    (MagneticFieldMap, "field_at_global_positions_t"),
    (CoordinateRegistration, "positions_global_to_local_m"),
    (CoordinateRegistration, "vectors_local_to_global"),
    (RegularGridInterpolator, "__call__"),
])
def test_mapped_replaced_methods_do_not_get_known_identity(monkeypatch, cls, method):
    original = getattr(cls, method)
    monkeypatch.setattr(cls, method, lambda self, *args, **kwargs: original(self, *args, **kwargs))
    # Even a delegating replacement is an unrecognised law, not the admitted one.
    scene = mapped_scene(monkeypatch)
    assert scene.physical_identity is None and scene.numerical_identity is None
    assert scene.support_metadata[0].model == "unknown_provider"


def test_deepcopied_writable_map_has_no_frozen_identity(monkeypatch):
    def copy_arrays(provider):
        object.__setattr__(provider, "field_map", deepcopy(provider.field_map))
        assert provider.field_map.components_t[0].flags.writeable

    scene = mapped_scene(monkeypatch, mutate_provider=copy_arrays)
    assert scene.physical_identity is None and scene.numerical_identity is None
    # This limits the identity claim only: an unknown provider remains queryable.
    np.testing.assert_allclose(scene.field_at_global_positions_t([[0., 0., .025]]), [[0., 0., .12]])


def _interpolator_storage_name(interpolator, public_name):
    return "_" + public_name if "_" + public_name in vars(interpolator) else public_name


@pytest.mark.parametrize("mutation", ["writable_component", "writable_axis", "different_values",
                                       "different_grid", "nearest", "extrapolation", "callback"])
def test_mapped_identity_requires_the_actual_readonly_linear_interpolator(monkeypatch, mutation):
    def alter(provider):
        field_map = provider.field_map
        interpolator = field_map._interpolators[0]
        values_name = _interpolator_storage_name(interpolator, "values")
        grid_name = _interpolator_storage_name(interpolator, "grid")
        if mutation in ("writable_component", "different_values"):
            values = field_map.components_t[0].copy()
            if mutation == "different_values":
                values += .001
                values.setflags(write=False)
            else:
                object.__setattr__(field_map, "components_t", (values, *field_map.components_t[1:]))
            setattr(interpolator, values_name, values)
        elif mutation in ("writable_axis", "different_grid"):
            axis = field_map.axes_m[0].copy()
            if mutation == "different_grid":
                axis[1] *= 1.1
                axis.setflags(write=False)
            else:
                object.__setattr__(field_map, "axes_m", (axis, *field_map.axes_m[1:]))
            setattr(interpolator, grid_name, (axis, *getattr(interpolator, grid_name)[1:]))
        elif mutation == "nearest":
            interpolator.method = "nearest"
        elif mutation == "extrapolation":
            interpolator.fill_value = None
        else:
            original = interpolator._evaluate_linear
            interpolator._evaluate_linear = lambda *args: original(*args) * 1.01

    scene = mapped_scene(monkeypatch, mutate_provider=alter)
    assert scene.physical_identity is None and scene.numerical_identity is None
    assert scene.support_metadata[0].model == "unknown_provider"


@pytest.mark.parametrize("attribute,value", [("strength_x_percent", 15.), ("channel_y_angle_deg", 40.),
                                              ("length_mm", 9.), ("z_mm", 26.)])
def test_multipole_identity_binds_effective_tensor_inputs_and_reference_momentum(attribute, value):
    component = Stigmator("Stigmator", "s", 25., strength_x_percent=10.)
    state = bare_state(stigmators=(component,))
    before = prepare_magnetic_scene(state)
    setattr(component, attribute, value)
    after = prepare_magnetic_scene(state)
    assert after.physical_identity != before.physical_identity
    state.beam_voltage_kv = 200.
    assert prepare_magnetic_scene(state).physical_identity != after.physical_identity
    info = before.support_metadata[0]
    assert info.reference_momentum_kg_m_s > 0. and info.reference_charge_c < 0.
    assert info.captured_time_s is None


def test_dynamic_deflector_identity_includes_captured_time_and_reference_energy():
    from temsim.optics.ac_deflector import AC_DEFLECTOR_DEFINITION
    component = AC_DEFLECTOR_DEFINITION.create_component()
    component.scan_enabled = True
    state = bare_state(deflectors=(component,))
    first = prepare_magnetic_scene(state)
    assert first.support_metadata[0].captured_time_s == state.simulation_time_s
    state.simulation_time_s += .25
    assert prepare_magnetic_scene(state).physical_identity != first.physical_identity
    state.simulation_time_s -= .25
    state.beam_voltage_kv = 200.
    assert prepare_magnetic_scene(state).physical_identity != first.physical_identity
    state.beam_voltage_kv = 300.
    component.effective_thickness_mm *= 1.1
    assert prepare_magnetic_scene(state).physical_identity != first.physical_identity


@pytest.mark.parametrize("dx,dy", [(1e-3, -2e-3), (-1e-3, 2e-3)])
def test_finite_deflector_signed_integral_and_angle_at_exit_reference_plane(dx, dy):
    from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
    from temsim.physics.core import electron

    # Two separated configured coils; compare the first coil's exit Z only.
    # The captured finite field is shared with main transport. This fixture
    # checks its signed paraxial integral, not a measured fringe-field shape.
    component = DeflectorPair("Deflector", "d", 25., 45., upper_x_mrad=dx * 1e3,
                               upper_y_mrad=dy * 1e3, thickness_mm=4.)
    state = bare_state(deflectors=(component,))
    scene = prepare_magnetic_scene(state)
    info = scene.support_metadata[0]
    lower, upper = np.asarray(info.bounds_m)[:, 2]
    z = np.linspace(lower, upper, 129)
    points = np.column_stack((np.zeros((len(z), 2)), z))
    field_integral = np.trapezoid(scene.field_at_global_positions_t(points), z, axis=0)
    charge, momentum, _ = electron(state)
    np.testing.assert_allclose((-charge * field_integral[1] / momentum,
                                charge * field_integral[0] / momentum), (dx, dy), rtol=2e-14)
    settings = TestElectronSettings(kinetic_energy_ev=300_000.,
        position_m=(0., 0., lower), max_path_length_m=(upper - lower) * 1.05,
        step_m=2e-5, relative_tolerance=1e-8, position_tolerance_m=1e-13)
    result = trace_test_electron(scene, settings, use_compiled=False)
    assert result.positions_m[-1, 2] > upper
    direction = np.array([np.interp(upper, result.positions_m[:, 2], result.directions[:, i]) for i in range(3)])
    # Small-angle equivalence is only claimed here for |theta| < 3 mrad,
    # fixed 300 keV, axial incidence and no other active field.
    np.testing.assert_allclose(direction[:2] / direction[2], (dx, dy), rtol=2e-5, atol=2e-9)
    assert info.model == "finite_coil_dipole"
    assert "fringe shape is unmodelled" in info.limitation


def test_empty_scene_is_known_zero_but_keeps_unsupported_filter_explicit():
    scene = prepare_magnetic_scene(bare_state())
    assert scene.physical_identity and scene.numerical_identity
    assert scene.support_metadata == ()
    assert any("bent coordinate" in note for note in scene.notes)
