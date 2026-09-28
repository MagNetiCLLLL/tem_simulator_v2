"""Identity integration and cache reuse with tiny non-solved field fixtures."""
from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.diagnostic_field_identity import electric_field_identity, identity_digest
from temsim.magnetic_field_scene import MagneticSceneField
from temsim.physics.closed_gun_field import ClosedGunField, closed_field_request
from temsim.physics.planar_gun_field import request_digest
from temsim.test_electron_scene import prepare_test_electron_scene, _prepare_electric_provider


@pytest.fixture
def gun():
    from temsim.optics.column import default_state
    return default_state().electron_gun


def tiny_field(request, *, offset=0.):
    domain = request["domain"]
    r = np.linspace(0., domain["outer_radius_m"], 3)
    z = np.linspace(domain["entrance_m"], domain["exit_m"], 4)
    return ClosedGunField(request, r, z, np.broadcast_to(z*1e5 + offset, (3, 4)),
                          {"request_sha256": request_digest(request)})


def empty_magnetic():
    bounds = np.array(((-.01, -.01, 0.), (.01, .01, 1.)))
    bounds.setflags(write=False)
    physical = identity_digest("fixture-no-magnets", {})
    return MagneticSceneField(bounds, .01, (), (), (), (), bounds,
                              physical_identity=physical).with_diagnostic_bounds(bounds)


def scene_from(monkeypatch, field, *, magnetic=None, apertures=(), energy=.3, stop_mm=200.):
    captured_gun = SimpleNamespace(
        emitter=SimpleNamespace(surface_model=None, emission_energy_ev=energy,
                                mechanical_center_from_tip_mm=-.5),
        bore_components=(), dpa_aperture=None, c1_aperture=None)
    state = SimpleNamespace(apertures=apertures, energy_filter_installed=False)
    monkeypatch.setattr("temsim.test_electron_scene._prepare_electric_provider",
                        lambda state, stop: (captured_gun, field, ()))
    return prepare_test_electron_scene(state, magnetic or empty_magnetic(),
                                       z_limits_mm=(0., stop_mm))


def test_known_electric_and_magnetic_scene_exposes_all_three_layers(monkeypatch, gun):
    field = tiny_field(closed_field_request(gun))
    scene = scene_from(monkeypatch, field)
    assert scene.physical_identity and scene.numerical_identity and scene.transport_identity
    assert len({scene.physical_identity, scene.numerical_identity, scene.transport_identity}) == 3
    assert any(scene.physical_identity in note for note in scene.notes)
    assert any(scene.numerical_identity in note for note in scene.notes)
    assert any("Requested diagnostic endpoint 200 mm" in note for note in scene.notes)
    assert scene.electric_base is field


def test_initial_energy_and_display_notes_do_not_change_field_or_transport_identity(monkeypatch, gun):
    field = tiny_field(closed_field_request(gun))
    magnetic = empty_magnetic()
    first = scene_from(monkeypatch, field, magnetic=magnetic, energy=.3)
    second = scene_from(monkeypatch, field, magnetic=replace(magnetic, notes=("Different display note",)), energy=.7)
    assert second.initial_energy_ev != first.initial_energy_ev
    assert second.physical_identity == first.physical_identity
    assert second.numerical_identity == first.numerical_identity
    assert second.transport_identity == first.transport_identity


def test_actual_field_arrays_and_requested_support_are_numerical_inputs(monkeypatch, gun):
    request = closed_field_request(gun)
    first = scene_from(monkeypatch, tiny_field(request))
    changed = scene_from(monkeypatch, tiny_field(request, offset=.01))
    cropped = scene_from(monkeypatch, first.electric_base, stop_mm=100.)
    assert first.physical_identity == changed.physical_identity == cropped.physical_identity
    assert first.numerical_identity != changed.numerical_identity
    assert first.numerical_identity != cropped.numerical_identity
    assert first.transport_identity != cropped.transport_identity


def test_hardware_stop_change_invalidates_transport_without_changing_fields(monkeypatch, gun):
    from temsim.optics.model import Aperture
    aperture = Aperture("Test aperture", "test_ap", 100., .1, enabled=True)
    field = tiny_field(closed_field_request(gun))
    first = scene_from(monkeypatch, field, apertures=(aperture,))
    aperture.radius_mm = .2
    second = scene_from(monkeypatch, field, apertures=(aperture,))
    assert first.numerical_identity == second.numerical_identity
    assert first.transport_identity and first.transport_identity != second.transport_identity
    assert first._apertures[0].radius_mm == .1  # Captured hardware remains detached.


def test_unknown_aperture_implementation_does_not_get_transport_identity(monkeypatch, gun):
    aperture = SimpleNamespace(key="custom", enabled=True, z_mm=100., radius_mm=.1,
                               offset_x_mm=0., offset_y_mm=0.)
    scene = scene_from(monkeypatch, tiny_field(closed_field_request(gun)), apertures=(aperture,))
    assert scene.physical_identity and scene.numerical_identity
    assert scene.transport_identity is None
    assert scene.diagnostic_segment_stop((.001, 0., 0.), (.001, 0., .2)) == (.5, "aperture:custom")


def test_unknown_magnetic_or_custom_electric_identity_does_not_prevent_existing_execution(monkeypatch, gun):
    field = tiny_field(closed_field_request(gun))
    unknown_magnetic = replace(empty_magnetic(), physical_identity=None, numerical_identity=None)
    scene = scene_from(monkeypatch, field, magnetic=unknown_magnetic)
    assert scene.physical_identity is scene.numerical_identity is scene.transport_identity is None
    assert scene.diagnostic_fields_at_global_position((0., 0., .001)) is not None
    field.interpolate = lambda points: (np.zeros(np.asarray(points).shape[:-1]), np.zeros_like(points))
    scene = scene_from(monkeypatch, field)
    assert scene.physical_identity is scene.numerical_identity is scene.transport_identity is None
    assert any("custom field method" in note for note in scene.notes)
    assert scene.diagnostic_fields_at_global_position((0., 0., .001)) is not None


def test_unidentified_wien_composition_does_not_inherit_scalar_base_identity(monkeypatch, gun):
    from temsim.optics.electron_gun.monochromator import CombinedElectricField

    class Wien:
        def field_at_global_positions_v_per_m(self, points):
            return np.zeros_like(points)

        def potential_v_at_global_positions(self, points):
            return np.zeros(np.asarray(points).shape[:-1])

    combined = CombinedElectricField(tiny_field(closed_field_request(gun)), Wien())
    scene = scene_from(monkeypatch, combined)
    assert scene.physical_identity is scene.numerical_identity is scene.transport_identity is None
    assert scene.diagnostic_fields_at_global_position((0., 0., .001)) is not None


def test_larger_field_reused_through_ordinary_accessor_without_relabelling_or_resolving(monkeypatch, gun):
    original_extension = gun._gun_field_exit_extension_mm if hasattr(gun, "_gun_field_exit_extension_mm") else 100.
    request = closed_field_request(gun, exit_extension_mm=200.)
    field = tiny_field(request)
    gun._closed_gun_field = field
    original_extension = gun._gun_field_exit_extension_mm

    def no_solve(*args, **kwargs):
        raise AssertionError("The covering captured field must avoid a new field solve")

    monkeypatch.setattr("temsim.physics.closed_gun_field._build_request_field", no_solve)
    stop = float(gun.exit_plane_z_mm)*1e-3 + .1
    clone, electric, notes = _prepare_electric_provider(SimpleNamespace(electron_gun=gun), stop)
    assert clone is not gun and electric is field
    assert gun._gun_field_exit_extension_mm == original_extension
    assert clone._gun_field_exit_extension_mm == 200.
    assert electric.request == request
    assert electric_field_identity(electric).status == "known"
    assert any("original numerical identity" in note for note in notes)


def test_worker_protocol_five_preserves_known_immutable_cache_identity(gun):
    from temsim.test_electron_execution import _dumps
    field = tiny_field(closed_field_request(gun))
    expected = electric_field_identity(field)
    received = pickle.loads(_dumps(field))
    assert received is not field
    assert not received.r.flags.writeable and not received.z.flags.writeable and not received.voltage.flags.writeable
    assert electric_field_identity(received) == expected


def test_plain_deepcopy_mutability_is_not_silently_upgraded(gun):
    field = tiny_field(closed_field_request(gun))
    cloned = deepcopy(field)
    assert cloned.voltage.flags.writeable
    assert electric_field_identity(cloned).status == "unknown"
