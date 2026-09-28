"""Private response-schema tests use synthetic arrays, never a field solve."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.electron_execution_protocol import (
    ElectronProtocolValueError, validate_scene_metadata, validate_trajectory_payload,
)
from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory
from temsim.test_electron_execution import RemoteElectronScene


def scene_metadata(**changes):
    bounds = ((-.01, -.01, 0.), (.01, .01, .1))
    return replace(RemoteElectronScene("token", "owner", (0., 0., 0.), .3, .1, bounds, bounds,
        ("Synthetic IPC fixture only",), physical_identity="physical", numerical_identity="numerical",
        transport_identity="transport"), **changes)


def trajectory(**changes):
    vector = np.zeros((2, 3))
    scalar = np.zeros(2)
    return replace(TestElectronTrajectory(positions_m=vector.copy(), directions=vector.copy(),
        momentum_kg_m_per_s=vector.copy(), time_s=np.array((0., 1e-12)), path_length_m=np.array((0., .001)),
        kinetic_energy_ev=np.full(2, .3), electrostatic_potential_v=scalar.copy(), speed_m_per_s=scalar.copy(),
        energy_invariant_error_ev=0., energy_invariant_relative_error=0., reason="path_limit", completed=True,
        steps=1, notes=("Synthetic arrays; physical accuracy is not asserted.",),
        physical_field_identity="physical", numerical_field_identity="numerical", execution_identity="execution"), **changes)


def check(value, *, progress=False, scene=None, execution_identity="execution"):
    return validate_trajectory_payload(value, settings=TestElectronSettings(max_steps=4),
        scene=scene or scene_metadata(), execution_identity=execution_identity, progress=progress)


def test_valid_payloads_return_the_original_objects_without_repair_or_freezing():
    metadata = scene_metadata()
    result = trajectory()
    before = tuple(getattr(result, name) for name in ("positions_m", "time_s", "speed_m_per_s"))
    assert validate_scene_metadata(metadata, process_identity="owner") is metadata
    assert check(result) is result
    assert all(getattr(result, name) is array for name, array in zip(("positions_m", "time_s", "speed_m_per_s"), before))
    assert result.positions_m.flags.writeable and np.all(result.speed_m_per_s == 0.)


@pytest.mark.parametrize("reason,completed", [("path_limit", True), ("domain_exit", True),
    ("tip_return", True), ("aperture:aperture-key", True), ("hardware:bore", True), ("aperture_stop", True),
    ("step_limit", False), ("numerical_limit", False), ("cancelled", False),
    ("initial_outside_domain", False), ("unsupported_field:filter", False)])
def test_existing_solver_stop_semantics_are_preserved(reason, completed):
    assert check(trajectory(reason=reason, completed=completed)).completed is completed
    with pytest.raises(ElectronProtocolValueError, match="completed"):
        check(trajectory(reason=reason, completed=not completed))


def test_progress_and_terminal_modes_cannot_be_interchanged():
    prefix = trajectory(reason="in_progress", completed=False)
    assert check(prefix, progress=True) is prefix
    with pytest.raises(ElectronProtocolValueError):
        check(prefix)
    with pytest.raises(ElectronProtocolValueError):
        check(trajectory(), progress=True)


@pytest.mark.parametrize("name", ["positions_m", "directions", "momentum_kg_m_per_s", "time_s",
    "path_length_m", "kinetic_energy_ev", "speed_m_per_s", "electrostatic_potential_v"])
@pytest.mark.parametrize("fault", ["shape", "nan", "infinity", "object", "complex"])
def test_invalid_arrays_are_rejected(name, fault):
    array = getattr(trajectory(), name).copy()
    if fault == "shape":
        array = array[:-1]
    elif fault in ("nan", "infinity"):
        array.flat[-1] = np.nan if fault == "nan" else np.inf
    else:
        array = array.astype(object if fault == "object" else complex)
    with pytest.raises(ElectronProtocolValueError):
        check(trajectory(**{name: array}))


@pytest.mark.parametrize("steps", [-1, 5, 0, True, 1.0])
def test_step_budget_and_sample_count_are_bound_to_the_request(steps):
    with pytest.raises(ElectronProtocolValueError):
        check(trajectory(steps=steps))


@pytest.mark.parametrize("name", ["time_s", "path_length_m"])
@pytest.mark.parametrize("values", [(1., 2.), (0., -1.), (0., 2., 1.)])
def test_time_and_path_must_start_at_zero_and_never_decrease(name, values):
    result = trajectory(**{name: np.array(values)})
    if len(values) == 3:
        result = replace(result, steps=2, **{other: np.zeros((3, 3)) if getattr(result, other).ndim == 2 else np.zeros(3)
            for other in ("positions_m", "directions", "momentum_kg_m_per_s", "time_s", "path_length_m",
                          "kinetic_energy_ev", "speed_m_per_s", "electrostatic_potential_v") if other != name})
    with pytest.raises(ElectronProtocolValueError):
        check(result)


@pytest.mark.parametrize("name", ["physical_field_identity", "numerical_field_identity", "execution_identity"])
@pytest.mark.parametrize("identity", [None, "another-request", "", 7])
def test_known_result_identity_must_match_the_request(name, identity):
    with pytest.raises(ElectronProtocolValueError, match="identity"):
        check(trajectory(**{name: identity}))


def test_unknown_identity_stays_unknown_and_cannot_be_invented():
    metadata = scene_metadata(physical_identity=None, numerical_identity=None, transport_identity=None)
    result = trajectory(physical_field_identity=None, numerical_field_identity=None, execution_identity=None)
    assert check(result, scene=metadata, execution_identity=None) is result
    for name in ("physical_field_identity", "numerical_field_identity", "execution_identity"):
        with pytest.raises(ElectronProtocolValueError):
            check(replace(result, **{name: "invented"}), scene=metadata, execution_identity=None)


@pytest.mark.parametrize("reason", ["initial_outside_domain", "cancelled"])
def test_unpropagated_unknown_electric_potential_remains_unknown(reason):
    result = trajectory(steps=0, reason=reason, completed=False, positions_m=np.zeros((1, 3)),
        directions=np.zeros((1, 3)), momentum_kg_m_per_s=np.zeros((1, 3)), time_s=np.zeros(1),
        path_length_m=np.zeros(1), kinetic_energy_ev=np.full(1, .3), speed_m_per_s=np.zeros(1),
        electrostatic_potential_v=np.full(1, np.nan), energy_invariant_error_ev=np.nan,
        energy_invariant_relative_error=np.nan)
    assert check(result) is result and np.isnan(result.electrostatic_potential_v[0])
    with pytest.raises(ElectronProtocolValueError):
        check(replace(result, energy_invariant_error_ev=0.))


@pytest.mark.parametrize("changes", [
    {"reason": ""}, {"completed": 1}, {"notes": "not a tuple"},
    {"kinetic_energy_ev": np.array((.3, -.1))}, {"speed_m_per_s": np.array((0., -1.))},
    {"energy_invariant_error_ev": -1.}, {"energy_invariant_relative_error": np.inf},
    {"energy_invariant_error_ev": np.nan},
])
def test_invalid_scalar_status_and_readout_contracts_are_rejected(changes):
    with pytest.raises(ElectronProtocolValueError):
        check(trajectory(**changes))


@pytest.mark.parametrize("changes", [
    {"token": ""}, {"token": 1}, {"process_identity": "another-owner"},
    {"bounds_m": ((0., 0., 0.), (0., 1., 1.))},
    {"bounds_m": ((0., 0., 0.), (1., 1., np.nan))},
    {"diagnostic_bounds_m": ((0., 0., 0.), (1., 1., -1.))},
    {"initial_position_m": (0., 0.)}, {"initial_position_m": (0., 0., np.inf)},
    {"initial_energy_ev": 0.}, {"initial_energy_ev": True}, {"default_path_length_m": -1.},
    {"physical_identity": ""}, {"notes": (1,)}, {"support_metadata": (None,)},
])
def test_remote_metadata_rejects_invalid_owner_and_display_inputs(changes):
    with pytest.raises(ElectronProtocolValueError):
        validate_scene_metadata(scene_metadata(**changes), process_identity="owner")


def test_arbitrary_objects_are_not_trajectory_or_remote_scene_payloads():
    with pytest.raises(ElectronProtocolValueError):
        check({"reason": "path_limit"})
    with pytest.raises(ElectronProtocolValueError):
        validate_scene_metadata({"token": "token"}, process_identity="owner")
