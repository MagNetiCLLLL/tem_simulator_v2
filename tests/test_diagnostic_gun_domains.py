"""Comparison accounting fixtures; no field solve or full-gun qualification."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.diagnostic_gun_comparison import (
    REFERENCE_CHECKS, compare_populations, emission_samples, field_difference,
    first_plane_crossing, trajectory_measurements,
)


def path(z=(0., 1., .5, 2.)):
    z = np.asarray(z)
    n = len(z)
    return SimpleNamespace(positions_m=np.column_stack((np.arange(n), np.zeros(n), z)),
        directions=np.tile([0., 0., 1.], (n, 1)), time_s=np.arange(n)*1e-9,
        kinetic_energy_ev=1.+z*3., electrostatic_potential_v=z*3., reason="domain_exit",
        completed=True, steps=n-1)


def row(identifier=7, trajectory=None):
    return {**trajectory_measurements(trajectory or path(), (("middle", .75), ("end", 2.))),
            "ray_id": identifier, "original_weight": .125}


def test_first_arrival_keeps_chronology_when_particle_turns():
    measured = first_plane_crossing(path(), .75)
    assert measured["position_m"] == pytest.approx([.75, 0., .75])
    assert measured["flight_time_s"] == pytest.approx(.75e-9)
    assert measured["kinetic_energy_ev"] == pytest.approx(3.25)
    assert "interpolated" in measured["sample_basis"]


def test_plane_before_source_or_beyond_stop_is_not_extrapolated():
    trajectory = path()
    assert first_plane_crossing(trajectory, -.01) is None
    assert first_plane_crossing(trajectory, 2.01) is None
    assert first_plane_crossing(trajectory, 0.)["flight_time_s"] == 0.


def test_zero_momentum_direction_is_undefined_instead_of_nan():
    trajectory = path((0., 1., 2.))
    trajectory.directions[:] = 0.
    assert first_plane_crossing(trajectory, 1.)["direction"] is None
    values = [row(trajectory=trajectory)]
    compared = compare_populations(values, values)
    assert all(item["direction_difference_rad"] is None for item in compared["per_particle_plane_differences"])


def test_completed_terminal_contact_accepts_float_roundoff_without_moving_endpoint():
    trajectory = path((0., .4499999999999996))
    actual = first_plane_crossing(trajectory, .45)
    assert actual is not None
    assert actual["position_m"][2] == trajectory.positions_m[-1, 2]
    assert actual["terminal_delta_z_m"] == trajectory.positions_m[-1, 2]-.45
    assert abs(actual["terminal_delta_z_m"]) <= actual["terminal_roundoff_window_m"]
    assert actual["flight_time_s"] == trajectory.time_s[-1]
    assert "actual endpoint retained" in actual["sample_basis"]


@pytest.mark.parametrize("reason,completed,end", [("domain_exit", True, .45-1e-12),
    ("step_limit", False, .4499999999999996), ("path_limit", True, .4499999999999996),
    ("cancelled", False, .4499999999999996), ("in_progress", False, .4499999999999996)])
def test_terminal_roundoff_does_not_extrapolate_unfinished_or_path_limited_trajectories(reason, completed, end):
    trajectory = path((0., end))
    trajectory.reason, trajectory.completed = reason, completed
    assert first_plane_crossing(trajectory, .45) is None


def test_terminal_roundoff_never_turns_backward_arrival_into_forward_arrival():
    trajectory = path((.6, .4499999999999996))
    assert first_plane_crossing(trajectory, .45) is None


def test_invariant_sign_is_kinetic_minus_potential_and_drift_is_measured():
    trajectory = path()
    assert trajectory_measurements(trajectory, ())["maximum_invariant_drift_ev"] == 0.
    trajectory.kinetic_energy_ev[-1] += .01
    result = trajectory_measurements(trajectory, ())
    assert result["initial_invariant_ev"] == 1.
    assert result["maximum_invariant_drift_ev"] == pytest.approx(.01)
    assert not result["reference_fixture_invariant_check"]
    assert REFERENCE_CHECKS["domain_equivalence_tolerance"] is None


def test_no_arrival_changes_population_instead_of_becoming_zero_error():
    first = [row()]
    stopped = path((0., .5))
    stopped.reason = "aperture:dpa"
    compared = compare_populations(first, [row(trajectory=stopped)])
    assert not compared["same_terminal_reasons"]
    assert compared["plane_populations"]["end"]["first"] == {"count": 1, "original_weight": .125}
    assert compared["plane_populations"]["end"]["second"] == {"count": 0, "original_weight": 0}
    for point in compared["per_particle_plane_differences"]:
        assert point["first_reached"] and not point["second_reached"]
        assert "xy_difference_m" not in point


@pytest.mark.parametrize("kind", ["duplicate", "changed_id", "changed_weight", "empty", "different_planes"])
def test_population_comparison_rejects_silent_reweighting_or_rematching(kind):
    first, second = [row()], [row()]
    if kind == "duplicate":
        second *= 2
    elif kind == "changed_id":
        second[0]["ray_id"] = 8
    elif kind == "changed_weight":
        second[0]["original_weight"] = 1.
    elif kind == "empty":
        first = []
    else:
        second[0]["planes"].pop("middle")
    with pytest.raises(ValueError):
        compare_populations(first, second)


def test_per_plane_differences_keep_units_and_identity():
    first = row()
    second = deepcopy(first)
    second["planes"]["middle"]["position_m"][0] += .01
    second["planes"]["middle"]["direction"] = [np.sin(.002), 0., np.cos(.002)]
    second["planes"]["middle"]["kinetic_energy_ev"] += .2
    second["planes"]["middle"]["flight_time_s"] += 1e-12
    result = compare_populations([first], [second])["per_particle_plane_differences"][0]
    assert result["xy_difference_m"] == pytest.approx(.01)
    assert result["direction_difference_rad"] == pytest.approx(.002)
    assert result["kinetic_energy_difference_ev"] == pytest.approx(.2)
    assert result["flight_time_difference_s"] == pytest.approx(1e-12)


class Field:
    r = np.array([0., 1.])
    z = np.array([0., 2.])

    def __init__(self, gain=1.):
        self.gain = gain

    def interpolate(self, points):
        return points[:, 2]*self.gain, np.column_stack((points[:, 0]*self.gain,
            np.zeros(len(points)), np.zeros(len(points))))


def test_local_field_error_is_not_normalised_by_another_points_peak():
    result = field_difference(Field(), Field(2.), [[.001, 0., .1], [.9, 0., 1.]])
    assert [p["local_relative_electric_difference"] for p in result["points"]] == pytest.approx([1., 1.])
    assert result["points"][0]["electric_difference_norm_v_per_m"] == pytest.approx(.001)
    zero = field_difference(Field(), Field(2.), [[0., 0., 0.]])["points"][0]
    assert zero["local_relative_electric_difference"] is None
    assert zero["local_relative_potential_difference"] is None


@pytest.mark.parametrize("points", [[[0., 0., 2.1]], [[1.01, 0., 1.]], [], [[float("nan"), 0., 0.]]])
def test_field_comparison_requires_real_common_support(points):
    with pytest.raises(ValueError):
        field_difference(Field(), Field(), points)


def test_samples_are_original_tip_population_with_unchanged_ids_weights():
    from temsim.optics.column import default_state
    gun = default_state().electron_gun
    gun.emitter.curvature_nm_inv = .01
    emitted = gun.emit(193)
    result = emission_samples(gun, 193, 3)
    assert result == emission_samples(gun, 193, 3)
    assert result["selected_count"] == 3
    for sample in result["samples"]:
        index = sample["original_index"]
        assert sample["ray_id"] == emitted.ray_id[index]
        assert sample["original_weight"] == emitted.weight[index]
        np.testing.assert_array_equal(sample["position_m"], emitted.surface_position_m[index])
        assert sample["kinetic_energy_ev"] == gun.emitter.emission_energy_ev+emitted.energy_offset_ev[index]
    assert result["selected_weight"] < result["emitted_weight"]


def test_near_aperture_input_does_not_create_or_move_the_source():
    from scripts.compare_diagnostic_gun_domains import inputs
    flat, edge = inputs("flat"), inputs("near_aperture")
    assert emission_samples(flat.electron_gun) == emission_samples(edge.electron_gun)
    aperture = edge.electron_gun.dpa_aperture
    assert aperture.offset_x_mm == pytest.approx(.999*aperture.radius_mm)
    assert flat.electron_gun.dpa_aperture.offset_x_mm == 0.


def test_reanalysis_uses_saved_arrays_and_preserves_original_evidence(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import compare_diagnostic_gun_domains as driver
    trajectory = path((0., .4499999999999996))
    measured = trajectory_measurements(trajectory, (("exit", .45),))
    measured["planes"]["exit"] = None  # Before the roundoff measurement repair.
    measured.update(ray_id=7, original_weight=.125)
    variants = {name for pair in driver.COMPARISON_PAIRS for name in pair[1:]}
    output = tmp_path/"flat"
    output.mkdir()
    checksums = {}
    for variant in variants:
        archive = output/f"{variant}-ray7.npz"
        np.savez_compressed(archive, position_m=trajectory.positions_m, direction=trajectory.directions,
            time_s=trajectory.time_s, path_length_m=np.array([0., 1.]),
            kinetic_energy_ev=trajectory.kinetic_energy_ev, potential_v=trajectory.electrostatic_potential_v,
            momentum_kg_m_per_s=trajectory.directions*1e-25)
        checksums[archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = {"schema": "diagnostic-gun-domain-comparison-v1", "status": "COMPARISON_COMPLETE_REVIEW_REQUIRED",
        "cases": {"flat": {"inputs": {"graph": {"nodes": []}}, "observation_planes_m": {"exit": .45},
            "source": {"samples": [{"ray_id": 7, "position_m": [0., 0., 0.]}]},
            "variants": {variant: [deepcopy(measured)] for variant in variants}}}}
    destination = tmp_path/"report.json"
    original = json.dumps(report, allow_nan=False).encode()
    destination.write_bytes(original)
    monkeypatch.setattr(driver, "inputs", lambda *args: pytest.fail("Reanalysis cannot use current defaults"))
    monkeypatch.setattr(driver, "trace_test_electron", lambda *args, **kwargs: pytest.fail("Reanalysis cannot propagate"))
    assert driver.reanalyse(tmp_path) == 0
    result = json.loads(destination.read_text())
    for rows in result["cases"]["flat"]["variants"].values():
        assert rows[0]["planes"]["exit"] is not None
        assert rows[0]["planes"]["exit"]["position_m"][2] == trajectory.positions_m[-1, 2]
    metadata = result["measurement_reanalysis"]
    assert metadata["field_solves"] == metadata["transport_executions"] == 0
    assert metadata["raw_arrays_unchanged"]
    assert "NOT_CAPTURED" in metadata["original_trace_source_provenance"]
    assert (tmp_path/metadata["prior_report"]).read_bytes() == original
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.npz")} == checksums
