from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.physics.first_order import TransverseTransfer
from temsim.physics.record_plane import (
    build_record_plane_plan,
    PlaneStop,
    RecordPlanePlan,
    project_sample_phase_space,
    resolved_runtime_geometry_fingerprint,
    route_record_planes,
)


def _transfer(z_mm, *, j_img=None, j_diff=None, k_img=None, k_diff=None):
    return TransverseTransfer(
        0.0,
        float(z_mm),
        np.eye(2) if j_img is None else np.asarray(j_img, dtype=float),
        np.zeros((2, 2)) if j_diff is None else np.asarray(j_diff, dtype=float),
        np.zeros((2, 2)) if k_img is None else np.asarray(k_img, dtype=float),
        np.eye(2) if k_diff is None else np.asarray(k_diff, dtype=float),
    )


def _plan(planes, transfers):
    return RecordPlanePlan(
        source_z_mm=0.0,
        planes=tuple(planes),
        transfers=tuple(transfers),
        resolved_geometry_fingerprint="1" * 64,
        fingerprint="2" * 64,
    )


def test_signed_mixed_plane_projection_uses_both_position_and_angle_blocks():
    transfer = _transfer(
        10.0,
        j_img=((2.0, -1.0), (0.5, 3.0)),
        j_diff=((0.10, -0.20), (-0.30, 0.40)),
        k_img=((4.0, 0.0), (0.0, -2.0)),
        k_diff=((-1.0, 0.5), (0.25, 2.0)),
    )
    position = np.array(((1.0e-6, -2.0e-6), (3.0e-6, 4.0e-6)))
    angle = np.array(((2.0e-3, -1.0e-3), (-4.0e-3, 3.0e-3)))

    projected = project_sample_phase_space(transfer, position, angle)

    expected_position = position @ transfer.j_img.T + angle @ transfer.j_diff_m_per_rad.T
    expected_angle = position @ transfer.k_img_rad_per_m.T + angle @ transfer.k_diff.T
    assert projected.position_m == pytest.approx(expected_position, abs=1.0e-18)
    assert projected.angle_rad == pytest.approx(expected_angle, abs=1.0e-18)
    # The negative off-diagonal terms are physically significant; replacing
    # the signed matrices by radial magnitudes must not reproduce this result.
    unsigned = position @ np.abs(transfer.j_img).T + angle @ np.abs(
        transfer.j_diff_m_per_rad
    ).T
    assert not np.allclose(projected.position_m, unsigned)


def test_sequential_aperture_and_detectors_conserve_weight_without_double_counting():
    aperture = PlaneStop(
        "sad", "Selected aperture", 1.0, "aperture", "disk", radius_mm=1.0
    )
    first = PlaneStop(
        "bf", "BF", 2.0, "detector", "disk", outer_width_mm=2.0,
        readout_enabled=True,
    )
    second = PlaneStop(
        "camera", "Camera", 3.0, "detector", "square", outer_width_mm=20.0,
        readout_enabled=True,
    )
    plan = _plan(
        (aperture, first, second),
        (_transfer(1.0), _transfer(2.0), _transfer(3.0)),
    )
    position = np.array(((-2.0e-3, 0.0), (-0.5e-3, 0.0), (0.5e-3, 0.0), (2.0e-3, 0.0)))
    angle = np.zeros_like(position)
    weights = np.array((0.1, 0.2, 0.3, 0.4))

    result = route_record_planes(plan, position, angle, weights=weights)

    by_key = {event.plane.key: event for event in result.interactions}
    assert by_key["sad"].intercepted_weight == pytest.approx(0.5)
    assert by_key["bf"].signal_weight == pytest.approx(0.5)
    assert by_key["camera"].signal_weight == pytest.approx(0.0)
    assert result.surviving_weight == pytest.approx(0.0)
    assert result.physically_intercepted_weight == pytest.approx(1.0)
    assert result.balance_error == pytest.approx(0.0, abs=1.0e-15)


def test_transverse_field_affine_offset_moves_beam_outside_detector():
    transfer = replace(_transfer(1.0), position_offset_m=(2e-3, 0),
                       angle_offset_rad=(1e-3, -2e-3))
    detector = PlaneStop("bf", "BF", 1.0, "detector", "disk",
                         outer_width_mm=2.0, readout_enabled=True)
    position = np.zeros((1,2))
    projected = project_sample_phase_space(transfer, position, position)
    np.testing.assert_allclose(projected.position_m, [[2e-3, 0]])
    np.testing.assert_allclose(projected.angle_rad, [[1e-3, -2e-3]])
    result = route_record_planes(_plan((detector,), (transfer,)), position, position)
    assert result.interactions[0].signal_weight == 0
    assert result.surviving_weight == 1


def test_inserted_detector_blocks_even_when_readout_is_disabled():
    detector = PlaneStop(
        "screen", "Screen", 1.0, "detector", "disk", outer_width_mm=10.0,
        readout_enabled=False,
    )
    downstream = PlaneStop(
        "camera", "Camera", 2.0, "detector", "square", outer_width_mm=20.0,
        readout_enabled=True,
    )
    result = route_record_planes(
        _plan((detector, downstream), (_transfer(1.0), _transfer(2.0))),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        weights=np.array((1.0, 2.0, 3.0)),
    )

    assert result.interactions[0].intercepted_weight == pytest.approx(6.0)
    assert result.interactions[0].signal_weight == pytest.approx(0.0)
    assert result.interactions[1].signal_weight == pytest.approx(0.0)
    assert result.balance_error == pytest.approx(0.0)


def test_resolved_geometry_fingerprint_changes_with_pole_piece_geometry():
    pole = SimpleNamespace(
        key="objective_upper_pole",
        start_z_mm=10.0,
        center_z_mm=11.0,
        end_z_mm=12.0,
        length_mm=2.0,
        data=MappingProxyType({
            "mechanical_outer_diameter_mm": 30.0,
            "mechanical_bore_diameter_mm": 3.0,
            "material": "soft_magnetic_alloy",
        }),
    )
    assembly = SimpleNamespace(
        selected_module_paths=(("column", "runtime.toml"),),
        parts=(pole,),
        vacuum_bore_segments=(),
        vacuum_liner_segments=(),
        exit_z_mm=100.0,
    )
    state = SimpleNamespace(
        sample=SimpleNamespace(z_mm=0.0),
        beam_voltage_kv=300.0,
        _resolved_assembly=assembly,
    )
    planes = (PlaneStop("camera", "Camera", 50.0, "detector", "square", outer_width_mm=10.0),)
    initial = resolved_runtime_geometry_fingerprint(state, planes)
    changed_pole = SimpleNamespace(**{
        **vars(pole),
        "data": MappingProxyType({**dict(pole.data), "mechanical_bore_diameter_mm": 2.5}),
    })
    state._resolved_assembly = SimpleNamespace(**{
        **vars(assembly),
        "parts": (changed_pole,),
    })

    changed = resolved_runtime_geometry_fingerprint(state, planes)

    assert initial != changed


def test_real_plan_reads_runtime_plane_positions_and_rejects_lens_position_edit():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    initial = build_record_plane_plan(state, maximum_step_mm=5.0)

    runtime_z = {
        component.key: float(component.z_mm)
        for component in (*state.apertures, *state.recording_planes)
    }
    assert initial.planes
    assert all(plane.z_mm == pytest.approx(runtime_z[plane.key]) for plane in initial.planes)

    projector_two = next(
        lens for lens in state.lenses if lens.key == "projector_lens_2"
    )
    projector_two.z_mm += 1.0
    changed = build_record_plane_plan(state, maximum_step_mm=5.0)

    assert changed.fingerprint != initial.fingerprint
    assert any(
        not np.allclose(before.matrix, after.matrix, rtol=1.0e-10, atol=1.0e-14)
        for before, after in zip(initial.transfers, changed.transfers)
    )
