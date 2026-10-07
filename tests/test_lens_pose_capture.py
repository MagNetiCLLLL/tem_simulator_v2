"""Saved physical lens placement survives requests and invalidates old results."""
from dataclasses import replace
from pathlib import Path

import pytest

from temsim.calculation_cache import calculation_signatures, incident_field_dependencies
from temsim.gui.calculation_request import CapturedCalculationRequest
from temsim.immutable_json import freeze_json, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot, decode_instrument
from temsim.lens_pose import PHYSICAL_POSE_FIELDS, lens_pose_registration
from temsim.optics.column import default_state


def _pose(state, part_key, *, remove=(), **values):
    assembly = state._resolved_assembly
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(part, data=freeze_json({**{key: value for key, value in part.data.items() if key not in remove},
                                       **values})) if part.key == part_key else part
        for part in assembly.parts))


def test_pose_snapshot_restores_without_reopening_live_geometry(monkeypatch):
    state = default_state()
    _pose(state, "condenser_lens_1", offset_x_mm=0.04, rotation_y_mrad=-1.25)
    expected = lens_pose_registration(state, "condenser_lens_1")
    snapshot = capture_instrument_snapshot(state)
    # Cache objects must neither enter an input digest nor require a decoder.
    state._runtime_posed_nonlinear_provider_cache = {"unused": object()}
    state._lens_pose_bore_cache = (state._resolved_assembly, {"unused": object()})
    assert capture_instrument_snapshot(state).digest == snapshot.digest
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", lambda *_a, **_kw: pytest.fail("Snapshot restoration reopened live files"))
        restored = decode_instrument(snapshot.graph)
        assert lens_pose_registration(restored, "condenser_lens_1") == expected
    assert restored._resolved_assembly.part("condenser_lens_1").data["rotation_y_mrad"] == -1.25
    assert not hasattr(restored, "_runtime_posed_nonlinear_provider_cache")
    assert not hasattr(restored, "_lens_pose_bore_cache")


def test_pose_registration_cache_does_not_change_input_snapshot():
    state = default_state()
    _pose(state, "condenser_lens_1", offset_y_mm=0.025, rotation_x_mrad=0.5)
    state.__dict__.pop("_lens_pose_registration_cache", None)
    before = capture_instrument_snapshot(state)
    expected = lens_pose_registration(state, "condenser_lens_1")
    lens_pose_registration(state, "objective_aperture")
    assert hasattr(state, "_lens_pose_registration_cache")
    after = capture_instrument_snapshot(state)
    assert after.digest == before.digest
    restored = decode_instrument(after.graph)
    assert not hasattr(restored, "_lens_pose_registration_cache")
    assert lens_pose_registration(restored, "condenser_lens_1") == expected
    _pose(state, "condenser_lens_1", offset_y_mm=0.075)
    assert lens_pose_registration(state, "condenser_lens_1") != expected


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
def test_captured_request_keeps_lens_pose_when_live_assembly_changes(quality):
    state = default_state()
    _pose(state, "condenser_lens_1", offset_x_mm=0.03, rotation_x_mrad=1.5)
    expected = lens_pose_registration(state, "condenser_lens_1")
    request = CapturedCalculationRequest.capture(state, quality, 9, 2.0, workflow="rays")
    try:
        _pose(state, "condenser_lens_1", offset_x_mm=-0.08, rotation_x_mrad=4.0)
        if request._instrument_graph is not None:
            restored = decode_instrument(request._instrument_graph, assets=request._input_assets)
        else:
            restored = request._model_state
        assert lens_pose_registration(restored, "condenser_lens_1") == expected
        assert lens_pose_registration(state, "condenser_lens_1") != expected
    finally:
        if request._input_assets is not None:
            request._input_assets.close()


@pytest.mark.parametrize("key", ["condenser_lens_1", "projector_lens_1"])
def test_lens_pose_invalidates_incident_cache_including_downstream_spatial_fields(key):
    state = default_state()
    before = calculation_signatures(state)
    _pose(state, key, rotation_y_mrad=1.25)
    after = calculation_signatures(state)
    assert after["incident"] != before["incident"]
    _pose(state, key, rotation_y_mrad=-1.25)
    assert calculation_signatures(state)["incident"] != after["incident"]
    if key == "projector_lens_1":
        row = next(row for row in incident_field_dependencies(state)["rows"] if row["key"] == key)
        assert row["reason"] == "posed spatial field"
        assert "physical_pose" in row


def test_legacy_rigid_lens_edits_invalidate_scoped_cache_and_migrate_without_new_identity():
    from temsim.calculation_manifest import resolved_assembly_geometry_fingerprints
    from temsim.lens_pose import migrate_legacy_lens_pose
    state = default_state()
    before = calculation_signatures(state)
    model = {"schema_version": 1, "transform": {"offset_mm": [0.025, 0.0, 0.0],
                                                  "rotation_deg": [0.0, 0.05, 0.0]}}
    _pose(state, "condenser_lens_1", remove=PHYSICAL_POSE_FIELDS, model_3d=model)
    assert calculation_signatures(state)["incident"] != before["incident"]
    legacy = resolved_assembly_geometry_fingerprints(state)
    data = thaw_json(state._resolved_assembly.part("condenser_lens_1").data)
    migrate_legacy_lens_pose(data)
    _pose(state, "condenser_lens_1", **data)
    assert resolved_assembly_geometry_fingerprints(state) == legacy
