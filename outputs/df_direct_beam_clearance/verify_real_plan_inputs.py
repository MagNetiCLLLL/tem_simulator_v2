"""Verify UI planner inputs against the recorded real 32x32 Si acquisition.

No new source-ray or wave calculation: only recorded probe statistics replace
the statistics reader; scan calibration and all record-plane maps remain real.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui.main_window import MainWindow
from temsim.optics.model import State
from temsim.physics.scan_geometry import calibrate_scan_system, paired_kick_response, raster_sample_grid
from temsim.profile_io import read_profile

HERE = Path(__file__).resolve().parent / "si110_32px_004nm_df60_100_final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parameters = json.loads((HERE / "parameters.json").read_text())
saved_state = json.loads((HERE / "final_acquisition_state.json").read_text())
preflight = json.loads((HERE / "mask_preflight.json").read_text())
metrics = json.loads((HERE / "metrics.json").read_text())["production_metrics"]
assert parameters["state"] == saved_state
inputs = [HERE / name for name in ("parameters.json", "final_acquisition_state.json", "raw_scan.npz", "mask_preflight.json")]
hashes = {path.name: sha(path) for path in inputs}
selection, _values = read_profile(HERE / "operating_profile.toml")
catalog = AssemblyCatalog(root=HERE / "instrument_inputs")
state = State.from_dict(saved_state)
catalog.apply(state, selection, preserve_operating_parameters=True)
apply_physical_layout_to_state(state, assembly_root=HERE / "instrument_inputs", preserve_operating_parameters=True)
before = deepcopy(state.to_dict())
with np.load(HERE / "raw_scan.npz", allow_pickle=False) as raw:
    frame = SimpleNamespace(scan_x_um=raw["scan_x_nm"] * 1e-3,
                            scan_y_um=raw["scan_y_nm"] * 1e-3,
                            metrics=metrics)
result = SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(incident=object()))
with patch("temsim.physics.wave_imaging._weighted_ray_statistics", return_value=parameters["probe"]) as recorded_stats:
    plan, positions_m, alpha, chief = MainWindow._df_geometry_plan_inputs(result, frame)
assert recorded_stats.call_count == 1
assert state.to_dict() == before, "UI planner mutated the captured state"

# Independently reproduce the production preflight's raster formula from its
# saved acquisition settings, not the UI implementation's intermediate data.
reference = State.from_dict(saved_state)
catalog.apply(reference, selection, preserve_operating_parameters=True)
apply_physical_layout_to_state(reference, assembly_root=HERE / "instrument_inputs", preserve_operating_parameters=True)
calibrate_scan_system(reference)
ac = reference.ac_deflector
xf, yf, times = raster_sample_grid(ac, maximum_count=None)
factors = np.stack((xf, yf), axis=-1)
kicks = np.einsum("ij,...j->...i", ac.scan_command_matrix_mrad, factors)
response = 1e3 * paired_kick_response(reference, ac, reference.sample.z_mm)
offsets_um = (kicks * 1e-3) @ response.T * 1e3
base_kick = np.asarray(ac.scan_kick_mrad(float(getattr(reference, "simulation_time_s", 0.))))
base_um = (base_kick * 1e-3) @ response.T * 1e3
probe = parameters["probe"]
origin_um = np.array([probe["mean_x_m"], probe["mean_y_m"]]) * 1e6 - base_um
scan_origin_um = np.array([reference.sample.scan_origin_x_nm, reference.sample.scan_origin_y_nm]) * 1e-3
reference_scan_um = offsets_um + scan_origin_um
reference_positions = (reference_scan_um + origin_um) * 1e-6
np.testing.assert_allclose(np.stack((frame.scan_x_um, frame.scan_y_um), axis=-1), reference_scan_um, rtol=0, atol=1e-14)
np.testing.assert_allclose(positions_m, reference_positions, rtol=0, atol=1e-18)
np.testing.assert_array_equal(plan.scan_times_s, times)
np.testing.assert_array_equal(chief, np.array([probe["mean_tx_rad"], probe["mean_ty_rad"]]) * 1e3)
assert alpha == probe["convergence_95_rad"] * 1e3

recorded = {row["key"]: row for row in preflight["planes"]}
plane_errors = {}
for plane, transfer in zip(plan.planes, plan.transfers):
    prior = recorded[plane.key]
    errors = {}
    for field in ("j_diff_m_per_rad", "j_img", "position_offset_m"):
        actual = np.asarray(getattr(transfer, field))
        expected = np.asarray(prior[field])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        errors[field] = float(np.max(np.abs(actual - expected)))
    # Resolving saved local coordinates again can change an absolute Z by a
    # floating-point addition ulp; record it and require sub-picometre error.
    errors["z_mm"] = abs(plane.z_mm - prior["z_mm"])
    np.testing.assert_allclose(plane.z_mm, prior["z_mm"], rtol=0, atol=1e-9)
    restored_stop = asdict(plane)
    restored_stop["z_mm"] = prior["stop"]["z_mm"]
    assert restored_stop == prior["stop"]
    plane_errors[plane.key] = errors
assert hashes == {path.name: sha(path) for path in inputs}
assert state.to_dict() == before
df_index = next(index for index, plane in enumerate(plan.planes) if plane.key == "df")
output = dict(
    passed=True,
    scope="Actual saved state and raw scan coordinates; recorded actual probe statistics only. No new source-ray or wave calculation. Real scan calibration and downstream first-order map traces were executed.",
    stats_source="parameters.json['probe'] (561 surviving measured rays)",
    input_sha256=hashes,
    state_serialization_unchanged=True,
    scan_shape=list(times.shape), scan_time_range_s=[float(times.min()), float(times.max())],
    scan_times_exact_match=True,
    raw_scan_vs_preflight_max_error_um=float(np.max(np.abs(np.stack((frame.scan_x_um, frame.scan_y_um), axis=-1) - reference_scan_um))),
    ui_positions_vs_preflight_max_error_m=float(np.max(np.abs(positions_m - reference_positions))),
    raster_origin_um=origin_um.tolist(), baseline_scan_shift_um=base_um.tolist(),
    probe_chief_angle_mrad=chief.tolist(), modeled_probe_semiangle_mrad=alpha,
    df_jdiff_m_per_rad=plan.transfers[df_index].j_diff_m_per_rad.tolist(),
    plane_matrix_max_errors=plane_errors,
    plan_fingerprint=plan.fingerprint, recorded_plan_fingerprint=preflight["plan_fingerprint"],
    plan_fingerprint_exact_match=plan.fingerprint == preflight["plan_fingerprint"],
)
(HERE / "real_plan_inputs_verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
