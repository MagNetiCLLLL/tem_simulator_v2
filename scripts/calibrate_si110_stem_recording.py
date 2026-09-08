"""Find an actual D/I/P1/P2 crossover at the fixed projection DPA.

Only output profiles/JSON are written. The microscope's four existing lens
strengths are optimised through its existing field integrator; no stop, hole,
transfer matrix, incident ray or geometry is replaced.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from temsim.detector.stem_signal import physical_angular_detectors
from temsim.operating_modes import direct_alignment_by_key
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import PROJECTOR_KEYS, _LiveFirstOrderModel, _rk4_axisymmetric_larmor_matrix
from temsim.physics.record_plane import build_record_plane_plan, prepare_record_plane_detector_masks, record_plane_plan_provenance
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "si110_cif_5nm_64px_002nm" / "recording_calibration")
    parser.add_argument("--haadf-inner-mrad", type=float, default=60.)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    state = default_state()
    selection, values = read_profile(args.profile)
    apply_profile_values(state, values)
    before = {lens.key: lens.percent for lens in state.lenses}
    plan = build_record_plane_plan(state)
    dpa = next(plane for plane in plan.planes if plane.key == "projection_chamber_dpa_aperture")
    haadf = next(plane for plane in plan.planes if plane.key == "haadf")
    distance_m = (haadf.z_mm - dpa.z_mm) * 1e-3
    requested_b_haadf = (haadf.inner_diameter_mm * .5e-3) / (args.haadf_inner_mrad * 1e-3)
    desired_d = requested_b_haadf / distance_m
    started = perf_counter()
    model = _LiveFirstOrderModel(state, state.sample.z_mm, dpa.z_mm, PROJECTOR_KEYS, step_mm=.025)
    if model.vector_maps or np.max(np.abs(model.sx_m2)) > 1e-12 or np.max(np.abs(model.sy_m2)) > 1e-12:
        raise ValueError("This bounded calibration requires the existing axisymmetric analytic projector fields")
    initial = np.array([before[key] for key in PROJECTOR_KEYS])
    seeds = [initial, *[np.asarray(row, dtype=float) for row in direct_alignment_by_key("diffraction_camera_length").targets["preset_vectors"]]]
    solutions = []
    with threadpool_limits(limits=1):
        for sign in (1, -1):
            target_d = sign * desired_d
            def residual(vector):
                radial = _rk4_axisymmetric_larmor_matrix(model._field_arrays(vector), model.z_m)
                return np.r_[radial[0, 1] / 1e-4, (radial[1, 1] - target_d) / abs(target_d),
                             1e-7 * (np.asarray(vector) - initial)]
            for seed in seeds:
                fit = least_squares(residual, np.clip(seed, 1e-7, model.upper - 1e-7), bounds=(np.zeros(4), model.upper),
                                    max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11, diff_step=1e-5)
                radial = _rk4_axisymmetric_larmor_matrix(model._field_arrays(fit.x), model.z_m)
                valid = abs(radial[0, 1]) < 5e-6 and abs(radial[1, 1] / target_d - 1) < .01
                row = {"strengths": dict(zip(PROJECTOR_KEYS, map(float, fit.x))), "radial_matrix": radial.tolist(),
                       "success": bool(valid), "nfev": fit.nfev, "distance_from_preset": float(np.linalg.norm(fit.x - initial)),
                       "elapsed_s": perf_counter() - started}
                solutions.append(row)
                print(json.dumps(row), flush=True)
                (args.output / "candidate_solutions.json").write_text(json.dumps(solutions, indent=2), encoding="utf-8")
                if valid:
                    break
    good = [row for row in solutions if row["success"]]
    if not good:
        raise RuntimeError("No bounded physical DPA crossover was found")
    selected = min(good, key=lambda row: row["distance_from_preset"])
    for lens in state.lenses:
        if lens.key in selected["strengths"]:
            lens.percent = selected["strengths"][lens.key]
    after = {lens.key: lens.percent for lens in state.lenses}
    assert all(before[key] == after[key] for key in before if key not in PROJECTOR_KEYS)
    live = build_record_plane_plan(state)
    index = next(index for index, plane in enumerate(live.planes) if plane.key == dpa.key)
    transfer = live.transfers[index]
    worst_position_m = np.linalg.norm(transfer.j_img, 2) * np.sqrt(2) * 5e-9
    worst_angle_m = np.linalg.norm(transfer.j_diff_m_per_rad, 2) * .36
    worst_offset_m = np.linalg.norm(transfer.position_offset_m)
    bound_mm = (worst_position_m + worst_angle_m + worst_offset_m) * 1e3
    if bound_mm >= dpa.radius_mm:
        raise ValueError(f"Production transfer did not clear the fixed DPA: {bound_mm} >= {dpa.radius_mm} mm")
    theta = np.linspace(0., .36, 3601)
    angle = np.stack((theta, np.zeros_like(theta)), axis=-1)
    masks = prepare_record_plane_detector_masks(live, angle)(np.zeros(2))
    accepted = {key: [float(theta[mask].min() * 1e3), float(theta[mask].max() * 1e3)] if np.any(mask) else None
                for key, mask in masks.items()}
    _, reference = physical_angular_detectors(state, state.stem_detectors)
    output = {"source_profile": str(args.profile.resolve()), "elapsed_s": perf_counter() - started,
              "purpose": "Actual four-lens DPA crossover; fixed 0.1 mm radius retained; not a detector-independent diffraction-plane calibration",
              "strengths": selected["strengths"], "radial_matrix": selected["radial_matrix"],
              "unchanged_upstream_strengths": {key: value for key, value in before.items() if key not in PROJECTOR_KEYS},
              "production_dpa_j_img": transfer.j_img.tolist(), "production_dpa_j_diff_m_per_rad": transfer.j_diff_m_per_rad.tolist(),
              "dpa_radius_mm": dpa.radius_mm, "bound_at_360mrad_plus_5nm_square_mm": bound_mm,
              "detector_reference_angles": {key: asdict(value) for key, value in reference.items()},
              "sequential_accepted_positive_x_angle_ranges_mrad": accepted,
              "angle_range_sampling_step_mrad": .1,
              "scope": "HAADF remains high-angle; DF is a physical low-angle annulus overlapping the incident disk; BF is the small on-axis disk. All real stops retained.",
              "production_record_plane_plan": dict(record_plane_plan_provenance(live))}
    (args.output / "recording_calibration.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    save_profile(args.output / "calibrated_profile.toml", state, selection)
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
