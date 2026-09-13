"""Observe local axial errors without replacing any executed gun operator.

The archived physical settings and numerical budget are re-executed from the
tip. A complete first-energy pilot (including every gun aperture and reflected
load) supplies the numerical chart. The optional occupied-wave audit completes
that energy's re-execution and retains both incoming ports. Other energies and
the image chain are not completed; no source checkpoint is published.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from temsim.calculation_manifest import solver_source_identity
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics import radial_gun_wave as radial
from temsim.physics import surface_gun_wave as surface_gun
from temsim.physics.adaptive_scattering import AxialRefinement, scattering_distance
from temsim.physics.magnus_scattering import cf4_slab, GAUSS_FRACTIONS
from temsim.physics.radial_coordinates import RadialCoordinateBlend
from temsim.physics.radial_mask_ledger import physical_to_chart
from temsim.physics.scattering_load import compose, hermitian_slab
from temsim.physics.surface_gun_wave import build_surface_gun_checkpoint
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.wave_following_chart import WaveFollowingNumerics
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement


def refinement_comparison(sample, width, reference_k):
    """Return unaligned complex two-port differences, not a current rescaling.

    sample(f) uses the SAME physical fields and continuous numerical chart at
    action-coordinate fraction f. Both directions and all channels are kept.
    """
    outputs, rows = {}, []
    for method in ("midpoint", "cf4"):
        for divisions in (1, 2, 4, 8, 16):
            result = None
            for index in range(divisions):
                if method == "midpoint":
                    q, g = sample((index+.5)/divisions)
                    piece, _ = hermitian_slab(q, g, width/divisions, reference_k, carrier_k=reference_k)
                else:
                    pair = [sample((index+f)/divisions) for f in GAUSS_FRACTIONS]
                    piece, _ = cf4_slab(*pair, width/divisions, reference_k)
                result = piece if result is None else compose(result, piece)
            outputs[method, divisions] = result
    reference = outputs["cf4", 16]
    for (method, divisions), output in outputs.items():
        row = {"method": method, "subdivisions": divisions,
               "difference_from_cf4_sixteen": scattering_distance(output, reference)}
        if divisions > 1:
            row["change_from_previous"] = scattering_distance(output, outputs[method, divisions//2])
        rows.append(row)
    return rows, outputs


def physical_action_sampler(scope):
    """Re-sample an observed production interval, retaining its frozen guide."""
    index, z, phi = scope["index"], scope["z"], scope["phi"]
    dz, energy = z[index+1]-z[index], scope["energy_ev"]
    nodes, weights = scope["gauss_x"], scope["gauss_w"]
    guide = np.array(((0., scope["drift"]), (scope["strength"], 0.)))
    def sample(action_fraction):
        def action(fraction):
            energies = energy+phi[index]+fraction*(phi[index+1]-phi[index])*(nodes+1)/2
            return fraction*float(np.sqrt(radial.squared_wave_number(energies))@(weights/2))/scope["average_k"][index]
        fraction = brentq(lambda f: action(f)-action_fraction, 0., 1., xtol=1e-14)
        mapping = expm(guide*(fraction*dz))
        def advance(value):
            return (mapping[1, 0]+mapping[1, 1]*value)/(mapping[0, 0]+mapping[0, 1]*value)
        return scope["sample_operator"](fraction, advance(scope["q"]), advance(scope["target_q"]))
    return sample


class AuditComplete(Exception):
    """Intentional diagnostic stop: never a publishable source checkpoint."""


def occupied_difference(first, second, incoming):
    """Relative complex error on the actual two-port input, with no phase fit."""
    def apply(blocks):
        size = len(incoming)//2
        left, right = incoming[:size], incoming[size:]
        return np.r_[blocks[0]@left+blocks[1]@right, blocks[2]@left+blocks[3]@right]
    norm = np.linalg.norm(incoming)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Occupied error requires a finite nonzero incident field")
    return float(np.linalg.norm(apply(first)-apply(second))/norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--occupied-waves", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    archived = json.loads(args.input.read_text(encoding="utf-8"))
    state = InstrumentSnapshot.from_dict(archived["instrument_snapshot"]).restore()
    settings = dict(archived["gun"]["record"]["radial_numerics"])
    settings["axial_refinement"] = AxialRefinement(**settings["axial_refinement"])
    settings["wave_following"] = WaveFollowingNumerics(**settings["wave_following"])
    settings["occupied_refinement"] = OccupiedAxialRefinement(**settings.get("occupied_refinement", {}))
    settings["coordinate_blend"] = (None if settings["coordinate_blend"] is None
                                    else RadialCoordinateBlend(**settings["coordinate_blend"]))
    settings["diagnostic_planes_mm"] = tuple(settings["diagnostic_planes_mm"])
    numerics = radial.RadialGunNumerics(**settings).validate()
    surface = SurfaceWaveNumerics(**archived["gun"]["record"]["surface_numerics"])
    if numerics.axial_integrator != "midpoint" or numerics.wave_following.iterations != 1:
        raise ValueError("This observer requires the one-iteration midpoint chart under investigation")
    started, implementation = perf_counter(), solver_source_identity()
    targets, seen, rows, arrays, steps = (100., 400., 1000., 10000.), set(), [], {}, []
    report = {"scope": "Local operator audit on first energy; NOT a full source or image calculation",
        "input": str(args.input), "input_sha256": sha256(args.input.read_bytes()).hexdigest(),
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation": implementation, "physical_settings_digest": archived["instrument_settings_digest"],
        "occupied_wave_requested": args.occupied_waves,
        "surface_numerics": asdict(surface), "radial_numerics": asdict(numerics)}
    original, original_boundary = radial.hermitian_slab, surface_gun.solve_joint_boundary
    def observed(q, g, width, kappa, *, carrier_k=None):
        original_result = original(q, g, width, kappa, carrier_k=carrier_k)
        scope = inspect.currentframe().f_back.f_locals
        i = scope["index"]
        left, right = scope["z"][i:i+2]
        stage = "pilot" if scope["executed_chart"] is None else "wave_following"
        for target in targets:
            if left <= target < right and (stage, target) not in seen:
                seen.add((stage, target))
                comparisons, outputs = refinement_comparison(physical_action_sampler(scope), width, kappa)
                record = {"stage": stage, "start_nm": float(left), "stop_nm": float(right),
                    "energy_ev": scope["energy_ev"], "action_width_nm": width, "reference_k_per_nm": kappa,
                    "row_index": len(scope["rows"]),
                    "physical_z_midpoint_vs_action_midpoint": scattering_distance(original_result[0], outputs["midpoint", 1]),
                    "comparisons": comparisons}
                rows.append(record)
                steps.append((record, original_result[0], outputs))
                for (method, divisions), blocks in outputs.items():
                    arrays[f"interval_{len(rows)}_{method}_{divisions}"] = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])
                print(json.dumps(record), flush=True)
        if len(seen) == 2*len(targets) and not args.occupied_waves:
            raise AuditComplete()
        # The completed pilot always consumes the original, unchanged step.
        return original_result
    def observed_boundary(problem, energy, load, frame, **kwargs):
        result = original_boundary(problem, energy, load, frame, **kwargs)
        if not args.occupied_waves:
            return result
        fields, derivatives = load.propagate(result[1])
        stage = "pilot" if frame["executed_chart_digest"] is None else "wave_following"
        for record, production, outputs in steps:
            if record["stage"] != stage:
                continue
            index = record["row_index"]
            left, _ = physical_to_chart(fields[index], derivatives[index], load.alpha[index],
                load.log_amplitude_derivative[index], load.kappa)
            _, right = physical_to_chart(fields[index+1], derivatives[index+1], load.alpha[index+1],
                load.log_amplitude_derivative[index+1], load.kappa)
            incoming = np.r_[left, right]
            arrays[f"{stage}_{index}_incoming"] = incoming
            record["occupied_production_vs_cf4_sixteen"] = occupied_difference(production, outputs["cf4", 16], incoming)
            record["occupied_production_vs_action_midpoint"] = occupied_difference(production, outputs["midpoint", 1], incoming)
            for row in record["comparisons"]:
                method, divisions = row["method"], row["subdivisions"]
                row["occupied_difference_from_cf4_sixteen"] = occupied_difference(outputs[method, divisions], outputs["cf4", 16], incoming)
                if divisions > 1:
                    row["occupied_change_from_previous"] = occupied_difference(outputs[method, divisions], outputs[method, divisions//2], incoming)
            print(json.dumps({"stage": stage, "start_nm": record["start_nm"],
                "occupied_production_vs_cf4_sixteen": record["occupied_production_vs_cf4_sixteen"],
                "occupied_cf4_eight_vs_sixteen": record["comparisons"][-1]["occupied_change_from_previous"]}), flush=True)
        if stage == "wave_following":
            if len(seen) != 2*len(targets):
                raise ValueError("The occupied audit did not observe every requested interval")
            raise AuditComplete()
        return result
    radial.hermitian_slab = observed
    surface_gun.solve_joint_boundary = observed_boundary
    try:
        with threadpool_limits(1):
            build_surface_gun_checkpoint(state.electron_gun, surface=surface, radial=numerics,
                grid_pixels=2048, column_state=state, cancelled=lambda: perf_counter()-started > args.timeout)
        report["status"] = "FAILED_MISSING_EXPECTED_INTERVALS"
    except AuditComplete:
        report["status"] = "LOCAL_OPERATOR_AUDIT_COMPLETE_NOT_SOURCE"
    except Exception:
        report["status"] = "FAILED"
        report["traceback"] = traceback.format_exc()
    finally:
        radial.hermitian_slab = original
        surface_gun.solve_joint_boundary = original_boundary
    report.update(intervals=rows, elapsed_s=perf_counter()-started,
                  implementation_unchanged=implementation == solver_source_identity())
    np.savez(args.output/"operators.npz", **arrays)
    with (args.output/"report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("status", "elapsed_s", "implementation_unchanged")}), flush=True)
    return int(report["status"] != "LOCAL_OPERATOR_AUDIT_COMPLETE_NOT_SOURCE")


if __name__ == "__main__":
    raise SystemExit(main())
