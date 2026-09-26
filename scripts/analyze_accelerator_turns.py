"""Read saved source-origin rays; distinguish transverse turning from backstreaming.

No particle integration or field solve is performed. Cached fields are read and
checked against their archived probes. Output is generated diagnostic data.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from temsim.cpu_resources import numerical_job
from scripts.diagnose_planar_gun import CHARGE, MASS, LIGHT, array_digest


def sign_changes(z, values, deadband=0.):
    """Opposite resolved signs; exact zeros/deadband do not add events.

    Return interpolated zero estimates plus bracketing stored samples. These
    are observation estimates, not newly integrated physical event states.
    """
    z, values = np.asarray(z, float), np.asarray(values, float)
    if (z.ndim != 1 or values.shape != z.shape or len(z) < 2
            or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
            or not np.isfinite(deadband) or deadband < 0):
        raise ValueError("Invalid sign-event samples")
    ids = np.flatnonzero(np.isfinite(values) & (abs(values) > deadband))
    events = []
    opposite = np.flatnonzero(np.signbit(values[ids[:-1]]) != np.signbit(values[ids[1:]]))
    for pair in opposite:
        left, right = ids[pair:pair+2]
        fraction = abs(values[left])/(abs(values[left])+abs(values[right]))
        events.append({"z_m": float(z[left]+fraction*(z[right]-z[left])),
                       "bracket_m": [float(z[left]), float(z[right])],
                       "before": int(np.sign(values[left])),
                       "after": int(np.sign(values[right]))})
    return events


def _validate(z, state, weights):
    z, state, weights = np.asarray(z, float), np.asarray(state, float), np.asarray(weights, float)
    if (z.ndim != 1 or len(z) < 2 or state.ndim != 3
            or state.shape[0] != len(z) or state.shape[2] != 6
            or weights.shape != (state.shape[1],) or not len(weights)
            or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
            or not np.isfinite(state).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Expected increasing z, finite (plane, particle, 6) state and nonnegative weights")
    return z, state, weights/weights.sum()


def _count_summary(counts, weights):
    counts = np.asarray(counts, int)
    return {"min": int(counts.min()), "median": float(np.median(counts)),
            "max": int(counts.max()), "weighted_mean": float(weights@counts),
            "weighted_fraction_with_event": float(weights@(counts > 0)),
            "per_particle": counts.tolist()}


def _channel(z, values, weights, deadband):
    return {"raw": _count_summary([len(sign_changes(z, values[:, i]))
                                   for i in range(len(weights))], weights),
            "resolved": _count_summary([len(sign_changes(z, values[:, i], deadband))
                                        for i in range(len(weights))], weights)}


def trajectory_metrics(z, state, weights, *, interval_m=(.03, .37),
                       slope_deadband_rad=1e-10, position_deadband_m=1e-12):
    """Pure saved-ray metrics: state columns x,y,px/mc,py/mc,pz/mc,time.

    X/Y turns use slopes. Radial extrema use dr/dz and do not imply momentum
    reversal along Z. Counts describe stored observations; deadbands are
    declared resolution thresholds, not assertions of physical insignificance.
    """
    z, state, weights = _validate(z, state, weights)
    low, high = map(float, interval_m)
    if not z[0] <= low < high <= z[-1]:
        raise ValueError("Analysis interval is outside saved source-origin coverage")
    inside = (z >= low) & (z <= high)
    zz, ss = z[inside], state[inside]
    if len(zz) < 2:
        raise ValueError("Analysis interval contains too few saved planes")
    uz = ss[..., 4]
    slope = np.divide(ss[..., 2:4], uz[..., None],
                      out=np.full_like(ss[..., 2:4], np.nan), where=uz[..., None] != 0)
    radius = np.linalg.norm(ss[..., :2], axis=-1)
    radial_rate = np.divide(np.sum(ss[..., :2]*slope, axis=-1), radius,
                            out=np.full_like(radius, np.nan), where=radius > position_deadband_m)
    gamma = np.sqrt(1+np.sum(ss[..., 2:5]**2, axis=-1))
    channels = {
        "x_slope_turns": _channel(zz, slope[..., 0], weights, slope_deadband_rad),
        "y_slope_turns": _channel(zz, slope[..., 1], weights, slope_deadband_rad),
        "radial_extrema": _channel(zz, radial_rate, weights, slope_deadband_rad),
        "x_axis_crossings": _channel(zz, ss[..., 0], weights, position_deadband_m),
        "y_axis_crossings": _channel(zz, ss[..., 1], weights, position_deadband_m),
    }
    return {"interval_m": [low, high], "sampled_interval_m": [float(zz[0]), float(zz[-1])],
            "particles": len(weights), "sampled_planes": len(zz),
            "thresholds": {"slope_deadband_rad": slope_deadband_rad,
                           "position_deadband_m": position_deadband_m},
            "longitudinal": {"all_sampled_forward": bool(np.all(uz > 0)),
                "nonforward_particle_count": int(np.count_nonzero(np.any(uz <= 0, axis=0))),
                "minimum_pz_over_mc": float(uz.min()),
                "minimum_vz_m_per_s": float(np.min(LIGHT*uz/gamma))},
            "peak_radius_um": float(radius.max()*1e6),
            "peak_transverse_slope_rad": float(np.nanmax(np.linalg.norm(slope, axis=-1))),
            **channels}


def curvature_components(state, electric):
    """Actual d²(x,y)/dz², separating transverse force and axial acceleration.

    Electric is SI V/m. A change in slope is not by itself evidence of a
    transverse force: increasing pz reduces slope even when px,py are fixed.
    """
    state, electric = np.asarray(state, float), np.asarray(electric, float)
    if state.shape[-1] != 6 or electric.shape != state.shape[:-1]+(3,):
        raise ValueError("State/electric field shapes differ")
    if np.any(state[..., 4] <= 0) or not np.isfinite(electric).all():
        raise ValueError("Curvature requires finite fields and forward momentum")
    gamma = np.sqrt(1+np.sum(state[..., 2:5]**2, axis=-1))
    factor = CHARGE*gamma/(MASS*LIGHT**2*state[..., 4]**2)
    slopes = state[..., 2:4]/state[..., 4, None]
    transverse = -factor[..., None]*electric[..., :2]
    acceleration = factor[..., None]*slopes*electric[..., 2, None]
    return transverse, acceleration


def force_metrics(z, state, electric, weights, *, interval_m=(.03, .37),
                  radial_force_over_e_deadband_v_per_m=1e-4,
                  position_deadband_m=1e-12, curvature_deadband_per_m=1e-10):
    """Field direction, momentum turning and geometric curvature stay separate.

    Signed force is outward radial projection F_r/e = -E dot r_hat; X near
    zero does not define its sign. Axis points have no radial direction.
    """
    z, state, weights = _validate(z, state, weights)
    electric = np.asarray(electric, float)
    if electric.shape != state.shape[:-1]+(3,) or not np.isfinite(electric).all():
        raise ValueError("Field samples must match the saved particle states")
    inside = (z >= interval_m[0]) & (z <= interval_m[1])
    zz, ss, ee = z[inside], state[inside], electric[inside]
    radius = np.linalg.norm(ss[..., :2], axis=-1)
    radial = np.divide(-np.sum(ss[..., :2]*ee[..., :2], axis=-1), radius,
                       out=np.full_like(radius, np.nan), where=radius > position_deadband_m)
    transverse, acceleration = curvature_components(ss, ee)
    total = transverse+acceleration
    return {"radial_force_definition": "Outward F_r/e = -E dot r_hat; positive defocusing, negative focusing",
            "radial_direction_min_radius_m": position_deadband_m,
            "force_over_e_deadband_v_per_m": radial_force_over_e_deadband_v_per_m,
            "peak_transverse_force_n": float(CHARGE*np.max(np.linalg.norm(ee[..., :2], axis=-1))),
            "peak_abs_radial_force_over_e_v_per_m": float(np.nanmax(abs(radial))),
            "peak_abs_axial_field_v_per_m": float(np.max(abs(ee[..., 2]))),
            "radial_force_direction_reversals": _channel(zz, radial, weights, radial_force_over_e_deadband_v_per_m),
            "x_curvature_sign_changes": _channel(zz, total[..., 0], weights, curvature_deadband_per_m),
            "peak_transverse_force_curvature_per_m": float(np.max(np.linalg.norm(transverse, axis=-1))),
            "peak_axial_acceleration_curvature_per_m": float(np.max(np.linalg.norm(acceleration, axis=-1))),
            "peak_total_curvature_per_m": float(np.max(np.linalg.norm(total, axis=-1)))}


def _read_archive(path, report):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data["input_snapshot_digest"]) != report["input_snapshot_digest"]:
        raise ValueError("Trajectory snapshot identity differs from the report")
    if array_digest(state=data["state"][0], ray_id=data["ray_id"], weight=data["weights"]) != report["emission_sha256"]:
        raise ValueError("Archive does not begin with the declared original emitted state")
    _validate(data["z_m"], data["state"], data["weights"])
    if data["z_m"][0] != 0.:
        raise ValueError("A downstream state cannot stand in for tip-origin rays")
    return data


def _archived_analytic_field(snapshot):
    """Read a narrow archived field description, never restore an active gun.

    Historical state remains read-only. Only scalar electrostatic parameters
    and stage records are reconstructed; there is no emitter or transport API.
    """
    from temsim.optics.electron_gun.electrostatic import (
        ExtractorElectrode, ElectrostaticGunLens, AcceleratorColumn,
        AcceleratorStage, FegElectrostaticField,
    )
    graph = snapshot.graph
    nodes = graph["nodes"]
    root = nodes[graph["root"]["ref"]]["attributes"]
    gun = nodes[root["electron_gun"]["ref"]]["attributes"]

    def attributes(reference, expected):
        node = nodes[reference["ref"]]
        if node["type"] != expected.__module__+":"+expected.__name__:
            raise ValueError("Unexpected archived electrostatic component")
        return node["attributes"]

    def scalars(row, cls, exclude=()):
        result = {}
        for field in fields(cls):
            if field.name in exclude:
                continue
            value = row[field.name]
            if not isinstance(value, (int, float, str, bool)):
                raise ValueError("Archived electrostatic parameter is not a scalar")
            result[field.name] = value
        return result

    emitter = nodes[gun["emitter"]["ref"]]
    if (emitter["type"] != "temsim.optics.electron_gun.emitter:ColdFieldEmitter"
            or emitter["attributes"].get("_surface_model") is not None
            or emitter["attributes"].get("_tip_curvature_nm_inv", 0.) != 0.):
        raise ValueError("This historical diagnostic must begin with its original flat emitter")
    extraction = ExtractorElectrode(**scalars(attributes(gun["extractor"], ExtractorElectrode), ExtractorElectrode))
    lens = ElectrostaticGunLens(**scalars(attributes(gun["electrostatic_lens"], ElectrostaticGunLens), ElectrostaticGunLens))
    row = attributes(gun["accelerator"], AcceleratorColumn)
    stages = [AcceleratorStage(**scalars(attributes(ref, AcceleratorStage), AcceleratorStage))
              for ref in row["stages"]["list"]]
    accelerator = AcceleratorColumn(**scalars(row, AcceleratorColumn, ("stages",)), stages=stages)
    for component in (extraction, lens, accelerator):
        component.validate()
    field = FegElectrostaticField(SimpleNamespace(emission_energy_ev=float(emitter["attributes"]["emission_energy_ev"])),
                                 extraction, lens, accelerator)
    field.compiled_axial = False
    return field


def analyze(directory, output, extra_directory=None):
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.physics.planar_gun_field import load_cached_field

    directory, output = Path(directory).resolve(), Path(output).resolve()
    source_report = json.loads((directory/"comparison.json").read_text(encoding="utf-8"))
    snapshot = InstrumentSnapshot.from_dict(json.loads((directory/"instrument-input-snapshot.json").read_text(encoding="utf-8")))
    # An unrelated newly added solver changes the global runtime identity.
    # Historical reading must not activate that archived profile. Verify the
    # actually consumed analytic implementations when narrower hashes exist.
    extra_report = None
    if extra_directory is not None:
        extra_directory = Path(extra_directory).resolve()
        extra_report = json.loads((extra_directory/"ramps-comparison.json").read_text(encoding="utf-8"))
        pinned = next(iter(extra_report["runs"].values()))["field"]["sha256"]
        for dependency, expected in pinned.items():
            if hashlib.sha256((Path(__file__).resolve().parents[1]/dependency).read_bytes()).hexdigest() != expected:
                raise ValueError("Consumed analytic implementation changed; archived metrics only")
    else:
        from temsim.calculation_manifest import solver_source_identity
        if snapshot.implementation != solver_source_identity():
            raise ValueError("Historical field evaluation requires pinned consumed implementation hashes")
    analytic = _archived_analytic_field(snapshot)
    report = {"scope": "Read-only analysis of saved full original emission trajectories; no new particle integration or field solve",
              "cpu_threads": 1, "source_report": str(directory/"comparison.json"),
              "input_snapshot_digest": snapshot.digest, "cases": {},
              "profile_restoration": "No active profile restored; only archived scalar field parameters read",
              "limits": ["Events are estimated between recorded samples; sub-sample turns may be missed.",
                         "Deadbands distinguish resolved signs; they do not declare all smaller effects unphysical.",
                         "Planar cathode and outlet-boundary limitations of the originating diagnostic remain.",
                         "A transverse momentum turn, axis crossing, force reversal and longitudinal backstreaming are distinct."]}
    display = {}
    stages = [dict(center_mm=s.center_from_tip_mm, edge_mm=s.soft_edge_mm,
                   rise_fraction=s.voltage_fraction) for s in analytic.accelerator.stages]
    report["analytic_accelerator_stages"] = stages
    cases = [(name, directory, source_report) for name in ("analytic", "planar16", "planar32")]
    if extra_directory is not None:
        if (extra_report["input_snapshot_digest"] != source_report["input_snapshot_digest"]
                or extra_report["emission_sha256"] != source_report["emission_sha256"]):
            raise ValueError("Width experiment source inputs differ from the baseline")
        report["width_experiment_source_report"] = str(extra_directory/"ramps-comparison.json")
        cases += [(name, extra_directory, extra_report) for name in extra_report["runs"]]
    for name, case_directory, case_report in cases:
        path = case_directory/f"{name}.npz"
        expected_archive_digest = case_report["runs"][name].get("archive_sha256")
        if expected_archive_digest and hashlib.sha256(path.read_bytes()).hexdigest() != expected_archive_digest:
            raise ValueError("Width experiment archive checksum mismatch")
        data = _read_archive(path, case_report)
        if name == "analytic" or name.startswith("analytic_width"):
            field = copy.deepcopy(analytic)
            if name.startswith("analytic_width"):
                description = case_report["runs"][name]["field"]
                for dependency, expected in description["sha256"].items():
                    if hashlib.sha256((Path(__file__).resolve().parents[1]/dependency).read_bytes()).hexdigest() != expected:
                        raise ValueError("Width experiment analytic field implementation changed")
                for stage in field.accelerator.stages:
                    stage.soft_edge_mm = description["stage_soft_edge_half_width_mm"]
        else:
            cache_file = Path(case_report["runs"][name]["field"]["cache_path"])
            manifest = json.loads(cache_file.with_suffix(".json").read_text(encoding="utf-8"))
            field = load_cached_field(manifest["request"], cache_file.parent)
            if field is None:
                raise ValueError("Required executed field cache missing; analysis will not solve it")
        checked = field.field_at_global_positions_v_per_m(data["probe_m"])
        np.testing.assert_allclose(checked, data["electric_v_per_m"], rtol=2e-12, atol=1e-7)
        z, state, weights = data["z_m"], data["state"], data["weights"]
        points = np.concatenate((state[..., :2], np.broadcast_to(z[:, None, None], state.shape[:-1]+(1,))), axis=-1)
        electric = field.field_at_global_positions_v_per_m(points)
        row = {"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "source_emission_verified": True, "intervals": {}}
        for key, interval in (("accelerator", (.03, .37)), ("interior", (.08, .34))):
            metrics = trajectory_metrics(z, state, weights, interval_m=interval)
            metrics["forces"] = force_metrics(z, state, electric, weights, interval_m=interval)
            pm = (data["probe_m"][:, 2] >= interval[0]) & (data["probe_m"][:, 2] <= interval[1])
            ee = data["electric_v_per_m"][pm]
            metrics["saved_r_1um_probe"] = {
                "radial_field_v_per_m_min": float(ee[:, 0].min()),
                "radial_field_v_per_m_max": float(ee[:, 0].max()),
                "axial_field_v_per_m_min": float(ee[:, 2].min()),
                "axial_field_v_per_m_max": float(ee[:, 2].max()),
                "zero_axial_field_sample_fraction": float(np.mean(ee[:, 2] == 0.))}
            if name == "analytic" or name.startswith("analytic_width"):
                _, d1, d2, d3 = field.axial_potential_v_and_derivatives_per_mm(z[(z >= interval[0]) & (z <= interval[1])]*1000)
                metrics["axis_potential_derivatives"] = {
                    "peak_abs_dphi_dz_v_per_m": float(np.max(abs(d1))*1e3),
                    "peak_abs_d2phi_dz2_v_per_m2": float(np.max(abs(d2))*1e6),
                    "peak_abs_d3phi_dz3_v_per_m3": float(np.max(abs(d3))*1e9),
                    "radial_expansion": "Er = r/2 * Phi_axis_second_derivative; Fr = -e Er"}
            row["intervals"][key] = metrics
        if name == "analytic":
            mask = (z >= .03) & (z <= .37)
            chosen = int(np.argmax(np.max(np.linalg.norm(state[mask, :, :2], axis=-1), axis=0)))
            report["selected_particle_index"] = chosen
            report["selected_ray_id"] = int(data["ray_id"][chosen])
            report["selection_rule"] = "Largest archived radial displacement in analytic accelerator; same ray ID compared in every case"
        if int(data["ray_id"][chosen]) != report["selected_ray_id"]:
            raise ValueError("Ray ordering changed between archives")
        mask = (z >= .03) & (z <= .37)
        zz, ss, ee = z[mask], state[mask, chosen], electric[mask, chosen]
        row["selected_ray"] = {
            "x_slope_turns": sign_changes(zz, ss[:, 2]/ss[:, 4], 1e-10),
            "x_axis_crossings": sign_changes(zz, ss[:, 0], 1e-12),
            "max_radius_um": float(np.max(np.linalg.norm(ss[:, :2], axis=-1))*1e6)}
        if name.startswith("analytic_width"):
            row["analytic_stage_half_width_mm"] = case_report["runs"][name]["field"]["stage_soft_edge_half_width_mm"]
        display[name] = (zz, ss, ee)
        report["cases"][name] = row
        print(name, {key: row["intervals"]["accelerator"][key]["resolved"]["weighted_mean"]
                     for key in ("x_slope_turns", "radial_extrema")}, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output/"turn-analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _render({name: values for name, values in display.items() if not name.startswith("analytic_width")},
            stages, output, report["selected_ray_id"])
    widths = {name: values for name, values in display.items() if name.startswith("analytic_width")}
    if widths:
        _render(widths, stages, output, report["selected_ray_id"], prefix="ramp-width-")
    return report


def _render(display, stages, output, ray_id, prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for name, (z, state, electric) in display.items():
        radial = np.linalg.norm(state[:, :2], axis=1)
        radial_force_over_e = -np.sum(state[:, :2]*electric[:, :2], axis=1)/radial
        transverse, acceleration = curvature_components(state, electric)
        axes[0].plot(z*1000, state[:, 0]*1e6, label=name)
        axes[1].plot(z*1000, state[:, 2]/state[:, 4]*1e6, label=name)
        axes[2].plot(z*1000, radial_force_over_e, label=name)
        axes[3].plot(z*1000, (transverse+acceleration)[:, 0], label=name)
    for ax, label in zip(axes, ("X position (um)", "dX/dZ (urad)",
                               "Outward radial force / e (V/m)", "d2X/dZ2 (1/m)")):
        ax.set_ylabel(label)
        ax.axhline(0, color="black", lw=.6)
        for stage in stages:
            ax.axvline(stage["center_mm"], color="grey", alpha=.25, lw=.6)
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("Z (mm)")
    fig.suptitle(f"Saved ray {ray_id}: position, momentum, force and curvature are different quantities\nPhysical values shown; panels use different axis units/scales")
    fig.savefig(output/(prefix+"accelerator-turn-components.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 2.8), constrained_layout=True)
    for name, (z, state, _electric) in display.items():
        ax.plot(z*1000, state[:, 0]*1000, label=name)
    ax.set(xlabel="Z (mm)", ylabel="X (mm)", xlim=(30, 370), ylim=(-20, 20))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Equal physical spatial scale: 1 mm horizontal = 1 mm vertical\nMicrometre transverse excursions are nearly flat at the full accelerator scale")
    fig.savefig(output/(prefix+"accelerator-equal-physical-scale.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tmp/planar-gun-20260926"))
    parser.add_argument("--output", type=Path, default=Path("tmp/accelerator-turns-20260926"))
    parser.add_argument("--extra-input", type=Path,
                        default=Path("tmp/accelerator-mechanism-20260926"),
                        help="Include saved ramp-width cases if this directory exists")
    args = parser.parse_args()
    with numerical_job(requested=1):
        analyze(args.input, args.output, args.extra_input if args.extra_input.exists() else None)


if __name__ == "__main__":
    main()
