"""Compare executed gun evidence; no source reconstruction or fitted phase."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(first, second, samples=16385):
    if type(samples) is not int or samples < 3:
        raise ValueError("Complex radial comparison needs at least three samples")
    a, b = (json.loads(Path(name).read_text(encoding="utf-8")) for name in (first, second))
    same_snapshot = a.get("instrument_digest") and a["instrument_digest"] == b.get("instrument_digest")
    same_settings = a.get("instrument_settings_digest") and a["instrument_settings_digest"] == b.get("instrument_settings_digest")
    if not (same_snapshot or same_settings):
        raise ValueError("Compare the same physical source, electrodes and column; instrument identities differ")
    ga, gb = a["gun"], b["gun"]
    lengths = [len(g[key]) for g in (ga, gb) for key in ("mode_records", "radial_output_modes")]
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError("Compare every energy mode; missing or unmatched mode sets are not comparable")
    # These older diagnostic files factor a near-tip axial action without
    # storing it separately. It cancels ONLY for the same grounded field,
    # emission energy and axial reference interval. Refuse other comparisons.
    with np.load(Path(first).parent/"complex_states.npz") as aa, np.load(Path(second).parent/"complex_states.npz") as bb:
        if not np.array_equal(aa["near_z_nm"][0, [0, -1]], bb["near_z_nm"][0, [0, -1]]):
            raise ValueError("Near-tip axial phase references differ; the omitted near carrier cannot be cancelled")
    rows, norm, error = [], 0., 0.
    for ma, mb, pa, pb in zip(ga["mode_records"], gb["mode_records"], ga["radial_output_modes"], gb["radial_output_modes"]):
        if ma["energy_ev"] != mb["energy_ev"] or pa["mode_id"] != pb["mode_id"]:
            raise ValueError("Compare matching physical energy modes, not a phase of their mixture")
        if ma["mixture_weight"] != mb["mixture_weight"]:
            raise ValueError("Energy mixture weights changed; this is not a numerical convergence comparison")
        ca = np.array(pa["coefficients_real"])+1j*np.array(pa["coefficients_imag"])
        cb = np.array(pb["coefficients_real"])+1j*np.array(pb["coefficients_imag"])
        width = max(pa["width_nm"], pb["width_nm"])
        r = np.linspace(0., width*np.sqrt(4*max(len(ca), len(cb))+160), samples)
        def field(c, p, m):
            from temsim.physics.quartic_radial_phase import radial_envelope
            value = radial_envelope(r, p["width_nm"], c)
            # Both fields keep their own executed carrier. Do not fit a
            # curvature, relative phase, beam centre or intensity scale.
            from temsim.physics.radial_gun_wave import squared_wave_number
            k = np.sqrt(squared_wave_number(m["far_field"]["exit_energy_ev"]))
            carrier = m["far_field"]["reference_carrier_phase_mod_rad"]
            return value*np.exp(1j*carrier+.5j*k*p["curvature_m1"]*1e-9*r*r
                +1j*p.get("quartic_phase_per_nm4", 0.)*r**4)*np.sqrt(m["mixture_weight"]*m["exit_fraction"])
        fa, fb = field(ca, pa, ma), field(cb, pb, mb)
        local_norm = np.trapezoid(2*np.pi*r*abs(fb)**2, r)
        local_error = np.trapezoid(2*np.pi*r*abs(fa-fb)**2, r)
        if not np.isfinite(local_norm+local_error) or local_norm <= 0:
            raise ValueError("Complex comparison requires a finite nonzero reference mode")
        rows.append({"energy_ev": ma["energy_ev"], "relative_complex_l2": float(np.sqrt(local_error/local_norm))})
        norm += local_norm; error += local_error
    return {"first": str(first), "second": str(second), "mode_comparisons": rows,
        "current_a": [ga["current_a"], gb["current_a"]],
        "relative_current_change": abs(ga["current_a"]-gb["current_a"])/gb["current_a"],
        "relative_complex_l2": float(np.sqrt(error/norm)),
        "radial_comparison_samples": samples,
        "metric": "Cylindrical complex L2 with recorded far-gun carrier restored; matching energy and near-reference only; no phase/curvature/current fit",
        "scope": "GUN_COMPARISON_NOT_TEM_STEM_ACCEPTANCE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(compare(args.first, args.second), indent=2)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload)
