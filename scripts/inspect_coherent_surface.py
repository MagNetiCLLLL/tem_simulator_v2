"""Actual assembled reference near-field refinements; not TEM/STEM acceptance.

Writes new evidence only. Explicitly selects the new coherent reservoir;
does not edit the default TOML, an open application, or historical profiles.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.tip_wave_pipeline import simulate_tip_wave, TipWaveRequest
from temsim.physics.surface_wave_export import export_surface_wave, draw_surface_wave
from temsim.profile_io import save_profile


def observations(result):
    """Fixed physical observation patch, no fitted phase or beam dimensions."""
    r = np.linspace(0., 15., 81)
    z = np.linspace(0., 1., 41)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    radius = result.record["source"]["geometry"]["apex_radius_nm"]
    bottom = -rr**2/(radius+np.sqrt(radius**2-rr**2))
    eta = (zz-bottom)/(result.z_nm[0, -1]-bottom)
    p = np.column_stack((rr.ravel(), eta.ravel()))
    fields = np.array([RegularGridInterpolator((result.radius_nm,
        np.linspace(0, 1, result.z_nm.shape[1])), mode.amplitude)(p) for mode in result.modes])
    density = sum(mode.weight*abs(field)**2 for mode, field in zip(result.modes, fields))
    return fields, density, rr.ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = default_state()
    model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    state.electron_gun.emitter.surface_model = model
    base = SurfaceWaveNumerics()
    field = model.field_numerics
    # Numerical variations only. Source physics, voltage and assembly fixed.
    cases = [("wave_49", replace(base, radial_nodes=49, axial_nodes=97), field),
             ("wave_97", base, field),
             ("wave_193", replace(base, radial_nodes=193, axial_nodes=385), field),
             ("radial_domain", replace(base, radial_nodes=257, axial_nodes=513, outer_radius_factor=2.), field),
             ("axial_domain", replace(base, radial_nodes=193, axial_nodes=513, exit_height_nm=4.), field),
             ("field_grid", replace(base, radial_nodes=193, axial_nodes=385),
                replace(field, radial_nodes=320, axial_nodes=640, apex_cells_per_radius=40)),
             ("energy_5", replace(base, radial_nodes=193, axial_nodes=385, energy_samples=5), field)]
    implementation = solver_source_identity()
    rows, measured = [], {}
    for name, numerics, field_numerics in cases:
        state.electron_gun.emitter.surface_model = replace(model, field_numerics=field_numerics)
        start = perf_counter()
        calculation = simulate_tip_wave(state, TipWaveRequest(stop="tip_near_field", surface=numerics), use_cache=False)
        elapsed = perf_counter()-start
        result = calculation.checkpoint
        export_surface_wave(calculation, args.output/f"{name}.npz")
        save_profile(args.output/f"{name}.toml", state, AssemblyCatalog().default_selection())
        measured[name] = observations(result)
        row = {"name": name, "elapsed_s": elapsed, "wave_numerics": asdict(numerics),
            "field_numerics": asdict(field_numerics), "digest": result.digest,
            "flux": {key: sum(m.weight*m.flux[key] for m in result.modes) for key in ("reflected", "top", "side")},
            "maximum_flux_balance_error": max(m.flux["balance_error"] for m in result.modes)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if name == "wave_193":
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(12, 5), dpi=150); FigureCanvasAgg(figure)
            draw_surface_wave(figure, result, 1)
            figure.savefig(args.output/"coherent_tip_near_field.png")
        del calculation, result
    comparisons = []
    for a, b in (("wave_49", "wave_97"), ("wave_97", "wave_193"),
            ("wave_193", "radial_domain"), ("wave_193", "axial_domain"),
            ("wave_193", "field_grid"), ("wave_193", "energy_5")):
        wa, da, r = measured[a]
        wb, db, _ = measured[b]
        density_change = float(np.sqrt(np.sum(r*(db-da)**2)/np.sum(r*db**2)))
        complex_change = (float(np.sqrt(np.sum(r*abs(wb-wa)**2)/np.sum(r*abs(wb)**2)))
                          if wa.shape == wb.shape else None)
        comparisons.append({"from": a, "to": b, "relative_density_l2_change": density_change,
            "relative_complex_l2_change_no_phase_fit": complex_change,
            "below_one_percent": density_change < .01 and (complex_change is None or complex_change < .01)})
    report = {"scope": "EXECUTED_GROUNDED_COHERENT_TIP_NEAR_FIELD_ONLY", "full_image": "NOT_CONNECTED",
        "implementation": implementation, "implementation_unchanged": solver_source_identity() == implementation,
        "cases": rows, "comparisons": comparisons,
        "metric": "Fixed 0<=r<=15 nm, 0<=z<=1 nm; cylindrical weighted L2; no global phase fit; interpolated observation grid",
        "qualification": "Flux balance is not mesh/field/energy/Robin-domain convergence. No full-image qualification."}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"comparisons": comparisons}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
