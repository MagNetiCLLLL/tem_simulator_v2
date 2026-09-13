"""Attempt physical-tip Si[110] readout without retuning source/optical inputs.

This repeatable development run selects the already defined tip-coherence
model explicitly, but retains the existing emitter values and energy law.
It saves a request/failure receipt, not a replacement source or a fake image.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence, tip_energy_samples, wavelength_m
from temsim.profile_io import save_profile
from temsim.specimen.scene import SpecimenScene
from trace_tip_wave import main as trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence directory")
    state = default_state()
    gun = state.electron_gun
    original = asdict(gun.emitter)
    gun.emitter.coherence = TipCoherence()
    assert asdict(gun.emitter) == original
    selection = AssemblyCatalog().selection_for_resolved(state._resolved_assembly)
    args.output.mkdir(parents=True, exist_ok=False)
    profile = args.output/"input.toml"
    save_profile(profile, state, selection)
    energies = tip_energy_samples(gun.emitter, 9)
    sigma = original["virtual_source_fwhm_nm"]*1e-9/np.sqrt(8*np.log(2))
    angles = wavelength_m(energies)/(4*np.pi*sigma)
    z_mm = np.array([0., 1e-6, .01, .05, .1, .5])
    points = np.zeros((len(z_mm), 3))
    points[:, 2] = z_mm*1e-3
    scene = SpecimenScene.from_state(state)
    probe = {
        "schema": "default-physical-tip-attempt-input-v1",
        "implementation": solver_source_identity(), "emitter": original,
        "explicit_model_selection": asdict(gun.emitter.coherence),
        "energy_samples_ev": energies.tolist(),
        "gaussian_closed_spectrum_weight": np.exp(-.5/angles**2).tolist(),
        "spectrum_note": "Boundary-density spectral weight, NOT emitted electron current or discarded probability",
        "field_sample_z_mm": z_mm.tolist(),
        "actual_gun_potential_v": gun.electric_field.potential_v_at_global_positions(points).tolist(),
        "actual_gun_electric_field_v_per_m": gun.electric_field.field_at_global_positions_v_per_m(points).tolist(),
        "requested_detector": "haadf",
        "detector_reason": "First inserted STEM detector, before the energy filter; no detector moved or removed",
        "sample": {"preset": scene.preset_key, "thickness_nm": scene.thickness_nm},
        "status": "INPUT_AUDIT_ONLY; see result/failure.json or result/receipt.json for execution",
    }
    (args.output/"input_audit.json").write_text(json.dumps(probe, indent=2)+"\n", encoding="utf-8")
    return trace(["--profile", str(profile), "--output", str(args.output/"result"),
                  "--detector", "haadf", "--stop", "detector"])


if __name__ == "__main__":
    raise SystemExit(main())
