"""Inspect/export executed near fields; never import them as a new source."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from temsim.immutable_json import thaw_json


def export_surface_wave(calculation, path):
    """Exclusive-create a pickle-free, self-describing evidence archive."""
    if calculation.request.stop != "tip_near_field":
        raise ValueError("A coherent tip near-field result is required")
    result = calculation.checkpoint
    receipt = {"schema": "surface-wave-export-v1", "scope": "TIP_NEAR_FIELD_ONLY_NOT_TEM_STEM",
        "status": "COMPUTED_DEVELOPMENT_SCOPE", "full_tem_stem": "NOT_CONNECTED_NOT_VALIDATED",
        "checkpoint_digest": result.digest, "instrument_digest": calculation.instrument_digest,
        "request": asdict(calculation.request), "reference_current_a": result.reference_current_a,
        "top_current_a": result.outgoing_current_a, "execution": thaw_json(result.record),
        "modes": [{"energy_ev": m.energy_ev, "weight": m.weight, "flux": thaw_json(m.flux),
                   "phase_reference": m.phase_reference} for m in result.modes],
        "usage": "Read-only numerical evidence; not a configurable downstream source"}
    arrays = {"radius_nm": result.radius_nm, "z_nm": result.z_nm,
              "potential_rise_v": result.potential_rise_v,
              "receipt_json": np.array(json.dumps(receipt, ensure_ascii=False))}
    arrays.update({f"mode_{i}_amplitude": mode.amplitude for i, mode in enumerate(result.modes)})
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return receipt


def draw_surface_wave(figure, result, mode_index=0):
    """Density is an incoherent energy sum; phase belongs to ONE energy mode."""
    from matplotlib.colors import LogNorm
    figure.clear()
    density = result.density
    mode = result.modes[mode_index]
    r = np.broadcast_to(result.radius_nm[:, None], result.z_nm.shape)
    # Shared triangular connectivity follows the actual curved FEM mesh.
    ids = np.arange(r.size).reshape(r.shape)
    a, b, c, d = [v.ravel() for v in (ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:])]
    triangles = np.vstack((np.column_stack((a,b,c)), np.column_stack((a,c,d))))
    positive = density[density > 0]
    upper = float(positive.max()) if positive.size else 1.
    panels = figure.subplots(1, 2)
    im = panels[0].tripcolor(r.ravel(), result.z_nm.ravel(), triangles, density.ravel(),
        shading="gouraud", norm=LogNorm(vmin=upper*1e-6, vmax=upper), cmap="magma")
    figure.colorbar(im, ax=panels[0], label="Sum of weighted |psi|² (flux units)")
    panels[0].set_title("Coherent tip near field — energy mixture")
    # Facewise phase avoids interpolating across the -pi/pi discontinuity.
    psi = mode.amplitude.ravel()
    face = psi[triangles].mean(axis=1)
    phase = np.ma.masked_where(abs(face)**2 < abs(psi).max()**2*1e-8, np.angle(face))
    im = panels[1].tripcolor(r.ravel(), result.z_nm.ravel(), triangles, facecolors=phase,
        cmap="twilight", vmin=-np.pi, vmax=np.pi)
    figure.colorbar(im, ax=panels[1], label="Phase (rad), one energy component")
    panels[1].set_title(f"{mode.energy_ev:.5g} eV | weight {mode.weight:.4g}")
    for ax in panels:
        ax.plot(result.radius_nm, result.z_nm[:, 0], color="cyan", lw=1., label="Tip surface")
        ax.set_xlabel("Radius (nm)")
        ax.set_ylabel("Z from tip apex (nm), +Z downstream")
        ax.invert_yaxis()
        ax.set_aspect("equal")
    figure.suptitle("Executed near-field segment; full gun and TEM/STEM not connected", fontsize=10)
    figure.tight_layout()
