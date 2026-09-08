"""Explain reported STEM levels with a labelled counting fixture, not an image simulation."""
from pathlib import Path
import json
import sys
from threading import Event
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from temsim.gui.calculation_request import CapturedCalculationRequest
from temsim.optics.column import default_state
from temsim.specimen.source import specimen_structure_available


def main():
    output = ROOT / "outputs/stem_geometry_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    n = 15000
    count_images = {
        "haadf": np.full((32, 32), 16, dtype=np.int64),
        "df": np.full((32, 32), 2423, dtype=np.int64),
        "bf": np.tile(np.repeat([3531, 3532], 16), (32, 1)),
    }
    arrays = {key + "_count": value for key, value in count_images.items()}
    arrays.update({key + "_fraction": value / n for key, value in count_images.items()})
    stats = {}
    for key, counts in count_images.items():
        values = counts / n
        low, high = float(values.min()), float(values.max())
        stats[key] = dict(count_levels=np.unique(counts).tolist(),
            fraction_levels=np.unique(values).tolist(), mean=float(values.mean()),
            absolute_peak_to_peak=high-low,
            relative_peak_to_peak=(high-low)/float(values.mean()),
            independent_auto_contrast_levels=[low, high if high>low else low+1e-12],
            fixed_source_fraction_levels=[0., 1.])
    for pitch in (.02, .05):
        centres = (np.arange(32)-15.5)*pitch
        arrays[f"x_nm_step_{pitch}"] = np.tile(centres, (32,1))
        arrays[f"y_nm_step_{pitch}"] = np.tile(centres[:,None], (1,32))
    np.savez_compressed(output / "labelled_counting_fixture.npz", **arrays)
    snapshots = []
    state = default_state()
    for enabled in (False, True):
        state.sample.stem_wave_enabled = enabled
        request = CapturedCalculationRequest.capture(state, "High accuracy", 15000, .1)
        prepared = request.prepare(Event())
        assert prepared.snapshot.sample.stem_wave_enabled is enabled
        snapshots.append(dict(requested_quality="High accuracy", editable_wave_enabled=enabled,
            captured_wave_enabled=prepared.snapshot.sample.stem_wave_enabled,
            structure_available=specimen_structure_available(prepared.snapshot.sample),
            ray_count=prepared.ray_count, step_mm=prepared.step_mm))
    cache_root = Path.home()/"AppData/Local/TEM Simulator v2/high_accuracy_artifacts/objects"
    manifests = sorted(cache_root.rglob("manifest.json"), key=lambda path:path.stat().st_mtime, reverse=True)
    cached_weight = None
    if manifests:
        path = manifests[0]
        manifest = json.loads(path.read_text())
        row = manifest["arrays"].get("incident.ray_weight")
        if row:
            weights = np.load(path.parent/row["file"], mmap_mode="r", allow_pickle=False)
            cached_weight = dict(manifest=str(path), stored_ray_count=int(weights.size),
                minimum_weight=float(weights.min()), maximum_weight=float(weights.max()),
                weight_sum=float(weights.sum()),
                scope="Actual latest incident weights only; no saved STEM frame or complete state is present.")
    report = dict(
        scope="Mechanism fixture constructed from screenshot-reported levels, NOT recovered pixels or a new specimen simulation.",
        counting_quantum=1/n, image_shape=[32,32], sampling=[dict(step_nm=p, pixel_edge_fov_nm=32*p) for p in (.02,.05)],
        channels=stats, high_accuracy_snapshot_checks=snapshots, latest_incident_weight_check=cached_weight,
        interpretation="The reported levels are consistent with equal-weight ray interception. One ray crossing a BF mask changes raw signal by 1/15000, but independent min/max contrast spans black to white. High accuracy preserves the explicit STEM wave switch.")
    (output/"diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
