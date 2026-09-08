"""Show stored Si STEM fractions with dose/count readout; no new wave solve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from temsim.detector.stem_signal import (
    CollectionAngle, DetectorSignal, StemScanResult, reweight_stem_scan,
)
from temsim.optics.model import State

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/si110_underfocus_100nm/near_focus_tail_corrected/raw_scan.npz"
ACQUISITION = ROOT / "outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final"
KEYS = ("haadf", "df", "bf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stem_poisson_display")
    parser.add_argument("--source-current-pa", type=float, default=100.)
    parser.add_argument("--dwell-us", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-ui", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        arrays = {key: value.copy() for key, value in archive.items()}
    parameters = json.loads((ACQUISITION / "parameters.json").read_text())
    metrics = json.loads((ACQUISITION / "metrics.json").read_text())["production_metrics"]
    state = State.from_dict(parameters["state"])
    emitted_pa = state.electron_gun.emitted_current_a * 1.e12
    if not 0. <= args.source_current_pa <= emitted_pa or not args.dwell_us > 0.:
        raise ValueError("Dose must be nonnegative and within the stored emitted-current ceiling; dwell must be positive.")
    state.column_current_limit_percent = 100. * args.source_current_pa / emitted_pa
    state.sample.stem_poisson_enabled = True
    state.sample.stem_poisson_seed = args.seed
    state.sample.stem_wave_enabled = True
    shape = arrays["scan_x_nm"].shape
    for scan in (state.ac_deflector, state.descan_deflector):
        scan.scan_pixels_x, scan.scan_lines = shape[1], shape[0]
        scan.scan_frame_period_s = args.dwell_us * 1.e-6 * np.prod(shape)
        scan.scan_pixel_size_nm = .02
        scan.scan_enabled = True
    fractions = {key: arrays[f"{key}_fraction"] for key in KEYS}
    signals = {
        key: DetectorSignal(key, key.upper(), float(fractions[key].mean()), 0., 0., 0.,
                            CollectionAngle(**parameters["physical_detector_reference_angles"][key]))
        for key in KEYS
    }
    frame = StemScanResult(
        scan_x_um=arrays["scan_x_nm"] * 1.e-3,
        scan_y_um=arrays["scan_y_nm"] * 1.e-3,
        fractions=fractions, detector_signals=signals,
        metrics={**metrics, "model": "multislice_angle_resolved",
                 "scan_pixel_size_nm": .02, "scan_field_of_view_x_nm": shape[1] * .02,
                 "scan_field_of_view_y_nm": shape[0] * .02,
                 "postprocessed_archived_probabilities": True},
    )
    readout = reweight_stem_scan(state, frame)
    repeated = reweight_stem_scan(state, frame)
    saved = {"scan_x_nm": arrays["scan_x_nm"], "scan_y_nm": arrays["scan_y_nm"]}
    channels = {}
    for key in KEYS:
        assert np.array_equal(frame.fractions[key], readout.fractions[key])
        assert np.array_equal(readout.poisson_counts[key], repeated.poisson_counts[key])
        for quantity, values in (("fraction", readout.fractions[key]),
                                 ("expected", readout.expected_electrons[key]),
                                 ("poisson", readout.poisson_counts[key])):
            saved[f"{key}_{quantity}"] = values
        channels[key] = {
            "mean_fraction": float(fractions[key].mean()),
            "mean_expected_electrons_per_pixel": float(readout.expected_electrons[key].mean()),
            "mean_poisson_electrons_per_pixel": float(readout.poisson_counts[key].mean()),
            "zero_count_pixels": int(np.count_nonzero(readout.poisson_counts[key] == 0)),
        }
    np.savez_compressed(output / "counts.npz", **saved)
    (output / "readout_state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({
        "scope": "Readout of archived Si [110] corrected fractions, not a new optical/wave acquisition or the user's current dose.",
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_parameters": str(ACQUISITION / "parameters.json"),
        "source_current_pa": args.source_current_pa, "dwell_us": args.dwell_us,
        "seed": args.seed, "pixels": shape, "channels": channels,
        "fractions_unchanged": True, "same_seed_reproduced": True,
        "limitations": "Uses the archived finite-grid wave/tail model; this readout check does not establish its physical or angular convergence.",
    }, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12., 10.4), constrained_layout=True)
    extent = (-.64, .64, -.64, .64)
    for col, key in enumerate(KEYS):
        expected, counts = readout.expected_electrons[key], readout.poisson_counts[key]
        upper = max(float(expected.max()), float(counts.max()), 1.)
        for row, (label, values) in enumerate((
            ("Ideal fraction", fractions[key]), ("Expected e-/pixel", expected), ("Poisson e-/pixel", counts)
        )):
            bounds = {} if row == 0 else {"vmin": 0., "vmax": upper}
            plot = axes[row, col].imshow(values, cmap="gray", origin="lower", interpolation="nearest", extent=extent, **bounds)
            axes[row, col].set_title(f"{key.upper()} | {label}")
            axes[row, col].set_xlabel("Scan X (nm)")
            axes[row, col].set_ylabel("Scan Y (nm)")
            fig.colorbar(plot, ax=axes[row, col], shrink=.8, pad=.02)
    fig.suptitle(f"Stored Si [110] signal: {args.source_current_pa:g} pA effective source, {args.dwell_us:g} us/pixel, seed {args.seed}\n"
                 "Readout only; expected / Poisson share each detector's count scale", fontsize=13)
    fig.savefig(output / "ideal_expected_poisson.png", dpi=140)
    plt.close(fig)
    if args.capture_ui:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtTest import QTest
        from temsim.app import create_application
        from temsim.gui.scan_panel import ScanControlView
        app = create_application([])
        for name in ("segoeui.ttf", "segoeuib.ttf"):
            QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts") / name))
        app.setFont(QFont("Segoe UI", 9))
        view = ScanControlView()
        view.set_state(state)
        view.resize(1520, 1000)
        view.result_tabs.setCurrentIndex(1)
        view.show()
        view.display_result(None, readout, complete=True, state_snapshot=state)
        for quantity in ("ideal", "expected", "poisson"):
            view.image_display_quantity.setCurrentIndex(view.image_display_quantity.findData(quantity))
            app.processEvents()
            QTest.qWait(150)
            assert view.grab().save(str(output / f"ui_{quantity}.png"))
        view.close()
    print(json.dumps(channels, indent=2))
    print(output)


if __name__ == "__main__":
    main()
