# TEM Simulator v2

Clean PySide6 reconstruction of the TEM simulator. The application UI and
saved parameter names are English-only.

## Development setup

1. Run `setup_env.py` with a 64-bit Python 3.12 interpreter.
2. Select `.venv\Scripts\python.exe` as the PyCharm project interpreter.
3. Run `main.py`.

The default high-accuracy calculation is sized for a 32 GiB workstation. A
24 GiB application memory budget is checked before submission so extreme
ray-count/step combinations fail safely instead of exhausting system memory.
Magnetic-lens excitation is consistently expressed on a 0–100% scale; lenses
that require stronger fields own correspondingly higher 100% field
calibrations instead of using over-100% excitation values.

## Current MVP

- Loads the FEG, FEG + monochromator and thermionic gun TOMLs.
- Loads five C2/C3/corrector column arrangements and both recording systems.
- Validates all 10 module TOMLs, 239 part definitions and 30 catalog assembly
  combinations at startup.
- Shows every active part, its derived assembly anchor and absolute positions.
- Edits operating parameters and saves/loads complete operating profiles as
  TOML.
- Edits existing module and part TOML values with validation and atomic
  rollback when an assembly becomes invalid.
- Recalculates a direct-beam preview after a lens excitation change without
  blocking the GUI.
- Marks every active component centre on the ray plot, keeps lens centres
  labelled, and reveals other component labels progressively while zooming.
- Uses distinct high-contrast markers for apertures and draws paired
  deflectors at both TOML-defined U/L interaction planes; coincident virtual
  planes remain explicitly identified rather than being given a false gap.
- Provides an optional auto-zoom mode that focuses both axes on the selected
  component while retaining mechanical and neighbouring ray-path context.
- Stops every displayed ray at its first physical intercept and reports the
  maximum physical X angle plus the transverse display magnification, so the
  schematic's deliberately unequal axis scales cannot be mistaken for a
  90-degree electron deflection.
- Applies an exact position-dependent circular vacuum-wall cutoff in X/Y and
  keeps the earliest stop when walls, apertures and recording devices compete.
  Every part owns `vacuum_inner_diameter_mm`; uncovered drift spaces use their
  module's `vacuum_drift_inner_diameter_mm`.
- Treats the vacuum wall strictly as a mechanical cutoff. Propagation and lens
  fields remain valid beyond the last defined wall segment and are not clipped
  to a lens body's mechanical dimensions.
- Draws the stepped vacuum walls at physical scale and marks ray stops by cause with
  clickable Z/X/Y/radius diagnostics.
- Draws an aperture's mechanical body centre and effective optical stop as
  separate references when the TOML intentionally assigns different planes,
  including the Objective Aperture stop at the nominal back focal plane.
- Renders each active aperture stop as two solid blocking segments separated
  by a blank X-opening derived from the current radius and offset controls;
  disabled apertures retain only a dotted, non-blocking reference.
- Detects and labels gun and lens crossovers in both interactive previews and
  one-shot calculations, including their axial position and RMS beam radius.
- Provides separate **Physical Layout** and **Magnetic Field** pages. The
  layout uses hollow-cylinder projections, resolved vacuum bores and optical
  references; the field page plots solver-identical total and per-lens Bz,
  field support, peak field and focal length with tree/plot selection linking.
- Models C1, C2 and C3 as lens assemblies with explicit upper/lower tapered
  pole-piece children, matching the Objective assembly's mechanical hierarchy.
- Runs a one-shot full calculation with a configurable ray count and axial
  integration step.
- Runs the optional TEM wave-imaging backend during a high-accuracy calculation
  and displays the image and diffraction pattern on a dedicated page.
- Keeps gun ray paths on a strict shared-Z grid while retaining equal-time
  electron snapshots and per-ray plane-arrival times for future wavefront/arc
  overlays at important optical planes.
- Samples the cold-FEG energy tail on a strictly positive kinetic-energy
  distribution while preserving its requested mean, FWHM and deterministic
  low-discrepancy ray ordering.

The `configs/instruments` files remain the authority for static mechanical
geometry and optical references. The **Anchors** tab must be checked whenever
a component is added, removed or restructured. A changed manifest is accepted
only after the complete catalog and active assembly validate.
