# TEM Simulator v2 — Living Function and Requirements Specification

> Purpose: the authoritative living entry point for major functional changes.
> Entry point: `main.py`; project version: `0.1.0`.
> English edition: 2026-09-05. Historical requirements and revision records are
> retained. Dated implementation counts and limits describe their recorded
> baseline, not a new verification of every feature.
> The current six-stage implementation and its evidence are recorded in
> [Six-stage physics implementation](docs/SIX_STAGE_PHYSICS_IMPLEMENTATION.md).

## 0. Overall objective and modelling principles

The range-bounded live tuning and optional advanced-bank contracts are specified
in [Live tuning and cached signals](docs/INTERACTIVE_CALCULATION.md). The toggleable
Live tuning dock uses the single Ray Diagram; the detached bank detector table
has its own Cached signals subpage. TEM/STEM bank images share the normal
Illuminating Image and Scanning Image viewers through independent Result source
selectors (Current calculation / Advanced bank). Selection changes presentation
only, never instrument parameters or the computation cache. Missing bank
products must not fall back to main products under a bank label. Hiding the dock
must not clear results or trigger
calculation. Live controls
intentionally update current settings; Preview/Medium results must never replace
or erase completed high-accuracy image/spectrum products. Advanced-bank draft,
running and completed experiments remain detached from main-window results.

Workspace geometry is independent of instrument configuration. Named layouts
under View > Layouts retain window/dock geometry, page splitter sizes, selected
presentation tabs and per-ray-panel visibility variants across restarts. Layout
switches must save the outgoing layout without clearing calculation results or
changing optical parameters. See [Workspace layouts](docs/WORKSPACE_LAYOUTS.md).

TEM Simulator v2 is a research simulator for new TEM designs: real TEM structure
and physics provide the baseline, while ideal continuous design variables
allow exploration beyond the fixed controls of existing instruments.

### 0.1 Real baseline and ideal design freedom

- Preserve component order, mechanical topology, proportions, vacuum channels,
  optical action, electron propagation, aperture interception, aberrations,
  specimen interactions and detection whenever photographs, public documents,
  traceable parameters or reliable models support them.
- Engineering limits imposed by manufacturing, discrete settings, travel,
  supplies, heating, magnetic saturation, fringe-field spillover or other
  hardware constraints may be relaxed explicitly for design exploration.
- Idealisation only removes declared engineering limits. A control is not
  physically adjustable unless its value reaches all relevant implemented
  propagation and imaging calculations; changing a label is insufficient.
- Build new or modified `mechanical_only` parts from structural evidence,
  assembly relationships and mechanical clearance first. Existing rays,
  conjugate planes, aperture stops or preset strengths must not prevent their
  addition or trigger automatic optimisation. Add optical constraints only
  when the user requests ray calculations or explicitly makes the part active
  in electron propagation.
- Separate real evidence, non-OEM engineering reconstruction and ideal design
  variables. Photograph ratios, provisional values and adjustable ranges are
  not manufacturer calibration or demonstrated instrument performance.

### 0.2 Continuous apertures and ideal lens parameters

- Every circular aperture opening radius/diameter is continuously adjustable:
  no quantisation or snapping to physical holder holes. Movable apertures have
  an independent insertion state. Gun/anode, projection-chamber DPA,
  spectrometer-entrance and installed blanker stops are always inserted;
  non-retractability does not prevent simulated size or TOML position edits.
- The four positions in the C2 aperture-holder photograph explain the carrier,
  perforated strip, screw and connecting rod. They do not require four simulator
  operating states or a four-position reconstruction.
- Lens strength, axial position and comparable variables support continuous
  exploration. Ideal mode must not impose coil heating, saturation, fringe
  fields, power ratings, travel or existing instrument settings as hard hardware
  limits unless the user explicitly enables a corresponding constraint model.
- Unlimited adjustment does not mean passing mathematical `inf` to a solver.
  Finite floating-point ranges, convergence, memory and invalid-state guards
  remain necessary; label these numerical limits, not TEM performance limits.
- Any future realistic hardware constraint mode must be explicit, identifiable
  and optional. Never silently clamp, snap or restore discrete hardware settings
  in ideal design mode.

### 0.3 EDS evidence and idealisation boundaries

- EDS is an off-axis X-ray collection system near the specimen, not an axial
  electron recording plane or an electron propagation stop.
- Install one generically named EDS array. Angular acceptance is defined once
  in `configs/detectors/eds/EDS.toml`; the five column TOMLs only reference its
  specimen-plane installation. Product documents are provenance, not a second
  detector system, model selector or branded signal/UI name.
- Do not transfer sensor specifications between products. Retain traceable
  windowless/six-segment evidence, solid angles and the single-instrument
  reference take-off angle. Active area, distance, crystal shape and package
  envelope remain unknown pending section/component drawings.
- Solid angle and equivalent-cone angle are physical acceptance quantities.
  Schematic head size and drawing distance are display-only and must not enter
  physical acceptance calculations.
- EDS solid polygons must not overlap resolved Objective pole material. The
  current 1 mm display separation is not a product clearance, real collimator
  clearance or proof of unobstructed collection. Angular centre/boundary lines
  are not material and can cross a projected axisymmetric pole section.
- Separate signal physics from detector branding. Primary and elastically
  scattered track segments in specimen/support can generate K/L/M vacancies.
  Explicit point calculations use event-driven elastic Monte Carlo in finite
  3-D geometry; aggregate core-loss MFP is not an elemental elastic/ionisation
  cross section. Straight paths are an explicitly labelled reference mode.
- The offline elastic provider is provisional relativistic screened Rutherford
  at 100–300 keV, not ELSEPA/full Mott or crystal channeling. Warn for `Z > 30`.
  A future high-accuracy provider requires legally sourced ELSEPA differential
  cross sections; do not bundle redistribution-restricted NIST SRD 64 tables.

### 0.4 Post-P2 detectors and mechanical changes

- Public Titan diagrams constrain the viewing/detector topology and relative
  order of HAADF, main screen, DF, BF and Camera, not absolute OEM dimensions.
- Recorded distances from the P2 housing end to these active surfaces are
  7.25, 127.25, 217.25, 287.25 and 399.75 mm respectively. Only HAADF is very
  close. Its 7.25 mm distance and chamber dimensions remain non-OEM provisional.
- `post_projector_detector_chamber` is TOML-owned mechanical context, not a
  new optical plane or vacuum cutoff. Mechanical edits must not automatically
  recalculate lens presets. Optical solving requires an explicit request or
  a deliberately changed active optical constraint.
- Show the fixed `projection_chamber_dpa_aperture` separately at the column /
  projection-chamber boundary, after P2 and before HAADF. Do not merge it with
  the downstream `energy_filter_entrance_aperture`. The boundary location is
  the default, not a locked design constraint. Its active circular stop clips
  downstream rays and coherent waves without a conjugacy prerequisite or
  automatic preset-strength recalculation.

## 1. Document governance

### 1.1 Responsibilities

1. Retain user requirements through rewriting, refactoring, replacement or retirement.
2. Describe functions, physical models, inputs, outputs and limitations.
3. Map functions to code, configuration and tests for impact analysis.
4. Read this baseline before changes; identify requirements, implement, then
   update implementation status and revision history.

`README.md` is the quick introduction, `HANDOFF.md` the development handoff,
and `CHANGELOG.md` the release summary. If they conflict with explicit
requirements here, first establish the user's latest intent and retain this
file as the requirements baseline.

### 1.2 Requirements must not be deleted

- Never delete an allocated requirement ID.
- Improve wording, terminology, layout or location without losing intent.
- Split requirements only with references back to their original IDs.
- Keep replaced requirements, label them **Superseded**, and identify successors.
- Keep cancelled unimplemented requirements as **Cancelled by user**.
- Restore accidentally removed IDs during later editing and record the conflict.
- Retain removed/retired features as **Retired** or **Removed**, with reasons
  and alternatives.
- Refactoring must not change requirement semantics.

### 1.3 Status definitions

- **Implemented:** supported by current code or test evidence.
- **Partial:** a core path exists, with explicit gaps or approximations.
- **Pending:** recorded without a satisfying implementation.
- **Constraint:** must remain true throughout future changes.
- **Superseded:** retained history governed by a referenced successor.
- **Retired:** no longer active, but its record must remain.

### 1.4 Change procedure

After the user edits this document:

1. Read the full file, not just new requirements.
2. Compare stable IDs, revision records and available Git differences.
3. Preserve history and allocate IDs for new requirements.
4. Identify affected GUI, state, configuration, physics, output and tests.
5. Implement code and necessary configuration changes.
6. Test proportionately; physical-topology or shared-state changes warrant the
   full relevant regression scope.
7. Update status, implementation locations, limitations and acceptance evidence.
8. Append revision history; do not replace earlier records.

## 2. Permanent user-requirement ledger

This ledger permits additions, wording improvements and status changes, not
deletion of existing IDs. Implementation notes retain their dated baseline;
the current implementation document qualifies later extensions.

| ID | Retained requirement | Recorded status | Implementation |
| --- | --- | --- | --- |
| UR-001 | Explain TEM Wave Image and make it a real specimen high-accuracy wave observable, not decoration. | Implemented | `physics/wave_imaging.py`, TEM wave page |
| UR-002 | Show the electron cross section at the selected axial plane in Transverse X-Y and explain its colours. | Implemented | `diagnostic_tabs.TransverseBeamView`; continuous initial polar angle about the bundle centroid |
| UR-003 | Real samples must not generate artificial diffraction rays; only Virtual samples may configure artificial diffraction, scattering and absorption channels. | Implemented; constraint | `physics/simulation.py`, `specimen/virtual.py` |
| UR-004 | Ray Diagram uses shade within a hue for convergence semi-angle, and different hues for interaction types. | Implemented | `physics/simulation.py`, `gui/visualization.py` |
| UR-005 | At any selected Z, report interaction-electron fractions from physical probabilities, not arbitrary display weights. | Implemented | `physics/interaction_budget.py`, selected-plane table |
| UR-006 | Real samples include plasmon, ionisation, other inelastic, plural collisions and effective absorption/removal. | Implemented compact probability-conserving transport | `specimen/inelastic.py`, material TOMLs, Energy Filter, STEM |
| UR-007 | Verify startup and principal functional chains after changes. | Implemented; constraint | Offscreen GUI smoke, compilation, focused/full regressions |
| UR-008 | Do not pre-emptively fix hypothetical compatibility issues while the program starts; address actual compatibility failures specifically. | Constraint | Development policy; no warning/speculation-driven compatibility rewrites |
| UR-009 | Maintain a detailed Markdown function specification that the user can edit extensively; read, compare, implement and update it. | Implemented | This file |
| UR-010 | Requirements may be reformatted/reworded but never deleted. | Constraint | Section 1.2 and this ledger |
| UR-011 | Transverse X-Y uses a DPC-style continuous 360-degree colour wheel, not four discrete quadrants, with an initial-direction legend. | Implemented | `InitialDirectionColourWheel`, `TransverseBeamView`; +X=0 degrees, counter-clockwise toward +Y |
| UR-012 | Allow component-centre Z and arbitrary Z selection; update immediately and retain the latest selection across recalculation. | Implemented | `VisualizationWorkspace.jump_to_ray_position`, `focus_component/focus_z`, movable Z cursor |
| UR-013 | Add PSF response at Camera, screen and BF/DF/HAADF physical recording planes; retain original rays and do not confuse arbitrary Z or Objective CTF with detector PSF. | First stage implemented | TOML `point_spread_*`, `detector/point_spread.py`, `detector_response_image`, transverse overlay; Zebra/EFTEM integration recorded as pending |
| UR-014 | Remove large gaps between D/I/P1/P2 envelopes, keep vacuum ID consistent, and place every detector/camera marker on its upstream signal-collection surface. | Implemented | Recording TOMLs: 5 mm envelope gaps, 20 mm vacuum ID, `signal_collection_surface = upstream_top_surface`; layout/ray/transverse views |
| UR-015 | Give every physical round lens traceable intrinsic Cs/Cc. Maintain complete effective aberrations only for probe/sample and Objective/image systems; compare correction physically and do not add duplicate Cs to corrector quadrupoles/hexapoles. | Principle model implemented | `optics/aberrations.py`, aberration controls, TEM/STEM phase; no OEM calibration |
| UR-016 | Use real TEM structure/physics with explicitly distinguished ideal continuous variables for future designs. | Constraint | Section 0; all future state, GUI, configuration, solver and validation work |
| UR-017 | Keep circular aperture openings continuous; the four C2-holder positions are structural evidence, not runtime quantisation, snapping or four settings. | Implemented; constraint | Floating-point radius/diameter and clipping; Pt strip, screw and rod are mechanical context |
| UR-018 | Allow continuous strengths/positions beyond engineering temperature, saturation, fringe-field, supply, travel and discrete-setting limits. Keep numerical guards explicit and hardware constraints optional. | Partial; constraint | Continuous controls exist; 0–100% excitation, alignment ranges and some position controls remain implementation limits to decouple |
| UR-019 | Add one off-axis EDS array near Objective/sample, with generic EDS UI/results. Distinguish manufacturer solid angles, reported segment count and single-instrument take-off angle; do not invent unavailable sensor area/distance/enclosure from other products or patent ranges. | Mechanical/angular stage implemented | `EDS.toml`, column installation references, two-azimuth layout, `detector/eds_geometry.py`; product names only in provenance |
| UR-020 | EDS solid schematics must not overlap Objective pole material; do not alter poles or fabricate product dimensions to fit an unknown package. | Implemented with evidence limits | Polygons positioned outside maximum pole radius with 1 mm display-only separation; vertex tests; true 3-D clearance/shadowing awaits sections |
| UR-021 | Review realistic post-P2 detector distances; show an independent viewing/detector chamber and separate topology evidence from provisional absolute dimensions. | Non-OEM mechanical stage implemented | Both recording TOMLs' `post_projector_detector_chamber`, containment/boundary checks and layout; active planes and presets retained |
| UR-022 | Add a distinct projection-chamber DPA; allow size/position edits while permanently inserted. Do not trigger preset solving. | Optical stop enabled by subsequent user request | `projection_chamber_dpa_aperture` in both recording TOMLs; existing 0.2 mm family reference retained, Titan dimension unconfirmed; shared ray/wave clipping and editable geometry |
| UR-023 | Add generic EDS: 3.05 mm Cu/Au commercial square-mesh grids, vacuum support and multiple mesh counts; primary/scattered paths in sample/support generate elemental/shell yields from physical cross sections and atomic databases. | Elastic-path/characteristic-line stage implemented | Finite sample, grid openings/sidewalls/bars/rim event-driven 3-D paths in `specimen/elastic_transport.py`; one history per ray actually reaching sample, carrying X/Y, incident slopes/rotation, energy and weight; straight reference retained. `eds_atomic.py`/`eds_signal.py` provide Bote–Salvat K/L/M lines, self-absorption, solid angle, ideal efficiency and Poisson. At this recorded stage Z>30 Rutherford was provisional and bremsstrahlung, slowing, cross-layer attenuation, vacancy cascades, fluorescence, ELSEPA and channeling remained pending; later implementations must be assessed separately. |
| UR-024 | Pausing Scanning Image refresh must show the previous complete frame, never a partial raster; scan clock and Ray Diagram playback continue. | Implemented | `stemPauseImageRefresh`, independent paused-frame cache and tests |
| UR-025 | Mode is the only structure-source switch: Real imports CIF/MCIF only; Si[110] and other ideal TOML references belong to Virtual. No duplicate Real TOML/CIF selector. | Implemented | `Sample.specimen_mode`, `specimen/source.py`, mode-specific controls and schema/profile migration |
| UR-026 | Every dark-theme checkbox needs a clearly distinguishable checked state. | Implemented | `app.APPLICATION_STYLE`; separate unchecked, checked, partial, hover and disabled SVG states |
| UR-027 | All project-facing UI, documentation and newly written project content must be English. | Implemented; constraint | English UI/source/configuration text; translated specification and historical geometry research, retaining IDs and evidence |
| UR-028 | Execute the six-stage extension through shared field transport, geometry fields, intermediate coherent propagation, field-derived aberrations, model evidence and multi-parameter design studies, without automatically recalculating lens presets. | Implemented initial numerical scope; explicit limits remain | `docs/SIX_STAGE_PHYSICS_IMPLEMENTATION.md`; linear axisymmetric fields, approximate finite-pupil fits and runtime-parameter sweeps, not arbitrary 3-D FEM or unrestricted geometry search |

## 3. System scope and architecture

### 3.1 Startup chain

```text
main.py
  -> temsim.app.run()
     -> QApplication
     -> MainWindow
        -> TOML catalogue and default assembly
        -> State
        -> Preview / High accuracy controller
        -> CalculationResult
        -> Central visualisation pages
```

- Keep `main.py` a lightweight launcher without geometry or physics.
- `temsim.app` creates/reuses QApplication, applies Fusion and shows MainWindow.
- MainWindow owns selection, state, menus, workers, alignment and result dispatch.
- `simulation_pipeline.calculate()` coordinates normalisation, layout, rays,
  optional TEM wave imaging, Energy Filter, scan/descan, STEM, crossovers and stops.

### 3.2 Authoritative sources

| Information | Authority | Rule |
| --- | --- | --- |
| Requirements/semantics | This file | Retain existing requirements |
| Static geometry, ownership, optical references | `configs/instruments/*.toml` | No second Python geometry authority |
| Assembly choices | `configs/instruments/catalog.toml` | One gun/column; fixed Energy Filter recording system |
| Modes/alignment targets | `configs/operating_modes/catalog.toml` | Targets, ranges, devices, tolerances and provenance |
| Specimen/material references | `configs/specimens/*.toml` | Custom CIF must not borrow another material's constants |
| Editable operating values | State/components | Preserve user values on ordinary recalculation |
| Saved operating settings | Operating-profile TOML | Allowlisted controls only; no copied static geometry |
| Algorithms/validation/display | `src/temsim` | Obey the authorities above |

### 3.3 Assembly catalogue

Recorded audit: 10 module TOMLs, 480 variant-level part definitions, 196 logical
part keys and 15 conflict-free choices.

Guns: FEG (cold field emission), FEG + Mono (Wien monochromator), Thermionic.
Columns: C2; C3; C3 + Probe Corrector; C3 + Image Corrector;
C3 + Probe Corrector + Image Corrector.

Energy Filter is permanently installed, with no recording-system selector.
The old No Energy Filter module is historical geometry; profiles migrate to
Energy Filter. Default: FEG + C3 + Probe Corrector + Energy Filter.

### 3.4 Operating modes

- Illumination: Microprobe (TEM) / Nanoprobe (STEM).
- Projection: Image / Diffraction.
- Specimen: Real (atomic) / Virtual.
- Holder: inserted / retracted.
- Quality: interactive Preview / one-shot High accuracy.

## 4. Installation, startup and main window

### 4.1 Environment

- Supported Python: >=3.12,<3.13.
- `setup_env.py` creates/reuses .venv and installs editable project/dev dependencies.
- Main dependencies: PySide6, PyQtGraph, NumPy, SciPy, Numba, Matplotlib, ASE,
  abTEM, Pillow, ImageIO, tifffile and tomli-w.
- CuPy CUDA is optional (`gpu` extra); CPU operation remains available.
- Follow UR-008: no preventive compatibility changes without a real failure.

### 4.2 Startup and exit

- Launch: `.venv\Scripts\python.exe main.py`.
- Title: TEM Simulator v2; initial default size: 1500 x 920.
- QSettings persists/restores window geometry and dock state.
- Exit cleans the worker pool and waits up to 3 seconds for background work.

### 4.3 Menus

File: Open operating profile (Ctrl+O), Save operating profile (Ctrl+S),
Reload and validate TOML catalogue (F5), Exit (Ctrl+Q).

View: show/hide instrument and calculation-log docks; restore default layout.

### 4.4 Calculation toolbar

- Preview: 49 rays, default 2.5 mm integration step, background execution.
- High-accuracy rays: 1,000–1,000,000; default 15,000.
- High-accuracy step: 0.01–1.0 mm; default 0.1 mm.
- Compute: Auto (GPU / CPU), CPU, Numba CPU, CUDA GPU.
- Run high-accuracy once uses current settings.
- Estimate memory before submission: recorded application budget 24 GiB on a
  32 GiB target machine. Include working/history rays, inelastic branches,
  wave grids, atomistic slices and frozen-phonon configurations.

### 4.5 Background execution and transactions

- Preview, High accuracy and Direct Alignment must not block the GUI thread.
- State edits invalidate affected results; stale jobs cannot overwrite current
  state. Preserve unaffected dependency-scoped products.
- Show progress; report duration, mode and ray/wave backends on completion.
- Report failures through status, log and error dialog.

## 5. Main-window pages

Historical nine-tab baseline: Ray Diagram, Physical Layout, Energy Filter,
Transverse X-Y, Sample, EDS, Scanning Image, Illuminating Image, Optical Transfer.

Later user changes supersede only the placement: Transverse X-Y is optional
inside Ray Diagram; Sample Interactions 3D is separate. This six-stage extension
adds Model Inspector before Design Explorer. Preserve the original functions.

Magnetic Field remains optional below Ray Diagram. Scanning Image is a horizontal
split: Scanning Parameters / Probe Aberrations on the left, Geometry / Images
on the right. Never nest the full Geometry/Images ScanControlView inside
Scanning Parameters. Illuminating Image owns image results / Image Aberrations;
Optical Transfer follows it.

Global checkbox indicators: white-edged dark unchecked, cyan-blue with white
tick checked, purple with white bar partial; separate hover/disabled states,
not low-contrast system defaults.

The instrument dock contains assembly selection, illumination/projection
presets, Optical/Mechanical trees, Direct Alignment and the selected part's
Operating/TOML editors. The log records startup, catalogue/assembly checks,
backends, calculations, alignment, filter matching and failures.

## 6. Configuration, editing and persistence

### 6.1 Catalogue validation

- Validate module format/type, unique files/keys, selection signatures and completeness.
- Validate part keys/order/structural fields, nesting, optical references and polarity provenance.
- Validate runtime-key conflicts and layout for every selectable assembly.
- Stable definition ID: `<module TOML>::parts[<canonical key>]`.

### 6.2 Optical and Mechanical trees

- Optical categories cover lenses, apertures, stigmators, deflectors, correctors,
  recording planes, guns and Energy Filter.
- Mechanical shows modules and housings, yokes, coils, poles, liners, holders,
  detector packages and other parts.
- Selection works from trees and reverse navigation in layout/field/filter plots.
- Do not duplicate Sample in the left editor; selecting it opens the central page.

### 6.3 Parameter editing

- Operating owns excitation, enable, offsets, scan, slit and model switches.
- TOML owns static structure/provenance.
- TOML saves are transactional: write, rebuild/validate, restore original on failure.
- Ordinary runtime edits schedule debounced Preview.
- The recorded excitation editor uses 0–100%; stronger fields require an explicit
  supported 100% calibration. This implementation boundary does not supersede
  UR-018's future ideal-control requirement.

### 6.4 Operating profiles

- Recorded format version: 2.
- Store assembly selection, allowlisted controls, sample quaternion, zone/in-plane
  axes, Virtual interaction/region tables and per-element frozen-phonon RMS.
- Do not store static geometry, positions or other TOML-owned fields.
- Use temporary write, flush, fsync and atomic replacement.
- Load into a candidate state, then reassert TOML geometry.
- Report unknown parameters as skipped; do not silently rewrite structure.

## 7. Electron gun

### 7.1 Common output contract

All guns supply a common phase space:

- `x_m`, `y_m`, `tx_rad`, `ty_rad`, `energy_offset_ev`.
- Per-ray weights and stable `ray_id`.
- Alive/blocked state.
- Shared-Z paths, equal-time history and arrival times at key planes.

### 7.2 Cold FEG

- Components: emitter, extractor, electrostatic gun lens, accelerator stages,
  DPA/gun aperture, deflector, stigmator and C1 aperture.
- Deterministic low-discrepancy position, angle and energy sampling.
- Keep kinetic energy positive while retaining requested mean/FWHM.
- Relativistic Boris transport through finite electric/magnetic fields.
- Physical bores/apertures intercept electrons and record the first cause.

### 7.3 FEG + monochromator

- Add a finite crossed-field Wien element and energy slit to the FEG path.
- Support electric/magnetic fields, soft edges, slit crossings and energy selection.
- Retain the common gun-exit contract.

### 7.4 Thermionic gun

- Cathode, Wehnelt, gun lens, accelerator, anode aperture, deflector,
  stigmator and C1 aperture.
- Emission combines Richardson–Laue–Dushman supply, Schottky barrier lowering
  and the Child–Langmuir space-charge limit.
- Flux-weighted planar Maxwell–Boltzmann positions/velocities.
- Share the finite-field relativistic transport path with FEG.

## 8. Column, electron optics and ray propagation

### 8.1 Coordinates and propagation

- Electrons travel along laboratory +Z.
- Transverse state order: `(x, y, theta_x, theta_y)`.
- End precisely at requested Z. Shorten the final step; never round specimen
  or detector positions to the display grid.
- CPU, Numba CPU and CUDA use the same interval-step definition.
- History includes positions, slopes, alive flags, blocked Z/key, energy and weights.

### 8.2 Magnetic lenses

- Round-lens fields use their Bz profile, excitation, field calibration and polarity.
- Recorded excitation range: non-negative 0–100%; polarity controls the sign.
  This normalised calibration coordinate is not a permanent hardware limit
  on ideal design freedom (UR-018).
- Lens TOMLs retain polarity, status and source.
- Diagnostics include focal length, Cs/Cc, Larmor rotation, signed integral,
  support and peak field.
- Mechanical housing/yoke/coil/pole children do not create duplicate lenses
  or abruptly truncate mathematical field support.
- The optional geometry-field extension is documented separately; the analytic
  profile itself is not a magnetostatic solution of the pole geometry.

### 8.3 Correctors and multipoles

- Probe/image correctors, hexapoles, quadrupoles, twelve-poles and finite fields.
- Preserve each component's mechanical structure and interaction planes.
- Production rays include nonlinear hexapole/aberration terms; first-order
  Jacobians explicitly disable nonlinear kicks.
- Diagnose corrector crossovers and residual spherical aberration separately.

### 8.4 Deflectors and stigmators

- Gun, condenser, beam shift/tilt, corrector, image/diffraction, AC scan and descan.
- Paired deflectors use both TOML interaction planes; do not invent a mechanical
  gap when virtual planes coincide.
- Stigmators have independent X/Y strengths and enable controls.

### 8.5 Apertures, recording devices and walls

- Circular hard-edge openings with radius and X/Y offsets; axial geometry in TOML.
- Apertures navigation includes installed gun/anode, DPA, entrance and blanker
  stops. Fixed stops show **Always inserted**, not an editable enable switch.
  Installation remains separate: absent optional hardware never intercepts rays.
- Radius/diameter is continuous; real holder hole counts never quantise it.
- The C2 holder's four holes are structural evidence; retain one continuous
  opening and photograph-supported holder/Pt strip/screw/rod topology.
- Distinguish mechanical body centre from optical stop plane.
- Active apertures show two blocking pieces around the opening; disabled ones
  retain a non-blocking reference.
- Vacuum walls use position-dependent circular X/Y cutoffs, not field clipping
  or termination of mathematical propagation through vacuum.
- Resolve competing aperture/wall/screen/camera/detector hits by the earliest
  physical intersection. Record Z, X, Y, radius and cause.

### 8.6 Crossovers and beam statistics

- Detect gun waist, C1/C2/C3, downstream-lens and corrector crossovers.
- Report axial Z, RMS radius and status.
- Sample statistics include chief ray; RMS/95%/99%/edge convergence; 95%
  illuminated diameter; wavefront curvature; and waist offset.

## 9. Direct Alignment

### 9.1 User controls

| ID | Control | Recorded range | Coupled devices | Target |
| --- | --- | --- | --- | --- |
| DA-001 | Nanoprobe convergence semi-angle | 20–40 mrad | C2, C3 | Weighted 95% radial containment and waist at sample |
| DA-002 | Microprobe illuminated diameter | 0.5–2.2 um | C2, C3 | 95% current diameter, wavefront curvature and at most 0.5 mrad semi-angle |
| DA-003 | Image magnification | 10–1,000,000x | Objective, D, I, P1, P2 | B=0 at the active recording stop; display abs(A) |
| DA-004 | Effective camera length | Requested 0.01–5 m | D, I, P1, P2 | Relay the live Objective back-focal plane |

### 9.2 Solver rules

- Targets, ranges, device sets, seeds, tolerances and calibration provenance
  come from operating-mode TOML.
- Solve a detached snapshot in a Qt worker.
- Commit all strengths atomically only after fine production validation of
  target/conjugacy and confirmation that live state is unchanged.
- Failed, unreachable, mismatched, out-of-bounds or stale solves change no lenses.
- The recorded Image mode uses equivalent thin-lens engineering calibration
  with signed Larmor rotation: non-OEM, not measured calibration.
- Diffraction uses distributed fields and BFP relay, not fictional single-lens magnification.
- A recorded 5 m request reached only about 2.59 m at P2=100%; a request range
  is not a guarantee of reachability.

## 10. Common specimen functions

### 10.1 Finite envelope

- Controls: insertion, mode, disk/rectangle, size/diameter, thickness,
  sample-centre X/Y and scan-origin X/Y.
- New-state default: Virtual Silicon [110], 3 mm disk, 10 nm thick,
  [110] -> +Z and [1 -1 0] -> +X (matching the preset transverse cell).
  The macroscopic disk never replaces the separate wave FOV or causes
  construction of a whole-disk atomic supercell.
- The disk boundary participates in straight/elastic paths, EDS, Virtual density
  and potential clipping. One Diameter control synchronises X/Y; Rectangle
  remains available. A relabelled square is not a disk implementation.
- Active instrument TOML owns sample Z.
- Retraction preserves the probe reference Z but gives zero interaction thickness.
  Do not access dormant/invalid CIF or calculate diffraction, inelastic or atomic
  interactions while retracted.
- Snapshots carry disk/box geometry, scan FOV, calculation ROI, probe, orientation
  and Virtual regions.

### 10.2 Real structure and orientation

- Real mode uses only Imported CIF / MCIF, without a second source selector.
- With no imported file, preserve the optical sample reference but disable
  sample-wave, EDS and material-inelastic interactions; never fall back to Virtual.
- Migrate retired CIF sources to Real and preset sources to Virtual.
- A normalised (w,x,y,z) quaternion is the only physical orientation state.
- Map the zone axis [uvw] to +Z and a non-collinear in-plane direction to +X.
- Support zone alignment, incremental XYZ tilt and explicit drag-edited drafts.
- By default dragging rotates the viewing camera only. Physical editing requires
  opt-in and Apply before it affects calculation.

### 10.2.1 Virtual references

- Virtual owns Vacuum, Silicon [110], Gold [001] and Amorphous carbon (model).
- The selected reference supplies ideal structure/material for high-accuracy
  TEM/STEM and EDS. User-defined ray angular channels remain a separate ideal
  probability model, not claimed crystal-derived scattering.
- Artificial angular channels default off and require explicit opt-in, avoiding
  conflation with reference crystals and unrequested large azimuth/history allocations.
- Dormant `specimen_preset_key` and `cif_path` may survive mode switches; only
  the active mode's source can contribute to physics.

### 10.3 Structure display

- PyQtGraph OpenGL/PyOpenGL 3-D with safe 2-D ball-and-stick fallback.
- Show finite envelope, cell, atoms, bonds, +Z beam, scan FOV and calculation ROI.
- ASE covalent neighbours define bonds; ASE/Jmol colours and reduced covalent
  radii define balls. A side legend lists displayed elements.
- Default 2,500-atom soft render limit affects display only, not multislice ROI.
- Above 3,000 selected atoms, OpenGL uses point-sphere level of detail.
- ROI-local pre-cropped structures have a 5,000,000-atom safety limit.

### 10.4 Real / Virtual separation

- No artificial +g/-g, diffuse rings or other user diffraction branches in Real mode.
- Real coherent elastic diffraction belongs to high-accuracy wave/multislice.
- The compact Real ray-branch model represents material energy-loss populations
  and removal, separately from explicit local elastic Monte Carlo.
- Virtual angular rays come from user tables; reference IAM/multislice can
  independently generate wave images. Do not claim these are one scattering model.
- Retired Real qualitative-diffraction fields may round-trip through profiles
  but must not influence Real ray calculations.

## 11. Compact Real inelastic transport

### 11.1 Channels

| Key | Meaning | Representative energy / angle |
| --- | --- | --- |
| real_zero_loss | No stochastic energy loss; coherent elastic redistribution may coexist | 0 eV; no interaction kick |
| real_plasmon | Single bulk-plasmon / low-loss event | Material/user loss; relativistic characteristic angle |
| real_ionisation | Single aggregate core-ionisation event | Material/user binding/loss energy |
| real_other_inelastic | Explicit additional inelastic channel | User MFP and loss |
| real_plural_inelastic | Two or more events | Conditional mean loss and RMS-angle quadrature |
| Effective absorption/removal | Removed from tracked transmitted population | No outgoing branch |

Removal is not literal surface adsorption of a 60–300 keV TEM electron.

### 11.2 Probabilities

For independent channels k:

```text
mu_k = thickness / lambda_k
mu = sum(mu_k)
P_zero_loss = exp(-mu)
P_single_k = exp(-mu) * mu_k
P_plural_2_or_more = 1 - exp(-mu) * (1 + mu)
```

With effective absorption enabled:

```text
S_absorbed_survival = exp(-thickness / lambda_abs)
P_absorbed = 1 - S_absorbed_survival
P_tracked_channel = S_absorbed_survival * P_channel
sum(P_tracked_channel) + P_absorbed = 1
```

The final conservation equality is checked explicitly.

### 11.3 Material anchors

| Reference | Total IMFP at 200 keV | Plasmon IMFP | Plasmon loss | Ionisation loss | Evidence |
| --- | --- | --- | --- | --- | --- |
| Silicon [110] | 145 nm | 168 nm | 16.7 eV | 99.2 eV | Measurement anchor |
| Gold [001] | 84 nm | 120 nm | 9.0 eV | 84.0 eV | Measurement anchor; approximate low/core split |
| Amorphous carbon | 150 nm | 154 nm | 25.0 eV | 284.2 eV | Density-scaled approximation; override with measured film data |
| Vacuum | Disabled | Disabled | — | — | No interaction |

- TOMLs retain anchor provenance and applicability.
- At reference energy, aggregate ionisation rate is
  `1/lambda_total - 1/lambda_plasmon`.
- Plasmon voltage scaling uses a relativistic log-angle factor relative to the anchor.
- Ionisation uses BEB with U=B for relative scaling only, not an absolute cross section.
- Characteristic angles define compact quadrature, not a complete differential cross section.

### 11.4 Overrides and imported CIF

- Input plasmon, ionisation, other and absorption MFPs; representative plasmon,
  ionisation and other losses.
- Zero built-in plasmon/ionisation overrides select material defaults.
  Zero other/absorption disables those channels.
- Imported CIF must not inherit the selected preset's inelastic constants.
- Each custom plasmon/ionisation channel requires both MFP and loss; ignore
  incomplete pairs with a warning.
- A custom specimen may enable one complete channel or explicit other/removal only.

### 11.5 Transport coupling

- Each tracked energy state has an absolute-probability population.
- Sample nonzero-loss azimuth rings uniformly among source rays.
- Branch energy offset equals source offset minus representative loss.
- Modified energy enters Objective chromatic response, downstream fields and filter.
- A 4,096 post-ray batch limit controls peak memory.
- TEM coherent waves are conditional-zero-loss observables. Elastic and inelastic
  events can share one history; do not force them into exclusive categories.

## 12. Virtual specimen model

### 12.1 Interaction table

Rows contain enabled, name, kind, absolute probability and JSON parameters.
Kinds: diffraction_spots, diffuse_ring, gaussian_diffuse, arbitrary_angular,
user_screened_power_law, physical_rutherford and absorption.

- No automatic renormalisation of absolute probabilities.
- Sum of enabled interactions and absorption must not exceed one.
- Direct/transmitted beam is exactly `1 - sum(enabled probabilities)`.
- Normalise quadrature within each channel only, without changing its total probability.
- physical_rutherford uses screened relativistic Rutherford with user Z,
  areal density, screening and angle range: `P=1-exp(-N_areal*sigma)`.
- Integrate with the solid-angle Jacobian `2*pi*sin(theta)dtheta`.
- Explicitly not a full Mott model.

### 12.2 Finite regions

- Rectangle, ellipse and greyscale-map regions; NPY, PNG, TIF and TIFF inputs.
- Density restricted to [0,1]; convert image row 0 correctly to laboratory +Y.
- Apply regions only inside the finite slab; outside is vacuum.
- Optional convolution with the calculated probe.
- Selected-plane budgets evaluate density at each source-ray position without
  convolving the probe a second time.

## 13. Ray Diagram

### 13.1 Geometry display

- Source-to-recording paths, active component centres, apertures, paired
  deflectors, specimen, crossovers, walls and first-intercept stops.
- Continuous transverse projection angle. Rotation must preserve Z, current
  zoom and the screen position of Z=0.
- Wheel zoom, drag pan, context menu, Fit, component focus and progressive
  component labels with zoom.
- Movable axial cursor and double-click navigation from other axial plots.
- Axial/transverse scales are intentionally unequal. Report maximum physical
  X angle and transverse display magnification; do not imply 90-degree electron bends.

### 13.2 Interaction hues

| Interaction | Hue meaning |
| --- | --- |
| Incident | Blue-cyan |
| Vacuum/reference | Neutral grey-blue |
| Real zero loss | Pale grey-blue |
| Real plasmon/low loss | Cyan |
| Real ionisation | Red-orange |
| Real other inelastic | Yellow |
| Real plural inelastic | Purple |
| Virtual transmitted | Green |
| Virtual diffraction spots | Purple-blue |
| Virtual diffuse ring | Orange |
| Virtual Gaussian diffuse | Yellow |
| Virtual arbitrary angular | Cyan-green |
| Virtual screened power law | Pink |
| Virtual physical Rutherford | Red |

### 13.3 Convergence shading

- Five dark-to-bright bins within each interaction hue.
- At the specimen, define semi-angle relative to the branch's current-weighted
  3-D chief ray.
- Saturate brightness at the incident bundle's weighted 99% convergence angle.
- Subtract compact Real inelastic characteristic-angle kicks before determining
  illumination convergence.
- Hue identifies interaction; shade identifies illumination convergence.

### 13.4 Arbitrary-Z interaction budget

Use all ray weights, not the at-most-48 displayed trajectories. Report:

- Selected Z relative to specimen.
- Source fraction reaching Z and source fraction reaching specimen.
- Per-interaction conditional probability at specimen incidence.
- Per-interaction source fraction reaching Z and composition among survivors at Z.
- Representative loss, pre-specimen stops, sample removal and downstream stops.
- Total conservation error.
- Real material, t/lambda and combined IMFP.
- When available, a separate non-exclusive conditional-zero-loss coherent
  redistribution observable from TEM waves.

## 14. Transverse X-Y

- Displays an electron cross section at component-centre/arbitrary Z, not
  automatically a diffraction pattern.
- Equal physical X/Y scale. The historical display used mm; later UI changes
  use explicit correctly formatted physical units.
- Before specimen, use incident rays; after specimen, use 000 or the first
  available outgoing branch. Exclude rays intercepted upstream.
- Display at most 2,000 rays; statistics use the selected surviving rays.
- Continuous colour is each ray's initial polar angle about its bundle centroid:
  +X=0 degrees, increasing counter-clockwise toward +Y. This shows lens rotation,
  not radius, energy or intensity.
- Show the matching DPC-style 360-degree ring with +X/+Y/-X/-Y and
  0/90/180/270-degree labels. The undefined initial-centroid direction is neutral grey.
- Component selection uses centre Z; Go to Z, axial selection and cyan cursor
  use arbitrary Z. Update continuously during dragging and preserve selection
  across recalculation.
- At Camera, screen and BF/DF/HAADF, overlay peak-normalised greyscale detector
  response below the original direction-coloured rays: apply physical
  disk/annulus/square hit_mask, Gaussian PSF, then finite-area mask again.
  Report weight lost into holes or beyond edges.
- PSF sigma_x/sigma_y and counter-clockwise principal-axis angle are TOML-owned,
  in recording-plane mm, labelled provisional_model_parameter rather than
  measured calibration.
- At arbitrary/non-recording planes, show geometric rays only. PSF never changes
  rays or convolves the specimen-to-Objective CTF image.
- Summary: branch, survivor count, RMS radius and orientation relative to bundle start.
- These direction colours are distinct from Ray Diagram interaction/convergence colours.
- Later placement is the optional Ray Diagram side panel, not a separate tab.
  User scale/centre remain fixed when Z changes; Fit beam remains explicit.

## 15. TEM wave image and multislice

### 15.1 Observable

- Requires an active Real CIF/MCIF or Virtual TOML reference, Microprobe (TEM),
  enabled TEM image / diffraction and High accuracy.
- The original left panel is local specimen-to-Objective CTF intensity, with
  percentile-clipped [0,1] display. The right panel is log exit-wave diffraction.
- That local observable is not itself the final projector/Camera image and
  excludes the curved Energy Filter branch. Later physical Camera propagation
  is a separate output; see the six-stage implementation.
- It shows projected/atomic potential, multislice, Objective defocus/Cs/aperture
  and coherent diffraction effects.
- Preserve linear diffraction probabilities independently; never read them from
  log-display pixels.

### 15.2 Potential

- Analytic continuous projected columns and finite atomistic IAM slices.
- Silicon [110] and Gold [001] crystal definitions.
- ASE structures and abTEM 1.0.10 Lobato–Van Dyck neutral-atom IAM potentials.
- Orthogonalise/periodically expand CIF only over scan ROI + probe padding
  intersected with the finite specimen, not a macroscopic full-sample supercell.
- CIF requires atomistic IAM/multislice; never silently substitute another material.

### 15.3 Multislice

- CPU reference: complex128 NumPy symmetric split operator.
- Optional CUDA: complex64 CuPy.
- Rectangular grids, independent X/Y sampling, projected 2-D potential or
  explicit (Z,Y,X) slices.
- Explicit angstrom / inverse-angstrom FFT convention, 2/3 anti-alias bandwidth,
  uniform or nonuniform slices.
- Report maximum slice phase, initial/final integrated intensity, maximum change
  and sampling support.
- Any CUDA allocation/propagation/FFT/integration failure discards all partial
  output and restarts the whole operation on CPU.

### 15.4 Frozen phonons

- Enable, 1–64 configurations, global one-axis RMS, per-element RMS and seed.
- Presets may supply qualified thermal RMS; CIF requires explicit global or
  per-element values.
- Reproducible independent isotropic Gaussian displacements (Einstein approximation).
- TEM/STEM average configuration intensities, not complex amplitudes.
- Report finite-ensemble relative standard error, without claiming a universal
  converged configuration count.

### 15.5 Wave-model limits

- Conditional-zero-loss coherent elastic wave.
- No bonded-charge redistribution, correlated phonons, absorptive/inelastic
  multislice potential, magnetic-specimen scattering or spin.
- Not a complete energy-differential EELS/dielectric-response model.
- Do not renormalise missing intensity outside reciprocal-space support.
- Optional high-angle Rutherford tails begin strictly outside wave support and
  are reported separately.

## 16. STEM, AC Scan and Descan

### 16.1 Raster controls

- AC/Descan share enable, pixel pitch, derived X/Y FOV, frame period, X pixels,
  Y lines, upper gain and derived lower coupling.
- Pixel pitch: 0.001 nm–1 mm; pixel/line counts: 2–4096.
- FOV = count * pixel pitch is the full footprint; centre-to-centre span is
  (count-1) * pitch.
- Shared clock, dimensions and pitch.
- Report and roll back physical-coil-limit or singular-transfer failures.

### 16.2 Calibration

- Solve upper/lower AC foils with active signed first-order optics for zero
  sample angular response: pure shift.
- Descan receives the exact negative AC command.
- Solve its lower coupling to hold the chief ray stationary near the Selected
  Area Aperture image-reference station.
- TOML AC/Descan geometry mirrors about specimen; reject broken symmetry.
- Classify Objective Aperture, Selected Area Aperture and calculated first
  image/diffraction planes from live Jacobians as image, diffraction or mixed.

### 16.3 Geometry panel

- Sample raster and selected downstream recording-plane trajectory.
- Requested/preview raster, pixel/FOV, sample span, drift pivot, foil symmetry,
  coupling matrices and residuals.
- Geometric and ray displays use the same physical foil planes.

### 16.4 Images panel

- Simultaneous HAADF/DF/BF images in equal-scale laboratory scan X/Y, with pan/zoom.
- Show each detector's TOML Z, inner/outer size and collection angles from the
  full signed 2x2 sample-to-detector map.
- Report ranges for anisotropic transfer, not a falsely precise scalar angle.
- Geometric Preview polygons/wedges are detector-clipping boundaries, not atomic contrast.
- High accuracy can use angle-resolved wave/multislice signal.
- Virtual signals use finite density and absolute interaction probabilities.
- Warn about vacuum outside finite sample and CIF pitch coarser than half the
  shortest atomic spacing.

### 16.5 Detector integration and outputs

- Evaluate hit_mask after complete signed transport.
- Intercept in axial order: an upstream detection cannot be counted again downstream.
- Return source-fraction images, pA, expected electrons/dwell, optional seeded
  Poisson counts, dwell, uncollected, absorbed, truncated and separate high-angle tail.
- The original frame path did not store a full 4D-STEM cube; any later optional
  cube path is distinct from this baseline.
- Separate inelastic absorption from source current and conserve tracked probability.
- The recorded high-accuracy approximation reuses coherent angular distributions
  for tracked loss populations; compact inelastic angles travel in rays/filter.
  Keep that approximation explicit wherever still used.

### 16.6 Playback

- Calculate/cache one complete frame, not fresh physics on every timer tick.
- With AC enabled, timer playback reveals lines according to frame period.
- Stopping scan retains the complete frame.
- Pause refresh fixes the previous full HAADF/DF/BF frame, never a partial raster;
  scan clock and Ray Diagram playback continue. Resume shows current progress.
- Ray playback reuses AC/Descan bases and cached positions; view rotation does
  not recalculate the column.

## 17. Energy Filter

### 17.1 Installation

- Iliad is permanently installed, with no filter-free assembly option.
- Optical branch enable is independent of installation.
- Controls include EELS/EFTEM, selected loss, window, optical integration,
  MultiEELS, alignment and ray tracing.
- Match rigidity, sector field, M12 and multipole scales to current high tension.

### 17.2 Topology

Entrance aperture; large tapered prism; independently powered M01–M10
(stable simulator indices, not claimed production names); XO crossover /
optional EFTEM slit; independent fast electrostatic shutter; dynamic-focus
electrostatic quadrupole placeholder; MultiEELS bias tube; Zebra deflector;
optional EFTEM output; Zebra EELS detector.

### 17.3 Physical tracing

- Extract representative entrance rays with position, direction, energy,
  colour and absolute source fraction.
- Real plasmon, ionisation and plural losses modify kinetic energy.
- Continuous relativistic Boris transport through sector and multipoles.
- Record reach/pass states for slit, EFTEM, EELS and Zebra, with stop keys.
- Slit acceptance uses dispersion, selected loss, energy window and blade travel.
- Report entrance/slit/camera/EELS fractions and pA.
- Never renormalise absolute branch weights to conceal absorption.

### 17.4 Page

- Dedicated curved branch, not flattened main-column axial markers.
- Show entrance, prism clear path, M01–M10, XO/slit, electrostatic envelopes,
  EFTEM output and Zebra surface.
- Independently adjustable X/Z scales.
- Click labels, centres or bodies to navigate to the corresponding editor.
- Dashed leaders are visual only and must not intercept body clicks.
- Later user layout moves filter-specific parameters here, separate from the
  main-column editor, with parameter and physical/ray-view groups.

### 17.5 Limits

- Prism radius/bend/gap and many multipole positions/envelopes are parameterised,
  non-OEM starting values.
- Dynamic-focus quadrupole remains mechanical without a validated field model.
- Straight-column J_img/J_diff stops at filter entrance; curved sector,
  M01–M10 and Zebra are not part of that first-order orientation map.

## 18. Diagnostic views

### 18.1 Physical Layout

- Proportional hollow-cylinder sections, vacuum bores, optical references,
  specimen/stage/holder and recording devices.
- Screen-space label packing into rows; leaders to centres/edges.
- Label movement never changes TOML geometry.
- Names, markers and bodies navigate to Optical/Mechanical editors.

### 18.2 Magnetic Field inside Ray Diagram

- Vertical splitter for ray plot, selected-plane budget and optional field plot;
  draggable high-contrast handles.
- Ray plot initially receives most space; toggling fields retains user sizes.
- Solver-identical total and individual Bz curves.
- Toggle individual fields/rotation labels.
- Tooltips: peak, excitation, formula, signed integral, polarity/status/source,
  per-lens/cumulative Larmor rotation.
- Mark image planes and specimen-to-plane orientation.
- Selection highlights support and exposes field/focal/Cs diagnostics.
- Double-click synchronises the ray cursor; the axial range follows Ray Diagram.

### 18.3 Optical Transfer

- Display `r_plane = J_img @ r_sample + J_diff @ theta_sample`.
- J_img is dimensionless; J_diff is m/rad, numerically equal to mm/mrad.
- Reference plus transverse basis rays; subtract the reference to remove affine shift.
- Report rotation, reflection/handedness, anisotropy, equivalent magnification /
  camera length and conjugacy residual.
- Capture Image and Diffraction states at the same plane to form a normalised
  diffraction-vector-to-image-direction map.
- Keep the full signed 2x2 map, never infer handedness from an absolute X coefficient.
- Camera-axis uncalibrated_identity is a placeholder, not absolute crystal calibration.

### 18.4 Illuminating Image

- Objective CTF image and exit-wave diffraction, plus separately identified later
  physical-Camera outputs.
- Summary: source/preset, model, slices, potential, configurations, thermal RMS,
  backend, FOV, pixel and surviving rays.
- Warnings: sampling truncation, conservation, CUDA/atomistic fallback and phonon uncertainty.
- Keep visible descriptions short; detailed qualifications belong in tooltips.

### 18.5 EDS and local interactions

- EDS is a separate top-level page to the right of specimen-related pages.
  Sample owns structure, envelope, orientation and interaction models.
- Use the complete bundle actually surviving to the physical sample plane;
  no independent Monte Carlo history-count input.
- Include X/Y, tx/ty, rotation, energy offset and source weight. Apply upstream
  survival exactly once.
- U-Z/V-Z projections reuse the same histories:
  U=X cos(phi)+Y sin(phi), V=-X sin(phi)+Y cos(phi).
  Synchronise phi with Ray Diagram; rotating redraws saved X/Y only.
  Spectrum/line list has its own subpage.
- The original explicit Calculate point EDS and sample-region actions must
  share compatible high-accuracy results after later cache unification.
  Mechanical edits do not automatically rerun EDS or presets.
- Local entry uses cached upstream phase space. Finite sample/support transport
  and EDS generate terminal states; forward electrons re-enter Objective /
  downstream lenses, apertures, physical recording planes and walls.
  Save the exit handoff explicitly; invalidate only affected dependencies.
- The historical ray overlay included primary/backscatter/elastic events,
  qualitative zero-weight secondary markers, generated characteristic photons
  and photons within EDS acceptance. Secondary display was later retired at
  the user's request and must not be reintroduced.
- Photon directions are uniform over a sphere. Unknown sensor face/distance
  requires an acceptance surrogate with the exact known aggregate solid angle;
  drawn endpoints are not sensor intersections.
- Rutherford is the elastic provider, not a separate particle species.
  Channeling belongs to coherent multislice, not another randomly added cross section
  or a duplicate Monte Carlo tail.
- Sample Interactions 3D is a cached local view, with selectable signal categories
  and default +Z downward. Tab switches and display filters must not erase
  trajectories or recalculate physical outcomes.

## 19. Backends, performance and failure handling

### 19.1 Rays

- NumPy CPU baseline; Numba CPU parallel kernels.
- Numba CUDA for sufficiently large independent-ray RK4 column propagation.
- Auto retains CPU for small tasks and selects accelerated backends above thresholds.

### 19.2 Waves

- NumPy complex128 reference; optional CuPy complex64.
- Auto accounts for work size to avoid small-task launch/transfer overhead.
- Resident STEM CUDA keeps scans, potentials, probe formation, multislice, FFT
  and masks on device, returning final detector arrays.
- Reuse frequency grids, anti-alias masks and slice propagators.

### 19.3 Fallback

- Report unavailable backends or actual execution failures, then use an allowed CPU path.
- CUDA operations are atomic: never combine partial GPU and CPU observables.
- UR-008 permits fallback after real errors; it prohibits speculative compatibility rewrites.

## 20. Physical definitions and conservation

- Source fractions are relative to emitted source current.
- Weights must be finite, non-negative and aligned with the ray bundle.
- Absolute branch sums must not exceed one.
- Real/Virtual interactions, plane budgets, Energy Filter and STEM must retain
  absorbed/removed probability; never renormalise it away.
- Coherent elastic scattering and inelastic energy states are non-exclusive.
- Nanoprobe control uses weighted 95% radial containment; wave pupils use the
  more conservative weighted 99% angular containment.
- Transverse views use equal physical X/Y scale; Ray Diagram is an axial schematic.

## 21. Explicit limits and provisional assumptions

The following recorded baseline limits must not be presented as completed
physics without separate implementation evidence:

- No complete energy-differential EELS/dielectric-loss model in the original
  compact transport. Later forward-spectrum modules are separate, not proof
  of a full dielectric model.
- No absorptive/inelastic complex-potential multislice, bonded-charge potential,
  correlated phonons, magnetic-specimen scattering or spin.
- Compact Real inelastic angles/losses are representative quadrature, not full line shapes.
- The recorded high-accuracy STEM approximation reused zero-loss angular distributions
  for tracked inelastic populations.
- Screened Rutherford is not full Mott. EDS elastic histories use independent-atom,
  bulk-density/mass-fraction, amorphous-style transport, not coherent channeling
  or unique classical paths extracted from multislice.
- Initial EDS electrons are geometric sample-plane phase space with source
  sampling, convergence, tilt/rotation, energy and weight, not a unique
  interpretation of an aberrated coherent probe.
- Elastic events conserve kinetic energy; the recorded elastic kernel excludes
  nuclear recoil and inelastic angular kicks.
- Carbon inelastic references are density-scaled, not universal film data.
- Atomistic potential is neutral-atom IAM.
- The local Objective CTF image is not the Camera image. Later Camera propagation
  is separately identified and sampling-limited.
- No default compensation outside wave angular support; optional tails remain separate.
- Dynamic-focus filter quadrupole has no implemented validated field.
- Curved-filter optics are outside the straight-column orientation map.
- Absolute detector/display axes are not measured/calibrated.
- Many lens polarities are provisional; projector calibration and numerous
  dimensions are non-OEM reconstruction.
- Analytic Bz is not reshaped by mechanical poles. The later optional linear
  axisymmetric field solve supplies a different, explicitly approximate mode:
  it is not calibrated alloy behaviour or arbitrary 3-D FEM.
- Geometric STEM Preview is not atomic contrast.
- Alignment request ranges do not guarantee reachable targets.
- The current 0–100% excitation, alignment ranges, aperture maxima and some
  static-position editors are implementation/calibration boundaries, not accepted
  final limits on ideal design freedom. UR-018 remains partially implemented.

## 22. Error handling and safety

- Reject invalid TOML, duplicate keys, missing structure and assembly conflicts.
- Reject invalid runtime assignments before mutation.
- Failed alignment changes no lenses.
- Failed scan/descan calibration restores both complete prior component states.
- Reject Virtual probabilities above one instead of normalising.
- Raise on Real probability-conservation failure.
- Clearly report missing/invalid CIF and atom-limit violations.
- Refuse high accuracy above the memory budget rather than exhaust the machine.
- Recalculate fully after real CUDA/CuPy failures; never deliver partial observables.
- Use recoverable/atomic profile and manifest writes.
- Compatibility fixes require actual failures (UR-008).

## 23. Function-to-code map

| Domain | Main code |
| --- | --- |
| Startup/window | main.py, src/temsim/app.py, gui/main_window.py |
| Catalogue/TOML | assembly_catalog.py, module_manifest.py, manifest_editor.py, column/* |
| State/profiles | optics/model.py, runtime_parameters.py, profile_io.py, state.py |
| Guns | optics/electron_gun/* |
| Lenses/fields/aberrations | optics/*lens*.py, physics/core.py, physics/magnetic_lens_aberration.py |
| Correctors/multipoles | optics/probe_corrector.py, optics/image_corrector.py, physics/*multipole*.py |
| Deflectors/scans | optics/*deflector*.py, physics/scan_geometry.py |
| Ray simulation | physics/simulation.py, physics/acceleration.py |
| Walls/stops/apertures | physics/column_wall.py, physics/aperture_clipping.py, physics/recording_clipping.py |
| Beam/crossovers | physics/beam_statistics.py, beam_waist.py, crossovers.py, all_lens_crossovers.py |
| Direct Alignment | optics/direct_alignment.py, gui/direct_alignment_* |
| Signed transfer | physics/first_order.py, gui/diagnostic_tabs.py |
| Specimen geometry/orientation | specimen/geometry.py, gui/sample_panel.py |
| Reference/atomic specimen | specimen/presets.py, specimen/atomistic.py, configs/specimens/* |
| Real inelastic | specimen/inelastic.py, physics/interaction_budget.py |
| Virtual specimen | specimen/virtual.py |
| TEM waves | physics/wave_imaging.py, physics/multislice.py, physics/wave_fft.py |
| STEM waves/CUDA | physics/stem_wave_imaging.py, physics/stem_cuda_pipeline.py, physics/cuda_multislice_plan.py |
| STEM signals | detector/stem_signal.py, gui/scan_panel.py |
| Recording devices | detector/* |
| Energy Filter | optics/energy_filter*.py, optics/energy_filter_detector.py |
| Central views | gui/visualization.py, gui/diagnostic_tabs.py |
| Unified results | simulation_pipeline.py |
| Shared specimen fields | specimen/vector_field_transport.py |
| Geometry fields | physics/axisymmetric_magnetostatics.py, physics/lens_field_provider.py |
| Coherent intermediate planes | physics/multiplane_wave.py, physics/camera_wave.py |
| Field-derived coefficients | optics/field_aberrations.py, optics/aberrations.py |
| Model evidence | gui/model_inspector.py |
| Detached design studies | design_experiments.py, design_sweep_execution.py, gui/design_explorer.py |

## 24. Test map and recorded verification

### 24.1 Test domains

- Atomic/CIF/phonons: test_atomistic_specimen.py, test_multislice.py, test_wave_imaging.py.
- Real inelastic/probability: test_real_inelastic.py, test_sample_plane_boundary.py.
- Virtual: test_virtual_specimen.py, test_sample_model_v2.py, test_sample_profile_v2.py.
- GUI/specimen: test_gui_shell.py, test_sample_page.py.
- Rays/fields/correctors: test_mvp_core.py, test_corrector_calibration.py,
  test_magnetic_lens_aberration.py.
- Alignment/first order: test_direct_alignment.py, test_first_order_transfer.py.
- Scan/STEM: test_scan_system.py, test_stem_observables_v2.py, test_stem_cuda_pipeline.py.
- CUDA/FFT: test_compute_backend.py, test_cuda_multislice_plan.py, test_wave_fft.py.
- TOML/layout: test_toml_authority.py, test_manifest_editing.py, test_column_wall.py,
  test_field_polarity_manifest.py.
- Detectors/filter: test_detector_orientation_manifest.py, test_detector_point_spread.py,
  test_energy_filter_physical_layout.py, test_eds_detector_geometry.py,
  test_post_projector_detector_chamber.py.
- EDS/elastic: test_eds_signal.py, test_elastic_transport.py, test_specimen_support.py.
- Guns/timing: test_electron_gun_timing.py.
- Six-stage additions: test_six_stage_physics.py, test_model_inspector.py,
  test_design_sweep_execution.py and the affected field/wave/cache/GUI suites.

### 24.2 Historical verification (not rerun merely by translation)

- 2026-08-31 pause-refresh/source-exclusivity groups: 196 passed
  (41 scan/sample/profile/aberration, 41 GUI, 114 specimen/wave/EDS/core).
  The existing 16 ms projection-timer test passed alone; a heavily loaded combined
  run exceeded its 1 s event-loop timeout. The threshold was not changed.
  Compilation and diff whitespace checks passed.
- 2026-08-31 elastic-path stage: focused cross-section/CDF/geometry/transport/EDS/
  profile/GUI checks passed; non-Nanoprobe-recalibration suite:
  405 passed, 1 skipped, 6 deselected. Deselections were the two C2/C3
  Nanoprobe calibration families excluded at user request.
  Compilation, main import, dependency and whitespace checks passed.
- 2026-08-31 sample-bundle EDS and page reorganisation: 408 passed,
  1 skipped, 6 deselected, no failures. Same excluded live-solve families.
  Focused EDS/profile/GUI, serial compilation, dependency check, offscreen
  startup/exact tab-order smoke and whitespace checks passed.
- 2026-08-30 mechanical DPA stage: 386 collected; non-recalibration set:
  379 passed, 1 skipped, 6 deselected. The six excluded points were not rerun;
  their preceding result remained 2 passed, 4 failed.
- Those four historical failures followed changed C2 length/calibration and
  aperture positions: 100 um -> 30 mrad, 200 um -> 60 mrad, 60 um -> 18 mrad,
  140 um -> 42 mrad. EDS mechanical work did not rewrite presets or warm starts.
- Five-column EDS geometry/provenance, angular derivation, layout, narrow-window
  and non-axial/non-optical checks passed.
- EDS polygon 1 mm display-only pole separation, both recording-chamber
  boundary/containment checks, P2-to-five-plane distances and offscreen rendering
  passed; related combined regression: 142 passed.
- Projector envelopes, upstream detector surfaces, PSF, orientation and GUI
  focused checks passed.
- Recorded Python 3.12.3 dependency check: no conflicts.
- main.py imported; offscreen MainWindow constructed, showed and closed;
  compileall and git diff --check passed.
- LF/CRLF notices were not functional failures and were not treated as speculative
  compatibility work (UR-008).

Current six-stage verification is recorded in
[the implementation report](docs/SIX_STAGE_PHYSICS_IMPLEMENTATION.md), independently
of these historical counts.

## 25. Future requirement editing

Append requirements with fresh IDs; the implementer assigns an ID if absent.

```markdown
### CR-NEW — Change title

- Status: Pending analysis
- Related requirement: UR-xxx / section
- Requested behaviour:
  - ...
- Behaviour that must remain unchanged:
  - ...
- Inputs/parameters:
  - ...
- Expected output/interface:
  - ...
- Acceptance:
  1. ...
  2. ...
- Notes/physical evidence:
  - ...
```

Natural-language edits are valid without this template. Existing requirements
must still be retained.

## 26. Append-only revision history

| Date | Change | Recorded outcome |
| --- | --- | --- |
| 2026-08-13 | Established the living function/requirements specification covering startup, assemblies, UI, rays, specimen, Real/Virtual interactions, waves, STEM, filter, diagnostics, limitations and tests. | Created permanent UR-001–UR-010 ledger for continued updates. |
| 2026-08-20 | Made Energy Filter permanent; removed recording-system installation choice and migrated old filter-free profiles. | All 15 gun/column combinations use Energy Filter; historical TOML retained for validation. |
| 2026-08-20 | Replaced four-quadrant transverse colours with continuous DPC-style 360-degree initial-direction wheel. | Off-axis bundles use their own initial centroid; direction/radius/energy/intensity remain distinct. |
| 2026-08-20 | Unified component-centre and arbitrary-Z transverse selection. | Immediate updates, live cursor drag and selection retained across recalculation. |
| 2026-08-20 | Added TOML-owned 2-D PSF at Camera, screen and BF/DF/HAADF. | Original rays retained; physical-plane overlays report retained weight; adjustable defaults explicitly non-calibrated. |
| 2026-08-20 | Tightened D-I-P1-P2 envelopes, unified vacuum tube and moved signal markers to upstream surfaces. | 5 mm neighbouring gaps, 20 mm ID, field centres retained; layout/ray/transverse markers share signal surfaces. |
| 2026-08-20 | Added layered hybrid aberrations. | Intrinsic lens Cs/Cc provenance; probe/image C1/A1/B2/A2/C3/S3/A3/C5/Cc with azimuths; correction comparison toggles nonlinear hexapoles; shared TEM/STEM coefficients. |
| 2026-08-30 | Established real-TEM baseline plus ideal continuous design variables and C2-holder evidence limits. | Added UR-016–UR-018; holder holes do not quantise apertures; hardware limits must not silently constrain future ideal mode. |
| 2026-08-30 | Recorded projector topology research and added an Ultra-X-based EDS schematic near sample. | D/I/P1/P2 two-pole topology remained provisional; six-segment aggregate and graded angular evidence; no invented dimensions or preset solve. |
| 2026-08-30 | Consolidated EDS geometry into one model TOML. | Historical UltraX.toml referenced by five columns; no second Super-X instance or model selector. |
| 2026-08-30 | Removed EDS/pole 2-D solid overlap and reviewed post-P2 spacing. | 1 mm display-only separation; no product/pole edits at this stage; non-OEM detector chamber added; HAADF-first topology supported, 7.25 mm not OEM; active planes/presets retained. |
| 2026-08-30 | Added projection-chamber DPA and separated mechanical addition from conjugacy constraints. | UR-022; distinct from Iliad entrance; 0.2 mm Tecnai/Talos reference only; no clipping/reference/preset solve. |
| 2026-08-30 | Began generic EDS signals and retired Ultra-X branding for the new system. | Revised UR-019/020, added UR-023; EDS.toml and generic UI; Cu/Au/vacuum supports, 50–500 mesh catalogue, Bote–Salvat K/L/M, xraylib direct relaxation, characteristic lines/self-absorption/solid angle/Poisson. Elastic paths and continuum were still pending at this date. |
| 2026-08-31 | Prioritised real elastic paths. | Event-driven finite-sample/square-mesh 3-D Monte Carlo, exponential flights, scatterer sampling, screened Rutherford, reproducible seed, terminal/truncation statistics and EDS path summaries. Straight reference retained; Z>30/ELSEPA/channeling/loss/continuum limitations explicit; no preset solve. |
| 2026-08-31 | Used actual sample-plane ray bundles for EDS and reorganised pages. | Removed independent history count; all incident phase-space fields/weights used. EDS gained orthogonal paths and spectra. Historical nine-tab order established; field panel embedded; probe/image aberrations moved into image controls. No preset solve. |
| 2026-08-31 | Added manual local entry/exit, photons and downstream handoff. | EDS local-run controls; real flights/terminal states; forward reinjection downstream. Uniform-sphere X-rays and exact aggregate acceptance; elastic/backscatter and historical zero-weight secondary markers. Channeling remained wave-owned; no artificial independent paths or preset solve. |
| 2026-08-31 | Set default Si[110] to a 3 mm disk, 10 nm thick. | Virtual reference, [1 -1 0] in-plane and 3,000,000 nm diameter; disk participates in all relevant geometry/density/wave/MC paths. Old no-shape schema retains rectangle semantics; no preset solve. |
| 2026-08-31 | Made internal Ray Diagram panel heights adjustable. | Three-way vertical splitter for rays/budget/optional field; ray area prioritised; high-contrast handles and toggle-size retention; no ray/field/lens physics changed. |
| 2026-09-05 | Translated the specification to English while retaining every UR/DA ID and historical revision. | Historical counts/limits are explicitly dated; later transverse placement, cache reuse, secondary retirement and physical Camera outputs are distinguished from their original baseline. |
| 2026-09-05 | Implemented the six-stage physics extension. | Shared specimen vector transport, optional linear axisymmetric geometry fields, coherent intermediate apertures, exclusive field-derived aberrations, Model Inspector and multi-runtime-parameter Design Explorer; evidence and limits in the implementation report. No automatic preset optimisation. |
