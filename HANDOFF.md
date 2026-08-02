# TEM Simulator v2 — Project Handoff

Last updated: 2026-08-02

## Purpose

This file is the persistent handoff for continuing development. Keep it focused
on current behaviour, confirmed design decisions, provisional assumptions and
the next concrete work. `README.md` remains the user/developer overview and
`CHANGELOG.md` remains the release history.

## Start and verify

- Workspace: `F:\tem_simulator_v2`
- Python: `.venv\Scripts\python.exe`
- Application entry point: `main.py`
- Run: `.venv\Scripts\python.exe main.py`
- Tests: `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest -q`
- Last full result: **52 passed**. The remaining messages are existing Pydantic
  `json_encoders` deprecation warnings.

## Current validated state

- The instrument catalog contains 10 module TOMLs, 441 part definitions and 30
  selectable assembly combinations.
- High-accuracy defaults target a 32 GiB workstation and use a conservative
  24 GiB application-memory preflight limit.
- Magnetic-lens excitation is limited to 0–100%. A lens needing a stronger
  field must raise its calibrated 100% field value rather than exceed 100%.
- Column/vacuum walls only cut off rays. They do not clip propagation or the
  mathematical support of a lens magnetic field.
- Ray stops use the earliest physical intersection among vacuum walls,
  apertures and recording devices.
- Ray Diagram supports continuous transverse viewing angle. Rotating X/Y keeps
  the current zoom, Z scale and Z=0 position unchanged.
- Gun trajectories retain equal-laboratory-time snapshots and per-ray arrival
  times for future wavefront/arrival-arc overlays at important planes.
- Operating-mode storage exists for condenser `micro_probe`/`nano_probe` and
  projector `imaging`/`diffraction`, but calibrated lens/aperture values remain
  intentionally empty until crossover constraints are supplied.

## Mechanical lens model

Every configured round magnetic lens currently has one optical parent plus
independent mechanical children:

```text
<lens>                         optical/field parent; no material drawing
├── <lens>_housing             non-magnetic outer housing
├── <lens>_yoke                soft-magnetic yoke
├── <lens>_excitation_coil     insulated copper winding package
├── <lens>_upper_pole          upstream pole piece, where applicable
└── <lens>_lower_pole          downstream pole piece, where applicable
```

Mechanical-only parts use these TOML fields:

- `local_start_z_mm`, `local_center_z_mm`, `local_end_z_mm`, `length_mm`
- `vacuum_inner_diameter_mm`
- `mechanical_inner_diameter_mm`, `mechanical_outer_diameter_mm`
- `mechanical_profile`, `mechanical_part_role`, `material_class`
- `mechanical_only = true`, `parent_key`
- `mechanical_overlap_group`, `mechanical_overlap_role`,
  `mechanical_overlap_reason`
- Excitation coils additionally use `field_source_key = "<parent lens>"`.

The mechanical children never create optical elements or additional magnetic
field sources. Existing lens strengths, apertures and field formulae were not
changed by the split.

Canonical profiles and key helpers are in
`src/temsim/mechanical_profiles.py`. Assembly validation is in
`src/temsim/module_manifest.py`. The repeatable migration is
`scripts/migrate_lens_mechanical_layers.py`.

## Confirmed and provisional topology

Confirmed:

- C1 and C2 are adjacent hollow cylindrical lens assemblies.
- The C1–C2 interface has one C1 lower pole and one C2 upper pole facing each
  other. Their gap midpoint is the confirmed target region for the C1–C2
  crossover.
- A three-condenser system gives C3 its own two-pole/single-gap assembly.
- Diffraction Lens and Intermediate Lens mechanical envelopes do not overlap;
  they retain at least 5 mm axial clearance for future pole-piece configuration.
- The innermost continuous wall is called the **vacuum liner**. Use
  **alignment tube** only for a separately adjustable alignment component.

Provisional:

- Housing/yoke/coil radial dimensions are schematic initial ratios derived
  from each existing lens envelope, not measurements of FEI hardware.
- Diffraction Lens, Intermediate Lens, P1 and P2 are presently represented as
  independent two-pole/single-gap lenses.
- A real FEI projector column may share a central pole piece or magnetic yoke
  between adjacent excitation stages. Public information does not yet confirm
  that every software-named projector lens owns two mechanically independent
  pole pieces.
- Pole geometry currently affects the mechanical drawing only; it does not
  reshape the solver magnetic field.
- Vacuum-liner wall thickness is a provisional 0.25 mm.

Do not label provisional projector topology as an exact FEI/Titan mechanical
reconstruction without a service drawing, section drawing or measured part.

## Real-part photo intake

Photos can be used to replace provisional dimensions. For a useful measurement
handoff, request:

- Front, side and top views, photographed as orthogonally as possible.
- A ruler, caliper or other known dimension in the same plane as the part.
- At least one confirmed dimension such as bore diameter, flange diameter or
  mounting-hole spacing.
- Part name/number and the lens/stage it came from.
- Disassembled or section views when internal geometry is relevant.

For each inferred dimension, record:

```text
value_mm = ...
uncertainty_mm = ...
evidence = "photo/drawing/measurement identifier"
confidence = "confirmed | measured | estimated | provisional"
```

A single unscaled oblique photo supports shape and ratio estimates only. It
cannot establish reliable absolute dimensions, hidden bores, winding data,
material grade, permeability or saturation field.

## Next work

1. Ingest real-part photographs or section drawings and replace the schematic
   housing/yoke/coil ratios with evidence-backed dimensions and uncertainties.
2. Determine whether the selected FEI projector system uses independent
   two-pole lenses or shared three-pole/multi-gap assemblies, then update TOML
   parent relationships without changing optical calibration prematurely.
3. After mechanical topology is confirmed, define the second-stage magnetic
   inputs: turns, current at 100%, maximum current, polarity, resistance,
   material B–H data, saturation, cooling and field-model selection.
4. Design an equal-emission-time arrival-front overlay at selected important
   Z planes using the stored plane-arrival data.

## Guardrails for future changes

- Do not alter all lens strengths while crossover locations are incomplete.
- Keep every operating excitation at or below 100%.
- Do not use mechanical lens dimensions to truncate a lens field.
- Do not make a mechanical-only child contribute a second magnetic field.
- Preserve TOML as the authority for static geometry and optical references.
- Validate all 30 assemblies and run the full test suite after topology edits.
- Update this file whenever a provisional assumption becomes measured or
  confirmed, or when the next-work ordering materially changes.

## Known documentation mismatch

`README.md` still reports an older part-definition count. The validated current
count is 441; update the README together with the next broader documentation
refresh.
