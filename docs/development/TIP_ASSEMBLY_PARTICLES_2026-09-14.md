# TOML-owned FEG tip and classical particles

Coherent wave imaging remains paused. New FEG assemblies now start with the
classical curved-tip source declared in their installed TOML part. No
independently configurable gun-exit or specimen-plane source was added.

## Files and editing

- `configs/sources/FEG_tip.toml`: shared, editable physical tip definition. Open
  it with Physical Layout > 3D model editor > Open TOML. Both FEG variants link
  to this file through `tip_definition_file`; their embedded values are portable
  snapshots, not competing definitions. A missing shared file is an error.
- `configs/instruments/gun/FEG.toml`, `FEG_Mono.toml`: select **Cold Field
  Emission Tip** to edit its shared dimensions. The existing TOML parameter
  table also edits emission and numerical fields. Saving through either variant
  or the shared file updates every linked assembly and refreshes the active gun,
  including when the edited variant is not installed. The 3D editor identifies
  the shared source. External file edits take effect after **Reload TOML**.
- Model Inspector > FEG tip edits a runtime draft. **Apply** changes the active
  tip, and saving an operating profile retains that override. **Reload installed
  tip TOML defaults** restores the current installed part into the draft.
- `configs/sources/cold_feg_tip.toml` remains a historical total-current emission
  template. Old planar-particle and coherent profiles retain their explicit source
  choices when read; opening a profile does not silently select the new default.

Shared edits validate all 30 compatible assemblies before writing. Stale drafts
cannot overwrite a newer shared definition; write or live-reload failures roll
back the affected files. Unchanged defaults preserve current operating overrides.
Extractor, gun-lens and accelerator voltage defaults still belong to their own
electrodes in each assembly. Shared tip edits do not change those voltages.

**Copy component** creates an independent mechanical solid with no shared link
or extra electron emitter. **Save copy** detaches the resolved values so the copy
is independent. Archived calculation inputs include their own shared definition,
and calculation manifests record its content hash. Historical output files are
not rewritten. This synchronization update does not resume coherent calculations.

The broader regression also exposed a particle-display issue: stored histories
after aperture/wall absorption were incorrectly submitted as vector-field query
positions or rejected by a global preview finiteness check. Undefined post-stop
placeholders now retain their source identities without becoming field queries.
Incoming and outgoing map histories are validated up to the actual physical
stop; non-finite live trajectories still fail explicitly. No current is assigned
to a stopped ray. The preview thread test now checks GUI timer responsiveness
and reports calculation errors directly while allowing a cold tip-field solve.

The installed apex remains at gun Z = 0. Changing shank length moves its upstream
end, and recalculates its centre and base diameter. The sphere joins the cone
tangentially. Radius, cone angle, shape and material identity are stored with
the physical part, and the same geometry supplies the 3D solid, emitting surface
and conducting boundary of the electrostatic calculation. Selecting a surface
or edge exposes its related dimension parameters. Independent Boolean CAD
overrides are rejected for this emitting tip because that field solver does not
consume them.

## Parameter definitions and defaults

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tip_shape` | `spherical_cap_tangent_cone` | Supported physical conducting contour |
| `tip_material` | `W(310)` | Material identity; no inferred tunnelling law |
| `length_mm` | 1 mm | Physical metal shank length upstream of apex |
| `tip_radius_nm` | 100 nm | Apex radius of curvature; curvature = 1/R |
| `tip_cone_half_angle_deg` | 5° | Cone angle relative to the axis |
| `emission_cap_half_angle_deg` | 10° | Geometric emitting-cap extent |
| `emission_flux_electrons_per_nm2_s` | 653864449.6443435 | Emitted electrons per nm² of curved surface per second |
| `emission_maximum_angle_deg` | 90° | Outgoing direction limit about each local surface normal |
| `emission_energy_distribution` | `normal_tangential_exponential` | Positive launch-energy/direction distribution |
| `emission_normal_mean_energy_ev` | 0.2 eV | Normal-component exponential scale |
| `emission_tangential_mean_energy_ev` | 0.1 eV | Tangential-component exponential scale |
| `emission_kinetic_mean_ev` | 0.3 eV | Used by `gamma` and `monoenergetic` laws |
| `emission_kinetic_sigma_ev` | 0.1 eV | Used by `gamma`; monoenergetic has zero width |

For cap angle θ, the actual area is `4π R² sin²(θ/2)` and the total current is
`electron charge × area × flux density`. Defaults therefore give 100 nA from
about 954.56 nm². This is prescribed emission, not a Fowler–Nordheim or
space-charge prediction. Keeping density fixed while doubling R quadruples
current. Cap area and maximum particle angle are separate controls.

The exponential law is conditioned on `E_tangent <= tan(angle_limit)² E_normal`.
Consequently narrowing its angular limit also changes total-energy statistics;
the editor displays those derived statistics. At zero angle the limiting
conditional distribution is used. Gamma and monoenergetic alternatives use
uniform solid-angle directions inside the limit, independently of total energy.
All launch positions and full direction vectors stay on the physical tip.

`tip_field_*` fields specify radial/axial mesh budgets, apex resolution, outer
boundary and residual tolerance. Numerical budgets do not switch off gun optics.

## Electrode ownership and connected acceleration

Electrical defaults are on the existing electrode parts, not on a downstream
source: extractor `default_voltage_kv = 4`, gun lens `default_voltage_kv = 1.2`,
accelerator `default_high_tension_kv = 300`. The accelerator also owns
`electrode_thickness_mm = 1`; stage positions remain its existing
`stage_centers_z_mm`. Existing voltage fractions rise through the ten stages to
exactly one at the grounded final anode.

At the default operating point, the tip is −300 kV, the extractor −296 kV,
the gun-lens electrode −294.8 kV and the final anode 0 V. The grounded Laplace
solution covers the whole tip/extractor/gun-lens/accelerator vacuum domain.
Spaces between metal electrodes remain solved vacuum. Historical compact field
windows do not create artificial zero-field gaps in this source model.

Particles retain extraction, acceleration, focusing, alignment and aperture
interception. The static discrete-gradient integrator obtains exit momentum
from actual transport; it does not project momentum onto a requested voltage.
The report compares exit kinetic energy with `initial kinetic energy + e × HT`,
using a 0.001 eV error budget. No transmitted particles produces an explicit
`no_transmitted_particles` result, not a vacuous zero-error acceptance.

Changing a consumed geometry, emission, voltage or numerical input invalidates
particle reuse. Emission-only changes reuse an unchanged electric field.
Unchanged assembly defaults preserve runtime overrides during reload; a changed
default is applied. Repeated catalog loading no longer temporarily replaces a
custom catalog with global gun defaults.

## Validation and limits

The shared-definition follow-up exercised **309 distinct test cases**, all with
passing final results, including **20 shared-tip cases**. The initial broader
runs exposed the stopped-history preview/map issues described above; their
affected transport and GUI cases passed after the fix. These counts include
overlapping reruns and are deduplicated, not added to the earlier runs below.
The final wheel's source/definition bytes were checked, and an isolated installed
copy propagated a shared radius edit to both variants and the runtime emitter.
Summary: `evidence/shared-tip-checks-2026-09-14.json`.

The main focused regression run passed **287 tests** covering source policy,
legacy source editing, profile/snapshot ownership, field and particle transport,
3D parts, parameter meanings, and manifest editing. Additional tip-assembly
checks cover density, every supported energy law and angular limit, strict input
types, copy placement, voltage reload in both FEG variants and current derivation.
The final focused run passed **96 tests**, including **25 new tip-assembly
cases** and parameter semantics/impact checks. These runs overlap; their counts
must not be added. Summary: `evidence/tip-assembly-checks-2026-09-14.json`.
Both source TOMLs
are included in the wheel; an isolated installed-wheel source editor opened
successfully. The tip solid and emission editor were rendered and inspected.

Actual particle runs kept the default physical apertures:

| HT | Extraction / gun lens | Samples → exit | Exit kinetic energy | Maximum energy error |
| --- | --- | --- | --- | --- |
| 300 kV | 4 / 1.2 kV | 9 → 3 | 300000.196031–300000.711070 eV | 1.98×10⁻⁹ eV |
| 200 kV | 4 / 1.2 kV | 49 → 0 | No exit energy acceptance | Not applicable |
| 200 kV | 2.666667 / 0.8 kV | 9 → 3 | 200000.196031–200000.711070 eV | 1.48×10⁻⁹ eV |

Changing only HT changes focusing: the unretuned 200 kV run lost 47 samples at
DPA and two at C1. The adjusted 200 kV case scales the electrode voltages; it
does not change aperture openings or create a replacement source. These small
particle bundles verify acceleration and interception, not converged current
or brightness. A small Laplace residual and energy error also do not certify
spatial field convergence, OEM geometry, tunnelling, space charge, or coherent
TEM/STEM imaging.

Scalar reports and exact input settings are retained in:

- `evidence/tip-assembly-particles-2026-09-14.json`
- `evidence/tip-assembly-particles-200kv-2026-09-14.json`
- `evidence/tip-assembly-particles-200kv-scaled-2026-09-14.json`

Reproduce the default case with:

```powershell
.venv/Scripts/python.exe -m scripts.validate_tip_particles --output outputs/tip-particles.json
```

The command writes no wave or particle array caches. No coherent image
calculation was run for this update.
