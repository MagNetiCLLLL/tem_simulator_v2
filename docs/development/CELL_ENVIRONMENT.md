# Specimen chamber, gas/liquid cell and windows

## Workflow

1. Define the specimen in **Sample**: material/CIF or vacuum, finite envelope,
   thickness, centre X/Y and axial position. These remain the authoritative
   specimen settings; the cell does not replace them.
2. Open **Vacuum map**, select **Inserted specimen cell**, and set **Insert cell**.
   Enter the aperture diameter, inner-face **Cell gap**, and offsets. Dimensions
   in the cell editor are nm; stored cell geometry retains mm for compatibility.
3. Set each window independently: SiN, graphene, or custom formula and mass
   density; enter its thickness in nm. Zero explicitly means no window.
4. Choose ideal vacuum, gas, or liquid for the interior. Use a single chemical
   formula or optional JSON **mole fractions**, for example
   `{"Ar": 0.9, "H2": 0.1}`. Fractions must sum to one; no silent normalisation.
   Gas uses total absolute pressure in mbar and temperature in K. Liquid uses
   supplied mass density in kg/m³; its pressure is recorded but does not change
   density or deform the membranes.
5. Apply. Inspect the gold Sample marker in the full chamber and the local
   X-Z section, where **+Z points down**. The local view uses independent X/Z
   scales and shows the cell aperture; a larger specimen may extend outside it.
6. Enable **Include vacuum / cell transport in calculations** before Preview
   when cell effects are wanted. It remains off by default. Editing the map
   alone does not enable physics. Later changes may invalidate all calculation
   caches. Save an operating profile or use **Save map as** to retain inputs.

The local section represents applied settings, not unapplied text edits. Its
gold marker always identifies the Sample reference plane; for an inserted
non-vacuum sample a gold envelope also shows the physical specimen thickness.
The full-column marker does not change the user-defined shared Z range.

## Geometry and ownership

Let `z_s` be Sample Z, `o_z` the cell offset, `g` the gap, and `t_u,t_d` the
upstream/downstream thicknesses, all converted to mm for transport:

| Part | Axial interval |
|---|---|
| Upstream window | `[z_s + o_z - g/2 - t_u, z_s + o_z - g/2]` |
| Interior | `[z_s + o_z - g/2, z_s + o_z + g/2]` |
| Downstream window | `[z_s + o_z + g/2, z_s + o_z + g/2 + t_d]` |

All three share the specified finite circular aperture and transverse centre.
X/Y cell centres use the column coordinate system; Z offset is relative to
Sample. The displayed relative X/Y displacement also accounts for Sample's
own centre. The specimen chamber interval remains separately editable.

The outer faces must fit inside the specimen chamber. Solid Sample/window
overlap is rejected, including after a Sample edit; touching faces are allowed.
The program never moves or resizes a specimen to satisfy this check. A specimen
outside the cell remains a separately placed sample; the model does not impose
mounting/seal constraints or require its lateral extent to fit the aperture.

## Executed physics

Window slabs use the existing independent-atom Wentzel–Moliere screened elastic
particle operator at a supplied solid mass density. Mixtures use mole-weighted
atomic stoichiometry and mean molecular mass. Gas density is `P/(kT)`; condensed
media use `rho / mean_molecule_mass`. For graphene, the carbon number per area
is therefore determined by `rho * thickness`, not a separate source parameter.

Trajectory segments are clipped to the finite cylinders. Their union replaces
ambient gas, including when the interior is ideal vacuum. The real Sample
envelope is excluded from the gas/liquid medium because its interactions remain
in the existing sample pipeline. No duplicate Sample model is created.
Window boundaries and optically thin steps are inserted into the optical plan,
including sub-nm layers. Existing lens fields, apertures, walls and downstream
detector stops remain active. Elastic collisions redirect particles without
an additional artificial absorption weight or energy loss; independent removal
cross sections remain explicit supplied inputs with provenance.

The transport model identity is
`independent-atom-Wentzel-Moliere-elastic-v3-cell-windows-mixtures`.
All window, mixture and geometry settings round-trip through maps, profiles and
snapshots and enter active cache identities. Old maps lacking window fields
load with **zero thickness**, not newly inserted material. New default maps
contain two illustrative 50 nm SiN windows, but both cell insertion and global
vacuum/cell transport start **off**.

## Preset provenance and limits

Presets are editable examples, not calibrated TEM holder data. Material inputs
are in `configs/environments/cell_window_materials.toml`.

- **SiN:** nominal Si3N4, 3100 kg/m³ bulk-density approximation, 50 nm example.
  [Silson](https://silson.com/product/silicon-nitride/) lists Si3N4 membrane
  thicknesses including 50 nm and distinguishes silicon-rich nitride from
  stoichiometric nitride. The [International Syalons property guide](https://www.syalons.com/wp-content/uploads/2018/03/guide-to-types-of-silicon-nitride-ceramics.pdf)
  reports bulk silicon nitride densities varying with processing, including
  3.1 g/cm³. This is not a measurement of an amorphous SiNx membrane; users must
  replace formula and density with their device data when known.
- **Graphene:** carbon, 2260 kg/m³ and nominal 0.3354 nm single-layer-equivalent
  thickness. [USGS Professional Paper 1802-J](https://pubs.usgs.gov/pp/1802/j/pp1802j.pdf)
  gives graphite interlayer spacing 3.354 Å and density up to 2.26 g/cm³.
  These define an approximate carbon sheet areal density, not graphene elastic
  lattice diffraction or a mechanically meaningful continuum thickness.
- The two-window geometry is consistent with [liquid-cell TEM research](https://pmc.ncbi.nlm.nih.gov/articles/PMC4763102/).
  Real membranes can bulge with pressure; this implementation keeps them flat.

Not implemented here: pressure-induced deformation, flow, seals or spacer-frame
occlusion; radiolysis, chemistry, charging, stopping power/ionisation in the new
media; window-generated EDS lines or X-ray attenuation; crystalline window
diffraction and coherent liquid/window multislice. The existing particle raster
uses reference trajectories and does not recalculate a separate cell collision
history at every scan pixel. Thus this is **not acceptance of quantitative
gas/liquid TEM or STEM wave imaging**. Paused coherent-source development has
not been restarted. The existing forward-Z and finite-step scattering limits
are documented in [Vacuum map](VACUUM_MAP_2026-09-14.md).

## Validation

2026-09-14: **132 targeted tests passed in 54.37 s**, covering the four vacuum
suites, both cell suites, particle tip geometry, calculation-cache reuse and
particle-point cache versioning. Changed Python files compiled successfully.
The styled editor and complete cell section were rendered offscreen at
1550 × 1050 with no horizontal overflow. Pyqtgraph emitted its non-fatal
ViewBox signal-disconnect warning during teardown; no test failed. No long
full-source high-accuracy or coherent imaging run was started.

Targeted tests exercise geometry, sample/window overlap, gas-mixture partial
densities, fixed liquid density, exact sub-nm boundary nodes, oblique path
lengths, replacement rather than double-counting, and an executed short
particle segment through both windows. They also cover default opt-out, old
windowless profiles, map/profile/snapshot persistence, cache invalidation and
GUI editing/markers. These are regression and analytic checks, not material
cross-section calibration or a converged full-column cell-imaging benchmark.
