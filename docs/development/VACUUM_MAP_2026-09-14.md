# Classical vacuum map and insertable specimen cell

The standalone Vacuum map tab, immediately after Sample, edits pressure, gas formula, temperature,
region boundaries and a finite cylindrical specimen cell. The default input is
`configs/environments/vacuum_map.toml`. Vacuum transport is **off by default**;
the map remains visible and editable. Choose whether to include it in
**Calculate setup** or **Vacuum map** before the first Preview. Later changes
remain allowed and may invalidate all cached calculation stages; this broad
invalidation is intentional. Apply
changes the live instrument; Save map as writes an independent TOML map (or the
default file when explicitly selected). Operating profiles and working-point
snapshots retain the complete active map. Historical profiles without a map
retain their original absence of residual-medium transport. Saved explicit
on/off choices are restored unchanged. Load normal-operation defaults restores
the pressure map with transport off; enable the checkbox explicitly when needed.

## Shared Z view and editing

Vacuum map uses actual millimetre Z intervals, with a separate row showing the
installed assembly modules. It no longer uses equal-width schematic boxes.
Physical Layout and Vacuum map share the same X range in both directions;
their plot gutters have matching widths. Pan/zoom in either page changes the
shared view. Selecting a region, applying a pressure edit, switching pages or
resizing the window does not fit the vacuum data to a new range.

The parameter editor is hidden until a region is selected. Click an interval
(or use the region selector for very short intervals) to edit its pressure.
The editor lists the assembly files intersecting that interval. Choose
`Z positions` to enter global Z or local coordinates of an installed module.
Module origins use the resolved assembly transform; they are not inferred from
display widths. Module-local ranges follow module placement changes. The
existing component anchors plus offsets remain available. The projection
entrance stays bound to the installed DPA.

Coordinates persist through the existing TOML anchor representation:
`axis_origin` for global Z and `module:<module-key>.origin` for module-local Z.
Additional `.start` and `.end` module anchors identify the installed interfaces.
Existing map files keep their component anchors until the user applies another
coordinate definition. Gaps remain linear transitions; overlaps remain errors.

In Physical Layout 2D, double-click no longer navigates. Right-click a component
and choose `Go to Ray Diagram`, `Go to 3D Parts`, or `Go to Vacuum map`.
The component remains highlighted across manual tab switches. Vacuum navigation
selects the containing interval and marks the component without changing Z scale.

Validation for this update: 47 targeted vacuum, transport, setup and parameter
semantics tests plus 7 context-menu/navigation tests passed. Offscreen rendering
of the styled main window measured identical 1390-pixel view widths and identical
Z limits in Physical Layout and Vacuum map; selection reveals the editor without
changing those limits. Images and raw test logs remain local in
`outputs/vacuum-validation/`.

## Default region assignments

| Beam-path region | Default pressure (mbar) | Reference / boundary |
|---|---:|---|
| Gun tip and extraction | 3e-11 | IGPa, rounded; physical tip start to accelerator entrance |
| Gun accelerator | 2e-8 | IGPb, rounded; accelerator entrance to gun exit |
| Column before specimen | 1e-7 | IGPcl, rounded |
| Specimen surroundings | 1e-7 | Assumed column ambient; specimen centre ± 5 mm |
| Column after specimen | 4e-8 | IGPco, rounded; ends at projection DPA |
| Projection / camera chamber | 8e-7 | CCGp, rounded; begins at installed projection DPA |

These values use the user-supplied vacuum overview photographs of 2026-09-14.
Assignments of gauges to axial regions are approximate. N2 composition and 293 K
temperature are editable assumptions, not measurements. The zero IGPf reading
does not establish perfect vacuum. Foreline, manifold and load-lock pressures
are excluded from the beam-path map. Valve state is not inferred from the photos;
this preset represents the user's requested open beam path under normal vacuum.
The 10 mm specimen surroundings span is an editable reservation, not a measured
OEM cell dimension.

Boundaries follow installed physical anchors with editable offsets. Each region
keeps independent start/end boundaries. A gas/vacuum gap is filled automatically
by a linear-pressure transition; overlaps and moving the projection boundary
away from its physical DPA are rejected. Transition boxes are generated from
the endpoints, not independently saved or editable. Liquid interfaces require
explicit boundaries; an automatic vacuum transition is not a liquid meniscus. Users can split regions and merge added
regions back into their predecessor. All 30 catalog selections resolve the map.

The cell starts retracted, with aperture diameter 10 µm and inner-face gap 1 µm.
Z offset is relative to Sample; X/Y centres use column coordinates. New defaults
include two independently editable 50 nm SiN windows; historical maps without
window fields remain windowless. These are example dimensions, not changes to
the Sample geometry. The windows and interior replace ambient gas, and the
finite solid specimen is excluded from the fluid to avoid duplicate material.
See [Cell environment](CELL_ENVIRONMENT.md) for the local +Z-down section,
material presets, mixtures, ownership and limitations.

## Transport and meaning of attenuation

Gas molecule density is `n = (100 P_mbar)/(k_B T)`. Liquid molecule density uses
mass density and formula mass instead of pressure. Each chemical formula is
expanded into constituent atoms, retaining stoichiometric multiplicities.

Elastic events use the independent-atom Wentzel–Molière screened Coulomb
approximation. The total cross section is integrated over the full sphere;
the conditional scattering angle is sampled from its analytic inverse CDF.
The implementation follows the screening parameter and differential cross
section in equations 95–96 of the [Geant4 electron scattering reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html).
This is an explicitly approximate model, not a bundled experimental database.

The cumulative hazard is `tau = integral n sigma(E) ds`. `ds` comes from the
executed three-dimensional piecewise trajectory, clipped to medium and hardware
boundaries, rather than substituting nominal axial thickness. The reported
uncollided survival is `exp(-tau)`. Elastic scattering conserves particle weight
and kinetic energy; it changes subsequent transport and detector acceptance.
It does **not** receive an additional exponential absorption weight. This avoids
counting one elastic collision twice as lost current.

An optional removal cross section per molecule models a separate supplied loss
process. It defaults to zero and requires an explicit measurement/model reference
when nonzero. Removal is sampled at its path coordinate. Upstream readouts retain
their arrivals; downstream readouts cannot collect the removed particle. This
is not an automatic model of ionisation or stopping power.

The gun evaluates gas interactions within actual time-domain extraction and
acceleration. Elastic events rotate momentum at constant magnitude. The column
executes collision operators between the existing optical integration steps,
including imported vector-field maps. Apertures, walls and absorbing recording
planes still apply in physical order, and accounting ends at their first hit.
Energy filters retain their existing routing; this change does not replace
their trajectory solver with a new source or a bypass.

Cell/slab boundaries, bore transitions and recording planes are inserted into
the integration grid. Configured optical-depth limits refine dense regions;
exceeding the numerical budget raises an explicit error. Current medium-enabled
column execution uses the CPU implementation. Whole executed calculation
products remain cacheable. Optical phase-space-only restart checkpoints are
not reused for a medium run because they do not retain its collision clocks.
All active map inputs, resolved boundaries, material exclusion and the medium
model version enter calculation identities; gun inputs enter the gun cache.

## Explicit limits

- The atomic elastic approximation neglects spin, molecular bonding and recoil.
  Its continuation below 50 eV is recorded as extrapolated path length. For
  comparison, [NIST SRD 64](https://www.nist.gov/publications/nist-electron-elastic-scattering-cross-section-database-version-40)
  provides atomic elastic data over 50 eV–300 keV. No NIST data have been imported.
- Liquid and solid windows use density-based independent-atom elastic transport.
  Inelastic energy loss, radiolysis, charging, fluid dynamics, membrane bulging,
  window crystal diffraction and photon attenuation require later models.
- Collision kicks use finite-step splitting. Reducing the optical step and
  optical-depth limit is necessary for convergence studies in dense media.
  Forward-Z column transport records backscattered particles as exiting that
  forward transport domain; it does not follow them back through the column.
- Existing affine raster previews reuse reference particle trajectories. They
  do not recalculate a cell collision history at each scan pixel; their metrics
  and image explanation state this limitation. This change does not claim a
  validated gas/liquid STEM image, nor enable paused coherent-wave calculations.
- The map covers the axial column through the recording chamber. It does not
  add residual-gas collisions along the separately bent energy-filter orbit.

## Validation

116 distinct regression cases passed, including 21 vacuum-specific cases and
the real background GUI preview. After the final recording-plane continuation
fix, the dedicated suite and preview were rerun successfully. The installation
wheel includes the default map and matches the tested transport source.

The dedicated tests cover gas/liquid density, pressure units, input rejection,
full-angle integration, seeded scattering statistics, exponential uncollided
survival, actual removal positions, oblique/cylindrical intersections, exclusion
of the solid specimen, downstream deflection, camera stops, snapshot/profile
round trips, GUI editing, cache invalidation and all catalog assemblies.

A real 9-particle 300 kV FEG check retained 3 exit particles and verified exit
kinetic energy within 1e-9 eV with the normal map. This checks execution and
energy conservation, not mesh convergence or experimentally validated gas
cross sections. Scalar evidence is retained under `evidence/`; rendered views
and test output remain local in `outputs/vacuum-validation/`.


## Horizontal map and Calculate setup update

The map runs left-to-right along +Z with physical Z widths shared with Physical
Layout. It remains visible/editable when transport is off. Geometry and medium
controls occupy two columns below the map; a separate local cell section uses
+Z downward and marks the Sample envelope/reference plane.

For a gap from z0 to z1, t=(z-z0)/(z1-z0). Each endpoint contributes its original
gas at partial pressure (1-t) P0 or t P1, with that endpoint's temperature and
cross sections. Thus total pressure is linear even when compositions differ.
Ideal-vacuum endpoints contribute zero density. For equal composition and
temperature this reduces to n(z)=P(z)/(kT). This is an explicit interpolation
rule, not a pumping/conductance solver or a measured pressure distribution.

At fixed step energy, rates are linear along each clipped straight trajectory.
Integrated collision/removal hazards use the analytic line integral; collision
species are selected at their sampled hazard coordinate. Removal positions use
the quadratic inverse integral, including a zero-density endpoint. Cell and
solid exclusion continue to clip the same path. Step budgets use maximum
endpoint hazard, and transition endpoints enter gun/column cache identities.

Calculate setup is on the top toolbar. It shares the actual state fields:

| Control | State | Meaning |
|---|---|---|
| STEM detector images | sample.stem_image_enabled | Optional frame output; does not disable scan coils or physical detector absorption |
| EDS spectrum | sample.eds_enabled | Explicit EDS acquisition |
| Vacuum scattering / attenuation | vacuum_map.enabled | Gun and column residual-medium transport |
| Seeded STEM / EDS counts | sample.stem_poisson_enabled / eds_poisson_enabled | Readout statistics, separate from trajectories |
| TEM / coherent STEM / 4D-STEM | Existing wave flags | Paused; historical requests remain readable and can be turned off |

New states select classical STEM particle readouts and leave coherent STEM off.
No historical explicit wave flags are silently converted. Turning off a readout
does not retract hardware, change emission, bypass optics, or admit a coherent
source. The current particle raster still has the previously documented affine
reference-trajectory limitations; this UI does not add atomic wave contrast.
