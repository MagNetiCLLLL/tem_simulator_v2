# Gun-field numerical repair before crossover-constrained matching

## Status

**Partial repair, not a qualified 30 mrad nanometre probe.** The historical
crossover count, order and component intervals remain required. No new lens
preset, electrode placement, voltage reference, emitting-cap size or source
distribution has been promoted. Coherent-wave development remains paused.

The numerical field repair is active for classical curved-tip calculations.
Cached gun traces use the executed field request, including the interpolation
version, mesh generation and numerical controls. A changed request cannot
reuse a trace from the old field. Historical source families stay readable.

## Reproduced near-axis force defect

The global tip-refined tensor grid retains nanometre radial cells hundreds
of millimetres downstream. Subtracting nearly identical kilovolt potentials
and dividing by their tiny squared-radius interval produced noisy radial
derivatives, including sign reversal.

For the original 166 x 461 field grid at Z = 200 mm:

| Radius | Original Er/r (V/m2) |
| --- | ---: |
| 1 pm to 1 nm | -1,756,112.81 |
| 10 nm | 14,012,210.11 |
| 100 nm | 581,207.66 |
| 1 um | 323,793.97 |
| 10 um | 319,091.59 |

`AxisRegularPotential` reconstructs the innermost interval in s = r2 from the
executed axis potential and a resolved radial node. Each axial node has its
own support, bounded by a fraction of the tip-distance/minimum-bore scale.
Interpolating these node profiles in Z keeps the scalar potential continuous.
Both electric-field components differentiate that same scalar potential.
Cut-tip cells and off-axis interpolation are unchanged. No focusing force is
added independently and no upstream stage is skipped.

On the original grid, fraction 0.01 gives Er/r = 319,205.39 V/m2 from 1 pm to
1 um. Halving the support is a separate numerical check, not a way to change
the physical source. Harmonic quadratic and quartic fixtures check precision,
the expected radial approximation error, symmetry and scalar-gradient
consistency; these fixtures do not qualify the full electron gun.

Axis expansion and the sensitivity of lens properties to potential
derivatives are discussed in [Nezhad et al., Scientific Reports (2024)](
https://www.nature.com/articles/s41598-024-55518-3). This reference supports
the numerical motivation, not an OEM calibration or validation of this model.

## Electrode-scale mesh defect and refinement

The old logarithmic axial mesh was fine at the tip but too coarse at the
accelerator rings. Doubling the global grid changed Er/r at Z = 200 mm from
319,205 to 175,616 V/m2, despite a tiny linear-solver residual.

`refine_electrode_axes` now subdivides existing intervals near physical bore
and outer-rim radii, within four bore radii of each electrode, and near the
grounded exit. All original tip and exact metal nodes remain. The **full**
vacuum domain is still solved; the local refinement is not a field cutoff.

Field-only measurements at r = 10 nm, unchanged physical gun:

| Electrode cells per bore radius | Grid | Er/r at Z=50 mm (V/m2) | Er/r at Z=200 mm (V/m2) | Solve time (s) |
| --- | --- | ---: | ---: | ---: |
| 0 (old mesh) | 166 x 461 | -446,981.09 | 319,205.39 | 0.28 |
| 8 (new default) | 264 x 1180 | -383,149.69 | 95,631.68 | 1.35 |
| 16 | 381 x 1974 | -384,489.30 | 97,613.70 | 3.79 |
| 32 | 616 x 3572 | -390,155.82 | 98,539.88 | 16.44 |

At Z = 200 mm the 8-to-16 discrepancy is about 2.0%, and 16-to-32 about
0.94%. Doubling the global tip grid as well, with 16 cells per bore, gives
97,991.61 V/m2. These are improved diagnostics, **not full field convergence**.
At Z = 8 mm the axial derivative still changes from -15,420 to -13,643 V/m
between local resolutions 8 and 16. Derivatives at piecewise-cell boundaries,
the remaining global mesh, outer boundary and trajectory tolerances need
independent refinement before nanometre focus can be certified.

The refined mesh also exposes a precision floor for the smallest radial
interpolation support: at Z = 200 mm, fractions 0.02 / 0.01 / 0.005 give
95,497.57 / 95,631.68 / 96,210.48 V/m2 on the 8-cell mesh. The earlier 0.5%
support-spread check passes for its original-mesh reproducer but does **not**
qualify this refined field. Reducing support indefinitely is not convergence
when subtraction error grows. This limitation is retained in the evidence.

## Actual tip-to-column executions

The same 193 physical samples use `apex_stratified_v1`, with all nine area
strata and their original current weights; eight directions per position.
The historical Nanoprobe/Diffraction lens branch, 0.05 mm column step, 4 kV
extractor and 1.2 kV gun lens relative to the extractor are unchanged.
Vacuum participation is off. Physical apertures and walls remain active.
These are limited-budget particle runs, not image calculations.

| Mesh / axis support | Gun exit d95 (mm) | Source current fraction at exit | Rays at C2 | Rays at sample entrance |
| --- | ---: | ---: | ---: | ---: |
| Old mesh / raw | 4.698225 | 0.6625 | 27 | 0 |
| Old mesh / 0.01 | 4.698225 | 0.6625 | 27 | 0 |
| Old mesh / 0.005 | 4.698225 | 0.6625 | 27 | 0 |
| 8 cells / 0.01 | 4.110305 | 0.7750 | 28 | 0 |
| 16 cells / 0.01 | 4.127949 | 0.7750 | 28 | 0 |

The last two runs took 116.88 and 119.02 seconds while sharing the host with
validation. Their maximum exit-energy errors were 3.20e-9 and 4.25e-9 eV.
The interpolation-only repair does not explain the gross beam expansion;
the electrode mesh has a larger effect. Even the refined exit beam is still
millimetre-scale, and **there is no accepted sample probe in these runs**.

The two resolved incident waist candidates are:

| Mesh | C1-to-C2 candidate Z (mm) | C2-to-aperture candidate Z (mm) |
| --- | ---: | ---: |
| 8 cells | 503.052047 | 667.981940 |
| 16 cells | 503.052697 | 667.981206 |

These are interpolated brackets requiring exact-plane refinement, not five
qualified crossovers. C1-centre alpha95 is still about 421--423 mrad: this
large-angle branch must not be treated as a qualified paraxial nm probe.
Zero surviving samples are not a proof of zero mathematical accepted current.

## Configuration and acceptance

The authoritative shared part `configs/sources/FEG_tip.toml` now exposes:

- `tip_field_axis_core_fraction = 0.01`;
- `tip_field_electrode_cells_per_bore = 8` (0 replays the historical mesh).

These are numerical controls, not new physical tip inputs. Nondefault values
round-trip through source snapshots and affect field/trace identities. Linked
assemblies missing these optional fields can materialize and edit them while
preserving their comments and unrelated tables. Unknown field names still fail.
The separate reference-model TOML carries the same numerical defaults.

The user's crossover requirement does not impose a minimum source-current
fraction. The surface-focus gate no longer silently requires 1% transmission.
It still requires positive current, at least 16 positive-current trajectories
and weighted effective sample size N_eff = (sum w)^2 / sum(w^2) >= 16.
An explicit experiment may request a minimum fraction. Angle, waist, diameter,
old crossover topology and sampling convergence remain independent gates.

## Reproduction and remaining work

```powershell
.\.venv\Scripts\python.exe scripts/check_gun_field_mesh.py --cells 0 8 16 32
.\.venv\Scripts\python.exe scripts/check_gun_field_mesh.py --cells 8 16 --global-scale 2
.\.venv\Scripts\python.exe scripts/check_gun_axis_regularisation.py --fraction .01 --electrode-cells 8 --output <new-scalar-report.json>
.\.venv\Scripts\python.exe scripts/check_gun_axis_regularisation.py --fraction .01 --electrode-cells 16 --output <another-new-scalar-report.json>
```

Local scalar execution evidence is under ignored
`outputs/crossover-audit-20260914/axis-regular-193.json`,
`electrode-mesh8-193.json` and `electrode-mesh16-193.json`. No array output or
calculation cache is intended for Git. The lightweight report records evidence
without turning it into an active source or qualified preset.

Next work remains gun-exit/C1 phase-space matching using the physical gun
geometry and permitted voltage ranges, followed by all five historical
crossover intervals and the surface d95/alpha95/focus targets together.
Do not restore the rejected weak-objective branch merely to transmit rays.

## Validation

Final executed checks: **213 passed** across the affected sets: 159 tip,
field, sampling, focus, topology, shared-source and source-contract tests;
53 manifest/part-document tests; and the persisted-working-point replay
test. The latter verifies that a fully blocked beam stays unavailable on
replay rather than acquiring a fictitious zero-angle signal. The initial
run exposed missing optional fields in linked TOML snapshots; insertion was
fixed and the affected tests passed on rerun.

The suites report existing Pydantic deprecation warnings. The blocked-beam
replay also emits column-polynomial overflow warnings on rejected paths;
passing persistence is not physical acceptance of those trajectories. No
full-project suite, visible desktop interaction, coherent propagation or
TEM/STEM image qualification was run. All five reported gun comparisons
are actual classical propagation, separate from the isolated test fixtures.
