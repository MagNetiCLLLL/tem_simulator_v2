# Magnetic circuits and lens-structure evidence

Implementation stages 1 and 2, 2026-09-05. All dimensions remain TOML-owned.
Changing geometry never runs lens-preset optimisation. The shipped D/I/P1/P2
dimensions, operating strengths and independent-pole drawings are retained.
Their topology is explicitly an engineering assumption, not a Titan service drawing.

## Supported scope

| Circuit topology | Mechanical meaning | Current numerical support |
| --- | --- | --- |
| `two_pole_single_gap` | One channel, two separate poles, positive gap | Analytic/maps; linear or static nonlinear FEM for supported profiles |
| `shared_pole_multi_gap` | Multiple channels, at least three poles in a common circuit | Separate linear responses, or one joint nonlinear solve for all currents |
| `air_core` | Excitation coil without circuit-owned magnetic bodies | Linear/nonlinear FEM with vacuum permeability; no invented poles |
| `monolithic_saturated_insert` | One profiled magnetic insert with a saturation waist, not separate poles | Joint B-H FEM or fixed-operating-point maps; linear FEM rejected |

The projector manifest/layout path now accepts alternative explicit circuits.
Existing named condenser/objective assembly constraints are not all replaced:
their installation-specific interfaces still require a compatible manifest.
The mixed-material C1/C2 cartridge has explicit dependencies on both channels,
but its unresolved carrier volume is not silently filled with homogeneous iron.

## TOML declarations

The following are field snippets, not complete installable instrument files.
Each participating optical lens parent declares a circuit ID and topology:

```toml
magnetic_circuit_id = "projection_doublet"
magnetic_circuit_topology = "shared_pole_multi_gap"
magnetic_circuit_evidence = "engineering_assumption"
magnetic_circuit_source = "User-defined design; not an OEM reconstruction"
```

Parents with the same ID share their material geometry, but remain distinct
optical/current channels. If a legacy `pole_piece_topology` is retained, it must
agree with the explicit declaration. Evidence levels are `engineering_assumption`,
`published_design`, `measured_component`, and `oem_drawing`; the last three need
real source evidence supplied by the author, not merely a successful calculation.
All channels in one circuit use the same topology and evidence level.

Magnetic parts inherit circuit membership from the nearest lens ancestor.
A passive body outside that tree may explicitly name the affected channels:

```toml
magnetic_lens_keys = ["condenser_lens_1", "condenser_lens_2"]
```

This dependency is independent of display overlap groups. It invalidates both
geometry-bound maps when the body changes, without adding optical sources.
New mechanical-only bodies must retain `mechanical_only = true` so the assembly
loader does not mistake them for new optical controls.

Each excitation coil has one current owner:

```toml
field_source_key = "projector_lens_1"
coil_ampere_turn_fraction = 1.0
```

Ownership defaults to its nearest lens ancestor. An explicit owner may differ
within the same declared circuit. The recipe's ampere-turns are the channel's
total absolute winding ampere-turns at 100% excitation. If a channel owns multiple
coils, omitted fractions divide that total equally, preserving legacy behaviour.
Alternatively, supply all fractions, with absolute values summing to one.
Negative fractions reverse winding direction. A linear basis solve drives only
its own channel; a nonlinear solve drives all configured channels jointly.

Generated responses within a shared circuit must use identical permeability,
material overrides, mesh and boundary settings; only excitation may differ.
Mismatches are rejected, including when a channel already has a cached basis.
Analytic fallback remains an explicitly provisional placeholder, not a
geometry-derived solution for any newly declared topology. A mixed analytic/map
circuit is not a validated compound-lens solution.

## One profile for drawing and material calculation

A magnetic pole or yoke may define a piecewise-linear radial profile:

```toml
# [Z offset from this part's start, inner radius, outer radius], all in mm.
# Synthetic shape only; not a measured material or OEM dimension.
magnetic_radial_profile_mm = [
    [0.0, 2.0, 7.0],
    [9.0, 2.0, 2.5],
    [11.0, 2.0, 2.5],
    [20.0, 2.0, 7.0],
]
```

Offsets must increase from zero to the part length. Radii must satisfy
`0 <= inner < outer`, clear the declared vacuum bore and remain within the part
envelope. Both Physical Layout and the FEM material mask consume these exact
knots; mesh axes include profile knots and radii. This is axisymmetric geometry,
not support for arbitrary 3-D poles or manufacturing tolerances.

The monolithic topology requires one body with
`magnetic_part_role = "saturating_insert"` and an explicit radial profile.
A linear constant-permeability calculation cannot reproduce its operating
principle. Imported maps for this topology are locked to their documented
reference excitation and polarity, including frozen downstream transport plans.
A new current requires a new operating-point map. Generated B-H recipes now
solve that operating point explicitly. Hysteresis and remanence are not modelled.
Disabling a coil removes its current; passive material can still be magnetized
by other active coils in a joint solve.

## Cache and user interface

Model Inspector contains a **Magnetic circuits** subtab listing topology,
control count, magnetic-body count and evidence. Member keys, coils and source
details are in tooltips. Opening this page never starts a field solve.

Geometry bindings include circuit membership, coil ownership, turn fractions,
material profiles and neighbouring magnetic bodies in a generated map's domain.
Evidence prose is excluded from geometry identity. Linear strength changes reuse
the same unit-current maps; shared-body edits invalidate every affected channel.
The [static B-H stage](STATIC_NONLINEAR_MAGNETICS.md) advances
the solver implementation identity to `temsim-solver-2026-09-static-bh-v1`,
so incompatible older persisted seeds cannot be silently reused.

## Evidence and validation boundaries

- [US4450357A](https://patents.google.com/patent/US4450357A/en) describes a
  three-pole, two-gap projector with two coils. This establishes a possible
  topology, not its adoption in a particular Titan instrument.
- [FEI US9595359B2](https://patents.google.com/patent/US9595359B2/en) describes a
  monolithic saturated insert. Figure 6 compares gapped and ungapped simulations;
  it does not establish the current D/I/P1/P2 bore, pole diameter or topology.
  The previous projector geometry-source wording has been corrected accordingly.
- [EOD](https://lencova.com/index.php/fem) and
  [MEBS](https://mebs.co.uk/software/the-sofem-family/) are relevant independent
  field-design references. No EOD/MEBS/COMSOL comparison was run in this stage.

`tests/test_magnetic_circuits.py` uses explicitly synthetic dimensions and
material values. It checks two separate responses against a combined-current
linear solve, source ownership, shared invalidation, fixed-point map guards,
radial masks/drawing and actual installation of a no-pole projector variant.
The existing finite-solenoid test compares a vacuum coil with an analytic field;
the manufactured-potential test checks mesh refinement. These are numerical
checks, not measured-material validation or a calibration of installed lenses.

## Static B-H extension and next validation

[Static nonlinear lens fields](STATIC_NONLINEAR_MAGNETICS.md) implements sourced
reference tables, CSV import and a joint current-vector solve. Prepared meshes
and completed operating points are cached separately. Manufactured and limiting
cases are covered; an independent published full benchmark and measured lens
validation remain necessary before claiming saturated-lens design accuracy.
No proprietary material curve, production dimension or preset is invented.
